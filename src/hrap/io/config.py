"""Motor configuration: MATLAB .mat import, JSON save/load, SI resolution."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from hrap.engine.nox import nox
from hrap.engine.types import Settings, State
from hrap.io.propellant import load_propellant
from hrap.layout import (
    MotorLayout,
    default_chamber_length,
    infer_component_stations,
    motor_layout,
    nozzle_exit_diameter,
)
from hrap.units import to_si

VENT_MAP = {"None": 0, "External": 1, "Internal": 2, 0: 0, 1: 1, 2: 2}


def default_cfg() -> dict[str, Any]:
    """MATLAB App Designer defaults plus a runnable example grain/tank."""
    return {
        "mtr_nm": "mtr_cfg",
        "tnk_V": 12624.05,
        "tnk_V_unit": "cm^3",
        "tnk_V_state": 0,
        "tnk_L": 0.0,
        "tnk_L_unit": "cm",
        "tnk_D": 3.75,
        "tnk_D_unit": "in",
        "cmbr_V_state": 1,
        "cmbr_V": 0.0,
        "cmbr_V_unit": "cm^3",
        "noz_thrt": 1.0,
        "noz_thrt_unit": "in",
        "noz_def": "Nozzle Expansion Ratio",
        "noz_ex": 5.5,
        "noz_ex_unit": "in",
        "noz_eff": 97.0,
        "noz_Cd": 0.95,
        "mp_state": 1,
        "tnk_start": 0.0,
        "tnk_start_unit": "in",
        "cmbr_start": 0.0,
        "cmbr_start_unit": "in",
        "cmbr_L": 0.0,
        "cmbr_L_unit": "in",
        "tnk_m": 0.0,
        "tnk_m_unit": "kg",
        "cmbr_m": 0.0,
        "cmbr_m_unit": "kg",
        "tnk_X": 0.0,
        "tnk_X_unit": "in",
        "cmbr_X": 0.0,
        "cmbr_X_unit": "in",
        "mtr_cg": 0.0,
        "mtr_cg_unit": "in",
        "mtr_m": 0.0,
        "mtr_m_unit": "kg",
        "tnk_dd": "Starting Tank Temperature",
        "tnk_cond": 293.15,
        "T_tnk_unit": "K",
        "P_cmbr": 1.0,
        "P_cmbr_unit": "atm",
        "fill_dd": "Tank Fill Percentage",
        "fill": 95.0,
        "fill_unit": "kg",
        "Pa": 1.0,
        "Pa_unit": "atm",
        "prop_file": None,
        "prop_nm": "ABS",
        "prop_id": "ABS",
        "prop_rho": 1070.0,
        "prop_rho_unit": "kg/m^3",
        "prop_a": 0.198,
        "prop_n": 0.325,
        "prop_m": 0.0,
        "const_OF": 6.71,
        "cstar_eff": 100.0,
        "grn_ID": 2.1875,
        "grn_ID_unit": "in",
        "grn_OD": 3.375,
        "grn_OD_unit": "in",
        "grn_L": 16.946,
        "grn_L_unit": "in",
        "inj_D": 0.25,
        "inj_D_unit": "in",
        "inj_N": 3,
        "inj_Cd": 0.361,
        "vnt_state": "Internal",
        "vnt_D": 0.028,
        "vnt_D_unit": "in",
        "vnt_Cd": 0.75,
        "t_max": 10.0,
        "t_burn": 0.0,
        "dt": 0.001,
        "reg_model": "Constant OF",
        "inj_model": "SPI",
        "inj_Cd_HEM": 0.0,
        "dyer_kappa": 1.0,
        "source": "hrap",
        "advanced": {
            "enabled": False,
            "ox_fluid": "N2O_legacy",
            "grain_shape": "cylindrical",
            "live_chem": False,
        },
    }


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _merge_cfg(data)


def save_json(path: str | Path, cfg: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _merge_cfg(data: dict[str, Any]) -> dict[str, Any]:
    cfg = default_cfg()
    cfg.update({k: v for k, v in data.items() if v is not None})
    if "prop_id" not in data or not data.get("prop_id"):
        cfg["prop_id"] = str(cfg.get("prop_nm") or "ABS")
    return cfg


def load_matlab_mat(path: str | Path) -> dict[str, Any]:
    import scipy.io

    raw = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)["cfg"]
    data: dict[str, Any] = {}
    for field in raw._fieldnames:
        val = getattr(raw, field)
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        elif hasattr(val, "item") and getattr(val, "shape", None) == ():
            val = val.item()
        elif type(val).__name__ == "MatlabOpaque":
            val = None
        if val is not None:
            data[field] = val
    cfg = _merge_cfg(data)
    cfg["source"] = "matlab"
    return cfg


def _u(cfg: dict, field_unit: str, group: str) -> float:
    return to_si(1.0, cfg[field_unit], group) if not isinstance(cfg[field_unit], (int, float)) else float(cfg[field_unit])


def _len(cfg: dict[str, Any], name: str, default: float = 0.0) -> float:
    if name not in cfg or cfg[name] is None:
        return default
    unit = cfg.get(f"{name}_unit") or "m"
    return float(cfg[name]) * to_si(1.0, unit, "length")


def _mass(cfg: dict[str, Any], name: str, default: float = 0.0) -> float:
    if name not in cfg or cfg[name] is None:
        return default
    unit = cfg.get(f"{name}_unit") or "kg"
    return float(cfg[name]) * to_si(1.0, unit, "mass")


def resolve_layout(cfg: dict[str, Any]) -> MotorLayout:
    """Hardware stations and dry masses used for CG, ENG/RSE length, and the schematic."""
    grn_OD = _len(cfg, "grn_OD")
    grn_L = _len(cfg, "grn_L")
    tnk_D = _len(cfg, "tnk_D")
    tnk_L = _len(cfg, "tnk_L")
    if int(cfg.get("tnk_V_state", 0)):
        tnk_V = tnk_L * 0.25 * math.pi * tnk_D ** 2
    else:
        tnk_V = float(cfg.get("tnk_V") or 0.0) * to_si(1.0, cfg.get("tnk_V_unit") or "m^3", "volume")
    noz_thrt = _len(cfg, "noz_thrt")
    if cfg.get("noz_def") == "Nozzle Exit Diameter":
        noz_exit = float(cfg.get("noz_ex") or 0.0) * to_si(1.0, cfg.get("noz_ex_unit") or "in", "length")
    else:
        noz_exit = nozzle_exit_diameter(noz_thrt, float(cfg.get("noz_ex") or 1.0))
    if tnk_L <= 1e-9 and tnk_D > 0 and tnk_V > 0:
        tnk_L = tnk_V / (0.25 * math.pi * tnk_D ** 2)
    auto_cmbr = default_chamber_length(grn_OD, grn_L, noz_thrt, noz_exit)
    tnk_start, cmbr_start, cmbr_L = infer_component_stations(
        tnk_start=_len(cfg, "tnk_start"),
        cmbr_start=_len(cfg, "cmbr_start"),
        cmbr_L=_len(cfg, "cmbr_L"),
        tnk_L=tnk_L,
        grn_L=grn_L,
        tnk_X=_len(cfg, "tnk_X"),
        cmbr_X=_len(cfg, "cmbr_X"),
        auto_cmbr_L=auto_cmbr,
    )
    return motor_layout(
        tnk_start=tnk_start,
        tnk_L=tnk_L,
        tnk_m=_mass(cfg, "tnk_m"),
        tnk_D=tnk_D,
        cmbr_start=cmbr_start,
        cmbr_L=cmbr_L,
        cmbr_m=_mass(cfg, "cmbr_m"),
        grn_L=grn_L,
        grn_OD=grn_OD,
        noz_thrt=noz_thrt,
        noz_exit=noz_exit,
    )


def apply_layout_mass(cfg: dict[str, Any], lay: MotorLayout) -> tuple[float, float, float, float]:
    """Return (mtr_m, mtr_cg, tnk_X, cmbr_X) in SI. Component dry mass replaces lumped empty mass."""
    legacy_m = _mass(cfg, "mtr_m")
    legacy_cg = _len(cfg, "mtr_cg")
    if lay.dry_mass > 0.0:
        return lay.dry_mass, lay.dry_cg, lay.tnk_aft, lay.grain_aft
    return legacy_m, legacy_cg, lay.tnk_aft, lay.grain_aft


def resolve(cfg: dict[str, Any], get_sat_props=None) -> tuple[Settings, State]:
    """Mirror MATLAB RunSimulation unit conversion and initial-state setup."""
    prop_key = cfg.get("prop_id") or cfg.get("prop_nm") or "ABS"
    prop = load_propellant(str(prop_key))

    adv = cfg.get("advanced") or {}
    fluid_name = str(adv.get("ox_fluid") or "") if adv.get("enabled") else ""
    cp_fluid = "Oxygen" if "Oxygen" in fluid_name else "NitrousOxide"
    # HEM and Dyer take their injector flow from CoolProp, so the tank uses CoolProp too
    if get_sat_props is None and ("CoolProp" in fluid_name or cfg.get("inj_model", "SPI") != "SPI"):
        from hrap.advanced.fluid import coolprop_sat
        get_sat_props = coolprop_sat(cp_fluid)

    grn_OD = cfg["grn_OD"] * to_si(1.0, cfg["grn_OD_unit"], "length")
    grn_ID = cfg["grn_ID"] * to_si(1.0, cfg["grn_ID_unit"], "length")
    grn_L = cfg["grn_L"] * to_si(1.0, cfg["grn_L_unit"], "length")
    cstar_eff = float(cfg["cstar_eff"]) / 100.0
    Pa = cfg["Pa"] * to_si(1.0, cfg["Pa_unit"], "pressure")
    P_cmbr = cfg["P_cmbr"] * to_si(1.0, cfg["P_cmbr_unit"], "pressure")

    tnk_D = cfg["tnk_D"] * to_si(1.0, cfg["tnk_D_unit"], "length")
    tnk_L = cfg["tnk_L"] * to_si(1.0, cfg["tnk_L_unit"], "length")
    if int(cfg.get("tnk_V_state", 0)):
        tnk_V = tnk_L * 0.25 * math.pi * tnk_D ** 2
    else:
        tnk_V = cfg["tnk_V"] * to_si(1.0, cfg["tnk_V_unit"], "volume")

    if int(cfg.get("cmbr_V_state", 0)):
        cmbr_V = grn_L * 0.25 * math.pi * grn_OD ** 2
    else:
        cmbr_V = cfg["cmbr_V"] * to_si(1.0, cfg["cmbr_V_unit"], "volume")

    noz_thrt = cfg["noz_thrt"] * to_si(1.0, cfg["noz_thrt_unit"], "length")
    if cfg.get("noz_def") == "Nozzle Exit Diameter":
        noz_ext = cfg["noz_ex"] * to_si(1.0, cfg["noz_ex_unit"], "length")
        noz_ER = (noz_ext ** 2) / (noz_thrt ** 2) if noz_thrt != 0 else 1.0
    else:
        noz_ER = float(cfg["noz_ex"])

    inj_D = cfg["inj_D"] * to_si(1.0, cfg["inj_D_unit"], "length")
    inj_CdA = 0.25 * math.pi * inj_D ** 2 * float(cfg["inj_Cd"])
    vnt_S = int(VENT_MAP.get(cfg.get("vnt_state", "None"), 0))
    vnt_D = cfg["vnt_D"] * to_si(1.0, cfg["vnt_D_unit"], "length")
    vnt_CdA = 0.25 * math.pi * vnt_D ** 2 * float(cfg.get("vnt_Cd") or 0.0)

    mp_calc = 1 if cfg.get("mp_state") else 0
    lay = resolve_layout(cfg)
    mtr_m, mtr_cg, tnk_X, cmbr_X = apply_layout_mass(cfg, lay)

    rho = cfg["prop_rho"] * to_si(1.0, cfg["prop_rho_unit"], "density")
    reg = np.array([float(cfg["prop_a"]), float(cfg["prop_n"]), float(cfg["prop_m"])], dtype=float)

    s = Settings(
        dt=float(cfg["dt"]),
        tmax=float(cfg["t_max"]),
        tburn=float(cfg["t_burn"]),
        Pa=Pa,
        tnk_V=tnk_V,
        tnk_D=tnk_D,
        inj_CdA=inj_CdA,
        inj_N=int(cfg["inj_N"]),
        vnt_S=vnt_S,
        vnt_CdA=vnt_CdA,
        grn_OD=grn_OD,
        grn_L=grn_L,
        grn_ID0=grn_ID,
        prop_Rho=rho,
        prop_Reg=reg,
        const_OF=float(cfg["const_OF"]),
        regression_model=str(cfg.get("reg_model") or "Constant OF"),
        cmbr_V=cmbr_V,
        cstar_eff=cstar_eff,
        noz_thrt=noz_thrt,
        noz_ER=noz_ER,
        noz_Cd=float(cfg["noz_Cd"]),
        noz_eff=float(cfg["noz_eff"]) / 100.0,
        mp_calc=mp_calc,
        mtr_m=mtr_m,
        mtr_cg=mtr_cg,
        tnk_X=tnk_X,
        cmbr_X=cmbr_X,
        mtr_nm=str(cfg.get("mtr_nm") or "mtr_cfg"),
        prop_nm=prop.name,
        prop_OF=prop.OF,
        prop_Pc=prop.Pc,
        prop_k=prop.k,
        prop_M=prop.M,
        prop_T=prop.T,
        get_sat_props=get_sat_props,
    )
    if adv.get("live_chem"):
        from hrap.advanced.chem import live_propellant_tables

        live = live_propellant_tables(prop)
        s.prop_OF = live.OF
        s.prop_Pc = live.Pc
        s.prop_k = live.k
        s.prop_M = live.M
        s.prop_T = live.T
        s.prop_nm = f"{prop.name} (live chem)"

    # Tank temperature
    if cfg.get("tnk_dd") == "Starting Tank Pressure":
        P_targ = cfg["tnk_cond"] * to_si(1.0, cfg["T_tnk_unit"], "pressure")
        from hrap.engine.fzero import matlab_fzero
        from hrap.engine.nox import vapor_pressure
        from scipy.optimize import brentq

        def residual(T: float) -> float:
            if get_sat_props is not None:
                return get_sat_props(T).Pv - P_targ
            return vapor_pressure(T) - P_targ

        try:
            T_tnk = float(brentq(residual, 183.15, 309.56, xtol=2.2e-16, maxiter=200))
        except ValueError:
            T_tnk = matlab_fzero(residual, 273.15)
    else:
        T_tnk = to_si(float(cfg["tnk_cond"]), cfg["T_tnk_unit"], "temperature")

    ox = get_sat_props(T_tnk) if get_sat_props is not None else nox(T_tnk)
    if cfg.get("fill_dd") == "Tank Fill Percentage":
        fill = float(cfg["fill"]) / 100.0
        m_o = fill * tnk_V * ox.rho_l + (1.0 - fill) * tnk_V * ox.rho_v
    else:
        unit = str(cfg.get("fill_unit") or "kg")
        if unit == "%":
            fill = float(cfg["fill"]) / 100.0
            m_o = fill * tnk_V * ox.rho_l + (1.0 - fill) * tnk_V * ox.rho_v
        else:
            m_o = float(cfg["fill"]) * to_si(1.0, unit, "mass")

    mLiq_new = (tnk_V - (m_o / ox.rho_v)) / ((1.0 / ox.rho_l) - (1.0 / ox.rho_v))
    m_f = 0.25 * math.pi * (grn_OD ** 2 - grn_ID ** 2) * rho * grn_L
    m_g = 1.225 * (cmbr_V - 0.25 * math.pi * (grn_OD ** 2 - grn_ID ** 2) * grn_L)
    OF0 = float(cfg["const_OF"]) if s.regression_model == "Constant OF" else 0.0

    x = State(
        T_tnk=T_tnk,
        P_tnk=ox.Pv,
        P_cmbr=P_cmbr,
        m_o=m_o,
        mLiq_new=mLiq_new,
        mLiq_old=mLiq_new + 1.0,
        mdot_o=0.0,
        mdot_v=0.0,
        mdot_f=0.0,
        mdot_n=0.0,
        OF=OF0,
        rdot=0.0,
        grn_ID=grn_ID,
        grn_ID_old=grn_ID,
        m_f=m_f,
        m_g=m_g,
        dP=0.0,
        F_thr=0.0,
        ox_props=ox,
    )
    if adv.get("enabled") and adv.get("grain_shape") == "star":
        from hrap.advanced.geometry import make_star_grain_fn
        s.grain_fn = make_star_grain_fn(int(adv.get("star_tips") or 6))
    if cfg.get("inj_model", "SPI") != "SPI":
        from hrap.advanced.injector import hem_flux_table
        s.inj_model = str(cfg["inj_model"])
        s.inj_CdA_HEM = 0.25 * math.pi * inj_D ** 2 * float(cfg.get("inj_Cd_HEM") or cfg["inj_Cd"])
        s.dyer_kappa = float(cfg.get("dyer_kappa") or 1.0)
        s.hem_flux = hem_flux_table(cp_fluid)
    return s, x


def bundled_motor(name: str) -> dict[str, Any]:
    from importlib import resources

    fname = f"{name}.json"
    try:
        text = resources.files("hrap.resources.motors").joinpath(fname).read_text(encoding="utf-8")
        return _merge_cfg(json.loads(text))
    except Exception:
        path = Path(__file__).resolve().parents[1] / "resources" / "motors" / fname
        return _merge_cfg(json.loads(path.read_text(encoding="utf-8")))


def clone_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(cfg)
