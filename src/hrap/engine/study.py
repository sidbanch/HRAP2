"""Parameter studies: run a motor over a grid of input values and fuel-flow models, in parallel."""
from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from hrap.engine.sim import run
from hrap.engine.summary import summarize
from hrap.io.config import clone_cfg, injector_cd, resolve
from hrap.units import to_si


def _set_length(key: str) -> Callable[[dict, float], None]:
    def apply(cfg: dict, value: float):
        cfg.update({key: value, f"{key}_unit": "m"})
    return apply


def _set_cd(cfg: dict, value: float):
    cfg.update(inj_Cd=value, inj_Cd_HEM=None, sw_cd_from_geometry=False)


def _set_cda(cfg: dict, value: float):
    """Total CdA over every injector, reached by setting the Cd on the injector's own area."""
    D = float(cfg["inj_D"]) * _length_si(cfg.get("inj_D_unit") or "in")
    _set_cd(cfg, value / (int(cfg["inj_N"]) * 0.25 * math.pi * D ** 2))


def _set_temperature(cfg: dict, value: float):
    cfg.update(tnk_dd="Starting Tank Temperature", tnk_cond=value, T_tnk_unit="K")


def _set_fill(cfg: dict, value: float):
    cfg.update(fill_dd="Tank Fill Percentage", fill=value)


def _scale_a(cfg: dict, value: float):
    cfg["prop_a"] = float(cfg["prop_a"]) * value


def _length_si(unit: str) -> float:
    return to_si(1.0, unit, "length")


@dataclass(frozen=True)
class Input:
    label: str
    quantity: str | None      # display quantity for the value; None for a bare number
    apply: Callable[[dict, float], None]
    current: Callable[[dict], float]


def _current_length(key: str) -> Callable[[dict], float]:
    return lambda cfg: float(cfg[key]) * _length_si(cfg.get(f"{key}_unit") or "in")


def total_cda(cfg: dict) -> float:
    """Total injector Cd × area (m²) over every injector."""
    D = _current_length("inj_D")(cfg)
    return int(cfg["inj_N"]) * 0.25 * math.pi * D ** 2 * injector_cd(cfg)


def _current_temperature(cfg: dict) -> float:
    if cfg.get("tnk_dd") == "Starting Tank Pressure":
        return resolve(cfg)[1].T_tnk
    return to_si(float(cfg["tnk_cond"]), cfg.get("T_tnk_unit") or "K", "temperature")


INPUTS: dict[str, Input] = {
    "throat": Input("Throat", "length", lambda c, v: c.update(noz_thrt=v, noz_thrt_unit="m"), _current_length("noz_thrt")),
    "inj_Cd": Input("Injector Cd", None, _set_cd, injector_cd),
    "inj_CdA": Input("Injector CdA (total)", "area", _set_cda, total_cda),
    "grain_L": Input("Grain length", "length", _set_length("grn_L"), _current_length("grn_L")),
    "port_D": Input("Starting port", "length", _set_length("grn_ID"), _current_length("grn_ID")),
    "tank_T": Input("Tank temperature", "temperature", _set_temperature, _current_temperature),
    "fill": Input("Fill (%)", None, _set_fill, lambda c: float(c.get("fill") or 0.0)),
    "OF": Input("Fixed O/F", None, lambda c, v: c.update(const_OF=v), lambda c: float(c.get("const_OF") or 6.0)),
    "a_scale": Input("Burn rate a ×", None, _scale_a, lambda c: 1.0),
    "cstar_eff": Input("C* efficiency (%)", None, lambda c, v: c.update(cstar_eff=v), lambda c: float(c["cstar_eff"])),
}

MODELS = {"Shifting OF": "Burn-rate law", "Constant OF": "Fixed O/F"}


@dataclass(frozen=True)
class Case:
    values: tuple[tuple[str, float], ...]  # (input key, SI value) in study order
    model: str                             # HRAP regression model the case ran with
    peak_P_cmbr: float    # Pa, absolute
    avg_inj_dP: float     # Pa, tank minus chamber, averaged over the burn; skips the chamber-filling spike
    total_impulse: float  # N·s
    peak_thrust: float    # N
    avg_thrust: float     # N
    burn_time: float      # s
    liquid_time: float    # s, when the liquid runs out
    OF_liquid: float      # average O/F over the liquid burn
    fuel_burned: float    # kg
    port_end: float       # m, largest port diameter reached
    burnout: bool         # the port reached the grain's outside diameter
    end_cond: str
    outside_table: bool
    traces: dict[str, np.ndarray] = field(repr=False, compare=False)

    def value(self, key: str) -> float:
        return dict(self.values)[key]


OUTPUTS = [  # (label, Case field, display quantity)
    ("Peak chamber pressure", "peak_P_cmbr", "pressure"),
    ("O/F over the liquid burn", "OF_liquid", "ratio"),
    ("Peak thrust", "peak_thrust", "force"),
    ("Average thrust", "avg_thrust", "force"),
    ("Total impulse", "total_impulse", "impulse"),
    ("Liquid runout time", "liquid_time", None),
    ("Burn time", "burn_time", None),
    ("Fuel burned", "fuel_burned", "mass"),
    ("Port at the end", "port_end", "length"),
    ("Average injector ΔP", "avg_inj_dP", "pressure"),
]


def case_cfg(cfg: dict[str, Any], values: Sequence[tuple[str, float]], model: str | None) -> dict[str, Any]:
    """The motor with the study's input values and fuel-flow model applied."""
    c = clone_cfg(cfg)
    for key, value in values:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{INPUTS[key].label} must be finite and above zero.")
        if key in ("fill", "cstar_eff") and value > 100:
            raise ValueError(f"{INPUTS[key].label} cannot exceed 100%.")
        INPUTS[key].apply(c, float(value))
    if model:
        c["reg_model"] = model
    return c


def run_case(cfg: dict[str, Any], values: Sequence[tuple[str, float]], model: str | None) -> Case:
    c = case_cfg(cfg, values, model)
    s, x = resolve(c)
    if not 0 < s.grn_ID0 < s.grn_OD or s.grn_L <= 0:
        raise ValueError("Starting port must be smaller than the grain outside diameter; length must be positive.")
    if s.noz_ER < 1:
        raise ValueError("Nozzle exit must be at least as large as the throat.")
    x, o = run(s, x)
    if not all(np.all(np.isfinite(getattr(o, name))) for name in ("P_cmbr", "F_thr", "grn_ID", "mdot_o", "mdot_f")):
        raise ValueError("Simulation produced non-finite results; check the motor inputs and timestep.")
    info = summarize(s, x, o)
    t, mdot_o, mdot_f = o.t, o.mdot_o, o.mdot_f
    burning = o.F_thr > 0
    inj_dP = o.P_tnk[burning] - o.P_cmbr[burning]
    liquid_time = o.extra.get("liquid_runout_time", float("nan"))
    k = int(np.searchsorted(t, liquid_time)) if math.isfinite(liquid_time) else len(t) - 1
    dt = np.diff(t[:k + 1])
    fuel = float(np.sum(mdot_f[1:k + 1] * dt))
    ox = float(np.sum(mdot_o[1:k + 1] * dt))
    indices = np.unique(np.linspace(0, len(t) - 1, min(1000, len(t))).astype(int))
    OD = float(c["grn_OD"]) * _length_si(c.get("grn_OD_unit") or "in")
    port_end = float(np.max(o.grn_ID))
    return Case(
        values=tuple((key, float(v)) for key, v in values),
        model=c.get("reg_model") or "Shifting OF",
        peak_P_cmbr=float(np.max(o.P_cmbr)),
        avg_inj_dP=float(np.mean(inj_dP)) if inj_dP.size else 0.0,
        total_impulse=info["total_impulse"],
        peak_thrust=info["peak_thrust"],
        avg_thrust=info["avg_thrust"],
        burn_time=info["burn_time"],
        liquid_time=liquid_time,
        OF_liquid=ox / fuel if fuel > 0 else float("nan"),
        fuel_burned=float(info["fuel_consumed"]),
        port_end=port_end,
        burnout=o.sim_end_cond == "Fuel Depleted" or port_end >= OD * (1 - 1e-6),
        end_cond=info["end_cond"],
        outside_table=bool(np.any((o.OF[burning] < np.min(s.prop_OF)) | (o.OF[burning] > np.max(s.prop_OF)))),
        traces={name: getattr(o, name)[indices] for name in ("t", "F_thr", "P_cmbr", "OF", "grn_ID")},
    )


def grid(axes: Sequence[tuple[str, Sequence[float]]]) -> list[tuple[tuple[str, float], ...]]:
    """Every combination of the axes' values, as (key, value) pairs."""
    keys = [key for key, _ in axes]
    return [tuple(zip(keys, combo)) for combo in itertools.product(*(values for _, values in axes))]


def study(cfg: dict[str, Any], axes: Sequence[tuple[str, Sequence[float]]],
          models: Sequence[str | None] = (None,),
          stopped: Callable[[], bool] = lambda: False) -> Iterator[Case]:
    """Run combinations in parallel; stop queues no further cases and drains running workers."""
    keys = [key for key, _ in axes]
    if len(set(keys)) != len(keys):
        raise ValueError("Choose different inputs for the two axes.")
    if "OF" in keys and any((m or cfg.get("reg_model")) != "Constant OF" for m in models):
        raise ValueError("Fixed O/F only varies the fixed-O/F model. Choose that model to sweep it.")
    if "a_scale" in keys and any((m or cfg.get("reg_model")) != "Shifting OF" for m in models):
        raise ValueError("Burn rate a only varies the burn-rate-law model. Choose that model to sweep it.")
    jobs = [(values, model) for values in grid(axes) for model in models]
    if not jobs:
        raise ValueError("Choose at least one value and fuel model.")
    workers = min(len(jobs), 8, max(1, (os.cpu_count() or 2) - 1))
    pending = iter(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        def submit():
            job = next(pending, None)
            if job is not None:
                values, model = job
                futures[pool.submit(run_case, cfg, values, model)] = job
        for _ in range(workers):
            if not stopped():
                submit()
        while futures:
            done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                values, model = futures.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    for f in futures:
                        f.cancel()
                    raise ValueError(f"Case {values}, {model or cfg.get('reg_model')}: {exc}") from exc
                if not stopped():
                    submit()


def uses_spi(cfg: dict[str, Any]) -> bool:
    """True for HRAP's liquid-only injector model, whose flow is only trustworthy under the ΔP warning."""
    return cfg.get("inj_model", "SPI") == "SPI"


def within_limits(case: Case, min_P_cmbr: float, max_P_cmbr: float, max_inj_dP: float) -> bool:
    return (min_P_cmbr <= case.peak_P_cmbr <= max_P_cmbr and case.avg_inj_dP <= max_inj_dP and not case.burnout
            and case.end_cond != "Max Simulation Time Reached")


def passing_values(cases: Sequence[Case], key: str, min_P_cmbr: float, max_P_cmbr: float,
                   max_inj_dP: float) -> list[float]:
    """Values of one input that stay within the limits for every other input value and model studied."""
    ok: dict[float, bool] = {}
    for c in cases:
        v = c.value(key)
        ok[v] = ok.get(v, True) and within_limits(c, min_P_cmbr, max_P_cmbr, max_inj_dP)
    return sorted(v for v, passed in ok.items() if passed)
