from pathlib import Path

import numpy as np

from hrap.engine.mass import mass_properties
from hrap.engine.types import Output
from hrap.io.config import apply_layout_mass, bundled_motor, default_cfg, load_json, resolve, resolve_layout
from hrap.io.export import export_eng, export_rse
from hrap.layout import FEED_GAP, INCH, PLATE_L, infer_component_stations, motor_layout, nozzle_cone_lengths, nozzle_exit_diameter
from hrap.units import to_si

HPS01_MASSED = Path(__file__).resolve().parent / "fixtures" / "HPS01-01_Massed.json"


def test_packed_layout_matches_legacy_plate_stack():
    lay = motor_layout(
        tnk_start=0.0,
        tnk_L=0.20,
        cmbr_start=None,
        grn_L=0.40,
        grn_OD=0.08,
        noz_thrt=0.025,
        noz_exit=0.05,
    )
    assert abs(lay.tnk1 - 0.20) < 1e-12
    assert abs(lay.plate0 - 0.20) < 1e-12
    assert abs(lay.plate1 - lay.plate0 - PLATE_L) < 1e-12
    assert lay.grn0 == lay.plate1
    assert lay.x_max >= lay.x_noz


def test_gap_between_tank_and_chamber():
    lay = motor_layout(
        tnk_start=0.10,
        tnk_L=0.20,
        cmbr_start=0.40,
        tnk_m=4.0,
        cmbr_m=6.0,
        grn_L=0.30,
        grn_OD=0.08,
        noz_thrt=0.025,
        noz_exit=0.05,
    )
    assert abs(lay.cmbr0 - lay.tnk1 - 0.10) < 1e-12
    assert abs(lay.dry_mass - 10.0) < 1e-12
    assert abs(lay.dry_cg - (4.0 * 0.20 + 6.0 * (0.40 + 0.5 * lay.cmbr_L)) / 10.0) < 1e-12
    assert abs(lay.overall_L - (lay.x_noz - 0.10)) < 1e-9
    assert abs(lay.overall_OD - 0.08) < 1e-12


def test_overall_od_is_widest_of_tank_and_tca():
    tank = motor_layout(
        tnk_start=0.0,
        tnk_L=0.20,
        tnk_D=0.12,
        grn_L=0.30,
        grn_OD=0.08,
        noz_thrt=0.025,
        noz_exit=0.05,
    )
    tca = motor_layout(
        tnk_start=0.0,
        tnk_L=0.20,
        tnk_D=0.06,
        grn_L=0.30,
        grn_OD=0.08,
        noz_thrt=0.025,
        noz_exit=0.05,
    )
    assert abs(tank.overall_OD - 0.12) < 1e-12
    assert abs(tca.overall_OD - 0.08) < 1e-12


def test_rse_dia_uses_widest_of_tank_and_tca(tmp_path):
    cfg = default_cfg()
    cfg["tnk_D"] = 5.0
    cfg["tnk_D_unit"] = "in"
    cfg["grn_OD"] = 3.625
    cfg["grn_OD_unit"] = "in"
    s, _x = resolve(cfg)
    n = 8
    o = Output(
        t=np.linspace(0.0, 1.0, n),
        m_o=np.linspace(1.0, 0.2, n),
        P_tnk=np.ones(n),
        P_cmbr=np.ones(n),
        mdot_o=np.ones(n) * 0.1,
        mdot_f=np.ones(n) * 0.02,
        OF=np.ones(n) * 5.0,
        grn_ID=np.ones(n) * s.grn_ID0,
        mdot_n=np.ones(n) * 0.12,
        rdot=np.zeros(n),
        m_f=np.linspace(0.5, 0.2, n),
        F_thr=np.ones(n) * 80.0,
        dP=np.zeros(n),
    )
    path = tmp_path / "wide.rse"
    export_rse(path, o, s, mfg="HCAT")
    dia_mm = float(path.read_text(encoding="utf-8").split('dia="')[1].split('"')[0])
    assert abs(dia_mm / 1000.0 - max(s.tnk_D, s.grn_OD)) < 1e-9


def test_infer_matlab_aft_stations():
    tnk_L = 1.0
    grn_L = 0.4
    tnk_X = 1.2
    cmbr_X = 1.8
    tnk0, cmbr0 = infer_component_stations(
        tnk_start=0.0,
        cmbr_start=0.0,
        tnk_L=tnk_L,
        grn_L=grn_L,
        pre_L=0.05,
        tnk_X=tnk_X,
        cmbr_X=cmbr_X,
    )
    assert abs(tnk0 - (tnk_X - tnk_L)) < 1e-12
    assert abs(cmbr0 - (cmbr_X - grn_L - 0.05 - PLATE_L)) < 1e-12


def test_new_motor_chamber_follows_tank():
    tnk0, cmbr0 = infer_component_stations(
        tnk_start=0.0,
        cmbr_start=0.0,
        tnk_L=0.3,
        grn_L=0.2,
        pre_L=0.0,
        tnk_X=0.0,
        cmbr_X=0.0,
    )
    assert tnk0 == 0.0
    assert abs(cmbr0 - (0.3 + FEED_GAP)) < 1e-12


def test_example_98mm_keeps_legacy_empty_mass():
    cfg = bundled_motor("example_98mm")
    s, _x = resolve(cfg)
    assert abs(s.mtr_m - 9.47769) < 1e-6
    assert abs(s.mtr_cg - to_si(52.33018, "in", "length")) < 1e-6
    assert abs(s.tnk_X - to_si(72.54685, "in", "length")) < 5e-3
    assert abs(s.cmbr_X - to_si(91.635, "in", "length")) < 5e-3


def test_component_dry_mass_replaces_legacy_empty():
    cfg = default_cfg()
    cfg["tnk_V_state"] = 1
    cfg["tnk_L"] = 15.0
    cfg["tnk_L_unit"] = "in"
    cfg["tnk_start"] = 0.0
    cfg["tnk_m"] = 4.0
    cfg["cmbr_start"] = 20.0
    cfg["cmbr_start_unit"] = "in"
    cfg["cmbr_m"] = 6.0
    cfg["mtr_m"] = 99.0
    lay = resolve_layout(cfg)
    mtr_m, mtr_cg, tnk_X, cmbr_X = apply_layout_mass(cfg, lay)
    assert abs(mtr_m - 10.0) < 1e-9
    assert abs(mtr_cg - lay.dry_cg) < 1e-12
    assert tnk_X == lay.tnk_aft
    assert cmbr_X == lay.grain_aft
    s, x = resolve(cfg)
    x.m_o = 2.0
    x.m_f = 1.0
    x.mLiq_new = 1.5
    from hrap.engine.nox import nox

    x.ox_props = nox(293.15)
    m_t, cg = mass_properties(s, x)
    assert abs(m_t - (10.0 + 2.0 + 1.0)) < 1e-9
    assert cg > 0.0


def test_pre_and_post_chambers_lengthen_the_chamber():
    lay = motor_layout(
        tnk_start=0.0,
        tnk_L=0.20,
        cmbr_start=0.25,
        pre_L=0.10,
        post_L=0.20,
        grn_L=0.20,
        grn_OD=0.08,
        noz_thrt=0.025,
        noz_exit=0.05,
    )
    L_conv, L_div = nozzle_cone_lengths(0.08, 0.025, 0.05)
    assert abs((lay.x_th - lay.x_case) - L_conv) < 1e-12
    assert abs((lay.x_noz - lay.x_th) - L_div) < 1e-12
    assert abs((lay.grn0 - lay.plate1) - 0.10) < 1e-12
    assert abs((lay.x_case - lay.grn1) - 0.20) < 1e-12
    assert abs(lay.cmbr_L - (PLATE_L + 0.10 + 0.20 + 0.20 + L_conv + L_div)) < 1e-12


def test_hps01_massed_layout():
    cfg = load_json(HPS01_MASSED)
    lay = resolve_layout(cfg)
    th = 1.0 * INCH
    exit_d = nozzle_exit_diameter(th, 3.2)
    L_conv, L_div = nozzle_cone_lengths(3.625 * INCH, th, exit_d)
    assert abs(lay.tnk_L - 45.0 * INCH) < 1e-6
    assert abs(lay.cmbr0 - 47.0 * INCH) < 1e-6
    assert abs(lay.cmbr_L - 24.0 * INCH) < 1e-6
    assert abs(lay.tnk_m - 14.0 * 0.453592) < 1e-6
    assert abs(lay.cmbr_m - 15.0 * 0.453592) < 1e-6
    assert abs(lay.overall_OD - 3.625 * INCH) < 1e-6
    assert abs((lay.x_noz - lay.x_th) - L_div) < 1e-9
    assert abs((lay.x_th - lay.x_case) - L_conv) < 1e-9
    assert abs((lay.grn0 - lay.plate1) - (lay.x_case - lay.grn1)) < 1e-9
    assert lay.grn0 - lay.plate1 > INCH
    s, x = resolve(cfg)
    assert s.mtr_nm == "HPS01-01"
    assert abs(s.mtr_m - 29.0 * 0.453592) < 1e-4
    assert x.m_o > 0.0


def _rse_init_wt_g(text: str) -> float:
    return float(text.split('initWt="')[1].split('"')[0])


def test_rse_initwt_includes_user_dry_mass_even_if_run_omitted_it(tmp_path):
    cfg = default_cfg()
    cfg["tnk_V_state"] = 1
    cfg["tnk_L"] = 15.0
    cfg["tnk_L_unit"] = "in"
    cfg["tnk_m"] = 4.0
    cfg["cmbr_start"] = 20.0
    cfg["cmbr_start_unit"] = "in"
    cfg["cmbr_m"] = 6.0
    s, _x = resolve(cfg)
    assert abs(s.mtr_m - 10.0) < 1e-9
    n = 25
    o = Output(
        t=np.linspace(0.0, 1.5, n),
        m_o=np.linspace(2.0, 0.2, n),
        P_tnk=np.ones(n),
        P_cmbr=np.ones(n),
        mdot_o=np.ones(n) * 0.1,
        mdot_f=np.ones(n) * 0.02,
        OF=np.ones(n) * 5.0,
        grn_ID=np.ones(n) * s.grn_ID0,
        mdot_n=np.ones(n) * 0.12,
        rdot=np.zeros(n),
        m_f=np.linspace(1.0, 0.4, n),
        F_thr=np.ones(n) * 150.0,
        dP=np.zeros(n),
        m_t=np.linspace(3.0, 0.6, n),
        cg=np.zeros(n),
    )
    path = tmp_path / "dry.rse"
    export_rse(path, o, s, mfg="HCAT")
    init_kg = _rse_init_wt_g(path.read_text(encoding="utf-8")) / 1000.0
    assert abs(init_kg - (10.0 + 2.0 + 1.0)) < 1e-6


def test_hps01_rse_has_shifting_cg(tmp_path):
    cfg = load_json(HPS01_MASSED)
    s, x = resolve(cfg)
    assert s.mp_calc == 0
    from hrap.engine.sim import run

    _x, o = run(s, x)
    path = tmp_path / "hps01.rse"
    export_rse(path, o, s, L=cfg.get("export_L"), mfg="HCAT")
    text = path.read_text(encoding="utf-8")
    init_kg = _rse_init_wt_g(text) / 1000.0
    wet0 = s.mtr_m + float(o.m_o[0]) + float(o.m_f[0])
    assert init_kg > s.mtr_m + 0.5
    assert abs(init_kg - wet0) < 0.02
    cgs = [float(part.split('cg="')[1].split('"')[0]) for part in text.split() if 'cg="' in part]
    assert o.cg is not None and o.cg.size == o.t.size
    assert len(cgs) > 5
    assert max(cgs) - min(cgs) > 1.0


def test_rse_rebuilds_shifting_cg_when_series_missing(tmp_path):
    cfg = bundled_motor("example_98mm")
    s, _x = resolve(cfg)
    n = 40
    o = Output(
        t=np.linspace(0.0, 2.0, n),
        m_o=np.linspace(3.0, 0.2, n),
        P_tnk=np.ones(n),
        P_cmbr=np.ones(n),
        mdot_o=np.ones(n) * 0.1,
        mdot_f=np.ones(n) * 0.02,
        OF=np.ones(n) * 5.0,
        grn_ID=np.ones(n) * s.grn_ID0,
        mdot_n=np.ones(n) * 0.12,
        rdot=np.zeros(n),
        m_f=np.linspace(1.5, 0.7, n),
        F_thr=np.ones(n) * 200.0,
        dP=np.zeros(n),
        m_t=None,
        cg=np.zeros(n),
    )
    path = tmp_path / "shift.rse"
    export_rse(path, o, s, mfg="HCAT")
    text = path.read_text(encoding="utf-8")
    assert 'auto-calc-cg="0"' in text
    cgs = [float(part.split('cg="')[1].split('"')[0]) for part in text.split() if 'cg="' in part]
    assert len(cgs) > 5
    assert max(cgs) - min(cgs) > 1.0


def test_eng_header_uses_layout_length_and_notes_cg(tmp_path):
    cfg = bundled_motor("example_98mm")
    cfg["tnk_m"] = 4.0
    cfg["cmbr_m"] = 6.0
    s, x = resolve(cfg)
    from hrap.engine.sim import run

    _x, o = run(s, x)
    path = tmp_path / "m.eng"
    export_eng(path, o, s, L=0.8)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("; HRAP-HCAT-Fork cg0=")
    header = next(line for line in text.splitlines() if not line.startswith(";"))
    parts = header.split()
    assert abs(float(parts[2]) - 800.0) < 1e-6
    assert float(parts[5]) > 10.0
