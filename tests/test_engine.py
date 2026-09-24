from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrap.engine.sim import run
from hrap.io.config import bundled_motor, default_cfg, resolve, save_json
from hrap.io.export import export_csv, export_eng, export_rse
from hrap.io.propellant import list_propellants, load_propellant

GOLDEN = Path(__file__).parent / "golden"


def test_propellant_presets_load():
    items = list_propellants()
    names = {i["id"] for i in items}
    assert "ABS" in names
    assert "HDPE" in names
    p = load_propellant("ABS")
    assert p.k.shape[1] == p.OF.size
    assert p.k.shape[0] == p.Pc.size


def test_example_98mm_const_of_runs(tmp_path):
    cfg = bundled_motor("example_98mm")
    s, x = resolve(cfg)
    assert s.regression_model == "Constant OF"
    assert s.vnt_S == 2
    assert s.prop_nm == "ABS"
    x, o = run(s, x)
    assert o.t.size > 50
    assert o.sim_end_cond in {
        "Oxidizer Depleted",
        "Fuel Depleted",
        "Burn Complete",
        "Max Simulation Time Reached",
    }
    assert np.max(o.F_thr) > 10.0
    assert np.all(np.isfinite(o.F_thr))
    assert np.all(np.isfinite(o.P_tnk))
    assert np.all(np.isfinite(o.P_cmbr))
    GOLDEN.mkdir(exist_ok=True)
    export_csv(GOLDEN / "example_98mm_python.csv", o, s)
    export_rse(tmp_path / "m.rse", o, s)
    text = (tmp_path / "m.rse").read_text(encoding="utf-8")
    assert "cg=" in text
    assert 'auto-calc-cg="0"' in text
    cgs = [float(part.split('cg="')[1].split('"')[0]) for part in text.split() if 'cg="' in part]
    assert len(cgs) > 5
    assert max(cgs) - min(cgs) > 1.0
    export_eng(tmp_path / "m.eng", o, s)
    eng = (tmp_path / "m.eng").read_text(encoding="utf-8")
    assert "cg0=" in eng
    info_path = GOLDEN / "example_98mm_python.csv"
    assert info_path.exists()


def test_shift_of_abs_runs():
    cfg = bundled_motor("example_98mm")
    cfg["reg_model"] = "Shifting OF"
    cfg["prop_a"] = 0.198
    cfg["prop_n"] = 0.325
    cfg["prop_m"] = 0.0
    s, x = resolve(cfg)
    x, o = run(s, x)
    assert o.t.size > 20
    assert np.all(np.isfinite(o.F_thr))
    GOLDEN.mkdir(exist_ok=True)
    export_csv(GOLDEN / "example_98mm_shift_python.csv", o, s)


def test_self_golden_repeatable():
    pytest.importorskip("numpy")
    path = GOLDEN / "example_98mm_python.csv"
    if not path.exists():
        pytest.skip("golden not generated yet")
    cfg = bundled_motor("example_98mm")
    s, x = resolve(cfg)
    _x, o = run(s, x)
    gold = np.genfromtxt(path, delimiter=",", names=True)
    n = min(o.t.size, gold.size)
    denom = np.maximum(np.abs(gold["F_thr"][:n]), 1e-8)
    rel = np.max(np.abs(o.F_thr[:n] - gold["F_thr"][:n]) / denom)
    assert rel < 1e-12


def test_json_roundtrip(tmp_path):
    cfg = default_cfg()
    p = tmp_path / "m.json"
    save_json(p, cfg)
    from hrap.io.config import load_json
    loaded = load_json(p)
    assert loaded["mtr_nm"] == cfg["mtr_nm"]


def test_rattworks_k240_runs():
    cfg = bundled_motor("Rattworks_K240")
    s, x = resolve(cfg)
    assert s.prop_nm == "HDPE"
    assert s.vnt_S == 1
    x, o = run(s, x)
    assert o.t.size > 20
    assert np.all(np.isfinite(o.F_thr))
    assert np.max(o.F_thr) > 1.0



def test_solved_tank_cooling_skips_the_fallback_and_stays_close():
    from hrap.engine.summary import summarize

    results = {}
    for fix in (False, True):
        cfg = bundled_motor("example_98mm")
        cfg["solve_tank_cooling"] = fix
        s, x = resolve(cfg)
        x, o = run(s, x)
        liquid = o.t < 8.2  # liquid runs out at about 8.25 s
        results[fix] = (summarize(s, x, o)["total_impulse"], o.P_tnk[liquid])
    (hrap_impulse, _), (impulse, P_tnk) = results[False], results[True]
    assert np.all(np.diff(P_tnk) <= 0)  # a draining tank only cools
    assert impulse == pytest.approx(hrap_impulse, rel=0.01)
