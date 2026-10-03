"""Compare Python engine traces to MATLAB-generated CSVs when present."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrap.engine.interp import interp2x
from hrap.engine.nox import nox
from hrap.engine.sim import run
from hrap.io.config import bundled_motor, resolve
from hrap.io.propellant import load_propellant

GOLDEN = Path(__file__).parent / "golden"
# Relative error is meaningless near burnout zeros; compare against peak-scaled abs too.
REL_TOL = 1e-8
ABS_FRAC = 1e-8
MATLAB_MAX_OF = 10.0  # MATLAB's tables stop here; HRAP2's ABS table continues to O/F 30


def _matlab_table(OF, *tables):
    """The combustion table MATLAB had: the O/F columns up to 10. Above that MATLAB reuses the O/F 10 values."""
    keep = np.asarray(OF) <= MATLAB_MAX_OF
    return (np.asarray(OF)[keep], *(np.asarray(t)[:, keep] for t in tables))


def _max_rel(a: np.ndarray, g: np.ndarray) -> float:
    n = min(a.size, g.size)
    peak = max(float(np.max(np.abs(g[:n]))), 1e-30)
    mask = np.abs(g[:n]) > ABS_FRAC * peak
    if not np.any(mask):
        return float(np.max(np.abs(a[:n] - g[:n])) / peak)
    denom = np.maximum(np.abs(g[:n][mask]), 1e-30)
    return float(np.max(np.abs(a[:n][mask] - g[:n][mask]) / denom))


def _compare_motor(name: str, csv_name: str, mutate=None, t_prefix: float = 5.0):
    path = GOLDEN / csv_name
    if not path.exists():
        pytest.skip(f"{csv_name} not generated (run scripts/make_matlab_golden.m)")
    cfg = bundled_motor(name)
    if mutate:
        mutate(cfg)
    s, x = resolve(cfg)
    s.prop_OF, s.prop_k, s.prop_M, s.prop_T = _matlab_table(s.prop_OF, s.prop_k, s.prop_M, s.prop_T)
    _x, o = run(s, x)
    gold = np.genfromtxt(path, delimiter=",", names=True)
    n = min(o.t.size, gold.size)
    # Peak thrust is a tight check that the Euler loop and tables match.
    peak_py = float(np.max(o.F_thr[:n]))
    peak_ml = float(np.max(gold["F_thr"][:n]))
    assert abs(peak_py - peak_ml) / max(abs(peak_ml), 1.0) <= REL_TOL
    # Prefix before MATLAB tank pressurizing branch (`mean(o.dP(1:sum(o.dP<0)))`).
    # After that discrete fzero/mean step, traces can drift; IC, peak thrust, and
    # the evaporating-liquid Euler prefix stay within 1e-8.
    m = n
    if t_prefix is not None:
        m = int(np.searchsorted(o.t[:n], t_prefix, side="right"))
        m = max(m, 1)
    assert _max_rel(o.F_thr[:m], gold["F_thr"][:m]) <= REL_TOL
    assert _max_rel(o.P_tnk[:m], gold["P_tnk"][:m]) <= REL_TOL
    assert _max_rel(o.P_cmbr[:m], gold["P_cmbr"][:m]) <= REL_TOL


def test_nox_vs_matlab_csv():
    path = GOLDEN / "nox_matlab.csv"
    if not path.exists():
        pytest.skip("nox_matlab.csv not generated")
    gold = np.genfromtxt(path, delimiter=",", names=True)
    for row in gold[::10]:
        op = nox(float(row["T"]))
        assert abs(op.Pv - row["Pv"]) / max(abs(row["Pv"]), 1.0) <= REL_TOL
        assert abs(op.rho_l - row["rho_l"]) / max(abs(row["rho_l"]), 1.0) <= REL_TOL
        assert abs(op.Z - row["Z"]) / max(abs(row["Z"]), 1e-6) <= 1e-10


def test_interp2x_vs_matlab_csv():
    path = GOLDEN / "interp2x_matlab.csv"
    if not path.exists():
        pytest.skip("interp2x_matlab.csv not generated")
    prop = load_propellant("ABS")
    OF, k = _matlab_table(prop.OF, prop.k)
    gold = np.genfromtxt(path, delimiter=",", names=True)
    for row in gold:
        zi = interp2x(OF, prop.Pc, k, float(row["OF"]), float(row["Pc"]))
        denom = max(abs(float(row["k"])), 1e-12)
        assert abs(zi - float(row["k"])) / denom <= REL_TOL


def test_example_98mm_vs_matlab():
    _compare_motor("example_98mm", "example_98mm_matlab.csv")


def test_example_98mm_shift_vs_matlab():
    def mutate(cfg):
        cfg["reg_model"] = "Shifting OF"
        cfg["prop_a"] = 0.198
        cfg["prop_n"] = 0.325
        cfg["prop_m"] = 0.0

    _compare_motor("example_98mm", "example_98mm_shift_matlab.csv", mutate)


def test_rattworks_vs_matlab():
    # Liquid pressurizing (MATLAB mean(dP(1:nneg))) begins ~3.54 s on this motor.
    _compare_motor("Rattworks_K240", "Rattworks_K240_matlab.csv", t_prefix=3.4)
