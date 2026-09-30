from __future__ import annotations

import pytest

from hrap.advanced.geometry import star_vertices
from hrap.engine.sim import run
from hrap.io.config import bundled_motor, resolve


def test_star_vertices_closed():
    v = star_vertices(0.01, 0.02, 6)
    assert v.shape == (12, 2)


def test_star_grain_does_not_crash():
    cfg = bundled_motor("example_98mm")
    cfg["advanced"] = {"enabled": True, "ox_fluid": "N2O_legacy", "grain_shape": "star", "star_tips": 6}
    s, x = resolve(cfg)
    assert s.grain_fn is not None
    _x, o = run(s, x)
    assert o.t.size > 10
    assert o.sim_end_cond


def test_polygon_grain_shift_of():
    from hrap.advanced.geometry import make_polygon_grain_fn, star_vertices

    cfg = bundled_motor("example_98mm")
    cfg["reg_model"] = "Shifting OF"
    cfg["prop_a"] = 0.198
    cfg["prop_n"] = 0.325
    cfg["advanced"] = {"enabled": True, "grain_shape": "cylindrical"}
    s, x = resolve(cfg)
    s.grain_fn = make_polygon_grain_fn(star_vertices(0.01, 0.02, 5))
    _x, o = run(s, x)
    assert o.t.size > 10


def test_chem_solver_one_point():
    from pathlib import Path

    from hrap.advanced.chem import ChemSolver, _thermo_path, make_basic_reactant

    path = _thermo_path()
    if not Path(path).exists():
        pytest.skip("thermo.dat missing")
    fuel = make_basic_reactant("HDPE", {"C": 2, "H": 4}, 28.05, 298.15, -52e6)
    solver = ChemSolver([path, fuel])
    res = solver.solve(3.0e6, {"N2O": 0.87, "HDPE": 0.13})
    assert res.T > 800.0
    assert 1.05 < res.gamma < 1.8
    assert res.M > 5.0
