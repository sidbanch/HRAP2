from __future__ import annotations

import pytest


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
