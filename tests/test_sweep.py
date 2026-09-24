import pytest

from hrap.engine.sim import run
from hrap.engine.summary import summarize
from hrap.engine.sweep import sweep
from hrap.io.config import bundled_motor, resolve


def test_sweep_case_matches_a_normal_run():
    cfg = bundled_motor("Rattworks_K240")
    s, x = resolve(cfg)
    x, o = run(s, x)
    (case,) = sweep(cfg, [s.noz_thrt], [cfg["inj_Cd"]])
    assert case.total_impulse == pytest.approx(summarize(s, x, o)["total_impulse"], rel=1e-12)
    assert case.peak_P_cmbr == pytest.approx(o.P_cmbr.max(), rel=1e-12)
