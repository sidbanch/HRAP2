import math

import pytest

from hrap.engine.study import case_cfg, run_case, study, total_cda
from hrap.io.config import bundled_motor, clone_cfg, resolve


def test_case_changes_are_isolated_and_cda_is_total():
    cfg = bundled_motor("Rattworks_K240")
    original = clone_cfg(cfg)
    c = case_cfg(cfg, [("inj_CdA", total_cda(cfg) * 1.5), ("a_scale", 2)], "Shifting OF")
    s, _ = resolve(c)
    assert s.inj_CdA * s.inj_N == pytest.approx(total_cda(cfg) * 1.5)
    assert c["prop_a"] == cfg["prop_a"] * 2
    assert cfg == original


def test_liquid_phase_metrics_do_not_confuse_early_stop_with_runout():
    cfg = bundled_motor("Rattworks_K240")
    full = run_case(cfg, [], "Constant OF")
    assert 4 < full.liquid_time < 6
    assert full.OF_liquid == pytest.approx(cfg["const_OF"])
    cfg["t_max"] = .02
    short = run_case(cfg, [], "Constant OF")
    assert math.isnan(short.liquid_time)
    assert short.OF_liquid == pytest.approx(cfg["const_OF"])
    assert short.end_cond == "Max Simulation Time Reached"


def test_invalid_model_axis_and_cancellation_before_start():
    cfg = bundled_motor("Rattworks_K240")
    with pytest.raises(ValueError, match="only applies to the Fixed O/F model"):
        list(study(cfg, [("OF", [6, 7])], ["Shifting OF"]))
    with pytest.raises(ValueError, match="Starting port"):
        run_case(cfg, [("port_D", 10)], "Constant OF")
    assert list(study(cfg, [("OF", [6, 7])], ["Constant OF"], stopped=lambda: True)) == []
