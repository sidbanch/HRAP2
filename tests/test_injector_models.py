import pytest

from hrap.advanced.injector import hem_flux_table
from hrap.engine.sim import run
from hrap.io.config import bundled_motor, resolve

PSI = 6894.757


def test_hem_flux_chokes_at_the_coolprop_value():
    flux = hem_flux_table()
    # Saturated N2O at 20 °C (733 psi): choked HEM flux is about 2.91 kg/s per cm²
    assert flux(293.15, 400 * PSI / (733 * PSI)) == pytest.approx(2.91e4, rel=0.01)
    assert flux(293.15, 100 / 733) == pytest.approx(flux(293.15, 400 / 733), rel=1e-9)
    assert flux(293.15, 1.0) == 0.0


def test_dyer_flow_is_between_spi_and_hem():
    peaks = {}
    for model in ("SPI", "HEM", "Dyer"):
        cfg = bundled_motor("example_98mm")
        cfg["inj_model"] = model
        s, x = resolve(cfg)
        _x, o = run(s, x)
        peaks[model] = o.mdot_o[1:200].mean()
    assert peaks["HEM"] < peaks["Dyer"] < peaks["SPI"]


def test_cda_curve_sets_the_injector_flow_over_the_burn():
    cfg = bundled_motor("example_98mm")
    s, x = resolve(cfg)
    total = s.inj_CdA * s.inj_N / 0.0254 ** 2  # in²
    _x, constant = run(s, x)
    _x, flat = run(*resolve(dict(cfg, inj_cda_curve=[[0.0, total], [4.0, total]], inj_cda_curve_unit="in^2")))
    _x, falling = run(*resolve(dict(cfg, inj_cda_curve=[[1.0, total], [4.0, total / 2]], inj_cda_curve_unit="in^2")))
    assert flat.t.size == constant.t.size
    assert flat.mdot_o == pytest.approx(constant.mdot_o, rel=1e-9)
    assert flat.F_thr == pytest.approx(constant.F_thr, rel=1e-9)
    i, late = (abs(constant.t - t).argmin() for t in (0.5, 6.0))
    assert falling.mdot_o[i] == pytest.approx(constant.mdot_o[i], rel=1e-9)  # held at the first value before 1 s
    assert falling.mdot_o[late] < 0.9 * constant.mdot_o[late]
    assert falling.inj_CdA[late] == pytest.approx(total / 2 * 0.0254 ** 2)  # held at the last value after 4 s
