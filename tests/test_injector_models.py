import pytest

pytest.importorskip("CoolProp")

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
        cfg["advanced"].update(enabled=model != "SPI", inj_model=model)
        s, x = resolve(cfg)
        _x, o = run(s, x)
        peaks[model] = o.mdot_o[1:200].mean()
    assert peaks["HEM"] < peaks["Dyer"] < peaks["SPI"]
