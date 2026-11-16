"""Physical balance, choking, and runout checks for the optional equilibrium tank."""
import math

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from hrap.advanced.tank import EquilibriumTank
from hrap.engine.sim import run
from hrap.engine.tank import tank
from hrap.io.config import bundled_motor, resolve


@pytest.mark.parametrize("vent", ["None", "External", "Internal"])
def test_liquid_step_conserves_mass_and_outlet_energy(vent):
    cfg = dict(bundled_motor("example_98mm"), solve_tank_cooling=True, vnt_state=vent, vnt_Cd=0.6)
    s, x = resolve(cfg)
    mass, energy, temp = x.m_o, x.U_tnk, x.T_tnk
    h_liq = CP.PropsSI("H", "T", temp, "Q", 0, "NitrousOxide")
    h_vap = CP.PropsSI("H", "T", temp, "Q", 1, "NitrousOxide")
    main = s.inj_CdA * s.inj_N * math.sqrt(2 * x.ox_props.rho_l * (x.P_tnk - x.P_cmbr))
    tank(s, None, x, s.dt)
    # Internal vent feeds the chamber; external vent leaves separately.
    vapor = x.mdot_o - main if vent == "Internal" else x.mdot_v
    assert mass - x.m_o == pytest.approx((x.mdot_o + x.mdot_v) * s.dt, abs=1e-12)
    assert energy - x.U_tnk == pytest.approx((main * h_liq + vapor * h_vap) * s.dt, abs=1e-7)
    assert x.T_tnk < temp


def test_vapor_flux_chokes_and_matches_dilute_gas_limit():
    model = EquilibriumTank("Nitrogen")
    gas = CP.AbstractState("HEOS", "Nitrogen")
    gas.update(CP.PT_INPUTS, 2e5, 300.0)
    h, entropy = gas.hmass(), gas.smass()
    flux = model.vapor_flux(2e5, h, entropy, 1e5)
    gamma = gas.cpmass() / gas.cvmass()
    R = gas.gas_constant() / gas.molar_mass()
    ideal = 2e5 / math.sqrt(R * 300) * math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1)/(2*(gamma - 1)))
    assert flux == pytest.approx(ideal, rel=0.003)
    assert model.vapor_flux(2e5, h, entropy, 1e3) == pytest.approx(flux, rel=1e-7)
    assert model.vapor_flux(2e5, h, entropy, 2e5) == 0
    assert model.vapor_flux(2e5, h, entropy, 3e5) == 0
    assert model.vapor_flux(2e5, h, entropy, 1.9e5) < flux


def test_valve_closure_preserves_tank_and_partial_step_balance():
    s, x = resolve(dict(bundled_motor("example_98mm"), solve_tank_cooling=True, t_burn=0.0005))
    mass = x.m_o
    tank(s, None, x, s.dt)
    assert mass - x.m_o == pytest.approx(x.mdot_o * s.dt, abs=1e-12)
    frozen = x.m_o, x.U_tnk, x.T_tnk, x.P_tnk
    tank(s, None, x, 2 * s.dt)
    assert (x.m_o, x.U_tnk, x.T_tnk, x.P_tnk) == frozen
    assert x.mdot_o == x.mdot_v == 0


def test_tank_only_runout_converges_without_reopening_liquid_outlet():
    results = []
    for dt in (0.002, 0.001):
        s, x = resolve(dict(bundled_motor("example_98mm"), solve_tank_cooling=True, dt=dt, vnt_state="None"))
        x.P_cmbr = 1e6  # fixed receiving pressure isolates the tank from chamber integration
        mass = x.m_o
        drained, runout = 0.0, None
        for i in range(round(10 / dt)):
            tank(s, None, x, (i + 1) * dt)
            drained += (x.mdot_o + x.mdot_v) * dt
            if x.equilibrium_tank.gas_discharge and runout is None:
                runout = (i + 1) * dt
            if runout is not None:
                assert x.equilibrium_tank.gas_discharge
        assert mass - x.m_o == pytest.approx(drained, abs=1e-10)
        assert runout is not None
        assert x.mLiq_new > 0  # equilibrium condensation is retained during vapor discharge
        results.append((runout, x.P_tnk, x.m_o))
    assert np.array(results[0]) == pytest.approx(results[1], rel=0.003)


def test_equilibrium_motor_exports_tank_history(tmp_path):
    from hrap.io.export import export_csv

    s, x = resolve(dict(bundled_motor("example_98mm"), solve_tank_cooling=True, t_max=12.0))
    _, o = run(s, x)
    assert 0 < o.extra["liquid_runout_time"] < o.t[-1]
    assert o.U_tnk.shape == o.m_liq.shape == o.mdot_v.shape == o.t.shape
    assert np.max(np.abs(np.diff(o.m_o) + (o.mdot_o[1:] + o.mdot_v[1:]) * s.dt)) < 1e-12
    export_csv(tmp_path / "tank.csv", o, s)
    assert {"U_tnk", "m_liq", "mdot_v", "T_tnk"} <= set((tmp_path / "tank.csv").read_text().splitlines()[0].split(","))
