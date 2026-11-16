"""Adiabatic, rigid equilibrium tank: dm/dt = -mdot; dU/dt = -sum(mdot*h_out).

Pure fluid, no pressurant or tank-wall heat transfer. A bottom outlet supplies liquid
until runout, then vapor; any later condensate remains in the tank. A vent supplies vapor. The same density/internal-energy flash continues
into vapor blowdown. Injector liquid flashing remains a separate SPI/HEM/Dyer choice.
Properties: https://coolprop.org/coolprop/LowLevelAPI.html
"""
from __future__ import annotations

import math

import CoolProp.CoolProp as CP
from scipy.optimize import minimize_scalar

from hrap.engine.types import OxProps, Settings, State


class EquilibriumTank:
    def __init__(self, fluid: str):
        self.bulk = CP.AbstractState("HEOS", fluid)
        self.liquid = CP.AbstractState("HEOS", fluid)
        self.vapor = CP.AbstractState("HEOS", fluid)
        self.throat = CP.AbstractState("HEOS", fluid)
        self.gas_discharge = False
        self.runout_time = None
        self.t_min = CP.PropsSI("Ttriple", fluid) + 1e-5
        self.t_max = CP.PropsSI("Tcrit", fluid) - 1e-5
        self.p_min = CP.PropsSI("ptriple", fluid) * (1 + 1e-7)

    def sat_props(self, T: float) -> OxProps:
        self.liquid.update(CP.QT_INPUTS, 0.0, T)
        self.vapor.update(CP.QT_INPUTS, 1.0, T)
        return OxProps(self.liquid.p(), self.liquid.rhomass(), self.vapor.rhomass(),
                       self.vapor.hmass() - self.liquid.hmass(), self.liquid.cpmass(), self.vapor.compressibility_factor())

    def initialize(self, s: Settings, x: State) -> None:
        self.sat_props(x.T_tnk)
        if not 0 < x.m_o / s.tnk_V < self.liquid.rhomass():
            raise ValueError("Equilibrium tank requires a positive fill with room for vapor.")
        self.bulk.update(CP.DmassT_INPUTS, x.m_o / s.tnk_V, x.T_tnk)
        x.U_tnk = x.m_o * self.bulk.umass()
        self.refresh(s, x)
        if self.gas_discharge:
            self.runout_time = 0.0

    def refresh(self, s: Settings, x: State) -> None:
        self.bulk.update(CP.DmassUmass_INPUTS, x.m_o / s.tnk_V, x.U_tnk / x.m_o)
        x.T_tnk, x.P_tnk = self.bulk.T(), self.bulk.p()
        x.ox_props = self.sat_props(x.T_tnk)
        q = self.bulk.Q()
        if q < 0 and self.bulk.phase() != CP.iphase_gas:
            raise ValueError("Equilibrium tank requires liquid/vapor or vapor conditions.")
        # The density/energy flash leaves sub-ppm traces at the saturation boundary.
        # Treat those as runout rather than repeatedly draining numerical residue.
        x.mLiq_new = x.m_o * (1.0 - q) if 0 <= q < 1 - 1e-6 else 0.0
        if x.mLiq_new == 0:
            self.gas_discharge = True

    def vapor_flux(self, P: float, h: float, entropy: float, back_pressure: float) -> float:
        """Real-fluid isentropic mass flux, maximized at choking, zero for no forward pressure drop."""
        if back_pressure >= P:
            return 0.0
        if P <= self.p_min:
            raise ValueError("Equilibrium tank has reached the fluid's triple-point pressure.")

        def flux(ratio: float) -> float:
            self.throat.update(CP.PSmass_INPUTS, ratio * P, entropy)
            return self.throat.rhomass() * math.sqrt(2 * max(h - self.throat.hmass(), 0.0))

        lo = max(back_pressure, self.p_min) / P
        result = minimize_scalar(lambda r: -flux(r), bounds=(lo, 1.0), method="bounded")
        if back_pressure < self.p_min and result.x <= lo + 1e-4:
            raise ValueError("Injector expansion reaches the triple point before choking; solid formation is not modeled.")
        return max(flux(lo), -float(result.fun))

    def step(self, s: Settings, x: State, t: float) -> State:
        from hrap.engine.tank import hole_CdA, liquid_flow

        old_pressure = x.P_tnk
        x.mdot_o = x.mdot_v = 0.0
        duration = s.dt if s.tburn == 0 else min(s.dt, max(s.tburn - (t - s.dt), 0.0))
        if duration == 0:
            x.dP = 0.0
            return x
        if s.vnt_S not in (0, 1, 2):
            raise ValueError("Error: Vent State Undefined")
        elapsed = 0.0
        chamber_mass = vent_mass = 0.0
        x.mLiq_old = x.mLiq_new
        while elapsed < duration:
            # After bulk runout, assume any condensate stays in the tank.
            liquid = not self.gas_discharge
            two_phase = 0 <= self.bulk.Q() <= 1
            h_v = self.vapor.hmass() if two_phase else self.bulk.hmass()
            entropy = self.vapor.smass() if two_phase else self.bulk.smass()
            sample_time = t - s.dt + elapsed
            CdA = hole_CdA(s, sample_time)[0] * s.inj_N
            mdot_main = (liquid_flow(s, x.T_tnk, x.ox_props.rho_l, x.P_tnk, x.P_cmbr, sample_time) if liquid else
                         CdA * self.vapor_flux(x.P_tnk, h_v, entropy, x.P_cmbr))
            vent_back = x.P_cmbr if s.vnt_S == 2 else s.Pa
            mdot_vent = (s.vnt_CdA * self.vapor_flux(x.P_tnk, h_v, entropy, vent_back)
                         if s.vnt_S and s.vnt_CdA > 0 else 0.0)
            total = mdot_main + mdot_vent
            if total == 0:
                break
            # Subdivide rather than clipping the physical flow rate. Resolve runout
            # within this timestep and use vapor flow for the remaining fraction.
            step = min(duration - elapsed, 0.02 * x.m_o / total)
            if liquid and mdot_main > 0:
                step = min(step, 0.5 * x.mLiq_new / mdot_main)
            if two_phase and mdot_vent > 0:
                step = min(step, 0.5 * (x.m_o - x.mLiq_new) / mdot_vent)
            if step <= 0 or elapsed + step == elapsed:
                raise ValueError("Tank timestep is too small to resolve the remaining phase mass.")
            h_main = self.liquid.hmass() if liquid else h_v
            x.m_o -= total * step
            x.U_tnk -= (mdot_main * h_main + mdot_vent * h_v) * step
            chamber_mass += (mdot_main + (mdot_vent if s.vnt_S == 2 else 0)) * step
            vent_mass += (mdot_vent if s.vnt_S == 1 else 0) * step
            elapsed += step
            self.refresh(s, x)
            if self.gas_discharge and self.runout_time is None:
                self.runout_time = t - s.dt + elapsed
        # Report rates averaged over the outer timestep, including partial valve closure.
        x.mdot_o, x.mdot_v = chamber_mass / s.dt, vent_mass / s.dt
        x.dP = x.P_tnk - old_pressure
        return x
