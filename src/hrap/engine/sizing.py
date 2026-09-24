"""Preliminary sizing: injector, nozzle and grain for a target chamber pressure and burn time.

Everything is evaluated at the start of the burn with the tank at its starting temperature, using
the same injector, combustion-table and nozzle equations as the simulation. The simulation then
shows how the motor drifts from these numbers as the tank cools and the port opens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hrap.engine.comb import comb
from hrap.engine.nox import nox
from hrap.engine.nozzle import nozzle
from hrap.engine.tank import liquid_flow
from hrap.io.config import resolve

G0 = 9.80665


@dataclass(frozen=True)
class SizingTargets:
    P_cmbr: float     # Pa absolute
    burn_time: float  # s, time to use the liquid at the starting oxidizer flow
    OF: float
    port_D: float     # m, starting port diameter


@dataclass(frozen=True)
class Sizing:
    P_tnk: float          # Pa, tank pressure at the starting temperature
    inj_dP: float         # Pa
    ox_liquid: float      # kg of liquid in the tank at the start
    mdot_o: float         # kg/s
    mdot_f: float         # kg/s
    flow_per_hole: float  # kg/s through one injector hole of the motor's diameter and Cd
    holes: float          # exact number of holes for mdot_o
    k: float              # ratio of specific heats from the combustion table
    cstar: float          # m/s, including C* efficiency
    throat_D: float       # m
    ER: float             # expansion ratio for exit pressure = ambient
    exit_D: float         # m
    thrust: float         # N
    isp: float            # s
    ox_flux: float        # kg/(m²·s) through the starting port
    grain_L: float        # m, grain length that gives the target O/F at the start (nan without a regression law)
    port_D_end: float     # m, port diameter when the liquid runs out
    OF_end: float         # O/F when the liquid runs out
    fuel_burned: float    # kg


def size_motor(cfg: dict[str, Any], t: SizingTargets) -> Sizing:
    s, x = resolve(cfg)
    op = s.get_sat_props(x.T_tnk) if s.get_sat_props is not None else nox(x.T_tnk)
    P_tnk = op.Pv
    if t.P_cmbr >= P_tnk:
        raise ValueError("The chamber pressure target has to be below the tank pressure.")
    if t.P_cmbr <= s.Pa:
        raise ValueError("The chamber pressure target has to be above ambient pressure.")

    mdot_o = x.mLiq_new / t.burn_time
    mdot_f = mdot_o / t.OF
    mdot = mdot_o + mdot_f
    flow_per_hole = liquid_flow(s, x.T_tnk, op.rho_l, P_tnk, t.P_cmbr) / s.inj_N

    x.OF, x.P_cmbr = t.OF, t.P_cmbr
    x = comb(s, x, 0.0)
    k = x.k
    throat_A = mdot * x.cstar / (t.P_cmbr * s.noz_Cd)
    throat_D = math.sqrt(4.0 * throat_A / math.pi)

    Me = math.sqrt(2.0 / (k - 1.0) * ((t.P_cmbr / s.Pa) ** ((k - 1.0) / k) - 1.0))
    ER = ((2.0 / (k + 1.0)) * (1.0 + 0.5 * (k - 1.0) * Me ** 2)) ** ((k + 1.0) / (2.0 * (k - 1.0))) / Me
    s.noz_thrt, s.noz_ER = throat_D, ER
    thrust = nozzle(s, x).F_thr

    a, n, m = (float(v) for v in s.prop_Reg[:3])
    rho = s.prop_Rho
    ox_flux = mdot_o / (0.25 * math.pi * t.port_D ** 2)
    grain_L = port_D_end = OF_end = fuel_burned = float("nan")
    if a > 0.0:
        # fuel flow = rho · (0.001·a·G^n·L^m) · π·D·L, solved for L
        grain_L = (mdot_f / (rho * 0.001 * a * ox_flux ** n * math.pi * t.port_D)) ** (1.0 / (1.0 + m))
        # dD/dt = 2·0.001·a·(4·mdot_o/(π·D²))^n·L^m integrates in closed form for constant mdot_o
        c = 2.0 * 0.001 * a * (4.0 * mdot_o / math.pi) ** n * grain_L ** m
        port_D_end = (t.port_D ** (2 * n + 1) + (2 * n + 1) * c * t.burn_time) ** (1.0 / (2 * n + 1))
        flux_end = mdot_o / (0.25 * math.pi * port_D_end ** 2)
        mdot_f_end = rho * 0.001 * a * flux_end ** n * grain_L ** m * math.pi * port_D_end * grain_L
        OF_end = mdot_o / mdot_f_end
        fuel_burned = rho * 0.25 * math.pi * (port_D_end ** 2 - t.port_D ** 2) * grain_L

    return Sizing(
        P_tnk=P_tnk,
        inj_dP=P_tnk - t.P_cmbr,
        ox_liquid=x.mLiq_new,
        mdot_o=mdot_o,
        mdot_f=mdot_f,
        flow_per_hole=flow_per_hole,
        holes=mdot_o / flow_per_hole,
        k=k,
        cstar=x.cstar,
        throat_D=throat_D,
        ER=ER,
        exit_D=throat_D * math.sqrt(ER),
        thrust=thrust,
        isp=thrust / (mdot * G0),
        ox_flux=ox_flux,
        grain_L=grain_L,
        port_D_end=port_D_end,
        OF_end=OF_end,
        fuel_burned=fuel_burned,
    )
