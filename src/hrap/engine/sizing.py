"""Preliminary sizing: injector, nozzle and grain for a target chamber pressure, or the chamber pressure
the motor's own nozzle gives.

Everything is evaluated at the start of the burn with the tank at its starting temperature, using
the same injector, combustion-table and nozzle equations as the simulation. The simulation then
shows how the motor drifts from these numbers as the tank cools and the port opens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy.optimize import brentq

from hrap.engine.comb import comb
from hrap.engine.nox import nox
from hrap.engine.nozzle import nozzle
from hrap.engine.tank import liquid_flow
from hrap.io.config import resolve

G0 = 9.80665


@dataclass(frozen=True)
class SizingTargets:
    P_cmbr: float | None  # Pa absolute; None keeps the motor's throat and expansion ratio and solves the pressure
    burn_time: float | None  # s, time to use the liquid at the starting oxidizer flow; ignored when holes is set.
                             # With neither, the oxidizer flow is solved from OF and grain_L.
    OF: float         # ignored when grain_L is set, unless it sets the oxidizer flow; a fixed-O/F motor uses its own
    port_D: float     # m, starting port diameter
    holes: int | None = None  # a fixed injector hole count; the burn time then follows from it
    grain_L: float | None = None  # m, a fixed grain length; the starting O/F then follows from it


@dataclass(frozen=True)
class Sizing:
    P_cmbr: float         # Pa, chamber pressure (the target, or what the motor's nozzle gives)
    P_tnk: float          # Pa, tank pressure at the starting temperature
    inj_dP: float         # Pa
    ox_liquid: float      # kg of liquid in the tank at the start
    mdot_o: float         # kg/s
    mdot_f: float         # kg/s
    OF: float             # at the start (the target when no grain length was given)
    flow_per_hole: float  # kg/s through one injector hole of the motor's diameter and Cd
    inj_CdA: float        # m², total injector Cd × area for mdot_o, with the motor's injector model
    holes: float          # exact number of holes for mdot_o (the fixed count when one was given)
    burn_time: float      # s, liquid burn time at the starting flow (the target when no hole count was given)
    OF_exp: float         # with the grain length fixed, O/F grows as oxidizer flow^OF_exp (1 − regression exponent)
    k: float              # ratio of specific heats from the combustion table
    cstar: float          # m/s, including C* efficiency
    throat_D: float       # m
    ER: float             # expansion ratio for exit pressure = ambient, or the motor's
    exit_D: float         # m
    thrust: float         # N
    isp: float            # s
    ox_flux: float        # kg/(m²·s) through the starting port
    grain_L: float        # m, grain length for the target O/F (the fixed length when one was given; nan without a regression law)
    port_D_end: float     # m, port diameter when the liquid runs out
    OF_end: float         # O/F when the liquid runs out
    fuel_burned: float    # kg


def size_motor(cfg: dict[str, Any], t: SizingTargets) -> Sizing:
    s, x = resolve(cfg)
    op = s.get_sat_props(x.T_tnk) if s.get_sat_props is not None else nox(x.T_tnk)
    if t.P_cmbr is not None:
        if t.P_cmbr >= op.Pv:
            raise ValueError("The chamber pressure target has to be below the tank pressure.")
        if t.P_cmbr <= s.Pa:
            raise ValueError("The chamber pressure target has to be above ambient pressure.")
        return _size(s, x, t, op, t.P_cmbr, None)
    # The motor's nozzle is set. A higher chamber pressure needs a smaller throat for the flow (and lets less
    # oxidizer in), so exactly one pressure between ambient and the tank matches the motor's throat.
    throat_D, ER = s.noz_thrt, s.noz_ER
    if throat_D <= 0.0:
        raise ValueError("The motor has no throat diameter.")

    def gap(P: float) -> float:
        return _size(s, x, t, op, P, ER).throat_D - throat_D

    low, high = s.Pa * (1.0 + 1e-6), op.Pv * (1.0 - 1e-6)
    if gap(low) < 0.0:
        raise ValueError("The throat is too big for this flow: the chamber would stay at ambient pressure.")
    if gap(high) > 0.0:
        raise ValueError("The throat is too small for this flow: the chamber would need more than the tank pressure.")
    return _size(s, x, t, op, brentq(gap, low, high, xtol=1.0), ER)


def _size(s, x, t: SizingTargets, op, P_cmbr: float, ER: float | None) -> Sizing:
    """Everything at one chamber pressure. Without an expansion ratio, it's sized for ambient exit pressure."""
    P_tnk = op.Pv
    flow_per_hole = liquid_flow(s, x.T_tnk, op.rho_l, P_tnk, P_cmbr) / s.inj_N
    a, n, m = (float(v) for v in s.prop_Reg[:3])
    rho = s.prop_Rho
    fixed_OF = s.regression_model == "Constant OF"  # the fuel flow is the oxidizer flow ÷ the motor's O/F
    if t.holes:
        mdot_o = t.holes * flow_per_hole
    elif t.burn_time:
        mdot_o = x.mLiq_new / t.burn_time
    else:
        if fixed_OF:
            raise ValueError("With a fixed O/F, the O/F doesn't depend on the oxidizer flow, so it can't size the injector.")
        if not t.grain_L or a <= 0.0 or n >= 1.0:
            raise ValueError("Sizing the oxidizer flow from O/F needs a grain length and the propellant's regression law.")
        # fuel flow = K · mdot_o^n, so O/F = mdot_o^(1−n) / K
        K = rho * 0.001 * a * (0.25 * math.pi * t.port_D ** 2) ** -n * t.grain_L ** m * math.pi * t.port_D * t.grain_L
        mdot_o = (t.OF * K) ** (1.0 / (1.0 - n))
    burn_time = t.burn_time if t.burn_time and not t.holes else x.mLiq_new / mdot_o
    ox_flux = mdot_o / (0.25 * math.pi * t.port_D ** 2)
    if fixed_OF:
        if not t.grain_L:
            raise ValueError("With a fixed O/F, the grain length doesn't change the fuel flow, so it can't be sized.")
        grain_L, OF = t.grain_L, s.const_OF
        mdot_f = mdot_o / OF
    elif t.grain_L:
        if a <= 0.0:
            raise ValueError("Sizing from a grain length needs the propellant's regression law (a > 0).")
        grain_L = t.grain_L
        mdot_f = rho * 0.001 * a * ox_flux ** n * grain_L ** m * math.pi * t.port_D * grain_L
        OF = mdot_o / mdot_f
    else:
        OF = t.OF
        mdot_f = mdot_o / OF
        # fuel flow = rho · (0.001·a·G^n·L^m) · π·D·L, solved for L
        grain_L = (mdot_f / (rho * 0.001 * a * ox_flux ** n * math.pi * t.port_D)) ** (1.0 / (1.0 + m)) if a > 0.0 else float("nan")
    mdot = mdot_o + mdot_f

    x.OF, x.P_cmbr = OF, P_cmbr
    x = comb(s, x, 0.0)
    k = x.k
    throat_A = mdot * x.cstar / (P_cmbr * s.noz_Cd)
    throat_D = math.sqrt(4.0 * throat_A / math.pi)

    if ER is None:
        Me = math.sqrt(2.0 / (k - 1.0) * ((P_cmbr / s.Pa) ** ((k - 1.0) / k) - 1.0))
        ER = ((2.0 / (k + 1.0)) * (1.0 + 0.5 * (k - 1.0) * Me ** 2)) ** ((k + 1.0) / (2.0 * (k - 1.0))) / Me
    s.noz_thrt, s.noz_ER = throat_D, ER
    thrust = nozzle(s, x).F_thr

    port_D_end = OF_end = fuel_burned = float("nan")
    if fixed_OF:  # the port opens by the fuel burned, as in the simulation
        fuel_burned, OF_end = mdot_f * burn_time, OF
        port_D_end = math.sqrt(t.port_D ** 2 + 4.0 * fuel_burned / (math.pi * rho * grain_L))
    elif a > 0.0:
        # dD/dt = 2·0.001·a·(4·mdot_o/(π·D²))^n·L^m integrates in closed form for constant mdot_o
        c = 2.0 * 0.001 * a * (4.0 * mdot_o / math.pi) ** n * grain_L ** m
        port_D_end = (t.port_D ** (2 * n + 1) + (2 * n + 1) * c * burn_time) ** (1.0 / (2 * n + 1))
        flux_end = mdot_o / (0.25 * math.pi * port_D_end ** 2)
        mdot_f_end = rho * 0.001 * a * flux_end ** n * grain_L ** m * math.pi * port_D_end * grain_L
        OF_end = mdot_o / mdot_f_end
        fuel_burned = rho * 0.25 * math.pi * (port_D_end ** 2 - t.port_D ** 2) * grain_L

    return Sizing(
        P_cmbr=P_cmbr,
        P_tnk=P_tnk,
        inj_dP=P_tnk - P_cmbr,
        ox_liquid=x.mLiq_new,
        mdot_o=mdot_o,
        mdot_f=mdot_f,
        OF=OF,
        flow_per_hole=flow_per_hole,
        inj_CdA=s.inj_CdA * mdot_o / flow_per_hole,  # the flow is proportional to CdA in every injector model
        holes=mdot_o / flow_per_hole,
        burn_time=burn_time,
        OF_exp=0.0 if fixed_OF else 1.0 - n,
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
