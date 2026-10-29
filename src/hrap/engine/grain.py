"""Grain regression. Port of MATLAB util/shift_OF.m and util/const_OF.m."""
from __future__ import annotations

import math

import numpy as np

from hrap.engine.types import Settings, State


def shift_of(s: Settings, x: State) -> State:
    dt = s.dt
    A = 0.25 * math.pi * x.grn_ID ** 2
    G = x.mdot_o / A if A != 0.0 else 0.0
    a, n, m = float(s.prop_Reg[0]), float(s.prop_Reg[1]), float(s.prop_Reg[2])
    # MATLAB: 0.001 * a * G^n * L^m   (rdot in m/s; a in mm/s units)
    if G == 0.0 and n > 0.0:
        x.rdot = 0.0
    else:
        x.rdot = 0.001 * a * (G ** n) * (s.grn_L ** m)
    x.mdot_f = s.prop_Rho * x.rdot * math.pi * x.grn_ID * s.grn_L
    if x.mdot_f == 0.0:
        x.OF = 0.0
    else:
        x.OF = x.mdot_o / x.mdot_f
    x.grn_ID_old = x.grn_ID
    x.grn_ID = x.grn_ID + 2.0 * x.rdot * dt
    x.m_f = x.m_f - x.mdot_f * dt
    return x


def const_of(s: Settings, x: State) -> State:
    dt = s.dt
    x.mdot_f = x.mdot_o / s.const_OF if s.const_OF != 0.0 else 0.0
    denom = s.prop_Rho * math.pi * x.grn_ID * s.grn_L
    x.rdot = x.mdot_f / denom if denom != 0.0 else 0.0
    x.grn_ID_old = x.grn_ID
    x.grn_ID = x.grn_ID + 2.0 * x.rdot * dt
    x.m_f = x.m_f - x.mdot_f * dt
    return x


def regress(s: Settings, x: State) -> State:
    if s.grain_geometry is not None:
        return regress_stl(s, x)
    if s.regression_model == "Shifting OF":
        return shift_of(s, x)
    return const_of(s, x)


def regress_stl(s: Settings, x: State) -> State:
    g = s.grain_geometry
    area = float(np.interp(x.grn_web, g.web, g.port_area))
    a, n, m = s.prop_Reg
    flux = x.mdot_o / area
    x.rdot = 0.001 * a * flux ** n * g.length ** m
    old_mass = x.m_f
    x.grn_web = min(g.web[-1], x.grn_web + x.rdot * s.dt)
    x.m_f = s.prop_Rho * float(np.interp(x.grn_web, g.web, g.fuel_volume))
    # Integrate dV/dd over the step so fuel loss and chamber growth agree exactly.
    x.mdot_f = (old_mass - x.m_f) / s.dt
    x.OF = x.mdot_o / x.mdot_f if x.mdot_f else 0.0
    x.grn_ID_old = x.grn_ID
    x.grn_ID = 2 * math.sqrt(float(np.interp(x.grn_web, g.web, g.port_area)) / math.pi)
    return x
