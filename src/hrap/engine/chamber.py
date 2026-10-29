"""Chamber pressure. Port of MATLAB util/chamber.m."""
from __future__ import annotations

import math

from hrap.engine.types import Settings, State


def chamber(s: Settings, x: State) -> State:
    dt = s.dt
    if s.grain_geometry is not None:
        V = (s.cmbr_V - x.m_f / s.prop_Rho if s.cmbr_V else
             0.25 * math.pi * x.grn_ID ** 2 * s.grn_L)
    elif s.cmbr_V == 0:
        V = 0.25 * math.pi * x.grn_ID ** 2 * s.grn_L
    else:
        V = s.cmbr_V - 0.25 * math.pi * (s.grn_OD ** 2 - x.grn_ID ** 2) * s.grn_L

    dV = 0.25 * math.pi * (x.grn_ID ** 2 - x.grn_ID_old ** 2) * s.grn_L / dt

    Ath = 0.25 * math.pi * s.noz_thrt ** 2
    x.mdot_n = x.P_cmbr * s.noz_Cd * Ath / x.cstar if x.cstar != 0.0 else 0.0

    dm_g = x.mdot_f + x.mdot_o - x.mdot_n
    if x.mdot_o == 0:
        x.dm_g = -x.mdot_n

    x.m_g = x.m_g + dm_g * dt

    dP = x.P_cmbr * (dm_g / x.m_g - dV / V) if x.m_g != 0.0 and V != 0.0 else 0.0
    x.P_cmbr = x.P_cmbr + dP * dt

    if x.P_cmbr <= s.Pa:
        x.P_cmbr = s.Pa
        x.mdot_n = 0.0
    return x
