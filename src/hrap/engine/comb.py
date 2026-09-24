"""Combustion property interpolation. Port of MATLAB util/comb.m."""
from __future__ import annotations

from hrap.engine.interp import interp2x
from hrap.engine.types import Settings, State

R_UNIV = 8314.5  # J/(kmol*K) as used in MATLAB comb.m


def comb(s: Settings, x: State, t: float) -> State:
    if t <= s.tburn or s.tburn == 0:
        x.k, x.M, x.T = interp2x(s.prop_OF, s.prop_Pc, (s.prop_k, s.prop_M, s.prop_T), x.OF, x.P_cmbr)
        x.R = R_UNIV / x.M
        x.rho = x.P_cmbr / (x.R * x.T)
        x.cstar = s.cstar_eff * (
            (x.R * x.T) / (x.k * (2.0 / (x.k + 1.0)) ** ((x.k + 1.0) / (x.k - 1.0)))
        ) ** 0.5
    return x
