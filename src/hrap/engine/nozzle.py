"""Isentropic CD nozzle. Port of MATLAB util/nozzle.m."""
from __future__ import annotations

import math

from scipy.optimize import brentq

from hrap.engine.types import Settings, State


def exit_mach(k: float, ER: float, guess: float = 0.0) -> float:
    """Supersonic exit Mach number for area ratio ``ER`` (MATLAB fzero from guess 3).

    Newton from ``guess`` (the previous step's value) converges in a few iterations; the
    bracketed solve covers the first step and any guess Newton can't use.
    """
    g = (k - 1.0) / 2.0
    e = (k + 1.0) / (2.0 * (k - 1.0))
    c = ((k + 1.0) / 2.0) ** -e

    def A_ratio(M: float) -> float:
        return c * (1.0 + g * M ** 2) ** e / M - ER

    M = guess
    for _ in range(20 if M > 1.0 else 0):
        A = c * (1.0 + g * M ** 2) ** e / M
        step = (A - ER) * M * (1.0 + g * M ** 2) / (A * (M ** 2 - 1.0))
        M -= step
        if M <= 1.0:
            break
        if abs(step) <= 1e-15 * M:
            return M
    return float(brentq(A_ratio, 1.0000001, 80.0, xtol=2.2e-16, maxiter=200))


def nozzle(s: Settings, x: State) -> State:
    if x.P_cmbr > s.Pa:
        k = x.k
        M = x.Me = exit_mach(k, s.noz_ER, x.Me)
        Pe = x.P_cmbr * (1.0 + 0.5 * (k - 1.0) * M ** 2) ** (-k / (k - 1.0))
        Ath = 0.25 * math.pi * s.noz_thrt ** 2
        Aex = Ath * s.noz_ER
        Cf = math.sqrt(
            ((2.0 * k ** 2) / (k - 1.0))
            * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0))
            * (1.0 - (Pe / x.P_cmbr) ** ((k - 1.0) / k))
        ) + ((Pe - s.Pa) * Aex) / (x.P_cmbr * Ath)
        x.F_thr = s.noz_eff * Cf * Ath * x.P_cmbr * s.noz_Cd
        if x.F_thr < 0.0:
            x.F_thr = 0.0
    else:
        x.F_thr = 0.0
    return x
