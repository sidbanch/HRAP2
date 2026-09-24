"""Two-phase injector flow for saturated liquid N2O (advanced mode, not MATLAB-identical).

SPI (HRAP's model) treats the injector flow as pure liquid. HEM assumes the liquid boils instantly
and completely as its pressure drops through the orifice, so it predicts less flow and chokes. Dyer
blends the two: mdot = κ/(1+κ)·SPI + 1/(1+κ)·HEM. For liquid at its own vapor pressure, as HRAP's
tank always is, Dyer's formula gives κ = 1 (an even blend); some test fits use a larger κ.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np


@lru_cache(maxsize=None)
def hem_flux_table(fluid: str = "NitrousOxide") -> Callable[[float, float], float]:
    """Return flux(T_tank, P_down / P_tank) -> HEM mass flux in kg/(m²·s), with choking.

    Built once per process from CoolProp: saturated liquid at T_tank expands isentropically to the
    downstream pressure. Below the choke pressure the flux stays at its maximum.
    """
    try:
        import CoolProp.CoolProp as CP
    except ImportError as exc:
        raise ImportError("CoolProp is required for the HEM and Dyer injector models. pip install hrap[advanced]") from exc

    state = CP.AbstractState("HEOS", fluid)
    T_grid = np.linspace(CP.PropsSI("Ttriple", fluid) + 1.0, CP.PropsSI("Tcrit", fluid) - 0.5, 250)
    r_grid = np.linspace(0.01, 1.0, 200)
    flux = np.zeros((T_grid.size, r_grid.size))
    for i, T in enumerate(T_grid):
        state.update(CP.QT_INPUTS, 0.0, T)
        P1, h1, s1 = state.p(), state.hmass(), state.smass()
        for j, r in enumerate(r_grid[:-1]):
            try:
                state.update(CP.PSmass_INPUTS, r * P1, s1)
            except ValueError:  # expansion below the triple point; flow is already choked there
                continue
            flux[i, j] = state.rhomass() * np.sqrt(2.0 * max(h1 - state.hmass(), 0.0))
        flux[i] = np.maximum.accumulate(flux[i][::-1])[::-1]

    def hem_flux(T: float, r: float) -> float:
        T = min(max(T, T_grid[0]), T_grid[-1])
        k = min(int(np.searchsorted(T_grid, T)), T_grid.size - 1)
        k = max(k, 1)
        w = (T - T_grid[k - 1]) / (T_grid[k] - T_grid[k - 1])
        lo = np.interp(r, r_grid, flux[k - 1])
        hi = np.interp(r, r_grid, flux[k])
        return float(lo + w * (hi - lo))

    return hem_flux
