"""Optional CoolProp saturation properties (advanced mode, not MATLAB-identical)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from hrap.engine.types import OxProps


@lru_cache(maxsize=None)
def coolprop_sat(fluid: str, T_min: float | None = None, T_max: float | None = None, n: int = 80):
    import CoolProp.CoolProp as CP

    if T_min is None:
        T_min = CP.PropsSI("Tmin", fluid) + 1.0
    if T_max is None:
        T_max = min(CP.PropsSI("Tcrit", fluid) - 0.5, T_min + 200.0)
    T_grid = np.linspace(T_min, T_max, n)
    Pv = np.array([CP.PropsSI("P", "T", T, "Q", 0, fluid) for T in T_grid])
    rho_l = np.array([CP.PropsSI("D", "T", T, "Q", 0, fluid) for T in T_grid])
    rho_v = np.array([CP.PropsSI("D", "T", T, "Q", 1, fluid) for T in T_grid])
    Hv = np.array(
        [CP.PropsSI("H", "T", T, "Q", 1, fluid) - CP.PropsSI("H", "T", T, "Q", 0, fluid) for T in T_grid]
    )
    Cp = np.array([CP.PropsSI("CPMASS", "T", T, "Q", 0, fluid) for T in T_grid])
    Z = np.array([CP.PropsSI("Z", "T", T, "Q", 1, fluid) for T in T_grid])

    rows = [tuple(map(float, r)) for r in zip(Pv, rho_l, rho_v, Hv, Cp, Z)]
    T0, dT = float(T_grid[0]), float(T_grid[1] - T_grid[0])

    def get_sat_props(T: float) -> OxProps:
        f = min(max((float(T) - T0) / dT, 0.0), n - 1.0)
        k = min(int(f), n - 2)
        w = f - k
        return OxProps(*(a + w * (b - a) for a, b in zip(rows[k], rows[k + 1])))

    return get_sat_props
