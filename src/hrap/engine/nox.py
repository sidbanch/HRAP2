"""Saturated nitrous oxide curve fits. Port of MATLAB util/NOX.m."""
from __future__ import annotations

from hrap.engine.types import OxProps

# Critical properties (MATLAB NOX.m)
PC = 7251000.0
TC = 309.57
RHOC = 452.0
R_N2O = 188.91


def nox(T: float) -> OxProps:
    """Thermophysical properties of saturated N2O as a function of temperature (K).

    Applicable range stated in MATLAB: about -90 to +36 C.
    """
    T = float(T)
    Tr = T / TC
    one_m = 1.0 - Tr

    a1, a2, a3, a4 = -6.71893, 1.35966, -1.3779, -4.051
    Pv = PC * _exp(
        (1.0 / Tr)
        * (
            a1 * one_m
            + a2 * _pow(one_m, 1.5)
            + a3 * _pow(one_m, 2.5)
            + a4 * _pow(one_m, 5.0)
        )
    )

    b1, b2, b3, b4 = 1.72328, -0.83950, 0.51060, -0.10412
    rho_l = RHOC * _exp(
        b1 * _pow(one_m, 1.0 / 3.0)
        + b2 * _pow(one_m, 2.0 / 3.0)
        + b3 * one_m
        + b4 * _pow(one_m, 4.0 / 3.0)
    )

    c1, c2, c3, c4, c5 = -1.00900, -6.28792, 7.50332, -7.90463, 0.629427
    tau = (TC / T) - 1.0
    rho_v = RHOC * _exp(
        c1 * _pow(tau, 1.0 / 3.0)
        + c2 * _pow(tau, 2.0 / 3.0)
        + c3 * tau
        + c4 * _pow(tau, 4.0 / 3.0)
        + c5 * _pow(tau, 5.0 / 3.0)
    )

    d1, d2, d3, d4, d5 = -200.0, 116.043, -917.225, 794.779, -589.587
    e1, e2, e3, e4, e5 = -200.0, 440.055, -459.701, 434.081, -485.338
    Hv = (
        (e1 - d1)
        + (e2 - d2) * _pow(one_m, 1.0 / 3.0)
        + (e3 - d3) * _pow(one_m, 2.0 / 3.0)
        + (e4 - d4) * one_m
        + (e5 - d5) * _pow(one_m, 4.0 / 3.0)
    )

    f1, f2, f3, f4, f5 = 2.49973, 0.023454, -3.80136, 13.0945, -14.5180
    Cp = f1 * (
        1.0
        + f2 * _pow(one_m, -1.0)
        + f3 * one_m
        + f4 * one_m ** 2.0
        + f5 * one_m ** 3.0
    )

    Z = Pv / (rho_v * R_N2O * T)
    return OxProps(Pv=Pv, rho_l=rho_l, rho_v=rho_v, Hv=Hv, Cp=Cp, Z=Z)


def vapor_pressure(T: float) -> float:
    """Wagner vapor-pressure fit from MATLAB NOX.m (used by GUI fzero and tank)."""
    T = float(T)
    Tr = T / TC
    one_m = 1.0 - Tr
    a1, a2, a3, a4 = -6.71893, 1.35966, -1.3779, -4.051
    return PC * _exp(
        (1.0 / Tr)
        * (
            a1 * one_m
            + a2 * _pow(one_m, 1.5)
            + a3 * _pow(one_m, 2.5)
            + a4 * _pow(one_m, 5.0)
        )
    )


def saturation_temperature(P: float) -> float | None:
    """The temperature whose vapor pressure is P, or None outside the fit."""
    import math

    from scipy.optimize import brentq

    if not math.isfinite(P) or P <= 1.0 or P >= 7.2e6:
        return None
    try:
        return float(brentq(lambda T: vapor_pressure(T) - P, 183.15, TC - 0.05, xtol=1e-9))
    except ValueError:
        return None


def _pow(base: float, exp: float) -> float:
    """Real power. Negative bases with fractional exponents are complex in MATLAB."""
    if base < 0.0:
        raise ValueError("complex residual")
    if base == 0.0 and exp < 0.0:
        raise ZeroDivisionError("0 ** negative")
    return base ** exp


def _exp(x: float) -> float:
    import math

    if isinstance(x, complex):
        raise TypeError("complex residual")
    return math.exp(float(x))
