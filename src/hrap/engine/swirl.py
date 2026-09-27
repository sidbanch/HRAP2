"""Discharge coefficient of a tangential-entry swirl injector from its geometry.

Abramovich's maximum-flow principle for an ideal (frictionless) liquid swirler. The swirl leaves
an air core in the exit orifice, so only a ring of liquid flows. The geometry parameter

    A = R_in · r_exit / (n · r_port²)

(R_in: distance from the swirler axis to each inlet port's axis, n: inlet port count) sets the
share φ of the exit area filled with liquid through A = (1 − φ)·√2 / φ^1.5, and then

    Cd = φ · √(φ / (2 − φ))

The ideal theory loses nothing entering the ports. Bazarov's inlet loss ξ (a share of the
port jets' velocity pressure) adds that, and matters most when the ports are small next to the exit:

    1 / Cd² = 1 / Cd_ideal² + ξ · (exit area ÷ total port area)²

ξ = 0 is the ideal theory; a sharp drilled hole is about 1.4. It's a liquid theory; boiling nitrous
flows somewhat differently, so fit ξ to a cold flow before trusting the Cd.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq


def swirl_A(D_exit: float, ports: int, D_port: float, R_in: float) -> float:
    return R_in * (0.5 * D_exit) / (ports * (0.5 * D_port) ** 2)


def swirl_fill(A: float) -> float:
    """Share of the exit area filled with liquid (1 − air-core area share)."""
    return brentq(lambda phi: (1.0 - phi) * math.sqrt(2.0) / phi ** 1.5 - A, 1e-6, 1.0)


def swirl_cd(D_exit: float, ports: int, D_port: float, R_in: float, xi: float = 0.0) -> float:
    phi = swirl_fill(swirl_A(D_exit, ports, D_port, R_in))
    ideal = phi * math.sqrt(phi / (2.0 - phi))
    return 1.0 / math.sqrt(1.0 / ideal ** 2 + xi * (D_exit ** 2 / (ports * D_port ** 2)) ** 2)


def swirl_sensitivity(D_exit: float, ports: int, D_port: float, R_in: float, xi: float = 0.0) -> tuple[float, float]:
    """The % flow change per % more total port area, and per % more exit area. Whichever is bigger limits the flow."""
    def log_cda(De: float, Dp: float) -> float:
        return math.log(swirl_cd(De, ports, Dp, R_in, xi) * De ** 2)

    step = 1.01  # a 1% larger area
    base = log_cda(D_exit, D_port)
    return ((log_cda(D_exit, D_port * math.sqrt(step)) - base) / math.log(step),
            (log_cda(D_exit * math.sqrt(step), D_port) - base) / math.log(step))


def swirl_port_D(D_exit: float, ports: int, R_in: float, cd: float, xi: float = 0.0) -> float:
    """Inlet port diameter that gives a Cd of cd. The ports can be at most twice their offset across."""
    if swirl_cd(D_exit, ports, 2.0 * R_in, R_in, xi) < cd:
        raise ValueError("No port size reaches this flow. Widen the exit, add ports or move them toward the axis.")
    return brentq(lambda d: swirl_cd(D_exit, ports, d, R_in, xi) - cd, 1e-4 * R_in, 2.0 * R_in)


def swirl_exit_D(cda: float, ports: int, D_port: float, R_in: float, xi: float = 0.0) -> float:
    """The exit diameter at which one swirler's Cd × exit area is cda, e.g. what a restricting fitting flows like.
    The exit can be at most as wide as the swirl chamber."""
    def gap(D: float) -> float:
        return swirl_cd(D, ports, D_port, R_in, xi) * 0.25 * math.pi * D ** 2 - cda

    widest = 2.0 * R_in + D_port
    if gap(widest) < 0.0:
        raise ValueError("Even an exit as wide as the swirl chamber can't pass that CdA through these holes. "
                         "Check the hole count, size and offset.")
    return brentq(gap, 1e-4 * widest, widest)


def swirl_xi(cda: float, D_exit: float, ports: int, D_port: float, R_in: float) -> float:
    """The inlet loss at which one swirler's Cd × exit area is cda."""
    cd, ideal = cda / (0.25 * math.pi * D_exit ** 2), swirl_cd(D_exit, ports, D_port, R_in)
    if cd >= ideal:
        raise ValueError(f"That's at least what the ideal theory gives (Cd {ideal:.3f} on this exit), so there's no "
                         "inlet loss to fit. Check the exit size.")
    return (1.0 / cd ** 2 - 1.0 / ideal ** 2) / (D_exit ** 2 / (ports * D_port ** 2)) ** 2


# Number drill diameters in inches, #80 to #1.
NUMBER_DRILLS = dict(zip(range(80, 0, -1), (
    0.0135, 0.0145, 0.016, 0.018, 0.020, 0.021, 0.0225, 0.024, 0.025, 0.026, 0.028, 0.0292, 0.031, 0.032, 0.033, 0.035,
    0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.0465, 0.052, 0.055, 0.0595, 0.0635, 0.067, 0.070, 0.073,
    0.076, 0.0785, 0.081, 0.082, 0.086, 0.089, 0.0935, 0.096, 0.098, 0.0995, 0.1015, 0.104, 0.1065, 0.110, 0.111, 0.113,
    0.116, 0.120, 0.1285, 0.136, 0.1405, 0.144, 0.147, 0.1495, 0.152, 0.154, 0.157, 0.159, 0.161, 0.166, 0.1695, 0.173,
    0.177, 0.180, 0.182, 0.185, 0.189, 0.191, 0.1935, 0.196, 0.199, 0.201, 0.204, 0.2055, 0.209, 0.213, 0.221, 0.228)))


def nearest_drill(D: float) -> tuple[int, float]:
    """The number drill closest to D (m), as (number, diameter in m)."""
    number = min(NUMBER_DRILLS, key=lambda k: abs(NUMBER_DRILLS[k] * 0.0254 - D))
    return number, NUMBER_DRILLS[number] * 0.0254
