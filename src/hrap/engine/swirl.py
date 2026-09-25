"""Discharge coefficient of a tangential-entry swirl injector from its geometry.

Abramovich's maximum-flow principle for an ideal (frictionless) liquid swirler. The swirl leaves
an air core in the exit orifice, so only a ring of liquid flows. The geometry parameter

    A = R_in · r_exit / (n · r_port²)

(R_in: distance from the swirler axis to each inlet port's axis, n: inlet port count) sets the
share φ of the exit area filled with liquid through A = (1 − φ)·√2 / φ^1.5, and then

    Cd = φ · √(φ / (2 − φ))

It's a liquid theory; boiling nitrous flows somewhat differently, so treat it as a starting
estimate until a cold flow measures the real Cd.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq


def swirl_A(D_exit: float, ports: int, D_port: float, R_in: float) -> float:
    return R_in * (0.5 * D_exit) / (ports * (0.5 * D_port) ** 2)


def swirl_fill(A: float) -> float:
    """Share of the exit area filled with liquid (1 − air-core area share)."""
    return brentq(lambda phi: (1.0 - phi) * math.sqrt(2.0) / phi ** 1.5 - A, 1e-6, 1.0)


def swirl_cd(D_exit: float, ports: int, D_port: float, R_in: float) -> float:
    phi = swirl_fill(swirl_A(D_exit, ports, D_port, R_in))
    return phi * math.sqrt(phi / (2.0 - phi))


def swirl_port_D(D_exit: float, ports: int, R_in: float, cd: float) -> float:
    """Inlet port diameter that gives a Cd of cd. The ports can be at most twice their offset across."""
    if swirl_cd(D_exit, ports, 2.0 * R_in, R_in) < cd:
        raise ValueError("No port size reaches this flow. Widen the exit, add ports or move them toward the axis.")
    return brentq(lambda d: swirl_cd(D_exit, ports, d, R_in) - cd, 1e-4 * R_in, 2.0 * R_in)


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
