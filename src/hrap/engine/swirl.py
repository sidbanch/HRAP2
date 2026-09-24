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
