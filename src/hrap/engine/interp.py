"""MATLAB interp1x / interp2x. Ported including edge clamping and histc indexing."""
from __future__ import annotations

import numpy as np


def _histc_bin(xi: float, edges: np.ndarray) -> int:
    """1-based MATLAB histc bin index: edges[k-1] <= xi < edges[k], last bin includes the right edge."""
    edges = np.asarray(edges, dtype=float).ravel()
    n = edges.size
    if xi >= edges[-1]:
        return n
    if xi <= edges[0]:
        return 1
    # searchsorted side='right' -> first index with edges[i] > xi
    i = int(np.searchsorted(edges, xi, side="right"))
    # MATLAB histc returns the index of the bin whose left edge is <= xi
    return max(1, i)  # i is 1-based equivalent of (0-based i)


def interp1x(X, Y, xi: float) -> float:
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if xi <= X[0]:
        return float(Y[0])
    if xi >= X[-1]:
        return float(Y[-1])
    k = _histc_bin(xi, X)  # 1-based preceding index
    k0 = k - 1
    return float(((Y[k0 + 1] - Y[k0]) / (X[k0 + 1] - X[k0])) * (xi - X[k0]) + Y[k0])


def interp2x(X, Y, Z, xi: float, yi: float):
    """Bilinear interpolation matching MATLAB util/interp2x.m.

    MATLAB comb.m calls interp2x(prop_OF, prop_Pc, prop_k, OF, P_cmbr).
    Saved .mat tables store Z with shape (n_Y, n_X) = (n_Pc, n_OF), so
    Z[:, k] is the Y (Pc) column for OF index k.

    Z may also be a tuple of same-shape tables, which returns a tuple and finds the cell once.
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    # k indexes X (OF); l indexes Y (Pc). MATLAB 1-based.
    if xi >= X[-1]:
        k = X.size
    elif xi <= X[0]:
        k = 1
    else:
        k = _histc_bin(xi, X)

    if yi >= Y[-1]:
        l = Y.size
    elif yi <= Y[0]:
        l = 1
    else:
        l = _histc_bin(yi, Y)

    k0 = k - 1
    l0 = l - 1

    def one(Z) -> float:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            Z = np.atleast_2d(Z)

        if xi >= X[-1]:
            Z1 = Z[:, X.size - 1]
        elif xi <= X[0]:
            Z1 = Z[:, 0]
        else:
            Z1 = Z[:, k0]

        if yi >= Y[-1]:
            zi1 = float(Z1[l0])
        elif yi <= Y[0]:
            zi1 = float(Z1[0])
        else:
            zi1 = float(((Z1[l0 + 1] - Z1[l0]) / (Y[l0 + 1] - Y[l0])) * (yi - Y[l0]) + Z1[l0])

        if xi >= X[-1]:
            Z2 = Z[:, X.size - 1]
        elif xi < X[0]:  # MATLAB uses < not <=
            Z2 = Z[:, 0]
        else:
            Z2 = Z[:, k0 + 1] if k0 + 1 < Z.shape[1] else Z[:, k0]

        if yi >= Y[-1]:
            zi2 = float(Z2[l0])
        else:
            # MATLAB does not special-case yi <= Y(1) here
            if l0 + 1 >= Z2.size:
                zi2 = float(Z2[l0])
            else:
                zi2 = float(((Z2[l0 + 1] - Z2[l0]) / (Y[l0 + 1] - Y[l0])) * (yi - Y[l0]) + Z2[l0])

        if xi >= X[-1]:
            return zi2
        if xi <= X[0]:
            return zi1
        return float(((zi2 - zi1) / (X[k0 + 1] - X[k0])) * (xi - X[k0]) + zi1)

    if isinstance(Z, tuple):
        return tuple(one(z) for z in Z)
    return one(Z)
