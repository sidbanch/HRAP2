"""Load a measured thrust (and optional pressure) curve from CSV."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hrap.units import to_si

_TIME_TPLUS_MS = {"tplusms", "tplus"}
_TIME_S = {"t", "time", "times"}
_TIME_MS = {"timems"}
_THRUST_LBF = {"weightlbs", "weightlb", "weightlbf", "thrustlbf", "flbf", "weight"}
_THRUST_N = {"fthr", "thrust", "fn", "thrustn", "f"}
_TANK_PSI = {"pt1psi", "ptnkpsi", "tankpsi"}
_TANK_PA = {"ptnk"}
_CMBR_PSI = {"pcmbrpsi", "chamberpsi"}
_CMBR_PA = {"pcmbr"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


@dataclass(frozen=True)
class MeasuredCurve:
    """Measured time series in SI (seconds, newtons, pascals)."""

    path: str
    label: str
    t: np.ndarray
    F_thr: np.ndarray | None = None
    P_tnk: np.ndarray | None = None
    P_cmbr: np.ndarray | None = None

    def series(self, key: str) -> np.ndarray | None:
        if key not in ("F_thr", "P_tnk", "P_cmbr"):
            return None
        return getattr(self, key)


def _pick(header: list[str], *groups: set[str]) -> int | None:
    for names in groups:
        for i, col in enumerate(header):
            if _norm(col) in names:
                return i
    return None


def _column(rows: list[list[str]], index: int, unit: str, group: str) -> np.ndarray:
    values = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if index >= len(row) or not str(row[index]).strip():
            continue
        try:
            values[i] = to_si(float(row[index]), unit, group)
        except ValueError:
            continue
    return values


def load_measured_csv(path: str | Path) -> MeasuredCurve:
    """Read a stand-test or HRAP CSV. Time is seconds; thrust and pressure are SI."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    header, body = rows[0], [row for row in rows[1:] if any(cell.strip() for cell in row)]
    if not body:
        raise ValueError(f"{path.name} has no data rows.")

    t_ms = _pick(header, _TIME_TPLUS_MS, _TIME_MS)
    t_s = _pick(header, _TIME_S)
    if t_ms is not None:
        t = np.array([float(row[t_ms]) if t_ms < len(row) and str(row[t_ms]).strip() else np.nan for row in body]) / 1000.0
    elif t_s is not None:
        t = np.array([float(row[t_s]) if t_s < len(row) and str(row[t_s]).strip() else np.nan for row in body])
    else:
        raise ValueError(f"{path.name} needs a time column (Tplus_ms, t, Time, …).")

    thrust_i = _pick(header, _THRUST_LBF, _THRUST_N)
    if thrust_i is not None:
        unit = "lbf" if _norm(header[thrust_i]) in _THRUST_LBF else "N"
        F = _column(body, thrust_i, unit, "force")
    else:
        F = None

    tank_i = _pick(header, _TANK_PSI, _TANK_PA)
    if tank_i is not None:
        unit = "psi" if _norm(header[tank_i]) in _TANK_PSI else "Pa"
        P_tnk = _column(body, tank_i, unit, "pressure")
    else:
        P_tnk = None

    cmbr_i = _pick(header, _CMBR_PSI, _CMBR_PA)
    if cmbr_i is not None:
        unit = "psi" if _norm(header[cmbr_i]) in _CMBR_PSI else "Pa"
        P_cmbr = _column(body, cmbr_i, unit, "pressure")
    else:
        P_cmbr = None

    keep = np.isfinite(t)
    for series in (F, P_tnk, P_cmbr):
        if series is not None:
            keep &= np.isfinite(series)
    t = t[keep]
    if t.size < 2:
        raise ValueError(f"{path.name} needs at least two finite samples.")
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    if F is not None:
        F = F[keep][order]
    if P_tnk is not None:
        P_tnk = P_tnk[keep][order]
    if P_cmbr is not None:
        P_cmbr = P_cmbr[keep][order]
    if F is None and P_tnk is None and P_cmbr is None:
        raise ValueError(f"{path.name} needs a thrust or pressure column (Weight_Lbs, F_thr, PT1_PSI, …).")
    return MeasuredCurve(str(path), path.stem, t, F, P_tnk, P_cmbr)
