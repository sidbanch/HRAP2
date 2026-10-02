"""CSV, RSE, and ENG export. RSE CG is millimeters (MATLAB v1.01)."""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from hrap.engine.impulse import impulse_class
from hrap.engine.types import Output, Settings


def export_csv(path: str | Path, o: Output, s: Settings | None = None) -> None:
    path = Path(path)
    cols = {
        "t": o.t,
        "F_thr": o.F_thr,
        "P_tnk": o.P_tnk,
        "P_cmbr": o.P_cmbr,
        "mdot_o": o.mdot_o,
        "mdot_f": o.mdot_f,
        "mdot_n": o.mdot_n,
        "OF": o.OF,
        "grn_ID": o.grn_ID,
        "rdot": o.rdot,
        "m_o": o.m_o,
        "m_f": o.m_f,
        "dP": o.dP,
    }
    if o.m_t is not None:
        cols["m_t"] = o.m_t
    if o.cg is not None:
        cols["cg"] = o.cg
    header = ",".join(cols.keys())
    data = np.column_stack([cols[k] for k in cols])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def _nonzero_window(F: np.ndarray) -> tuple[int, int]:
    nz = np.flatnonzero(F > 0.0)
    if nz.size == 0:
        return 0, max(0, F.size - 1)
    return int(nz[0]), int(nz[-1])


def _bin_resample(t: np.ndarray, bins: np.ndarray, *series: np.ndarray):
    Ibin = np.searchsorted(bins, t, side="right") - 1
    Ibin = np.clip(Ibin, 0, bins.size - 1)
    Nbin = np.bincount(Ibin, minlength=bins.size)
    it = np.flatnonzero(Nbin > 0)
    out = [(np.bincount(Ibin, y, minlength=bins.size) / np.maximum(Nbin, 1))[it] for y in series]
    return bins[it], *out


def get_impulse_letter(Itot: float) -> str:
    letter, _pct = impulse_class(Itot)
    return letter if isinstance(letter, str) else "U"


def _mass_cg_series(
    o: Output,
    s: Settings,
    *,
    dry_mass: float | None = None,
    dry_cg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wet mass and time-varying CG (meters) for RSE / ENG.

    Always ``dry + remaining oxidizer + remaining fuel`` so tank/chamber dry
    mass from the form is present even if the last run stored propellant-only ``m_t``.
    """
    dry = float(s.mtr_m if dry_mass is None else dry_mass)
    dry_x = float(s.mtr_cg if dry_cg is None else dry_cg)
    m = dry + o.m_o + o.m_f
    grain_cg = s.cmbr_X - 0.5 * s.grn_L
    tank_len = s.tnk_V / (0.25 * math.pi * s.tnk_D ** 2) if s.tnk_D else 0.0
    tank_cg = s.tnk_X - 0.5 * tank_len
    tot = np.maximum(m, 1e-12)
    rebuilt = (o.m_o * tank_cg + o.m_f * grain_cg + dry * dry_x) / tot
    if (
        o.cg is not None
        and o.m_t is not None
        and o.cg.size == m.size
        and np.any(np.abs(o.cg) > 0.0)
        and abs(float(o.m_t[0]) - float(m[0])) < 1e-6
    ):
        return m, o.cg
    return m, rebuilt


def export_rse(
    path: str | Path,
    o: Output,
    s: Settings,
    *,
    OD: float | None = None,
    L: float | None = None,
    mfg: str = "HRAP",
    Nt_max: int = 200,
    dry_mass: float | None = None,
    dry_cg: float | None = None,
) -> None:
    t, F = o.t, o.F_thr
    mdot = o.mdot_o + o.mdot_f
    m, Cg = _mass_cg_series(o, s, dry_mass=dry_mass, dry_cg=dry_cg)

    i0, i1 = _nonzero_window(F)
    t, F, mdot, m, Cg = t[i0 : i1 + 1], F[i0 : i1 + 1], mdot[i0 : i1 + 1], m[i0 : i1 + 1], Cg[i0 : i1 + 1]
    T_burn = float(t[-1] - t[0]) if t.size > 1 else 0.0
    Itot = float(np.trapezoid(F, t)) if t.size > 1 else 0.0
    F_max = float(np.max(F)) if F.size else 0.0
    prop_burnt = float(np.trapezoid(mdot, t)) if t.size > 1 else 0.0
    Isp = Itot / (prop_burnt * 9.81) if prop_burnt > 0 else 0.0
    F_avg = Itot / T_burn if T_burn > 0 else 0.0

    if t.size > Nt_max:
        T_start, Nt_start = t[0] + 0.1, Nt_max // 10
        bins = np.concatenate(
            [
                np.linspace(t[0], T_start, Nt_start, endpoint=False),
                np.linspace(T_start, t[-1], Nt_max - Nt_start),
            ]
        )
        t, F, mdot, m, Cg = _bin_resample(t, bins, F, mdot, m, Cg)

    if F.size and F[-1] != 0.0:
        t = np.append(t, t[-1] + 1e-5)
        F = np.append(F, 0.0)
        mdot = np.append(mdot, 0.0)
        m = np.append(m, m[-1])
        Cg = np.append(Cg, Cg[-1])

    if t.size > 1 and np.trapezoid(F, t) != 0:
        F = F * (Itot / np.trapezoid(F, t))

    OD = max(s.tnk_D, s.grn_OD) if OD is None else OD
    L = max(s.tnk_X, s.cmbr_X) if L is None else L
    D_exit = math.sqrt(s.noz_ER) * s.noz_thrt
    code = get_impulse_letter(Itot)

    lines = [
        "<engine-database>",
        "    <engine-list>",
        (
            f'    <engine FDiv="10" FFix="1" FStep="-1." Isp="{Isp}" Itot="{Itot}"'
            f' Type="Hybrid" auto-calc-cg="0" auto-calc-mass="0" avgThrust="{F_avg}"'
            f' burn-time="{T_burn}" cgDiv="10" cgFix="1" cgStep="-1." code="{code}{int(round(F_avg))}" delays="0"'
            f' dia="{OD * 1000.0}" D_exit="{D_exit * 1000.0}" initWt="{m[0] * 1000.0}" len="{L * 1000.0}"'
            f' mDiv="10" mFix="1" mStep="-1." massFrac="{(m[0] - m[-1]) / m[0] if m[0] else 0.0}"'
            f' mfg={quoteattr(mfg)} peakThrust="{F_max}"'
            f' propWt="{(m[0] - m[-1]) * 1000.0}" tDiv="10" tFix="1" tStep="-1."'
            f' throatDia="{s.noz_thrt * 1000.0}">'
        ),
        "    <data>",
    ]
    for i in range(t.size):
        lines.append(
            f'        <eng-data cg="{Cg[i] * 1000.0}" f="{max(0.0, float(F[i]))}" '
            f'm="{m[i] * 1000.0}" t="{t[i]}"/>'
        )
    lines += ["    </data>", "    </engine>", "    </engine-list>", "</engine-database>"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_eng(
    path: str | Path,
    o: Output,
    s: Settings,
    *,
    OD: float | None = None,
    L: float | None = None,
    mfg: str = "HRAP",
    dry_mass: float | None = None,
    dry_cg: float | None = None,
) -> None:
    t, F = o.t, o.F_thr
    m, _cg = _mass_cg_series(o, s, dry_mass=dry_mass, dry_cg=dry_cg)
    i0, i1 = _nonzero_window(F)
    t, F, m = t[i0 : i1 + 1], F[i0 : i1 + 1], m[i0 : i1 + 1]
    T_burn = float(t[-1] - t[0]) if t.size > 1 else 0.0
    Itot = float(np.trapezoid(F, t)) if t.size > 1 else 0.0
    bins = np.linspace(t[0], t[-1], 31) if t.size else np.array([0.0])
    if t.size:
        t, F = _bin_resample(t, bins, F)
    t = np.append(t, (t[0] + T_burn * 31 / 30) if t.size else 0.0)
    F = np.append(F, 0.0)
    if t.size > 1 and np.trapezoid(F, t) != 0:
        F = F * (Itot / np.trapezoid(F, t))
    OD = max(s.tnk_D, s.grn_OD) if OD is None else OD
    L = max(s.tnk_X, s.cmbr_X) if L is None else L
    code = get_impulse_letter(Itot)
    F_avg = int(round(Itot / T_burn)) if T_burn else 0
    dry = float(s.mtr_m if dry_mass is None else dry_mass)
    cg0 = float(o.cg[i0]) if o.cg is not None and o.cg.size else float(s.mtr_cg if dry_cg is None else dry_cg)
    lines = [
        f"; HRAP-HCAT-Fork cg0={cg0:.6f} m dry={dry:.6f} kg (time-varying CG in .rse)",
        f"{code}{F_avg} {1000.0 * OD} {1000.0 * L} P {m[0] - m[-1] if m.size else 0.0} {m[0] if m.size else 0.0}"
        f" {mfg.replace(' ', '_')}",
    ]
    for i in range(min(32, t.size)):
        lines.append(f" {t[i]} {F[i]} ")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
