from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def compare_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare an HRAP JSON motor to a MATLAB/CSV golden trace.")
    parser.add_argument("motor", type=Path, help="Motor JSON (or MATLAB .mat)")
    parser.add_argument("csv", type=Path, help="Golden CSV with columns t,F_thr,P_tnk,P_cmbr,...")
    parser.add_argument("--dt-tol", type=float, default=1e-8)
    args = parser.parse_args(argv)

    from hrap.engine.sim import run
    from hrap.io.config import load_json, load_matlab_mat, resolve

    if args.motor.suffix.lower() == ".mat":
        cfg = load_matlab_mat(args.motor)
    else:
        cfg = load_json(args.motor)
    s, x = resolve(cfg)
    _x, o = run(s, x)

    gold = np.genfromtxt(args.csv, delimiter=",", names=True)
    mapping = {
        "t": o.t,
        "F_thr": o.F_thr,
        "P_tnk": o.P_tnk,
        "P_cmbr": o.P_cmbr,
        "mdot_o": o.mdot_o,
        "mdot_f": o.mdot_f,
        "OF": o.OF,
        "grn_ID": o.grn_ID,
        "m_o": o.m_o,
        "m_f": o.m_f,
    }
    n = min(o.t.size, gold.shape[0] if gold.dtype.names is None else gold.size)
    print(f"python n={o.t.size}  golden n={gold.size if gold.dtype.names else gold.shape[0]}  end={o.sim_end_cond}")
    max_rel = 0.0
    for name, arr in mapping.items():
        if gold.dtype.names and name not in gold.dtype.names:
            continue
        g = gold[name][:n] if gold.dtype.names else None
        if g is None:
            continue
        a = arr[:n]
        denom = np.maximum(np.abs(g), 1e-30)
        rel = np.max(np.abs(a - g) / denom)
        abserr = np.max(np.abs(a - g))
        max_rel = max(max_rel, float(rel))
        print(f"  {name:8s}  max_rel={rel:.3e}  max_abs={abserr:.3e}")
    ok = max_rel <= args.dt_tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HRAP from a motor JSON and write CSV.")
    parser.add_argument("motor", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("HRAP_output.csv"))
    args = parser.parse_args(argv)
    from hrap.engine.sim import run
    from hrap.engine.summary import format_summary, summarize
    from hrap.io.config import load_json, load_matlab_mat, resolve
    from hrap.io.export import export_csv

    cfg = load_matlab_mat(args.motor) if args.motor.suffix.lower() == ".mat" else load_json(args.motor)
    s, x = resolve(cfg)
    x, o = run(s, x)
    export_csv(args.output, o, s)
    print(format_summary(summarize(s, x, o)))
    print(f"wrote {args.output}")
    return 0


def _linspace(text: str) -> np.ndarray:
    lo, hi, n = text.split(":")
    return np.linspace(float(lo), float(hi), int(n))


def sweep_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep nozzle throat diameter and injector Cd for a motor JSON.")
    parser.add_argument("motor", type=Path)
    parser.add_argument("--throat", required=True, type=_linspace, help="min:max:count, e.g. 0.3:0.6:7")
    parser.add_argument("--throat-unit", default="in")
    parser.add_argument("--cd", required=True, type=_linspace, help="min:max:count, e.g. 0.15:0.4:6")
    parser.add_argument("--max-chamber", type=float, default=500.0, help="chamber pressure limit, psi absolute")
    parser.add_argument("--max-dp", type=float, default=300.0,
                        help="average injector dP above which HRAP's liquid-only injector model overpredicts flow, psi")
    parser.add_argument("-o", "--output", type=Path, default=Path("HRAP_sweep.csv"))
    args = parser.parse_args(argv)
    from hrap.engine.sweep import passing_throats, sweep, uses_spi
    from hrap.io.config import load_json, load_matlab_mat
    from hrap.units import from_si, to_si

    cfg = load_matlab_mat(args.motor) if args.motor.suffix.lower() == ".mat" else load_json(args.motor)
    limits = "both limits"
    if not uses_spi(cfg):  # HEM and Dyer model the high-ΔP flow, so the warning doesn't apply
        args.max_dp, limits = float("inf"), "the chamber limit"
    throats = [to_si(v, args.throat_unit, "length") for v in args.throat]
    psi = lambda pa: from_si(pa, "psi", "pressure")
    unit = lambda m: from_si(m, args.throat_unit, "length")
    total = len(throats) * len(args.cd)
    cases = []
    for c in sweep(cfg, throats, args.cd):
        cases.append(c)
        print(f"\r{len(cases)}/{total} cases", end="", flush=True)
    print()
    cases.sort(key=lambda c: (c.throat, c.inj_Cd))
    header = (f"throat_{args.throat_unit},inj_Cd,peak_P_cmbr_psi,avg_inj_dP_psi,"
              "total_impulse_Ns,peak_thrust_N,burn_time_s,end_cond,flags")
    lines = [header]
    for c in cases:
        flags = [f for f, bad in (("over_chamber_limit", psi(c.peak_P_cmbr) > args.max_chamber),
                                  ("high_injector_dP", psi(c.avg_inj_dP) > args.max_dp)) if bad]
        lines.append(f"{unit(c.throat):.6g},{c.inj_Cd:.6g},{psi(c.peak_P_cmbr):.6g},{psi(c.avg_inj_dP):.6g},"
                     f"{c.total_impulse:.6g},{c.peak_thrust:.6g},{c.burn_time:.6g},{c.end_cond},{' '.join(flags)}")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = passing_throats(cases, to_si(args.max_chamber, "psi", "pressure"), to_si(args.max_dp, "psi", "pressure"))
    if ok:
        print(f"Throats under {limits} for every Cd: {', '.join(f'{unit(t):.4g}' for t in ok)} {args.throat_unit}")
    else:
        print(f"No throat in this range stays under {limits} for every Cd.")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(run_main())
