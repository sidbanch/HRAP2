"""Nozzle throat × injector Cd sweep, for sizing a throat against a chamber pressure limit."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from hrap.engine.sim import run
from hrap.engine.summary import summarize
from hrap.io.config import clone_cfg, resolve


@dataclass(frozen=True)
class SweepCase:
    throat: float         # m
    inj_Cd: float
    peak_P_cmbr: float    # Pa, absolute
    avg_inj_dP: float     # Pa, tank minus chamber, averaged over the burn; skips the chamber-filling spike
    total_impulse: float  # N·s
    peak_thrust: float    # N
    burn_time: float      # s
    end_cond: str


def run_case(cfg: dict[str, Any], throat: float, cd: float) -> SweepCase:
    case_cfg = clone_cfg(cfg)
    case_cfg.update(noz_thrt=float(throat), noz_thrt_unit="m", inj_Cd=float(cd))
    s, x = resolve(case_cfg)
    x, o = run(s, x)
    info = summarize(s, x, o)
    burning = o.F_thr > 0
    inj_dP = o.P_tnk[burning] - o.P_cmbr[burning]
    return SweepCase(
        throat=float(throat),
        inj_Cd=float(cd),
        peak_P_cmbr=float(np.max(o.P_cmbr)),
        avg_inj_dP=float(np.mean(inj_dP)) if inj_dP.size else 0.0,
        total_impulse=info["total_impulse"],
        peak_thrust=info["peak_thrust"],
        burn_time=info["burn_time"],
        end_cond=info["end_cond"],
    )


def sweep(cfg: dict[str, Any], throats: Sequence[float], cds: Sequence[float]) -> Iterator[SweepCase]:
    """Run ``cfg`` once per throat diameter (m) and injector Cd, keeping every other input.

    Cases run in parallel and are yielded as they finish, not in grid order. A nozzle defined by
    exit diameter keeps that exit, so its expansion ratio changes with the throat.
    """
    pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
    try:
        futures = [pool.submit(run_case, cfg, t, c) for t in throats for c in cds]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def uses_spi(cfg: dict[str, Any]) -> bool:
    """True for HRAP's liquid-only injector model, whose flow is only trustworthy under the ΔP warning."""
    return cfg.get("inj_model", "SPI") == "SPI"


def passing_throats(cases: Sequence[SweepCase], max_P_cmbr: float, max_inj_dP: float) -> list[float]:
    """Throats that stay under both limits for every Cd swept."""
    ok: dict[float, bool] = {}
    for c in cases:
        ok[c.throat] = ok.get(c.throat, True) and c.peak_P_cmbr <= max_P_cmbr and c.avg_inj_dP <= max_inj_dP
    return sorted(t for t, passed in ok.items() if passed)
