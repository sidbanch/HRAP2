"""Sequential Euler loop. Port of MATLAB core/sim_loop.m and sim_iteration.m."""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from hrap.engine.chamber import chamber
from hrap.engine.comb import comb
from hrap.engine.grain import regress
from hrap.engine.mass import mass_properties
from hrap.engine.nozzle import nozzle
from hrap.engine.tank import tank
from hrap.engine.types import Output, Settings, State


def _nsteps(s: Settings) -> int:
    return int(math.floor(s.tmax / s.dt + 1e-12)) + 1


def _blank_output(s: Settings) -> Output:
    n = _nsteps(s)
    def z():
        return np.zeros(n, dtype=float)
    o = Output(
        t=z(),
        m_o=z(),
        P_tnk=z(),
        P_cmbr=z(),
        mdot_o=z(),
        mdot_f=z(),
        OF=z(),
        grn_ID=z(),
        mdot_n=z(),
        rdot=z(),
        m_f=z(),
        F_thr=z(),
        dP=z(),
        m_t=z(),
        cg=z(),
        T_tnk=z(),
    )
    return o


def record(o: Output, x: State, t: float, i: int, s: Settings) -> None:
    o.t[i] = t
    o.m_o[i] = x.m_o
    o.P_tnk[i] = x.P_tnk
    o.P_cmbr[i] = x.P_cmbr
    o.mdot_o[i] = x.mdot_o
    o.mdot_f[i] = x.mdot_f
    o.OF[i] = x.OF
    o.grn_ID[i] = x.grn_ID
    o.mdot_n[i] = x.mdot_n
    o.rdot[i] = x.rdot
    o.m_f[i] = x.m_f
    o.dP[i] = x.dP
    o.F_thr[i] = x.F_thr
    o.T_tnk[i] = x.T_tnk
    mp = mass_properties(s, x)
    o.m_t[i] = mp[0]
    o.cg[i] = mp[1]


def sim_iteration(s: Settings, x: State, o: Output, t: float, i: int) -> tuple[Settings, State, Output, float]:
    """One step. ``i`` is MATLAB 1-based (first written sample is 2)."""
    dt = s.dt
    t = t + dt
    x = tank(s, o, x, t)
    if s.grain_fn is not None:
        x = s.grain_fn(s, x)
    else:
        x = regress(s, x)
    x = comb(s, x, t)
    x = chamber(s, x)
    x = nozzle(s, x)
    record(o, x, t, i - 1, s)
    return s, x, o, t


def sim_loop(
    s: Settings,
    x: State,
    o: Output,
    t: float = 0.0,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[Settings, State, Output, float]:
    i = 1  # MATLAB 1-based
    dt = s.dt
    n = o.t.size  # MATLAB length(o.t) == tmax/dt + 1
    last_emit = 0.0
    while True:
        t = (i - 1) * dt
        i = i + 1
        if i > n:
            o.sim_end_cond = "Max Simulation Time Reached"
            break
        s, x, o, t = sim_iteration(s, x, o, t, i)
        if on_progress is not None:
            now = time.monotonic()
            if now - last_emit >= 0.05 or i >= n or i <= 3:
                on_progress(min(i, n), n)
                last_emit = now
        if x.grn_ID >= s.grn_OD:
            o.sim_end_cond = "Fuel Depleted"
            break
        if x.m_o <= 0:
            o.sim_end_cond = "Oxidizer Depleted"
            break
        if t >= s.tmax:
            o.sim_end_cond = "Max Simulation Time Reached"
            break
        if x.P_cmbr <= s.Pa:
            o.sim_end_cond = "Burn Complete"
            break
    if on_progress is not None:
        on_progress(min(i, n), n)

    last = int(np.sum(o.t > 0)) + 1
    last = min(last, o.t.size)
    for name in (
        "t", "m_o", "P_tnk", "P_cmbr", "mdot_o", "mdot_f", "OF",
        "grn_ID", "mdot_n", "rdot", "m_f", "F_thr", "dP", "m_t", "cg", "T_tnk",
    ):
        setattr(o, name, getattr(o, name)[:last])
    o.inj_dP = o.P_tnk - o.P_cmbr
    return s, x, o, t


def run(
    s: Settings,
    x: State,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[State, Output]:
    """Run a full MATLAB-parity simulation from initialized settings and state."""
    o = _blank_output(s)
    # GUI stores initial conditions at index 0 (MATLAB index 1) before the loop
    o.m_o[0] = x.m_o
    o.P_tnk[0] = x.P_tnk
    o.P_cmbr[0] = x.P_cmbr
    o.mdot_o[0] = x.mdot_o
    o.mdot_f[0] = x.mdot_f
    o.OF[0] = x.OF
    o.grn_ID[0] = x.grn_ID
    o.mdot_n[0] = x.mdot_n
    o.rdot[0] = x.rdot
    o.m_f[0] = x.m_f
    o.T_tnk[0] = x.T_tnk
    mp = mass_properties(s, x)
    o.m_t[0] = mp[0]
    o.cg[0] = mp[1]
    if on_progress is not None:
        on_progress(0, o.t.size)
    s, x, o, _t = sim_loop(s, x, o, 0.0, on_progress=on_progress)
    return x, o
