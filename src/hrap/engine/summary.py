"""Performance summary matching MATLAB HRAP results panel."""
from __future__ import annotations

import numpy as np

from hrap.engine.impulse import impulse_class
from hrap.engine.types import Output, Settings, State
from hrap.units import DisplayUnits


def summarize(s: Settings, x: State, o: Output) -> dict:
    mask = o.F_thr > 0
    if np.any(mask):
        total_impulse = float(np.trapezoid(o.F_thr[mask], o.t[mask]))
        burn_time = float(o.t[mask][-1])
        peak_thrust = float(np.max(o.F_thr))
        avg_thrust = float(np.mean(o.F_thr[mask]))
    else:
        total_impulse = burn_time = peak_thrust = avg_thrust = 0.0
    motor_class, percent = impulse_class(total_impulse)
    if not isinstance(motor_class, str):
        motor_class, percent = "—", 0.0
    peak_pressure = float(np.max(o.P_cmbr) / 1e5) if o.P_cmbr.size else 0.0
    avg_pressure = float(np.mean(o.P_cmbr[o.P_cmbr > 0]) / 1e5) if np.any(o.P_cmbr > 0) else 0.0
    fuel_consumed = float(o.m_f[0] - o.m_f[-1]) if o.m_f.size else 0.0
    ox_consumed = float(o.m_o[0] - o.m_o[-1]) if o.m_o.size else 0.0
    avg_OF = ox_consumed / fuel_consumed if fuel_consumed else 0.0
    int_pressure = float(np.trapezoid(o.P_cmbr, o.t)) if o.t.size > 1 else 0.0
    Ath = 0.25 * np.pi * s.noz_thrt ** 2
    cstar = int_pressure * Ath / (fuel_consumed + ox_consumed) if (fuel_consumed + ox_consumed) else 0.0
    isp = total_impulse / ((ox_consumed + fuel_consumed) * 9.81) if (ox_consumed + fuel_consumed) else 0.0
    port_cm = float(x.grn_ID * 100.0)
    avg_inj_dP = float(np.mean(o.P_tnk[mask] - o.P_cmbr[mask])) if np.any(mask) else 0.0
    return {
        "name": s.mtr_nm,
        "propellant": s.prop_nm,
        "tnk_V_cc": s.tnk_V * 1e6,
        "burn_time": burn_time,
        "peak_thrust": peak_thrust,
        "avg_thrust": avg_thrust,
        "total_impulse": total_impulse,
        "peak_pressure_bar": peak_pressure,
        "avg_pressure_bar": avg_pressure,
        "port_cm": port_cm,
        "fuel_consumed": fuel_consumed,
        "ox_consumed": ox_consumed,
        "avg_OF": avg_OF,
        "cstar": cstar,
        "isp": isp,
        "avg_inj_dP": avg_inj_dP,
        "impulse_class": motor_class,
        "impulse_percent": percent,
        "end_cond": o.sim_end_cond,
    }


def format_summary(info: dict, units: DisplayUnits | None = None) -> str:
    u = units or DisplayUnits(pressure="bar", length="cm", volume="cc")
    return (
        f"Motor Name: {info['name']}\n"
        f"    Propellant: {info['propellant']}\n"
        f"    Oxidizer Tank Volume: {u.text(info['tnk_V_cc'] * 1e-6, 'volume', 6)}\n"
        f"    Burn Time: {info['burn_time']:.3f} s\n"
        f"    Peak Thrust: {u.text(info['peak_thrust'], 'force', 6)}\n"
        f"    Average Thrust: {u.text(info['avg_thrust'], 'force', 6)}\n"
        f"    Total Impulse: {u.text(info['total_impulse'], 'impulse', 7)}\n"
        f"    Peak Chamber Pressure: {u.text(info['peak_pressure_bar'] * 1e5, 'pressure', 6)} (absolute)\n"
        f"    Average Chamber Pressure: {u.text(info['avg_pressure_bar'] * 1e5, 'pressure', 6)} (absolute)\n"
        f"    Average Injector ΔP: {u.text(info['avg_inj_dP'], 'pressure', 6)}\n"
        f"    Port Diameter at Burnout: {u.text(info['port_cm'] * .01, 'length', 6)}\n"
        f"    Fuel Consumed: {u.text(info['fuel_consumed'], 'mass', 6)}\n"
        f"    Oxidizer Consumed: {u.text(info['ox_consumed'], 'mass', 6)}\n"
        f"    Average OF Ratio: {info['avg_OF']:.3f}\n"
        f"    Characteristic Velocity: {u.text(info['cstar'], 'speed', 6)}\n"
        f"    Specific Impulse: {info['isp']:.1f} s\n"
        f"    Motor Classification: {info['impulse_percent']:3.0f}% {info['impulse_class']}{info['avg_thrust']:.0f}\n"
        f"    Simulation Termination Condition: {info['end_cond']}"
    )
