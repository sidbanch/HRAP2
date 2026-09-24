"""Saturated N2O tank blowdown. Port of MATLAB util/tank.m."""
from __future__ import annotations

import math

import numpy as np

from hrap.engine.fzero import matlab_fzero
from hrap.engine.nox import nox, vapor_pressure
from hrap.engine.types import Output, Settings, State


def _sat_props(s: Settings, T: float):
    if s.get_sat_props is not None:
        return s.get_sat_props(T)
    return nox(T)


def tank(s: Settings, o: Output, x: State, t: float) -> State:
    dt = s.dt
    x.ox_props = _sat_props(s, x.T_tnk)
    x.P_tnk = x.ox_props.Pv

    dP = x.P_tnk - x.P_cmbr
    Mcc = math.sqrt(
        x.ox_props.Z * 1.31 * 188.91 * x.T_tnk * (x.P_cmbr / x.P_tnk) ** (0.31 / 1.31)
    )
    Matm = math.sqrt(
        x.ox_props.Z * 1.31 * 188.91 * x.T_tnk * (s.Pa / x.P_tnk) ** (0.31 / 1.31)
    )
    if Mcc >= 1:
        Mcc = 1.0
    if Matm >= 1:
        Matm = 1.0
    if dP < 0:
        dP = 0.0

    def vap_mdot(CdA: float, N: float, M: float) -> float:
        return (
            (CdA * N * x.P_tnk / math.sqrt(x.T_tnk))
            * math.sqrt(1.31 / (x.ox_props.Z * 188.91))
            * M
            * (1.0 + (0.31) / 2.0 * M ** 2) ** (-2.31 / 0.62)
        )

    def liq_mdot() -> float:
        spi = s.inj_CdA * s.inj_N * math.sqrt(2.0 * x.ox_props.rho_l * dP)
        if s.hem_flux is None:
            return spi
        hem = s.inj_CdA_HEM * s.inj_N * s.hem_flux(x.T_tnk, x.P_cmbr / x.P_tnk)
        if s.inj_model == "HEM":
            return hem
        return (s.dyer_kappa * spi + hem) / (1.0 + s.dyer_kappa)

    if s.tburn == 0 or t <= s.tburn:
        if s.vnt_S == 0:
            x.mdot_v = 0.0
            x.mdot_o = vap_mdot(s.inj_CdA, s.inj_N, Mcc) if x.mLiq_new == 0 else liq_mdot()
            mD = (x.mdot_o + x.mdot_v) * dt
        elif s.vnt_S == 1:
            x.mdot_v = vap_mdot(s.vnt_CdA, 1.0, Matm)
            x.mdot_o = vap_mdot(s.inj_CdA, s.inj_N, Mcc) if x.mLiq_new == 0 else liq_mdot()
            mD = (x.mdot_o + x.mdot_v) * dt
        elif s.vnt_S == 2:
            x.mdot_v = vap_mdot(s.vnt_CdA, 1.0, Matm)
            if x.mLiq_new == 0:
                x.mdot_o = vap_mdot(s.inj_CdA, s.inj_N, Mcc)
            else:
                x.mdot_o = liq_mdot() + x.mdot_v
            mD = x.mdot_o * dt
        else:
            raise ValueError("Error: Vent State Undefined")
    elif s.tburn > 0 and t > s.tburn:
        x.mdot_o = 0.0
        x.mdot_v = 0.0
        mD = 0.0
    else:
        mD = 0.0

    m_o_old = x.m_o
    x.m_o = x.m_o - x.mdot_o * dt

    if x.mLiq_new < x.mLiq_old and x.mLiq_new > 0 and x.mdot_o > 0:
        x.mLiq_old = x.mLiq_new - mD
        x.ox_props = _sat_props(s, x.T_tnk)
        x.mLiq_new = (s.tnk_V - (x.m_o / x.ox_props.rho_v)) / (
            (1.0 / x.ox_props.rho_l) - (1.0 / x.ox_props.rho_v)
        )
        mv = x.mLiq_old - x.mLiq_new
        dT = -mv * x.ox_props.Hv / (x.mLiq_new * x.ox_props.Cp)
        x.T_tnk = x.T_tnk + dT
        op = _sat_props(s, x.T_tnk)
        x.dP = op.Pv - x.P_tnk
        # MATLAB assigns op only into dP; ox_props is refreshed at the next tank() call.
    elif x.mLiq_new >= x.mLiq_old and x.mLiq_new > 0 and x.mdot_o > 0:
        # MATLAB: dP_avg = mean(o.dP(1:sum(o.dP<0))) — mean of the first N samples,
        # where N is the count of negative dP, not the mean of the negative values.
        nneg = int(np.sum(o.dP < 0))
        dP_avg = float(np.mean(o.dP[:nneg])) if nneg > 0 else float("nan")
        P_new = x.P_tnk + dP_avg

        def vp(T: float) -> float:
            # MATLAB uses the Wagner Pv fit, not a full NOX() call.
            if s.get_sat_props is not None:
                return s.get_sat_props(T).Pv - P_new
            return vapor_pressure(T) - P_new

        from scipy.optimize import brentq

        try:
            x.T_tnk = float(brentq(vp, 183.15, 309.56, xtol=2.2e-16, maxiter=200))
        except ValueError:
            x.T_tnk = matlab_fzero(vp, x.T_tnk)
        x.dP = x.ox_props.Pv - x.P_tnk
        x.ox_props = _sat_props(s, x.T_tnk)
        x.mLiq_new = (s.tnk_V - (x.m_o / x.ox_props.rho_v)) / (
            (1.0 / x.ox_props.rho_l) - (1.0 / x.ox_props.rho_v)
        )
        x.mLiq_old = 0.0
    elif x.mLiq_new <= 0 and x.mdot_o > 0:
        if x.mLiq_new != 0:
            x.mLiq_new = 0.0
        Z_old = x.ox_props.Z
        Zguess = Z_old
        epsilon = 1.0
        Ti = x.T_tnk
        Pi = x.P_tnk
        while epsilon >= 0.000001:
            T_ratio = ((Zguess * x.m_o) / (Z_old * m_o_old)) ** 0.3
            x.T_tnk = T_ratio * Ti
            P_ratio = T_ratio ** (1.3 / 0.3)
            x.P_tnk = P_ratio * Pi
            x.ox_props = _sat_props(s, x.T_tnk)
            Z = x.ox_props.Z
            epsilon = abs(Zguess - Z)
            Zguess = (Zguess + Z) / 2.0
    return x
