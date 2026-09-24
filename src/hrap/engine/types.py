from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class OxProps:
    Pv: float
    rho_l: float
    rho_v: float
    Hv: float
    Cp: float
    Z: float


@dataclass
class Propellant:
    name: str
    opt_OF: float
    rho: float
    reg: np.ndarray
    OF: np.ndarray
    Pc: np.ndarray
    k: np.ndarray
    M: np.ndarray
    T: np.ndarray


@dataclass
class Settings:
    dt: float
    tmax: float
    tburn: float
    Pa: float
    tnk_V: float
    tnk_D: float
    inj_CdA: float
    inj_N: float
    vnt_S: int
    vnt_CdA: float
    grn_OD: float
    grn_L: float
    grn_ID0: float
    prop_Rho: float
    prop_Reg: np.ndarray
    const_OF: float
    regression_model: str
    cmbr_V: float
    cstar_eff: float
    noz_thrt: float
    noz_ER: float
    noz_Cd: float
    noz_eff: float
    mp_calc: int
    mtr_m: float
    mtr_cg: float
    tnk_X: float
    cmbr_X: float
    mtr_nm: str
    prop_nm: str
    prop_OF: np.ndarray
    prop_Pc: np.ndarray
    prop_k: np.ndarray
    prop_M: np.ndarray
    prop_T: np.ndarray
    get_sat_props: Optional[Callable[[float], OxProps]] = None
    grain_fn: Optional[Callable] = None
    inj_model: str = "SPI"  # "SPI" (HRAP), "HEM" or "Dyer"
    inj_CdA_HEM: float = 0.0
    dyer_kappa: float = 1.0
    hem_flux: Optional[Callable[[float, float], float]] = None


@dataclass
class State:
    T_tnk: float
    P_tnk: float
    P_cmbr: float
    m_o: float
    mLiq_new: float
    mLiq_old: float
    mdot_o: float
    mdot_v: float
    mdot_f: float
    mdot_n: float
    OF: float
    rdot: float
    grn_ID: float
    grn_ID_old: float
    m_f: float
    m_g: float
    dP: float
    F_thr: float
    k: float = 1.4
    M: float = 29.0
    T: float = 300.0
    R: float = 287.0
    rho: float = 1.225
    cstar: float = 1.0
    ox_props: Optional[OxProps] = None
    dm_g: float = 0.0
    m_t: float = 0.0
    cg: float = 0.0


@dataclass
class Output:
    t: np.ndarray
    m_o: np.ndarray
    P_tnk: np.ndarray
    P_cmbr: np.ndarray
    mdot_o: np.ndarray
    mdot_f: np.ndarray
    OF: np.ndarray
    grn_ID: np.ndarray
    mdot_n: np.ndarray
    rdot: np.ndarray
    m_f: np.ndarray
    F_thr: np.ndarray
    dP: np.ndarray
    m_t: Optional[np.ndarray] = None
    cg: Optional[np.ndarray] = None
    sim_end_cond: str = ""
    extra: dict = field(default_factory=dict)
