"""Tank and thrust-chamber hardware layout (SI meters).

Station 0 is the user datum. Tank and chamber starts are the forward faces.
The chamber runs from the injector face through the nozzle: plate, pre-combustion
chamber, grain, post-combustion chamber, nozzle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INCH = 0.0254
PLATE_L = 1.0 * INCH
INJECTOR_L = 1.5 * INCH
FEED_GAP = 2.0 * INCH
CONV_HALF_DEG = 60.0
DIV_HALF_DEG = 15.0


def cylinder_length(volume: float, diameter: float) -> float:
    area = 0.25 * math.pi * max(diameter, 0.0) ** 2
    if area <= 0.0:
        return 0.0
    return max(float(volume) / area, 0.0)


def effective_tank_length(tnk_L: float, tnk_V: float, tnk_D: float) -> float:
    if tnk_L > 1e-9:
        return float(tnk_L)
    return cylinder_length(tnk_V, tnk_D)


def nozzle_cone_lengths(grn_OD: float, noz_thrt: float, noz_exit: float) -> tuple[float, float]:
    r_th = max(float(noz_thrt) * 0.5, 1e-6)
    r_ex = max(float(noz_exit) * 0.5, r_th)
    r_in = max(float(grn_OD) * 0.5, r_th)
    tan_c = math.tan(math.radians(CONV_HALF_DEG))
    tan_d = math.tan(math.radians(DIV_HALF_DEG))
    L_conv = max(0.0, (r_in - r_th) / tan_c) if tan_c > 0 else 0.0
    L_div = max(0.0, (r_ex - r_th) / tan_d) if tan_d > 0 else 0.0
    return L_conv, L_div


def nozzle_exit_diameter(noz_thrt: float, noz_ER: float) -> float:
    return math.sqrt(max(float(noz_ER), 0.0)) * float(noz_thrt)


@dataclass(frozen=True)
class MotorLayout:
    tnk0: float
    tnk1: float
    cmbr0: float
    cmbr1: float
    plate0: float
    plate1: float
    inj0: float
    inj1: float
    grn0: float
    grn1: float
    x_th: float
    x_noz: float
    x_case: float
    L_conv: float
    L_div: float
    tnk_L: float
    cmbr_L: float
    tnk_m: float
    cmbr_m: float
    tnk_D: float = 0.0
    grn_OD: float = 0.0

    @property
    def x_min(self) -> float:
        return min(self.tnk0, self.cmbr0)

    @property
    def x_max(self) -> float:
        return max(self.tnk1, self.x_noz, self.inj1)

    @property
    def overall_L(self) -> float:
        return max(self.x_max - self.x_min, 0.0)

    @property
    def overall_OD(self) -> float:
        """RSE / ENG motor diameter: wider of tank and TCA (grain/case)."""
        return max(self.tnk_D, self.grn_OD)

    @property
    def tnk_mid(self) -> float:
        return 0.5 * (self.tnk0 + self.tnk1)

    @property
    def cmbr_mid(self) -> float:
        return 0.5 * (self.cmbr0 + self.cmbr1)

    @property
    def dry_mass(self) -> float:
        return max(self.tnk_m, 0.0) + max(self.cmbr_m, 0.0)

    @property
    def dry_cg(self) -> float:
        m = self.dry_mass
        if m <= 0.0:
            return 0.0
        return (self.tnk_m * self.tnk_mid + self.cmbr_m * self.cmbr_mid) / m

    @property
    def tnk_aft(self) -> float:
        """MATLAB ``tnk_X``: aft (injector-side) tank face."""
        return self.tnk1

    @property
    def grain_aft(self) -> float:
        """MATLAB ``cmbr_X``: aft grain face."""
        return self.grn1


def motor_layout(
    *,
    tnk_start: float,
    tnk_L: float,
    tnk_m: float = 0.0,
    tnk_D: float = 0.0,
    cmbr_start: float | None = None,
    pre_L: float = 0.0,
    post_L: float = 0.0,
    cmbr_m: float = 0.0,
    grn_L: float,
    grn_OD: float,
    noz_thrt: float,
    noz_exit: float,
) -> MotorLayout:
    tnk_L = max(float(tnk_L), 1e-4)
    tnk0 = float(tnk_start)
    tnk1 = tnk0 + tnk_L
    if cmbr_start is None:
        cmbr0 = tnk1
    else:
        cmbr0 = float(cmbr_start)
    L_conv, L_div = nozzle_cone_lengths(grn_OD, noz_thrt, noz_exit)
    grn_L = max(float(grn_L), 1e-4)
    plate0 = cmbr0
    plate1 = cmbr0 + PLATE_L
    inj0 = plate0
    inj1 = plate0 + INJECTOR_L
    grn0 = plate1 + max(float(pre_L), 0.0)
    grn1 = grn0 + grn_L
    x_case = grn1 + max(float(post_L), 0.0)
    x_th = x_case + L_conv
    x_noz = x_th + L_div
    return MotorLayout(
        tnk0=tnk0,
        tnk1=tnk1,
        cmbr0=cmbr0,
        cmbr1=x_noz,
        plate0=plate0,
        plate1=plate1,
        inj0=inj0,
        inj1=inj1,
        grn0=grn0,
        grn1=grn1,
        x_th=x_th,
        x_noz=x_noz,
        x_case=x_case,
        L_conv=L_conv,
        L_div=L_div,
        tnk_L=tnk_L,
        cmbr_L=x_noz - cmbr0,
        tnk_m=max(float(tnk_m), 0.0),
        cmbr_m=max(float(cmbr_m), 0.0),
        tnk_D=max(float(tnk_D), 0.0),
        grn_OD=max(float(grn_OD), 0.0),
    )


def infer_component_stations(
    *,
    tnk_start: float,
    cmbr_start: float,
    tnk_L: float,
    grn_L: float,
    pre_L: float,
    tnk_X: float,
    cmbr_X: float,
) -> tuple[float, float]:
    """Forward starts from new fields, or from MATLAB aft stations when starts are unset."""
    if abs(tnk_start) <= 1e-12 and tnk_X > tnk_L + 1e-9:
        tnk_start = tnk_X - tnk_L
    if abs(cmbr_start) <= 1e-12 and cmbr_X > grn_L + 1e-9:
        cmbr_start = cmbr_X - grn_L - pre_L - PLATE_L
    elif abs(cmbr_start) <= 1e-12:
        cmbr_start = tnk_start + tnk_L + FEED_GAP
    return float(tnk_start), float(cmbr_start)
