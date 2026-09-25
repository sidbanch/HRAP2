from PySide6.QtCore import QRectF

from hrap.gui.viz import MotorView, _geom, _vent_visible, liquid_rect, pack_hrects
from hrap.layout import INJECTOR_L, PLATE_L


def test_plate_and_injector_lengths():
    m = MotorView(
        tnk_L=0.20,
        tnk_D=0.10,
        grn_L=0.40,
        grn_OD=0.08,
        grn_ID=0.04,
        inj_D=0.006,
        inj_N=3,
        vnt_state="Internal",
        vnt_D=0.002,
        noz_thrt=0.025,
        noz_exit=0.05,
        fill_frac=0.9,
    )
    g = _geom(m)
    assert abs((g.x_plate1 - g.x_plate0) - PLATE_L) < 1e-12
    assert abs((g.x_inj1 - g.x_inj0) - INJECTOR_L) < 1e-12
    assert g.x_inj0 == g.x_plate0
    assert g.x_inj1 > g.x_plate1  # orifices stick out to the right of the plate
    assert g.L_conv > 0
    assert g.L_div > g.L_conv  # 15° diverge is longer than 60° converge
    assert g.vnt_on
    assert abs(g.x_tnk0 - 0.0) < 1e-12
    assert abs(g.x_cmbr0 - g.x_tnk1) < 1e-12


def test_explicit_starts_open_a_feed_gap():
    m = MotorView(
        tnk_L=0.20,
        tnk_D=0.10,
        grn_L=0.40,
        grn_OD=0.08,
        grn_ID=0.04,
        inj_D=0.006,
        inj_N=3,
        vnt_state="Internal",
        vnt_D=0.002,
        noz_thrt=0.025,
        noz_exit=0.05,
        fill_frac=0.9,
        tnk_start=0.05,
        cmbr_start=0.40,
        tnk_dry_kg=4.0,
        cmbr_dry_kg=6.0,
    )
    g = _geom(m)
    assert abs(g.x_tnk0 - 0.05) < 1e-12
    assert abs(g.x_cmbr0 - 0.40) < 1e-12
    assert g.x_cmbr0 > g.x_tnk1
    assert g.x_min == g.x_tnk0
    assert abs(g.x_end - g.x_noz) < 1e-12
    assert abs((g.x_noz - g.x_th) - g.L_div) < 1e-12


def test_orifice_vent_stays_on_for_internal_and_nonzero_diameter():
    base = dict(
        tnk_L=0.20,
        tnk_D=0.10,
        grn_L=0.20,
        grn_OD=0.08,
        grn_ID=0.04,
        inj_D=0.006,
        inj_N=3,
        noz_thrt=0.025,
        noz_exit=0.05,
        fill_frac=0.9,
    )
    internal = MotorView(**base, vnt_state="Internal", vnt_D=0.002)
    assert _geom(internal).vnt_on
    assert _vent_visible("Internal", 0.0)
    tiny = MotorView(**base, vnt_state="None", vnt_D=0.0003)
    assert _geom(tiny).vnt_on
    hidden = MotorView(**base, vnt_state="None", vnt_D=0.0)
    assert not _geom(hidden).vnt_on


def test_liquid_sits_on_the_right():
    tank = QRectF(0.0, 0.0, 100.0, 20.0)
    liq = liquid_rect(tank, 0.4)
    assert abs(liq.right() - tank.right()) < 1e-9
    assert abs(liq.width() - 40.0) < 1e-9
    assert abs(liq.left() - 60.0) < 1e-9
    full = liquid_rect(tank, 1.0)
    assert abs(full.left() - tank.left()) < 1e-9
    empty = liquid_rect(tank, 0.0)
    assert empty.width() == 0.0


def _assert_packed(placed, x0, x1, gap):
    assert placed
    assert placed[0][0] >= x0 - 1e-9
    assert placed[-1][0] + placed[-1][1] <= x1 + 1e-9
    for i in range(len(placed) - 1):
        left, width = placed[i]
        nxt, _ = placed[i + 1]
        assert nxt + 1e-9 >= left + width + gap


def test_pack_hrects_no_overlap_when_centers_collide():
    gap = 6.0
    placed = pack_hrects([10.0, 12.0, 14.0], [40.0, 40.0, 40.0], 0.0, 400.0, gap=gap)
    _assert_packed(placed, 0.0, 400.0, gap)
    for i in range(len(placed) - 1):
        assert placed[i + 1][0] - (placed[i][0] + placed[i][1]) >= gap - 1e-9


def test_pack_hrects_bunches_in_narrow_band():
    gap = 4.0
    placed = pack_hrects([0.0, 80.0, 160.0, 240.0, 320.0], [90.0] * 5, 0.0, 200.0, gap=gap, min_width=20.0)
    _assert_packed(placed, 0.0, 200.0, gap)
    span = (placed[-1][0] + placed[-1][1]) - placed[0][0]
    assert span <= 200.0 + 1e-9
    for _, width in placed:
        assert width < 90.0
