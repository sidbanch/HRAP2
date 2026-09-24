"""Horizontal hybrid-motor schematic (vent, tank, injector, grain, nozzle)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hrap.layout import INJECTOR_L, PLATE_L, motor_layout
from hrap.units import DisplayUnits, LENGTH

SPEC_GAP = 6.0
SPEC_MIN_W = 56.0
SPAN_H = 28.0
RULER_H = 28.0
TITLE_H = 18.0
VENT_HEAD_PX = 16.0
VENT_MIN_W = 5.0
VENT_MIN_H = 12.0

DARK_VIZ = {
    "bg": "#1a1d23",
    "border": "#3a404c",
    "tank": "#5b8def",
    "tank_empty": "#252d3c",
    "grain": "#c98652",
    "port": "#14171c",
    "plate": "#d8dce3",
    "inj": "#8fb4f0",
    "outline": "#c5c9d1",
    "text": "#e8eaed",
    "muted": "#9aa3b0",
    "card": "#252a33",
    "overlay": "#f4f6fa",
    "span": "#7aa4f0",
    "ruler": "#c5c9d1",
}
LIGHT_VIZ = {
    "bg": "#f4f5f7",
    "border": "#c5c9d1",
    "tank": "#3d6fd4",
    "tank_empty": "#d5dce8",
    "grain": "#c47a3c",
    "port": "#f7f8fa",
    "plate": "#ffffff",
    "inj": "#8aa8de",
    "outline": "#111111",
    "text": "#1b1d21",
    "muted": "#5c6370",
    "card": "#ffffff",
    "overlay": "#1b1d21",
    "span": "#2f5fbf",
    "ruler": "#5c6370",
}


@dataclass
class MotorView:
    tnk_L: float
    tnk_D: float
    grn_L: float
    grn_OD: float
    grn_ID: float
    inj_D: float
    inj_N: int
    vnt_state: str
    vnt_D: float
    noz_thrt: float
    noz_exit: float
    fill_frac: float
    name: str = ""
    tank_lines: tuple[str, ...] = ()
    inj_lines: tuple[str, ...] = ()
    grain_lines: tuple[str, ...] = ()
    noz_lines: tuple[str, ...] = ()
    vent_lines: tuple[str, ...] = ()
    overlay_tank: tuple[str, ...] = ()
    overlay_grain: tuple[str, ...] = ()
    display_units: DisplayUnits = field(default_factory=DisplayUnits)
    time_s: float | None = None
    tnk_start: float = 0.0
    cmbr_start: float | None = None
    cmbr_L: float = 0.0
    tnk_dry_kg: float = 0.0
    cmbr_dry_kg: float = 0.0


@dataclass
class _Geom:
    tnk_L: float
    tnk_R: float
    grn_L: float
    grn_R: float
    port_R: float
    inj_D: float
    inj_N: int
    vnt_on: bool
    vnt_D: float
    r_th: float
    r_ex: float
    r_in: float
    L_conv: float
    L_div: float
    x_tnk0: float = 0.0
    x_tnk1: float = 0.0
    x_cmbr0: float = 0.0
    x_cmbr1: float = 0.0
    x_plate0: float = 0.0
    x_plate1: float = 0.0
    x_inj0: float = 0.0
    x_inj1: float = 0.0
    x_grn0: float = 0.0
    x_grn1: float = 0.0
    x_case: float = 0.0
    x_th: float = 0.0
    x_noz: float = 0.0
    x_end: float = 0.0
    x_min: float = 0.0
    x_span: float = 1.0
    y_max: float = 0.0


def _vent_visible(state: str, diameter: float) -> bool:
    if str(state).lower() not in {"none", "0", ""}:
        return True
    return float(diameter) > 1e-6


def _geom(m: MotorView) -> _Geom:
    tnk_L = max(float(m.tnk_L), 1e-4)
    tnk_R = max(float(m.tnk_D), 1e-4) * 0.5
    grn_L = max(float(m.grn_L), 1e-4)
    grn_R = max(float(m.grn_OD), 1e-4) * 0.5
    port_R = max(min(float(m.grn_ID) * 0.5, grn_R * 0.98), 0.0)
    r_th = max(float(m.noz_thrt) * 0.5, 1e-6)
    r_ex = max(float(m.noz_exit) * 0.5, r_th)
    r_in = max(grn_R, r_th)
    lay = motor_layout(
        tnk_start=float(m.tnk_start),
        tnk_L=tnk_L,
        tnk_m=float(m.tnk_dry_kg),
        tnk_D=float(m.tnk_D),
        cmbr_start=m.cmbr_start,
        cmbr_L=float(m.cmbr_L),
        cmbr_m=float(m.cmbr_dry_kg),
        grn_L=grn_L,
        grn_OD=float(m.grn_OD),
        noz_thrt=float(m.noz_thrt),
        noz_exit=float(m.noz_exit),
    )
    g = _Geom(
        tnk_L=tnk_L,
        tnk_R=tnk_R,
        grn_L=grn_L,
        grn_R=grn_R,
        port_R=port_R,
        inj_D=max(float(m.inj_D), 1e-6),
        inj_N=max(int(m.inj_N), 1),
        vnt_on=_vent_visible(m.vnt_state, m.vnt_D),
        vnt_D=max(float(m.vnt_D), 0.0),
        r_th=r_th,
        r_ex=r_ex,
        r_in=r_in,
        L_conv=lay.L_conv,
        L_div=lay.L_div,
        x_tnk0=lay.tnk0,
        x_tnk1=lay.tnk1,
        x_cmbr0=lay.cmbr0,
        x_cmbr1=lay.x_noz,
        x_plate0=lay.plate0,
        x_plate1=lay.plate1,
        x_inj0=lay.inj0,
        x_inj1=lay.inj1,
        x_grn0=lay.grn0,
        x_grn1=lay.grn1,
        x_case=lay.x_case,
        x_th=lay.x_th,
        x_noz=lay.x_noz,
        x_end=lay.x_noz,
        x_min=lay.x_min,
        x_span=max(lay.overall_L, 1e-6),
    )
    g.y_max = max(tnk_R, grn_R, r_ex, r_in)
    return g


def _ruler_step(span: float) -> float:
    if span <= 0:
        return 1.0
    raw = span / 8.0
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for mult in (1.0, 2.0, 5.0, 10.0):
        if raw <= mult * mag:
            return mult * mag
    return 10.0 * mag


def _tick_label(tick: float, step: float, unit: str = "") -> str:
    if abs(tick - round(tick)) < 1e-6:
        text = f"{int(round(tick))}"
    else:
        decimals = max(0, -math.floor(math.log10(step)))
        text = f"{tick:.{decimals}f}"
    return f"{text} {unit}" if unit else text


def liquid_rect(tank: QRectF, fill: float) -> QRectF:
    """Remaining oxidizer sits on the right; ullage grows from the left."""
    fill = min(max(float(fill), 0.0), 1.0)
    w = tank.width() * fill
    return QRectF(tank.right() - w, tank.top(), w, tank.height())


def pack_hrects(
    centers: list[float],
    widths: list[float],
    x0: float,
    x1: float,
    gap: float = SPEC_GAP,
    min_width: float = SPEC_MIN_W,
) -> list[tuple[float, float]]:
    """Place boxes at preferred centers with no overlap and a small gap.

    If the band is too narrow, widths shrink and the boxes bunch together.
    """
    n = len(centers)
    if n == 0:
        return []
    if n != len(widths):
        raise ValueError("centers and widths must be the same length")

    avail = max(x1 - x0, 0.0)
    gaps = gap * max(n - 1, 0)
    ws = [max(float(w), 1.0) for w in widths]
    scale_space = avail - gaps
    if scale_space <= 0:
        ws = [max(avail / n, 1.0) for _ in ws]
    elif sum(ws) + gaps > avail:
        total = sum(ws) or 1.0
        ws = [w * scale_space / total for w in ws]
        floor = min(min_width, scale_space / n) if n else 1.0
        ws = [max(w, floor) for w in ws]
        extra = sum(ws) + gaps - avail
        if extra > 0 and sum(ws) > 0:
            ws = [w * scale_space / sum(ws) for w in ws]

    lefts = [c - 0.5 * w for c, w in zip(centers, ws)]
    lefts[0] = max(x0, min(lefts[0], x1 - ws[0]))
    for i in range(1, n):
        lefts[i] = max(lefts[i], lefts[i - 1] + ws[i - 1] + gap)

    if lefts[-1] + ws[-1] > x1 + 1e-9:
        lefts[-1] = x1 - ws[-1]
        for i in range(n - 2, -1, -1):
            lefts[i] = min(lefts[i], lefts[i + 1] - ws[i] - gap)

    if lefts[0] < x0 - 1e-9:
        lefts[0] = x0
        for i in range(1, n):
            lefts[i] = lefts[i - 1] + ws[i - 1] + gap

    return list(zip(lefts, ws))


def _c(colors: dict, key: str) -> QColor:
    return QColor(colors[key])


class MotorVizWidget(QWidget):
    """2D side-view motor that scales with tank, grain, injector, and nozzle sizes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors = dict(DARK_VIZ)
        self._framed = False
        self._model = MotorView(
            tnk_L=0.2,
            tnk_D=0.1,
            grn_L=0.4,
            grn_OD=0.08,
            grn_ID=0.04,
            inj_D=0.006,
            inj_N=3,
            vnt_state="None",
            vnt_D=0.002,
            noz_thrt=0.02,
            noz_exit=0.04,
            fill_frac=0.95,
        )
        self.setMinimumHeight(300)  # below this the spec blocks leave no room for the drawing
        self.setMaximumHeight(480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_framed(self, framed: bool) -> None:
        self._framed = framed
        self.update()

    def set_theme(self, name: str) -> None:
        self._colors = dict(LIGHT_VIZ if name == "light" else DARK_VIZ)
        self.update()

    def set_model(self, model: MotorView) -> None:
        self._model = model
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = self._colors
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.fillRect(rect, _c(colors, "bg"))
        if self._framed:
            p.setPen(QPen(_c(colors, "border"), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)

        m = self._model
        g = _geom(m)
        pad = 10.0
        title_font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        body_font = QFont("Segoe UI", 8)
        title_fm = QFontMetrics(title_font)
        body_fm = QFontMetrics(body_font)
        spec_items = [ln for ln in (m.vent_lines, m.tank_lines, m.inj_lines, m.grain_lines, m.noz_lines) if ln]
        max_lines = max((len(ln) for ln in spec_items), default=2)
        label_h = title_fm.height() + max(0, max_lines - 1) * body_fm.height() + 14.0
        inner = rect.adjusted(pad, pad, -pad, -pad)
        motor_h = inner.height() - label_h - SPAN_H - RULER_H - TITLE_H
        if motor_h < 36 or inner.width() < 40 or g.x_span <= 0:
            return
        head_px = VENT_HEAD_PX if g.vnt_on else 0.0
        body_h = max(motor_h - head_px, 24.0)
        sx = inner.width() / g.x_span
        sy = body_h / max(2.0 * g.y_max, 1e-6)
        scale = min(sx, sy)
        used_w = g.x_span * scale
        ox = inner.left() + 0.5 * (inner.width() - used_w)
        cy = inner.top() + label_h + head_px + 0.5 * body_h

        def X(x: float) -> float:
            return ox + (x - g.x_min) * scale

        def Y(y: float) -> float:
            return cy - y * scale

        def box(x0: float, x1: float, r: float) -> QRectF:
            return QRectF(X(x0), Y(r), max((x1 - x0) * scale, 1.0), 2.0 * r * scale)

        outline = QPen(_c(colors, "outline"), 1.2)
        fill = min(max(m.fill_frac, 0.0), 1.0)

        tank = box(g.x_tnk0, g.x_tnk1, g.tnk_R)
        p.setPen(outline)
        p.setBrush(_c(colors, "tank_empty"))
        p.drawRect(tank)
        if fill > 0:
            liq = liquid_rect(tank, fill)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_c(colors, "tank"))
            p.drawRect(liq)
            p.setPen(outline)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(tank)

        if g.vnt_on:
            vw = max(g.vnt_D * scale, VENT_MIN_W)
            room = max(tank.top() - (inner.top() + label_h) - 1.0, VENT_MIN_H)
            vh = min(max(0.36 * tank.height(), VENT_MIN_H), room)
            vx = X(g.x_tnk0 + 0.05 * g.tnk_L)
            vent = QRectF(vx - 0.5 * vw, tank.top() - vh, vw, vh)
            p.setBrush(_c(colors, "plate"))
            p.setPen(outline)
            p.drawRect(vent)

        if g.x_cmbr0 - g.x_tnk1 > 1e-6:
            feed_r = min(g.tnk_R, g.grn_R) * 0.18
            feed = box(g.x_tnk1, g.x_cmbr0, max(feed_r, 2.0 / scale / 2.0))
            p.setPen(outline)
            p.setBrush(_c(colors, "tank") if fill > 0 else _c(colors, "inj"))
            p.drawRect(feed)

        if g.x_case > g.x_plate1 + 1e-9:
            # Pre- and post-combustion chambers: open gas space, the full case bore.
            case = box(g.x_plate1, g.x_case, g.grn_R)
            p.setBrush(_c(colors, "port"))
            p.setPen(outline)
            p.drawRect(case)

        grain = box(g.x_grn0, g.x_grn1, g.grn_R)
        p.setBrush(_c(colors, "grain"))
        p.setPen(outline)
        p.drawRect(grain)
        if g.port_R > 0:
            port = box(g.x_grn0, g.x_grn1, g.port_R)
            p.setBrush(_c(colors, "port"))
            p.setPen(QPen(_c(colors, "outline"), 0.8))
            p.drawRect(port)

        plate_r = max(g.tnk_R, g.grn_R)
        plate = box(g.x_plate0, g.x_plate1, plate_r)
        p.setBrush(_c(colors, "plate"))
        p.setPen(outline)
        p.drawRect(plate)

        n_draw = min(g.inj_N, 12)
        hole_h = g.inj_D
        span = min(2.0 * plate_r * 0.72, max(n_draw, 1) * hole_h * 1.35)
        ys = [0.0] if n_draw == 1 else [span * (0.5 - i / (n_draw - 1)) for i in range(n_draw)]
        p.setBrush(_c(colors, "tank") if fill > 0 else _c(colors, "inj"))
        p.setPen(outline)
        for y in ys:
            hr = max(hole_h * 0.5, 1.4 / scale)
            inj = QRectF(X(g.x_inj0), Y(y + hr), (g.x_inj1 - g.x_inj0) * scale, 2.0 * hr * scale)
            p.drawRect(inj)

        poly = QPolygonF(
            [
                QPointF(X(g.x_case), Y(g.r_in)),
                QPointF(X(g.x_th), Y(g.r_th)),
                QPointF(X(g.x_noz), Y(g.r_ex)),
                QPointF(X(g.x_noz), Y(-g.r_ex)),
                QPointF(X(g.x_th), Y(-g.r_th)),
                QPointF(X(g.x_case), Y(-g.r_in)),
            ]
        )
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        p.setBrush(_c(colors, "plate"))
        p.setPen(QPen(_c(colors, "outline"), 1.3))
        p.drawPath(path)

        self._draw_state_overlay(p, colors, tank, m.overlay_tank)
        self._draw_state_overlay(p, colors, grain, m.overlay_grain)

        band = QRectF(inner.left(), inner.top(), inner.width(), label_h - 4)
        self._draw_specs(p, colors, band, g, X, m, title_font, body_font, title_fm, body_fm)

        span_top = inner.top() + label_h + motor_h
        self._draw_span_labels(p, colors, g, X, m, QRectF(inner.left(), span_top, inner.width(), SPAN_H))
        ruler = QRectF(inner.left(), span_top + SPAN_H, inner.width(), RULER_H)
        self._draw_ruler(p, colors, g, X, ruler, m.display_units)
        p.setPen(_c(colors, "text"))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        p.drawText(
            QRectF(inner.left(), ruler.bottom(), inner.width(), TITLE_H),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            m.name or "motor",
        )

    def _draw_span_labels(self, p: QPainter, colors: dict, g: _Geom, X, m: MotorView, band: QRectF) -> None:
        spans = (
            (g.x_tnk0, g.x_tnk1, "Oxidizer Tank", m.tnk_dry_kg),
            (g.x_cmbr0, g.x_cmbr1, "Thrust Chamber Assembly", m.cmbr_dry_kg),
        )
        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = QFontMetrics(font)
        for x0, x1, title, mass_kg in spans:
            left, right = X(x0), X(x1)
            if right - left < 12:
                continue
            y = band.top() + 2.0
            p.setPen(QPen(_c(colors, "span"), 2.0))
            p.drawLine(QPointF(left, y + fm.height() + 3.0), QPointF(right, y + fm.height() + 3.0))
            p.setPen(_c(colors, "span"))
            text = f"{title}  {m.display_units.text(mass_kg, 'mass', 3)}"
            p.drawText(
                QRectF(left, y, max(right - left, 8.0), fm.height() + 2.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                fm.elidedText(text, Qt.TextElideMode.ElideRight, max(int(right - left), 1)),
            )

    def _draw_ruler(self, p: QPainter, colors: dict, g: _Geom, X, band: QRectF, units: DisplayUnits) -> None:
        y = band.top() + 7.0
        x0, x1 = X(g.x_min), X(g.x_min + g.x_span)
        p.setPen(QPen(_c(colors, "ruler"), 1.0))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        scale = LENGTH[units.length]
        start = g.x_min / scale
        end = (g.x_min + g.x_span) / scale
        step = _ruler_step(end - start)
        label_font = QFont("Segoe UI", 7)
        p.setFont(label_font)
        fm = QFontMetrics(label_font)
        half = step / 2.0
        first = math.ceil((start - 1e-9) / half) * half
        majors: list[float] = []
        tick = first
        while tick <= end + 1e-6:
            xt = X(tick * scale)
            major = abs(tick / step - round(tick / step)) < 1e-6
            h = 6.0 if major else 3.0
            p.setPen(QPen(_c(colors, "ruler"), 1.0))
            p.drawLine(QPointF(xt, y - h), QPointF(xt, y + h))
            if major:
                majors.append(tick)
            tick += half
        p.setPen(_c(colors, "muted"))
        for i, tick in enumerate(majors):
            label = _tick_label(tick, step, units.length if i == len(majors) - 1 else "")
            tw = fm.horizontalAdvance(label)
            xt = X(tick * scale)
            p.drawText(QPointF(xt - 0.5 * tw, y + 6.0 + fm.ascent() + 1.0), label)

    def _draw_specs(
        self,
        p: QPainter,
        colors: dict,
        band: QRectF,
        g: _Geom,
        X,
        m: MotorView,
        title_font: QFont,
        body_font: QFont,
        title_fm: QFontMetrics,
        body_fm: QFontMetrics,
    ) -> None:
        items: list[tuple[float, tuple[str, ...]]] = []
        if m.vent_lines:
            items.append((X(g.x_tnk0 + 0.05 * g.tnk_L), m.vent_lines))
        if m.tank_lines:
            items.append((0.5 * (X(g.x_tnk0) + X(g.x_tnk1)), m.tank_lines))
        if m.inj_lines:
            items.append((0.5 * (X(g.x_inj0) + X(g.x_inj1)), m.inj_lines))
        if m.grain_lines:
            items.append((0.5 * (X(g.x_grn0) + X(g.x_grn1)), m.grain_lines))
        if m.noz_lines:
            items.append((0.5 * (X(g.x_case) + X(g.x_noz)), m.noz_lines))
        if not items:
            return

        def pref_w(lines: tuple[str, ...]) -> float:
            w = 0.0
            for i, line in enumerate(lines):
                fm = title_fm if i == 0 else body_fm
                w = max(w, fm.horizontalAdvance(line))
            return w + 12.0

        centers = [cx for cx, _ in items]
        widths = [pref_w(lines) for _, lines in items]
        placed = pack_hrects(centers, widths, band.left(), band.right(), gap=SPEC_GAP, min_width=SPEC_MIN_W)
        for (left, width), (_, lines) in zip(placed, items):
            r = QRectF(left, band.top(), width, band.height())
            self._paint_spec(p, colors, r, lines, title_font, body_font, title_fm, body_fm)

    @staticmethod
    def _draw_state_overlay(p: QPainter, colors: dict, rect: QRectF, lines: tuple[str, ...]) -> None:
        if not lines or rect.width() < 28 or rect.height() < 16:
            return
        pt = 9 if rect.height() >= 28 else 7
        font = QFont("Segoe UI", pt, QFont.Weight.DemiBold)
        p.setFont(font)
        text = "\n".join(lines)
        flags = int(
            Qt.AlignmentFlag.AlignHCenter
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextWordWrap
        )
        halo = QColor(12, 14, 18, 170)
        p.setPen(halo)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p.drawText(rect.adjusted(dx, dy, dx, dy), flags, text)
        p.setPen(_c(colors, "overlay"))
        p.drawText(rect, flags, text)

    @staticmethod
    def _paint_spec(
        p: QPainter,
        colors: dict,
        r: QRectF,
        lines: tuple[str, ...],
        title_font: QFont,
        body_font: QFont,
        title_fm: QFontMetrics,
        body_fm: QFontMetrics,
    ) -> None:
        p.setBrush(_c(colors, "card"))
        p.setPen(QPen(_c(colors, "border"), 1.0))
        p.drawRect(r)
        inner = r.adjusted(4, 5, -4, -5)
        y = inner.top()
        for i, line in enumerate(lines):
            fm = title_fm if i == 0 else body_fm
            p.setFont(title_font if i == 0 else body_font)
            p.setPen(_c(colors, "text" if i == 0 else "muted"))
            row = QRectF(inner.left(), y, inner.width(), fm.height())
            text = fm.elidedText(line, Qt.TextElideMode.ElideRight, max(int(inner.width()), 1))
            p.drawText(row, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter), text)
            y += fm.height()


class MotorPanel(QFrame):
    """Schematic with spec blocks above components and the motor name below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("motorPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.viz = MotorVizWidget()
        lay.addWidget(self.viz)

    def set_theme(self, name: str) -> None:
        self.viz.set_theme(name)

    def set_model(self, model: MotorView) -> None:
        self.viz.set_model(model)
