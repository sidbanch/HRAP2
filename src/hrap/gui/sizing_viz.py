"""Small to-scale sketches for the sizing cards: injector face, grain cross-section and nozzle profile."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from hrap.gui.viz import DARK_VIZ, LIGHT_VIZ
from hrap.units import from_si, to_si

SIZE = 104  # px, square drawing area of each sketch
CAPTION_H = 30
SCALE_H = 16  # scale bar under the caption, so sketches drawn at different scales can be compared


class Sketch(QWidget):
    def __init__(self, width: int = int(SIZE * 1.3)):
        super().__init__()
        self.setFixedSize(width, SIZE + CAPTION_H + SCALE_H)
        self._colors = dict(DARK_VIZ)
        self._data: dict | None = None

    def set_theme(self, name: str):
        self._colors = dict(LIGHT_VIZ if name == "light" else DARK_VIZ)
        self.update()

    def show_data(self, data: dict | None):
        self._data = data
        self.update()

    def color(self, key: str) -> QColor:
        return QColor(self._colors[key])

    def paintEvent(self, _event):
        if not self._data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        data = dict(self._data)
        caption, unit = data.pop("caption"), data.pop("unit")
        px = self.draw(p, QRectF(2, 2, self.width() - 4, SIZE - 4), **data)
        p.setPen(self.color("muted"))
        p.drawText(QRectF(0, SIZE, self.width(), CAPTION_H),
                   int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap), caption)
        self._draw_scale(p, px, unit)

    def _draw_scale(self, p: QPainter, px: float, unit: str):
        """A bar of a round length (1, 2 or 5 × a power of ten) about 30 px long."""
        target = from_si(30.0 / px, unit, "length")
        step = 10.0 ** math.floor(math.log10(target))
        length = max(m * step for m in (1, 2, 5) if m * step <= target)
        bar = to_si(length, unit, "length") * px
        label = f"{length:g} {unit}"
        y = SIZE + CAPTION_H + 0.5 * SCALE_H
        text_w = p.fontMetrics().horizontalAdvance(label)
        x0 = 0.5 * (self.width() - bar - 6 - text_w)
        p.setPen(QPen(self.color("muted"), 1.2))
        p.drawLine(QPointF(x0, y), QPointF(x0 + bar, y))
        for x in (x0, x0 + bar):
            p.drawLine(QPointF(x, y - 3), QPointF(x, y + 3))
        p.drawText(QRectF(x0 + bar + 6, y - 8, text_w + 2, 16), int(Qt.AlignmentFlag.AlignVCenter), label)

    def draw(self, p: QPainter, r: QRectF, **data) -> float:
        """Draw to scale and return the scale in px per metre."""
        raise NotImplementedError


def _circle(p: QPainter, c: QPointF, d: float):
    p.drawEllipse(c, d / 2, d / 2)


class InjectorSketch(Sketch):
    """Injector face inside the chamber bore, holes to scale (at least 4 px so they stay visible),
    or a swirler looking down its axis."""

    def draw(self, p: QPainter, r: QRectF, swirler: dict | None = None, **holes) -> float:
        return self._draw_swirler(p, r, **swirler) if swirler else self._draw_holes(p, r, **holes)

    def _draw_swirler(self, p: QPainter, r: QRectF, exit: float, ports: int, port: float, offset: float, fill: float) -> float:
        """Looking down the swirler's axis: drilled tangential ports through the body wall into the swirl
        chamber, the flow's spin, and the exit orifice beyond it with its air core. All to scale."""
        c = r.center()
        chamber = offset + 0.5 * port  # the ports run along the chamber wall
        body = chamber + max(2.5 * port, 0.4 * chamber)
        px = 0.5 * min(r.width(), r.height()) / body
        outline, liquid = self.color("outline"), self.color("tank")
        rim = QPainterPath()
        rim.addEllipse(c, body * px, body * px)
        p.setPen(QPen(outline, 1.0))
        p.setBrush(self.color("plate"))
        p.drawPath(rim)

        # Each port is a drilled channel tangent to the chamber wall, cut off at the body's rim.
        p.save()
        p.setClipPath(rim)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(liquid)
        for i in range(ports):
            a = 2 * math.pi * i / ports
            radial, along = QPointF(math.cos(a), math.sin(a)), QPointF(-math.sin(a), math.cos(a))
            inner, outer = (offset - 0.5 * port) * radial, (offset + 0.5 * port) * radial
            reach = 2 * body * along
            p.drawPolygon(QPolygonF([c + q * px for q in (inner, outer, outer + reach, inner + reach)]))
        p.restore()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(liquid)
        _circle(p, c, 2 * chamber * px)

        # The swirl: flow enters along each channel toward the chamber, so it turns the same way.
        arc_r = 0.5 * (0.5 * exit + chamber) * px
        spin = QPen(self.color("overlay"), 1.2)
        spin.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(spin)
        p.setBrush(Qt.BrushStyle.NoBrush)
        box = QRectF(c.x() - arc_r, c.y() - arc_r, 2 * arc_r, 2 * arc_r)
        start, sweep = 20.0, 250.0  # degrees, counterclockwise like the entering flow
        p.drawArc(box, int(start * 16), int(sweep * 16))
        end = math.radians(start + sweep)
        tip = c + QPointF(arc_r * math.cos(end), -arc_r * math.sin(end))
        heading = QPointF(-math.sin(end), -math.cos(end))  # counterclockwise tangent on screen
        side = QPointF(-heading.y(), heading.x())
        head = 4.0
        p.setBrush(self.color("overlay"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([tip + heading * head, tip - heading * head + side * head, tip - heading * head - side * head]))

        # The exit orifice sits beyond the chamber: dashed, with the air core the swirl leaves in it.
        p.setPen(QPen(self.color("overlay"), 1.0, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        _circle(p, c, exit * px)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color("port"))
        _circle(p, c, math.sqrt(1.0 - fill) * exit * px)
        return px

    def _draw_holes(self, p: QPainter, r: QRectF, bore: float, hole: float, holes: int) -> float:
        c, px = r.center(), min(r.width(), r.height()) / bore
        p.setPen(QPen(self.color("outline"), 1.0))
        p.setBrush(self.color("plate"))
        _circle(p, c, bore * px)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color("tank"))  # oxidizer blue, as in the motor diagram
        ring = 0.0 if holes == 1 else 0.25 * bore * px
        for i in range(holes):
            a = 2 * math.pi * i / holes - math.pi / 2
            _circle(p, c + QPointF(ring * math.cos(a), ring * math.sin(a)), max(hole * px, 4.0))
        return px


class GrainSketch(Sketch):
    """Grain cross-section: the starting port, and the port when the liquid runs out as a dashed circle."""

    def draw(self, p: QPainter, r: QRectF, od: float, port: float, port_end: float) -> float:
        c, px = r.center(), min(r.width(), r.height()) / od
        p.setPen(QPen(self.color("outline"), 1.0))
        p.setBrush(self.color("grain"))
        _circle(p, c, od * px)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color("port"))
        _circle(p, c, port * px)
        if math.isfinite(port_end):
            pen = QPen(self.color("text"), 1.2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            _circle(p, c, min(port_end, od) * px)
        return px


class NozzleSketch(Sketch):
    """Nozzle side profile from the chamber bore through the throat to the exit: 45° in, 15° out."""

    def __init__(self):
        super().__init__(int(SIZE * 1.4))

    def draw(self, p: QPainter, r: QRectF, bore: float, throat: float, exit: float) -> float:
        rc, rt, re = bore / 2, throat / 2, max(exit, throat) / 2
        l_in = rc - rt  # tan 45° = 1
        l_out = (re - rt) / math.tan(math.radians(15))
        lip = 0.15 * bore  # a short stretch of chamber wall before the converging section
        px = min(r.width() / (lip + l_in + l_out), r.height() / (2 * max(rc, re)))
        x0 = r.center().x() - 0.5 * (lip + l_in + l_out) * px
        y0 = r.center().y()
        xs = [x0, x0 + lip * px, x0 + (lip + l_in) * px, x0 + (lip + l_in + l_out) * px]
        radii = [rc, rc, rt, re]
        top = [QPointF(x, y0 - rad * px) for x, rad in zip(xs, radii)]
        bottom = [QPointF(x, y0 + rad * px) for x, rad in zip(xs, radii)]
        gas = QPainterPath(top[0])
        for pt in top[1:] + bottom[::-1]:
            gas.lineTo(pt)
        gas.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.color("tank_empty"))
        p.drawPath(gas)
        p.setPen(QPen(self.color("outline"), 1.6))
        for wall in (top, bottom):
            p.drawPolyline(wall)
        p.setPen(QPen(self.color("muted"), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(xs[0], y0), QPointF(xs[-1], y0))
        return px
