"""Small to-scale sketches for the sizing cards: injector face, grain cross-section and nozzle profile."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from hrap.gui.viz import DARK_VIZ, LIGHT_VIZ

SIZE = 104  # px, square drawing area of each sketch


class Sketch(QWidget):
    def __init__(self, width: int = SIZE):
        super().__init__()
        self.setFixedSize(width, SIZE)
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
        self.draw(p, QRectF(self.rect()).adjusted(2, 2, -2, -2), **self._data)

    def draw(self, p: QPainter, r: QRectF, **data):
        raise NotImplementedError


def _circle(p: QPainter, c: QPointF, d: float):
    p.drawEllipse(c, d / 2, d / 2)


class InjectorSketch(Sketch):
    """Injector face inside the chamber bore, holes to scale (at least 3 px so they stay visible)."""

    def draw(self, p: QPainter, r: QRectF, bore: float, hole: float, holes: int):
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


class GrainSketch(Sketch):
    """Grain cross-section: the starting port, and the port when the liquid runs out as a dashed circle."""

    def draw(self, p: QPainter, r: QRectF, od: float, port: float, port_end: float):
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


class NozzleSketch(Sketch):
    """Nozzle side profile from the chamber bore through the throat to the exit: 45° in, 15° out."""

    def __init__(self):
        super().__init__(int(SIZE * 1.4))

    def draw(self, p: QPainter, r: QRectF, bore: float, throat: float, exit: float):
        rc, rt, re = bore / 2, throat / 2, max(exit, throat) / 2
        l_in = rc - rt  # tan 45° = 1
        l_out = (re - rt) / math.tan(math.radians(15))
        lip = 0.15 * bore  # a short stretch of chamber wall before the converging section
        outer = max(rc, re)
        px = min(r.width() / (lip + l_in + l_out), r.height() / (2.3 * outer))
        x0 = r.center().x() - 0.5 * (lip + l_in + l_out) * px
        y0 = r.center().y()
        xs = [x0, x0 + lip * px, x0 + (lip + l_in) * px, x0 + (lip + l_in + l_out) * px]
        radii = [rc, rc, rt, re]
        wall = 0.15 * outer * px
        for sign in (-1, 1):
            path = QPainterPath(QPointF(xs[0], y0 + sign * (outer * px + wall)))
            for x, rad in zip(xs, radii):
                path.lineTo(x, y0 + sign * rad * px)
            path.lineTo(xs[-1], y0 + sign * (outer * px + wall))
            path.closeSubpath()
            p.setPen(QPen(self.color("outline"), 1.0))
            p.setBrush(self.color("plate"))
            p.drawPath(path)
        p.setPen(QPen(self.color("muted"), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(xs[0], y0), QPointF(xs[-1], y0))
