"""Simulation tab: what to run on the left; on the right its plots, one per unit on a shared time axis, with a list
of their lines to read values from and turn them on and off, and under them the motor drawing, grain and summary."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hrap.gui.grain_view import GrainScene, RunGrainView
from hrap.gui.theme import colors_for
from hrap.gui.viz import MotorPanel
from hrap.gui.widgets import (
    CARD_GAP,
    PAGE_MARGIN,
    ElidedLabel,
    FieldGrid,
    HintList,
    PlainComboBox,
    PlainDoubleSpinBox,
    UnitRow,
    card_frame,
    line_up,
    page_layout,
    sentence_case,
)
from hrap.units import PRESSURE_ITEMS, DisplayUnits, to_si


@dataclass(frozen=True)
class Line:
    key: str           # the run's output field
    name: str
    quantity: str
    color: str         # theme colour
    on: bool = True    # until it's turned off
    stl: bool = False  # only an STL grain has it


# Each plot tab's lines. Lines in the same unit share a plot; the tab stacks one plot per unit that has a line on.
TABS: dict[str, tuple[Line, ...]] = {
    "Thrust": (Line("F_thr", "Thrust", "force", "green"),),
    "Pressure": (Line("P_tnk", "Tank", "pressure", "oxidizer"), Line("P_cmbr", "Chamber", "pressure", "purple"),
                 Line("inj_dP", "Injector ΔP", "pressure", "teal"),
                 Line("T_tnk", "Tank temp", "temperature", "oxidizer", on=False)),
    "Flow": (Line("mdot_o", "Oxidizer", "mass_flow", "oxidizer"), Line("mdot_f", "Fuel", "mass_flow", "fuel"),
             Line("mdot_n", "Nozzle", "mass_flow", "grey", on=False), Line("OF", "O/F", "ratio", "teal"),
             Line("inj_CdA", "Injector CdA", "area", "purple")),
    "Mass": (Line("m_o", "Oxidizer", "mass", "oxidizer"), Line("m_f", "Fuel", "mass", "fuel"),
             Line("m_t", "Total", "mass", "grey", on=False), Line("cg", "CG", "length", "purple", on=False)),
    "Grain": (Line("grn_ID", "Port", "length", "purple"),
              Line("grn_web", "Burned distance", "length", "fuel", on=False, stl=True),
              Line("burn_area", "Burning area", "area", "fuel", stl=True),
              Line("case_area", "Exposed case wall", "area", "limit", stl=True),
              Line("rdot", "Regression rate", "speed", "fuel", on=False)),
}
MEASURED = {"Thrust": (("F_thr", "Measured", "force"),),  # a measured curve's series on each tab
            "Pressure": (("P_tnk", "Measured tank", "pressure"), ("P_cmbr", "Measured chamber", "pressure"))}
AXES = {"force": "Thrust", "pressure": "Pressure", "temperature": "Temperature", "mass_flow": "Mass flow", "ratio": "O/F",
        "mass": "Mass", "length": "Length", "area": "Area", "speed": "Regression rate"}
DASHES = (Qt.PenStyle.DashLine, Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine, Qt.PenStyle.DashDotDotLine)
LEGEND_W = 220
STL_KEYS = {line.key for lines in TABS.values() for line in lines if line.stl}


def curve(o, settings, key: str) -> np.ndarray | None:
    """A run's curve for a line; None if it has none, as a round grain has no STL curves.
    An STL grain's burning area and exposed case wall follow from its burned distance."""
    g = getattr(settings, "grain_geometry", None)
    if key in STL_KEYS and g is None:
        return None
    if key in ("burn_area", "case_area"):
        return np.interp(o.grn_web, g.web, getattr(g, key))
    return np.asarray(getattr(o, key), dtype=float)


def case_contact(o, settings) -> float | None:
    """When an STL grain's burn first reached the case wall, if it did."""
    g = getattr(settings, "grain_geometry", None)
    if g is None or not math.isfinite(g.case_web):
        return None
    reached = np.flatnonzero(o.grn_web >= g.case_web)
    return float(o.t[reached[0]]) if reached.size else None


def summary_text(info: dict, u: DisplayUnits) -> str:
    """A run's headline numbers, in two columns."""
    rows = [
        ("Propellant", info["propellant"]),
        ("Tank volume", u.text(info["tnk_V_cc"] * 1e-6, "volume", 6)),
        ("Burn time", f"{info['burn_time']:.3f} s"),
        ("Peak thrust", u.text(info["peak_thrust"], "force", 6)),
        ("Average thrust", u.text(info["avg_thrust"], "force", 6)),
        ("Total impulse", u.text(info["total_impulse"], "impulse", 7)),
        ("Peak chamber pressure", f"{u.text(info['peak_pressure_bar'] * 1e5, 'pressure', 6)} (absolute)"),
        ("Average chamber pressure", f"{u.text(info['avg_pressure_bar'] * 1e5, 'pressure', 6)} (absolute)"),
        ("Average injector ΔP", u.text(info["avg_inj_dP"], "pressure", 6)),
        (sentence_case(info["port_label"]), u.text(info["port_cm"] * .01, "length", 6)),
        ("Fuel consumed", u.text(info["fuel_consumed"], "mass", 6)),
        ("Oxidizer consumed", u.text(info["ox_consumed"], "mass", 6)),
        ("Average O/F", f"{info['avg_OF']:.3f}"),
        ("Characteristic velocity", u.text(info["cstar"], "speed", 6)),
        ("Specific impulse", f"{info['isp']:.1f} s"),
        ("Motor classification", f"{info['impulse_percent']:.0f}% {info['impulse_class']}{info['avg_thrust']:.0f}"),
        ("End of the run", sentence_case(info["end_cond"])),
    ]
    if (contact := info.get("case_contact_time")) is not None:
        rows.append(("Case wall first exposed", f"{contact:.3f} s, {info['case_contact_fuel_percent']:.1f}% fuel left"))
    if info.get("case_area_end") is not None:
        rows.append(("Exposed case wall at burnout", u.text(info["case_area_end"], "area", 6)))
    width = max(len(label) for label, _ in rows) + 3
    lines = [info["name"], *(f"{label:<{width}}{value}" for label, value in rows)]
    return "\n".join(lines + ([info["grain_warning"]] if info.get("grain_warning") else []))


def line_pixmap(color: str, style: Qt.PenStyle = Qt.PenStyle.SolidLine, width: int = 16) -> QPixmap:
    """A short line in a colour and style, as the plots draw it."""
    pm = QPixmap(QSize(2 * width, 20))
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 2.0, style)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setPen(pen)
    p.drawLine(QPointF(0, 5), QPointF(width, 5))
    p.end()
    return pm


def dash_icon(color: str, style: Qt.PenStyle | None) -> QIcon:
    """The Runs list's key to how a run is drawn; blank for a run that isn't."""
    if style is None:
        pm = QPixmap(QSize(40, 20))
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.GlobalColor.transparent)
        return QIcon(pm)
    return QIcon(line_pixmap(color, style, 20))


@dataclass
class Shown:
    """What the plots show: the run on screen, ticked runs drawn dashed under it, and a measured curve."""
    output: object = None
    settings: object = None
    overlays: tuple = ()  # (output, settings) of each ticked run
    measured: object = None  # MeasuredCurve
    cfg: dict | None = None  # for the chamber limit
    spi: bool = False  # the 300 psi ΔP line only means something for the SPI injector model


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()  # at once; it's deleted once the app is idle
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


class LegendRow(QWidget):
    """One line in the legend: its colour and style, its name, and its value at the time hovered. Click to turn
    it on or off."""
    clicked = Signal()

    def __init__(self, name: str, pixmap: QPixmap, value_w: int, tip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        swatch = QLabel()
        swatch.setPixmap(pixmap)
        swatch.setFixedSize(16, 10)
        self.name = ElidedLabel(name)
        self.value = QLabel("")
        self.value.setObjectName("cardLabel")
        self.value.setFixedWidth(value_w)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(swatch)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.value)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tip or "Click to show or hide it.")
        self._dim = QGraphicsOpacityEffect(self)
        self._dim.setOpacity(0.38)
        self.setGraphicsEffect(self._dim)

    def set_on(self, on: bool):
        self._dim.setEnabled(not on)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("legendDivider")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


class PlotTab(QWidget):
    """One plot tab: its plots stacked on a shared time axis, and beside them the legend of its lines."""

    def __init__(self, panel: PlotPanel, name: str):
        super().__init__()
        self.panel, self.name = panel, name
        self.graph = pg.GraphicsLayoutWidget()
        self.graph.ci.setSpacing(6)
        self.graph.ci.setContentsMargins(0, 4, 8, 0)
        self.graph.viewport().installEventFilter(self)
        self.graph.scene().sigMouseMoved.connect(self._mouse_moved)
        self.empty = QLabel("")
        self.empty.setObjectName("cardLabel")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.graph)
        self.legend = QWidget()
        self.legend.setFixedWidth(LEGEND_W)
        self.legend_l = QVBoxLayout(self.legend)
        self.legend_l.setContentsMargins(12, 10, 0, 0)
        self.legend_l.setSpacing(4)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.stack, 1)
        row.addWidget(self.legend)
        self.plots: dict[str, pg.PlotItem] = {}
        self.hover_lines: list[pg.InfiniteLine] = []
        self.rows: dict[str, LegendRow] = {}  # each line's legend row, by id
        self.header = QLabel("")
        self.plotted = False  # there's a run or a measured curve to plot

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Leave:
            self.panel.leave()
        return super().eventFilter(obj, event)

    def _mouse_moved(self, pos):
        for plot in self.plots.values():
            vb = plot.vb
            if vb.sceneBoundingRect().contains(pos):
                self.panel.hover(vb.mapSceneToView(pos).x())
                return
        self.panel.leave()

    def lines(self, shown: Shown) -> list[Line]:
        stl = any(getattr(s, "grain_geometry", None) is not None
                  for s in (shown.settings, *(s for _o, s in shown.overlays)))
        return [line for line in TABS[self.name] if stl or not line.stl]

    def build(self, shown: Shown, units: DisplayUnits, colors: dict):
        """Draw the plots and the legend afresh."""
        panel = self.panel
        lines = self.lines(shown)
        stl = any(line.stl for line in lines)
        runs = ([(shown.output, shown.settings)] if shown.output is not None else []) + list(shown.overlays)
        measured = [(key, name, q) for key, name, q in MEASURED.get(self.name, ())
                    if shown.measured is not None and shown.measured.series(key) is not None]

        # The legend: lines, then reference lines and a measured curve, each after a divider.
        _clear_layout(self.legend_l)
        self.rows.clear()
        self.header = QLabel("")
        self.header.setObjectName("cardLabel")
        self.legend_l.addWidget(self.header)
        self.legend_l.addSpacing(4)
        fm = self.fontMetrics()
        value_w = max(fm.horizontalAdvance(f"-0.0000 {units.unit(q) if q != 'ratio' else ''}") + 2
                      for q in {line.quantity for line in lines} | {q for _k, _n, q in measured})
        last_q = None
        for line in lines:
            if last_q is not None and line.quantity != last_q:
                self.legend_l.addSpacing(6)
            last_q = line.quantity
            name = "Same-area port" if line.key == "grn_ID" and stl else line.name
            self._add_row(line.key, name, line_pixmap(colors[line.color]), value_w, line.on)
        if self.name == "Pressure":
            self.legend_l.addSpacing(4)
            self.legend_l.addWidget(_divider())
            self._add_row("limit", "Chamber limit", line_pixmap(colors["limit"], Qt.PenStyle.DashLine), value_w, True,
                          "The chamber's design pressure, from Targets on the Motor tab. Click to show or hide it.")
            if shown.spi:
                self._add_row("dp300", "300 psi ΔP", line_pixmap(colors["warn"], Qt.PenStyle.DashLine), value_w, False,
                              "Above about 300 psi of injector ΔP, real nitrous boils in the hole and flows less than\n"
                              "SPI predicts. Click to show or hide it.")
        if self.name in MEASURED:
            self.legend_l.addSpacing(4)
            self.legend_l.addWidget(_divider())
            if shown.measured is not None:
                for key, name, _q in measured:
                    self._add_row(f"measured:{key}", name, line_pixmap(self._measured_color(key, colors)), value_w,
                                  True, f"{shown.measured.path}\nClick to show or hide it.")
                source = QHBoxLayout()
                label = ElidedLabel(shown.measured.label)
                label.setObjectName("cardLabel")
                label.setToolTip(shown.measured.path)
                clear = QPushButton("Clear")
                clear.setObjectName("flat")
                clear.setToolTip("Stop showing the measured curve.")
                clear.clicked.connect(panel.clear_measured)
                source.addWidget(label, 1)
                source.addWidget(clear)
                self.legend_l.addLayout(source)
            else:
                load = QPushButton("Overlay measured CSV…")
                load.setObjectName("flat")
                load.setToolTip("Plot a measured thrust curve over the simulation. Also draws tank pressure if the file\n"
                                "has it (Weight_Lbs / F_thr, Tplus_ms / t, PT1_PSI / P_tnk).")
                load.clicked.connect(panel.load_measured)
                self.legend_l.addWidget(load, 0, Qt.AlignmentFlag.AlignLeft)
        self.legend_l.addStretch(1)

        # The plots: one per unit with a line on.
        self.graph.clear()
        self.plots.clear()
        self.hover_lines.clear()
        quantities = []
        for line in lines:
            if panel.is_on(line.key, line.on) and line.quantity not in quantities:
                quantities.append(line.quantity)
        for key, _name, q in measured:
            if panel.is_on(f"measured:{key}", True) and q not in quantities:
                quantities.append(q)
        self.plotted = bool(runs) or shown.measured is not None
        if not self.plotted:
            self.empty.setText("Press Run to simulate this motor.")
            self.stack.setCurrentWidget(self.empty)
            return
        if not quantities:
            self.empty.setText("Click a line in the list to plot it.")
            self.stack.setCurrentWidget(self.empty)
            return
        self.stack.setCurrentWidget(self.graph)
        contact = case_contact(shown.output, shown.settings) if shown.output is not None else None
        first = None
        for r, q in enumerate(quantities):
            plot = self.graph.addPlot(row=r, col=0)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", AXES[q], units=None if q == "ratio" else units.unit(q), color=colors["plot_fg"])
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("left").setWidth(64)
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(colors["plot_fg"])
                plot.getAxis(side).setTextPen(colors["plot_fg"])
            if r == len(quantities) - 1:
                plot.setLabel("bottom", "Time", units="s", color=colors["plot_fg"])
            else:
                plot.getAxis("bottom").setStyle(showValues=False)
                plot.getAxis("bottom").setHeight(4)

            def scaled(values, q=q):
                return values if q == "ratio" else units.value(values, q)
            for line in lines:
                if line.quantity != q or not panel.is_on(line.key, line.on):
                    continue
                color = colors[line.color]
                if shown.output is not None and (values := curve(shown.output, shown.settings, line.key)) is not None:
                    plot.plot(shown.output.t, scaled(values), pen=pg.mkPen(color, width=2))
                for j, (o, s) in enumerate(shown.overlays):
                    if (values := curve(o, s, line.key)) is not None:
                        plot.plot(o.t, scaled(values), pen=pg.mkPen(color, width=1.5, style=DASHES[j % len(DASHES)]))
            for key, _name, mq in measured:
                if mq == q and panel.is_on(f"measured:{key}", True):
                    plot.plot(shown.measured.t, scaled(np.asarray(shown.measured.series(key), dtype=float)),
                              pen=pg.mkPen(self._measured_color(key, colors), width=2.5))
            if q == "pressure":
                for ref, color in (("limit", colors["limit"]), ("dp300", colors["warn"])):
                    value = panel.reference(ref, shown)
                    if value is not None and panel.is_on(ref, ref == "limit"):
                        plot.addItem(pg.InfiniteLine(pos=units.value(value, "pressure"), angle=0, movable=False,
                                                     pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine)))
            if contact is not None:
                plot.addItem(pg.InfiniteLine(
                    pos=contact, angle=90, movable=False,
                    pen=pg.mkPen(colors["limit"], width=1.5, style=Qt.PenStyle.DashLine),
                    label=f"Case wall exposed, {contact:.2f} s" if r == 0 else None,
                    labelOpts={"position": 0.96, "color": colors["limit"], "anchors": [(-0.03, 0), (-0.03, 0)]}),
                    ignoreBounds=True)
            hover = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(colors["accent"], width=1, style=Qt.PenStyle.DashLine))
            hover.setVisible(False)
            plot.addItem(hover, ignoreBounds=True)
            self.hover_lines.append(hover)
            plot.vb.setXRange(*panel.time_range, padding=0)
            if first is None:
                first = plot
                plot.vb.sigXRangeChanged.connect(panel.time_range_changed)
            else:
                plot.setXLink(first)
            self.graph.ci.layout.setRowStretchFactor(r, 2 if r == 0 and len(quantities) > 1 else 1)
            self.plots[q] = plot

    @staticmethod
    def _measured_color(key: str, colors: dict) -> str:
        return colors["grey"] if key == "P_cmbr" else colors["plot_fg"]

    def _add_row(self, key: str, name: str, pixmap: QPixmap, value_w: int, default_on: bool, tip: str = ""):
        row = LegendRow(name, pixmap, value_w, tip)
        row.set_on(self.panel.is_on(key, default_on))
        row.clicked.connect(lambda: self.panel.toggle(key, default_on))
        self.legend_l.addWidget(row)
        self.rows[key] = row

    def show_values(self, values: dict[str, str], time: float | None):
        """The lines' values at a time, or none while nothing is hovered."""
        self.header.setText("" if not self.plotted else "Hover a plot for values" if time is None else f"At {time:.2f} s")
        for key, row in self.rows.items():
            row.value.setText(values.get(key, ""))
        for line in self.hover_lines:
            line.setVisible(time is not None)
            if time is not None:
                line.setValue(time)


class PlotPanel(QWidget):
    """The plot tabs: Thrust, Pressure, Flow, Mass and Grain."""
    hovered = Signal(object)  # the run's time index hovered, or None once the mouse leaves
    load_measured = Signal()
    clear_measured = Signal()

    def __init__(self, prefs: QSettings, get_units: Callable[[], DisplayUnits]):
        super().__init__()
        self._prefs, self._get_units = prefs, get_units
        self._theme = "dark"
        self.shown = Shown()
        self.time_range = (0.0, 1.0)
        self._syncing = False
        try:
            self._on: dict[str, bool] = dict(json.loads(str(self._prefs.value("simulationLines") or "{}")))
        except (ValueError, TypeError):
            self._on = {}
        self.tabs = QTabWidget()
        self.tabs.setObjectName("subTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.pages = {name: PlotTab(self, name) for name in TABS}
        for name, page in self.pages.items():
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(lambda _: self.leave())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.redraw()

    @property
    def plots(self) -> dict[tuple[str, str], pg.PlotItem]:
        """Every plot on screen, by tab and unit."""
        return {(name, q): plot for name, page in self.pages.items() for q, plot in page.plots.items()}

    def is_on(self, key: str, default: bool) -> bool:
        return self._on.get(key, default)

    def toggle(self, key: str, default: bool):
        """Turn a line on or off, and remember it for next time."""
        self._on[key] = not self.is_on(key, default)
        self._prefs.setValue("simulationLines", json.dumps(self._on))
        self.redraw()

    def reference(self, ref: str, shown: Shown) -> float | None:
        """A reference line's pressure, if it applies."""
        if ref == "limit" and shown.cfg is not None:
            return float(shown.cfg["P_cmbr_max"]) * to_si(1.0, shown.cfg["P_cmbr_max_unit"], "pressure")
        if ref == "dp300" and shown.spi:
            return to_si(300.0, "psi", "pressure")
        return None

    def show_runs(self, shown: Shown):
        """Plot a run (or none), ticked runs dashed under it, and a measured curve."""
        times = [x.t for x in ([shown.output] if shown.output is not None else []) + [o for o, _s in shown.overlays]]
        if shown.measured is not None:
            times.append(shown.measured.t)
        new = (shown.output is not self.shown.output or shown.measured is not self.shown.measured
               or len(shown.overlays) != len(self.shown.overlays))
        self.shown = shown
        if new and times:
            self.time_range = (min(0.0, *(float(t[0]) for t in times)), max(float(t[-1]) for t in times))
        self.redraw()

    def set_theme(self, name: str):
        self._theme = name
        self.redraw()

    def redraw(self):
        colors = colors_for(self._theme)
        for page in self.pages.values():
            page.graph.setBackground(colors["plot_bg"])
            page.build(self.shown, self._get_units(), colors)
        self.leave()

    def time_range_changed(self, _view, time_range):
        """Zooming one tab's time axis zooms them all."""
        if self._syncing:
            return
        self.time_range = tuple(time_range)
        self._syncing = True
        try:
            for page in self.pages.values():
                for plot in page.plots.values():
                    plot.vb.setXRange(*self.time_range, padding=0)
        finally:
            self._syncing = False

    def hover(self, x: float):
        """Show the values at a time on the current tab, and tell the drawing which time it is."""
        shown = self.shown
        page = self.tabs.currentWidget()
        i = None
        time = x
        values: dict[str, str] = {}
        if shown.output is not None and len(shown.output.t) and shown.output.t[0] <= x <= shown.output.t[-1]:
            t = np.asarray(shown.output.t)
            i = int(np.clip(np.searchsorted(t, x), 0, t.size - 1))
            time = float(t[i])
            for line in page.lines(shown):
                if (series := curve(shown.output, shown.settings, line.key)) is not None:
                    values[line.key] = self.text(float(series[i]), line.quantity)
        if shown.measured is not None:
            for key, _name, q in MEASURED.get(page.name, ()):
                if (series := shown.measured.series(key)) is not None:
                    value = float(np.interp(time, shown.measured.t, series, left=np.nan, right=np.nan))
                    values[f"measured:{key}"] = self.text(value, q) if math.isfinite(value) else "—"
        page.show_values({**self._reference_values(), **values}, time)
        if i is not None:
            self.hovered.emit(i)

    def leave(self):
        page = self.tabs.currentWidget()
        if page is not None:
            page.show_values(self._reference_values(), None)
        self.hovered.emit(None)

    def _reference_values(self) -> dict[str, str]:
        return {ref: self.text(value, "pressure") for ref in ("limit", "dp300")
                if (value := self.reference(ref, self.shown)) is not None}

    def text(self, value: float, quantity: str) -> str:
        return f"{value:.3g}" if quantity == "ratio" else self._get_units().text(value, quantity, 3)


class SimulationPage(QSplitter):
    """The Simulation tab. The main window runs the motor, keeps its runs and fills these in."""

    def __init__(self, prefs: QSettings, get_units: Callable[[], DisplayUnits], grain_scene: GrainScene):
        super().__init__(Qt.Orientation.Horizontal)
        self._prefs = prefs
        self.form = self._make_form()
        self.addWidget(self.form)
        self.addWidget(self._make_results(get_units, grain_scene))
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 1)
        self.setSizes([500, 900])

    def _make_form(self) -> QWidget:
        wrap = QWidget()
        wl = page_layout(QVBoxLayout(wrap))
        wl.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runButton")
        wl.addWidget(self.run_btn)

        # The motor is set on the Motor tab; this only says which one runs.
        motor, ml = card_frame("Motor")
        title = ml.itemAt(0).widget()
        ml.removeWidget(title)
        head = QHBoxLayout()
        self.edit_btn = QPushButton("Edit on Motor tab")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.edit_btn)
        ml.addLayout(head)
        self.motor_summary = QLabel("—")
        self.motor_summary.setObjectName("cardLabel")
        self.motor_summary.setWordWrap(True)
        ml.addWidget(self.motor_summary)
        self.unapplied = QWidget()
        ul = QHBoxLayout(self.unapplied)
        ul.setContentsMargins(0, 0, 0, 0)
        self.unapplied_text = QLabel("")
        self.unapplied_text.setObjectName("notApplied")
        self.unapplied_text.setWordWrap(True)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Copy the Motor tab's sized parts into the motor.")
        ul.addWidget(self.unapplied_text, 1)
        ul.addWidget(self.apply_btn, 0, Qt.AlignmentFlag.AlignTop)
        ml.addWidget(self.unapplied)

        # The cards scroll under the Run button, keeping their edges in line when the scrollbar shows.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(CARD_GAP)
        root.addWidget(motor)

        self.solve_tank_cooling = QCheckBox("Conserve tank mass and energy (experimental)")
        self.solve_tank_cooling.setToolTip(
            "Uses the same mass and energy balance through liquid runout and gas discharge.\n"
            "Assumes liquid leaves until runout, phase equilibrium, and no heat from the tank wall.\n"
            "After runout, vapor leaves and any condensate stays in the tank.\n"
            "Uses CoolProp properties. Leave unchecked for the original HRAP tank model."
        )
        self.tmax = PlainDoubleSpinBox()
        self.tmax.setRange(0.01, 120)
        self.tmax.setValue(10)
        self.tburn = PlainDoubleSpinBox()
        self.tburn.setRange(0.0, 120)
        self.tburn.setToolTip("Closes the oxidizer valve at this time. 0 = never.")
        self.dt = PlainDoubleSpinBox()
        self.dt.setDecimals(3)
        self.dt.setRange(0.01, 100)
        self.dt.setValue(1.0)
        self.dt.setToolTip("1 ms is usually accurate to about 0.1%. To check a new motor, rerun at 0.2 ms and compare.")
        self.P_cmbr = UnitRow(PRESSURE_ITEMS, "atm")
        self.P_cmbr.setToolTip("Chamber pressure before ignition. Leave it at 1 atm.")
        sim, sim_l = card_frame("Run")
        sf = FieldGrid()
        sf.add("Max run time", self.tmax, "s")
        sf.add("Close valve at", self.tburn, "s")
        sf.add("Timestep", self.dt, "ms")
        sf.add("Chamber start pressure", self.P_cmbr)
        sf.add_line(self.solve_tank_cooling)
        sim_l.addLayout(sf)
        root.addWidget(sim)

        runs, runs_l = card_frame("Runs")
        self.run_list = HintList("No runs yet.")
        self.run_list.setToolTip("Every simulation of this motor lands here, with what changed from its previous run.\n"
                                 "Click a run to load its settings and results; tick runs to draw them under it,\n"
                                 "each in the line style beside its name. Right-click to rename, export or delete.")
        self.run_list.setMinimumHeight(120)
        self.run_list.setIconSize(QSize(20, 10))
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.run_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.run_list.setStyleSheet("QListView::item { padding: 3px 4px; }")  # room around each checkbox
        self.run_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.clear_runs = QPushButton("Clear runs")
        runs_l.addWidget(self.run_list)
        runs_l.addWidget(self.clear_runs)
        root.addWidget(runs)

        adv, adv_l = card_frame("Advanced")
        af = FieldGrid()
        self.adv_on = QCheckBox("Enable advanced options")
        self.adv_on.setToolTip("These options don't necessarily match MATLAB HRAP's results.")
        self.live_chem = QCheckBox("Live chemistry (WIP)")
        self.ox_fluid = PlainComboBox()
        self.ox_fluid.addItems(["N2O_legacy", "NitrousOxide (CoolProp)"])
        self.ox_fluid.setToolTip("Which nitrous property data the tank uses with the SPI injector model.\n"
                                 "HEM and Dyer always use CoolProp.")
        for box in (self.adv_on, self.live_chem):
            af.add_line(box)
        af.add("Oxidizer fluid", self.ox_fluid)
        adv_l.addLayout(af)
        root.addWidget(adv)
        line_up(sf, af)

        # Never let the splitter squeeze the panel narrower than its fields; there's no horizontal scrollbar.
        scroll.setMinimumWidth(inner.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        root.addStretch(1)
        scroll.setWidget(inner)
        wl.addWidget(scroll, 1)
        return wrap

    def _make_results(self, get_units: Callable[[], DisplayUnits], grain_scene: GrainScene) -> QWidget:
        box = QWidget()
        layout = page_layout(QVBoxLayout(box))
        self.limit_warning = QLabel("")
        self.limit_warning.setObjectName("sizingError")
        self.limit_warning.setWordWrap(True)
        self.limit_warning.hide()
        self.plots = PlotPanel(self._prefs, get_units)

        # Under the plots: the motor drawing, an STL grain in 3D (both at the time hovered), and the run's summary.
        self.motor_panel = MotorPanel()
        self.run_grain = RunGrainView(grain_scene, get_units)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.summary.setPlaceholderText("Press Run to simulate this motor.")
        self.bottom = QTabWidget()
        self.bottom.setObjectName("subTabs")
        self.bottom.setDocumentMode(True)
        self.bottom.tabBar().setDrawBase(False)
        self.bottom.addTab(self.motor_panel, "Motor drawing")
        self.bottom.addTab(self.run_grain, "Grain")
        self.bottom.addTab(self.summary, "Summary")
        self.bottom.setTabVisible(1, False)  # until an STL grain is in
        chosen = str(self._prefs.value("simulationPanel") or "")
        for i in range(self.bottom.count()):
            if self.bottom.tabText(i) == chosen and self.bottom.isTabVisible(i):
                self.bottom.setCurrentIndex(i)
        self.bottom.tabBarClicked.connect(lambda i: self._prefs.setValue("simulationPanel", self.bottom.tabText(i)))

        # Drag the gap between them to trade plot height for the drawing.
        column = QSplitter(Qt.Orientation.Vertical)
        column.setObjectName("plainSplitter")
        column.setChildrenCollapsible(False)
        column.setHandleWidth(10)
        column.addWidget(self.plots)
        column.addWidget(self.bottom)
        column.setStretchFactor(0, 1)
        column.setSizes([2000, 1])  # the drawing at its smallest; the plots get the rest
        layout.addWidget(self.limit_warning)
        layout.addWidget(column, 1)
        return box

    def show_grain_tab(self, shown: bool):
        """The Grain tab under the plots, while an STL grain is loaded."""
        self.bottom.setTabVisible(1, shown)
        if shown and str(self._prefs.value("simulationPanel") or "") == "Grain":
            self.bottom.setCurrentIndex(1)
