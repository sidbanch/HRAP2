"""Parameter studies, frozen run history, and comparisons of selected cases."""
from __future__ import annotations

import csv
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.study import INPUTS, MODELS, OUTPUTS, Case, case_cfg, check_axes, grid, passing_values, study, uses_spi
from hrap.gui.sizing import card_frame
from hrap.gui.widgets import UnitRow
from hrap.io.config import chamber_limit, clone_cfg
from hrap.units import AREA_ITEMS, LENGTH_ITEMS, PRESSURE_ITEMS, TEMP_ITEMS, DisplayUnits, from_si, to_si

OK, WARN, BAD = QColor("#2f7d4f"), QColor("#a87a22"), QColor("#a8413b")


def _small(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("cardLabel")
    label.setWordWrap(True)
    return label


def _chip(color: QColor, text: str) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    swatch = QLabel()
    swatch.setFixedSize(12, 12)
    swatch.setStyleSheet(f"background: {color.name()}; border-radius: 3px;")
    h.addWidget(swatch)
    label = QLabel(text)
    label.setObjectName("cardLabel")
    h.addWidget(label)
    return w


class TileDelegate(QStyledItemDelegate):
    """Result cells as rounded colored tiles with a gap between them; selected tiles get an outline."""

    def paint(self, painter: QPainter, option, index):
        color = index.data(Qt.ItemDataRole.BackgroundRole)
        if color is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gap = 3 if min(option.rect.width(), option.rect.height()) > 30 else 1
        rect = option.rect.adjusted(gap, gap, -gap, -gap)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, 5, 5)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor("white"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        font = painter.font()
        bounds = painter.fontMetrics().boundingRect(text)
        scale = min(1.0, (rect.width() - 4) / max(1, bounds.width()), rect.height() / max(1, bounds.height()))
        if scale < 1.0:
            font.setPointSizeF(max(6.0, font.pointSizeF() * scale))
            painter.setFont(font)
        painter.setPen(QColor("white"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class Axis(QWidget):
    def __init__(self, optional=False):
        super().__init__()
        self.key = QComboBox()
        if optional:
            self.key.addItem("None", "")
        for key, spec in INPUTS.items():
            self.key.addItem(spec.label, key)
        self.values = QLineEdit()
        self.values.setPlaceholderText("12, 15, 18, 24   or   12:24:5")
        self.values.setToolTip("Comma-separated values, or start:end:count for evenly spaced values including both ends.")
        self.unit = QComboBox()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.addWidget(self.key, 0, 0)
        layout.addWidget(self.unit, 0, 1)
        layout.addWidget(self.values, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        self.key.currentIndexChanged.connect(self._changed)
        self.unit.currentTextChanged.connect(self._convert)
        self.cfg = {}
        self._unit = ""
        self._changed()

    def _convert(self, unit):
        key = self.key.currentData()
        if unit and self._unit and key:
            try:
                values = self.read()[1]
                self.values.setText(", ".join(f"{from_si(v, unit, INPUTS[key].quantity):.8g}" for v in values))
            except ValueError:
                pass
        self._unit = unit

    def _changed(self):
        key = self.key.currentData()
        self.unit.blockSignals(True)
        self.unit.clear()
        quantity = INPUTS[key].quantity if key else None
        choices = {"length": LENGTH_ITEMS, "area": AREA_ITEMS, "temperature": TEMP_ITEMS}.get(quantity, [])
        self.unit.addItems(choices)
        default = {"length": "in", "area": "in^2", "temperature": "F"}.get(quantity, "")
        self.unit.setCurrentText(default)
        self._unit = default
        self.unit.blockSignals(False)
        self.unit.setVisible(bool(choices))
        self.values.setEnabled(bool(key))
        if key and self.cfg:
            value = INPUTS[key].current(self.cfg)
            if quantity:
                value = from_si(value, default, quantity)
            values = [value - 10, value, value + 10] if quantity == "temperature" else [value * f for f in (0.8, 1, 1.2)]
            if key in ("fill", "cstar_eff"):
                values = [min(v, 100) for v in values]
            self.values.setText(", ".join(f"{v:.3g}" for v in dict.fromkeys(values)))

    def set_axis(self, key, values=None):
        self.key.setCurrentIndex(self.key.findData(key))
        self._changed()
        if values is not None:
            quantity = INPUTS[key].quantity if key else None
            shown = [from_si(v, self._unit, quantity) if quantity else v for v in values]
            self.values.setText(", ".join(f"{v:.9g}" for v in shown))

    def read(self):
        key = self.key.currentData()
        if not key:
            return None
        text = self.values.text().strip()
        try:
            if ":" in text:
                lo, hi, count = text.split(":")
                n = int(count)
                if not 1 <= n <= 100:
                    raise ValueError()
                values = np.linspace(float(lo), float(hi), n).tolist()
            else:
                values = [float(v.strip()) for v in text.split(",")]
            if not values or len(values) > 100 or not all(math.isfinite(v) for v in values):
                raise ValueError()
        except ValueError as exc:
            raise ValueError(f"{INPUTS[key].label}: enter numbers separated by commas, or start:end:count (1–100).") from exc
        quantity = INPUTS[key].quantity
        values = [to_si(v, self._unit, quantity) if quantity else v for v in values]
        return key, list(dict.fromkeys(values))


@dataclass
class StudyRun:
    cfg: dict
    axes: list
    models: list
    cases: list[Case] = field(default_factory=list)
    error: str = ""
    stopped: bool = False
    name: str = ""
    throat_P: float | None = None  # Pa; each case's throat was sized for this chamber pressure

    def key(self) -> str:
        """Identifies what was run: the motor, the input values, the fuel models and the throat sizing."""
        return json.dumps([self.cfg, self.axes, self.models, self.throat_P], sort_keys=True, default=str)

    @property
    def complete(self) -> bool:
        return len(self.cases) == self.total and not self.error and not self.stopped

    @property
    def total(self):
        return len(grid(self.axes)) * len(self.models)


class StudyWorker(QObject):
    case_done = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, result):
        super().__init__()
        self.result = result
        self.stop = Event()

    def run(self):
        try:
            r = self.result
            for case in study(r.cfg, r.axes, r.models, stopped=self.stop.is_set, throat_P=r.throat_P):
                self.case_done.emit(case)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class StudyPage(QWidget):
    started = Signal()
    finished = Signal()
    runs_changed = Signal()

    def __init__(self, get_cfg: Callable[[], dict], get_units: Callable[[], DisplayUnits], on_open,
                 get_unapplied: Callable[[], list[str]] = lambda: [],
                 ask_save: Callable[[str, str, str], str] | None = None):
        super().__init__()
        self._get_cfg, self._get_units, self._on_open = get_cfg, get_units, on_open
        self._get_unapplied = get_unapplied
        self._ask_save = ask_save or (lambda caption, name, filters: QFileDialog.getSaveFileName(self, caption, name, filters)[0])
        self._loaded_from = ""
        self._thread = self._worker = None
        self.runs: list[StudyRun] = []
        self.result = None
        self._cells = {}
        self._base_override = None

        # Left: what to study. Right: the result grid, and the curves of the selected cases.
        self.axes = [Axis(), Axis(optional=True)]
        self.models = QComboBox()
        self.models.addItem("Current motor's fuel model", None)
        self.models.addItem("Both fuel models", list(MODELS))
        for key, label in MODELS.items():
            self.models.addItem(label, [key])
        self.axes[1].key.setCurrentIndex(self.axes[1].key.findData("inj_CdA"))
        for axis in self.axes:
            axis.key.activated.connect(self._edited)
            axis.key.currentIndexChanged.connect(lambda *_: self._show_throat_option())
            axis.values.textEdited.connect(self._edited)
        self._axes_edited = False  # until then, the values follow the open motor
        self.size_throat = QCheckBox("Size the throat for each case")
        self.size_throat.setToolTip("Each case gets the throat and expansion ratio that the Motor tab would size for its\n"
                                    "chamber pressure target, instead of the motor's own nozzle.")
        self.size_throat.toggled.connect(lambda *_: self._update_source())
        self.size_throat.hide()
        self.source = _small("")
        self.source.setStyleSheet(f"color: {WARN.lighter(150).name()};")
        self.source.hide()

        self.setup, setup = card_frame("Study")
        fields = QGridLayout()
        fields.setHorizontalSpacing(10)
        fields.setVerticalSpacing(8)
        fields.setColumnStretch(1, 1)
        for i, (name, axis) in enumerate(zip(("Columns", "Rows"), self.axes)):
            fields.addWidget(QLabel(name), i, 0, Qt.AlignmentFlag.AlignTop)
            fields.addWidget(axis, i, 1)
        fields.addWidget(QLabel("Fuel model"), 2, 0)
        fields.addWidget(self.models, 2, 1)
        fields.addWidget(self.size_throat, 3, 0, 1, 2)
        setup.addLayout(fields)
        self.run_btn = QPushButton("Run study")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn, 1)
        buttons.addWidget(self.stop_btn)
        setup.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        setup.addWidget(self.progress)
        setup.addWidget(self.source)

        limits, limits_l = card_frame("Limits")
        self.max_pressure = QLabel("—")
        self.max_pressure.setObjectName("cardValue")
        self.max_pressure.setToolTip("The chamber pressure limit, set in Targets on the Motor tab.")
        self.min_pressure = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.min_pressure.set_si(0, "pressure")
        self.min_pressure.setToolTip("Lowest acceptable peak chamber pressure (absolute).")
        self.max_dp = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.max_dp.set_si(to_si(300, "psi", "pressure"), "pressure")
        self.max_dp.setToolTip("The SPI injector model treats the nitrous as liquid all the way through the hole. Above roughly\n"
            "300 psi of pressure drop, real nitrous starts boiling in the hole and flows less, so SPI\n"
            "overpredicts the flow, and with it the thrust and chamber pressure. Cells whose burn-average\n"
            "injector ΔP is over this are marked. It only applies to motors using SPI.")
        grid_l = QGridLayout()
        grid_l.setHorizontalSpacing(10)
        grid_l.setVerticalSpacing(8)
        grid_l.setColumnStretch(1, 1)
        for r, (label, control) in enumerate((("Max peak Pc", self.max_pressure), ("Min peak Pc", self.min_pressure),
                                               ("Injector ΔP warning", self.max_dp))):
            grid_l.addWidget(QLabel(label), r, 0)
            grid_l.addWidget(control, r, 1)
        limits_l.addLayout(grid_l)
        for control in (self.min_pressure, self.max_dp):
            control.spin.valueChanged.connect(self.refresh)
            control.unit.currentTextChanged.connect(self.refresh)

        runs, runs_l = card_frame("Runs")
        self.history = QListWidget()
        self.history.setMinimumHeight(90)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.history.setToolTip("Double-click a run to rename it.")
        self.history.currentRowChanged.connect(self._show_run)
        self.history.itemChanged.connect(self._renamed)
        runs_l.addWidget(self.history, 1)
        files = QHBoxLayout()
        self.load_btn = QPushButton("Load…")
        self.load_btn.clicked.connect(self._load)
        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._save)
        self.export_btn = QPushButton("CSV…")
        self.export_btn.setToolTip("Export the shown run's results as CSV.")
        self.export_btn.clicked.connect(self._export)
        for w in (self.load_btn, self.save_btn, self.export_btn):
            files.addWidget(w)
        runs_l.addLayout(files)

        left = QWidget()
        left.setFixedWidth(400)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(12)
        left_l.addWidget(self.setup)
        left_l.addWidget(limits)
        left_l.addWidget(runs, 1)

        # The result grid.
        results, results_l = card_frame("Results")
        head = results_l.itemAt(0).widget()
        results_l.removeWidget(head)
        top = QHBoxLayout()
        top.addWidget(head)
        top.addStretch(1)
        top.addWidget(QLabel("Show"))
        self.shown_output = QComboBox()
        self.shown_output.addItems([m[0] for m in OUTPUTS])
        self.shown_output.setToolTip("O/F is nitrous mass ÷ fuel mass over the simulated liquid burn.")
        self.shown_output.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.shown_output)
        results_l.addLayout(top)
        self.answer = QLabel("")
        self.answer.setWordWrap(True)
        self.answer.setObjectName("cardValue")
        results_l.addWidget(self.answer)
        self.status = _small("")
        results_l.addWidget(self.status)
        self.col_caption = _small("")
        self.col_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_l.addWidget(self.col_caption)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setShowGrid(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setHighlightSections(False)
        self.table.setStyleSheet("QTableWidget { border: none; background: transparent; selection-background-color: transparent; }")
        self.table.setItemDelegate(TileDelegate(self.table))
        for header in (self.table.horizontalHeader(), self.table.verticalHeader()):
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            header.setMinimumSectionSize(16)
        self.table.itemSelectionChanged.connect(self._selection)
        self.table.cellDoubleClicked.connect(lambda *_: self._open())
        results_l.addWidget(self.table, 1)
        legend = QHBoxLayout()
        legend.setSpacing(16)
        for color, text in ((OK, "Within limits"), (WARN, "Warning (click a cell for why)"), (BAD, "Over the pressure limit or burned out")):
            legend.addWidget(_chip(color, text))
        legend.addStretch(1)
        self.legend = QWidget()
        self.legend.setLayout(legend)
        legend.setContentsMargins(0, 0, 0, 0)
        self.legend.hide()
        results_l.addWidget(self.legend)

        # Curves of the selected cases.
        compare, compare_l = card_frame("Compare")
        head = compare_l.itemAt(0).widget()
        compare_l.removeWidget(head)
        line = QHBoxLayout()
        line.addWidget(head)
        line.addStretch(1)
        self.trace = QComboBox()
        for name, key, quantity in (("Thrust", "F_thr", "force"), ("Chamber pressure", "P_cmbr", "pressure"),
                                     ("O/F", "OF", "ratio"), ("Port diameter", "grn_ID", "length")):
            self.trace.addItem(name, (key, quantity))
        self.trace.currentIndexChanged.connect(self._selection)
        self.open_btn = QPushButton("Open as motor")
        self.open_btn.setToolTip("Open the selected case as a new motor tab.")
        self.open_btn.clicked.connect(self._open)
        line.addWidget(QLabel("Plot"))
        line.addWidget(self.trace)
        line.addWidget(self.open_btn)
        compare_l.addLayout(line)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet(f"color: {WARN.lighter(150).name()}; font-weight: 600;")
        self.warnings.hide()
        compare_l.addWidget(self.warnings)
        self.details = _small("Click cells to plot them, ⌘/Ctrl-click to add more. Double-click a cell to open it as a motor in a new tab.")
        self.details.setWordWrap(True)
        compare_l.addWidget(self.details)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMinimumHeight(160)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        compare_l.addWidget(self.plot, 1)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(results)
        split.addWidget(compare)
        split.setSizes([520, 360])

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(left)
        root.addWidget(split, 1)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.open_btn.setEnabled(False)

    def motor_changed(self):
        if self.busy():
            return
        self._base_override = None
        cfg = self._get_cfg()
        for axis in self.axes:
            axis.cfg = cfg
            if not self._axes_edited:
                axis._changed()
        self._loaded_from = ""
        self._show_throat_option()
        self._update_source()
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_source()

    def _update_source(self):
        """Warn when the study won't include Motor-tab sizing that hasn't been applied, or runs a loaded motor."""
        if self._loaded_from:
            text = f"Runs the motor saved in {self._loaded_from}, not the open one."
        else:
            changes = self._get_unapplied()
            if self.size_throat.isChecked() and self.size_throat.isVisibleTo(self):  # each case sizes its own nozzle
                changes = [c for c in changes if not c.startswith(("throat ", "expansion ratio "))]
            text = (f"Not applied from the Motor tab: {', '.join(changes)}. The study runs the motor without these."
                    if changes else "")
        self.source.setText(text)
        self.source.setVisible(bool(text))

    def _edited(self, *_):
        self._axes_edited = True

    def _show_throat_option(self):
        """Only when the Motor tab sizes the nozzle for a chamber pressure, and the study isn't varying the throat."""
        target = self._throat_target(self._base_override or self._get_cfg())
        varied = any(axis.key.currentData() == "throat" for axis in self.axes)
        self.size_throat.setVisible(target is not None and not varied)
        if target is not None:
            self.size_throat.setText(f"Size the throat for each case ({self._get_units().text(target, 'pressure')} target)")
        self._update_source()

    @staticmethod
    def _throat_target(cfg: dict) -> float | None:
        """The Motor tab's chamber pressure target (Pa), when it sizes the nozzle for one."""
        sizing = cfg.get("sizing") or {}
        return float(sizing["P_cmbr"]) if sizing.get("nozzle_from") == "P_cmbr" and sizing.get("P_cmbr") else None

    def busy(self):
        return self._thread is not None

    def stop(self):
        if self._worker is not None:
            self._worker.stop.set()
            self.result.stopped = True
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping after the simulations already running finish…")

    def _run(self):
        if self.busy():
            return
        try:
            axes = [a.read() for a in self.axes]
            axes = [a for a in axes if a]
            cfg = clone_cfg(self._base_override or self._get_cfg())
            models = self.models.currentData() or [cfg.get("reg_model", "Shifting OF")]
            check_axes(cfg, [key for key, _ in axes], models)
            throat_P = self._throat_target(cfg) if self.size_throat.isChecked() and self.size_throat.isVisibleTo(self) else None
            result = StudyRun(cfg, axes, models, throat_P=throat_P)
            same = next((i for i, run in enumerate(self.runs) if run.complete and run.key() == result.key()), None)
            if same is not None:
                self.history.setCurrentRow(same)
                self.status.setText(f"Same inputs as “{self.runs[same].name}”, so it's shown instead of running again.")
                return
            if result.total > 500:
                raise ValueError(f"That is {result.total} simulations. Narrow the ranges to at most 500.")
            for values in grid(axes):
                case_cfg(cfg, values, models[0], throat_P)
        except Exception as exc:
            self.status.setText(str(exc))
            return
        result.name = f"{cfg.get('mtr_nm') or 'Motor'} · " + " × ".join(INPUTS[k].label for k, _ in axes)
        self.runs.append(result)
        self.history.addItem(self._history_item(result))
        self.history.setCurrentRow(len(self.runs) - 1)
        self.result = result
        self.progress.setRange(0, result.total)
        self.progress.setValue(0)
        self.progress.show()
        for w in (self.setup, self.run_btn, self.history, self.load_btn):
            w.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._thread = QThread()
        self._worker = StudyWorker(result)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.case_done.connect(self._case)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._finished)
        self._thread.start()
        self.started.emit()
        self.refresh()

    def _case(self, case):
        self.result.cases.append(case)
        self.progress.setValue(len(self.result.cases))
        self.refresh()

    def _failed(self, message):
        self.result.error = message

    def _finished(self):
        self._thread.wait()
        self._thread.deleteLater()
        self._thread = self._worker = None
        for w in (self.setup, self.run_btn, self.history, self.load_btn):
            w.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.hide()
        self.refresh()
        self.finished.emit()
        self.runs_changed.emit()

    def _renamed(self, item):
        row = self.history.row(item)
        name = item.text().strip()
        if 0 <= row < len(self.runs):
            if name:
                self.runs[row].name = name
                self.runs_changed.emit()
            else:
                item.setText(self.runs[row].name)

    def dump_runs(self) -> bytes:
        """The finished runs, for the session."""
        return pickle.dumps([r for r in self.runs if r is not self.result or not self.busy()])

    def load_runs(self, data: bytes):
        """Bring back a session's runs, showing the last one."""
        runs = pickle.loads(data)
        if not all(isinstance(r, StudyRun) for r in runs):
            return
        self.runs = runs
        self.history.blockSignals(True)
        self.history.clear()
        for run in runs:
            self.history.addItem(self._history_item(run))
        self.history.blockSignals(False)
        if runs:
            self.history.setCurrentRow(len(runs) - 1)

    def _history_item(self, run: StudyRun) -> QListWidgetItem:
        item = QListWidgetItem(run.name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setToolTip(f"{run.name}\n{run.total} runs. Double-click to rename.")
        return item

    def _show_run(self, index):
        if 0 <= index < len(self.runs):
            self.result = self.runs[index]
            self.table.clearSelection()
            self.refresh()

    def _value_text(self, key, value, digits=4):
        spec = INPUTS[key]
        return self._get_units().text(value, spec.quantity, digits) if spec.quantity else f"{value:.{digits}g}"

    def _value_texts(self, key, values) -> list[str]:
        """Labels with the fewest digits (at least 3) that still tell the values apart."""
        for digits in range(3, 10):
            texts = [self._value_text(key, v, digits) for v in values]
            if len(set(texts)) == len(texts):
                return texts
        return texts

    def _label(self, case):
        return ", ".join(f"{INPUTS[key].label} {self._value_text(key, v)}" for key, v in case.values) + " · " + MODELS[case.model]

    def _flags(self, case):
        flags = []
        if case.peak_P_cmbr > chamber_limit(self.result.cfg):
            flags.append("over pressure limit")
        if case.peak_P_cmbr < self.min_pressure.si("pressure"):
            flags.append("below minimum pressure")
        if case.burnout:
            flags.append("fuel depleted")
        if case.outside_table:
            flags.append("O/F outside combustion table")
        if case.end_cond == "Max Simulation Time Reached":
            flags.append("time limit reached")
        if uses_spi(self.result.cfg) and case.avg_inj_dP > self.max_dp.si("pressure"):
            flags.append("SPI ΔP above warning")
        return flags

    def refresh(self):
        r = self.result
        u = self._get_units()
        cfg = r.cfg if r is not None else self._base_override or self._get_cfg()
        self.max_pressure.setText(f"{u.text(chamber_limit(cfg), 'pressure')} (Motor tab)")
        if r is None:
            return
        self.legend.show()
        selected = {(i.row(), i.column()) for i in self.table.selectedItems()}
        self.table.blockSignals(True)
        self.table.clearContents()
        xkey, xs = r.axes[0]
        ykey, ys = r.axes[1] if len(r.axes) > 1 else (None, [None])
        rows = [(y, model) for y in ys for model in r.models]
        self.table.setColumnCount(len(xs))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(self._value_texts(xkey, xs))
        many_models = len(r.models) > 1
        y_texts = dict(zip(ys, self._value_texts(ykey, ys))) if ykey else {}
        self.table.setVerticalHeaderLabels([
            "  " + " · ".join(filter(None, [y_texts.get(y, ""), MODELS[model] if many_models or not ykey else ""])) + "  "
            for y, model in rows])
        label, name, quantity = OUTPUTS[self.shown_output.currentIndex()]
        unit = u.unit(quantity) if quantity else "s"
        rows_label = INPUTS[ykey].label if ykey else "Fuel model"
        self.col_caption.setText(f"{label}{f' ({unit})' if unit else ''}   ·   Columns: {INPUTS[xkey].label}   ·   "
                                 f"Rows: {rows_label}" + ("" if many_models or not ykey else f" ({MODELS[r.models[0]]})")
                                 + ("   ·   Throat sized for each case" if r.throat_P else ""))
        self._cells = {}
        for case in r.cases:
            col = xs.index(case.value(xkey))
            row = rows.index((case.value(ykey) if ykey else None, case.model))
            self._cells[row, col] = case
            value = getattr(case, name)
            flags = self._flags(case)
            if not math.isfinite(value):
                text = "—"
            else:
                shown = u.value(value, quantity) if quantity else value
                text = f"{shown:.0f}" if abs(shown) >= 1e4 else f"{shown:.4g}"
            item = QTableWidgetItem(text + (" ⚠" if flags else ""))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            bad = case.burnout or case.peak_P_cmbr > chamber_limit(r.cfg)
            item.setBackground(BAD if bad else WARN if flags else OK)
            item.setForeground(QColor("white"))
            item.setToolTip(self._label(case) + ("\n" + "\n".join(flags) if flags else "") + f"\nEnd: {case.end_cond}")
            self.table.setItem(row, col, item)
            item.setSelected((row, col) in selected)
        self.table.blockSignals(False)
        n = len(r.cases)
        warned = sum(bool(self._flags(c)) for c in r.cases)
        if self.busy():
            text = f"Running {n} of {r.total}"
        elif r.error:
            text = f"Failed after {n} of {r.total} runs: {r.error}"
        elif r.stopped:
            text = f"Stopped after {n} of {r.total} runs · {warned} with warnings"
        else:
            text = f"{n} runs · {warned} with warnings"
        self.status.setText(text)
        if n == r.total and not r.error and not self.busy():
            passing = passing_values(r.cases, xkey, self.min_pressure.si("pressure"), chamber_limit(r.cfg),
                                     self.max_dp.si("pressure") if uses_spi(r.cfg) else float("inf"))
            every = "every row" if len(rows) > 1 else "this run"
            if passing:
                listed = ", ".join(self._value_texts(xkey, passing))
                self.answer.setText(f"{INPUTS[xkey].label} within limits for {every}: {listed}")
            else:
                self.answer.setText(f"No {INPUTS[xkey].label.lower()} stays within limits for {every}.")
        else:
            self.answer.setText("")
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(r.cases))
        self._selection()

    def _selected(self):
        return [self._cells[i.row(), i.column()] for i in self.table.selectedItems() if (i.row(), i.column()) in self._cells]

    def _selection(self):
        cases = self._selected()
        self.open_btn.setEnabled(len(cases) == 1 and not self.busy())
        self.plot.clear()
        key, quantity = self.trace.currentData()
        u = self._get_units()
        self.plot.setLabel("left", self.trace.currentText(), units=u.unit(quantity))
        for i, c in enumerate(cases[:8]):
            values = np.array([u.value(v, quantity) for v in c.traces[key]])
            self.plot.plot(c.traces["t"], values, name=" / ".join(self._value_text(k, v) for k, v in c.values) + " · " + MODELS[c.model], pen=pg.mkPen(pg.intColor(i, hues=8), width=2))
        if len(cases) == 1:
            c = cases[0]
            why = self._why(c)
            self.warnings.setText("\n".join(f"⚠ {w}" for w in why))
            self.warnings.setVisible(bool(why))
            sized = []
            if "start_OF" in dict(c.values):
                sized.append(f"injector CdA {u.text(c.inj_CdA, 'area', 3)}")
            if self.result.throat_P:
                sized.append(f"throat {u.text(c.throat, 'length', 3)}")
            self.details.setText(f"{self._label(c)}\nPeak Pc {u.text(c.peak_P_cmbr, 'pressure')} · "
                                 f"O/F {c.OF_liquid:.3g} · impulse {u.text(c.total_impulse, 'impulse')} · "
                                 + "".join(f"{x} · " for x in sized) + f"end: {c.end_cond}")
        else:
            self.warnings.hide()
            self.details.setText(f"{len(cases)} selected. Up to 8 are plotted.")

    def _why(self, case) -> list[str]:
        """Each warning on a case, with the numbers behind it."""
        u, r = self._get_units(), self.result
        pc, limit, low = case.peak_P_cmbr, chamber_limit(r.cfg), self.min_pressure.si("pressure")
        why = []
        if pc > limit:
            why.append(f"Peak chamber pressure {u.text(pc, 'pressure')} is over the {u.text(limit, 'pressure')} limit.")
        if pc < low:
            why.append(f"Peak chamber pressure {u.text(pc, 'pressure')} is under the {u.text(low, 'pressure')} minimum.")
        if case.burnout:
            why.append("The grain burned through to its outside diameter before the tank emptied.")
        if case.outside_table:
            why.append("The O/F left HRAP's combustion table (O/F 1 to 10), so thrust and pressure read high there.")
        if case.end_cond == "Max Simulation Time Reached":
            why.append("The run hit the maximum run time before the burn finished.")
        if uses_spi(r.cfg) and case.avg_inj_dP > self.max_dp.si("pressure"):
            why.append(f"Average injector ΔP {u.text(case.avg_inj_dP, 'pressure')} is over the "
                       f"{u.text(self.max_dp.si('pressure'), 'pressure')} warning, where the SPI injector model "
                       "overpredicts nitrous flow.")
        return why

    def _open(self):
        selected = self._selected()
        if self.busy() or len(selected) != 1:
            return
        c = selected[0]
        cfg = case_cfg(self.result.cfg, c.values, c.model, self.result.throat_P)
        cfg["sizing"] = {}  # open the exact built geometry, without old target sizing
        cfg["mtr_nm"] = f"{cfg.get('mtr_nm') or 'Motor'} ({self._label(c).replace(' · ', ', ')})"
        self._on_open(cfg)

    def _save(self):
        if self.result is None:
            return
        r = self.result
        default = "".join(c if c.isalnum() or c in " -_" else "_" for c in r.name).strip() or "study"
        path = self._ask_save("Save study", f"{default}.json", "Study JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps({"name": r.name, "motor": r.cfg, "axes": r.axes, "models": r.models,
                                                 "throat_P": r.throat_P,
                                                 "min_pressure": self.min_pressure.si("pressure"),
                                                 "max_dp": self.max_dp.si("pressure")}, indent=2) + "\n")
            except OSError as exc:
                self.status.setText(str(exc))

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load study", "", "Study JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            cfg, axes, models = data["motor"], data["axes"], data["models"]
            if not isinstance(cfg, dict) or not 1 <= len(axes) <= 2 or any(m not in MODELS for m in models):
                raise ValueError("Invalid study file.")
            if not models or len(set(k for k, _ in axes)) != len(axes):
                raise ValueError("Invalid axes or fuel models.")
            if math.prod(len(v) for _, v in axes) * len(models) > 500:
                raise ValueError("A study can contain at most 500 cases.")
            check_axes(cfg, [k for k, _ in axes], models)
            throat_P = data.get("throat_P")
            for values in grid(axes):
                case_cfg(cfg, values, models[0], throat_P)
            self._base_override = cfg
            self.size_throat.setChecked(bool(throat_P))
            for axis in self.axes:
                axis.cfg = cfg
            for axis, (key, values) in zip(self.axes, axes + [["", []]]):
                axis.set_axis(key, values)
            self.models.setCurrentIndex(next(i for i in range(self.models.count()) if self.models.itemData(i) == models))
            self._axes_edited = True
            self._show_throat_option()
            self.min_pressure.set_si(float(data.get("min_pressure", 0)), "pressure")
            self.max_dp.set_si(float(data.get("max_dp", to_si(300, "psi", "pressure"))), "pressure")
            self._loaded_from = Path(path).name
            self._update_source()
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
            self.status.setText(f"Could not load study: {exc}")

    def _export(self):
        r = self.result
        if r is None:
            return
        default = "".join(c if c.isalnum() or c in " -_" else "_" for c in r.name).strip() or "study"
        path = self._ask_save("Export study results", f"{default}.csv", "CSV (*.csv)")
        if not path:
            return
        u = self._get_units()
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{INPUTS[k].label} ({u.unit(INPUTS[k].quantity)})" if INPUTS[k].quantity else INPUTS[k].label
                                 for k, _ in r.axes] + ["Fuel model"] +
                                [f"{label} ({u.unit(q) if q else 's'})" for label, _, q in OUTPUTS] + ["Fuel depleted", "End", "Warnings", "Study status"])
                for c in sorted(r.cases, key=lambda c: (c.values, c.model)):
                    writer.writerow([u.value(v, INPUTS[k].quantity) if INPUTS[k].quantity else v for k, v in c.values] +
                                    [MODELS[c.model]] + [u.value(getattr(c, name), q) if q else getattr(c, name) for _, name, q in OUTPUTS] +
                                    [c.burnout, c.end_cond, "; ".join(self._flags(c)), "complete" if len(r.cases) == r.total else "partial"])
        except OSError as exc:
            self.status.setText(str(exc))
