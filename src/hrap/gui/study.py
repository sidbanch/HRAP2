"""Parameter studies, frozen run history, and comparisons of selected cases."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.study import INPUTS, MODELS, OUTPUTS, Case, case_cfg, grid, passing_values, study, uses_spi
from hrap.gui.widgets import UnitRow
from hrap.io.config import chamber_limit, clone_cfg
from hrap.units import AREA_ITEMS, LENGTH_ITEMS, PRESSURE_ITEMS, TEMP_ITEMS, DisplayUnits, from_si, to_si


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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.key)
        layout.addWidget(self.values, 1)
        layout.addWidget(self.unit)
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
            self.values.setText(", ".join(f"{v:.7g}" for v in dict.fromkeys(values)))

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
            for case in study(self.result.cfg, self.result.axes, self.result.models, stopped=self.stop.is_set):
                self.case_done.emit(case)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class StudyPage(QWidget):
    started = Signal()
    finished = Signal()

    def __init__(self, get_cfg: Callable[[], dict], get_units: Callable[[], DisplayUnits], on_open):
        super().__init__()
        self._get_cfg, self._get_units, self._on_open = get_cfg, get_units, on_open
        self._thread = self._worker = None
        self.runs: list[StudyRun] = []
        self.result = None
        self._cells = {}
        self._base_override = None

        root = QVBoxLayout(self)
        self.setup = QFrame()
        self.setup.setObjectName("sizingCard")
        form = QGridLayout(self.setup)
        form.setColumnStretch(1, 1)
        form.setVerticalSpacing(6)
        title = QLabel("Compare simulations")
        title.setObjectName("cardTitle")
        note = QLabel("Vary one or two inputs; every other setting stays fixed. Each cell runs the full simulation.")
        note.setWordWrap(True)
        form.addWidget(title, 0, 0, 1, 3)
        form.addWidget(note, 1, 0, 1, 3)
        self.preset = QComboBox()
        self.preset.addItems(["Throat × injector Cd", "Grain length × total injector CdA", "Grain length × burn rate a", "Custom"])
        self.preset.activated.connect(self._preset)
        form.addWidget(QLabel("Start with"), 2, 0)
        form.addWidget(self.preset, 2, 1, 1, 2)
        self.axes = [Axis(), Axis(optional=True)]
        for i, axis in enumerate(self.axes):
            form.addWidget(QLabel("Columns" if i == 0 else "Rows"), 3 + i, 0)
            form.addWidget(axis, 3 + i, 1, 1, 2)
        self.models = QComboBox()
        self.models.addItem("Current motor's fuel model", None)
        self.models.addItem("Both fuel models", list(MODELS))
        for key, label in MODELS.items():
            self.models.addItem(label, [key])
        form.addWidget(QLabel("Fuel model"), 5, 0)
        form.addWidget(self.models, 5, 1)
        self.source = QLabel("")
        self.source.setWordWrap(True)
        form.addWidget(self.source, 6, 0, 1, 3)
        self.show_setup = QPushButton("Study inputs ▾")
        self.show_setup.setCheckable(True)
        self.show_setup.setChecked(True)
        self.show_setup.toggled.connect(self.setup.setVisible)
        self.show_setup.toggled.connect(lambda visible: self.show_setup.setText("Study inputs ▾" if visible else "Study inputs ▸"))
        root.addWidget(self.show_setup)
        root.addWidget(self.setup)
        for axis in self.axes:
            axis.key.activated.connect(lambda *_: self.preset.setCurrentIndex(3))
            axis.values.textEdited.connect(lambda *_: self.preset.setCurrentIndex(3))
        controls = QHBoxLayout()
        self.run_btn = QPushButton("Run study")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.load_btn = QPushButton("Load study…")
        self.load_btn.clicked.connect(self._load)
        self.save_btn = QPushButton("Save study…")
        self.save_btn.clicked.connect(self._save)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export)
        self.progress = QProgressBar()
        self.progress.hide()
        for w in (self.run_btn, self.stop_btn, self.load_btn, self.save_btn, self.export_btn, self.progress):
            controls.addWidget(w)
        controls.addStretch()
        root.addLayout(controls)
        self.status = QLabel("Enter values separated by commas, or start:end:count. Apply any pending Motor sizing before running.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        row = QHBoxLayout()
        self.history = QComboBox()
        self.history.setMinimumWidth(260)
        self.history.currentIndexChanged.connect(self._show_run)
        self.metric = QComboBox()
        self.metric.addItems([m[0] for m in OUTPUTS])
        self.metric.currentIndexChanged.connect(self.refresh)
        row.addWidget(QLabel("Results"))
        row.addWidget(self.history, 1)
        row.addWidget(QLabel("Show"))
        row.addWidget(self.metric)
        root.addLayout(row)
        criteria = QHBoxLayout()
        self.min_pressure = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.min_pressure.set_si(0, "pressure")
        self.max_dp = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.max_dp.set_si(to_si(300, "psi", "pressure"), "pressure")
        self.max_dp.setToolTip("Burn-average injector pressure-drop warning for SPI only. This is a user-set screening threshold.")
        for label, control in (("Minimum peak Pc", self.min_pressure), ("SPI ΔP warning", self.max_dp)):
            criteria.addWidget(QLabel(label))
            criteria.addWidget(control)
            control.spin.valueChanged.connect(self.refresh)
            control.unit.currentTextChanged.connect(self.refresh)
        criteria.addStretch()
        root.addLayout(criteria)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selection)
        self.table.cellDoubleClicked.connect(lambda *_: self._open())
        self.table.setMinimumHeight(160)
        bottom = QWidget()
        layout = QVBoxLayout(bottom)
        layout.setContentsMargins(0, 0, 0, 0)
        line = QHBoxLayout()
        self.open_btn = QPushButton("Open selected case as motor")
        self.open_btn.clicked.connect(self._open)
        self.trace = QComboBox()
        for name, key, quantity in (("Thrust", "F_thr", "force"), ("Chamber pressure", "P_cmbr", "pressure"),
                                     ("O/F", "OF", "ratio"), ("Port diameter", "grn_ID", "length")):
            self.trace.addItem(name, (key, quantity))
        self.trace.currentIndexChanged.connect(self._selection)
        line.addWidget(self.open_btn)
        line.addStretch()
        line.addWidget(QLabel("Overlay selected cells"))
        line.addWidget(self.trace)
        layout.addLayout(line)
        self.details = QLabel("Select cells to compare curves. Ctrl/⌘-click adds cases; double-click opens one as a motor.")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMinimumHeight(180)
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "Time", units="s")
        layout.addWidget(self.plot)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(bottom)
        split.setSizes([260, 250])
        root.addWidget(split, 1)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.open_btn.setEnabled(False)

    def motor_changed(self):
        if self.busy():
            return
        self._base_override = None
        self.show_setup.setChecked(True)
        cfg = self._get_cfg()
        for axis in self.axes:
            axis.cfg = cfg
        self._preset(self.preset.currentIndex())
        self.source.setText(f"Next study: {cfg.get('mtr_nm') or 'current motor'} — applied Motor settings and current Simulation settings.")

    def _preset(self, index):
        self.preset.setCurrentIndex(index)
        if index == 3:
            return
        cfg = self._base_override or self._get_cfg()
        for axis in self.axes:
            axis.cfg = cfg
        first, second = (("throat", "inj_Cd"), ("grain_L", "inj_CdA"), ("grain_L", "a_scale"))[index]
        self.axes[0].set_axis(first)
        self.axes[1].set_axis(second)
        self.models.setCurrentIndex(2 if index == 2 else 0)

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
            if len({key for key, _ in axes}) != len(axes):
                raise ValueError("Choose different inputs for the two axes.")
            cfg = clone_cfg(self._base_override or self._get_cfg())
            models = self.models.currentData() or [cfg.get("reg_model", "Shifting OF")]
            keys = [key for key, _ in axes]
            if "OF" in keys and models != ["Constant OF"]:
                raise ValueError("Choose Fixed O/F to vary its value; the burn-rate law ignores that setting.")
            if "a_scale" in keys and models != ["Shifting OF"]:
                raise ValueError("Choose Burn-rate law to vary a; fixed O/F ignores that setting.")
            result = StudyRun(cfg, axes, models)
            if result.total > 500:
                raise ValueError(f"That is {result.total} simulations. Narrow the ranges to at most 500.")
            for values in grid(axes):
                case_cfg(cfg, values, models[0])
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.source.setText(f"Next study: {cfg.get('mtr_nm') or 'Motor'} — settings captured when Run study is pressed.")
        self.show_setup.setChecked(False)
        self.runs.append(result)
        self.history.addItem(f"{len(self.runs)}. {cfg.get('mtr_nm') or 'Motor'} · {result.total} cases")
        self.history.setCurrentIndex(len(self.runs) - 1)
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

    def _show_run(self, index):
        if 0 <= index < len(self.runs):
            self.result = self.runs[index]
            self.table.clearSelection()
            self.refresh()

    def _value_text(self, key, value):
        spec = INPUTS[key]
        return self._get_units().text(value, spec.quantity, 5) if spec.quantity else f"{value:.5g}"

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
        if r is None:
            return
        u = self._get_units()
        selected = {(i.row(), i.column()) for i in self.table.selectedItems()}
        self.table.blockSignals(True)
        self.table.clearContents()
        xkey, xs = r.axes[0]
        ykey, ys = r.axes[1] if len(r.axes) > 1 else (None, [None])
        rows = [(y, model) for y in ys for model in r.models]
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch if len(xs) <= 8 else QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnCount(len(xs))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([f"{INPUTS[xkey].label}\n{self._value_text(xkey, x)}" for x in xs])
        self.table.setVerticalHeaderLabels([
            (f"{INPUTS[ykey].label} {self._value_text(ykey, y)} · " if ykey else "") + MODELS[model]
            for y, model in rows])
        self._cells = {}
        label, name, quantity = OUTPUTS[self.metric.currentIndex()]
        for case in r.cases:
            col = xs.index(case.value(xkey))
            row = rows.index((case.value(ykey) if ykey else None, case.model))
            self._cells[row, col] = case
            value = getattr(case, name)
            flags = self._flags(case)
            text = (u.text(value, quantity) if quantity else f"{value:.3g} s") if math.isfinite(value) else "Not reached"
            if not math.isfinite(value) and name == "OF_liquid":
                text = "—"
            item = QTableWidgetItem(text + (" ⚠" if flags else ""))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(QColor("#803c37" if case.burnout or case.peak_P_cmbr > chamber_limit(r.cfg)
                                       else "#705e31" if flags else "#285845"))
            item.setForeground(QColor("white"))
            item.setToolTip(self._label(case) + "\n" + "\n".join(flags) + f"\nEnd: {case.end_cond}")
            self.table.setItem(row, col, item)
            item.setSelected((row, col) in selected)
        self.table.blockSignals(False)
        n = len(r.cases)
        state = "Running" if self.busy() else "Stopped" if r.stopped else "Failed" if r.error else "Complete"
        flags = sum(bool(self._flags(c)) for c in r.cases)
        self.status.setText(f"{state}: {n}/{r.total} simulations · {label} · {flags} cases with warnings. "
                            f"Pressure limit: {u.text(chamber_limit(r.cfg), 'pressure')} absolute."
                            + (f"\n{r.error}" if r.error else "")
                            + "\nO/F is oxidizer mass ÷ fuel mass during the simulated liquid phase. Hover warnings for details.")
        if n == r.total and not r.error and not self.busy():
            passing = passing_values(r.cases, xkey, self.min_pressure.si("pressure"), chamber_limit(r.cfg),
                                     self.max_dp.si("pressure") if uses_spi(r.cfg) else float("inf"))
            listed = ", ".join(self._value_text(xkey, v) for v in passing) or "none"
            self.status.setText(self.status.text() + f"\n{INPUTS[xkey].label} values completing every row within pressure/ΔP limits without fuel depletion: {listed}.")
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
            self.details.setText(f"{self._label(c)}\nPeak Pc {u.text(c.peak_P_cmbr, 'pressure')} · "
                                 f"O/F {c.OF_liquid:.3g} · impulse {u.text(c.total_impulse, 'impulse')} · "
                                 f"end: {c.end_cond}. " + "; ".join(self._flags(c)))
        else:
            self.details.setText(f"{len(cases)} selected; overlay shows up to 8. Ctrl/⌘-click cells to compare them.")

    def _open(self):
        selected = self._selected()
        if self.busy() or len(selected) != 1:
            return
        c = selected[0]
        cfg = case_cfg(self.result.cfg, c.values, c.model)
        cfg["sizing"] = {}  # open the exact built geometry, without old target sizing
        cfg["mtr_nm"] = f"{cfg.get('mtr_nm') or 'Motor'} — {self._label(c)}"
        self._on_open(cfg)

    def _save(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save reproducible study", "study.json", "Study JSON (*.json)")
        if path:
            try:
                r = self.result
                Path(path).write_text(json.dumps({"motor": r.cfg, "axes": r.axes, "models": r.models,
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
            for values in grid(axes):
                case_cfg(cfg, values, models[0])
            self._base_override = cfg
            for axis in self.axes:
                axis.cfg = cfg
            for axis, (key, values) in zip(self.axes, axes + [["", []]]):
                axis.set_axis(key, values)
            self.models.setCurrentIndex(next(i for i in range(self.models.count()) if self.models.itemData(i) == models))
            self.preset.setCurrentIndex(3)
            self.min_pressure.set_si(float(data.get("min_pressure", 0)), "pressure")
            self.max_dp.set_si(float(data.get("max_dp", to_si(300, "psi", "pressure"))), "pressure")
            self.show_setup.setChecked(True)
            self.source.setText(f"Next study uses the saved motor snapshot from {Path(path).name}. Switching motors returns to current inputs.")
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
            self.status.setText(f"Could not load study: {exc}")

    def _export(self):
        r = self.result
        if r is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export study results", "study.csv", "CSV (*.csv)")
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
