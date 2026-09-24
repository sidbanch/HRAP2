"""HRAP desktop application — PySide6 + pyqtgraph."""
from __future__ import annotations

import math
import multiprocessing
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QEvent, QObject, QSettings, QThread, Signal
from PySide6.QtGui import QFontDatabase, QIcon
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hrap import APP_NAME, __version__
from hrap.engine.nox import nox
from hrap.engine.sim import run
from hrap.engine.types import Settings, State
from hrap.engine.summary import format_summary, summarize
from hrap.gui.theme import apply_theme
from hrap.gui.sweep import SweepDialog
from hrap.gui.viz import MotorPanel, MotorView, _vent_visible
from hrap.io.config import bundled_motor, default_cfg, load_json, load_matlab_mat, resolve, resolve_layout, save_json
from hrap.io.export import export_csv, export_eng, export_rse
from hrap.io.propellant import list_propellants, load_propellant
from hrap.layout import motor_layout
from hrap.units import (
    DENSITY_ITEMS,
    DisplayUnits,
    LENGTH_ITEMS,
    MASS_ITEMS,
    PRESSURE_ITEMS,
    TEMP_ITEMS,
    VOLUME_ITEMS,
    compatible_units,
    convert,
    from_si,
    to_si,
)

# Label, output field, physical quantity. Engine arrays remain in SI units.
TRACES = [
    ("Thrust", "F_thr", "force"),
    ("Tank pressure", "P_tnk", "pressure"),
    ("Chamber pressure", "P_cmbr", "pressure"),
    ("O/F", "OF", "ratio"),
    ("Oxidizer flow", "mdot_o", "mass_flow"),
    ("Fuel flow", "mdot_f", "mass_flow"),
    ("Nozzle flow", "mdot_n", "mass_flow"),
    ("Regression rate", "rdot", "speed"),
    ("Port ID", "grn_ID", "length"),
    ("Oxidizer mass", "m_o", "mass"),
    ("Fuel mass", "m_f", "mass"),
    ("Total mass", "m_t", "mass"),
    ("CG", "cg", "length"),
]
PLOT_LABELS = {
    "force": "Thrust", "pressure": "Pressure (absolute)", "ratio": "O/F",
    "mass_flow": "Mass flow", "speed": "Regression rate", "length": "Length", "mass": "Mass",
}
DISPLAY_UNIT_OPTIONS = {
    "pressure": ["psi", "bar", "kPa", "MPa", "Pa", "atm"],
    "length": LENGTH_ITEMS,
    "mass": MASS_ITEMS,
    "force": ["N", "kN", "lbf"],
    "volume": VOLUME_ITEMS,
    "temperature": TEMP_ITEMS,
    "speed": ["m/s", "mm/s", "in/s", "ft/s"],
}


def _t_sat(P: float) -> float | None:
    """Invert N2O Wagner Pv(T) for display. None if out of range."""
    from hrap.engine.nox import TC, vapor_pressure
    from scipy.optimize import brentq

    if not math.isfinite(P) or P <= 1.0 or P >= 7.2e6:
        return None
    try:
        return cast(float, brentq(lambda T: vapor_pressure(T) - P, 183.15, TC - 0.05, xtol=1e-4))
    except Exception:
        return None


class _NoWheel:
    """Let the settings panel scroll instead of changing the focused control."""

    def wheelEvent(self, event):
        event.ignore()
        parent = cast(QWidget, self).parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                QApplication.sendEvent(parent.viewport(), event)
                return
            parent = parent.parentWidget()


class PlainSpinBox(_NoWheel, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class PlainDoubleSpinBox(_NoWheel, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class PlainComboBox(_NoWheel, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class CollapsibleBox(QWidget):
    """Settings category that can be collapsed to just its title."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsibleBox")
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._btn = QPushButton(f"▾  {title}")
        self._btn.setObjectName("collapseHeader")
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setFlat(True)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._body = QWidget()
        self._body.setObjectName("collapseBody")
        self._form = QFormLayout(self._body)
        self._form.setContentsMargins(10, 6, 10, 10)
        outer.addWidget(self._btn)
        outer.addWidget(self._body)
        self._btn.toggled.connect(self._set_open)

    def form(self) -> QFormLayout:
        return self._form

    def _set_open(self, open_: bool) -> None:
        self._body.setVisible(open_)
        mark = "▾" if open_ else "▸"
        self._btn.setText(f"{mark}  {self._title}")


class UnitRow(QWidget):
    def __init__(self, items: list[str], unit: str, decimals: int = 4, maximum: float = 1e12):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin = PlainDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(0.0, maximum)
        self.spin.setKeyboardTracking(False)
        self.unit = PlainComboBox()
        self.unit.addItems(items)
        if unit in items:
            self.unit.setCurrentText(unit)
        self.unit.setFixedWidth(72)
        self._unit = self.unit.currentText()
        self.unit.currentTextChanged.connect(self._on_unit_changed)
        layout.addWidget(self.spin, 1)
        layout.addWidget(self.unit)

    def _on_unit_changed(self, new_unit: str):
        old = self._unit
        self._unit = new_unit
        if not old or old == new_unit:
            return
        try:
            new_val = convert(self.spin.value(), old, new_unit)
        except Exception:
            return
        self.spin.blockSignals(True)
        self.spin.setValue(new_val)
        self.spin.blockSignals(False)

    def set_display(self, value: float, unit: str | None = None):
        if unit:
            self.unit.blockSignals(True)
            self.unit.setCurrentText(unit)
            self.unit.blockSignals(False)
            self._unit = self.unit.currentText()
        self.spin.blockSignals(True)
        self.spin.setValue(float(value or 0.0))
        self.spin.blockSignals(False)


def _remember_unit(combo: QComboBox, unit: str | None = None) -> None:
    combo.setProperty("hrap_unit", combo.currentText() if unit is None else unit)


def _set_unit_text(combo: QComboBox, unit: str) -> None:
    """Set a unit combo without converting the paired numeric field."""
    combo.blockSignals(True)
    combo.setCurrentText(unit)
    combo.blockSignals(False)
    _remember_unit(combo, combo.currentText())


def _wire_unit_combo(spin: QDoubleSpinBox, combo: QComboBox, group: str | None = None, should_convert=None) -> None:
    """Keep the SI quantity fixed when the user changes a standalone unit combo."""
    _remember_unit(combo)

    def _on_unit(new_unit: str):
        old = combo.property("hrap_unit") or new_unit
        _remember_unit(combo, new_unit)
        if not old or old == new_unit:
            return
        if should_convert is not None and not should_convert(old, new_unit):
            return
        if not compatible_units(old, new_unit):
            return
        try:
            new_val = convert(spin.value(), old, new_unit, group)
        except Exception:
            return
        spin.blockSignals(True)
        spin.setValue(new_val)
        spin.blockSignals(False)

    combo.currentTextChanged.connect(_on_unit)


class SimWorker(QObject):
    finished = Signal(object, object, object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            s, x = resolve(self.cfg)
            x, o = run(s, x, on_progress=self.progress.emit)
            self.finished.emit(s, x, o)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1280, 860)
        self.cfg = bundled_motor("example_98mm") if _has_bundled("example_98mm") else default_cfg()
        self._output = None
        self._settings = None
        self._state = None
        self._theme = "dark"
        self._thread = None
        self._worker = None
        self._close_when_finished = False
        self._result_cfg = None
        self._hover_index = None
        self._prefs = QSettings("HCAT", APP_NAME)
        defaults = asdict(DisplayUnits())
        saved = {key: str(self._prefs.value(f"displayUnits/{key}", default))
                 for key, default in defaults.items()}
        self.display_units = DisplayUnits(**{
            key: unit if unit in DISPLAY_UNIT_OPTIONS[key] else defaults[key]
            for key, unit in saved.items()
        })
        self._last_dir = str(self._prefs.value("lastFileDir") or "")
        self._build()
        self._cfg_to_form(self.cfg)
        self._connect_derived()
        for control_type, signal in (
            (QDoubleSpinBox, "valueChanged"), (QSpinBox, "valueChanged"),
            (QComboBox, "currentIndexChanged"), (QCheckBox, "toggled"),
        ):
            for control in self._form.findChildren(control_type):
                getattr(control, signal).connect(self._invalidate_results)
        self.name.textChanged.connect(self._invalidate_results)
        self.mfg.textChanged.connect(self._invalidate_results)

    def _build(self):
        file_menu = self._file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("New", self._new)
        file_menu.addAction("Open JSON…", self._open_json)
        file_menu.addAction("Import MATLAB .mat…", self._import_mat)
        file_menu.addAction("Save JSON…", self._save_json)
        file_menu.addSeparator()
        file_menu.addAction("Export CSV…", lambda: self._export("csv"))
        file_menu.addAction("Export RSE…", lambda: self._export("rse"))
        file_menu.addAction("Export ENG…", lambda: self._export("eng"))
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        ex_menu = self._examples_menu = self.menuBar().addMenu("&Examples")
        ex_menu.addAction("example_98mm (const O/F, ABS)", lambda: self._load_bundled("example_98mm"))
        ex_menu.addAction("Rattworks K240 (const O/F, HDPE)", lambda: self._load_bundled("Rattworks_K240"))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("Dark theme", lambda: self._set_theme("dark"))
        view_menu.addAction("Light theme", lambda: self._set_theme("light"))

        settings_menu = self.menuBar().addMenu("&Settings")
        settings_menu.addAction("Units…", self._choose_display_units)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(f"About {APP_NAME}", self._about)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._form = self._make_form()
        splitter.addWidget(self._form)
        splitter.addWidget(self._make_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 760])
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(splitter, 1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(18)
        self._progress.setFormat("Running %p%")
        self._progress.hide()
        root.addWidget(self._progress)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — MATLAB-parity engine")

    def _make_form(self) -> QWidget:
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(8, 8, 8, 8)
        wl.setSpacing(8)

        header = QWidget()
        header.setObjectName("configHeader")
        header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        header_l = QVBoxLayout(header)
        header_l.setContentsMargins(10, 8, 10, 8)
        header_l.setSpacing(6)

        run_row = QHBoxLayout()
        self.name = QLineEdit()
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        self.sweep_btn = QPushButton("Sweep…")
        self.sweep_btn.setToolTip("Run this motor across a range of throat diameters and injector Cds")
        self.sweep_btn.clicked.connect(self._open_sweep)
        run_row.addWidget(QLabel("Motor Name"))
        run_row.addWidget(self.name, 1)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.sweep_btn)
        header_l.addLayout(run_row)

        io_row = QHBoxLayout()
        self.mfg = QLineEdit("HRAP")
        save_btn = QPushButton("Save")
        load_btn = QPushButton("Load")
        save_btn.clicked.connect(self._save_json)
        load_btn.clicked.connect(self._open_json)
        io_row.addWidget(QLabel("Manufacturer"))
        io_row.addWidget(self.mfg, 1)
        io_row.addWidget(save_btn)
        io_row.addWidget(load_btn)
        header_l.addLayout(io_row)

        rse_row = QHBoxLayout()
        self.rse_btn = QPushButton("Export .RSE")
        self.rse_btn.clicked.connect(lambda: self._export("rse"))
        rse_row.addWidget(self.rse_btn, 1)
        header_l.addLayout(rse_row)
        wl.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setSpacing(8)

        self.prop_combo = PlainComboBox()
        for item in list_propellants():
            self.prop_combo.addItem(f"{item['name']} ({item['id']})", item["id"])
        self.prop_combo.currentIndexChanged.connect(self._on_propellant)

        self.tnk_V = UnitRow(VOLUME_ITEMS, "cm^3", 3)
        self.tnk_D = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_L = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_by_dims = QCheckBox("Tank volume from diameter × length")
        self.cmbr_V = UnitRow(VOLUME_ITEMS, "cm^3", 3)
        self.cmbr_by_dims = QCheckBox("Chamber volume from grain envelope")
        self.cmbr_by_dims.setChecked(True)

        tank = CollapsibleBox("Tank")
        tf = tank.form()
        tf.addRow(self.tnk_by_dims)
        tf.addRow("Volume", self.tnk_V)
        tf.addRow("Diameter", self.tnk_D)
        tf.addRow("Length", self.tnk_L)
        tf.addRow(self.cmbr_by_dims)
        tf.addRow("Chamber volume", self.cmbr_V)
        root.addWidget(tank)

        self.noz_thrt = UnitRow(LENGTH_ITEMS, "in")
        self.noz_mode = PlainComboBox()
        self.noz_mode.addItems(["Nozzle Expansion Ratio", "Nozzle Exit Diameter"])
        self.noz_ex = PlainDoubleSpinBox()
        self.noz_ex.setRange(0.0, 1e6)
        self.noz_ex.setDecimals(4)
        self.noz_ex_unit = PlainComboBox()
        self.noz_ex_unit.addItems(LENGTH_ITEMS)
        self.noz_eff = PlainDoubleSpinBox()
        self.noz_eff.setRange(0.0, 100.0)
        self.noz_eff.setValue(97.0)
        self.noz_Cd = PlainDoubleSpinBox()
        self.noz_Cd.setRange(0.0, 1.0)
        self.noz_Cd.setDecimals(3)
        self.noz_Cd.setValue(0.95)
        noz = CollapsibleBox("Nozzle")
        nf = noz.form()
        nf.addRow("Throat", self.noz_thrt)
        nf.addRow("Define by", self.noz_mode)
        ex_row = QWidget()
        ex_l = QHBoxLayout(ex_row)
        ex_l.setContentsMargins(0, 0, 0, 0)
        ex_l.addWidget(self.noz_ex, 1)
        ex_l.addWidget(self.noz_ex_unit)
        nf.addRow("Exit / ER", ex_row)
        nf.addRow("Efficiency %", self.noz_eff)
        nf.addRow("Cd", self.noz_Cd)
        root.addWidget(noz)

        self.rho = UnitRow(DENSITY_ITEMS, "kg/m^3", 4)
        self.prop_a = PlainDoubleSpinBox(); self.prop_a.setDecimals(5); self.prop_a.setRange(0, 1e3)
        self.prop_n = PlainDoubleSpinBox(); self.prop_n.setDecimals(5); self.prop_n.setRange(-2, 5)
        self.prop_m = PlainDoubleSpinBox(); self.prop_m.setDecimals(5); self.prop_m.setRange(-2, 5)
        self.const_OF = PlainDoubleSpinBox(); self.const_OF.setDecimals(4); self.const_OF.setRange(0.01, 100)
        self.cstar = PlainDoubleSpinBox(); self.cstar.setRange(0.0, 100.0); self.cstar.setValue(100.0)
        self.grn_ID = UnitRow(LENGTH_ITEMS, "in")
        self.grn_OD = UnitRow(LENGTH_ITEMS, "in")
        self.grn_L = UnitRow(LENGTH_ITEMS, "in")
        grain = CollapsibleBox("Propellant / grain")
        gf = grain.form()
        gf.addRow("Preset", self.prop_combo)
        gf.addRow("Density", self.rho)
        gf.addRow("a (mm/s)", self.prop_a)
        gf.addRow("n", self.prop_n)
        gf.addRow("m", self.prop_m)
        gf.addRow("Constant O/F", self.const_OF)
        gf.addRow("C* efficiency %", self.cstar)
        gf.addRow("Port ID", self.grn_ID)
        gf.addRow("Grain OD", self.grn_OD)
        gf.addRow("Grain length", self.grn_L)
        root.addWidget(grain)

        self.inj_D = UnitRow(LENGTH_ITEMS, "in", 5)
        self.inj_Cd = PlainDoubleSpinBox(); self.inj_Cd.setRange(0, 1); self.inj_Cd.setDecimals(4)
        self.inj_N = PlainSpinBox(); self.inj_N.setRange(1, 200)
        self.vnt_state = PlainComboBox(); self.vnt_state.addItems(["None", "External", "Internal"])
        self.vnt_D = UnitRow(LENGTH_ITEMS, "mm", 4)
        self.vnt_Cd = PlainDoubleSpinBox(); self.vnt_Cd.setRange(0, 1); self.vnt_Cd.setDecimals(3)
        inj = CollapsibleBox("Injector / vent")
        iff = inj.form()
        iff.addRow("Injector diameter", self.inj_D)
        iff.addRow("Injector Cd", self.inj_Cd)
        iff.addRow("Injector count", self.inj_N)
        self.inj_cda = QLabel("—")
        iff.addRow("Injector CdA (computed)", self.inj_cda)
        iff.addRow("Vent", self.vnt_state)
        iff.addRow("Vent diameter", self.vnt_D)
        iff.addRow("Vent Cd", self.vnt_Cd)
        root.addWidget(inj)

        self.mp_on = QCheckBox("Calculate mass properties (ENG / RSE CG)")
        self.tnk_start = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_m = UnitRow(MASS_ITEMS, "kg")
        self.cmbr_start = UnitRow(LENGTH_ITEMS, "in")
        self.cmbr_L = UnitRow(LENGTH_ITEMS, "in")
        self.cmbr_m = UnitRow(MASS_ITEMS, "kg")
        self.dry_OD = UnitRow(LENGTH_ITEMS, "in")
        self.dry_L = UnitRow(LENGTH_ITEMS, "in")
        self.mass_info = QLabel("Empty mass / CG: —")
        self.mass_info.setWordWrap(True)
        mass = CollapsibleBox("Mass properties")
        mf = mass.form()
        mf.addRow(self.mp_on)
        mf.addRow("Oxidizer tank start", self.tnk_start)
        mf.addRow("Oxidizer tank dry mass", self.tnk_m)
        mf.addRow("Chamber start", self.cmbr_start)
        mf.addRow("Chamber length (incl. injector + nozzle)", self.cmbr_L)
        mf.addRow("Chamber dry mass (no grain)", self.cmbr_m)
        mf.addRow("Motor OD (export)", self.dry_OD)
        mf.addRow("Motor length (export)", self.dry_L)
        mf.addRow(self.mass_info)
        root.addWidget(mass)
        self._legacy_mtr_m = 0.0
        self._legacy_mtr_m_unit = "kg"
        self._legacy_mtr_cg = 0.0
        self._legacy_mtr_cg_unit = "in"

        self.tmax = PlainDoubleSpinBox(); self.tmax.setRange(0.01, 120); self.tmax.setValue(10)
        self.tburn = PlainDoubleSpinBox(); self.tburn.setRange(0.0, 120)
        self.dt = PlainDoubleSpinBox(); self.dt.setDecimals(5); self.dt.setRange(1e-5, 0.1); self.dt.setValue(0.001)
        self.reg_model = PlainComboBox(); self.reg_model.addItems(["Constant OF", "Shifting OF"])
        sim = CollapsibleBox("Simulation")
        sf = sim.form()
        sf.addRow("Max run time [s]", self.tmax)
        sf.addRow("Max burn time [s] (0 = none)", self.tburn)
        sf.addRow("Timestep [s]", self.dt)
        sf.addRow("Regression model", self.reg_model)
        root.addWidget(sim)

        self.tnk_dd = PlainComboBox(); self.tnk_dd.addItems(["Starting Tank Temperature", "Starting Tank Pressure"])
        self.tnk_cond = PlainDoubleSpinBox(); self.tnk_cond.setDecimals(4); self.tnk_cond.setRange(0, 1e8); self.tnk_cond.setValue(293.15)
        self.T_tnk_unit = PlainComboBox(); self.T_tnk_unit.addItems(TEMP_ITEMS + PRESSURE_ITEMS)
        self.P_cmbr = UnitRow(PRESSURE_ITEMS, "atm")
        self.fill_dd = PlainComboBox(); self.fill_dd.addItems(["Tank Fill Percentage", "Starting Oxidizer Mass"])
        self.fill = PlainDoubleSpinBox(); self.fill.setDecimals(4); self.fill.setRange(0, 1e6); self.fill.setValue(95)
        self.fill_unit = PlainComboBox(); self.fill_unit.addItems(["%"] + MASS_ITEMS)
        self.Pa = UnitRow(PRESSURE_ITEMS, "atm")
        ic = CollapsibleBox("Initial conditions")
        icf = ic.form()
        icf.addRow("Tank specified by", self.tnk_dd)
        trow = QWidget(); tl = QHBoxLayout(trow); tl.setContentsMargins(0,0,0,0); tl.addWidget(self.tnk_cond, 1); tl.addWidget(self.T_tnk_unit)
        icf.addRow("Tank T / P", trow)
        icf.addRow("Chamber pressure", self.P_cmbr)
        icf.addRow("Oxidizer specified by", self.fill_dd)
        frow = QWidget(); fl = QHBoxLayout(frow); fl.setContentsMargins(0,0,0,0); fl.addWidget(self.fill, 1); fl.addWidget(self.fill_unit)
        icf.addRow("Fill / mass", frow)
        icf.addRow("Ambient pressure", self.Pa)
        self.sat_info = QLabel("Saturation: —")
        self.sat_info.setWordWrap(True)
        icf.addRow(self.sat_info)
        root.addWidget(ic)

        adv = CollapsibleBox("Advanced (not MATLAB-identical)")
        af = adv.form()
        self.adv_on = QCheckBox("Enable advanced options — results will not match original HRAP")
        self.live_chem = QCheckBox("Live chemistry (NASA thermo.dat Gibbs solver)")
        self.ox_fluid = PlainComboBox()
        self.ox_fluid.addItems(["N2O_legacy", "NitrousOxide (CoolProp)", "Oxygen (CoolProp)"])
        self.grain_shape = PlainComboBox()
        self.grain_shape.addItems(["cylindrical", "star"])
        self.star_tips = PlainSpinBox(); self.star_tips.setRange(3, 16); self.star_tips.setValue(6)
        self.inj_model = PlainComboBox()
        self.inj_model.addItems(["SPI", "HEM", "Dyer"])
        self.inj_model.setToolTip(
            "SPI: pure liquid through the injector (original HRAP); overpredicts flow at high ΔP.\n"
            "HEM: liquid boils instantly in the orifice; underpredicts flow and chokes. Needs CoolProp.\n"
            "Dyer: κ/(1+κ)·SPI + 1/(1+κ)·HEM. Needs CoolProp."
        )
        self.inj_Cd_HEM = PlainDoubleSpinBox(); self.inj_Cd_HEM.setRange(0, 1); self.inj_Cd_HEM.setDecimals(4)
        self.inj_Cd_HEM.setSpecialValueText("same as injector Cd")
        self.inj_Cd_HEM.setToolTip("Discharge coefficient for the HEM part. Water flow tests can't measure it; a nitrous cold flow can.")
        self.dyer_kappa = PlainDoubleSpinBox(); self.dyer_kappa.setRange(0.01, 100); self.dyer_kappa.setDecimals(2)
        self.dyer_kappa.setValue(1.0)
        self.dyer_kappa.setToolTip("Dyer weighting. 1 = the formula's value for a tank at its own vapor pressure (even blend). Larger leans toward SPI.")
        af.addRow(self.adv_on)
        af.addRow(self.live_chem)
        af.addRow("Oxidizer fluid", self.ox_fluid)
        af.addRow("Grain shape", self.grain_shape)
        af.addRow("Star tips", self.star_tips)
        af.addRow("Injector model", self.inj_model)
        af.addRow("HEM Cd", self.inj_Cd_HEM)
        af.addRow("Dyer κ", self.dyer_kappa)
        root.addWidget(adv)
        root.addStretch(1)
        scroll.setWidget(inner)
        wl.addWidget(scroll, 1)
        return wrap

    def _make_results(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        mid = QSplitter(Qt.Orientation.Horizontal)
        self.plot = QTabWidget()
        self._plots: dict[str, pg.PlotItem] = {}
        self._plot_widgets: dict[str, pg.PlotWidget] = {}
        self._hover_lines: list[pg.InfiniteLine] = []
        self._plot_viewports: list[QWidget] = []
        self._plotted_output = None
        self._syncing_time = False
        self._time_range = (0.0, 1.0)
        self.plot.currentChanged.connect(lambda _: self._reset_viz_to_start())
        self.plot_readout = QLabel("Run a simulation, then hover over a plot to inspect a time.")
        self.plot_readout.setWordWrap(True)
        self.plot_readout.setMinimumHeight(36)
        self.trace_list = QListWidget()
        for i, (label, _key, _quantity) in enumerate(TRACES):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if i < 3 or 4 <= i <= 6 else Qt.CheckState.Unchecked)
            self.trace_list.addItem(item)
        self.trace_list.itemChanged.connect(lambda *_: self._refresh_plot())
        mid.addWidget(self.trace_list)
        mid.addWidget(self.plot)
        mid.setSizes([160, 600])
        layout.addWidget(mid, 3)
        layout.addWidget(self.plot_readout)
        self.motor_panel = MotorPanel()
        self.viz = self.motor_panel.viz
        layout.addWidget(self.motor_panel)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self.summary, 2)
        return box

    def _form_to_cfg(self) -> dict:
        lay = self._form_layout()
        empty_m, empty_cg = self._empty_mass_si(lay)
        cfg = default_cfg()
        cfg.update({
            "mtr_nm": self.name.text() or "mtr_cfg",
            "tnk_V": self.tnk_V.spin.value(),
            "tnk_V_unit": self.tnk_V.unit.currentText(),
            "tnk_V_state": int(self.tnk_by_dims.isChecked()),
            "tnk_D": self.tnk_D.spin.value(),
            "tnk_D_unit": self.tnk_D.unit.currentText(),
            "tnk_L": self.tnk_L.spin.value(),
            "tnk_L_unit": self.tnk_L.unit.currentText(),
            "cmbr_V": self.cmbr_V.spin.value(),
            "cmbr_V_unit": self.cmbr_V.unit.currentText(),
            "cmbr_V_state": int(self.cmbr_by_dims.isChecked()),
            "noz_thrt": self.noz_thrt.spin.value(),
            "noz_thrt_unit": self.noz_thrt.unit.currentText(),
            "noz_def": self.noz_mode.currentText(),
            "noz_ex": self.noz_ex.value(),
            "noz_ex_unit": self.noz_ex_unit.currentText(),
            "noz_eff": self.noz_eff.value(),
            "noz_Cd": self.noz_Cd.value(),
            "mp_state": int(self.mp_on.isChecked()),
            "tnk_start": self.tnk_start.spin.value(),
            "tnk_start_unit": self.tnk_start.unit.currentText(),
            "cmbr_start": self.cmbr_start.spin.value(),
            "cmbr_start_unit": self.cmbr_start.unit.currentText(),
            "cmbr_L": self.cmbr_L.spin.value(),
            "cmbr_L_unit": self.cmbr_L.unit.currentText(),
            "tnk_m": self.tnk_m.spin.value(),
            "tnk_m_unit": self.tnk_m.unit.currentText(),
            "cmbr_m": self.cmbr_m.spin.value(),
            "cmbr_m_unit": self.cmbr_m.unit.currentText(),
            "tnk_X": from_si(lay.tnk_aft, self.tnk_start.unit.currentText(), "length"),
            "tnk_X_unit": self.tnk_start.unit.currentText(),
            "cmbr_X": from_si(lay.grain_aft, self.cmbr_start.unit.currentText(), "length"),
            "cmbr_X_unit": self.cmbr_start.unit.currentText(),
            "mtr_cg": from_si(empty_cg, self.tnk_start.unit.currentText(), "length"),
            "mtr_cg_unit": self.tnk_start.unit.currentText(),
            "mtr_m": from_si(empty_m, self.tnk_m.unit.currentText(), "mass"),
            "mtr_m_unit": self.tnk_m.unit.currentText(),
            "tnk_dd": self.tnk_dd.currentText(),
            "tnk_cond": self.tnk_cond.value(),
            "T_tnk_unit": self.T_tnk_unit.currentText(),
            "P_cmbr": self.P_cmbr.spin.value(),
            "P_cmbr_unit": self.P_cmbr.unit.currentText(),
            "fill_dd": self.fill_dd.currentText(),
            "fill": self.fill.value(),
            "fill_unit": self.fill_unit.currentText(),
            "Pa": self.Pa.spin.value(),
            "Pa_unit": self.Pa.unit.currentText(),
            "prop_id": self.prop_combo.currentData() or "ABS",
            "prop_nm": (self.prop_combo.currentText() or "ABS").split(" (")[0],
            "prop_rho": self.rho.spin.value(),
            "prop_rho_unit": self.rho.unit.currentText(),
            "prop_a": self.prop_a.value(),
            "prop_n": self.prop_n.value(),
            "prop_m": self.prop_m.value(),
            "const_OF": self.const_OF.value(),
            "cstar_eff": self.cstar.value(),
            "grn_ID": self.grn_ID.spin.value(),
            "grn_ID_unit": self.grn_ID.unit.currentText(),
            "grn_OD": self.grn_OD.spin.value(),
            "grn_OD_unit": self.grn_OD.unit.currentText(),
            "grn_L": self.grn_L.spin.value(),
            "grn_L_unit": self.grn_L.unit.currentText(),
            "inj_D": self.inj_D.spin.value(),
            "inj_D_unit": self.inj_D.unit.currentText(),
            "inj_N": self.inj_N.value(),
            "inj_Cd": self.inj_Cd.value(),
            "vnt_state": self.vnt_state.currentText(),
            "vnt_D": self.vnt_D.spin.value(),
            "vnt_D_unit": self.vnt_D.unit.currentText(),
            "vnt_Cd": self.vnt_Cd.value(),
            "t_max": self.tmax.value(),
            "t_burn": self.tburn.value(),
            "dt": self.dt.value(),
            "reg_model": self.reg_model.currentText(),
            "advanced": {
                "enabled": self.adv_on.isChecked(),
                "ox_fluid": self.ox_fluid.currentText(),
                "grain_shape": self.grain_shape.currentText(),
                "star_tips": self.star_tips.value(),
                "live_chem": self.live_chem.isChecked(),
                "inj_model": self.inj_model.currentText(),
                "inj_Cd_HEM": self.inj_Cd_HEM.value(),
                "dyer_kappa": self.dyer_kappa.value(),
            },
            "export_OD": to_si(self.dry_OD.spin.value(), self.dry_OD.unit.currentText(), "length"),
            "export_L": to_si(self.dry_L.spin.value(), self.dry_L.unit.currentText(), "length"),
            "mfg": self.mfg.text() or "HRAP",
        })
        return cfg

    def _cfg_to_form(self, cfg: dict):
        self.name.setText(str(cfg.get("mtr_nm") or ""))
        self.mfg.setText(str(cfg.get("mfg") or "HRAP"))
        self.tnk_V.set_display(cfg.get("tnk_V", 0), cfg.get("tnk_V_unit", "cm^3"))
        self.tnk_D.set_display(cfg.get("tnk_D", 0), cfg.get("tnk_D_unit", "in"))
        self.tnk_L.set_display(cfg.get("tnk_L", 0), cfg.get("tnk_L_unit", "in"))
        self.tnk_by_dims.setChecked(bool(cfg.get("tnk_V_state")))
        self.cmbr_V.set_display(cfg.get("cmbr_V", 0), cfg.get("cmbr_V_unit", "cm^3"))
        self.cmbr_by_dims.setChecked(bool(cfg.get("cmbr_V_state", 1)))
        self.noz_thrt.set_display(cfg.get("noz_thrt", 0), cfg.get("noz_thrt_unit", "in"))
        self.noz_mode.setCurrentText(cfg.get("noz_def") or "Nozzle Expansion Ratio")
        self.noz_ex.setValue(float(cfg.get("noz_ex") or 0))
        _set_unit_text(self.noz_ex_unit, cfg.get("noz_ex_unit") or "in")
        self.noz_eff.setValue(float(cfg.get("noz_eff") or 100))
        self.noz_Cd.setValue(float(cfg.get("noz_Cd") or 1))
        self.mp_on.setChecked(bool(cfg.get("mp_state")))
        self._legacy_mtr_m = float(cfg.get("mtr_m") or 0.0)
        self._legacy_mtr_m_unit = str(cfg.get("mtr_m_unit") or "kg")
        self._legacy_mtr_cg = float(cfg.get("mtr_cg") or 0.0)
        self._legacy_mtr_cg_unit = str(cfg.get("mtr_cg_unit") or "in")
        lay = resolve_layout(cfg)
        self.tnk_start.set_display(from_si(lay.tnk0, cfg.get("tnk_start_unit") or "in", "length"), cfg.get("tnk_start_unit") or "in")
        self.cmbr_start.set_display(from_si(lay.cmbr0, cfg.get("cmbr_start_unit") or "in", "length"), cfg.get("cmbr_start_unit") or "in")
        self.cmbr_L.set_display(from_si(lay.cmbr_L, cfg.get("cmbr_L_unit") or "in", "length"), cfg.get("cmbr_L_unit") or "in")
        self.tnk_m.set_display(cfg.get("tnk_m", 0), cfg.get("tnk_m_unit", "kg"))
        self.cmbr_m.set_display(cfg.get("cmbr_m", 0), cfg.get("cmbr_m_unit", "kg"))
        self.tnk_dd.setCurrentText(cfg.get("tnk_dd") or "Starting Tank Temperature")
        self.tnk_cond.setValue(float(cfg.get("tnk_cond") or 0))
        _set_unit_text(self.T_tnk_unit, cfg.get("T_tnk_unit") or "K")
        self.P_cmbr.set_display(cfg.get("P_cmbr", 1), cfg.get("P_cmbr_unit", "atm"))
        self.fill_dd.setCurrentText(cfg.get("fill_dd") or "Tank Fill Percentage")
        self.fill.setValue(float(cfg.get("fill") or 0))
        fu = cfg.get("fill_unit") or "%"
        if fu not in [self.fill_unit.itemText(i) for i in range(self.fill_unit.count())]:
            fu = "%"
        _set_unit_text(
            self.fill_unit,
            fu if self.fill_dd.currentText() == "Tank Fill Percentage" else (cfg.get("fill_unit") or "kg"),
        )
        self.Pa.set_display(cfg.get("Pa", 1), cfg.get("Pa_unit", "atm"))
        pid = cfg.get("prop_id") or cfg.get("prop_nm") or "ABS"
        idx = self.prop_combo.findData(pid)
        if idx < 0:
            for i in range(self.prop_combo.count()):
                if pid.lower() in str(self.prop_combo.itemText(i)).lower() or pid.lower() in str(self.prop_combo.itemData(i)).lower():
                    idx = i
                    break
        if idx >= 0:
            self.prop_combo.setCurrentIndex(idx)
        self.rho.set_display(cfg.get("prop_rho", 1000), cfg.get("prop_rho_unit", "kg/m^3"))
        self.prop_a.setValue(float(cfg.get("prop_a") or 0))
        self.prop_n.setValue(float(cfg.get("prop_n") or 0))
        self.prop_m.setValue(float(cfg.get("prop_m") or 0))
        self.const_OF.setValue(float(cfg.get("const_OF") or 1))
        self.cstar.setValue(float(cfg.get("cstar_eff") or 100))
        self.grn_ID.set_display(cfg.get("grn_ID", 0), cfg.get("grn_ID_unit", "in"))
        self.grn_OD.set_display(cfg.get("grn_OD", 0), cfg.get("grn_OD_unit", "in"))
        self.grn_L.set_display(cfg.get("grn_L", 0), cfg.get("grn_L_unit", "in"))
        self.inj_D.set_display(cfg.get("inj_D", 0), cfg.get("inj_D_unit", "in"))
        self.inj_Cd.setValue(float(cfg.get("inj_Cd") or 1))
        self.inj_N.setValue(int(cfg.get("inj_N") or 1))
        self.vnt_state.setCurrentText(str(cfg.get("vnt_state") or "None"))
        self.vnt_D.set_display(cfg.get("vnt_D", 0), cfg.get("vnt_D_unit", "mm"))
        self.vnt_Cd.setValue(float(cfg.get("vnt_Cd") or 0))
        self.tmax.setValue(float(cfg.get("t_max") or 10))
        self.tburn.setValue(float(cfg.get("t_burn") or 0))
        self.dt.setValue(float(cfg.get("dt") or 0.001))
        self.reg_model.setCurrentText(cfg.get("reg_model") or "Constant OF")
        self.dry_OD.set_display(from_si(cfg.get("export_OD") or lay.overall_OD, "in", "length"), "in")
        self.dry_L.set_display(from_si(cfg.get("export_L") or lay.overall_L, "in", "length"), "in")
        adv = cfg.get("advanced") or {}
        self.adv_on.setChecked(bool(adv.get("enabled")))
        if adv.get("ox_fluid"):
            self.ox_fluid.setCurrentText(str(adv["ox_fluid"]))
        if adv.get("grain_shape"):
            self.grain_shape.setCurrentText(str(adv["grain_shape"]))
        if adv.get("star_tips"):
            self.star_tips.setValue(int(adv["star_tips"]))
        self.live_chem.setChecked(bool(adv.get("live_chem")))
        self.inj_model.setCurrentText(str(adv.get("inj_model") or "SPI"))
        self.inj_Cd_HEM.setValue(float(adv.get("inj_Cd_HEM") or 0.0))
        self.dyer_kappa.setValue(float(adv.get("dyer_kappa") or 1.0))
        self._update_derived_labels()

    def _connect_derived(self):
        self._wire_standalone_units()
        for w in (
            self.inj_D.spin, self.inj_Cd, self.inj_N,
            self.tnk_cond, self.fill, self.tnk_V.spin, self.tnk_D.spin, self.tnk_L.spin,
            self.grn_ID.spin, self.grn_OD.spin, self.grn_L.spin, self.rho.spin, self.const_OF,
            self.noz_thrt.spin, self.noz_ex, self.vnt_D.spin, self.vnt_Cd, self.P_cmbr.spin,
            self.tnk_start.spin, self.tnk_m.spin, self.cmbr_start.spin, self.cmbr_L.spin, self.cmbr_m.spin,
        ):
            w.valueChanged.connect(self._update_derived_labels)
        self.rho.unit.currentTextChanged.connect(self._update_derived_labels)
        self.P_cmbr.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_dd.currentTextChanged.connect(self._update_derived_labels)
        self.fill_dd.currentTextChanged.connect(self._update_derived_labels)
        self.T_tnk_unit.currentTextChanged.connect(self._update_derived_labels)
        self.fill_unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_V.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_D.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_L.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_start.unit.currentTextChanged.connect(self._update_derived_labels)
        self.cmbr_start.unit.currentTextChanged.connect(self._update_derived_labels)
        self.cmbr_L.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_m.unit.currentTextChanged.connect(self._update_derived_labels)
        self.cmbr_m.unit.currentTextChanged.connect(self._update_derived_labels)
        self.inj_D.unit.currentTextChanged.connect(self._update_derived_labels)
        self.grn_ID.unit.currentTextChanged.connect(self._update_derived_labels)
        self.grn_OD.unit.currentTextChanged.connect(self._update_derived_labels)
        self.grn_L.unit.currentTextChanged.connect(self._update_derived_labels)
        self.noz_thrt.unit.currentTextChanged.connect(self._update_derived_labels)
        self.noz_ex_unit.currentTextChanged.connect(self._update_derived_labels)
        self.vnt_D.unit.currentTextChanged.connect(self._update_derived_labels)
        self.tnk_by_dims.toggled.connect(self._update_derived_labels)
        self.noz_mode.currentTextChanged.connect(self._update_derived_labels)
        self.vnt_state.currentTextChanged.connect(self._update_derived_labels)
        self.name.textChanged.connect(self._update_derived_labels)

    def _wire_standalone_units(self):
        _wire_unit_combo(
            self.noz_ex,
            self.noz_ex_unit,
            "length",
            should_convert=lambda _o, _n: "Expansion" not in self.noz_mode.currentText(),
        )
        _wire_unit_combo(self.tnk_cond, self.T_tnk_unit)
        _wire_unit_combo(self.fill, self.fill_unit)

    def _update_derived_labels(self):
        D = to_si(self.inj_D.spin.value(), self.inj_D.unit.currentText(), "length")
        cda = 0.25 * math.pi * D ** 2 * self.inj_Cd.value() * self.inj_N.value()
        self.inj_cda.setText(self.display_units.text(cda, "area"))
        if self.tnk_by_dims.isChecked():
            d = to_si(self.tnk_D.spin.value(), self.tnk_D.unit.currentText(), "length")
            L = to_si(self.tnk_L.spin.value(), self.tnk_L.unit.currentText(), "length")
            V = L * 0.25 * math.pi * d ** 2
            self.tnk_V.set_display(from_si(V, self.tnk_V.unit.currentText(), "volume"))
        try:
            if self.tnk_dd.currentText() == "Starting Tank Temperature":
                T = to_si(self.tnk_cond.value(), self.T_tnk_unit.currentText(), "temperature")
            else:
                P = to_si(self.tnk_cond.value(), self.T_tnk_unit.currentText(), "pressure")
                from hrap.engine.fzero import matlab_fzero
                from hrap.engine.nox import vapor_pressure
                T = matlab_fzero(lambda t: vapor_pressure(t) - P, 273.15)
            ox = nox(T)
            if self.tnk_by_dims.isChecked():
                d = to_si(self.tnk_D.spin.value(), self.tnk_D.unit.currentText(), "length")
                L = to_si(self.tnk_L.spin.value(), self.tnk_L.unit.currentText(), "length")
                V = L * 0.25 * math.pi * d ** 2
            else:
                V = to_si(self.tnk_V.spin.value(), self.tnk_V.unit.currentText(), "volume")
            if self.fill_dd.currentText() == "Tank Fill Percentage":
                fill = self.fill.value() / 100.0
                m_o = fill * V * ox.rho_l + (1.0 - fill) * V * ox.rho_v
            else:
                unit = self.fill_unit.currentText()
                m_o = self.fill.value() * (1.0 if unit == "%" else to_si(1.0, unit, "mass"))
                if unit == "%":
                    fill = self.fill.value() / 100.0
                    m_o = fill * V * ox.rho_l + (1.0 - fill) * V * ox.rho_v
            self.sat_info.setText(
                f"Saturation: T={self.display_units.text(T, 'temperature')}, "
                f"P={self.display_units.text(ox.Pv, 'pressure')} (absolute), "
                f"ox mass={self.display_units.text(m_o, 'mass')}"
            )
        except Exception:
            self.sat_info.setText("Saturation: (out of N2O fit range)")
        try:
            lay = self._form_layout()
            empty_m, empty_cg = self._empty_mass_si(lay)
            unit = self.tnk_start.unit.currentText()
            self.mass_info.setText(
                f"Empty mass {self.display_units.text(empty_m, 'mass')} at CG {self.display_units.text(empty_cg, 'length')}. "
                f"Overall length {self.display_units.text(lay.overall_L, 'length')} "
                f"(tank L {self.display_units.text(lay.tnk_L, 'length')})."
            )
            if not self.dry_OD.spin.hasFocus():
                self.dry_OD.set_display(from_si(lay.overall_OD, self.dry_OD.unit.currentText(), "length"))
            if not self.dry_L.spin.hasFocus():
                self.dry_L.set_display(from_si(lay.overall_L, self.dry_L.unit.currentText(), "length"))
        except Exception:
            self.mass_info.setText("Empty mass / CG: —")
        self._refresh_viz()

    def _len_si(self, row: UnitRow) -> float:
        return to_si(row.spin.value(), row.unit.currentText(), "length")

    def _tank_geometry(self) -> tuple[float, float, float]:
        d = self._len_si(self.tnk_D)
        if d <= 0:
            d = self._len_si(self.grn_OD) or 0.05
        if self.tnk_by_dims.isChecked():
            L = self._len_si(self.tnk_L)
            V = L * 0.25 * math.pi * d ** 2
        else:
            V = to_si(self.tnk_V.spin.value(), self.tnk_V.unit.currentText(), "volume")
            L = self._len_si(self.tnk_L)
            if L <= 1e-9 and d > 0 and V > 0:
                L = V / (0.25 * math.pi * d ** 2)
        return max(L, 1e-4), max(d, 1e-4), max(V, 0.0)

    def _form_layout(self):
        tnk_L, _tnk_D, tnk_V = self._tank_geometry()
        if tnk_L <= 1e-9:
            tnk_L = tnk_V / (0.25 * math.pi * max(_tnk_D, 1e-9) ** 2)
        th = self._len_si(self.noz_thrt)
        exit_d, _er = self._nozzle_exit()
        return motor_layout(
            tnk_start=self._len_si(self.tnk_start),
            tnk_L=tnk_L,
            tnk_m=to_si(self.tnk_m.spin.value(), self.tnk_m.unit.currentText(), "mass"),
            tnk_D=_tnk_D,
            cmbr_start=self._len_si(self.cmbr_start),
            cmbr_L=self._len_si(self.cmbr_L),
            cmbr_m=to_si(self.cmbr_m.spin.value(), self.cmbr_m.unit.currentText(), "mass"),
            grn_L=self._len_si(self.grn_L),
            grn_OD=self._len_si(self.grn_OD),
            noz_thrt=th,
            noz_exit=exit_d,
        )

    def _empty_mass_si(self, lay=None) -> tuple[float, float]:
        lay = lay or self._form_layout()
        if lay.dry_mass > 0.0:
            return lay.dry_mass, lay.dry_cg
        return (
            to_si(self._legacy_mtr_m, self._legacy_mtr_m_unit, "mass"),
            to_si(self._legacy_mtr_cg, self._legacy_mtr_cg_unit, "length"),
        )

    def _nozzle_exit(self) -> tuple[float, float]:
        th = self._len_si(self.noz_thrt)
        if "Expansion" in self.noz_mode.currentText():
            er = max(float(self.noz_ex.value()), 1e-9)
            return th * math.sqrt(er), er
        exit_d = to_si(self.noz_ex.value(), self.noz_ex_unit.currentText(), "length")
        er = (exit_d / th) ** 2 if th > 0 else 1.0
        return exit_d, er

    def _initial_fill_and_T(self) -> tuple[float, float, float]:
        """Return (fill fraction, tank T [K], oxidizer mass [kg]) from the form."""
        L, d, V = self._tank_geometry()
        try:
            if self.tnk_dd.currentText() == "Starting Tank Temperature":
                T = to_si(self.tnk_cond.value(), self.T_tnk_unit.currentText(), "temperature")
            else:
                P = to_si(self.tnk_cond.value(), self.T_tnk_unit.currentText(), "pressure")
                T = _t_sat(P) or 293.15
            ox = nox(T)
        except Exception:
            T, ox = 293.15, None
        if self.fill_dd.currentText() == "Tank Fill Percentage" or self.fill_unit.currentText() == "%":
            fill = self.fill.value() / 100.0
            m_o = 0.0
            if ox is not None:
                m_o = fill * V * ox.rho_l + (1.0 - fill) * V * ox.rho_v
        else:
            m_o = to_si(self.fill.value(), self.fill_unit.currentText(), "mass")
            fill = 0.0
            if ox is not None and V > 0 and ox.rho_l > 0:
                fill = min(max(m_o / (ox.rho_l * V), 0.0), 1.0)
        return fill, T, m_o

    def _motor_view(self, index: int | None = None) -> MotorView:
        tnk_L, tnk_D, tnk_V = self._tank_geometry()
        grn_L = self._len_si(self.grn_L)
        grn_OD = self._len_si(self.grn_OD)
        grn_ID = self._len_si(self.grn_ID)
        inj_D = self._len_si(self.inj_D)
        inj_N = int(self.inj_N.value())
        inj_Cd = float(self.inj_Cd.value())
        inj_A = 0.25 * math.pi * inj_D ** 2 * inj_N
        vnt = self.vnt_state.currentText()
        vnt_D = self._len_si(self.vnt_D)
        th = self._len_si(self.noz_thrt)
        exit_d, er = self._nozzle_exit()
        fill0, T0, m0 = self._initial_fill_and_T()
        rho = to_si(self.rho.spin.value(), self.rho.unit.currentText(), "density")
        m_f0 = max(0.25 * math.pi * max(grn_OD ** 2 - grn_ID ** 2, 0.0) * rho * grn_L, 0.0)
        fill = fill0
        T = T0
        m_o = m0
        m_f = m_f0
        ox_mdot = 0.0
        fuel_mdot = 0.0
        noz_mdot = 0.0
        of_ratio = float(self.const_OF.value())
        thrust = 0.0
        time_s = None
        P_tnk = 0.0
        try:
            P_tnk = float(nox(T0).Pv)
        except Exception:
            pass
        P_cmbr = to_si(self.P_cmbr.spin.value(), self.P_cmbr.unit.currentText(), "pressure")

        o = self._output
        if o is not None and o.t.size and index is not None:
            i = int(np.clip(index, 0, o.t.size - 1))
            self._hover_index = i
            time_s = float(o.t[i])
            P_tnk = float(o.P_tnk[i])
            m_o = float(o.m_o[i])
            T = _t_sat(P_tnk) or T0
            m_init = float(o.m_o[0]) if float(o.m_o[0]) > 0 else m0 or 1.0
            fill = fill0 * (m_o / m_init) if m_init else fill0
            grn_ID = float(o.grn_ID[i])
            m_f = float(o.m_f[i])
            ox_mdot = float(o.mdot_o[i])
            fuel_mdot = float(o.mdot_f[i])
            noz_mdot = float(o.mdot_n[i])
            of_ratio = float(o.OF[i])
            thrust = float(o.F_thr[i])
            P_cmbr = float(o.P_cmbr[i])

        u = self.display_units
        def size(diameter: float, length: float) -> str:
            return f"Ø{u.value(diameter, 'length'):.3g} × {u.text(length, 'length', 3)}"

        mass_pct = 100.0 * m_f / m_f0 if m_f0 > 1e-12 else 0.0
        dp_inj = P_tnk - P_cmbr
        stiff = (dp_inj / P_cmbr) if P_cmbr > 1e-9 else 0.0
        vnt_on = _vent_visible(vnt, vnt_D)
        vent_lines = ("Orifice Vent", f"Ø{u.text(vnt_D, 'length', 3)}") if vnt_on else ()
        tank_lines = (
            "Oxidizer Tank",
            f"size: {size(tnk_D, tnk_L)}",
            f"volume: {u.text(tnk_V, 'volume')}",
            f"mass = {u.text(m_o, 'mass')} ({100.0 * fill:.0f}%)",
            f"T = {u.text(T, 'temperature')}",
        )
        inj_lines = (
            "Injectors",
            f"{inj_N} × Ø{u.text(inj_D, 'length', 3)}",
            f"Cd: {inj_Cd:.2f}",
            f"A: {u.text(inj_A, 'area')}",
            f"ox flow = {u.text(ox_mdot, 'mass_flow', 3)}",
            f"ΔP = {u.text(dp_inj, 'pressure')}",
            f"stiffness = {100.0 * stiff:.0f}%",
        )
        grain_lines = (
            "Fuel Grain",
            f"size: {size(grn_OD, grn_L)}",
            f"port = {u.text(grn_ID, 'length')}",
            f"mass = {u.text(m_f, 'mass')} ({mass_pct:.0f}%)",
            f"fuel flow = {u.text(fuel_mdot, 'mass_flow', 3)}",
        )
        noz_lines = (
            "Nozzle",
            f"throat: Ø{u.text(th, 'length')}",
            f"exit: Ø{u.text(exit_d, 'length')}",
            f"expansion ratio: {er:.2f}",
            f"O/F = {of_ratio:.2f}",
            f"flow = {u.text(noz_mdot, 'mass_flow', 3)}",
            f"thrust = {u.text(thrust, 'force')}",
        )

        lay = self._form_layout()
        return MotorView(
            tnk_L=tnk_L,
            tnk_D=tnk_D,
            grn_L=grn_L,
            grn_OD=grn_OD,
            grn_ID=grn_ID,
            inj_D=inj_D,
            inj_N=inj_N,
            vnt_state=vnt,
            vnt_D=vnt_D,
            noz_thrt=th,
            noz_exit=exit_d,
            fill_frac=fill,
            name=self.name.text().strip() or "motor",
            tnk_start=lay.tnk0,
            cmbr_start=lay.cmbr0,
            cmbr_L=lay.cmbr_L,
            tnk_dry_kg=lay.tnk_m,
            cmbr_dry_kg=lay.cmbr_m,
            tank_lines=tank_lines,
            inj_lines=inj_lines,
            grain_lines=grain_lines,
            noz_lines=noz_lines,
            vent_lines=vent_lines,
            overlay_tank=(
                f"{100.0 * fill:.0f}%",
                u.text(P_tnk, "pressure"),
            ),
            overlay_grain=(
                f"{mass_pct:.0f}%",
                u.text(P_cmbr, "pressure"),
            ),
            time_s=time_s,
            display_units=u,
        )

    def _refresh_viz(self):
        if not hasattr(self, "motor_panel"):
            return
        self.motor_panel.set_model(self._motor_view(self._hover_index))

    def _reset_viz_to_start(self):
        for line in self._hover_lines:
            line.setVisible(False)
        self.plot_readout.setText("Hover over a plot to inspect a time.")
        if self._hover_index is None:
            return
        self._hover_index = None
        self._refresh_viz()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Leave and obj in self._plot_viewports:
            self._reset_viz_to_start()
        return super().eventFilter(obj, event)

    def _mouse_moved(self, quantity: str, pos):
        if self._output is None:
            return
        plot = self._plots[quantity]
        vb = cast(pg.ViewBox, plot.vb)
        if not vb.sceneBoundingRect().contains(pos):
            self._reset_viz_to_start()
            return
        x = vb.mapSceneToView(pos).x()
        t = np.asarray(self._output.t)
        if t.size == 0:
            return
        i = int(np.clip(np.searchsorted(t, x), 0, t.size - 1))
        for line in self._hover_lines:
            line.setValue(t[i])
            line.setVisible(True)
        bits = [f"Time: {t[i]:.3f} s"]
        for k, (label, key, trace_quantity) in enumerate(TRACES):
            if trace_quantity != quantity or self.trace_list.item(k).checkState() != Qt.CheckState.Checked:
                continue
            value = float(getattr(self._output, key)[i])
            bits.append(f"{label}: {self.display_units.text(value, quantity)}")
        self.plot_readout.setText("   ·   ".join(bits))
        self._refresh_viz_at(i)

    def _refresh_viz_at(self, index: int | None):
        self._hover_index = index
        self._refresh_viz()

    def _load_bundled(self, name: str):
        self._output = None
        self._hover_index = None
        self.cfg = bundled_motor(name)
        self._cfg_to_form(self.cfg)
        self.summary.clear()
        self._clear_plot()
        self._refresh_viz()

    def _invalidate_results(self):
        if self._output is None:
            return
        self._output = self._settings = self._state = self._result_cfg = None
        self._hover_index = None
        self.summary.clear()
        self._clear_plot()
        self._refresh_viz()
        self.statusBar().showMessage("Inputs changed — run again to update results.")

    def _on_propellant(self):
        ident = self.prop_combo.currentData()
        if not ident:
            return
        try:
            p = load_propellant(ident)
        except FileNotFoundError:
            return
        self.rho.set_display(p.rho, "kg/m^3")
        self.prop_a.setValue(float(p.reg[0]))
        self.prop_n.setValue(float(p.reg[1]))
        self.prop_m.setValue(float(p.reg[2]) if p.reg.size > 2 else 0.0)
        if p.opt_OF:
            self.const_OF.setValue(float(p.opt_OF))

    def _run(self):
        if self._thread is not None:
            return
        self.cfg = self._form_to_cfg()
        self._running_cfg = self.cfg
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.show()
        self._set_running(True)
        self.statusBar().showMessage("Running simulation…")
        self._thread = QThread()
        self._worker = SimWorker(self.cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    def _open_sweep(self):
        SweepDialog(self._form_to_cfg(), self.display_units, self).exec()

    def _set_running(self, running: bool):
        self._form.setEnabled(not running)
        self.sweep_btn.setEnabled(not running)
        self._file_menu.setEnabled(not running)
        self._examples_menu.setEnabled(not running)

    def _thread_finished(self):
        thread = cast(QThread, self._thread)
        # finished can arrive before deferred worker deletion has completed.
        # Join here (after finished) before Qt destroys the thread object.
        thread.wait()
        thread.deleteLater()
        self._thread = None
        self._worker = None
        self._set_running(False)
        if self._close_when_finished:
            self.close()

    def closeEvent(self, event):
        if self._thread is not None:
            self._close_when_finished = True
            self.statusBar().showMessage("Closing when the current simulation finishes…")
            event.ignore()
        else:
            event.accept()

    def _on_progress(self, i: int, n: int):
        n = max(int(n), 1)
        self._progress.setRange(0, n)
        self._progress.setValue(min(int(i), n))
        self.statusBar().showMessage(f"Running simulation… {100.0 * i / n:.0f}%")

    def _stop_progress(self):
        self._progress.hide()
        self._progress.setValue(0)

    def _on_finished(self, s, x, o):
        self._settings, self._state, self._output = s, x, o
        self._result_cfg = self._running_cfg
        self._hover_index = None
        info = summarize(s, x, o)
        self.summary.setPlainText(format_summary(info, self.display_units))
        self._refresh_plot()
        self._refresh_viz()
        self._stop_progress()
        self.statusBar().showMessage(f"Done — {o.sim_end_cond}  Total impulse: {self.display_units.text(info['total_impulse'], 'impulse')}")

    def _on_failed(self, msg: str):
        self._stop_progress()
        self.statusBar().showMessage("Simulation failed")
        QMessageBox.critical(self, "Simulation error", msg)

    def _hover_line_pen(self):
        accent = "#5b8def" if getattr(self, "_theme", "dark") != "light" else "#2f5fbf"
        return pg.mkPen(accent, width=1, style=Qt.PenStyle.DashLine)

    def _clear_plot(self):
        self._plots.clear()
        self._hover_lines.clear()
        self._plot_viewports.clear()
        self._plotted_output = None
        self.plot.clear()
        for widget in self._plot_widgets.values():
            widget.deleteLater()
        self._plot_widgets.clear()
        self.plot_readout.setText("Run a simulation, then hover over a plot to inspect a time.")

    def _sync_time_range(self, _view, time_range):
        if self._syncing_time:
            return
        self._time_range = tuple(time_range)
        self._syncing_time = True
        try:
            for plot in self._plots.values():
                cast(pg.ViewBox, plot.vb).setXRange(*self._time_range, padding=0)
        finally:
            self._syncing_time = False

    def _refresh_plot(self):
        o = self._output
        if o is None:
            return
        current = self.plot.currentWidget()
        active_quantity = next((q for q, widget in self._plot_widgets.items() if widget is current), None)
        if self._plotted_output is not o:
            self._time_range = (float(o.t[0]), float(o.t[-1]))
        self._clear_plot()
        palette = ["#5b8def", "#f0c14b", "#e06c75", "#98c379", "#c678dd", "#56b6c2", "#d19a66", "#abb2bf"]
        for i, (label, key, quantity) in enumerate(TRACES):
            if self.trace_list.item(i).checkState() != Qt.CheckState.Checked:
                continue
            unit = self.display_units.unit(quantity)
            if quantity not in self._plots:
                widget = pg.PlotWidget()
                plot = cast(pg.PlotItem, widget.getPlotItem())
                plot.showGrid(x=True, y=True, alpha=0.25)
                plot.addLegend(offset=(-10, 5))
                plot.setLabel("left", PLOT_LABELS[quantity], units=unit)
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.getAxis("left").setWidth(100)
                plot.setLabel("bottom", "Time", units="s")
                self._plots[quantity] = plot
                self._plot_widgets[quantity] = widget
                self.plot.addTab(widget, PLOT_LABELS[quantity].replace(" (absolute)", ""))
                viewport = widget.viewport()
                self._plot_viewports.append(viewport)
                viewport.installEventFilter(self)
                cast(pg.GraphicsScene, widget.scene()).sigMouseMoved.connect(
                    lambda pos, q=quantity: self._mouse_moved(q, pos))
                line = pg.InfiniteLine(angle=90, movable=False, pen=self._hover_line_pen())
                line.setVisible(False)
                plot.addItem(line, ignoreBounds=True)
                self._hover_lines.append(line)
            pen = pg.mkPen(palette[i % len(palette)], width=2)
            self._plots[quantity].plot(o.t, np.asarray(getattr(o, key), dtype=float) * self.display_units.value(1.0, quantity),
                                      pen=pen, name=label)
        for plot in self._plots.values():
            vb = cast(pg.ViewBox, plot.vb)
            vb.setXRange(*self._time_range, padding=0)
            vb.sigXRangeChanged.connect(self._sync_time_range)
        if active_quantity in self._plot_widgets:
            self.plot.setCurrentWidget(self._plot_widgets[active_quantity])
        self._plotted_output = o
        self.plot_readout.setText("Hover over a plot to inspect a time." if self._plots
                                 else "Select a quantity from the list to display its plot.")
        self._style_plots()

    def _style_plots(self):
        dark = getattr(self, "_theme", "dark") == "dark"
        fg = "#e6e8ee" if dark else "#1b1d21"
        for widget in self._plot_widgets.values():
            widget.setBackground("#1a1d23" if dark else "#ffffff")
        for plot in self._plots.values():
            for side in ("left", "bottom"):
                axis = plot.getAxis(side)
                axis.setPen(fg)
                axis.setTextPen(fg)
                axis.setLabel(text=axis.labelText, units=axis.labelUnits, **{"color": fg})
            cast(pg.LegendItem, plot.legend).setLabelTextColor(fg)
        for line in self._hover_lines:
            line.setPen(self._hover_line_pen())

    def _new(self):
        self._output = None
        self._hover_index = None
        self.cfg = default_cfg()
        self._cfg_to_form(self.cfg)
        self.summary.clear()
        self._clear_plot()
        self._refresh_viz()

    def _dialog_dir(self) -> str:
        if self._last_dir and Path(self._last_dir).is_dir():
            return self._last_dir
        return ""

    def _dialog_path(self, filename: str = "") -> str:
        folder = self._dialog_dir()
        if not folder:
            return filename
        return str(Path(folder) / filename) if filename else folder

    def _remember_file_dir(self, path: str) -> None:
        folder = Path(path)
        if not folder.is_dir():
            folder = folder.parent
        if folder.is_dir():
            self._last_dir = str(folder)
            self._prefs.setValue("lastFileDir", self._last_dir)

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open motor", self._dialog_path(), "HRAP JSON (*.json)")
        if path:
            self._remember_file_dir(path)
            self._output = None
            self._hover_index = None
            self.cfg = load_json(path)
            self._cfg_to_form(self.cfg)
            self.summary.clear()
            self._clear_plot()

    def _import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MATLAB motor", self._dialog_path(), "MATLAB (*.mat)")
        if path:
            self._remember_file_dir(path)
            self._output = None
            self._hover_index = None
            self.cfg = load_matlab_mat(path)
            self._cfg_to_form(self.cfg)
            self.summary.clear()
            self._clear_plot()

    def _save_json(self):
        suggested = (self.name.text() or "motor") + ".json"
        path, _ = QFileDialog.getSaveFileName(self, "Save motor", self._dialog_path(suggested), "HRAP JSON (*.json)")
        if path:
            self._remember_file_dir(path)
            save_json(path, self._form_to_cfg())

    def _export(self, kind: str):
        if self._output is None or self._settings is None:
            QMessageBox.information(self, "Export", "Run a simulation first.")
            return
        cfg = cast(dict, self._result_cfg)
        if kind == "csv":
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", self._dialog_path("HRAP_output.csv"), "CSV (*.csv)")
            if path:
                self._remember_file_dir(path)
                export_csv(path, self._output, self._settings)
        elif kind == "rse":
            stem = cfg["mtr_nm"].strip() or "motor"
            path, _ = QFileDialog.getSaveFileName(self, "Export RSE", self._dialog_path(f"{stem}.rse"), "RSE (*.rse)")
            if path:
                self._remember_file_dir(path)
                export_rse(
                    path,
                    self._output,
                    self._settings,
                    OD=cfg["export_OD"],
                    L=cfg["export_L"],
                    mfg=cfg["mfg"],
                )
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export ENG", self._dialog_path("motor.eng"), "ENG (*.eng)")
            if path:
                self._remember_file_dir(path)
                export_eng(
                    path,
                    self._output,
                    self._settings,
                    OD=cfg["export_OD"],
                    L=cfg["export_L"],
                    mfg=cfg["mfg"],
                )

    def _choose_display_units(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Display units")
        layout = QVBoxLayout(dialog)
        note = QLabel("Used in plots, the motor diagram, and results.\n"
                      "Input fields keep their own labeled unit selectors.\n"
                      "Pressure is absolute; injector ΔP is a pressure difference.")
        layout.addWidget(note)
        form = QFormLayout()
        controls = {}
        for quantity, choices in DISPLAY_UNIT_OPTIONS.items():
            combo = QComboBox()
            combo.addItems(choices)
            combo.setCurrentText(getattr(self.display_units, quantity))
            controls[quantity] = combo
            form.addRow(quantity.capitalize(), combo)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_display_units(DisplayUnits(**{key: combo.currentText() for key, combo in controls.items()}))

    def _set_display_units(self, units: DisplayUnits):
        self.display_units = units
        for key, unit in asdict(units).items():
            self._prefs.setValue(f"displayUnits/{key}", unit)
        self._refresh_plot()
        self._update_derived_labels()
        if self._output is not None:
            info = summarize(cast(Settings, self._settings), cast(State, self._state), self._output)
            self.summary.setPlainText(format_summary(info, units))
            self.statusBar().showMessage(f"Done — {self._output.sim_end_cond}  Total impulse: {units.text(info['total_impulse'], 'impulse')}")

    def _set_theme(self, name: str):
        self._theme = name
        apply_theme(cast(QApplication, QApplication.instance()), name)
        self._style_plots()
        if hasattr(self, "motor_panel"):
            self.motor_panel.set_theme(name)

    def _about(self):
        QMessageBox.about(
            self,
            "About HRAP (HCAT Fork)",
            f"{APP_NAME} {__version__}\n"
            "Hybrid Rocket Analysis Program\n\n"
            "HCAT fork of the original MATLAB HRAP (GPL-3.0) by Robert Nickel. "
            "The default engine is a line-for-line port of that sequential Euler loop. "
            "Advanced CoolProp / non-cylindrical options are optional and will not match original HRAP.",
        )


def _has_bundled(name: str) -> bool:
    try:
        bundled_motor(name)
        return True
    except Exception:
        return False


def _prepare_qt_environment() -> None:
    """Make sure a desktop Qt plugin is used and Windows can find fonts."""
    if sys.platform != "win32":
        return
    plat = str(os.environ.get("QT_QPA_PLATFORM", "")).lower()
    if plat in {"offscreen", "minimal", "null"} and not os.environ.get("HRAP_OFFSCREEN"):
        os.environ.pop("QT_QPA_PLATFORM", None)
    windir = os.environ.get("WINDIR", r"C:\Windows")
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(windir) / "Fonts"))


def main():
    multiprocessing.freeze_support()  # sweep worker processes in the frozen Windows build
    _prepare_qt_environment()
    try:
        app = QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(__version__)
        app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / "resources" / "icon.ico")))
        apply_theme(app, "dark")
        pg.setConfigOptions(antialias=True, background="#1a1d23", foreground="#e6e8ee")
        win = MainWindow()
        win.show()
        win.raise_()
        win.activateWindow()
        sys.exit(app.exec())
    except SystemExit:
        raise
    except Exception:
        log = Path.cwd() / "hrap_launch.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            QMessageBox.critical(None, APP_NAME, f"Failed to start.\n\nDetails written to:\n{log}")
        except Exception:
            print(f"Failed to start. Details: {log}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
