"""HRAP desktop application — PySide6 + pyqtgraph."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QEvent, QObject, QProcess, QSettings, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
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
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hrap import APP_NAME, __version__, update
from hrap.engine.nox import nox, saturation_temperature
from hrap.engine.sim import run
from hrap.engine.summary import format_summary, summarize
from hrap.engine.types import Settings, State
from hrap.gui.sizing import SizingPage
from hrap.gui.study import StudyPage
from hrap.gui.theme import apply_theme
from hrap.gui.viz import MotorPanel, MotorView, _vent_visible
from hrap.gui.widgets import CollapsibleBox, PlainComboBox, PlainDoubleSpinBox, PlainSpinBox, UnitRow
from hrap.io.config import (
    bundled_motor,
    chamber_limit,
    default_cfg,
    load_json,
    load_matlab_mat,
    resolve,
    resolve_layout,
    save_json,
)
from hrap.io.export import export_csv, export_eng, export_rse
from hrap.layout import motor_layout
from hrap.units import (
    LENGTH_ITEMS,
    MASS_ITEMS,
    PRESSURE_ITEMS,
    TEMP_ITEMS,
    VOLUME_ITEMS,
    DisplayUnits,
    from_si,
    to_si,
)

# Label, output field, physical quantity. Engine arrays remain in SI units.
TRACES = [
    ("Thrust", "F_thr", "force"),
    ("Tank pressure", "P_tnk", "pressure"),
    ("Chamber pressure", "P_cmbr", "pressure"),
    ("Injector ΔP", "inj_dP", "pressure"),
    ("Tank temperature", "T_tnk", "temperature"),
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
    "force": "Thrust", "pressure": "Pressure", "ratio": "O/F",
    "mass_flow": "Mass flow", "speed": "Regression rate", "length": "Length", "mass": "Mass",
    "temperature": "Temperature",
}
DEFAULT_TRACES = {"Thrust", "Tank pressure", "Chamber pressure", "Injector ΔP", "Oxidizer flow", "Fuel flow"}
DISPLAY_UNIT_OPTIONS = {
    "pressure": ["psi", "bar", "kPa", "MPa", "Pa", "atm"],
    "length": LENGTH_ITEMS,
    "mass": MASS_ITEMS,
    "force": ["N", "kN", "lbf"],
    "volume": VOLUME_ITEMS,
    "temperature": TEMP_ITEMS,
    "speed": ["m/s", "mm/s", "in/s", "ft/s"],
}


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


@dataclass
class OpenMotor:
    """A motor in the header's tabs: its file, its settings, and its last run."""
    path: str  # "" until it's saved to a file
    title: str
    cfg: dict  # the settings as the pages last showed them
    saved: dict | None = None  # the settings as last loaded or saved, to tell whether it has unsaved edits
    run: tuple | None = None  # (settings, state, output, cfg) of its last run


def _same_cfg(a, b) -> bool:
    """Equal settings, apart from float rounding from unit conversions."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_cfg(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    return a == b


class _Updater(QObject):
    """Checks GitHub and installs updates off the UI thread."""
    found = Signal(object, bool)   # the newer commit (or None), quiet
    installed = Signal()
    failed = Signal(str, bool)     # message, quiet

    def check(self, repo: str, branch: str, current: str, quiet: bool):
        def work():
            try:
                commit = update.latest(repo, branch)
                self.found.emit(commit if commit.sha != current else None, quiet)
            except Exception as exc:
                self.failed.emit(str(exc), quiet)
        threading.Thread(target=work, daemon=True).start()

    def install(self, root: Path, commit: update.Commit):
        def work():
            try:
                update.update(root, commit)
                self.installed.emit()
            except Exception as exc:
                self.failed.emit(str(getattr(exc, "stderr", "") or exc).strip()[-500:], False)
        threading.Thread(target=work, daemon=True).start()


class MainWindow(QMainWindow):
    def __init__(self, prefs: QSettings | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1280, 860)
        self._shown: OpenMotor | None = None  # the motor the pages show
        self._output = None
        self._settings = None
        self._state = None
        self._theme = "dark"
        self._thread = None
        self._worker = None
        self._close_when_finished = False
        self._result_cfg = None
        self._hover_index = None
        self._prefs = prefs or QSettings("HCAT", APP_NAME)
        # The session (open motors, unsaved edits, study runs) is kept beside the settings a test passes in,
        # or in the app's data folder, so it survives quitting, crashes and update restarts.
        self._session_dir = (Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
                             if prefs is None else Path(prefs.fileName()).with_suffix(".session"))
        self._session_text = ""
        defaults = asdict(DisplayUnits())
        saved = {key: str(self._prefs.value(f"displayUnits/{key}", default))
                 for key, default in defaults.items()}
        self.display_units = DisplayUnits(**{
            key: unit if unit in DISPLAY_UNIT_OPTIONS[key] else defaults[key]
            for key, unit in saved.items()
        })
        self._last_dir = str(self._prefs.value("lastFileDir") or "")
        self._build()
        self._connect_derived()
        for control_type, signal in (
            (QDoubleSpinBox, "valueChanged"), (QSpinBox, "valueChanged"),
            (QComboBox, "currentIndexChanged"), (QCheckBox, "toggled"),
        ):
            for page in (self._form, self.mass_page):
                for control in page.findChildren(control_type):
                    getattr(control, signal).connect(self._invalidate_results)
                    getattr(control, signal).connect(self._update_tab_marker)
        for edit in (self.name, self.mfg):
            edit.textChanged.connect(self._invalidate_results)
            edit.textChanged.connect(self._update_tab_marker)
        self.sizing_page.sized.connect(self._update_tab_marker)
        self._restore_session()
        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._save_session)
        self._session_timer.start(2000)
        self.study_page.runs_changed.connect(self._save_study_runs)
        if update.install_root() is not None:
            QTimer.singleShot(3000, lambda: self._check_updates(quiet=True))

    def _build(self):
        file_menu = self._file_menu = self.menuBar().addMenu("&File")
        for text, slot, key in (
            ("New", self._new, QKeySequence.StandardKey.New),
            ("Open…", self._open_json, QKeySequence.StandardKey.Open),
            ("Import MATLAB .mat…", self._import_mat, None),
            ("Save", lambda: self._save_motor(self._shown), QKeySequence.StandardKey.Save),
            ("Save As…", lambda: self._save_motor(self._shown, choose_path=True), QKeySequence.StandardKey.SaveAs),
            ("Close motor", lambda: self._close_motor(self.motor_tabs.currentIndex()), QKeySequence.StandardKey.Close),
        ):
            action = file_menu.addAction(text, slot)
            if key is not None:
                action.setShortcut(key)
        file_menu.addSeparator()
        file_menu.addAction("Export CSV…", lambda: self._export("csv"))
        file_menu.addAction("Export RSE…", lambda: self._export("rse"))
        file_menu.addAction("Export ENG…", lambda: self._export("eng"))
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        ex_menu = self._examples_menu = self.menuBar().addMenu("&Examples")
        ex_menu.addAction("example_98mm (const O/F, ABS)", lambda: self._open_bundled("example_98mm"))
        ex_menu.addAction("Rattworks K240 (const O/F, HDPE)", lambda: self._open_bundled("Rattworks_K240"))

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction("Dark theme", lambda: self._set_theme("dark"))
        view_menu.addAction("Light theme", lambda: self._set_theme("light"))

        settings_menu = self.menuBar().addMenu("&Settings")
        settings_menu.addAction("Units…", self._choose_display_units)
        settings_menu.addSeparator()
        settings_menu.addAction("Check for updates…", lambda: self._check_updates(quiet=False))
        settings_menu.addAction("Update branch…", self._choose_update_branch)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(f"About {APP_NAME}", self._about)

        top = QWidget()
        top.setObjectName("configHeader")
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(12, 6, 12, 6)
        # One tab per open motor. Each keeps its unsaved edits and last run, so switching is instant.
        self.motor_tabs = QTabBar()
        self.motor_tabs.setObjectName("motorTabs")
        self.motor_tabs.setTabsClosable(True)
        self.motor_tabs.setMovable(True)
        self.motor_tabs.setExpanding(False)
        self.motor_tabs.setDrawBase(False)
        self.motor_tabs.currentChanged.connect(self._on_motor_tab)
        self.motor_tabs.tabCloseRequested.connect(self._close_motor)
        for key, step in ((QKeySequence.StandardKey.NextChild, 1), (QKeySequence.StandardKey.PreviousChild, -1)):
            QShortcut(key, self, lambda step=step: self._step_motor(step))
        self._open_btn = QPushButton("Open…")
        self._open_btn.setToolTip("Open a motor file in a new tab.")
        self._open_btn.clicked.connect(self._open_json)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Motor name")
        self.name.setToolTip("The motor's name, used in the drawing and in exported RSE and ENG files.")
        self.name.setMaximumWidth(320)
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save this motor to its file.")
        save_btn.clicked.connect(lambda: self._save_motor(self._shown))
        top_l.addWidget(self.motor_tabs)
        top_l.addWidget(self._open_btn)
        top_l.addStretch(1)
        top_l.addWidget(QLabel("Name"))
        top_l.addWidget(self.name, 1)
        top_l.addWidget(save_btn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._form = self._make_form()
        splitter.addWidget(self._form)
        splitter.addWidget(self._make_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 900])
        self.sizing_page = SizingPage(self._form_to_cfg, lambda: self.display_units)
        self.sizing_page.motor_edited.connect(self._on_motor_edited)
        self.sizing_page.sized.connect(self._update_motor_summary)
        self.sizing_page.applied.connect(self._on_applied)
        self.study_page = StudyPage(self._form_to_cfg, lambda: self.display_units, self._open_study_case,
                                    self.sizing_page.unapplied)
        self.study_page.started.connect(self._sync_busy)
        self.study_page.finished.connect(self._sync_busy)
        self.mass_page = self._make_mass_page()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("pageTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.addTab(self.sizing_page, "Motor")
        self.tabs.addTab(splitter, "Simulation")
        self.tabs.addTab(self.study_page, "Study")
        self.tabs.addTab(self.mass_page, "Mass && export")
        self.tabs.setCurrentWidget(splitter)
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(top)
        root.addWidget(self.tabs, 1)
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
        self._update_btn = QPushButton("")
        self._update_btn.hide()
        self._update_btn.clicked.connect(self._update_clicked)
        self.statusBar().addPermanentWidget(self._update_btn)
        self._updater = _Updater()
        self._updater.found.connect(self._update_found)
        self._updater.installed.connect(self._update_installed)
        self._updater.failed.connect(self._update_failed)
        self._pending = None  # the commit an update would install
        self.statusBar().showMessage("Ready")

    def _make_form(self) -> QWidget:
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(8, 8, 8, 8)
        wl.setSpacing(8)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        wl.addWidget(self.run_btn)

        # The motor is set on the Motor tab; this only says which one runs.
        motor = QFrame()
        motor.setObjectName("sizingCard")
        ml = QVBoxLayout(motor)
        ml.setContentsMargins(12, 8, 12, 8)
        ml.setSpacing(4)
        head = QHBoxLayout()
        title = QLabel("Motor")
        title.setObjectName("cardTitle")
        edit = QPushButton("Edit on Motor tab")
        edit.clicked.connect(lambda: self.tabs.setCurrentWidget(self.sizing_page))
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(edit)
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
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Copy the Motor tab's sized parts into the motor.")
        apply_btn.clicked.connect(lambda: self.sizing_page.apply())
        ul.addWidget(self.unapplied_text, 1)
        ul.addWidget(apply_btn, 0, Qt.AlignmentFlag.AlignTop)
        ml.addWidget(self.unapplied)
        wl.addWidget(motor)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setSpacing(8)

        # Run
        self.solve_tank_cooling = QCheckBox("Solve tank cooling each step")
        self.solve_tank_cooling.setToolTip(
            "Near the end of the liquid, HRAP's tank model runs into a numerical problem and switches to an\n"
            "averaged pressure drop. This works out the tank cooling properly instead.\n"
            "Total impulse usually changes by under 1%. Works without Enable advanced options."
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
        sim = CollapsibleBox("Run")
        sf = sim.form()
        sf.addRow("Max run time [s]", self.tmax)
        sf.addRow("Close valve at [s]", self.tburn)
        sf.addRow("Timestep [ms]", self.dt)
        sf.addRow("Chamber start pressure", self.P_cmbr)
        root.addWidget(sim)

        # Advanced
        adv = CollapsibleBox("Advanced (not necessarily MATLAB-identical)")
        af = adv.form()
        self.adv_on = QCheckBox("Enable advanced options")
        self.live_chem = QCheckBox("Live chemistry (WIP)")
        self.ox_fluid = PlainComboBox()
        self.ox_fluid.addItems(["N2O_legacy", "NitrousOxide (CoolProp)"])
        self.ox_fluid.setToolTip("Which nitrous property data the tank uses with the SPI injector model.\n"
                                 "HEM and Dyer always use CoolProp.")
        self.grain_shape = PlainComboBox()
        self.grain_shape.addItems(["cylindrical", "star"])
        self.star_tips = PlainSpinBox()
        self.star_tips.setRange(3, 16)
        self.star_tips.setValue(6)
        af.addRow(self.solve_tank_cooling)
        af.addRow(self.adv_on)
        af.addRow(self.live_chem)
        af.addRow("Oxidizer fluid", self.ox_fluid)
        af.addRow("Grain shape", self.grain_shape)
        af.addRow("Star tips", self.star_tips)
        self.grain_shape.currentTextChanged.connect(lambda shape: af.setRowVisible(self.star_tips, shape == "star"))
        af.setRowVisible(self.star_tips, False)
        root.addWidget(adv)

        # One label column width for every section, so the fields line up down the panel.
        labels = [item.widget() for box in inner.findChildren(CollapsibleBox) for r in range(box.form().rowCount())
                  if (item := box.form().itemAt(r, QFormLayout.ItemRole.LabelRole)) is not None]
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setMinimumWidth(width)
        # Never let the splitter squeeze the panel narrower than its fields; there's no horizontal scrollbar.
        scroll.setMinimumWidth(inner.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        root.addStretch(1)
        scroll.setWidget(inner)
        wl.addWidget(scroll, 1)
        return wrap

    def _make_mass_page(self) -> QWidget:
        page = QScrollArea()
        page.setWidgetResizable(True)
        inner = QWidget()
        outer = QHBoxLayout(inner)
        outer.setContentsMargins(16, 16, 16, 16)
        column = QVBoxLayout()
        column.setSpacing(12)
        outer.addLayout(column)
        outer.addStretch(1)
        inner.setMaximumWidth(1100)

        self.mp_on = QCheckBox("Calculate mass properties (ENG / RSE CG)")
        self.tnk_start = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_m = UnitRow(MASS_ITEMS, "kg")
        self.cmbr_start = UnitRow(LENGTH_ITEMS, "in")
        self.cmbr_m = UnitRow(MASS_ITEMS, "kg")
        self.dry_OD = UnitRow(LENGTH_ITEMS, "in")
        self.dry_L = UnitRow(LENGTH_ITEMS, "in")
        self.mfg = QLineEdit("HRAP")
        self.mass_info = QLabel("Empty mass / CG: —")
        self.mass_info.setWordWrap(True)
        self._legacy_mtr_m = 0.0
        self._legacy_mtr_m_unit = "kg"
        self._legacy_mtr_cg = 0.0
        self._legacy_mtr_cg_unit = "in"

        mass = CollapsibleBox("Mass properties")
        mf = mass.form()
        mf.addRow(self.mp_on)
        mf.addRow("Tank front, from datum", self.tnk_start)
        mf.addRow("Tank dry mass", self.tnk_m)
        mf.addRow("Chamber front, from datum", self.cmbr_start)
        mf.addRow("Chamber dry mass (no grain)", self.cmbr_m)
        mf.addRow(self.mass_info)
        mass_fields = (self.tnk_start, self.tnk_m, self.cmbr_start, self.cmbr_m)
        self.mp_on.toggled.connect(lambda on: [w.setEnabled(on) for w in mass_fields])
        for w in mass_fields:
            w.setEnabled(False)
        column.addWidget(mass)

        export = CollapsibleBox("Export")
        ef = export.form()
        ef.addRow("Manufacturer", self.mfg)
        ef.addRow("Motor OD", self.dry_OD)
        ef.addRow("Motor length", self.dry_L)
        buttons = QHBoxLayout()
        for label, kind in (("Export .RSE", "rse"), ("Export .ENG", "eng"), ("Export CSV", "csv")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=kind: self._export(k))
            buttons.addWidget(btn)
        self.rse_btn = cast(QPushButton, buttons.itemAt(0).widget())
        ef.addRow(buttons)
        column.addWidget(export)
        column.addStretch(1)
        for box in (mass, export):
            box.setMinimumWidth(520)
        page.setWidget(inner)
        return page

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
        self.plot_readout = QLabel("Hover a plot to read values.")
        self.plot_readout.setWordWrap(True)
        self.plot_readout.setMinimumHeight(36)
        self.trace_list = QListWidget()
        for i, (label, _key, _quantity) in enumerate(TRACES):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if label in DEFAULT_TRACES else Qt.CheckState.Unchecked)
            self.trace_list.addItem(item)
        self.trace_list.itemChanged.connect(lambda *_: self._refresh_plot())
        mid.addWidget(self.trace_list)
        mid.addWidget(self.plot)
        mid.setSizes([160, 600])
        plots = QWidget()
        plots_l = QVBoxLayout(plots)
        plots_l.setContentsMargins(0, 0, 0, 0)
        plots_l.addWidget(mid, 1)
        plots_l.addWidget(self.plot_readout)
        plots.setMinimumHeight(260)
        self.motor_panel = MotorPanel()
        self.viz = self.motor_panel.viz
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.summary.setMinimumHeight(60)
        # Drag the dividers to trade plot height for the diagram or the summary.
        column = QSplitter(Qt.Orientation.Vertical)
        column.setChildrenCollapsible(False)
        for part, stretch in ((plots, 5), (self.motor_panel, 3), (self.summary, 2)):
            column.addWidget(part)
            column.setStretchFactor(column.indexOf(part), stretch)
        column.setSizes([520, 300, 120])
        self.limit_warning = QLabel("")
        self.limit_warning.setObjectName("sizingError")
        self.limit_warning.setWordWrap(True)
        self.limit_warning.hide()
        layout.addWidget(self.limit_warning)
        layout.addWidget(column, 1)
        return box

    def _form_to_cfg(self) -> dict:
        lay = self._form_layout()
        empty_m, empty_cg = self._empty_mass_si(lay)
        cfg = default_cfg()
        cfg.update(self.sizing_page.motor_cfg())
        cfg.update({
            "mtr_nm": self.name.text() or "mtr_cfg",
            "solve_tank_cooling": self.solve_tank_cooling.isChecked(),
            "mp_state": int(self.mp_on.isChecked()),
            "tnk_start": self.tnk_start.spin.value(),
            "tnk_start_unit": self.tnk_start.unit.currentText(),
            "cmbr_start": self.cmbr_start.spin.value(),
            "cmbr_start_unit": self.cmbr_start.unit.currentText(),
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
            "P_cmbr": self.P_cmbr.spin.value(),
            "P_cmbr_unit": self.P_cmbr.unit.currentText(),
            "t_max": self.tmax.value(),
            "t_burn": self.tburn.value(),
            "dt": self.dt.value() / 1000.0,
            "advanced": {
                "enabled": self.adv_on.isChecked(),
                "ox_fluid": self.ox_fluid.currentText(),
                "grain_shape": self.grain_shape.currentText(),
                "star_tips": self.star_tips.value(),
                "live_chem": self.live_chem.isChecked(),
            },
            "export_OD": self.dry_OD.si("length"),
            "export_L": self.dry_L.si("length"),
            "mfg": self.mfg.text() or "HRAP",
            "sizing": self.sizing_page.targets_cfg(),
        })
        return cfg

    def _cfg_to_form(self, cfg: dict):
        self.name.setText(str(cfg.get("mtr_nm") or ""))
        self.mfg.setText(str(cfg.get("mfg") or "HRAP"))
        self.solve_tank_cooling.setChecked(bool(cfg.get("solve_tank_cooling")))
        self.mp_on.setChecked(bool(cfg.get("mp_state")))
        self._legacy_mtr_m = float(cfg.get("mtr_m") or 0.0)
        self._legacy_mtr_m_unit = str(cfg.get("mtr_m_unit") or "kg")
        self._legacy_mtr_cg = float(cfg.get("mtr_cg") or 0.0)
        self._legacy_mtr_cg_unit = str(cfg.get("mtr_cg_unit") or "in")
        lay = resolve_layout(cfg)
        self.tnk_start.set_display(from_si(lay.tnk0, cfg.get("tnk_start_unit") or "in", "length"), cfg.get("tnk_start_unit") or "in")
        self.cmbr_start.set_display(from_si(lay.cmbr0, cfg.get("cmbr_start_unit") or "in", "length"), cfg.get("cmbr_start_unit") or "in")
        self.tnk_m.set_display(cfg.get("tnk_m", 0), cfg.get("tnk_m_unit", "kg"))
        self.cmbr_m.set_display(cfg.get("cmbr_m", 0), cfg.get("cmbr_m_unit", "kg"))
        self.P_cmbr.set_display(cfg.get("P_cmbr", 1), cfg.get("P_cmbr_unit", "atm"))
        self.tmax.setValue(float(cfg.get("t_max") or 10))
        self.tburn.setValue(float(cfg.get("t_burn") or 0))
        self.dt.setValue(1000.0 * float(cfg.get("dt") or 0.001))
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
        self.sizing_page.load_motor(cfg)
        self.sizing_page.set_targets(cfg.get("sizing") or {}, cfg)
        self.sizing_page.refresh()
        self._on_motor_edited()

    def _on_motor_edited(self):
        self.ox_fluid.setEnabled(self.sizing_page.inj_model.currentText() == "SPI")
        self._invalidate_results()
        self._update_derived_labels()

    def _update_motor_summary(self):
        """Say which motor runs, and what the Motor tab's sizing would still change about it."""
        sp, u = self.sizing_page, self.display_units
        n = sp.holes.value()
        if sp.inj_type.currentText() == "Swirler":
            injector = f"{n} swirler{'s' if n != 1 else ''} ({sp.sw_ports.value()} × {u.text(sp.sw_D_port.si('length'), 'length')} holes)"
        else:
            injector = f"{n} × {u.text(sp.hole_D.si('length'), 'length')} holes"
        throat, _exit, _er = sp.nozzle_size()
        self.motor_summary.setText(" · ".join((f"{u.text(sp.tank_geometry()[2], 'volume')} tank", injector,
                                               f"{u.text(sp.grain_L.si('length'), 'length')} grain",
                                               f"{u.text(throat, 'length')} throat")))
        changes = sp.unapplied()
        self.unapplied_text.setText("Not applied: " + ", ".join(changes) + ".")
        self.unapplied.setVisible(bool(changes))

    def _on_applied(self):
        self.statusBar().showMessage("Applied to motor.")
        self.tabs.setCurrentIndex(1)

    def _connect_derived(self):
        for row in (self.tnk_start, self.tnk_m, self.cmbr_start, self.cmbr_m, self.P_cmbr):
            row.spin.valueChanged.connect(self._update_derived_labels)
            row.unit.currentTextChanged.connect(self._update_derived_labels)
        self.name.textChanged.connect(self._update_derived_labels)

    def _update_derived_labels(self):
        try:
            lay = self._form_layout()
            empty_m, empty_cg = self._empty_mass_si(lay)
            self.mass_info.setText(
                f"Empty mass {self.display_units.text(empty_m, 'mass')} at CG {self.display_units.text(empty_cg, 'length')}. "
                f"Overall length {self.display_units.text(lay.overall_L, 'length')} "
                f"(tank L {self.display_units.text(lay.tnk_L, 'length')})."
            )
            if not self.dry_OD.spin.hasFocus():
                self.dry_OD.set_si(lay.overall_OD, "length")
            if not self.dry_L.spin.hasFocus():
                self.dry_L.set_si(lay.overall_L, "length")
        except Exception:
            self.mass_info.setText("Empty mass / CG: —")
        self._refresh_viz()

    def _form_layout(self):
        sp = self.sizing_page
        tnk_L, tnk_D, _V = sp.tank_geometry()
        throat, exit_D, _er = sp.nozzle_size()
        return motor_layout(
            tnk_start=self.tnk_start.si("length"),
            tnk_L=tnk_L,
            tnk_m=self.tnk_m.si("mass"),
            tnk_D=tnk_D,
            cmbr_start=self.cmbr_start.si("length"),
            pre_L=sp.pre_L.si("length"),
            post_L=sp.post_L.si("length"),
            cmbr_m=self.cmbr_m.si("mass"),
            grn_L=sp.grain_L.si("length"),
            grn_OD=sp.grain_OD.si("length"),
            noz_thrt=throat,
            noz_exit=exit_D,
        )

    def _empty_mass_si(self, lay=None) -> tuple[float, float]:
        lay = lay or self._form_layout()
        if lay.dry_mass > 0.0:
            return lay.dry_mass, lay.dry_cg
        return (
            to_si(self._legacy_mtr_m, self._legacy_mtr_m_unit, "mass"),
            to_si(self._legacy_mtr_cg, self._legacy_mtr_cg_unit, "length"),
        )

    def _motor_view(self, index: int | None = None) -> MotorView:
        sp = self.sizing_page
        tnk_L, tnk_D, tnk_V = sp.tank_geometry()
        grn_L = sp.grain_L.si("length")
        grn_OD = sp.grain_OD.si("length")
        grn_ID = sp.port_D.si("length")
        inj_D = sp.hole_D.si("length")
        inj_N = sp.holes.value()
        inj_Cd = sp.inj_Cd.value()
        inj_A = 0.25 * math.pi * inj_D ** 2 * inj_N
        vnt = sp.vent.currentText()
        vnt_D = sp.vent_D.si("length")
        th, exit_d, er = sp.nozzle_size()
        fill0, T0, m0 = sp.tank_state()
        rho = sp.rho.si("density")
        m_f0 = max(0.25 * math.pi * max(grn_OD ** 2 - grn_ID ** 2, 0.0) * rho * grn_L, 0.0)
        fill = fill0
        T = T0
        m_o = m0
        m_f = m_f0
        ox_mdot = 0.0
        fuel_mdot = 0.0
        noz_mdot = 0.0
        of_ratio = sp.result.OF if sp.result is not None else float("nan")  # at the start of the burn
        thrust = 0.0
        time_s = None
        P_tnk = 0.0
        try:
            P_tnk = float(nox(T0).Pv)
        except Exception:
            pass
        P_cmbr = self.P_cmbr.si("pressure")

        o = self._output
        if o is not None and o.t.size and index is not None:
            i = int(np.clip(index, 0, o.t.size - 1))
            self._hover_index = i
            time_s = float(o.t[i])
            P_tnk = float(o.P_tnk[i])
            m_o = float(o.m_o[i])
            T = saturation_temperature(P_tnk) or T0
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
            (f"{inj_N} swirler{'s' if inj_N > 1 else ''}, exit Ø{u.text(inj_D, 'length', 3)}"
             if sp.inj_type.currentText() == "Swirler" else f"{inj_N} × Ø{u.text(inj_D, 'length', 3)}"),
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
            pre_L=lay.grn0 - lay.plate1,
            post_L=lay.x_case - lay.grn1,
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
        self.plot_readout.setText("Hover a plot to read values.")
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
            if self.trace_list.item(k).checkState() != Qt.CheckState.Checked:
                continue
            value = float(getattr(self._output, key)[i])
            bits.append(f"{label}: {self.display_units.text(value, trace_quantity)}")
        self.plot_readout.setText("   ·   ".join(bits))
        self._refresh_viz_at(i)

    def _refresh_viz_at(self, index: int | None):
        self._hover_index = index
        self._refresh_viz()

    def _invalidate_results(self):
        if self._output is None:
            return
        self._output = self._settings = self._state = self._result_cfg = None
        self._hover_index = None
        self.limit_warning.hide()
        self.summary.clear()
        self._clear_plot()
        self._refresh_viz()
        self.statusBar().showMessage("Inputs changed, run again to update.")

    def _run(self):
        if self._thread is not None:
            return
        self._running_cfg = self._form_to_cfg()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.show()
        self.statusBar().showMessage("Running simulation…")
        self._thread = QThread()
        self._set_running(True)
        self._worker = SimWorker(self._running_cfg)
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

    def _set_running(self, running: bool):
        for page in (self._form, self.sizing_page, self.mass_page):
            page.setEnabled(not running)
        self._sync_busy()

    def _sync_busy(self):
        """Motors can't be switched or opened while a simulation or study is running."""
        idle = self._thread is None and not self.study_page.busy()
        for page in (self._form, self.sizing_page, self.mass_page):
            page.setEnabled(idle)
        self.study_page.setEnabled(self._thread is None)
        for w in (self.motor_tabs, self._open_btn, self._file_menu, self._examples_menu):
            w.setEnabled(idle)

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
        study = self.study_page
        if study.busy():
            study.stop()
            study.finished.connect(self.close)
            self.statusBar().showMessage("Closing when the running study simulations finish…")
            event.ignore()
            return
        if self._thread is not None:
            self._close_when_finished = True
            self.statusBar().showMessage("Closing when the current simulation finishes…")
            event.ignore()
            return
        self._save_session()
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
        self._stop_progress()
        self._show_results()

    def _show_results(self):
        s, x, o = cast(Settings, self._settings), cast(State, self._state), self._output
        self._hover_index = None
        info = summarize(s, x, o)
        self.summary.setPlainText(format_summary(info, self.display_units))
        self._refresh_plot()
        self._refresh_viz()
        u = self.display_units
        peak, limit = float(np.max(o.P_cmbr)), chamber_limit(cast(dict, self._result_cfg))
        over = peak > limit
        self.limit_warning.setText(f"Peak chamber pressure {u.text(peak, 'pressure')} is above the "
                                   f"{u.text(limit, 'pressure')} chamber pressure limit.")
        self.limit_warning.setVisible(over)
        self.statusBar().showMessage(f"Done: {o.sim_end_cond}. Total impulse {u.text(info['total_impulse'], 'impulse')}"
                                     + ("  ⚠ Over the chamber pressure limit" if over else ""))

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
        self.plot_readout.setText("Hover a plot to read values.")

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
        checked = [i for i in range(len(TRACES)) if self.trace_list.item(i).checkState() == Qt.CheckState.Checked]
        per_plot: dict[str, int] = {}
        for i in checked:
            per_plot[TRACES[i][2]] = per_plot.get(TRACES[i][2], 0) + 1
        for i in checked:
            label, key, quantity = TRACES[i]
            unit = self.display_units.unit(quantity)
            if quantity not in self._plots:
                widget = pg.PlotWidget()
                plot = cast(pg.PlotItem, widget.getPlotItem())
                plot.showGrid(x=True, y=True, alpha=0.25)
                if per_plot[quantity] > 1:
                    plot.addLegend(offset=(-10, 5))
                plot.setLabel("left", PLOT_LABELS[quantity], units=unit)
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.getAxis("left").setWidth(100)
                plot.setLabel("bottom", "Time", units="s")
                self._plots[quantity] = plot
                self._plot_widgets[quantity] = widget
                self.plot.addTab(widget, PLOT_LABELS[quantity])
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
            self._plots[quantity].plot(o.t, self.display_units.value(np.asarray(getattr(o, key), dtype=float), quantity),
                                      pen=pen, name=label)
        if "pressure" in self._plots and self._result_cfg is not None:
            limit = self.display_units.value(chamber_limit(self._result_cfg), "pressure")
            self._plots["pressure"].addItem(pg.InfiniteLine(
                pos=limit, angle=0, movable=False, pen=pg.mkPen("#e06c75", width=1, style=Qt.PenStyle.DashLine),
                label="Chamber limit", labelOpts={"position": 0.05, "color": "#e06c75", "anchors": [(0, 1), (0, 1)]}))
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
            if plot.legend is not None:
                cast(pg.LegendItem, plot.legend).setLabelTextColor(fg)
        for line in self._hover_lines:
            line.setPen(self._hover_line_pen())

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

    def _motors(self) -> list[OpenMotor]:
        return [self.motor_tabs.tabData(i) for i in range(self.motor_tabs.count())]

    def _edited(self, motor: OpenMotor) -> bool:
        return not _same_cfg(motor.cfg, motor.saved)

    def _add_motor(self, cfg: dict, path: str = "", title: str = "", show: bool = True):
        """Open a motor in a new tab, or switch to its tab if its file is already open."""
        for i, motor in enumerate(self._motors()):
            if path and motor.path == path:
                self.motor_tabs.setCurrentIndex(i)
                return
        motor = OpenMotor(path, title or Path(path).stem, cfg)
        i = self.motor_tabs.addTab(motor.title)
        self.motor_tabs.setTabData(i, motor)
        self.motor_tabs.setTabToolTip(i, path or "Not saved to a file yet")
        if show:
            self.motor_tabs.setCurrentIndex(i)
            self._on_motor_tab(i)  # the first tab is current before its motor is attached

    def _on_motor_tab(self, index: int):
        motor = self.motor_tabs.tabData(index) if index >= 0 else None
        if motor is None or motor is self._shown:
            return
        if self._shown is not None:
            self._stash()
        self._shown = None  # loading the pages mustn't mark anything edited
        self._output = self._settings = self._state = self._result_cfg = None
        self._cfg_to_form(motor.cfg)
        if motor.saved is None:
            motor.saved = self._form_to_cfg()
        self._shown = motor
        self.study_page.motor_changed()
        if motor.run is not None:
            self._settings, self._state, self._output, self._result_cfg = motor.run
            self._show_results()
        else:
            self.limit_warning.hide()
            self.summary.clear()
            self._clear_plot()
            self._refresh_viz()
        self._update_tab_marker()

    def _open_study_case(self, cfg):
        self._add_motor(cfg, title=cfg["mtr_nm"])
        self._shown.saved = None
        self._update_tab_marker()
        self.tabs.setCurrentIndex(1)

    def _stash(self):
        """Keep the shown motor's settings and last run, so switching back brings them back."""
        motor = self._shown
        if motor is not None:
            motor.cfg = self._form_to_cfg()
            motor.run = (self._settings, self._state, self._output, self._result_cfg) if self._output is not None else None

    def _update_tab_marker(self, *_):
        motor = self._shown
        if motor is None:
            return
        motor.cfg = self._form_to_cfg()
        i = self._motors().index(motor)
        self.motor_tabs.setTabText(i, motor.title + (" •" if self._edited(motor) else ""))
        self.motor_tabs.setTabToolTip(i, motor.path or "Not saved to a file yet")
        self.setWindowTitle(f"{motor.title} · {APP_NAME} {__version__}")

    def _step_motor(self, step: int):
        if self.motor_tabs.isEnabled() and self.motor_tabs.count() > 1:
            self.motor_tabs.setCurrentIndex((self.motor_tabs.currentIndex() + step) % self.motor_tabs.count())

    def _close_motor(self, index: int):
        motor = self.motor_tabs.tabData(index)
        if motor is None or not self.motor_tabs.isEnabled():
            return
        if motor is self._shown:
            self._stash()
        if self._edited(motor):
            answer = QMessageBox.question(
                self, APP_NAME, f"{motor.title} has unsaved changes.",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Save and not self._save_motor(motor):
                return
        self.motor_tabs.removeTab(self._motors().index(motor))
        if self.motor_tabs.count() == 0:
            self._shown = None
            self._new()

    def _save_motor(self, motor: OpenMotor | None, choose_path: bool = False) -> bool:
        """Save a motor to its file, asking for one if it has none. Returns whether it was saved."""
        if motor is None:
            return False
        if motor is self._shown:
            motor.cfg = self._form_to_cfg()
        path = motor.path
        if choose_path or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save motor", path or self._dialog_path(f"{motor.title}.json"),
                                                  "HRAP JSON (*.json)")
            if not path:
                return False
            self._remember_file_dir(path)
        path = str(Path(path).resolve())
        try:
            save_json(path, motor.cfg)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save {motor.title} to {path}.\n\n{exc}")
            return False
        motor.path, motor.title = path, Path(path).stem
        motor.saved = motor.cfg
        if motor is self._shown:
            self._update_tab_marker()
        else:
            self.motor_tabs.setTabText(self._motors().index(motor), motor.title)
        self.statusBar().showMessage(f"Saved {motor.path}")
        return True

    def _open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open motor", self._dialog_path(), "HRAP JSON (*.json)")
        if path:
            self._remember_file_dir(path)
            self._open_path(path)

    def _open_path(self, path: str, show: bool = True):
        path = str(Path(path).resolve())
        self._add_motor(load_json(path), path, show=show)

    def _import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MATLAB motor", self._dialog_path(), "MATLAB (*.mat)")
        if path:
            self._remember_file_dir(path)
            self._add_motor(load_matlab_mat(path), title=Path(path).stem)

    def _open_bundled(self, name: str):
        self._add_motor(bundled_motor(name), title=name)

    def _new(self):
        self._add_motor(default_cfg(), title="Untitled")

    def _check_updates(self, quiet: bool):
        root = update.install_root()
        if root is None:
            if not quiet:
                QMessageBox.information(self, "Updates", "This copy runs from a source checkout, so it doesn't update "
                                        "itself. Update it with git pull.")
            return
        info = update.read_info(root)
        self._updater.check(info["repo"], info["branch"], info["sha"], quiet)

    def _choose_update_branch(self):
        root = update.install_root()
        if root is None:
            self._check_updates(quiet=False)
            return
        info = update.read_info(root)
        branch, ok = QInputDialog.getText(self, "Update branch", "Follow this GitHub branch (main is the reviewed one):",
                                          text=info["branch"])
        if ok and branch.strip() and branch.strip() != info["branch"]:
            info["branch"] = branch.strip()
            update.write_info(root, info)
            self._check_updates(quiet=False)

    def _update_found(self, commit, quiet: bool):
        if commit is None:
            if not quiet:
                QMessageBox.information(self, "Updates", "HRAP is up to date.")
            return
        self._pending = commit
        self._update_btn.setText("Update available")
        self._update_btn.setToolTip(f"{commit.message}\n{commit.date[:10]} · {commit.sha[:7]}")
        self._update_btn.setEnabled(True)
        self._update_btn.show()
        if not quiet:
            self._update_clicked()

    def _update_clicked(self):
        if self._pending == "restart":
            self._restart()
            return
        if self._pending is None:
            return
        self._update_btn.setText("Updating…")
        self._update_btn.setEnabled(False)
        self._updater.install(update.install_root(), self._pending)

    def _update_installed(self):
        self._pending = "restart"
        self._update_btn.setText("Restart to finish updating")
        self._update_btn.setEnabled(True)

    def _update_failed(self, message: str, quiet: bool):
        if self._pending is not None and self._pending != "restart":
            self._update_btn.setText("Update failed, try again")
            self._update_btn.setToolTip(message)
            self._update_btn.setEnabled(True)
        elif not quiet:
            QMessageBox.warning(self, "Updates", f"Couldn't check for updates:\n{message}")

    def _restart(self):
        """Start the updated app, then quit this one; the session brings everything back."""
        info = update.read_info(update.install_root())
        if sys.platform == "darwin" and info.get("app"):
            QProcess.startDetached("open", ["-n", info["app"]])
        else:
            exe = Path(sys.executable)
            pythonw = exe.with_name("pythonw.exe")
            QProcess.startDetached(str(pythonw if pythonw.exists() else exe), ["-m", "hrap"])
        self.close()

    def _save_session(self):
        """Write the open motors (with unsaved edits), the page and the window size, if they changed."""
        if self._shown is None:
            return
        self._stash()
        motors = [{"path": m.path, "title": m.title, "cfg": m.cfg if self._edited(m) or not m.path else None}
                  for m in self._motors()]
        text = json.dumps({"motors": motors, "current": self.motor_tabs.currentIndex(), "page": self.tabs.currentIndex(),
                           "geometry": bytes(self.saveGeometry().toBase64()).decode()}, default=str)
        if text != self._session_text:
            self._write_session_file("session.json", text.encode())
            self._session_text = text

    def _save_study_runs(self):
        self._write_session_file("study_runs.pickle", self.study_page.dump_runs())

    def _write_session_file(self, name: str, data: bytes):
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._session_dir / f"{name}.tmp"
            tmp.write_bytes(data)
            tmp.replace(self._session_dir / name)
        except OSError:
            pass  # a read-only or full disk only costs the restore

    def _restore_session(self):
        """Reopen last session's motors, unsaved edits, page, window size and study runs.

        A motor whose file was moved or deleted comes back only if it had unsaved edits (as unsaved).
        """
        try:
            data = json.loads((self._session_dir / "session.json").read_text())
        except (OSError, ValueError):
            data = {}
        for m in data.get("motors", []):
            path, cfg = m.get("path") or "", m.get("cfg")
            try:
                saved = load_json(path) if path else None
            except (OSError, ValueError):
                saved, path = None, ""
            if cfg is None and saved is None:
                continue
            self._add_motor(cfg or saved, path, m.get("title") or Path(path).stem or "Untitled", show=False)
            # A motor with no file that was edited stays marked; one with a file compares against it.
            self._motors()[-1].saved = saved if saved is not None else ({} if cfg else None)
        if self.motor_tabs.count() == 0:
            self._open_bundled("example_98mm") if _has_bundled("example_98mm") else self._new()
        else:
            i = min(int(data.get("current", 0)), self.motor_tabs.count() - 1)
            self.motor_tabs.setCurrentIndex(i)
            self._on_motor_tab(i)
        if 0 <= int(data.get("page", -1)) < self.tabs.count():
            self.tabs.setCurrentIndex(int(data["page"]))
        if data.get("geometry"):
            self.restoreGeometry(QByteArray.fromBase64(data["geometry"].encode()))
        try:
            self.study_page.load_runs((self._session_dir / "study_runs.pickle").read_bytes())
        except Exception:  # missing, or saved by a version whose classes changed
            pass

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
        note = QLabel("Pressures include the atmosphere (gauge + 14.7 psi).")
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
        self.sizing_page.refresh()
        self.study_page.refresh()
        if self._output is not None:
            info = summarize(cast(Settings, self._settings), cast(State, self._state), self._output)
            self.summary.setPlainText(format_summary(info, units))
            self.statusBar().showMessage(f"Done: {self._output.sim_end_cond}. Total impulse {units.text(info['total_impulse'], 'impulse')}")

    def _set_theme(self, name: str):
        self._theme = name
        apply_theme(cast(QApplication, QApplication.instance()), name)
        self._style_plots()
        if hasattr(self, "motor_panel"):
            self.motor_panel.set_theme(name)
        if hasattr(self, "sizing_page"):
            self.sizing_page.set_theme(name)

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


def _warm_coolprop_tables() -> None:
    from hrap.advanced.fluid import coolprop_sat
    from hrap.advanced.injector import hem_flux_table

    coolprop_sat("NitrousOxide")
    hem_flux_table("NitrousOxide")


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
        app.setOrganizationName("HCAT")
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(__version__)
        app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / "resources" / "icon.ico")))
        apply_theme(app, "dark")
        pg.setConfigOptions(antialias=True, background="#1a1d23", foreground="#e6e8ee")
        # Loading CoolProp and building its nitrous tables takes ~0.8 s and blocks Qt even from a
        # background thread, so do it before the window is up instead of on the first sizing or run.
        _warm_coolprop_tables()
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
