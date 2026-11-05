"""HRAP desktop application — PySide6 + pyqtgraph."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QByteArray, QEvent, QObject, QProcess, QSettings, QStandardPaths, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hrap import APP_NAME, __version__, update
from hrap.engine.nox import nox, saturation_temperature
from hrap.engine.sim import run
from hrap.engine.study import uses_spi
from hrap.engine.summary import format_summary, summarize
from hrap.engine.types import Settings, State
from hrap.gui.grain_view import GrainScene, GrainView, RunGrainView
from hrap.gui.runs import delete_run, new_run_file, read_run, run_files, write_run
from hrap.gui.sizing import SizingPage
from hrap.gui.study import StudyPage
from hrap.gui.theme import apply_theme, colors_for
from hrap.gui.viz import MotorPanel, MotorView, _vent_visible
from hrap.gui.widgets import (
    CARD_GAP,
    PAGE_MARGIN,
    FieldGrid,
    PlainComboBox,
    PlainDoubleSpinBox,
    UnitRow,
    card_frame,
    line_up,
    page_layout,
)
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
from hrap.io.measured import MeasuredCurve, load_measured_csv
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
    ("Burned distance", "grn_web", "length"),
    ("Burning area", "burn_area", "area"),
    ("Exposed case wall", "case_area", "area"),
    ("Oxidizer mass", "m_o", "mass"),
    ("Fuel mass", "m_f", "mass"),
    ("Total mass", "m_t", "mass"),
    ("CG", "cg", "length"),
]
PLOT_LABELS = {
    "force": "Thrust", "pressure": "Pressure", "ratio": "O/F",
    "mass_flow": "Mass flow", "speed": "Regression rate", "length": "Length", "mass": "Mass",
    "temperature": "Temperature", "area": "Area",
}
DEFAULT_TRACES = {"Thrust", "Tank pressure", "Chamber pressure", "Injector ΔP", "Oxidizer flow", "Fuel flow",
                  "Burning area", "Exposed case wall"}
STL_TRACES = {"grn_web", "burn_area", "case_area"}  # only an STL grain has these
# Each trace's theme colour: oxidizer blue, fuel orange, the case wall red; traces sharing a plot differ.
TRACE_COLORS = {
    "F_thr": "green", "P_tnk": "oxidizer", "P_cmbr": "purple", "inj_dP": "teal", "T_tnk": "oxidizer", "OF": "teal",
    "mdot_o": "oxidizer", "mdot_f": "fuel", "mdot_n": "grey", "rdot": "fuel", "grn_ID": "purple", "grn_web": "fuel",
    "burn_area": "fuel", "case_area": "limit", "m_o": "oxidizer", "m_f": "fuel", "m_t": "grey", "cg": "grey",
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


def _curve(o, settings, key: str) -> np.ndarray | None:
    """A run's curve for a trace; None if it has none, as a round grain has no STL curves.
    An STL grain's burning area and exposed case wall follow from its burned distance."""
    g = getattr(settings, "grain_geometry", None)
    if key in STL_TRACES and g is None:
        return None
    if key in ("burn_area", "case_area"):
        return np.interp(o.grn_web, g.web, getattr(g, key))
    return np.asarray(getattr(o, key), dtype=float)


def _case_contact(o, settings) -> float | None:
    """When an STL grain's burn first reached the case wall, if it did."""
    g = getattr(settings, "grain_geometry", None)
    if g is None or not math.isfinite(g.case_web):
        return None
    reached = np.flatnonzero(o.grn_web >= g.case_web)
    return float(o.t[reached[0]]) if reached.size else None


class SimWorker(QObject):
    finished = Signal(object, object, object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, cfg: dict, grain_tables=None):
        super().__init__()
        self.cfg, self.grain_tables = cfg, grain_tables  # an STL grain's tables, if the Motor tab has loaded it

    def run(self):
        try:
            s, x = resolve(self.cfg, grain_tables=self.grain_tables)
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


@dataclass
class SimRun:
    """A finished simulation in the Simulation tab's Runs list, kept in its file (see hrap.gui.runs)."""
    file: Path
    label: str
    cfg: dict
    changes: tuple = ()  # what changed from the motor's previous run
    results: dict = field(default_factory=dict)  # headline numbers (SI) from when it was first run
    time: str = ""  # when it was first run
    output: object = None  # its curves, simulated again after it's read from its file
    state: object = None  # the simulation's final state, for the summary
    settings: object = None  # what it simulated
    ticked: bool = False  # drawn dashed under the shown run

    def save(self):
        write_run(self.file, {"label": self.label, "time": self.time, "changes": list(self.changes),
                              "results": self.results, "motor": self.cfg})

    @classmethod
    def read(cls, path: Path) -> SimRun | None:
        data = read_run(path)
        if data is None:
            return None
        return cls(path, str(data.get("label") or path.stem), data["motor"], tuple(data.get("changes") or ()),
                   dict(data.get("results") or {}), str(data.get("time") or ""))


# Readable names for the settings a run's "what changed" mentions most; others show their file key.
_SETTING_NAMES = {
    "tnk_cond": "tank temp", "fill": "fill", "tnk_V": "tank volume", "noz_thrt": "throat", "noz_ex": "expansion ratio",
    "inj_D": "injector bore", "inj_N": "injectors", "grn_L": "grain length", "grn_ID": "starting port",
    "grn_OD": "grain OD", "prop_a": "burn rate a", "prop_n": "burn rate n", "cstar_eff": "C* eff", "const_OF": "O/F",
    "reg_model": "fuel flow", "inj_model": "flow model", "grn_shape": "grain shape", "grn_stl": "STL",
    "grn_stl_voxel": "cube size", "grn_ends_burn": "ends burn",
    "sw_ports": "swirler holes", "sw_D_port": "swirler hole", "sw_R_in": "hole offset", "sw_xi": "inlet loss",
    "noz_eff": "nozzle eff", "solve_tank_cooling": "tank cooling", "inj_type": "injector type",
    "cmbr_L": "TCA length", "tnk_L": "tank length",
    "sw_cd_from_geometry": "Cd from geometry", "ptc_stock": "stock PTC", "ox_fluid": "oxidizer fluid",
    "enabled": "advanced options", "live_chem": "live chemistry", "dt": "timestep", "t_max": "max run time",
}
_SETTING_UNITS = {"tnk_cond": "T_tnk_unit", "grn_stl_voxel": "grn_stl_unit"}  # where a setting's unit lives, when not at "<key>_unit"

# Settings that follow from other settings, so a run's "what changed" leaves them out.
_DERIVED_KEYS = {"tnk_L", "tnk_X", "cmbr_X", "cmbr_V", "export_L", "export_OD", "mtr_cg", "tnk_m", "cmbr_m", "mtr_m"}


def _changed_settings(old: dict, new: dict) -> list[str]:
    """What a run changed from an earlier one, as short "setting value" pieces."""
    def flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[prefix + k] = v
        return out

    a, b = flat(old), flat(new)
    changes = []
    for k, v in b.items():
        key = k.split(".")[-1]
        if (key in _DERIVED_KEYS or key.endswith("_unit") or key == "mtr_nm" or k.startswith("sizing.round_grain.")
                or _same_cfg(a.get(k), v)):
            continue
        name = _SETTING_NAMES.get(key, key.replace("_", " "))
        if key == "inj_Cd":  # the Motor tab shows the injector as a CdA
            D = to_si(float(b["inj_D"]), b.get("inj_D_unit") or "in", "length")
            changes.append(f"CdA {from_si(float(v) * 0.25 * math.pi * D ** 2, 'in^2', 'area'):.4g} in²")
        elif key == "grn_stl":  # the file's name, not its whole path
            changes.append(f"{name} {Path(v).name}")
        elif isinstance(v, float):
            unit = "%" if key == "fill" else str(b.get(_SETTING_UNITS.get(key, f"{key}_unit")) or "")
            changes.append(f"{name} {v:.4g}{' ' + unit if unit else ''}")
        else:
            changes.append(f"{name} {'auto' if v is None else v}")
    return changes


def _same_cfg(a, b) -> bool:
    """Equal settings, apart from float rounding from unit conversions."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_cfg(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    return a == b


class _Resimulator(QObject):
    """Simulates runs read from their files again, off the UI thread, as an STL grain can take a minute to load."""
    done = Signal(object, object)  # the run, and (settings, state, output) or an error message
    _lock = threading.Lock()  # one at a time, as each may hold a large grain in memory

    def start(self, sim_run: SimRun):
        def work():
            with self._lock:
                try:
                    s, x = resolve(sim_run.cfg)
                    x, o = run(s, x)
                    self.done.emit(sim_run, (s, x, o))
                except Exception as exc:
                    self.done.emit(sim_run, str(exc) or type(exc).__name__)
        threading.Thread(target=work, daemon=True).start()


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
        self._sim_runs: list[SimRun] = []
        self._waiting_runs: dict[Path, list[Callable[[], None]]] = {}  # runs being simulated again, and what then
        self._wanted_run: SimRun | None = None  # the run last clicked, which shows once it's simulated again
        self._grain_was_loading = False  # the grain's loading was on screen, so its end gets a status message
        self._measured: MeasuredCurve | None = None
        self._plotted_measured = None
        self._hover_index = None
        self._prefs = prefs or QSettings("HCAT", APP_NAME)
        # The session (open motors, unsaved edits, runs of motors without a file) is kept beside the settings a test
        # passes in, or in the app's data folder, so it survives quitting, crashes and update restarts.
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
        self.name.textChanged.connect(self._invalidate_results)
        self.name.textChanged.connect(self._update_tab_marker)
        self.sizing_page.sized.connect(self._update_tab_marker)
        self._restore_session()
        self._reload_measured(str(self._prefs.value("measuredOverlay") or ""), quiet=True)
        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._save_session)
        self._session_timer.start(2000)
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
        file_menu.addAction("Overlay measured curve…", self._load_measured)
        file_menu.addAction("Clear measured overlay", self._clear_measured)
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
        self.name.setToolTip("The motor's name, shown on its tab and used in the drawing and in exported RSE and ENG files.")
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

        self.grain_scene = GrainScene()  # the STL grain in 3D, shared by the Grain tab and the Simulation tab
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
        self.sizing_page.grain_changed.connect(self._sync_grain_view)
        self.sizing_page.show_grain.connect(lambda: self.tabs.setCurrentWidget(self.grain_view))
        self.sizing_page.grain_progress.connect(self._show_grain_progress)
        self._load_learned_timing()
        self.sizing_page.load_timing_learned.connect(lambda: self._prefs.setValue(
            "grainLoadTimeShares", ",".join(str(x) for x in self.sizing_page.load_time_shares)))
        self.grain_view = GrainView(self.grain_scene, lambda: self.display_units, lambda: self.sizing_page.rho.si("density"))
        self.grain_view.cancel_load.connect(self.sizing_page.stop_grain_load)
        self.grain_view.message_action.connect(self._on_grain_message_action)
        self.study_page = StudyPage(self._form_to_cfg, lambda: self.display_units, self._open_unsaved,
                                    lambda: self._runs_dir(self._shown), self.sizing_page.unapplied, self._save_output)
        self.study_page.started.connect(self._sync_busy)
        self.study_page.finished.connect(self._sync_busy)
        self.mass_page = self._make_mass_page()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("pageTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)
        self.tabs.addTab(self.sizing_page, "Motor")
        self.tabs.addTab(self.grain_view, "Grain")
        self.tabs.setTabVisible(self.tabs.indexOf(self.grain_view), False)
        self.tabs.addTab(splitter, "Simulation")
        self._sim_page = splitter
        self.tabs.addTab(self.study_page, "Sweep")
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
        self._resimulator = _Resimulator()
        self._resimulator.done.connect(self._on_resimulated)
        self._updater = _Updater()
        self._updater.found.connect(self._update_found)
        self._updater.installed.connect(self._update_installed)
        self._updater.failed.connect(self._update_failed)
        self._pending = None  # the commit an update would install
        self.statusBar().showMessage("Ready")

    def _make_form(self) -> QWidget:
        wrap = QWidget()
        wl = page_layout(QVBoxLayout(wrap))
        wl.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        wl.addWidget(self.run_btn)

        # The motor is set on the Motor tab; this only says which one runs.
        motor, ml = card_frame("Motor")
        title = ml.itemAt(0).widget()
        ml.removeWidget(title)
        head = QHBoxLayout()
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

        # The cards scroll under the Run button, keeping their edges in line when the scrollbar shows.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(CARD_GAP)
        root.addWidget(motor)

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
        sim, sim_l = card_frame("Run")
        sf = FieldGrid()
        sf.add("Max run time", self.tmax, "s")
        sf.add("Close valve at", self.tburn, "s")
        sf.add("Timestep", self.dt, "ms")
        sf.add("Chamber start pressure", self.P_cmbr)
        sim_l.addLayout(sf)
        root.addWidget(sim)

        # Runs: every finished simulation, to draw under the current one
        runs, runs_l = card_frame("Runs")
        self.run_list = QListWidget()
        self.run_list.setToolTip("Every simulation of this motor lands here, with what changed from its previous run.\n"
                                 "Click a run to load its settings and results; tick runs to draw them dashed under it.\n"
                                 "Right-click to rename, export or delete.")
        self.run_list.setMinimumHeight(120)
        self.run_list.setStyleSheet("QListView::item { padding: 3px 4px; }")  # room around each checkbox
        self.run_list.itemChanged.connect(self._on_run_item_changed)
        self.run_list.itemPressed.connect(lambda item: setattr(self, "_pressed_check", item.checkState()))
        self.run_list.itemClicked.connect(self._on_run_clicked)
        self.run_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.run_list.customContextMenuRequested.connect(self._run_menu)
        clear_runs = QPushButton("Clear runs")
        clear_runs.clicked.connect(self._clear_runs)
        runs_l.addWidget(self.run_list)
        runs_l.addWidget(clear_runs)
        root.addWidget(runs)

        # Advanced
        adv, adv_l = card_frame("Advanced (not necessarily MATLAB-identical)")
        af = FieldGrid()
        self.adv_on = QCheckBox("Enable advanced options")
        self.live_chem = QCheckBox("Live chemistry (WIP)")
        self.ox_fluid = PlainComboBox()
        self.ox_fluid.addItems(["N2O_legacy", "NitrousOxide (CoolProp)"])
        self.ox_fluid.setToolTip("Which nitrous property data the tank uses with the SPI injector model.\n"
                                 "HEM and Dyer always use CoolProp.")
        for box in (self.solve_tank_cooling, self.adv_on, self.live_chem):
            af.add_line(box)
        af.add("Oxidizer fluid", self.ox_fluid)
        self.adv_on.toggled.connect(self._sync_advanced)
        adv_l.addLayout(af)
        root.addWidget(adv)
        line_up(sf, af)

        # Never let the splitter squeeze the panel narrower than its fields; there's no horizontal scrollbar.
        scroll.setMinimumWidth(inner.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        root.addStretch(1)
        scroll.setWidget(inner)
        wl.addWidget(scroll, 1)
        return wrap

    def _make_mass_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        column = page_layout(QVBoxLayout(inner))

        self._tank_len_custom = False
        self._tca_len_custom = False
        self.mp_on = QCheckBox("Calculate mass properties (ENG / RSE CG)")
        self.tnk_start = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_L = UnitRow(LENGTH_ITEMS, "in")
        self.tnk_m = UnitRow(MASS_ITEMS, "kg")
        self.cmbr_start = UnitRow(LENGTH_ITEMS, "in")
        self.cmbr_L = UnitRow(LENGTH_ITEMS, "in")
        self.cmbr_m = UnitRow(MASS_ITEMS, "kg")
        self.dry_OD = UnitRow(LENGTH_ITEMS, "in")
        self.dry_L = UnitRow(LENGTH_ITEMS, "in")
        self.mass_info = QLabel("Empty mass / CG: —")
        self.mass_info.setWordWrap(True)
        self._legacy_mtr_m = 0.0
        self._legacy_mtr_m_unit = "kg"
        self._legacy_mtr_cg = 0.0
        self._legacy_mtr_cg_unit = "in"
        self.tnk_L.setToolTip(
            "Length of the oxidizer tank in the drawing, and where its dry mass sits.\n"
            "It follows the internal length on the Motor tab until you type a length."
        )
        self.cmbr_L.setToolTip(
            "Length of the thrust chamber from the injector face through the nozzle.\n"
            "It starts from the grain, the pre- and post-combustion chambers, and the nozzle.\n"
            "A longer length adds empty chamber, with the grain kept in the middle.\n"
            "A shorter length is kept too. The drawing overlaps where the parts no longer fit."
        )
        self.tnk_start.setToolTip("Distance from the datum to the front of the tank.")
        self.cmbr_start.setToolTip("Distance from the datum to the injector face.")

        mass, mass_l = card_frame("Mass properties")
        mf = FieldGrid()
        mf.add_line(self.mp_on)
        mf.add("Tank front, from datum", self.tnk_start)
        mf.add("Tank length", self.tnk_L)
        mf.add("Tank dry mass", self.tnk_m)
        mf.add("TCA front, from datum", self.cmbr_start)
        mf.add("TCA length", self.cmbr_L)
        mf.add("Chamber dry mass (no grain)", self.cmbr_m)
        mf.add_line(self.mass_info)
        mass_l.addLayout(mf)
        mass_fields = (self.tnk_start, self.tnk_m, self.cmbr_start, self.cmbr_m)
        self.mp_on.toggled.connect(lambda on: [w.setEnabled(on) for w in mass_fields])
        for w in mass_fields:
            w.setEnabled(False)
        column.addWidget(mass)

        export, export_l = card_frame("Export")
        ef = FieldGrid()
        ef.add("Motor OD", self.dry_OD)
        ef.add("Motor length", self.dry_L)
        export_l.addLayout(ef)
        buttons = QHBoxLayout()
        for label, kind in (("Export .RSE", "rse"), ("Export .ENG", "eng"), ("Export CSV", "csv")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=kind: self._export(k))
            buttons.addWidget(btn)
        export_l.addLayout(buttons)
        column.addWidget(export)
        column.addStretch(1)
        line_up(mf, ef)
        for box in (mass, export):
            box.setMinimumWidth(520)
        scroll.setWidget(inner)

        self.mass_preview = MotorPanel()
        self.mass_preview.viz.set_framed(True)
        self.mass_preview.viz.setMinimumHeight(360)
        self.mass_preview.viz.setMaximumHeight(16777215)
        row.addWidget(scroll, 0)
        row.addWidget(self.mass_preview, 1)
        self._sync_length_fields()
        return page

    def _make_results(self) -> QWidget:
        box = QWidget()
        layout = page_layout(QVBoxLayout(box))
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
        self.spi_dp_line = QCheckBox("300 psi ΔP line (SPI)")
        self.spi_dp_line.setToolTip("Mark 300 psi on the pressure plot. Above about 300 psi of injector ΔP, real nitrous\n"
                                    "boils in the hole and flows less than SPI predicts. Only shown for SPI motors.")
        self.spi_dp_line.toggled.connect(lambda *_: self._refresh_plot())
        self.overlay_btn = QPushButton("Overlay CSV…")
        self.overlay_btn.setToolTip("Plot a measured thrust curve on the simulation. Also draws tank pressure\n"
                                    "if the file has it (Weight_Lbs / F_thr, Tplus_ms / t, PT1_PSI / P_tnk).")
        self.overlay_btn.clicked.connect(self._load_measured)
        self.overlay_clear = QPushButton("Clear")
        self.overlay_clear.clicked.connect(self._clear_measured)
        self.overlay_label = QLabel("No measured curve")
        self.overlay_label.setWordWrap(True)
        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 0, 0, 0)
        overlay_row.addWidget(self.overlay_btn, 1)
        overlay_row.addWidget(self.overlay_clear)
        traces = QWidget()
        traces_l = QVBoxLayout(traces)
        traces_l.setContentsMargins(0, 0, 0, 0)
        traces_l.addWidget(self.trace_list, 1)
        traces_l.addWidget(self.spi_dp_line)
        traces_l.addLayout(overlay_row)
        traces_l.addWidget(self.overlay_label)
        mid.addWidget(traces)
        mid.addWidget(self.plot)
        mid.setSizes([160, 600])
        plots = QWidget()
        plots_l = QVBoxLayout(plots)
        plots_l.setContentsMargins(0, 0, 0, 0)
        # Under the plots: the motor drawing or, for an STL grain, the grain in 3D, both at the time hovered.
        self.motor_panel = MotorPanel()
        self.run_grain = RunGrainView(self.grain_scene, lambda: self.display_units)
        self._panel = QStackedWidget()
        self._panel.addWidget(self.motor_panel)
        self._panel.addWidget(self.run_grain)
        self._panel_switch = QWidget()
        switch_l = QHBoxLayout(self._panel_switch)
        switch_l.setContentsMargins(0, 0, 0, 0)
        switch_l.setSpacing(0)
        choice = QButtonGroup(self._panel_switch)
        for text in ("Motor drawing", "Grain"):
            button = QPushButton(text)
            button.setObjectName("switch")
            button.setCheckable(True)
            choice.addButton(button)
            switch_l.addWidget(button)
        self._show_grain = cast(QPushButton, choice.buttons()[1])
        (self._show_grain if self._prefs.value("simulationPanel") == "grain" else choice.buttons()[0]).setChecked(True)
        self._show_grain.toggled.connect(self._on_panel_choice)
        self._panel_switch.hide()  # until an STL grain is in
        readout = QHBoxLayout()
        readout.addWidget(self.plot_readout, 1)
        readout.addWidget(self._panel_switch, 0, Qt.AlignmentFlag.AlignBottom)
        plots_l.addWidget(mid, 1)
        plots_l.addLayout(readout)
        plots.setMinimumHeight(260)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.summary.setMinimumHeight(60)
        # Drag the dividers to trade plot height for the diagram or the summary.
        column = QSplitter(Qt.Orientation.Vertical)
        column.setChildrenCollapsible(False)
        for part, stretch in ((plots, 5), (self._panel, 3), (self.summary, 2)):
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
                "live_chem": self.live_chem.isChecked(),
            },
            "export_OD": self.dry_OD.si("length"),
            "export_L": self.dry_L.si("length"),
            "sizing": self.sizing_page.targets_cfg(),
        })
        if getattr(self, "_tank_len_custom", False):
            cfg["tnk_L"] = self.tnk_L.spin.value()
            cfg["tnk_L_unit"] = self.tnk_L.unit.currentText()
        if getattr(self, "_tca_len_custom", False):
            cfg["cmbr_L"] = self.cmbr_L.spin.value()
            cfg["cmbr_L_unit"] = self.cmbr_L.unit.currentText()
        return cfg

    def _cfg_to_form(self, cfg: dict):
        self.name.setText(str(cfg.get("mtr_nm") or ""))
        self.solve_tank_cooling.setChecked(bool(cfg.get("solve_tank_cooling")))
        self.mp_on.setChecked(bool(cfg.get("mp_state")))
        self._legacy_mtr_m = float(cfg.get("mtr_m") or 0.0)
        self._legacy_mtr_m_unit = str(cfg.get("mtr_m_unit") or "kg")
        self._legacy_mtr_cg = float(cfg.get("mtr_cg") or 0.0)
        self._legacy_mtr_cg_unit = str(cfg.get("mtr_cg_unit") or "in")
        lay = resolve_layout({**cfg, "grn_shape": "round"})  # only the tank and chamber positions, which the grain doesn't move
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
        self._sync_advanced()
        if adv.get("ox_fluid"):
            self.ox_fluid.setCurrentText(str(adv["ox_fluid"]))
        self.live_chem.setChecked(bool(adv.get("live_chem")))
        self.sizing_page.load_motor(cfg)
        self.sizing_page.set_targets(cfg.get("sizing") or {}, cfg)
        self._tank_len_custom = False
        self._tca_len_custom = False
        self._sync_length_fields()
        self.sizing_page.refresh()
        self._apply_saved_lengths(cfg)
        self._on_motor_edited()

    def _on_motor_edited(self):
        self._sync_advanced()
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

    def _sync_grain_view(self):
        """The Grain tab shows while the grain comes from an STL: the grain, its loading, or what's in the way.
        The Simulation tab can show a loaded STL grain instead of the motor drawing."""
        sp, gv = self.sizing_page, self.grain_view
        self.tabs.setTabVisible(self.tabs.indexOf(gv), sp.grain_shape.currentData() == "stl")
        if sp.grain_geom is not None:
            gv.set_grain(sp.grain_geom)
            self._panel_switch.show()
            if self._grain_was_loading:
                self.statusBar().showMessage(f"Loaded {sp.grain_name()} in {sp.grain_load_seconds():.0f} s.")
        elif sp.grain_loading_shown():
            gv.set_loading(sp.grain_name(), sp.grain_mesh)
            self._panel_switch.hide()
            self._show_grain_progress()
        elif not sp.grain_loading():  # while it loads, a grain from the cache is a moment away: leave things be
            gv.set_message(sp.grain_message(), "Load again" if sp.grain_stopped() else "Open the Motor tab")
            self._panel_switch.hide()
            if self._grain_was_loading or sp.grain_failed():
                self.statusBar().showMessage(sp.grain_message())
        self._grain_was_loading = sp.grain_loading_shown()
        self._sync_panel()

    def _on_panel_choice(self):
        self._prefs.setValue("simulationPanel", "grain" if self._show_grain.isChecked() else "drawing")
        self._sync_panel()

    def _sync_panel(self):
        grain = not self._panel_switch.isHidden() and self._show_grain.isChecked()
        self._panel.setCurrentWidget(self.run_grain if grain else self.motor_panel)

    def _load_learned_timing(self):
        """How the last cold STL loads' time was spread over their steps, for the time left on the next."""
        try:
            shares = tuple(float(x) for x in str(self._prefs.value("grainLoadTimeShares") or "").split(","))
        except ValueError:
            return
        if len(shares) == len(self.sizing_page.load_time_shares) and list(shares) == sorted(shares):
            self.sizing_page.load_time_shares = shares

    def _on_grain_message_action(self):
        if self.sizing_page.grain_stopped():
            self.sizing_page.load_grain_again()
        else:
            self.tabs.setCurrentWidget(self.sizing_page)

    def _show_grain_progress(self):
        sp = self.sizing_page
        if sp.grain_loading_shown():
            text = sp.grain_progress_text()
            self.grain_view.set_progress(text, sp.grain_fraction())
            self.statusBar().showMessage(f"Loading {sp.grain_name()}: {text}")

    def _on_applied(self):
        self.statusBar().showMessage("Applied to motor.")
        self.tabs.setCurrentWidget(self._sim_page)

    def _connect_derived(self):
        for row in (self.tnk_start, self.tnk_m, self.cmbr_start, self.cmbr_m, self.P_cmbr):
            row.spin.valueChanged.connect(self._update_derived_labels)
            row.unit.currentTextChanged.connect(self._update_derived_labels)
        for row in (self.tnk_L, self.cmbr_L):
            row.spin.valueChanged.connect(self._on_length_edited)
            row.unit.currentTextChanged.connect(self._on_length_edited)
        self.name.textChanged.connect(self._update_derived_labels)

    def _design_lengths(self) -> tuple[float, float]:
        """Tank length from volume and diameter, and TCA length from the sized parts, both in meters."""
        sp = self.sizing_page
        tnk_L, _tnk_D, _V = sp.tank_geometry()
        throat, exit_D, _er = sp.nozzle_size()
        lay = motor_layout(
            tnk_start=0.0,
            tnk_L=tnk_L,
            cmbr_start=0.0,
            pre_L=sp.pre_L.si("length"),
            post_L=sp.post_L.si("length"),
            grn_L=sp.grain_L.si("length"),
            grn_OD=sp.grain_OD.si("length"),
            noz_thrt=throat,
            noz_exit=exit_D,
        )
        return tnk_L, lay.cmbr_L

    def _sync_length_fields(self):
        """Keep the length fields on the sized motor until someone types a different length."""
        if not hasattr(self, "tnk_L"):
            return
        design_tank, design_tca = self._design_lengths()
        if not self._tank_len_custom:
            self.tnk_L.set_si(design_tank, "length")
        if not self._tca_len_custom:
            self.cmbr_L.set_si(design_tca, "length")

    def _apply_saved_lengths(self, cfg: dict):
        design_tank, design_tca = self._design_lengths()
        stored_tank = to_si(float(cfg.get("tnk_L") or 0.0), cfg.get("tnk_L_unit") or "m", "length")
        self._tank_len_custom = stored_tank > 1e-6 and abs(stored_tank - design_tank) > 1e-4
        if self._tank_len_custom:
            self.tnk_L.set_display(float(cfg.get("tnk_L") or 0.0), cfg.get("tnk_L_unit") or "in")
        stored_tca = 0.0 if cfg.get("cmbr_L") in (None, "") else to_si(
            float(cfg.get("cmbr_L") or 0.0), cfg.get("cmbr_L_unit") or "m", "length")
        self._tca_len_custom = stored_tca > 1e-6 and abs(stored_tca - design_tca) > 1e-4
        if self._tca_len_custom:
            self.cmbr_L.set_display(float(cfg["cmbr_L"]), cfg.get("cmbr_L_unit") or "in")
        else:
            self.cmbr_L.set_si(design_tca, "length")

    def _on_length_edited(self):
        design_tank, design_tca = self._design_lengths()
        self._tank_len_custom = abs(self.tnk_L.si("length") - design_tank) > 1e-4
        self._tca_len_custom = abs(self.cmbr_L.si("length") - design_tca) > 1e-4
        self._update_derived_labels()

    def _update_derived_labels(self):
        try:
            self._sync_length_fields()
            lay = self._form_layout()
            empty_m, empty_cg = self._empty_mass_si(lay)
            u = self.display_units
            self.mass_info.setText(
                f"Empty mass {u.text(empty_m, 'mass')} at CG {u.text(empty_cg, 'length')}. "
                f"Tank {u.text(lay.tnk_L, 'length')}, TCA {u.text(lay.cmbr_L, 'length')}, "
                f"overall {u.text(lay.overall_L, 'length')}."
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
        design_tank, _design_tca = self._design_lengths()
        tnk_D = sp.tank_geometry()[1]
        throat, exit_D, _er = sp.nozzle_size()
        tnk_L = self.tnk_L.si("length") if self._tank_len_custom else design_tank
        return motor_layout(
            tnk_start=self.tnk_start.si("length"),
            tnk_L=tnk_L,
            tnk_m=self.tnk_m.si("mass"),
            tnk_D=tnk_D,
            cmbr_start=self.cmbr_start.si("length"),
            pre_L=sp.pre_L.si("length"),
            post_L=sp.post_L.si("length"),
            cmbr_L=self.cmbr_L.si("length") if self._tca_len_custom else 0.0,
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
        inj_Cd = sp.cd()
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
        stl = sp.grain_shape.currentData() == "stl"
        web = case = 0.0

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
            if (g := getattr(self._settings, "grain_geometry", None)) is not None:
                web = float(o.grn_web[i])
                case = float(np.interp(web, g.web, g.case_area))

        u = self.display_units
        lay = self._form_layout()
        def size(diameter: float, length: float) -> str:
            return f"Ø{u.value(diameter, 'length'):.3g} × {u.text(length, 'length', 3)}"

        def at_time(text: str) -> str:  # flows and thrust only mean something at a time in a run
            return text if time_s is not None else "—"

        mass_pct = 100.0 * m_f / m_f0 if m_f0 > 1e-12 else 0.0
        dp_inj = P_tnk - P_cmbr
        vnt_on = _vent_visible(vnt, vnt_D)
        vent_card = ("Vent", (("Orifice", f"Ø{u.text(vnt_D, 'length', 3)}"),)) if vnt_on else None
        tank_card = ("Oxidizer tank", (
            ("Size", size(tnk_D, lay.tnk_L)),
            ("Volume", u.text(tnk_V, "volume", 3)),
            ("Mass", f"{u.text(m_o, 'mass', 3)} ({100.0 * fill:.0f}%)"),
            ("Temperature", u.text(T, "temperature", 3)),
        ))
        swirler = sp.inj_type.currentText() == "Swirler"
        inj_card = ("Injector", (
            ("Swirlers" if swirler else "Holes", f"{inj_N} × Ø{u.text(inj_D, 'length', 3)}"),
            ("Cd", f"{inj_Cd:.3g}"),
            ("Area", u.text(inj_A, "area", 3)),
            ("Oxidizer flow", at_time(u.text(ox_mdot, "mass_flow", 3))),
            ("ΔP", at_time(u.text(dp_inj, "pressure", 3))),
            ("Stiffness", at_time(f"{100.0 * dp_inj / P_cmbr:.0f}%" if P_cmbr > 1e-9 else "—")),
        ))
        grain_card = ("Fuel grain (STL)" if stl else "Fuel grain", (
            ("Size", size(grn_OD, grn_L)),
            *((("Burned", u.text(web, "length", 3)), ("Same-area port", u.text(grn_ID, "length", 3))) if stl else
              (("Port", u.text(grn_ID, "length", 3)),)),
            ("Mass", f"{u.text(m_f, 'mass', 3)} ({mass_pct:.0f}%)"),
            ("Fuel flow", at_time(u.text(fuel_mdot, "mass_flow", 3))),
            *((("Case wall exposed", u.text(case, "area", 3)),) if case > 0 else ()),
        ))
        noz_card = ("Nozzle", (
            ("Throat", f"Ø{u.text(th, 'length', 3)}"),
            ("Exit", f"Ø{u.text(exit_d, 'length', 3)}"),
            ("Expansion ratio", f"{er:.3g}"),
            ("O/F", f"{of_ratio:.3g}" if math.isfinite(of_ratio) else "—"),
            ("Flow", at_time(u.text(noz_mdot, "mass_flow", 3))),
            ("Thrust", at_time(u.text(thrust, "force", 3))),
        ))

        drawn_fill = fill * (tnk_L / lay.tnk_L) if lay.tnk_L > 1e-9 else fill
        return MotorView(
            tnk_L=lay.tnk_L,
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
            fill_frac=min(max(drawn_fill, 0.0), 1.0),
            name=self.name.text().strip() or "motor",
            tnk_start=lay.tnk0,
            cmbr_start=lay.cmbr0,
            pre_L=sp.pre_L.si("length"),
            post_L=sp.post_L.si("length"),
            cmbr_L=lay.cmbr_L,
            tnk_dry_kg=lay.tnk_m,
            cmbr_dry_kg=lay.cmbr_m,
            tank_card=tank_card,
            inj_card=inj_card,
            grain_card=grain_card,
            noz_card=noz_card,
            vent_card=vent_card,
            overlay_tank=(f"{100.0 * fill:.0f}%", u.text(P_tnk, "pressure", 3)),
            overlay_grain=(f"{mass_pct:.0f}%", u.text(P_cmbr, "pressure", 3)),
            time_s=time_s,
            display_units=u,
        )

    def _refresh_viz(self):
        if not hasattr(self, "motor_panel"):
            return
        if hasattr(self, "mass_preview"):
            self.mass_preview.set_model(self._motor_view(None))
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
        if self._output is None and self._measured is None:
            return
        plot = self._plots[quantity]
        vb = cast(pg.ViewBox, plot.vb)
        if not vb.sceneBoundingRect().contains(pos):
            self._reset_viz_to_start()
            return
        x = vb.mapSceneToView(pos).x()
        t_show = x
        if self._output is not None:
            t = np.asarray(self._output.t)
            if t.size == 0:
                return
            i = int(np.clip(np.searchsorted(t, x), 0, t.size - 1))
            t_show = float(t[i])
            self._refresh_viz_at(i)
        for line in self._hover_lines:
            line.setValue(t_show)
            line.setVisible(True)
        bits = [f"Time: {t_show:.3f} s"]
        if self._output is not None:
            for k, (_label, key, trace_quantity) in enumerate(TRACES):
                item = self.trace_list.item(k)
                values = _curve(self._output, self._settings, key)
                if item.checkState() == Qt.CheckState.Checked and not item.isHidden() and values is not None:
                    bits.append(f"{item.text()}: {self.display_units.text(float(values[i]), trace_quantity)}")
            self.run_grain.show_at(t_show, float(self._output.grn_web[i]))
        if self._measured is not None:
            for label, key, trace_quantity in TRACES:
                series = self._measured.series(key)
                if series is None:
                    continue
                value = float(np.interp(t_show, self._measured.t, series, left=np.nan, right=np.nan))
                if np.isfinite(value):
                    bits.append(f"{self._measured.label} {label}: {self.display_units.text(value, trace_quantity)}")
        self.plot_readout.setText("   ·   ".join(bits))

    def _refresh_viz_at(self, index: int | None):
        self._hover_index = index
        self._refresh_viz()

    def _sync_advanced(self, *_):
        """The advanced options only apply with Enable advanced options ticked, so grey them out otherwise."""
        on = self.adv_on.isChecked()
        controls = (self.live_chem, self.ox_fluid)
        tips = self.__dict__.setdefault("_advanced_tips", {w: w.toolTip() for w in controls})
        for w in controls:
            w.setEnabled(on)
            w.setToolTip(tips[w] if on else "Tick Enable advanced options to use this.")
        if on and self.sizing_page.inj_model.currentText() != "SPI":  # HEM and Dyer always use CoolProp
            self.ox_fluid.setEnabled(False)
            self.ox_fluid.setToolTip("HEM and Dyer always use CoolProp nitrous properties.")

    def _sync_run_button(self):
        """Grey the Run button while the results already match the inputs; it still runs if clicked."""
        fresh = self._output is not None
        self.run_btn.setProperty("upToDate", fresh)
        self.run_btn.setToolTip("These results match the current inputs." if fresh else "")
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)

    def _invalidate_results(self):
        if self._output is None:
            return
        self._output = self._settings = self._state = self._result_cfg = None
        self._hover_index = None
        self.limit_warning.hide()
        self.summary.clear()
        self._clear_plot()
        self._refresh_viz()
        self.run_grain.set_run(False)
        self._sync_run_button()
        self.statusBar().showMessage("Inputs changed, run again to update.")

    def _run(self):
        if self._thread is not None:
            return
        if self.sizing_page.grain_loading():
            self.statusBar().showMessage("The grain STL is still loading. Run once it's in.")
            return
        self._running_cfg = self._form_to_cfg()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.show()
        self.statusBar().showMessage("Running simulation…")
        self._thread = QThread()
        self._sync_busy()
        self._worker = SimWorker(self._running_cfg, self.sizing_page.grain_tables())
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
        self._sync_busy()
        if self._close_when_finished:
            self.close()

    def closeEvent(self, event):
        study = self.study_page
        if study.busy():
            study.stop()
            study.finished.connect(self.close)
            self.statusBar().showMessage("Closing when the running sweep simulations finish…")
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
        self._add_run(self._result_cfg, s, x, o)
        self._show_results()

    def _runs_dir(self, motor: OpenMotor | None) -> Path:
        """Where a motor's runs are kept: a runs folder in its export folder, or the app's own folder until it has a file."""
        return Path(motor.path).with_suffix("") / "runs" if motor is not None and motor.path else self._session_dir / "runs"

    def _add_run(self, cfg: dict, s, x, o):
        """Keep a finished run in the Runs list and in its file, labeled with what changed since the motor's last run."""
        mine = self._motor_runs()
        same = next((r for r in mine if _same_cfg(r.cfg, cfg)), None)
        if same is not None:  # rerunning an earlier run's settings shows that run instead of adding a copy
            same.output, same.state, same.settings = o, x, s
            self._select_run(same)
            return
        name = str(cfg.get("mtr_nm") or "motor")
        previous = mine[-1] if mine else None
        changes = _changed_settings(previous.cfg, cfg) if previous else []
        what = (", ".join(changes[:3]) + (f" +{len(changes) - 3} more" if len(changes) > 3 else "")) if changes else (
            "no changes" if previous else "first run")
        file, number = new_run_file(self._runs_dir(self._shown), "sim")
        info = summarize(s, x, o)
        results = {"total_impulse": info["total_impulse"], "peak_P_cmbr": info["peak_pressure_bar"] * 1e5,
                   "burn_time": info["burn_time"]}
        run = SimRun(file, f"{number}. {name}: {what}", cfg, tuple(changes), results,
                     datetime.now().isoformat(timespec="seconds"), o, x, s)
        run.save()
        self._sim_runs.append(run)
        self._add_run_item(run)
        self._select_run(run)

    def _with_curves(self, sim_run: SimRun, then: Callable[[], None]):
        """Do something with a run's curves, once it's simulated again if it was read from its file."""
        if sim_run.output is not None:
            then()
            return
        waiting = self._waiting_runs.setdefault(sim_run.file, [])
        if then not in waiting:
            waiting.append(then)
        if len(waiting) == 1:
            self._resimulator.start(sim_run)
            stl = " Its grain STL can take a minute to load." if sim_run.cfg.get("grn_shape") == "stl" else ""
            self.statusBar().showMessage(f"Simulating {sim_run.label} again…{stl}")

    def _on_resimulated(self, sim_run: SimRun, result):
        waiting = self._waiting_runs.pop(sim_run.file, [])
        if isinstance(result, str):
            self.statusBar().showMessage(f"Couldn't simulate {sim_run.label} again: {result}")
        elif sim_run in self._sim_runs:  # not deleted meanwhile
            sim_run.settings, sim_run.state, sim_run.output = result
            self.statusBar().clearMessage()
            for then in waiting:
                then()

    def _sync_runs(self):
        """Read the saved runs of every open motor, keeping the ones already read (with their curves and ticks), and
        list the shown motor's."""
        listed = {r.file: r for r in self._sim_runs}
        runs = (listed.get(p) or SimRun.read(p) for p in run_files([self._runs_dir(m) for m in self._motors()], "sim"))
        self._sim_runs = sorted((r for r in runs if r is not None), key=lambda r: r.time)
        self._list_runs()
        self._refresh_plot()

    def _motor_name(self) -> str | None:
        """How the shown motor's runs are told apart when it has no file, as those share the app's runs folder."""
        return None if self._shown is not None and self._shown.path else self.name.text() or "mtr_cfg"

    def _motor_runs(self) -> list[SimRun]:
        """The shown motor's runs: the ones in its runs folder, or with its name when it has no file."""
        folder, name = self._runs_dir(self._shown), self._motor_name()
        return [r for r in self._sim_runs if r.file.parent == folder and name in (None, r.cfg.get("mtr_nm"))]

    def _list_runs(self):
        """The Runs list and the Sweep tab's saved sweeps show only the shown motor's."""
        self.run_list.clear()
        for r in self._motor_runs():
            self._add_run_item(r)
        shown = next((r for r in self._motor_runs() if r.output is not None and r.output is self._output), None)
        if shown is not None:
            self._select_run(shown)
        self.study_page.sync_runs(self._runs_dir(self._shown), self._motor_name())

    def _select_run(self, run: SimRun):
        """Highlight the run whose settings and results are shown."""
        for i in range(self.run_list.count()):
            if self.run_list.item(i).data(Qt.ItemDataRole.UserRole) == str(run.file):
                self.run_list.blockSignals(True)
                self.run_list.setCurrentRow(i)
                self.run_list.blockSignals(False)

    def _on_run_clicked(self, item: QListWidgetItem):
        if item.checkState() != getattr(self, "_pressed_check", item.checkState()):
            return  # the click ticked or unticked it
        run = self._run_of(item)
        if run is not None:
            self._load_run(run)

    def _load_run(self, run: SimRun):
        """Put a run's settings back into its motor and show its results, to compare or carry on from it."""
        self._wanted_run = run
        if self._thread is not None:
            return
        if run.output is None:
            motor, cfg = self._shown, self._form_to_cfg()

            def show():  # never over newer edits, another motor, or another run clicked meanwhile
                if self._wanted_run is not run:
                    return
                if self._shown is motor and _same_cfg(self._form_to_cfg(), cfg):
                    self._load_run(run)
                else:
                    self.statusBar().showMessage(f"{run.label} is simulated again. Click it to show it.")
            self._with_curves(run, show)
            return
        self._output = self._settings = self._state = self._result_cfg = None
        self._cfg_to_form(run.cfg)
        self._update_tab_marker()
        self._settings, self._state = run.settings, run.state
        self._output, self._result_cfg = run.output, run.cfg
        self._show_results()
        self._select_run(run)
        message = f"Loaded {run.label}. Save to keep these settings in the motor file."
        then, now = run.results.get("total_impulse"), summarize(self._settings, run.state, run.output)["total_impulse"]
        if then and abs(now - then) > 1e-3 * then:
            u = self.display_units
            message += (f" Its total impulse was {u.text(then, 'impulse')} when first run and is {u.text(now, 'impulse')} "
                        "now, so the simulation has changed since.")
        self.statusBar().showMessage(message)

    def _add_run_item(self, run: SimRun):
        item = QListWidgetItem(run.label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
        item.setCheckState(Qt.CheckState.Checked if run.ticked else Qt.CheckState.Unchecked)
        item.setData(Qt.ItemDataRole.UserRole, str(run.file))
        item.setToolTip("\n".join([*run.changes, str(run.file)]))
        self.run_list.blockSignals(True)
        self.run_list.insertItem(0, item)  # newest first
        self.run_list.blockSignals(False)

    def _run_of(self, item: QListWidgetItem) -> SimRun | None:
        return next((r for r in self._sim_runs if str(r.file) == item.data(Qt.ItemDataRole.UserRole)), None)

    def _on_run_item_changed(self, item: QListWidgetItem):
        run = self._run_of(item)
        if run is not None:
            run.ticked = item.checkState() == Qt.CheckState.Checked
        if run is not None and item.text().strip() != run.label:
            if item.text().strip():
                run.label = item.text().strip()
                run.save()
            else:
                self.run_list.blockSignals(True)
                item.setText(run.label)  # an empty name keeps the old one
                self.run_list.blockSignals(False)
        self._refresh_plot()

    def _run_menu(self, pos):
        item = self.run_list.itemAt(pos)
        run = self._run_of(item) if item is not None else None
        if run is None:
            return
        menu = QMenu(self)
        rename, csv, delete = menu.addAction("Rename"), menu.addAction("Export CSV…"), menu.addAction("Delete")
        chosen = menu.exec(self.run_list.viewport().mapToGlobal(pos))
        if chosen is rename:
            self.run_list.editItem(item)
        elif chosen is csv:
            self._with_curves(run, lambda: self._export_run(run))
        elif chosen is delete:
            shown = run.output is not None and run.output is self._output
            self._sim_runs.remove(run)
            delete_run(run.file)
            self.run_list.takeItem(self.run_list.row(item))
            mine = self._motor_runs()
            if shown and mine:
                self._load_run(mine[-1])  # the deleted run was on screen: show the newest one left
            elif shown:
                self._invalidate_results()
            else:
                self._refresh_plot()

    def _export_run(self, run: SimRun):
        name = "".join(c if c.isalnum() or c in " -_." else "_" for c in run.label).strip() or "run"
        path, _ = QFileDialog.getSaveFileName(self, "Export run", self._dialog_path(f"{name}.csv"), "CSV (*.csv)")
        if path:
            self._remember_file_dir(path)
            export_csv(path, run.output)
            self.statusBar().showMessage(f"Saved {path}")

    def _clear_runs(self):
        mine = self._motor_runs()
        if not mine or QMessageBox.question(
                self, APP_NAME, f"Delete all {len(mine)} runs in the list, and their files?") != QMessageBox.StandardButton.Yes:
            return
        for r in mine:
            delete_run(r.file)
            self._sim_runs.remove(r)
        self.run_list.clear()
        self._refresh_plot()

    def _overlay_runs(self) -> list[SimRun]:
        """Ticked runs other than the one shown. Ones read from their files show once they're simulated again."""
        ticked = [r for r in self._motor_runs() if r.ticked]
        for r in ticked:
            if r.output is None:
                self._with_curves(r, self._refresh_plot)
        return [r for r in ticked if r.output is not None and r.output is not self._output]

    def _sync_overlay_ui(self):
        loaded = self._measured is not None
        if hasattr(self, "overlay_clear"):
            self.overlay_clear.setEnabled(loaded)
            self.overlay_label.setText(self._measured.label if loaded else "No measured curve")
        for action in self._file_menu.actions():
            if action.text() == "Clear measured overlay":
                action.setEnabled(loaded)

    def _reload_measured(self, path: str, *, quiet: bool = False) -> bool:
        if not path:
            self._measured = None
            self._prefs.setValue("measuredOverlay", "")
            self._sync_overlay_ui()
            return False
        try:
            self._measured = load_measured_csv(path)
        except (OSError, ValueError) as exc:
            self._measured = None
            if not quiet:
                QMessageBox.warning(self, "Overlay CSV", str(exc))
            self._sync_overlay_ui()
            return False
        self._prefs.setValue("measuredOverlay", path)
        self._remember_file_dir(path)
        self._sync_overlay_ui()
        self._refresh_plot()
        bits = []
        if self._measured.F_thr is not None:
            bits.append("thrust")
        if self._measured.P_tnk is not None:
            bits.append("tank pressure")
        if self._measured.P_cmbr is not None:
            bits.append("chamber pressure")
        self.statusBar().showMessage(f"Overlaying {self._measured.label} ({', '.join(bits)}).")
        return True

    def _load_measured(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Overlay measured curve", self._dialog_path(),
            "CSV (*.csv);;All files (*.*)")
        if path:
            self._reload_measured(path)

    def _clear_measured(self):
        if self._measured is None:
            return
        self._measured = None
        self._prefs.setValue("measuredOverlay", "")
        self._sync_overlay_ui()
        self._refresh_plot()
        self.statusBar().showMessage("Cleared measured overlay.")

    def _show_results(self):
        s, x, o = cast(Settings, self._settings), cast(State, self._state), self._output
        self._hover_index = None
        info = summarize(s, x, o)
        self.summary.setPlainText(format_summary(info, self.display_units))
        self._refresh_plot()
        self._refresh_viz()
        self.run_grain.set_run(True)
        u = self.display_units
        peak, limit = float(np.max(o.P_cmbr)), chamber_limit(cast(dict, self._result_cfg))
        over = peak > limit
        self.limit_warning.setText(f"Peak chamber pressure {u.text(peak, 'pressure')} is above the "
                                   f"{u.text(limit, 'pressure')} chamber pressure limit.")
        self.limit_warning.setVisible(over)
        self.statusBar().showMessage(f"Done: {o.sim_end_cond}. Total impulse {u.text(info['total_impulse'], 'impulse')}"
                                     + ("  ⚠ Over the chamber pressure limit" if over else ""))
        self._sync_run_button()

    def _on_failed(self, msg: str):
        self._stop_progress()
        self.statusBar().showMessage("Simulation failed")
        QMessageBox.critical(self, "Simulation error", msg)

    def _colors(self) -> dict:
        return colors_for(getattr(self, "_theme", "dark"))

    def _hover_line_pen(self):
        return pg.mkPen(self._colors()["accent"], width=1, style=Qt.PenStyle.DashLine)

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

    def _measured_pen(self):
        return pg.mkPen(self._colors()["plot_fg"], width=2.5)

    def _refresh_plot(self):
        o = self._output
        overlays = self._overlay_runs()
        measured = self._measured
        if o is None and not overlays and measured is None:
            self._clear_plot()
            return
        current = self.plot.currentWidget()
        active_quantity = next((q for q, widget in self._plot_widgets.items() if widget is current), None)
        if self._plotted_output is not o or o is None or self._plotted_measured is not measured:
            shown_t = []
            if o is not None:
                shown_t.append(o.t)
            shown_t.extend(r.output.t for r in overlays)
            if measured is not None:
                shown_t.append(measured.t)
            self._time_range = (min(float(a[0]) for a in shown_t), max(float(a[-1]) for a in shown_t))
        self._clear_plot()
        colors = self._colors()
        runs = [self._settings, *(r.settings for r in overlays)]
        stl = any(getattr(settings, "grain_geometry", None) is not None for settings in runs)
        self.trace_list.blockSignals(True)  # renaming an item would redraw the plots midway through this
        for i, (label, key, _quantity) in enumerate(TRACES):
            item = self.trace_list.item(i)
            item.setHidden(key in STL_TRACES and not stl)
            if key == "grn_ID":
                item.setText("Same-area port" if stl else label)
        self.trace_list.blockSignals(False)
        checked = [i for i in range(len(TRACES))
                   if self.trace_list.item(i).checkState() == Qt.CheckState.Checked and not self.trace_list.item(i).isHidden()]
        per_plot: dict[str, int] = {}
        measured_on: set[str] = set()
        for i in checked:
            per_plot[TRACES[i][2]] = per_plot.get(TRACES[i][2], 0) + 1
            if measured is not None and measured.series(TRACES[i][1]) is not None:
                measured_on.add(TRACES[i][2])
        for i in checked:
            _label, key, quantity = TRACES[i]
            label = self.trace_list.item(i).text()
            has_sim = o is not None or bool(overlays)
            has_meas = measured is not None and measured.series(key) is not None
            if not has_sim and not has_meas:
                continue
            unit = self.display_units.unit(quantity)
            if quantity not in self._plots:
                widget = pg.PlotWidget()
                plot = cast(pg.PlotItem, widget.getPlotItem())
                plot.showGrid(x=True, y=True, alpha=0.25)
                if per_plot[quantity] > 1 or overlays or quantity in measured_on:
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
            color = colors[TRACE_COLORS[key]]
            if o is not None and (values := _curve(o, self._settings, key)) is not None:
                self._plots[quantity].plot(o.t, self.display_units.value(values, quantity), pen=pg.mkPen(color, width=2),
                                          name=label)
            dashes = (Qt.PenStyle.DashLine, Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine, Qt.PenStyle.DashDotDotLine)
            for j, overlay in enumerate(overlays):
                if (values := _curve(overlay.output, overlay.settings, key)) is not None:
                    self._plots[quantity].plot(
                        overlay.output.t, self.display_units.value(values, quantity),
                        pen=pg.mkPen(color, width=1.5, style=dashes[j % len(dashes)]), name=f"{label} ({overlay.label})")
            series = measured.series(key) if measured is not None else None
            if series is not None:
                self._plots[quantity].plot(
                    measured.t, self.display_units.value(np.asarray(series, dtype=float), quantity),
                    pen=self._measured_pen(), name=f"{measured.label} {label}")
        cfg = self._result_cfg if self._result_cfg is not None else (overlays[0].cfg if overlays else None)
        if "pressure" in self._plots and cfg is not None:
            limit = self.display_units.value(chamber_limit(cfg), "pressure")
            self._plots["pressure"].addItem(pg.InfiniteLine(
                pos=limit, angle=0, movable=False, pen=pg.mkPen(colors["limit"], width=1, style=Qt.PenStyle.DashLine),
                label="Chamber limit", labelOpts={"position": 0.05, "color": colors["limit"], "anchors": [(0, 1), (0, 1)]}))
            if self.spi_dp_line.isChecked() and uses_spi(cfg):
                dp = self.display_units.value(to_si(300.0, "psi", "pressure"), "pressure")
                self._plots["pressure"].addItem(pg.InfiniteLine(
                    pos=dp, angle=0, movable=False, pen=pg.mkPen(colors["warn"], width=1, style=Qt.PenStyle.DashLine),
                    label="300 psi ΔP (SPI)", labelOpts={"position": 0.05, "color": colors["warn"], "anchors": [(0, 1), (0, 1)]}))
        contact = _case_contact(o, self._settings) if o is not None else None
        for plot in self._plots.values() if contact is not None else ():
            plot.addItem(pg.InfiniteLine(
                pos=contact, angle=90, movable=False, pen=pg.mkPen(colors["limit"], width=1.5, style=Qt.PenStyle.DashLine),
                label=f"Case wall exposed, {contact:.2f} s",
                labelOpts={"position": 0.96, "color": colors["limit"], "anchors": [(-0.03, 0), (-0.03, 0)]}), ignoreBounds=True)
        for plot in self._plots.values():
            vb = cast(pg.ViewBox, plot.vb)
            vb.setXRange(*self._time_range, padding=0)
            vb.sigXRangeChanged.connect(self._sync_time_range)
        if active_quantity in self._plot_widgets:
            self.plot.setCurrentWidget(self._plot_widgets[active_quantity])
        self._plotted_output = o
        self._plotted_measured = measured
        self.plot_readout.setText("Hover over a plot to inspect a time." if self._plots
                                 else "Select a quantity from the list to display its plot.")
        self._style_plots()

    def _style_plots(self):
        colors = self._colors()
        fg = colors["plot_fg"]
        for widget in self._plot_widgets.values():
            widget.setBackground(colors["plot_bg"])
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

    def _output_path(self, filename: str) -> str:
        """Where an export of the shown motor goes by default: a folder named after its file, beside it."""
        motor = self._shown
        if motor is None or not motor.path:
            return self._dialog_path(filename)
        folder = Path(motor.path).with_suffix("")
        folder.mkdir(exist_ok=True)
        return str(folder / filename)

    def _save_output(self, caption: str, filename: str, filters: str) -> str:
        """Ask where to save an export, starting in the motor's output folder; drop the folder if it stays empty."""
        start = self._output_path(filename)
        path, _ = QFileDialog.getSaveFileName(self, caption, start, filters)
        folder = Path(start).parent
        if self._shown is not None and self._shown.path and folder == Path(self._shown.path).with_suffix(""):
            try:
                folder.rmdir()  # only succeeds if nothing was saved there
            except OSError:
                pass
        return path

    def _remember_file_dir(self, path: str) -> None:
        folder = Path(path)
        if not folder.is_dir():
            folder = folder.parent
        if folder.is_dir():
            self._last_dir = str(folder)
            self._prefs.setValue("lastFileDir", self._last_dir)

    @staticmethod
    def _title(cfg: dict, path: str, fallback: str = "Untitled") -> str:
        """A tab shows the motor's Name; a motor without one shows its file name."""
        name = str(cfg.get("mtr_nm") or "").strip()
        if name and name != "mtr_cfg":
            return name
        return Path(path).stem if path else fallback

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
        motor = OpenMotor(path, self._title(cfg, path, title or "Untitled"), cfg)
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
        self._wanted_run = None  # a run clicked on the last motor doesn't load into this one
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
            self.run_grain.set_run(False)
        self._list_runs()
        self._sync_run_button()
        self._update_tab_marker()

    def _open_unsaved(self, cfg):
        """Open settings (a sweep case) as a new, unsaved motor."""
        self._add_motor(cfg, title=cfg["mtr_nm"])
        self._shown.saved = None
        self._update_tab_marker()
        self.tabs.setCurrentWidget(self._sim_page)

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
        motor.title = self._title(motor.cfg, motor.path)
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
        self._sync_runs()

    def _save_motor(self, motor: OpenMotor | None, choose_path: bool = False) -> bool:
        """Save a motor to its file. A new name opens Save As with that name. Returns whether it was saved."""
        if motor is None:
            return False
        if motor is self._shown:
            motor.cfg = self._form_to_cfg()
        path = motor.path
        name = str(motor.cfg.get("mtr_nm") or "").strip()
        saved_name = str((motor.saved or {}).get("mtr_nm") or "").strip()
        renamed = bool(path) and name not in ("", "mtr_cfg") and name != saved_name
        if choose_path or not path or renamed:
            if renamed:
                stem = "".join("_" if c in '<>:"/\\|?*' else c for c in name).strip().rstrip(". ") or "motor"
                start = str(Path(path).with_name(f"{stem}.json"))
            elif path:
                start = path
            else:
                start = self._dialog_path(f"{motor.title}.json")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save As" if choose_path or renamed else "Save motor", start, "HRAP JSON (*.json)")
            if not path:
                return False
            self._remember_file_dir(path)
        path = str(Path(path).resolve())
        try:
            save_json(path, motor.cfg)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save {motor.title} to {path}.\n\n{exc}")
            return False
        moved = path != motor.path
        motor.path, motor.title = path, self._title(motor.cfg, path)
        motor.saved = motor.cfg
        if motor is self._shown:
            self._update_tab_marker()
        else:
            self.motor_tabs.setTabText(self._motors().index(motor), motor.title)
        if moved:
            self._sync_runs()
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
        self._sync_runs()

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

    def _write_session_file(self, name: str, data: bytes):
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._session_dir / f"{name}.tmp"
            tmp.write_bytes(data)
            tmp.replace(self._session_dir / name)
        except OSError:
            pass  # a read-only or full disk only costs the restore

    def _restore_session(self):
        """Reopen last session's motors, unsaved edits, page and window size, and list the open motors' runs.

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
        if 0 <= int(data.get("page", -1)) < self.tabs.count() and self.tabs.isTabVisible(int(data["page"])):
            self.tabs.setCurrentIndex(int(data["page"]))
        if data.get("geometry"):
            self.restoreGeometry(QByteArray.fromBase64(data["geometry"].encode()))
        self._sync_runs()

    def _export(self, kind: str):
        if self._output is None or self._settings is None:
            QMessageBox.information(self, "Export", "Run a simulation first.")
            return
        cfg = cast(dict, self._result_cfg)
        if kind == "csv":
            path = self._save_output("Export CSV", f"{(Path(self._shown.path).stem if self._shown and self._shown.path else 'HRAP_output')}.csv", "CSV (*.csv)")
            if path:
                self._remember_file_dir(path)
                export_csv(path, self._output, self._settings)
        elif kind == "rse":
            stem = cfg["mtr_nm"].strip() or "motor"
            path = self._save_output("Export RSE", f"{stem}.rse", "RSE (*.rse)")
            if path:
                self._remember_file_dir(path)
                export_rse(
                    path,
                    self._output,
                    self._settings,
                    OD=cfg["export_OD"],
                    L=cfg["export_L"],
                    mfg=stem,
                )
        else:
            path = self._save_output("Export ENG", f"{(Path(self._shown.path).stem if self._shown and self._shown.path else 'HRAP_output')}.eng", "ENG (*.eng)")
            if path:
                self._remember_file_dir(path)
                export_eng(
                    path,
                    self._output,
                    self._settings,
                    OD=cfg["export_OD"],
                    L=cfg["export_L"],
                    mfg=cfg["mtr_nm"].strip() or "motor",
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
        self.grain_view.refresh()
        self.run_grain.refresh()
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
        if hasattr(self, "mass_preview"):
            self.mass_preview.set_theme(name)
        if hasattr(self, "sizing_page"):
            self.sizing_page.set_theme(name)
            self.grain_scene.set_theme(name)
            self.grain_view.set_theme()
            self.run_grain.set_theme()

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
        pg.setConfigOptions(antialias=True, background=colors_for()["plot_bg"], foreground=colors_for()["plot_fg"])
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
