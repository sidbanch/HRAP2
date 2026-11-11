"""Motor tab: the whole motor, with each part either entered by hand or sized for targets at the start of the burn."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable, cast

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.nox import nox, saturation_temperature
from hrap.engine.sizing import Sizing, SizingTargets, size_motor
from hrap.engine.swirl import swirl_A, swirl_cd, swirl_exit_D, swirl_fill, swirl_sensitivity, swirl_xi
from hrap.gui.cda_curve import CdaCurve
from hrap.gui.sizing_viz import GrainSketch, InjectorSketch, NozzleSketch
from hrap.gui.swirler_options import Layout, SwirlerOptions, Target, drilled_layout
from hrap.gui.widgets import (
    CARD_GAP,
    COLUMN_GAP,
    FIELD_GAP,
    PAGE_MARGIN,
    UNIT_W,
    Card,
    ElidedLabel,
    FieldGrid,
    PlainComboBox,
    PlainDoubleSpinBox,
    PlainSpinBox,
    UnitRow,
    beside,
    card_frame,
    page_layout,
)
from hrap.io.config import chamber_limit, injector_cd
from hrap.io.propellant import list_propellants, load_propellant
from hrap.units import (
    AREA_ITEMS,
    DENSITY_ITEMS,
    LENGTH_ITEMS,
    PRESSURE_ITEMS,
    TEMP_ITEMS,
    VOLUME_ITEMS,
    DisplayUnits,
    from_si,
    to_si,
)

# m, a very rough estimate. A stock 1/4 in PTC has a 0.188 in hex inside, but its tube stop and collet restrict more:
# HPS01-1's liquid ran out at 5.8 s in the fire video when its four swirlers (6 × 0.100 in holes, offset guessed at
# 0.10 in) exit through a clean hole this size, with the team-standard SPI model, the burn-rate law, tank cooling and
# the tube's measured 3.655 in bore over 45 in (0.118 in with Dyer κ 1). One video timing, so a cold flow of a bare
# stock PTC should replace it.
STOCK_PTC_D = 0.097 * 0.0254


class LoadStopped(Exception):
    """Raised from a load's progress callback to stop it."""


class GrainLoader(QObject):
    """Reads grain STLs off the UI thread, one at a time, as each can take a minute and a lot of memory.
    A newer request stops the one loading, as does stop()."""
    mesh = Signal(object, object)  # request, (vertices, faces): the STL's shape, read in a moment, to show while it loads
    progress = Signal(object, str, float)  # request, stage, share of the whole load done
    done = Signal(object, object)  # request, and its GrainGeometry, an error message, or None if it was stopped
    _read = Signal(object, object)

    def __init__(self):
        super().__init__()
        self._reading: tuple | None = None  # (request, its stop flag)
        self._waiting: tuple | None = None
        self._read.connect(self._on_read)

    def load(self, request: tuple):
        if self._reading is not None:
            reading, stop = self._reading
            if reading == request and not stop.is_set():
                self._waiting = None  # it's loading already
            else:  # stop the one loading; once it has, this one loads (even if it's the same, as it was told to stop)
                stop.set()
                self._waiting = request
            return
        stop = threading.Event()
        self._reading = (request, stop)
        path, units, voxel, ends_burn = request

        def report(stage: str, fraction: float):
            if stop.is_set():
                raise LoadStopped
            self.progress.emit(request, stage, fraction)

        def read_mesh():
            try:
                from hrap.engine.grain_stl import read_grain_mesh
                self.mesh.emit(request, read_grain_mesh(path, units=units))
            except Exception:
                pass  # the full load says what's wrong

        def work():
            try:
                from hrap.engine.grain_stl import load_grain_stl
                result = load_grain_stl(path, units=units, voxel=voxel, ends_burn=ends_burn, progress=report)
            except LoadStopped:
                result = None
            except Exception as exc:
                result = str(exc) or type(exc).__name__
            self._read.emit(request, result)

        threading.Thread(target=read_mesh, daemon=True).start()
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        if self._reading is not None:
            self._reading[1].set()
        waiting, self._waiting = self._waiting, None
        if waiting is not None:
            self.done.emit(waiting, None)  # stopped before it started

    def _on_read(self, request: tuple, result):
        waiting, self._reading, self._waiting = self._waiting, None, None
        if result is None and waiting == request:
            self.load(request)  # stopped for another request, but wanted again since: load it afresh
            return
        self.done.emit(request, result)
        if waiting is not None and waiting != request:
            self.load(waiting)


STOPPED = "Loading stopped."
# Share of a load's time against the engine's own progress fraction, to start from: measured on cold loads of a 20 in
# twisted grain, where filling the cubes and measuring burn distances take nearly all of it. Each cold load updates it.
LOAD_PROGRESS = (0.0, 0.04, 0.22, 0.28, 0.78, 0.88, 1.0)
LOAD_TIME = (0.0, 0.04, 0.48, 0.48, 0.88, 0.95, 1.0)

# What the engine's load errors mean for someone holding an STL, and what to do about them.
_LOAD_ADVICE = (
    ("open edges", "Its surface has holes (open edges). Export the fuel from CAD again as a closed solid."),
    ("triangular", "It isn't a mesh HRAP can read. Export the fuel from CAD again as an STL."),
    ("Align the grain axis", "The grain's axis has to run along the STL's X, Y or Z axis, with a round outside. "
                             "Line it up in CAD and export it again."),
    ("Voxel size is too small", "That cube size needs over 35 million cubes. Make it larger, or set it to Auto."),
    ("check the STL units", "Its size looks wrong for a fuel grain. Check the STL units (in or mm)."),
    ("no open port", "No port was found inside it. Check the file is the fuel grain and its port runs all the way "
                     "through."),
    ("open at both ends", "The port has to run all the way through the grain, open at both ends."),
    ("No fuel is resolved", "The cubes are too big to see the fuel between port and case. Make the cube size smaller."),
    ("port surface", "Its port surface couldn't be found. Check the file is the fuel grain."),
)


def plain_load_error(name: str, message: str) -> str:
    advice = next((text for key, text in _LOAD_ADVICE if key in message), f"The STL reader says: {message}")
    return f"Couldn't load {name}. {advice}"


def mid_section(geom) -> np.ndarray:
    """The grain's cross-section halfway along: 0 outside it, 1 fuel, 2 port."""
    d = geom.distance[:, :, geom.distance.shape[2] // 2].T
    return np.ascontiguousarray(np.where(np.isinf(d), 0, np.where(d > 0, 1, 2)), dtype=np.uint8)


class SizingPage(QWidget):
    motor_edited = Signal()  # a motor setting on this page changed
    sized = Signal()  # the sizing was worked out again, so what Apply to motor would change may differ
    applied = Signal()
    grain_changed = Signal()  # the STL grain is loading (and on screen), loaded, failed, stopped or dropped
    grain_progress = Signal()  # loading went on
    load_timing_learned = Signal()  # load_time_shares changed after a cold load
    show_grain = Signal()  # View in 3D was clicked

    def __init__(self, get_cfg: Callable[[], dict], get_units: Callable[[], DisplayUnits]):
        super().__init__()
        self._get_cfg, self._get_units = get_cfg, get_units
        self._result: Sizing | None = None
        self._built: Sizing | None = None  # the current motor, at the start of the burn
        self._cfg: dict | None = None
        self._layout: Layout | None = None  # the swirler drilled for the flow, when a burn time or O/F sets it
        self._port_error = ""
        self._loading = False
        self._swirler = False
        self.grain_geom = None  # the loaded STL grain (GrainGeometry)
        self._grain_request: tuple | None = None  # what grain_geom is, or is loading, for
        self._grain_error = ""
        self.grain_mesh: tuple | None = None  # the STL's shape (vertices, faces), shown while its burn-back loads
        self._grain_stage = ("", 0.0)  # loading: the step it's on, and the share of the whole load done
        self._grain_started = self._grain_took = 0.0
        self._load_samples: list[tuple[float, float]] = []  # (engine progress, seconds) through this load
        self.load_time_shares = LOAD_TIME  # share of a load's time at each of LOAD_PROGRESS
        self._grain_shown = False  # loading is on screen; a grain in the cache loads too fast to show it
        self._show_loading = QTimer(self, singleShot=True, interval=250)
        self._show_loading.timeout.connect(self._on_slow_load)
        self._tick = QTimer(self, interval=1000)  # the elapsed time and time left move on between progress reports
        self._tick.timeout.connect(self._on_tick)
        self._section: np.ndarray | None = None
        self._grain_loader = GrainLoader()
        self._grain_loader.done.connect(self._on_grain_loaded)
        self._grain_loader.progress.connect(self._on_grain_progress)
        self._grain_loader.mesh.connect(self._on_grain_mesh)
        self._tank_sync = False

        self.P_cmbr = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.burn_time = PlainDoubleSpinBox()
        self.burn_time.setRange(0.1, 120)
        self.burn_time.setDecimals(2)
        self.OF = PlainDoubleSpinBox()
        self.OF.setRange(0.1, 50)
        self.OF.setDecimals(2)
        self.P_cmbr.setToolTip("Chamber pressure at the start of the burn (absolute). It falls as the tank cools.")
        self.burn_time.setToolTip("How long the liquid should last. It sets the oxidizer flow, and the hole count follows.\n"
                                  "The real burn runs a little longer, because the flow drops as the tank cools.")
        self.size_from = PlainComboBox()
        self.size_from.addItems(["Manual", "For liquid burn time", "For O/F"])
        self.size_from.setToolTip("Manual: use the holes or swirlers you enter.\n"
                                  "For liquid burn time: size the injector to empty the liquid in that time.\n"
                                  "For O/F: size the injector for the starting O/F, with the grain length you enter.")
        self.holes = PlainSpinBox()
        self.holes.setRange(1, 200)
        self.holes.setToolTip("Injector hole count. The oxidizer flow and burn time follow from it.")
        self.OF.setToolTip("Oxidizer-to-fuel ratio at the start of the burn. It sets the grain length,\n"
                           "or the oxidizer flow when the injector is sized from O/F.\n"
                           "With the burn-rate law it drifts during the burn.")
        self.grain_from = PlainComboBox()
        self.grain_from.addItems(["Manual", "For O/F"])
        self.grain_from.setToolTip("Manual: use the grain length you enter.\n"
                                   "For O/F: size the grain length for the starting O/F.")
        self.port_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.grain_OD = UnitRow(LENGTH_ITEMS, "in", 4)
        self.grain_L = UnitRow(LENGTH_ITEMS, "in", 4)
        self.port_D.setToolTip("Starting port diameter.")
        self.grain_OD.setToolTip("Grain outer diameter. It's also the chamber bore in the drawings.")
        self.grain_L.setToolTip("Grain length. The fuel flow and starting O/F follow from it.")

        self.P_limit = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.P_limit.setToolTip("The chamber's design pressure (absolute). The Motor, Simulation and Sweep tabs warn above it.")

        # Beside each target, what the current motor gives; clicking one makes it the target.
        self._badges: dict[str, QToolButton] = {}
        for name in ("P_cmbr", "burn_time", "OF"):
            badge = self._badges[name] = QToolButton()
            badge.setObjectName("builtBadge")
            badge.setCursor(Qt.CursorShape.PointingHandCursor)
            badge.hide()
            badge.clicked.connect(lambda _=False, n=name: self._revert([n]))

        targets, tl = card_frame("Targets")
        form = FieldGrid()
        for name, label, field, unit in (("P_cmbr", "Chamber pressure", self.P_cmbr, ""),
                                         ("burn_time", "Liquid burn time", self.burn_time, "s"),
                                         ("OF", "O/F", self.OF, "")):
            setattr(self, f"_{name}_row", form.add(label, field, unit))
            form.add_below(self._badges[name])
        form.add("Pressure limit", self.P_limit)
        tl.addLayout(form)

        # The motor's physical settings live on this page; the Simulation tab only picks models and run settings.
        self.tank_L = UnitRow(LENGTH_ITEMS, "in", 3)
        self.tank_V = UnitRow(VOLUME_ITEMS, "cm^3", 3)
        self.tank_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.tank_T = UnitRow(TEMP_ITEMS, "C", 3)
        self.fill = PlainDoubleSpinBox()
        self.fill.setRange(0, 100)
        self.fill.setDecimals(3)
        self.tank_L.setToolTip("Inside length of the tank. With an inside diameter, volume is π/4 × diameter² × this length.")
        self.tank_D.setToolTip("Inside diameter. With the internal length, it sets the tank volume.")
        self.tank_V.setToolTip("Tank volume. With an inside diameter it follows the internal length.\n"
                               "Type a volume to set the length from it instead.")
        self.tank_T.setToolTip("Starting tank temperature. It sets the tank pressure.")
        self.fill.setToolTip("Share of the tank volume that starts as liquid. The dip tube's length sets it.")
        self.tank_P = QLabel("—")
        self.ox_liquid = QLabel("—")
        self.vent = PlainComboBox()
        self.vent.addItems(["None", "External", "Internal"])
        self.vent.setToolTip("External: an orifice at the top of the tank vents vapor overboard.\n"
                             "Internal: the vent flow goes into the chamber along with the injector flow.")
        self.vent_D = UnitRow(LENGTH_ITEMS, "mm", 4)
        self.vent_Cd = PlainDoubleSpinBox()
        self.vent_Cd.setRange(0, 1)
        self.vent_Cd.setDecimals(3)
        tank, tank_layout = card_frame("Tank")
        tank_form = FieldGrid()
        tank_form.add("Internal length", self.tank_L)
        tank_form.add("Diameter", self.tank_D)
        tank_form.add("Volume", self.tank_V)
        tank_form.add("Temperature", self.tank_T)
        tank_form.add("Fill", self.fill, "%")
        tank_form.add("Tank pressure", self.tank_P, muted=True)
        tank_form.add("Liquid oxidizer", self.ox_liquid, muted=True)
        tank_form.add("Vent", self.vent)
        self._vent_rows = [*tank_form.add("Vent diameter", self.vent_D), *tank_form.add("Vent Cd", self.vent_Cd)]
        tank_layout.addLayout(tank_form)

        self.propellant = PlainComboBox()
        for item in list_propellants():
            plain_id = item["id"].replace("_", " ")  # "ABS", but "50P (HTPB Paraffin)"
            self.propellant.addItem(item["name"] if plain_id == item["name"] else f"{item['name']} ({plain_id})", item["id"])
        # Long propellant names would otherwise widen the whole inputs column.
        self.propellant.setSizeAdjustPolicy(PlainComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.propellant.setMinimumContentsLength(12)
        self.propellant.setToolTip("Picks the combustion table, and fills in the fuel's density and burn rate.")
        self.propellant.currentIndexChanged.connect(self._on_propellant)
        self.rho = UnitRow(DENSITY_ITEMS, "kg/m^3", 1)
        self.rho.setToolTip("Fuel density. A 3D-printed grain can be lighter than solid plastic, so weigh one.")
        self.prop_a, self.prop_n, self.prop_m = PlainDoubleSpinBox(), PlainDoubleSpinBox(), PlainDoubleSpinBox()
        for spin, low in ((self.prop_a, 0.0), (self.prop_n, -2.0), (self.prop_m, -2.0)):
            spin.setDecimals(5)
            spin.setRange(low, 1e3)
        self.prop_a.setToolTip("Burn rate = a × G^n × L^m, in mm/s with the oxidizer flux G in kg/(m²·s) and L in m.\n"
                               "a scales the whole burn rate.")
        self.prop_n.setToolTip("How strongly the burn rate follows oxidizer flux. Usually 0.3–0.8.")
        self.prop_m.setToolTip("Grain-length effect. Almost always 0.")
        self.reg_model = PlainComboBox()
        self.reg_model.addItem("Burn-rate law", "Shifting OF")
        self.reg_model.addItem("Fixed O/F", "Constant OF")
        self.reg_model.setToolTip("Burn-rate law (HRAP's Shifting OF): the fuel burns back at a × G^n × L^m, so the O/F\n"
                                  "drifts as the port opens.\n"
                                  "Fixed O/F (HRAP's Constant OF): fuel flow = oxidizer flow ÷ the O/F below, and a, n and m\n"
                                  "aren't used. Only for motors with no burn-rate data, or to reproduce old HRAP runs.")
        self.const_OF = PlainDoubleSpinBox()
        self.const_OF.setDecimals(3)
        self.const_OF.setRange(0.01, 100)
        self.const_OF.setToolTip("The O/F held for the whole burn.")
        self.cstar = PlainDoubleSpinBox()
        self.cstar.setRange(0, 100)
        self.cstar.setDecimals(1)
        self.cstar.setToolTip("C* efficiency: how completely the propellants burn. Small hybrids are usually 85–95%.")
        fuel, fuel_layout = card_frame("Fuel")
        fuel_form = FieldGrid()
        fuel_form.add("Propellant", self.propellant)
        fuel_form.add("Density", self.rho)
        fuel_form.add("Fuel flow", self.reg_model)
        self._burn_rate_rows = [*fuel_form.add("Burn rate a", self.prop_a), *fuel_form.add("Burn rate n", self.prop_n),
                                *fuel_form.add("Burn rate m", self.prop_m)]
        self._const_OF_row = fuel_form.add("O/F", self.const_OF)
        fuel_form.add("C* efficiency", self.cstar, "%")
        fuel_layout.addLayout(fuel_form)

        self.hole_D = UnitRow(LENGTH_ITEMS, "in", 5)
        self.inj_CdA = UnitRow(AREA_ITEMS, "in^2", 6)
        self.inj_Cd = PlainDoubleSpinBox()
        self.inj_Cd.setRange(0, 5)
        self.inj_Cd.setDecimals(4)
        self.cd_as = PlainComboBox()
        self.cd_as.addItems(["CdA", "Cd"])
        self.cd_as.setFixedWidth(UNIT_W)
        self.cd_as.setToolTip("CdA is the discharge coefficient times the hole area.\n"
                              "Cd is the coefficient alone, on that same area. CdA = Cd × area.")
        self.inj_model = PlainComboBox()
        self.inj_model.addItems(["SPI", "HEM", "Dyer"])
        self.inj_model.setToolTip("SPI: treats the nitrous as liquid all the way through the hole (original HRAP).\n"
                                  "  Overpredicts flow above roughly 300 psi of ΔP.\n"
                                  "HEM: the nitrous boils instantly in the hole. Underpredicts flow.\n"
                                  "Dyer: a blend of the two, weighted by κ. The usual choice for nitrous.\n"
                                  "HEM and Dyer also use CoolProp nitrous properties in the tank.")
        self.inj_Cd_HEM = PlainDoubleSpinBox()
        self.inj_Cd_HEM.setRange(0.01, 1)
        self.inj_Cd_HEM.setDecimals(3)
        self.inj_Cd_HEM.setToolTip("Discharge coefficient for the HEM part. Water flow tests can't measure it. A nitrous cold flow can.")
        self.hem_same = QCheckBox("Same as Cd")
        self.dyer_kappa = PlainDoubleSpinBox()
        self.dyer_kappa.setRange(0.01, 100)
        self.dyer_kappa.setDecimals(2)
        self.dyer_kappa.setToolTip("How the Dyer model weights SPI against HEM. 1 is an even blend, the formula's value for a tank\n"
                                   "at its own vapor pressure. Higher leans toward SPI.")
        self.inj_flow = PlainComboBox()
        self.inj_flow.addItems(["Constant", "Changes over the burn"])
        self.inj_flow.setToolTip("Constant: the holes and their Cd set the injector's flow for the whole burn.\n"
                                 "Changes over the burn: a table of total CdA against time sets it instead, to reproduce a\n"
                                 "fire whose injector flow fell (or rose) as it burned.")
        self.cda_curve = CdaCurve()
        self.inj_type = PlainComboBox()
        self.inj_type.addItems(["Holes", "Swirler"])
        self.inj_type.setToolTip("Holes: straight drilled holes.\n"
                                 "Swirler: tangential ports spin the nitrous in a small chamber before the exit orifice.")
        self.sw_ports = PlainSpinBox()
        self.sw_ports.setRange(1, 12)
        self.sw_D_port = UnitRow(LENGTH_ITEMS, "in", 4)
        self.sw_R_in = UnitRow(LENGTH_ITEMS, "in", 4)
        self.sw_ports.setToolTip("Holes drilled tangentially into the swirler plug.")
        self.sw_D_port.setToolTip("Drill size of each swirler hole.")
        self.sw_R_in.setToolTip("Distance from the swirler's axis to each hole's axis. For holes tangent to the\n"
                                "plug's bore, it's the bore radius minus half a hole.")
        self.ptc_stock = QCheckBox("Stock")
        self.ptc_stock.setToolTip("A stock 1/4 in PTC. Its insides restrict more than its 0.188 in hex, so it's entered as\n"
                                  "the clean hole it flows like. Untick it for a PTC bored out to a size.")
        self.ptc_stock.toggled.connect(self._on_ptc_stock)
        self.sw_xi = PlainDoubleSpinBox()
        self.sw_xi.setRange(0, 100)
        self.sw_xi.setDecimals(2)
        self.sw_xi.setToolTip("Pressure lost entering the swirler holes, as a share of the jets' velocity pressure (Bazarov's ξ).\n"
                              "0 is the ideal theory, which flows the most, so holes sized with it come out small.\n"
                              "A sharp drilled hole is about 1.4. It matters most when the holes are small next to the exit.\n"
                              "Fit it to a cold flow of the swirler.")
        self.sw_CdA_meas = UnitRow(AREA_ITEMS, "in^2", 5)
        self.sw_CdA_meas.setToolTip("One swirler's CdA from a cold flow: water flow ÷ √(2 × 998 kg/m³ × ΔP), with ΔP read right\n"
                                    "at the injector. If the cold flow gives a Cd on the exit area, CdA = Cd × exit area.")
        self.sw_CdA_meas.spin.setSpecialValueText("none yet")  # optional: only once a cold flow measures one
        fit = self._fit_btn = QPushButton("Fit")
        fit.setToolTip("Stock ticked: set what the stock PTC acts like, keeping ξ.\n"
                       "Bored PTC: set the inlet loss ξ, keeping the bore.\n"
                       "Either way the swirl theory then gives the measured CdA.")
        fit.setEnabled(False)
        fit.clicked.connect(self._on_fit)
        self.sw_CdA_meas.spin.valueChanged.connect(lambda cda: fit.setEnabled(cda > 0))
        self.sw_cd_geom = QCheckBox("From geometry")
        self.sw_cd_geom.setToolTip("Work the swirler's Cd out from its geometry (Abramovich's theory for an ideal liquid).\n"
                                   "Untick it to type a CdA measured in a cold flow.")
        inj_form = FieldGrid()
        inj_form.add("Sizing", self.size_from)
        inj_form.add("Type", self.inj_type)
        self._holes_row = inj_form.add("Hole count", self.holes)
        self._hole_D_row = inj_form.add("Hole diameter", beside(self.hole_D, self.ptc_stock), span=True)
        ports_row = inj_form.add("Swirler holes", self.sw_ports)
        self._sw_D_port_row = inj_form.add("Hole diameter", self.sw_D_port)
        self._swirler_rows = [*ports_row, *inj_form.add("Hole offset", self.sw_R_in), *inj_form.add("Inlet loss (ξ)", self.sw_xi),
                              *inj_form.add("Measured CdA", beside(self.sw_CdA_meas, fit), span=True)]
        cd_entry = QWidget()
        cd_lay = QHBoxLayout(cd_entry)
        cd_lay.setContentsMargins(0, 0, 0, 0)
        cd_lay.setSpacing(FIELD_GAP)
        cd_lay.addWidget(self.inj_CdA, 1)
        cd_lay.addWidget(self.inj_Cd, 1)
        cd_lay.addWidget(self.cd_as)
        cd_lay.addWidget(self.sw_cd_geom)
        self.inj_Cd.hide()
        inj_form.add("Injector flow", self.inj_flow)
        self._cd_row = inj_form.add("CdA per hole", cd_entry, span=True)
        self._curve_row = inj_form.add("Total CdA", self.cda_curve, span=True)
        self._curve_row[0].setAlignment(Qt.AlignmentFlag.AlignTop)
        inj_form.add("Flow model", self.inj_model)
        self._hem_row = inj_form.add("HEM Cd", beside(self.inj_Cd_HEM, self.hem_same), span=True)
        self._dyer_row = inj_form.add("Dyer κ", self.dyer_kappa)
        self.show_layouts = QPushButton("Show")
        self.show_layouts.setCheckable(True)
        self.show_layouts.setToolTip("Swirler hole drills that give the target flow through this PTC bore.")
        self.show_layouts.toggled.connect(lambda _on: self._show_size_from_rows())
        self._layouts_row = inj_form.add("Drill layouts", self.show_layouts)
        self.injector = Card("Injector", ["Oxidizer flow", "Injector ΔP", "Stiffness (ΔP / Pc)", "Flow per hole",
                                          "Holes", "Total CdA", "Hole drill", "Swirler Cd", "Limits the flow",
                                          "Liquid lasts"],
                             InjectorSketch(), inj_form, sketch_over_results=True)

        self.noz_Cd = PlainDoubleSpinBox()
        self.noz_Cd.setRange(0, 1)
        self.noz_Cd.setDecimals(3)
        self.noz_Cd.setToolTip("How much of the throat area actually flows. About 0.97–0.99 for a smooth, rounded throat.\n"
                               "Acts like a smaller throat. Put combustion losses in C* efficiency instead.")
        self.noz_eff = PlainDoubleSpinBox()
        self.noz_eff.setRange(0, 100)
        self.noz_eff.setDecimals(1)
        self.noz_eff.setToolTip("Thrust lost to the nozzle's cone angle and friction. It scales thrust only.\n"
                                "A 15° cone loses about 2% to the angle alone. 92–97% is typical.")
        self.Pa = UnitRow(PRESSURE_ITEMS, "atm", 3)
        self.Pa.setToolTip("Outside pressure. A sized expansion ratio matches it. Lower it to model a motor at altitude.")
        self.nozzle_from = PlainComboBox()
        self.nozzle_from.addItems(["Manual", "For chamber pressure"])
        self.nozzle_from.setToolTip("Manual: use the throat and expansion ratio you enter.\n"
                                    "For chamber pressure: size them for the chamber pressure target.")
        self.throat_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.noz_ER = PlainDoubleSpinBox()
        self.noz_ER.setRange(1, 1000)
        self.noz_ER.setDecimals(3)
        self.noz_ER.setToolTip("Exit area ÷ throat area.")
        noz_form = FieldGrid()
        noz_form.add("Sizing", self.nozzle_from)
        self._throat_rows = [*noz_form.add("Throat", self.throat_D), *noz_form.add("Expansion ratio", self.noz_ER)]
        noz_form.add("Throat Cd", self.noz_Cd)
        noz_form.add("Efficiency", self.noz_eff, "%")
        noz_form.add("Ambient pressure", self.Pa)
        self.nozzle = Card("Nozzle", ["Chamber pressure", "Throat diameter", "Expansion ratio", "Exit diameter",
                                      "C*"], NozzleSketch(), noz_form)

        self.pre_L = UnitRow(LENGTH_ITEMS, "in", 4)
        self.post_L = UnitRow(LENGTH_ITEMS, "in", 4)
        self.pre_L.setToolTip("Empty space between the injector plate and the front of the grain.\n"
                              "The chamber's gas volume sets how fast its pressure builds at ignition.")
        self.post_L.setToolTip("Empty space between the back of the grain and the nozzle.")
        self.grain_shape = PlainComboBox()
        self.grain_shape.addItem("Round", "round")
        self.grain_shape.addItem("From STL", "stl")
        self.grain_shape.setToolTip("Round: a straight round port, from the sizes below.\n"
                                    "From STL: any port shape, from a 3D model of the fuel. It burns back evenly from the\n"
                                    "port's whole surface at the burn-rate law's rate.")
        self._stl_path = ""
        self.stl_file = ElidedLabel()
        self.stl_file.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # long names don't widen the card
        load_stl = QPushButton("Load STL…")
        load_stl.setToolTip("A closed (watertight) model of the fuel, with the grain's axis along the STL's X, Y or Z axis.")
        load_stl.clicked.connect(self._pick_stl)
        self.stl_unit = PlainComboBox()
        self.stl_unit.addItems(["in", "mm"])
        self.stl_unit.setToolTip("The unit the STL's numbers are in. STL files don't say, so check the CAD export.")
        self.stl_voxel = PlainDoubleSpinBox()
        self.stl_voxel.setDecimals(3)
        self.stl_voxel.setRange(0, 10)
        self.stl_voxel.setSpecialValueText("Auto")
        self.stl_voxel.setKeyboardTracking(False)  # each change loads the grain again
        self.stl_voxel.setToolTip("Size of the cubes the grain is split into to work out how it burns back. Auto uses about\n"
                                  "24 million cubes, none under 0.5 mm. Half the size is 8 times the cubes and slower to load.")
        self.ends_burn = QCheckBox("Ends burn")
        self.ends_burn.setToolTip("The grain also burns back from its two end faces. Leave it off for inhibited or bonded ends.")
        self.stl_status = QLabel("")
        self.stl_status.setObjectName("cardLabel")
        self.stl_status.setWordWrap(True)
        self.stl_progress = QProgressBar()
        self.stl_progress.setRange(0, 1000)
        self.stl_progress.setTextVisible(False)
        self.stl_progress.setFixedSize(90, 8)
        self._grain_action = QPushButton()  # View in 3D, Cancel or Load again
        self._grain_action.clicked.connect(self._on_grain_action)
        grain_form = FieldGrid()
        grain_form.add("Shape", self.grain_shape)
        self._stl_rows = [*grain_form.add("STL file", beside(self.stl_file, load_stl), span=True),
                          *grain_form.add("STL units", self.stl_unit)]
        voxel_row = grain_form.add("Cube size", self.stl_voxel, "in")
        self._voxel_unit = voxel_row[2]
        status = beside(self.stl_status, beside(self.stl_progress, self._grain_action))
        self._stl_rows += [*voxel_row, *grain_form.add("", self.ends_burn), status]
        grain_form.add_line(status)
        self._grain_from_row = grain_form.add("Sizing", self.grain_from)
        self._grain_L_row = grain_form.add("Grain length", self.grain_L)
        self._round_rows = [*grain_form.add("Starting port", self.port_D), *grain_form.add("Outer diameter", self.grain_OD)]
        # An STL grain's sizes come from its file. The motor's round sizes stay as they are, for a round grain.
        self.stl_L, self.stl_OD, self.stl_port = QLabel("—"), QLabel("—"), QLabel("—")
        self.stl_port.setToolTip("The round port with the same area as the STL's starting port, averaged along the grain\n"
                                 "(its volume ÷ length). The drawings use it.")
        self._stl_size_rows = [*grain_form.add("Length", self.stl_L, muted=True),
                               *grain_form.add("Outer diameter", self.stl_OD, muted=True),
                               *grain_form.add("Same-area port", self.stl_port, muted=True)]
        grain_form.add("Pre-combustion", self.pre_L)
        grain_form.add("Post-combustion", self.post_L)
        self.grain = Card("Grain", ["Fuel flow", "Oxidizer flux", "Burning area", "Grain length", "O/F",
                                    "Port at liquid burnout", "O/F at liquid burnout", "Fuel burned"], GrainSketch(), grain_form)
        self.performance = Card("Start of burn", ["Thrust", "Isp", "Impulse over the burn time"])

        self.limit_warning = QLabel("")
        self.limit_warning.setObjectName("sizingError")
        self.limit_warning.setWordWrap(True)
        self.limit_warning.hide()
        self.error = QLabel("")
        self.error.setObjectName("sizingError")
        self.error.setWordWrap(True)
        self.error.hide()
        self.apply_btn = QPushButton("Apply to motor")
        self.apply_btn.setObjectName("runButton")
        self.apply_btn.setToolTip("Set the motor's throat, expansion ratio, hole count, swirler holes and grain length to the sized ones.")
        self.apply_btn.clicked.connect(self.apply)
        self.apply_summary = QLabel("")
        self.apply_summary.setWordWrap(True)
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.setToolTip("Set the targets to what the current motor gives, instead of changing the motor.")
        self.revert_btn.clicked.connect(lambda: self._revert(list(self._badges)))
        bar = self._apply_bar = QFrame()
        bar.setObjectName("applyBar")
        buttons = QHBoxLayout(bar)
        buttons.setContentsMargins(PAGE_MARGIN, 10, PAGE_MARGIN, 10)
        buttons.addWidget(self.apply_summary, 1)
        buttons.addWidget(self.revert_btn)
        buttons.addWidget(self.apply_btn)
        self.swirler_options = SwirlerOptions()
        self.swirler_options.picked.connect(self._on_layout_picked)
        injector_layout = self.injector.layout()
        injector_layout.insertWidget(injector_layout.count() - 1, self.swirler_options)  # above the card's closing stretch


        # Two columns of about the same height; the last card in each takes up any difference.
        left_column = QWidget()
        left_column.setFixedWidth(380)
        left = QVBoxLayout(left_column)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(CARD_GAP)
        for card in (targets, tank, fuel):
            left.addWidget(card)
        left.addWidget(self.performance, 1)
        right = QVBoxLayout()
        right.setSpacing(CARD_GAP)
        right.addWidget(self.injector)
        right.addWidget(self.grain)
        right.addWidget(self.nozzle, 1)
        body = QGridLayout()
        body.setHorizontalSpacing(COLUMN_GAP)
        body.setVerticalSpacing(CARD_GAP)
        body.addWidget(self.limit_warning, 0, 0, 1, 2)
        body.addWidget(self.error, 1, 0, 1, 2)
        body.addWidget(left_column, 2, 0)
        body.addLayout(right, 2, 1)
        body.setColumnStretch(1, 1)

        inner = QWidget()
        page = page_layout(QVBoxLayout(inner))
        page.addLayout(body, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        outer.addWidget(bar)

        for spin in (self.P_cmbr.spin, self.burn_time, self.OF):
            spin.valueChanged.connect(self.refresh)
        self.P_cmbr.unit.currentTextChanged.connect(self.refresh)
        for w in (self.P_limit, self.tank_T, self.vent_D, self.rho, self.hole_D, self.inj_CdA,
                  self.sw_D_port, self.sw_R_in,
                  self.port_D, self.grain_OD, self.grain_L, self.pre_L, self.post_L, self.Pa, self.throat_D):
            w.spin.valueChanged.connect(self._on_motor_edited)
            w.unit.currentTextChanged.connect(self._on_motor_edited)
        for spin in (self.fill, self.vent_Cd, self.prop_a, self.prop_n, self.prop_m, self.const_OF, self.cstar, self.sw_xi,
                     self.inj_Cd, self.inj_Cd_HEM, self.dyer_kappa, self.holes, self.sw_ports, self.noz_Cd, self.noz_eff, self.noz_ER):
            spin.valueChanged.connect(self._on_motor_edited)
        self.stl_voxel.valueChanged.connect(self._on_motor_edited)
        self.tank_L.spin.valueChanged.connect(self._on_tank_length)
        self.tank_L.unit.currentTextChanged.connect(self._on_tank_length)
        self.tank_D.spin.valueChanged.connect(self._on_tank_diameter)
        self.tank_D.unit.currentTextChanged.connect(self._on_tank_diameter)
        self.tank_V.spin.valueChanged.connect(self._on_tank_volume)
        self.tank_V.unit.currentTextChanged.connect(self._on_tank_volume)
        self.cd_as.currentIndexChanged.connect(self._on_cd_mode)
        self.inj_flow.currentIndexChanged.connect(self._on_inj_flow)
        self.cda_curve.edited.connect(self._on_motor_edited)
        for combo in (self.vent, self.inj_type, self.inj_model, self.reg_model, self.stl_unit):
            combo.currentIndexChanged.connect(self._on_motor_edited)
        self.grain_shape.currentIndexChanged.connect(self._on_motor_edited)
        for box in (self.sw_cd_geom, self.hem_same, self.ends_burn):
            box.toggled.connect(self._on_motor_edited)
        self.size_from.currentIndexChanged.connect(self._on_size_from)
        self.grain_from.currentIndexChanged.connect(self._on_size_from)
        self.nozzle_from.currentIndexChanged.connect(self._on_size_from)
        self._sync_motor_rows()

    def _by_holes(self) -> bool:
        return self.size_from.currentIndex() == 0

    def _by_burn_time(self) -> bool:
        return self.size_from.currentIndex() == 1

    def _by_OF(self) -> bool:
        return self.size_from.currentIndex() == 2

    @property
    def result(self) -> Sizing | None:
        """The motor at the start of the burn, from the last refresh."""
        return self._result

    def _fixed_OF(self) -> bool:
        return self.reg_model.currentData() == "Constant OF"

    def _fixed_nozzle(self) -> bool:
        return self.nozzle_from.currentIndex() == 0

    def _solve_port(self) -> bool:
        """A burn time or O/F sets the flow, and the swirler's inlet port size is sized to it."""
        return self._swirler and not self._by_holes()

    def _show_size_from_rows(self):
        solve = self._solve_port()
        for w in self._burn_time_row:
            w.setVisible(self._by_burn_time())
        for w in self._holes_row:
            w.setVisible(self._by_holes())
        for w in self._swirler_rows:
            w.setVisible(self._swirler)
        for w in self._sw_D_port_row:
            w.setVisible(self._swirler and not solve)
        curve = self._curve()
        for w in self._cd_row:
            w.setVisible(not solve and not curve)
        for w in self._curve_row:
            w.setVisible(curve)
        self.injector.show_row("Flow per hole", not solve and not curve)
        self.injector.show_row("Holes", not solve and not self._by_holes())  # with a chosen count it's an input
        self.injector.show_row("Hole drill", solve)
        self.injector.show_row("Swirler Cd", solve)
        self.injector.show_row("Limits the flow", self._swirler)
        for w in self._layouts_row:
            w.setVisible(solve)
        self.swirler_options.setVisible(solve and self.show_layouts.isChecked())
        self.show_layouts.setText("Hide" if self.show_layouts.isChecked() else "Show")
        for w in self._OF_row:
            w.setVisible(not self._by_length() or self._by_OF())
        self.grain_from.setEnabled(not self._by_OF())  # an O/F-sized flow needs a set grain length
        stl = self._stl()  # an STL grain has its own sizes
        for w in self._grain_L_row:
            w.setVisible(self._by_length() and not stl)
        for w in self._round_rows:
            w.setVisible(not stl)
        for w in (*self._stl_rows, *self._stl_size_rows):
            w.setVisible(stl)
        for w in self._grain_from_row:
            w.setVisible(not stl)
        # Until an STL grain is in, its results are blank; the status line says why. One from the cache is a moment away.
        sized = not stl or self.grain_geom is not None or (self.grain_loading() and not self._grain_shown)
        for name in self.grain.values:
            self.grain.show_row(name, sized and (name != "Burning area" or stl))
        self.grain.labels["Port at liquid burnout"].setText("Same-area port at liquid burnout" if stl else "Port at liquid burnout")
        self.grain.show_row("Grain length", sized and not self._by_length())
        self.grain.show_row("O/F", sized and self._by_length())
        fixed = self._fixed_nozzle()
        for w in self._P_cmbr_row:
            w.setVisible(not fixed)
        for w in self._throat_rows:
            w.setVisible(fixed)
        for name in ("Throat diameter", "Expansion ratio"):  # with a manual nozzle they're inputs
            self.nozzle.show_row(name, not fixed)
        self.nozzle.show_row("Chamber pressure", fixed)

    def _by_length(self) -> bool:
        return self.grain_from.currentIndex() == 0

    def _on_size_from(self, *_):
        if self._by_OF():
            self.grain_from.setCurrentIndex(0)
        self._show_size_from_rows()
        self.refresh()

    def grain_size(self) -> tuple[float, float, float]:
        """The grain's length, outer diameter and starting port [m]: a loaded STL's own, or the motor's round sizes."""
        g = self.grain_geom
        if g is not None:
            return g.length, g.outer_diameter, 2 * math.sqrt(g.port_area[0] / math.pi)
        return self.grain_L.si("length"), self.grain_OD.si("length"), self.port_D.si("length")

    def targets(self) -> SizingTargets:
        length, _outer, port = self.grain_size()
        return SizingTargets(
            P_cmbr=None if self._fixed_nozzle() else self.P_cmbr.si("pressure"),
            burn_time=self.burn_time.value() if self._by_burn_time() else None,
            OF=self.OF.value(),
            port_D=port,
            holes=self.holes.value() if self._by_holes() else None,
            grain_L=length if self._by_length() else None,
        )

    def targets_cfg(self) -> dict:
        return {"size_from": ("holes", "burn_time", "OF")[self.size_from.currentIndex()],
                "grain_from": "grain_L" if self._by_length() else "OF",
                "nozzle_from": "throat" if self._fixed_nozzle() else "P_cmbr",
                "P_cmbr": self.P_cmbr.si("pressure"), "burn_time": self.burn_time.value(), "OF": self.OF.value()}

    def set_targets(self, saved: dict, motor_cfg: dict):
        """Load saved targets. A motor without any opens with every part manual, so nothing is sized."""
        self._loading = True
        self.P_cmbr.set_si(saved.get("P_cmbr") or to_si(400.0, "psi", "pressure"), "pressure")
        self.burn_time.setValue(float(saved.get("burn_time") or 5.0))
        # Injector mode first: while it's still on the last motor's O/F, it holds the grain on Grain length.
        self.size_from.setCurrentIndex({"burn_time": 1, "OF": 2}.get(saved.get("size_from"), 0))
        self.grain_from.setCurrentIndex(1 if saved.get("grain_from") == "OF" else 0)
        self.nozzle_from.setCurrentIndex(1 if saved.get("nozzle_from", "P_cmbr" if saved else "throat") == "P_cmbr" else 0)
        self.OF.setValue(float(saved.get("OF") or motor_cfg.get("const_OF") or 6.0))
        self._loading = False

    def _on_motor_edited(self, *_):
        if self._loading:
            return
        self._sync_motor_rows()
        self.refresh()
        self.motor_edited.emit()

    def _sync_motor_rows(self):
        """Show the rows the injector type, flow model and vent use, and fill in the Cds that follow from others."""
        self._swirler = swirler = self.inj_type.currentText() == "Swirler"
        self._holes_row[0].setText("Swirler count" if swirler else "Hole count")
        stock = swirler and self.ptc_stock.isChecked()
        rough = stock and abs(self.hole_D.si("length") - STOCK_PTC_D) < 1e-6  # still the fire-video estimate
        self._hole_D_row[0].setText("Stock PTC acts like (rough)" if rough else "Stock PTC acts like" if stock else
                                    "PTC bore" if swirler else "Hole diameter")
        self.hole_D.setToolTip(
            f"The clean hole a stock PTC flows like. {STOCK_PTC_D / 0.0254:.3f} in is a very rough estimate, fit to HPS01-1's\n"
            "liquid running out at 5.8 s in the fire video, with a guessed hole offset. Replace it with a cold flow\n"
            "of a bare stock PTC (Measured CdA, then Fit)." if stock else
            "The PTC fitting's bore after the swirler: the narrowest point the swirling flow leaves through." if swirler else
            "Diameter of each injector hole.")
        self.ptc_stock.setVisible(swirler)
        self.injector.labels["Flow per hole"].setText("Flow per swirler" if swirler else "Flow per hole")
        self.injector.labels["Holes"].setText("Swirlers" if swirler else "Holes")
        self.sw_cd_geom.setVisible(swirler)
        as_cd = self.cd_as.currentText() == "Cd"
        noun = "swirler" if swirler else "hole"
        self._cd_row[0].setText(f"Cd per {noun}" if as_cd else f"CdA per {noun}")
        self.inj_Cd.setVisible(as_cd)
        self.inj_CdA.setVisible(not as_cd)
        geometry = swirler and self.sw_cd_geom.isChecked()
        self.inj_CdA.setEnabled(not geometry)
        self.inj_Cd.setEnabled(not geometry)
        bore = self.hole_D.si("length")
        cd_tip = (
            "Worked out from the swirler geometry." if geometry else
            "Discharge coefficient on the hole or PTC bore. The motor stores this. CdA = Cd × that area." if as_cd else
            f"The flow area after losses: a cold flow's water flow ÷ √(2 × 998 kg/m³ × ΔP). Divided by the\n"
            f"{'PTC bore' if swirler else 'hole'} area it's a Cd of {self.cd():.4f}, which is what the motor file stores."
        )
        self.inj_CdA.setToolTip(cd_tip)
        self.inj_Cd.setToolTip(cd_tip)
        if geometry and self.sw_D_port.si("length") > 0:
            cd = self.cd()
            self.inj_Cd.blockSignals(True)
            self.inj_Cd.setValue(cd)
            self.inj_Cd.blockSignals(False)
            self.inj_CdA.spin.blockSignals(True)
            self.inj_CdA.set_si(cd * 0.25 * math.pi * bore ** 2, "area")
            self.inj_CdA.spin.blockSignals(False)
        model = self.inj_model.currentText()
        for w in self._hem_row:
            w.setVisible(model != "SPI")
        for w in self._dyer_row:
            w.setVisible(model == "Dyer")
        # With a CdA curve, the holes' sizes and Cds don't set the flow; the count and sizes still draw the injector.
        curve = self._curve()
        for w in (self.holes, self.hole_D, self.ptc_stock, self.sw_ports, self.sw_D_port, self.sw_R_in, self.sw_xi,
                  self.sw_CdA_meas, self.hem_same):
            w.setEnabled(not curve)
        self._fit_btn.setEnabled(not curve and self.sw_CdA_meas.si("area") > 0)
        self.inj_Cd_HEM.setEnabled(not curve and not self.hem_same.isChecked())
        if self.hem_same.isChecked():
            self.inj_Cd_HEM.blockSignals(True)
            self.inj_Cd_HEM.setValue(self.cd())
            self.inj_Cd_HEM.blockSignals(False)
        for w in self._vent_rows:
            w.setVisible(self.vent.currentText() != "None")
        stl = self._stl()
        fixed_item = cast(QStandardItemModel, self.reg_model.model()).item(1)
        fixed_item.setEnabled(not stl)
        fixed_item.setToolTip("An STL grain burns back by the burn-rate law." if stl else "")
        if stl and self._fixed_OF():
            self.reg_model.setCurrentIndex(0)
        fixed = self._fixed_OF()
        for w in self._burn_rate_rows:
            w.setVisible(not fixed)
        for w in self._const_OF_row:
            w.setVisible(fixed)
        # With a fixed O/F, neither the flow nor the grain length changes the O/F, so neither can be sized for one.
        # An STL grain has its own length, and a CdA curve sets the injector.
        for combo, index, off in ((self.size_from, 1, curve), (self.size_from, 2, fixed or curve),
                                  (self.grain_from, 1, fixed or stl)):
            cast(QStandardItemModel, combo.model()).item(index).setEnabled(not off)
            if off and combo.currentIndex() == index:
                combo.setCurrentIndex(0)
        self._voxel_unit.setText(self.stl_unit.currentText())
        self._sync_grain()
        self._show_size_from_rows()

    def _stl(self) -> bool:
        return self.grain_shape.currentData() == "stl"

    def _curve(self) -> bool:
        """A table of CdA against time sets the injector's flow."""
        return self.inj_flow.currentIndex() == 1

    def injector_curve(self) -> list[list[float | None]] | None:
        """The table's [time, total CdA] rows in its unit when they set the injector's flow, else None."""
        return self.cda_curve.points() if self._curve() else None

    def _on_inj_flow(self, *_):
        """Start a new curve from the injector's current total CdA, so switching doesn't change the flow at first."""
        if not self._loading and self._curve() and not self.cda_curve.points():
            unit = self.cda_curve.unit.currentText()
            total = self.cd() * 0.25 * math.pi * self.hole_D.si("length") ** 2 * self.holes.value()
            self.cda_curve.set_points([[0.0, from_si(total, unit, "area")]], unit)
        self._on_motor_edited()

    def _pick_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load grain STL", str(Path(self._stl_path).parent) if self._stl_path else "",
                                              "STL (*.stl)")
        if path:
            self._stl_path = str(Path(path).resolve())
            self._on_motor_edited()

    def _sync_grain(self):
        """Load the STL grain when its file or settings change, off the UI thread since it takes a while."""
        request = (self._stl_path, self.stl_unit.currentText(), self.stl_voxel.value() or None,
                   self.ends_burn.isChecked()) if self._stl() and self._stl_path else None
        if request == self._grain_request:
            return
        self._grain_request, self.grain_geom, self.grain_mesh, self._section = request, None, None, None
        self._grain_error, self._grain_stage, self._grain_shown = "", ("Reading STL", 0.0), False
        if request is None:
            self._grain_loader.stop()
        elif not Path(request[0]).is_file():
            self._grain_loader.stop()
            self._grain_error = f"Can't find {request[0]}. If it moved, load it again with Load STL…"
        else:
            self._grain_started, self._load_samples = time.monotonic(), []
            self._grain_loader.load(request)
            self._show_loading.start()
        self._show_grain_status()
        self.grain_changed.emit()

    def _on_slow_load(self):
        """Still loading after a moment, so it isn't coming from the cache: show its progress."""
        if self.grain_loading():
            self._grain_shown = True
            self._tick.start()
            self._show_grain_status()
            self._show_size_from_rows()
            self.refresh()
            self.grain_changed.emit()

    def _on_tick(self):
        if self.grain_loading_shown():
            self._show_grain_status()
            self.grain_progress.emit()
        else:
            self._tick.stop()

    def _on_grain_progress(self, request: tuple, stage: str, fraction: float):
        if request == self._grain_request and self.grain_geom is None:
            self._grain_stage = (stage, fraction)
            self._load_samples.append((fraction, time.monotonic() - self._grain_started))
            if self._grain_shown:
                self._show_grain_status()
                self.grain_progress.emit()

    def _on_grain_mesh(self, request: tuple, mesh: tuple):
        if request == self._grain_request and self.grain_geom is None:
            self.grain_mesh = mesh
            if self._grain_shown:
                self.grain_changed.emit()

    def _on_grain_loaded(self, request: tuple, result):
        if request != self._grain_request:
            return  # the STL or its settings changed while it loaded
        self._show_loading.stop()
        self._tick.stop()
        self._grain_took = time.monotonic() - self._grain_started
        if result is None:
            self._grain_error = STOPPED
        elif isinstance(result, str):
            self._grain_error = plain_load_error(Path(request[0]).name, result)
        else:
            self.grain_geom, self._section, self.grain_mesh = result, mid_section(result), None
            self._learn_load_timing()
        self._show_grain_status()
        self._show_size_from_rows()  # the Grain card's results come back
        self.grain_changed.emit()
        self.refresh()

    def _learn_load_timing(self):
        """Share the next loads' time out more like this one's, as the steps' speeds change with the engine and computer."""
        samples = self._load_samples
        if self._grain_took < 2 or len(samples) < 10 or samples[-1][0] < 0.9:
            return  # from the cache, or too little to go on
        progress, seconds = np.array(samples).T
        measured = np.maximum.accumulate(np.interp(LOAD_PROGRESS, progress, seconds) / self._grain_took)
        shares = 0.5 * np.array(self.load_time_shares) + 0.5 * np.clip(measured, 0, 1)
        shares[0], shares[-1] = 0.0, 1.0
        self.load_time_shares = tuple(round(float(x), 4) for x in shares)
        self.load_timing_learned.emit()

    def _on_grain_action(self):
        if self.grain_loading():
            self.stop_grain_load()
        elif self.grain_geom is not None:
            self.show_grain.emit()
        else:
            self.load_grain_again()

    def stop_grain_load(self):
        self._grain_loader.stop()

    def load_grain_again(self):
        self._grain_request = None
        self._sync_grain()

    def grain_failed(self) -> bool:
        """The last load failed or was stopped."""
        return bool(self._grain_error)

    def grain_stopped(self) -> bool:
        return self._grain_error == STOPPED

    def grain_load_seconds(self) -> float:
        """How long the last load took."""
        return self._grain_took

    def grain_name(self) -> str:
        return Path(self._stl_path).name

    def grain_message(self) -> str:
        """What's in the way of showing the STL grain, if anything."""
        if not self._stl_path:
            return "Load the fuel's STL file on the Motor tab's Grain card."
        if self._grain_error:
            return self._grain_error
        return f"Loading {self.grain_name()}…" if self.grain_geom is None else ""

    def grain_progress_text(self) -> str:
        """The loading step, how far along the whole load is, and roughly how long is left."""
        done, elapsed = self.grain_fraction(), time.monotonic() - self._grain_started
        text = f"{self._grain_stage[0]} · {100 * done:.0f}% · {elapsed:.0f} s in"
        if elapsed > 2 and 0.1 < done < 0.97:
            text += f", about {max(1.0, elapsed * (1 - done) / done):.0f} s left"
        return text

    def grain_fraction(self) -> float:
        """How far through the load, by time, as the engine's steps take very different shares of it."""
        return float(np.interp(self._grain_stage[1], LOAD_PROGRESS, self.load_time_shares))

    def grain_tables(self):
        """The loaded STL grain's burn-back tables, so the engine needn't load it again; None for a round grain."""
        return self.grain_geom.tables() if self.grain_geom is not None else None

    def grain_loading(self) -> bool:
        return self._grain_request is not None and self.grain_geom is None and not self._grain_error

    def grain_loading_shown(self) -> bool:
        """Loading, and long enough that its progress is on screen."""
        return self.grain_loading() and self._grain_shown

    def _show_grain_status(self):
        self.stl_file.set_full_text(self.grain_name() or "none")
        self.stl_file.setToolTip(self._stl_path)
        g, loading = self.grain_geom, self.grain_loading()
        if loading and not self._grain_shown:
            return  # a moment in: a grain in the cache is about to arrive
        if loading:
            text = f"Loading: {self.grain_progress_text()}"
        elif g is None:
            text = self.grain_message()
        else:
            text = g.warning
        warn = (self._grain_error and not self.grain_stopped()) or (g is not None and bool(text))
        self.stl_status.setText(text or "The numbers on this tab use the STL's starting port and burning area.")
        self.stl_status.setObjectName("notApplied" if warn else "cardLabel")
        self.stl_status.style().unpolish(self.stl_status)
        self.stl_status.style().polish(self.stl_status)
        self.stl_progress.setVisible(loading)
        self.stl_progress.setValue(round(1000 * self.grain_fraction()))
        self._grain_action.setText("Cancel" if loading else "View in 3D" if g is not None else "Load again")
        self._grain_action.setVisible(bool(self._stl_path) and (loading or g is not None or bool(self._grain_error)))

    def _on_propellant(self):
        """Fill in the propellant's own density and burn rate."""
        if self._loading or not self.propellant.currentData():
            return
        p = load_propellant(self.propellant.currentData())
        self._loading = True
        self.rho.set_si(p.rho, "density")
        self.prop_a.setValue(float(p.reg[0]))
        self.prop_n.setValue(float(p.reg[1]))
        self.prop_m.setValue(float(p.reg[2]) if p.reg.size > 2 else 0.0)
        self._loading = False
        self._on_motor_edited()

    def load_motor(self, cfg: dict):
        """Show a motor. A tank set by its length, starting pressure or oxidizer mass, a nozzle set by its exit
        diameter, and a typed-in chamber volume are shown as their equivalents in this page's fields."""
        def si(name: str, quantity: str = "length") -> float:
            return to_si(float(cfg[name] or 0.0), cfg[f"{name}_unit"], quantity)

        def show(row: UnitRow, name: str):  # in the file's own unit, so saving it again keeps its numbers
            row.set_display(float(cfg[name] or 0.0), cfg[f"{name}_unit"])

        self._loading = True
        show(self.tank_D, "tnk_D")
        D = si("tnk_D")
        if int(cfg.get("tnk_V_state", 0)) and D > 0:
            L = si("tnk_L")
            self.tank_L.set_display(float(cfg.get("tnk_L") or 0.0), cfg.get("tnk_L_unit") or "in")
            self.tank_V.set_si(L * 0.25 * math.pi * D ** 2, "volume")
        else:
            show(self.tank_V, "tnk_V")
            if D > 0:
                self.tank_L.set_si(self.tank_V.si("volume") / (0.25 * math.pi * D ** 2), "length")
            elif float(cfg.get("tnk_L") or 0.0) > 0:
                self.tank_L.set_display(float(cfg.get("tnk_L") or 0.0), cfg.get("tnk_L_unit") or "in")
        if cfg.get("tnk_dd") == "Starting Tank Pressure":
            P = to_si(float(cfg["tnk_cond"]), cfg["T_tnk_unit"], "pressure")
            self.tank_T.set_si(saturation_temperature(P) or 293.15, "temperature")
        else:
            self.tank_T.set_display(float(cfg["tnk_cond"]), cfg["T_tnk_unit"])
        if cfg.get("fill_dd") == "Tank Fill Percentage" or cfg.get("fill_unit") == "%":
            self.fill.setValue(float(cfg["fill"]))
        else:
            ox, V = nox(self.tank_T.si("temperature")), self.tank_V.si("volume")
            m_o = to_si(float(cfg["fill"]), cfg["fill_unit"], "mass")
            self.fill.setValue(100.0 * (m_o / max(V, 1e-12) - ox.rho_v) / (ox.rho_l - ox.rho_v))
        self.vent.setCurrentText(str(cfg.get("vnt_state") or "None"))
        show(self.vent_D, "vnt_D")
        self.vent_Cd.setValue(float(cfg.get("vnt_Cd") or 0.0))

        self.propellant.setCurrentIndex(max(self.propellant.findData(cfg.get("prop_id") or "ABS"), 0))
        show(self.rho, "prop_rho")
        self.prop_a.setValue(float(cfg.get("prop_a") or 0.0))
        self.prop_n.setValue(float(cfg.get("prop_n") or 0.0))
        self.prop_m.setValue(float(cfg.get("prop_m") or 0.0))
        self.reg_model.setCurrentIndex(max(self.reg_model.findData(cfg.get("reg_model") or "Constant OF"), 0))
        self.const_OF.setValue(float(cfg.get("const_OF") or 1.0))
        self.cstar.setValue(float(cfg.get("cstar_eff") or 100.0))

        self.inj_type.setCurrentText(str(cfg["inj_type"]))
        show(self.hole_D, "inj_D")
        self.ptc_stock.blockSignals(True)
        self.ptc_stock.setChecked(bool(cfg.get("ptc_stock")))
        self.ptc_stock.blockSignals(False)
        self.sw_xi.setValue(float(cfg.get("sw_xi") or 0.0))
        self.sw_CdA_meas.set_display(0.0)
        self._fit_btn.setEnabled(False)
        self.holes.setValue(int(cfg.get("inj_N") or 1))
        self.sw_ports.setValue(int(cfg["sw_ports"]))
        show(self.sw_D_port, "sw_D_port")
        show(self.sw_R_in, "sw_R_in")
        self.sw_cd_geom.setChecked(bool(cfg["sw_cd_from_geometry"]))
        self.cd_as.setCurrentText("Cd" if cfg.get("inj_cd_as") == "Cd" else "CdA")
        self.inj_Cd.setValue(float(cfg.get("inj_Cd") or 0.0))
        self.inj_CdA.set_si(float(cfg.get("inj_Cd") or 1.0) * 0.25 * math.pi * self.hole_D.si("length") ** 2, "area")
        self.inj_model.setCurrentText(str(cfg.get("inj_model") or "SPI"))
        hem_cd = float(cfg.get("inj_Cd_HEM") or 0.0)
        self.hem_same.setChecked(not hem_cd)
        if hem_cd:
            self.inj_Cd_HEM.setValue(hem_cd)
        curve = cfg.get("inj_cda_curve")
        self.cda_curve.set_points(curve or [], cfg.get("inj_cda_curve_unit") or "in^2")
        self.inj_flow.setCurrentIndex(0 if curve is None else 1)
        self.dyer_kappa.setValue(float(cfg.get("dyer_kappa") or 1.0))

        self.grain_shape.setCurrentIndex(max(self.grain_shape.findData(cfg.get("grn_shape") or "round"), 0))
        self._stl_path = str(cfg.get("grn_stl") or "")
        self.stl_unit.setCurrentText(str(cfg.get("grn_stl_unit") or "in"))
        self.stl_voxel.setValue(float(cfg.get("grn_stl_voxel") or 0.0))
        self.ends_burn.setChecked(bool(cfg.get("grn_ends_burn")))
        for row, name in ((self.port_D, "grn_ID"), (self.grain_OD, "grn_OD"), (self.grain_L, "grn_L"),
                          (self.pre_L, "cmbr_pre_L"), (self.post_L, "cmbr_post_L")):
            show(row, name)
        if not int(cfg.get("cmbr_V_state", 1)):  # extra chamber volume becomes post-combustion length
            area = 0.25 * math.pi * si("grn_OD") ** 2
            extra = si("cmbr_V", "volume") / area - si("cmbr_pre_L") - si("grn_L") - si("cmbr_post_L")
            if extra > 0:
                self.post_L.set_si(si("cmbr_post_L") + extra, "length")

        show(self.throat_D, "noz_thrt")
        exit_D = si("noz_ex") if cfg.get("noz_def") == "Nozzle Exit Diameter" else 0.0
        self.noz_ER.setValue((exit_D / si("noz_thrt")) ** 2 if exit_D and si("noz_thrt") else float(cfg.get("noz_ex") or 1.0))
        self.noz_Cd.setValue(float(cfg.get("noz_Cd") or 1.0))
        self.noz_eff.setValue(float(cfg.get("noz_eff") or 100.0))
        show(self.Pa, "Pa")
        show(self.P_limit, "P_cmbr_max")
        self._loading = False
        self._sync_motor_rows()

    def motor_cfg(self) -> dict:
        """The motor's settings as config fields."""
        def field(name: str, row: UnitRow) -> dict:
            return {name: row.value(), f"{name}_unit": row.unit.currentText()}

        L, _D, V = self.tank_geometry()
        vol_unit = self.tank_V.unit.currentText()
        len_unit = self.tank_L.unit.currentText()
        curve = self.injector_curve()
        return {
            "tnk_V": from_si(V, vol_unit, "volume"), "tnk_V_unit": vol_unit, "tnk_V_state": 0,
            **field("tnk_D", self.tank_D),
            "tnk_L": from_si(L, len_unit, "length"), "tnk_L_unit": len_unit,
            "tnk_dd": "Starting Tank Temperature", "tnk_cond": self.tank_T.value(),
            "T_tnk_unit": self.tank_T.unit.currentText(),
            "fill_dd": "Tank Fill Percentage", "fill": self.fill.value(), "fill_unit": "%",
            "vnt_state": self.vent.currentText(), **field("vnt_D", self.vent_D), "vnt_Cd": self.vent_Cd.value(),
            "prop_id": self.propellant.currentData() or "ABS",
            "prop_nm": (self.propellant.currentText() or "ABS").split(" (")[0],
            **field("prop_rho", self.rho), "prop_a": self.prop_a.value(), "prop_n": self.prop_n.value(),
            "prop_m": self.prop_m.value(), "reg_model": self.reg_model.currentData(), "const_OF": self.const_OF.value(),
            "cstar_eff": self.cstar.value(),
            "inj_type": self.inj_type.currentText(), **field("inj_D", self.hole_D), "inj_N": self.holes.value(),
            "inj_Cd": self.cd(), "inj_cd_as": self.cd_as.currentText(), "sw_ports": self.sw_ports.value(), **field("sw_D_port", self.sw_D_port),
            **field("sw_R_in", self.sw_R_in), "sw_xi": self.sw_xi.value(), "sw_cd_from_geometry": self.sw_cd_geom.isChecked(),
            "ptc_stock": self.inj_type.currentText() == "Swirler" and self.ptc_stock.isChecked(),
            "inj_model": self.inj_model.currentText(),
            "inj_Cd_HEM": 0.0 if self.hem_same.isChecked() else self.inj_Cd_HEM.value(),
            "dyer_kappa": self.dyer_kappa.value(),
            **({"inj_cda_curve": curve, "inj_cda_curve_unit": self.cda_curve.unit.currentText()} if curve is not None else {}),
            **field("P_cmbr_max", self.P_limit),
            "grn_shape": self.grain_shape.currentData(), "grn_stl": self._stl_path, "grn_stl_unit": self.stl_unit.currentText(),
            "grn_stl_voxel": self.stl_voxel.value() or None, "grn_ends_burn": self.ends_burn.isChecked(),
            **field("grn_ID", self.port_D), **field("grn_OD", self.grain_OD), **field("grn_L", self.grain_L),
            **field("cmbr_pre_L", self.pre_L), **field("cmbr_post_L", self.post_L), "cmbr_V_state": 1,
            **field("noz_thrt", self.throat_D), "noz_def": "Nozzle Expansion Ratio", "noz_ex": self.noz_ER.value(),
            "noz_Cd": self.noz_Cd.value(),
            "noz_eff": self.noz_eff.value(), **field("Pa", self.Pa),
        }

    def tank_geometry(self) -> tuple[float, float, float]:
        """Tank length, diameter and volume in SI. With an inside diameter, volume follows the internal length."""
        D = self.tank_D.si("length")
        L = self.tank_L.si("length")
        if D > 1e-9 and L > 1e-9:
            return L, D, L * 0.25 * math.pi * D ** 2
        D = D or self.grain_OD.si("length") or 0.05
        V = self.tank_V.si("volume")
        return max(V / (0.25 * math.pi * D ** 2), 1e-4), D, V

    def tank_state(self) -> tuple[float, float, float]:
        """Starting fill fraction, temperature [K] and oxidizer mass [kg]."""
        fill, T = self.fill.value() / 100.0, self.tank_T.si("temperature")
        try:
            ox = nox(T)
        except Exception:  # outside the nitrous fit
            return fill, T, 0.0
        V = self.tank_V.si("volume")
        return fill, T, fill * V * ox.rho_l + (1.0 - fill) * V * ox.rho_v

    def nozzle_size(self) -> tuple[float, float, float]:
        """The motor's throat and exit diameters [m] and expansion ratio."""
        throat = self.throat_D.si("length")
        return throat, throat * math.sqrt(self.noz_ER.value()), self.noz_ER.value()

    def refresh(self):
        if self._loading:
            return
        u = self._get_units()
        cfg = self._get_cfg()
        for label, size in zip((self.stl_L, self.stl_OD, self.stl_port), self.grain_size()):
            label.setText(u.text(size, "length") if self.grain_geom is not None else "—")
        limit = chamber_limit(cfg)
        self.limit_warning.hide()
        if self._stl() and self.grain_geom is None:
            # Sizing would load the STL here, holding up the UI. Until it's in, the other cards keep their last
            # numbers, nothing can be applied from them, and the Grain card's status line says what's going on.
            if self.grain_loading() and not self._grain_shown:
                return  # a moment in: a grain in the cache is about to arrive
            self._result = self._cfg = None
            self.error.hide()
            self.grain.sketch.show_data(None)
            self._show_apply()
            return
        try:
            z = size_motor(cfg, self.targets(), grain_tables=self.grain_tables())
        except Exception as exc:  # the motor form can hold any combination; show why sizing can't run
            self._result = self._cfg = None
            self.swirler_options.update_target(None)
            self.error.setText(str(exc) or type(exc).__name__)
            self.error.setVisible(not self.grain_loading())  # the Grain card says it's loading
            for card in (self.injector, self.nozzle, self.grain, self.performance):
                card.clear()
            self._show_apply()
            return
        self._result, self._cfg = z, cfg
        try:
            t = self.targets()
            self._built = size_motor(cfg, SizingTargets(P_cmbr=None, burn_time=None, OF=t.OF, port_D=t.port_D,
                                                        holes=int(cfg["inj_N"]), grain_L=self.grain_size()[0]),
                                     grain_tables=self.grain_tables())
        except Exception:
            self._built = None
        self._layout = None
        if self._solve_port():
            target = Target(z.inj_CdA, z.mdot_o, z.OF if self._by_length() else None, z.OF_exp, z.ox_liquid, self.holes.value(),
                            to_si(float(cfg["sw_R_in"]), cfg["sw_R_in_unit"], "length"), self._hole_D(cfg), int(cfg["sw_ports"]),
                            self.sw_xi.value())
            try:
                self._layout = drilled_layout(target, target.exit_D, target.ports)
            except ValueError as exc:
                self._port_error = str(exc)
            self.swirler_options.update_target(target)
        self.error.hide()
        what = "starting chamber pressure" if self._fixed_nozzle() else "chamber pressure target"
        self.limit_warning.setText(f"The {u.text(z.P_cmbr, 'pressure')} {what} is above the "
                                   f"{u.text(limit, 'pressure')} chamber pressure limit.")
        self.limit_warning.setVisible(z.P_cmbr > limit)

        self.tank_P.setText(u.text(z.P_tnk, "pressure"))
        self.ox_liquid.setText(u.text(z.ox_liquid, "mass"))

        t = self.targets()
        holes = self._values()["holes"]
        solve = self._solve_port()
        lasts = z.burn_time if solve else z.ox_liquid / (holes * z.flow_per_hole)
        flow, per_hole = u.text(z.mdot_o, "mass_flow", 3), u.text(z.flow_per_hole, "mass_flow", 3)
        dP, P_cmbr = u.text(z.inj_dP, "pressure"), u.text(z.P_cmbr, "pressure")
        hole, inj_Cd, model = u.text(self._hole_D(cfg), "length"), injector_cd(cfg), cfg.get("inj_model") or "SPI"
        noun = "swirler" if self._swirler else "hole"
        curve = self._curve()
        if curve:
            self.injector.set("Oxidizer flow", flow, f"The flow at the start of the burn, with the CdA curve's "
                                                     f"{u.text(z.inj_CdA, 'area')} at 0 s.")
        elif t.holes:
            self.injector.set("Oxidizer flow", flow,
                              f"The flow through your {noun}s:\n{noun} count × flow per {noun} = {holes} × {per_hole} = {flow}")
        elif t.burn_time is None:
            self.injector.set("Oxidizer flow", flow,
                              f"The flow that gives O/F {t.OF:.3g} with the {u.text(cast(float, t.grain_L), 'length')} grain.\n"
                              "The fuel flow grows as oxidizer flow^n, so O/F grows as oxidizer flow^(1 − n), solved for the flow.")
        else:
            self.injector.set("Oxidizer flow", flow,
                              f"The flow that empties the liquid in the burn time:\n"
                              f"liquid oxidizer ÷ liquid burn time = {u.text(z.ox_liquid, 'mass')} ÷ {t.burn_time:.3g} s = {flow}")
        self.injector.set("Injector ΔP", dP,
                          f"tank pressure − chamber pressure = {u.text(z.P_tnk, 'pressure')} − {P_cmbr} = {dP}")
        self.injector.set("Stiffness (ΔP / Pc)", f"{100 * z.inj_dP / z.P_cmbr:.0f}%",
                          f"Injector ΔP as a share of chamber pressure: {dP} ÷ {P_cmbr}.\n"
                          "Above about 20%, chamber pressure swings barely change the injector flow,\n"
                          "which keeps the motor from chugging.")
        self.injector.set("Flow per hole", per_hole,
                          f"Flow through one {hole} {'swirler exit' if self._swirler else 'hole'} at Cd {inj_Cd:.3g} with {dP} across it,\n"
                          f"from the {model} injector model.")
        if t.holes:
            self.injector.set("Holes", f"{holes}", f"Your {noun} count, from Targets.")
        else:
            self.injector.set("Holes", f"{z.holes:.2f} → {holes}",
                              f"oxidizer flow ÷ flow per {noun} = {flow} ÷ {per_hole} = {z.holes:.2f},\n"
                              f"rounded to {holes}. Apply to motor uses {holes}.")
        cda, cda_now = u.text(z.inj_CdA, "area"), u.text(z.inj_CdA * holes / z.holes, "area")
        if curve:
            self.injector.set("Total CdA", cda, "The CdA curve's value at the start of the burn. The simulation follows the curve.")
        elif t.holes:
            self.injector.set("Total CdA", cda, f"Cd × area over all your {noun}s. A cold flow measures this directly.")
        elif solve:
            self.injector.set("Total CdA", cda, "The Cd × area that gives the oxidizer flow. A cold flow measures this directly.")
        else:
            self.injector.set("Total CdA", cda,
                              f"The Cd × area over all {noun}s that gives the oxidizer flow. A cold flow measures this directly.\n"
                              f"{holes} of your {noun}s give {cda_now}.")
        if curve:
            self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                              f"liquid oxidizer ÷ starting flow = {u.text(z.ox_liquid, 'mass')} ÷ {flow} = {lasts:.2f} s.\n"
                              "The CdA curve and the cooling tank change the flow, so the full simulation's liquid time differs.")
        elif solve:
            self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                              f"liquid oxidizer ÷ oxidizer flow = {u.text(z.ox_liquid, 'mass')} ÷ {flow} = {lasts:.2f} s.\n"
                              "The real flow falls as the tank cools, so the full simulation runs a little longer.")
        else:
            self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                              f"With {holes} {noun}s the flow is {u.text(holes * z.flow_per_hole, 'mass_flow', 3)}, so\n"
                              f"liquid oxidizer ÷ flow = {u.text(z.ox_liquid, 'mass')} ÷ {u.text(holes * z.flow_per_hole, 'mass_flow', 3)} = {lasts:.2f} s.\n"
                              "The real flow falls as the tank cools, so the full simulation runs a little longer.")
        bore = self.grain_size()[1]
        if self._swirler:
            D_port = to_si(float(cfg["sw_D_port"]), cfg["sw_D_port_unit"], "length")
            R_in = to_si(float(cfg["sw_R_in"]), cfg["sw_R_in_unit"], "length")
            ports = int(cfg["sw_ports"])
            if solve:
                cd = z.inj_CdA / (holes * 0.25 * math.pi * self._hole_D(cfg) ** 2)
                if self._layout is None:
                    self.injector.set("Hole drill", "none fits", self._port_error)
                    self.injector.set("Swirler Cd", "—")
                else:
                    D_port = self._layout.port_D
                    self.injector.set("Hole drill", f"#{self._layout.drill} ({u.text(D_port, 'length')})",
                                      f"The number drill nearest the port size that gives Cd {cd:.3f} with {ports} ports\n"
                                      f"{u.text(R_in, 'length')} off the axis, from Abramovich's swirl theory with inlet loss ξ "
                                      f"{self.sw_xi.value():.2f}.\n"
                                      "Apply to motor copies it into the swirler.")
                    self.injector.set("Swirler Cd", f"{self._layout.cd:.3f}",
                                      f"The drilled swirler's Cd on its {hole} exit. The flow needs {cd:.3f}.")
            holes_gain, bore_gain = swirl_sensitivity(self._hole_D(cfg), ports, D_port, R_in, self.sw_xi.value())
            limit = ("Swirler holes" if holes_gain > bore_gain + 0.15 else
                     "PTC bore" if bore_gain > holes_gain + 0.15 else "Both")
            self.injector.set("Limits the flow", limit,
                              f"10% more swirler hole area gives {10 * holes_gain:.1f}% more flow;\n"
                              f"10% more PTC bore area gives {10 * bore_gain:.1f}% more.\n"
                              "With little swirl the PTC bore works like a plain hole.")
            fill = swirl_fill(swirl_A(self._hole_D(cfg), ports, D_port, R_in))
            ptc = f"stock PTC acting like {hole}" if self.ptc_stock.isChecked() else f"{hole} PTC bore"
            self.injector.sketch.show_data({
                "bore": bore, "swirlers": holes,
                "swirler": {"exit": self._hole_D(cfg), "ports": ports, "port": D_port, "offset": R_in, "fill": fill},
                "caption": f"{holes} swirler{'s' if holes != 1 else ''}, {ports} holes each, {ptc}"})
        else:
            self.injector.sketch.show_data({"bore": bore, "hole": self._hole_D(cfg), "holes": holes,
                                            "caption": f"{holes} × {hole} holes"})
        end = f" → {from_si(z.port_D_end, u.length, 'length'):.3g}" if math.isfinite(z.port_D_end) else ""
        g = self.grain_geom
        if self._stl() and g is not None:
            self.grain.sketch.show_data({"od": g.outer_diameter, "port": t.port_D, "port_end": math.nan,
                                         "section": self._section, "section_width": g.distance.shape[0] * g.spacing,
                                         "caption": "halfway along the grain"})
            self.grain.set("Burning area", u.text(float(g.burn_area[0]), "area"), "The starting port's surface, from the STL.")
        else:
            self.grain.sketch.show_data({"od": bore, "port": t.port_D, "port_end": z.port_D_end,
                                         "caption": f"port {from_si(t.port_D, u.length, 'length'):.3g}{end} {u.length}"})

        self._show_throat()
        self.nozzle.set("Expansion ratio", f"{z.ER:.2f}",
                        f"Exit area ÷ throat area, sized so the exhaust leaves at ambient pressure\n"
                        f"({u.text(to_si(float(cfg['Pa']), cfg['Pa_unit'], 'pressure'), 'pressure')}) when the chamber is at {P_cmbr}, with γ {z.k:.3f}.")
        self.nozzle.set("C*", f"{z.cstar:.0f} m/s",
                        f"Characteristic velocity from the {self.propellant.currentText()} combustion table at O/F {z.OF:.3g}\n"
                        f"and {P_cmbr}, times the {self.cstar.value():.3g}% C* efficiency.")

        port = u.text(t.port_D, "length")
        fuel = u.text(z.mdot_f, "mass_flow", 3)
        if self._fixed_OF():
            self.grain.set("Fuel flow", fuel, f"oxidizer flow ÷ the fixed O/F = {flow} ÷ {z.OF:.3g}")
            self.grain.set("O/F", f"{z.OF:.2f}", "Fixed on the Fuel card.")
        elif t.grain_L:
            self.grain.set("Fuel flow", fuel,
                           f"The fuel the {u.text(t.grain_L, 'length')} grain burns at the starting flux:\n"
                           "fuel flow = density × burn rate × port wall area, with burn rate = a × flux^n from the propellant.")
            self.grain.set("O/F", f"{z.OF:.2f}", f"oxidizer flow ÷ fuel flow = {flow} ÷ {fuel}")
        else:
            self.grain.set("Fuel flow", fuel, f"oxidizer flow ÷ O/F = {flow} ÷ {t.OF:.3g}")
        self.grain.set("Oxidizer flux", f"{z.ox_flux:.0f} kg/(m²·s)",
                       f"Oxidizer flow per unit of port area: {flow} ÷ the area of a {port} port.\n"
                       "It sets how fast the fuel burns back.")
        if math.isfinite(z.grain_L):
            self.grain.set("Grain length", u.text(z.grain_L, "length"),
                           f"The length whose burning wall gives the fuel flow for O/F {t.OF:.3g}.\n"
                           f"burn rate = a × flux^n from the propellant, and fuel flow = density × burn rate × port wall area.")
            self.grain.set("Port at liquid burnout", u.text(z.port_D_end, "length"),
                           f"The port after burning back for {z.burn_time:.3g} s at the starting oxidizer flow.\n"
                           "The dashed circle in the drawing.")
            self.grain.set("O/F at liquid burnout", f"{z.OF_end:.2f}",
                           "Fixed on the Fuel card." if self._fixed_OF() else
                           "The wider port lowers the flux but adds burning wall, so the O/F drifts.")
            self.grain.set("Fuel burned", u.text(z.fuel_burned, "mass"),
                           "The fuel between the starting port and the port at liquid burnout.")
        else:
            why = "The fuel's burn rate a on the Fuel card is 0, so no grain length makes fuel."
            self.grain.set("Grain length", "set burn rate a (Fuel card)", why)
            for name in ("Port at liquid burnout", "O/F at liquid burnout", "Fuel burned"):
                self.grain.set(name, "—", why)

        self.performance.set("Thrust", u.text(z.thrust, "force"))
        self.performance.set("Isp", f"{z.isp:.0f} s")
        self.performance.set("Impulse over the burn time", u.text(z.thrust * z.burn_time, "impulse"))
        self._show_apply()

    def _show_throat(self):
        z, u = cast(Sizing, self._result), self._get_units()
        throat = self._values()["throat_D"]
        formula = (f"throat area = total flow × C* ÷ (chamber pressure × throat Cd)\n"
                   f"total flow = {u.text(z.mdot_o + z.mdot_f, 'mass_flow', 3)}, C* = {z.cstar:.0f} m/s, "
                   f"throat Cd = {self.noz_Cd.value():.3g}")
        how = f"Sized throat: the throat that holds {u.text(z.P_cmbr, 'pressure')} in the chamber at the starting flow.\n" + formula
        self.nozzle.set("Chamber pressure", u.text(z.P_cmbr, "pressure"),
                        f"The chamber pressure at which the {u.text(throat, 'length')} throat passes the starting flow:\n"
                        + formula + "\nsolved for the chamber pressure. A higher chamber pressure also lets in less oxidizer.")
        self.nozzle.set("Throat diameter", u.text(throat, "length"), how)
        self.nozzle.set("Exit diameter", u.text(throat * math.sqrt(z.ER), "length"))
        bore = self.grain_size()[1]
        self.nozzle.sketch.show_data({"bore": bore, "throat": throat, "exit": throat * math.sqrt(z.ER),
                                      "caption": f"{u.text(throat, 'length')} throat"})

    def unapplied(self) -> list[str]:
        """What Apply to motor would change, as "old → new"."""
        if self._result is None:
            return []
        u, v = self._get_units(), self._values()
        pairs = [("throat", u.text(self.throat_D.si("length"), "length"), u.text(v["throat_D"], "length")),
                 ("expansion ratio", f"{self.noz_ER.value():.2f}", f"{v['ER']:.2f}"),
                 ("swirlers" if self._swirler else "holes", str(self.holes.value()), str(v["holes"]))]
        if v["sw_D_port"]:
            pairs.append(("swirler holes", u.text(self.sw_D_port.si("length"), "length"), u.text(v["sw_D_port"], "length")))
        if math.isfinite(v["grain_L"]) and not self._by_length():
            pairs.append(("grain", u.text(self.grain_L.si("length"), "length"), u.text(v["grain_L"], "length")))
        return [f"{name} {old} → {new}" for name, old, new in pairs if old != new]

    def _built_targets(self) -> dict[str, tuple[float, str]]:
        """For each target in use, what the current motor gives at the start of the burn: (value, as shown)."""
        b = self._built
        if self._result is None or b is None:
            return {}
        built = {}
        if not self._fixed_nozzle():
            unit = self.P_cmbr.unit.currentText()
            built["P_cmbr"] = (b.P_cmbr, f"{from_si(b.P_cmbr, unit, 'pressure'):.4g} {unit}")
        if self._by_burn_time():
            built["burn_time"] = (b.burn_time, f"{b.burn_time:.3g} s")
        if (self._by_OF() or not self._by_length()) and not self._fixed_OF():
            built["OF"] = (b.OF, f"{b.OF:.3g}")
        return built

    def _target_shown(self, name: str) -> str:
        if name == "P_cmbr":
            unit = self.P_cmbr.unit.currentText()
            return f"{self.P_cmbr.spin.value():.4g} {unit}"
        return f"{self.burn_time.value():.3g} s" if name == "burn_time" else f"{self.OF.value():.3g}"

    def _revert(self, names: list[str]):
        """Make the current motor's values the targets."""
        built = self._built_targets()
        self._loading = True
        for name in names:
            if name not in built:
                continue
            value = built[name][0]
            if name == "P_cmbr":
                self.P_cmbr.set_si(value, "pressure")
            else:
                getattr(self, name).setValue(value)
        self._loading = False
        self.refresh()

    def _show_apply(self):
        """The Apply bar only shows when a sized part differs from the motor's."""
        changes = self.unapplied()
        built = self._built_targets()
        for name, badge in self._badges.items():
            shown = built.get(name, (0.0, ""))[1]
            differs = bool(shown) and shown != self._target_shown(name)
            badge.setText(f"current motor: {shown}")
            badge.setToolTip(f"The motor is currently sized for {shown}. Apply to motor sizes it for the target;\n"
                             "click here (or Revert) to set the target back to this.")
            badge.setVisible(differs)
            field = self.P_cmbr.spin if name == "P_cmbr" else getattr(self, name)
            field.setProperty("changed", differs)
            field.style().unpolish(field)
            field.style().polish(field)
        self.revert_btn.setVisible(any(b.isVisibleTo(self) for b in self._badges.values()))
        self.apply_summary.setText("Applies: " + ", ".join(changes))
        self._apply_bar.setVisible(bool(changes))
        self.sized.emit()

    @staticmethod
    def _hole_D(cfg: dict) -> float:
        return to_si(float(cfg["inj_D"]), cfg["inj_D_unit"], "length")

    def set_theme(self, name: str):
        for card in (self.injector, self.nozzle, self.grain):
            card.sketch.set_theme(name)

    def _on_layout_picked(self, ports: int):
        """Use a layout's hole count; the injector then shows its drill."""
        self.sw_ports.setValue(ports)

    def cd(self) -> float:
        """The Cd on the hole or PTC bore: from the swirler geometry, a typed Cd, or the typed CdA ÷ that area."""
        bore, D_port = self.hole_D.si("length"), self.sw_D_port.si("length")
        if self._swirler and self.sw_cd_geom.isChecked() and D_port > 0:
            return swirl_cd(bore, self.sw_ports.value(), D_port, self.sw_R_in.si("length"), self.sw_xi.value())
        if self.cd_as.currentText() == "Cd":
            return self.inj_Cd.value()
        area = 0.25 * math.pi * bore ** 2
        return self.inj_CdA.si("area") / area if area > 0 else 0.0

    def _on_tank_length(self, *_):
        """Internal length is the tank size. Volume follows when an inside diameter is already set."""
        if self._loading or self._tank_sync:
            return
        D = self.tank_D.si("length")
        if D > 1e-9:
            self._tank_sync = True
            self.tank_V.set_si(self.tank_L.si("length") * 0.25 * math.pi * D ** 2, "volume")
            self._tank_sync = False
        self._on_motor_edited()

    def _on_tank_diameter(self, *_):
        self._on_tank_length()

    def _on_tank_volume(self, *_):
        """A typed volume sets the internal length when the diameter is known."""
        if self._loading or self._tank_sync:
            return
        D = self.tank_D.si("length")
        if D > 1e-9:
            self._tank_sync = True
            self.tank_L.set_si(self.tank_V.si("volume") / (0.25 * math.pi * D ** 2), "length")
            self._tank_sync = False
        self._on_motor_edited()

    def _on_cd_mode(self, *_):
        """Switch the injector entry between CdA and Cd without changing the flow."""
        if not self._loading and not (self._swirler and self.sw_cd_geom.isChecked()):
            area = 0.25 * math.pi * self.hole_D.si("length") ** 2
            if self.cd_as.currentText() == "Cd" and area > 0:
                self.inj_Cd.blockSignals(True)
                self.inj_Cd.setValue(self.inj_CdA.si("area") / area)
                self.inj_Cd.blockSignals(False)
            elif area > 0:
                self.inj_CdA.set_si(self.inj_Cd.value() * area, "area")
        self._on_motor_edited()

    def _on_ptc_stock(self, stock: bool):
        if stock:
            self.hole_D.set_si(STOCK_PTC_D, "length")
        self._on_motor_edited()

    def _on_fit(self):
        """Fit the stock PTC's size or the inlet loss to one swirler's measured CdA."""
        cda = self.sw_CdA_meas.si("area")
        geometry = (self.sw_ports.value(), self.sw_D_port.si("length"), self.sw_R_in.si("length"))
        try:
            if self.ptc_stock.isChecked():
                self.hole_D.set_si(swirl_exit_D(cda, *geometry, self.sw_xi.value()), "length")
            else:
                self.sw_xi.setValue(swirl_xi(cda, self.hole_D.si("length"), *geometry))
        except ValueError as exc:
            QMessageBox.information(self, "Fit", str(exc))
            return
        self.sw_cd_geom.setChecked(True)  # the fitted theory sets the Cd from here on
        self._on_motor_edited()

    def _values(self) -> dict:
        z, t = cast(Sizing, self._result), self.targets()
        return {
            "throat_D": self.throat_D.si("length") if self._fixed_nozzle() else z.throat_D,
            "ER": z.ER,
            "holes": t.holes or (self.holes.value() if self._solve_port() else max(1, round(z.holes))),
            "sw_D_port": self._layout.port_D if self._layout else None,
            "grain_L": z.grain_L,
        }

    def apply(self):
        """Make the sized throat, expansion ratio, hole count, swirler holes and grain length the motor's."""
        if self._result is None:
            return
        v = self._values()
        self._loading = True
        self.throat_D.set_si(v["throat_D"], "length")
        self.noz_ER.setValue(v["ER"])
        self.holes.setValue(v["holes"])
        if v["sw_D_port"]:
            self.sw_D_port.set_si(v["sw_D_port"], "length")
        if math.isfinite(v["grain_L"]) and not self._by_length():
            self.grain_L.set_si(v["grain_L"], "length")
        self._loading = False
        self._on_motor_edited()
        self.applied.emit()
