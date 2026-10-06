"""Motor tab: the whole motor, with each part either entered by hand or sized for targets at the start of the burn."""
from __future__ import annotations

import math
from typing import Callable, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.nox import nox, saturation_temperature
from hrap.engine.sizing import Sizing, SizingTargets, size_motor
from hrap.engine.swirl import swirl_A, swirl_cd, swirl_exit_D, swirl_fill, swirl_sensitivity, swirl_xi
from hrap.gui.sizing_viz import GrainSketch, InjectorSketch, NozzleSketch, Sketch
from hrap.gui.swirler_options import Layout, SwirlerOptions, Target, drilled_layout
from hrap.gui.widgets import PlainComboBox, PlainDoubleSpinBox, PlainSpinBox, UnitRow
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
LABEL_W = 150  # one label column width, so the Targets and Motor fields line up
UNIT_W = 72    # UnitRow's unit dropdown


def _beside(field: QWidget, box: QWidget) -> QWidget:
    """A field with a checkbox or button after it, on one row."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(field, 1)
    layout.addWidget(box)
    return row


def card_frame(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("sizingCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("cardTitle")
    layout.addWidget(heading)
    return frame, layout


class FieldGrid(QGridLayout):
    """Label / field / unit rows with the same column widths in every card."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(6)
        self.setVerticalSpacing(8)
        self.setColumnMinimumWidth(0, LABEL_W)
        self.setColumnMinimumWidth(2, UNIT_W)
        self.setColumnStretch(1, 1)
        self._rows = 0

    def add(self, label: str, field: QWidget, unit: str = "", muted: bool = False, span: bool = False) -> list[QWidget]:
        """Add a row and return its widgets, so the row can be hidden."""
        name = QLabel(label)
        if muted:
            name.setObjectName("cardLabel")
        self.addWidget(name, self._rows, 0)
        row: list[QWidget] = [name, field]
        if span or isinstance(field, (UnitRow, PlainComboBox)):
            self.addWidget(field, self._rows, 1, 1, 2)
        else:
            self.addWidget(field, self._rows, 1)
            if unit:
                row.append(QLabel(unit))
                self.addWidget(row[-1], self._rows, 2)
        self._rows += 1
        return row

    def add_below(self, widget: QWidget):
        """A line under the last row's field."""
        self.addWidget(widget, self._rows, 1, 1, 2, Qt.AlignmentFlag.AlignLeft)
        self._rows += 1


class Card(QFrame):
    """A titled block of label / value rows."""

    def __init__(self, title: str, rows: list[str], sketch: Sketch | None = None, inputs: QGridLayout | None = None,
                 sketch_over_results: bool = False):
        super().__init__()
        self.setObjectName("sizingCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.values: dict[str, QLabel] = {}
        self.labels: dict[str, QLabel] = {}
        for r, name in enumerate(rows):
            label = QLabel(name)
            label.setObjectName("cardLabel")
            value = QLabel("—")
            value.setObjectName("cardValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(label, r, 0)
            grid.addWidget(value, r, 1)
            self.values[name], self.labels[name] = value, label
        grid.setColumnStretch(0, 1)
        self.sketch = sketch
        if sketch is None:
            layout.addLayout(grid)
        else:
            row = QHBoxLayout()
            row.setSpacing(12)
            if not sketch_over_results:  # otherwise it tops the results column, keeping a wide sketch from widening the card
                row.addWidget(sketch, 0, Qt.AlignmentFlag.AlignTop)
            if inputs is not None:  # editable settings between the drawing and the results they give
                row.addLayout(inputs, 1)
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                divider.setObjectName("cardDivider")
                row.addSpacing(8)
                row.addWidget(divider)
                row.addSpacing(8)
                grid.setAlignment(Qt.AlignmentFlag.AlignTop)
            if sketch_over_results:
                column = QVBoxLayout()
                column.setSpacing(10)
                column.addWidget(sketch, 0, Qt.AlignmentFlag.AlignHCenter)
                column.addLayout(grid)
                column.addStretch(1)
                row.addLayout(column, 1)
            else:
                row.addLayout(grid, 1)
            layout.addLayout(row)
        layout.addStretch(1)

    def set(self, name: str, text: str, tip: str = ""):
        """Show a value; the tooltip says how it was worked out."""
        self.values[name].setText(text)
        self.labels[name].setToolTip(tip)
        self.values[name].setToolTip(tip)

    def show_row(self, name: str, visible: bool):
        self.labels[name].setVisible(visible)
        self.values[name].setVisible(visible)

    def clear(self):
        for value in self.values.values():
            value.setText("—")
        if self.sketch is not None:
            self.sketch.show_data(None)


class SizingPage(QWidget):
    motor_edited = Signal()  # a motor setting on this page changed
    sized = Signal()  # the sizing was worked out again, so what Apply to motor would change may differ
    applied = Signal()

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
        self.P_limit.setToolTip("The chamber's design pressure (absolute). The Motor, Simulation and Study tabs warn above it.")

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
        self.tank_V = UnitRow(VOLUME_ITEMS, "cm^3", 3)
        self.tank_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.tank_T = UnitRow(TEMP_ITEMS, "C", 3)
        self.fill = PlainDoubleSpinBox()
        self.fill.setRange(0, 100)
        self.fill.setDecimals(3)
        self.tank_D.setToolTip("Inside diameter. It only sets the tank's length in the drawings and mass properties.")
        self.tank_T.setToolTip("Starting tank temperature. It sets the tank pressure.")
        self.fill.setToolTip("Share of the tank volume that starts as liquid. The dip tube's length sets it.")
        self.tank_P = QLabel("—")
        self.ox_liquid = QLabel("—")
        self.tank_L = QLabel("—")
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
        tank_form.add("Volume", self.tank_V)
        tank_form.add("Diameter", self.tank_D)
        tank_form.add("Temperature", self.tank_T)
        tank_form.add("Fill", self.fill, "%")
        tank_form.add("Tank pressure", self.tank_P, muted=True)
        tank_form.add("Liquid oxidizer", self.ox_liquid, muted=True)
        tank_form.add("Length", self.tank_L, muted=True)
        tank_form.add("Vent", self.vent)
        self._vent_rows = [*tank_form.add("Vent diameter", self.vent_D), *tank_form.add("Vent Cd", self.vent_Cd)]
        tank_layout.addLayout(tank_form)

        self.propellant = PlainComboBox()
        for item in list_propellants():
            self.propellant.addItem(f"{item['name']} ({item['id']})", item["id"])
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
        self._hole_D_row = inj_form.add("Hole diameter", _beside(self.hole_D, self.ptc_stock), span=True)
        ports_row = inj_form.add("Swirler holes", self.sw_ports)
        self._sw_D_port_row = inj_form.add("Hole diameter", self.sw_D_port)
        self._swirler_rows = [*ports_row, *inj_form.add("Hole offset", self.sw_R_in), *inj_form.add("Inlet loss (ξ)", self.sw_xi),
                              *inj_form.add("Measured CdA", _beside(self.sw_CdA_meas, fit), span=True)]
        self._cd_row = inj_form.add("CdA per hole", _beside(self.inj_CdA, self.sw_cd_geom), span=True)
        inj_form.add("Flow model", self.inj_model)
        self._hem_row = inj_form.add("HEM Cd", _beside(self.inj_Cd_HEM, self.hem_same), span=True)
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
        grain_form = FieldGrid()
        grain_form.add("Sizing", self.grain_from)
        self._grain_L_row = grain_form.add("Grain length", self.grain_L)
        grain_form.add("Starting port", self.port_D)
        grain_form.add("Outer diameter", self.grain_OD)
        grain_form.add("Pre-combustion", self.pre_L)
        grain_form.add("Post-combustion", self.post_L)
        self.grain = Card("Grain", ["Fuel flow", "Oxidizer flux", "Grain length", "O/F", "Port at liquid burnout",
                                    "O/F at liquid burnout", "Fuel burned"], GrainSketch(), grain_form)
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
        buttons.setContentsMargins(16, 10, 16, 10)
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
        left.setSpacing(12)
        for card in (targets, tank, fuel):
            left.addWidget(card)
        left.addWidget(self.performance, 1)
        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self.injector)
        right.addWidget(self.grain)
        right.addWidget(self.nozzle, 1)
        body = QGridLayout()
        body.setHorizontalSpacing(16)
        body.setVerticalSpacing(12)
        body.addWidget(self.limit_warning, 0, 0, 1, 2)
        body.addWidget(self.error, 1, 0, 1, 2)
        body.addWidget(left_column, 2, 0)
        body.addLayout(right, 2, 1)
        body.setColumnStretch(1, 1)

        inner = QWidget()
        page = QVBoxLayout(inner)
        page.setContentsMargins(16, 16, 16, 16)
        page.setSpacing(12)
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
        for w in (self.P_limit, self.tank_V, self.tank_D, self.tank_T, self.vent_D, self.rho, self.hole_D, self.inj_CdA,
                  self.sw_D_port, self.sw_R_in,
                  self.port_D, self.grain_OD, self.grain_L, self.pre_L, self.post_L, self.Pa, self.throat_D):
            w.spin.valueChanged.connect(self._on_motor_edited)
            w.unit.currentTextChanged.connect(self._on_motor_edited)
        for spin in (self.fill, self.vent_Cd, self.prop_a, self.prop_n, self.prop_m, self.const_OF, self.cstar, self.sw_xi,
                     self.inj_Cd_HEM, self.dyer_kappa, self.holes, self.sw_ports, self.noz_Cd, self.noz_eff, self.noz_ER):
            spin.valueChanged.connect(self._on_motor_edited)
        for combo in (self.vent, self.inj_type, self.inj_model, self.reg_model):
            combo.currentIndexChanged.connect(self._on_motor_edited)
        for box in (self.sw_cd_geom, self.hem_same):
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
        for w in self._cd_row:
            w.setVisible(not solve)
        self.injector.show_row("Flow per hole", not solve)
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
        for w in self._grain_L_row:
            w.setVisible(self._by_length())
        self.grain.show_row("Grain length", not self._by_length())
        self.grain.show_row("O/F", self._by_length())
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

    def targets(self) -> SizingTargets:
        return SizingTargets(
            P_cmbr=None if self._fixed_nozzle() else self.P_cmbr.si("pressure"),
            burn_time=self.burn_time.value() if self._by_burn_time() else None,
            OF=self.OF.value(),
            port_D=to_si(self.port_D.spin.value(), self.port_D.unit.currentText(), "length"),
            holes=self.holes.value() if self._by_holes() else None,
            grain_L=to_si(self.grain_L.spin.value(), self.grain_L.unit.currentText(), "length") if self._by_length() else None,
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
        self._cd_row[0].setText("CdA per swirler" if swirler else "CdA per hole")
        geometry = swirler and self.sw_cd_geom.isChecked()
        self.inj_CdA.setEnabled(not geometry)
        bore = self.hole_D.si("length")
        self.inj_CdA.setToolTip(
            "Worked out from the swirler geometry." if geometry else
            f"The flow area after losses: a cold flow's water flow ÷ √(2 × 998 kg/m³ × ΔP). Divided by the\n"
            f"{'PTC bore' if swirler else 'hole'} area it's a Cd of {self.cd():.4f}, which is what the motor file stores.")
        if geometry and self.sw_D_port.si("length") > 0:
            self.inj_CdA.spin.blockSignals(True)
            self.inj_CdA.set_si(self.cd() * 0.25 * math.pi * bore ** 2, "area")
            self.inj_CdA.spin.blockSignals(False)
        model = self.inj_model.currentText()
        for w in self._hem_row:
            w.setVisible(model != "SPI")
        for w in self._dyer_row:
            w.setVisible(model == "Dyer")
        self.inj_Cd_HEM.setEnabled(not self.hem_same.isChecked())
        if self.hem_same.isChecked():
            self.inj_Cd_HEM.blockSignals(True)
            self.inj_Cd_HEM.setValue(self.cd())
            self.inj_Cd_HEM.blockSignals(False)
        for w in self._vent_rows:
            w.setVisible(self.vent.currentText() != "None")
        fixed = self._fixed_OF()
        for w in self._burn_rate_rows:
            w.setVisible(not fixed)
        for w in self._const_OF_row:
            w.setVisible(fixed)
        # With a fixed O/F, neither the flow nor the grain length changes the O/F, so neither can be sized for one.
        for combo, index in ((self.size_from, 2), (self.grain_from, 1)):
            cast(QStandardItemModel, combo.model()).item(index).setEnabled(not fixed)
            if fixed and combo.currentIndex() == index:
                combo.setCurrentIndex(0)
        self._show_size_from_rows()

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
        if int(cfg.get("tnk_V_state", 0)):
            self.tank_V.set_si(si("tnk_L") * 0.25 * math.pi * si("tnk_D") ** 2, "volume")
        else:
            show(self.tank_V, "tnk_V")
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
        self.inj_CdA.set_si(float(cfg.get("inj_Cd") or 1.0) * 0.25 * math.pi * self.hole_D.si("length") ** 2, "area")
        self.inj_model.setCurrentText(str(cfg.get("inj_model") or "SPI"))
        hem_cd = float(cfg.get("inj_Cd_HEM") or 0.0)
        self.hem_same.setChecked(not hem_cd)
        if hem_cd:
            self.inj_Cd_HEM.setValue(hem_cd)
        self.dyer_kappa.setValue(float(cfg.get("dyer_kappa") or 1.0))

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
            return {name: row.spin.value(), f"{name}_unit": row.unit.currentText()}

        L, _D, _V = self.tank_geometry()
        return {
            **field("tnk_V", self.tank_V), "tnk_V_state": 0, **field("tnk_D", self.tank_D),
            "tnk_L": from_si(L, self.tank_D.unit.currentText(), "length"), "tnk_L_unit": self.tank_D.unit.currentText(),
            "tnk_dd": "Starting Tank Temperature", "tnk_cond": self.tank_T.spin.value(),
            "T_tnk_unit": self.tank_T.unit.currentText(),
            "fill_dd": "Tank Fill Percentage", "fill": self.fill.value(), "fill_unit": "%",
            "vnt_state": self.vent.currentText(), **field("vnt_D", self.vent_D), "vnt_Cd": self.vent_Cd.value(),
            "prop_id": self.propellant.currentData() or "ABS",
            "prop_nm": (self.propellant.currentText() or "ABS").split(" (")[0],
            **field("prop_rho", self.rho), "prop_a": self.prop_a.value(), "prop_n": self.prop_n.value(),
            "prop_m": self.prop_m.value(), "reg_model": self.reg_model.currentData(), "const_OF": self.const_OF.value(),
            "cstar_eff": self.cstar.value(),
            "inj_type": self.inj_type.currentText(), **field("inj_D", self.hole_D), "inj_N": self.holes.value(),
            "inj_Cd": self.cd(), "sw_ports": self.sw_ports.value(), **field("sw_D_port", self.sw_D_port),
            **field("sw_R_in", self.sw_R_in), "sw_xi": self.sw_xi.value(), "sw_cd_from_geometry": self.sw_cd_geom.isChecked(),
            "ptc_stock": self.inj_type.currentText() == "Swirler" and self.ptc_stock.isChecked(),
            "inj_model": self.inj_model.currentText(),
            "inj_Cd_HEM": 0.0 if self.hem_same.isChecked() else self.inj_Cd_HEM.value(),
            "dyer_kappa": self.dyer_kappa.value(),
            **field("P_cmbr_max", self.P_limit),
            **field("grn_ID", self.port_D), **field("grn_OD", self.grain_OD), **field("grn_L", self.grain_L),
            **field("cmbr_pre_L", self.pre_L), **field("cmbr_post_L", self.post_L), "cmbr_V_state": 1,
            **field("noz_thrt", self.throat_D), "noz_def": "Nozzle Expansion Ratio", "noz_ex": self.noz_ER.value(),
            "noz_Cd": self.noz_Cd.value(),
            "noz_eff": self.noz_eff.value(), **field("Pa", self.Pa),
        }

    def tank_geometry(self) -> tuple[float, float, float]:
        """Tank length, diameter and volume in SI. The length follows from the volume and diameter."""
        D = self.tank_D.si("length") or self.grain_OD.si("length") or 0.05
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
        self.tank_L.setText(u.text(self.tank_geometry()[0], "length"))
        limit = chamber_limit(cfg)
        self.limit_warning.hide()
        try:
            z = size_motor(cfg, self.targets())
        except Exception as exc:  # the motor form can hold any combination; show why sizing can't run
            self._result = self._cfg = None
            self.swirler_options.update_target(None)
            self.error.setText(str(exc) or type(exc).__name__)
            self.error.show()
            for card in (self.injector, self.nozzle, self.grain, self.performance):
                card.clear()
            self._show_apply()
            return
        self._result, self._cfg = z, cfg
        try:
            t = self.targets()
            self._built = size_motor(cfg, SizingTargets(P_cmbr=None, burn_time=None, OF=t.OF, port_D=t.port_D,
                                                        holes=int(cfg["inj_N"]), grain_L=self.grain_L.si("length")))
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
        if t.holes:
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
        if t.holes:
            self.injector.set("Total CdA", cda, f"Cd × area over all your {noun}s. A cold flow measures this directly.")
        elif solve:
            self.injector.set("Total CdA", cda, "The Cd × area that gives the oxidizer flow. A cold flow measures this directly.")
        else:
            self.injector.set("Total CdA", cda,
                              f"The Cd × area over all {noun}s that gives the oxidizer flow. A cold flow measures this directly.\n"
                              f"{holes} of your {noun}s give {cda_now}.")
        if solve:
            self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                              f"liquid oxidizer ÷ oxidizer flow = {u.text(z.ox_liquid, 'mass')} ÷ {flow} = {lasts:.2f} s.\n"
                              "The real flow falls as the tank cools, so the full simulation runs a little longer.")
        else:
            self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                              f"With {holes} {noun}s the flow is {u.text(holes * z.flow_per_hole, 'mass_flow', 3)}, so\n"
                              f"liquid oxidizer ÷ flow = {u.text(z.ox_liquid, 'mass')} ÷ {u.text(holes * z.flow_per_hole, 'mass_flow', 3)} = {lasts:.2f} s.\n"
                              "The real flow falls as the tank cools, so the full simulation runs a little longer.")
        bore = to_si(float(cfg["grn_OD"]), cfg["grn_OD_unit"], "length")
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
        bore = to_si(float(self._cfg["grn_OD"]), self._cfg["grn_OD_unit"], "length")
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
        """The Cd on the hole or PTC bore: from the swirler geometry, or the typed CdA ÷ that area."""
        bore, D_port = self.hole_D.si("length"), self.sw_D_port.si("length")
        if self._swirler and self.sw_cd_geom.isChecked() and D_port > 0:
            return swirl_cd(bore, self.sw_ports.value(), D_port, self.sw_R_in.si("length"), self.sw_xi.value())
        area = 0.25 * math.pi * bore ** 2
        return self.inj_CdA.si("area") / area if area > 0 else 0.0

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
