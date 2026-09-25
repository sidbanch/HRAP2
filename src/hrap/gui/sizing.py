"""Sizing page: injector, nozzle and grain for a target chamber pressure and burn time."""
from __future__ import annotations

import math
from typing import Callable, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.sizing import Sizing, SizingTargets, size_motor
from hrap.engine.swirl import swirl_A, swirl_fill
from hrap.gui.sizing_viz import GrainSketch, InjectorSketch, NozzleSketch, Sketch
from hrap.gui.sweep import SweepPanel
from hrap.gui.widgets import PlainComboBox, PlainDoubleSpinBox, PlainSpinBox, UnitRow
from hrap.io.config import chamber_limit, injector_cd
from hrap.io.propellant import list_propellants
from hrap.units import LENGTH_ITEMS, PRESSURE_ITEMS, TEMP_ITEMS, VOLUME_ITEMS, DisplayUnits, from_si, to_si


LABEL_W = 150  # one label column width, so the Targets and Motor fields line up
UNIT_W = 72    # UnitRow's unit dropdown


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


class Card(QFrame):
    """A titled block of label / value rows."""

    def __init__(self, title: str, rows: list[str], sketch: Sketch | None = None, inputs: QGridLayout | None = None):
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
    motor_edited = Signal()  # a motor field on this page changed; the Simulation settings should follow

    def __init__(
        self,
        get_cfg: Callable[[], dict],
        get_units: Callable[[], DisplayUnits],
        apply: Callable[[dict], None],
    ):
        super().__init__()
        self._get_cfg, self._get_units, self._apply = get_cfg, get_units, apply
        self._result: Sizing | None = None
        self._cfg: dict | None = None
        self._picked_throat: float | None = None
        self._loading = False
        self._swirler = False

        self.P_cmbr = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.burn_time = PlainDoubleSpinBox(); self.burn_time.setRange(0.1, 120); self.burn_time.setDecimals(2)
        self.OF = PlainDoubleSpinBox(); self.OF.setRange(0.1, 50); self.OF.setDecimals(2)
        self.P_cmbr.setToolTip("Chamber pressure at the start of the burn (absolute). It falls as the tank cools.")
        self.burn_time.setToolTip("How long the liquid lasts at the starting oxidizer flow. It sets the oxidizer flow,\n"
                                  "and the hole count follows from it. The real flow falls during the burn, so the\n"
                                  "liquid lasts somewhat longer. A weak vapor tail follows once the liquid runs out.")
        self.size_from = PlainComboBox()
        self.size_from.addItems(["Liquid burn time", "Hole count"])
        self.size_from.setToolTip("Pick the liquid burn time and get the hole count, or pick the hole count and get the burn time.")
        self.holes = PlainSpinBox(); self.holes.setRange(1, 200)
        self.holes.setToolTip("Injector hole count, shared with the Simulation tab. The oxidizer flow and burn time follow from it.")
        self.OF.setToolTip("Oxidizer-to-fuel ratio at the start of the burn. It sets the grain length.\n"
                           "With a regression law it drifts during the burn.")
        self.grain_from = PlainComboBox()
        self.grain_from.addItems(["O/F", "Grain length"])
        self.grain_from.setToolTip("Pick the starting O/F and get the grain length, or pick the grain length and get the O/F.")
        self.port_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.grain_OD = UnitRow(LENGTH_ITEMS, "in", 4)
        self.grain_L = UnitRow(LENGTH_ITEMS, "in", 3)
        self.port_D.setToolTip("Starting port diameter, shared with the Simulation tab.")
        self.grain_OD.setToolTip("Grain outer diameter, shared with the Simulation tab. It's also the chamber bore in the drawings.")
        self.grain_L.setToolTip("Grain length, shared with the Simulation tab. The fuel flow and starting O/F follow from it.")

        targets, tl = card_frame("Targets")
        form = FieldGrid()
        form.add("Chamber pressure", self.P_cmbr)
        self._burn_time_row = form.add("Liquid burn time", self.burn_time, "s")
        self._OF_row = form.add("O/F", self.OF)
        tl.addLayout(form)

        # Motor inputs that drive sizing. They mirror the Simulation tab's fields (MainWindow keeps them in step).
        self.tank_V = UnitRow(VOLUME_ITEMS, "cm^3", 1)
        self.tank_T = UnitRow(TEMP_ITEMS, "C", 2)
        self.fill = PlainDoubleSpinBox(); self.fill.setRange(0, 100); self.fill.setDecimals(1)
        self.hole_D = UnitRow(LENGTH_ITEMS, "in", 5)
        self.inj_Cd = PlainDoubleSpinBox(); self.inj_Cd.setRange(0, 1); self.inj_Cd.setDecimals(3)
        self.inj_model = PlainComboBox(); self.inj_model.addItems(["SPI", "HEM", "Dyer"])
        self.inj_type = PlainComboBox(); self.inj_type.addItems(["Holes", "Swirler"])
        self.inj_type.setToolTip("Holes: straight drilled holes.\n"
                                 "Swirler: tangential ports spin the nitrous in a small chamber before the exit orifice.")
        self.sw_ports = PlainSpinBox(); self.sw_ports.setRange(1, 12)
        self.sw_D_port = UnitRow(LENGTH_ITEMS, "in", 4)
        self.sw_R_in = UnitRow(LENGTH_ITEMS, "in", 4)
        self.sw_ports.setToolTip("Number of tangential inlet ports into the swirl chamber.")
        self.sw_D_port.setToolTip("Diameter of each tangential inlet port.")
        self.sw_R_in.setToolTip("Distance from the swirler's axis to each inlet port's axis.")
        self.propellant = PlainComboBox()
        for item in list_propellants():
            self.propellant.addItem(f"{item['name']} ({item['id']})", item["id"])
        # Long propellant names would otherwise widen the whole inputs column.
        self.propellant.setSizeAdjustPolicy(PlainComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.propellant.setMinimumContentsLength(12)
        self.cstar = PlainDoubleSpinBox(); self.cstar.setRange(0, 100); self.cstar.setDecimals(1)
        self.noz_Cd = PlainDoubleSpinBox(); self.noz_Cd.setRange(0, 1); self.noz_Cd.setDecimals(3)
        self.tank_T.setToolTip("Starting tank temperature. It sets the tank pressure.")
        self.cstar.setToolTip("C* efficiency: how completely the propellants burn. Small hybrids are usually 85–95%.")
        self.noz_Cd.setToolTip("Throat Cd: how much of the throat area flows, about 0.97–0.99 for a smooth throat.")
        self.tank_P = QLabel("—")
        self.ox_liquid = QLabel("—")
        motor, ml = card_frame("Motor")
        mform = FieldGrid()
        mform.add("Tank volume", self.tank_V)
        mform.add("Tank temperature", self.tank_T)
        mform.add("Fill", self.fill, "%")
        mform.add("Tank pressure", self.tank_P, muted=True)
        mform.add("Liquid oxidizer", self.ox_liquid, muted=True)
        mform.add("Propellant", self.propellant)
        mform.add("C* efficiency", self.cstar, "%")
        mform.add("Throat Cd", self.noz_Cd)
        ml.addLayout(mform)
        shared = QLabel("Shared with the Simulation tab, which has the rest of the motor.")
        shared.setObjectName("cardLabel")
        shared.setWordWrap(True)
        ml.addWidget(shared)
        self.motor_fields = (self.tank_V.spin, self.tank_V.unit, self.tank_T.spin, self.tank_T.unit, self.fill,
                             self.hole_D.spin, self.hole_D.unit, self.inj_Cd, self.inj_model, self.propellant,
                             self.cstar, self.noz_Cd, self.inj_type, self.sw_ports, self.sw_D_port.spin,
                             self.sw_D_port.unit, self.sw_R_in.spin, self.sw_R_in.unit, self.port_D.spin, self.port_D.unit,
                             self.grain_OD.spin, self.grain_OD.unit, self.grain_L.spin, self.grain_L.unit)
        for w in self.motor_fields:
            signal = w.currentIndexChanged if isinstance(w, PlainComboBox) else w.valueChanged
            signal.connect(self._on_motor_edited)
        inj_form = FieldGrid()
        inj_form.add("Size from", self.size_from)
        inj_form.add("Type", self.inj_type)
        self._holes_row = inj_form.add("Hole count", self.holes)
        self._hole_D_label = inj_form.add("Hole diameter", self.hole_D)[0]
        self._swirler_rows = [*inj_form.add("Inlet ports", self.sw_ports), *inj_form.add("Inlet port diameter", self.sw_D_port),
                              *inj_form.add("Port offset from axis", self.sw_R_in)]
        self.sw_cd_geom = QCheckBox("From geometry")
        self.sw_cd_geom.setToolTip("Work the swirler's Cd out from its geometry (Abramovich's theory for an ideal liquid).\n"
                                   "Untick it to type a Cd measured in a cold flow.")
        self.sw_cd_geom.toggled.connect(self._on_motor_edited)
        cd_row = QWidget()
        cd_layout = QHBoxLayout(cd_row)
        cd_layout.setContentsMargins(0, 0, 0, 0)
        cd_layout.setSpacing(6)
        cd_layout.addWidget(self.inj_Cd, 1)
        cd_layout.addWidget(self.sw_cd_geom)
        inj_form.add("Cd", cd_row, span=True)
        inj_form.add("Flow model", self.inj_model)
        self.injector = Card("Injector", ["Oxidizer flow", "Injector ΔP", "ΔP / chamber", "Flow per hole",
                                          "Holes", "Liquid lasts"], InjectorSketch(), inj_form)
        self.nozzle = Card("Nozzle", ["Throat diameter", "Sized throat", "Expansion ratio", "Exit diameter", "C*"], NozzleSketch())
        self.nozzle.show_row("Sized throat", False)
        grain_form = FieldGrid()
        grain_form.add("Size from", self.grain_from)
        self._grain_L_row = grain_form.add("Grain length", self.grain_L)
        grain_form.add("Starting port", self.port_D)
        grain_form.add("Outer diameter", self.grain_OD)
        self.grain = Card("Grain", ["Fuel flow", "Oxidizer flux", "Grain length", "O/F", "Port at liquid burnout",
                                    "O/F at liquid burnout", "Fuel burned"], GrainSketch(), grain_form)
        self.performance = Card("Performance at the start", ["Thrust", "Isp", "Impulse over the burn time"])

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
        self.apply_btn.setToolTip("Copy the throat, expansion ratio, rounded hole count, grain length and O/F into the motor.")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_summary = QLabel("")
        bar = QFrame()
        bar.setObjectName("applyBar")
        buttons = QHBoxLayout(bar)
        buttons.setContentsMargins(16, 10, 16, 10)
        buttons.addWidget(self.apply_summary, 1)
        buttons.addWidget(self.apply_btn)
        self.sweep = SweepPanel(self.sized_cfg, get_units, self._on_pick)
        self.sweep.limit_edited.connect(self._on_motor_edited)

        intro = QLabel("Sizes the injector, nozzle and grain for conditions at the start of the burn, using the "
                       "simulation's own injector, combustion and nozzle equations. Apply the result, then run "
                       "the simulation to see the whole burn.")
        intro.setObjectName("cardLabel")
        intro.setWordWrap(True)

        inputs = QWidget()
        inputs.setFixedWidth(380)
        left = QVBoxLayout(inputs)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(targets)
        left.addWidget(motor)
        left.addStretch(1)
        results = QGridLayout()
        results.setSpacing(12)
        results.addWidget(self.injector, 0, 0, 1, 2)
        results.addWidget(self.grain, 1, 0, 1, 2)
        results.addWidget(self.nozzle, 2, 0)
        results.addWidget(self.performance, 2, 1)
        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self.limit_warning)
        right.addWidget(self.error)
        right.addLayout(results)
        right.addWidget(self.sweep, 1)
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(inputs, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(right, 1)

        inner = QWidget()
        page = QVBoxLayout(inner)
        page.setContentsMargins(16, 16, 16, 16)
        page.setSpacing(12)
        page.addWidget(intro)
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
        self.holes.valueChanged.connect(self._on_motor_edited)
        self.size_from.currentIndexChanged.connect(self._on_size_from)
        self.grain_from.currentIndexChanged.connect(self._on_size_from)
        self._show_size_from_rows()

    def _by_holes(self) -> bool:
        return self.size_from.currentIndex() == 1

    def _show_size_from_rows(self):
        for w in self._burn_time_row:
            w.setVisible(not self._by_holes())
        for w in self._holes_row:
            w.setVisible(self._by_holes())
        self.injector.show_row("Holes", not self._by_holes())  # with a chosen count it's an input
        for w in self._OF_row:
            w.setVisible(not self._by_length())
        for w in self._grain_L_row:
            w.setVisible(self._by_length())
        self.grain.show_row("Grain length", not self._by_length())
        self.grain.show_row("O/F", self._by_length())

    def _by_length(self) -> bool:
        return self.grain_from.currentIndex() == 1

    def _on_size_from(self, *_):
        self._show_size_from_rows()
        self.refresh()

    def targets(self) -> SizingTargets:
        return SizingTargets(
            P_cmbr=to_si(self.P_cmbr.spin.value(), self.P_cmbr.unit.currentText(), "pressure"),
            burn_time=self.burn_time.value(),
            OF=self.OF.value(),
            port_D=to_si(self.port_D.spin.value(), self.port_D.unit.currentText(), "length"),
            holes=self.holes.value() if self._by_holes() else None,
            grain_L=to_si(self.grain_L.spin.value(), self.grain_L.unit.currentText(), "length") if self._by_length() else None,
        )

    def targets_cfg(self) -> dict:
        t = self.targets()
        return {"size_from": "holes" if self._by_holes() else "burn_time", "grain_from": "grain_L" if self._by_length() else "OF",
                "P_cmbr": t.P_cmbr, "burn_time": t.burn_time, "OF": t.OF}

    def set_targets(self, saved: dict, motor_cfg: dict):
        """Load saved targets, or start from the motor's own O/F."""
        self._loading = True
        self.P_cmbr.set_display(from_si(saved.get("P_cmbr") or to_si(400.0, "psi", "pressure"), self.P_cmbr.unit.currentText(), "pressure"))
        self.burn_time.setValue(float(saved.get("burn_time") or 5.0))
        self.size_from.setCurrentIndex(1 if saved.get("size_from") == "holes" else 0)
        self.grain_from.setCurrentIndex(1 if saved.get("grain_from") == "grain_L" else 0)
        self.OF.setValue(float(saved.get("OF") or motor_cfg.get("const_OF") or 6.0))
        self.sweep.set_cd_range(injector_cd(motor_cfg))
        self._loading = False
        if self.isVisible():
            self.refresh()

    def _on_motor_edited(self, *_):
        if not self._loading:
            self.motor_edited.emit()

    def show_motor(self, values: dict):
        """Load the Simulation tab's current values into this page's motor fields, without echoing back."""
        self._loading = True
        self.tank_V.set_display(from_si(values["tank_V"], self.tank_V.unit.currentText(), "volume"))
        self.tank_V.setEnabled(values["tank_V_editable"])
        self.tank_T.set_display(from_si(values["tank_T"], self.tank_T.unit.currentText(), "temperature"))
        self.fill.setValue(100.0 * values["fill"])
        self.hole_D.set_display(from_si(values["hole_D"], self.hole_D.unit.currentText(), "length"))
        self.inj_Cd.setValue(values["inj_Cd"])
        self.inj_Cd.setEnabled(values["inj_Cd_editable"])
        self.inj_Cd.setToolTip("" if values["inj_Cd_editable"] else "Worked out from the swirler geometry.")
        self.sw_cd_geom.setChecked(values["cd_from_geometry"])
        self.sw_cd_geom.setVisible(values["inj_type"] == "Swirler")
        self._swirler = values["inj_type"] == "Swirler"
        self.inj_type.setCurrentText(values["inj_type"])
        self.sw_ports.setValue(values["sw_ports"])
        self.sw_D_port.set_display(from_si(values["sw_D_port"], self.sw_D_port.unit.currentText(), "length"))
        self.sw_R_in.set_display(from_si(values["sw_R_in"], self.sw_R_in.unit.currentText(), "length"))
        for w in self._swirler_rows:
            w.setVisible(self._swirler)
        self.size_from.setItemText(1, "Swirler count" if self._swirler else "Hole count")
        self._hole_D_label.setText("Exit diameter" if self._swirler else "Hole diameter")
        self._holes_row[0].setText("Swirler count" if self._swirler else "Hole count")
        self.injector.labels["Flow per hole"].setText("Flow per swirler" if self._swirler else "Flow per hole")
        self.injector.labels["Holes"].setText("Swirlers" if self._swirler else "Holes")
        self.sweep.set_chamber_limit(values["P_cmbr_max"])
        self.inj_model.setCurrentText(values["inj_model"])
        self.propellant.setCurrentIndex(max(self.propellant.findData(values["prop_id"]), 0))
        self.cstar.setValue(values["cstar"])
        self.noz_Cd.setValue(values["noz_Cd"])
        self.holes.setValue(values["holes"])
        self.port_D.set_display(from_si(values["port_D"], self.port_D.unit.currentText(), "length"))
        self.grain_OD.set_display(from_si(values["grain_OD"], self.grain_OD.unit.currentText(), "length"))
        self.grain_L.set_display(from_si(values["grain_L"], self.grain_L.unit.currentText(), "length"))
        self._loading = False

    def motor_values(self) -> dict:
        return {
            "tank_V": to_si(self.tank_V.spin.value(), self.tank_V.unit.currentText(), "volume"),
            "tank_T": to_si(self.tank_T.spin.value(), self.tank_T.unit.currentText(), "temperature"),
            "fill": self.fill.value() / 100.0,
            "hole_D": to_si(self.hole_D.spin.value(), self.hole_D.unit.currentText(), "length"),
            "inj_Cd": self.inj_Cd.value(),
            "inj_model": self.inj_model.currentText(),
            "prop_id": self.propellant.currentData(),
            "cstar": self.cstar.value(),
            "noz_Cd": self.noz_Cd.value(),
            "holes": self.holes.value(),
            "P_cmbr_max": self.sweep.chamber_limit(),
            "inj_type": self.inj_type.currentText(),
            "cd_from_geometry": self.sw_cd_geom.isChecked(),
            "sw_ports": self.sw_ports.value(),
            "sw_D_port": to_si(self.sw_D_port.spin.value(), self.sw_D_port.unit.currentText(), "length"),
            "sw_R_in": to_si(self.sw_R_in.spin.value(), self.sw_R_in.unit.currentText(), "length"),
            "port_D": to_si(self.port_D.spin.value(), self.port_D.unit.currentText(), "length"),
            "grain_OD": to_si(self.grain_OD.spin.value(), self.grain_OD.unit.currentText(), "length"),
            "grain_L": to_si(self.grain_L.spin.value(), self.grain_L.unit.currentText(), "length"),
        }

    def refresh(self):
        if self._loading:
            return
        u = self._get_units()
        cfg = self._get_cfg()
        target, limit = self.targets().P_cmbr, chamber_limit(cfg)
        self.limit_warning.setText(f"The {u.text(target, 'pressure')} chamber pressure target is above the "
                                   f"{u.text(limit, 'pressure')} chamber pressure limit.")
        self.limit_warning.setVisible(target > limit)
        try:
            z = size_motor(cfg, self.targets())
        except Exception as exc:  # the motor form can hold any combination; show why sizing can't run
            self._result = self._cfg = None
            self.sweep.update_motor(None, 0.0)
            self.error.setText(str(exc) or type(exc).__name__)
            self.error.show()
            self.apply_btn.setEnabled(False)
            self.apply_summary.setText("")
            for card in (self.injector, self.nozzle, self.grain, self.performance):
                card.clear()
            return
        if self._result is None or abs(z.throat_D - self._result.throat_D) > 1e-12:
            self._picked_throat = None
            self.sweep.clear_pick()
        self._result, self._cfg = z, cfg
        self.sweep.update_motor(self.sized_cfg(), z.throat_D)
        self.error.hide()
        self.apply_btn.setEnabled(True)

        self.tank_P.setText(u.text(z.P_tnk, "pressure"))
        self.ox_liquid.setText(u.text(z.ox_liquid, "mass"))

        t = self.targets()
        holes = self._values()["holes"]
        lasts = z.ox_liquid / (holes * z.flow_per_hole)
        flow, per_hole = u.text(z.mdot_o, "mass_flow", 3), u.text(z.flow_per_hole, "mass_flow", 3)
        dP, P_cmbr = u.text(z.inj_dP, "pressure"), u.text(t.P_cmbr, "pressure")
        hole, inj_Cd, model = u.text(self._hole_D(cfg), "length"), injector_cd(cfg), cfg.get("inj_model") or "SPI"
        noun = "swirler" if self._swirler else "hole"
        if t.holes:
            self.injector.set("Oxidizer flow", flow,
                              f"The flow through your {noun}s:\n{noun} count × flow per {noun} = {holes} × {per_hole} = {flow}")
        else:
            self.injector.set("Oxidizer flow", flow,
                              f"The flow that empties the liquid in the burn time:\n"
                              f"liquid oxidizer ÷ liquid burn time = {u.text(z.ox_liquid, 'mass')} ÷ {t.burn_time:.3g} s = {flow}")
        self.injector.set("Injector ΔP", dP,
                          f"tank pressure − chamber pressure = {u.text(z.P_tnk, 'pressure')} − {P_cmbr} = {dP}")
        self.injector.set("ΔP / chamber", f"{100 * z.inj_dP / t.P_cmbr:.0f}%",
                          f"Injector ΔP as a share of chamber pressure: {dP} ÷ {P_cmbr}.\n"
                          "Above about 20%, chamber pressure swings barely change the injector flow,\n"
                          "which avoids feed-coupled combustion instability.")
        self.injector.set("Flow per hole", per_hole,
                          f"Flow through one {hole} {'swirler exit' if self._swirler else 'hole'} at Cd {inj_Cd:.3g} with {dP} across it,\n"
                          f"from the {model} injector model.")
        if t.holes:
            self.injector.set("Holes", f"{holes}", f"Your {noun} count, from Targets.")
        else:
            self.injector.set("Holes", f"{z.holes:.2f} → {holes}",
                              f"oxidizer flow ÷ flow per {noun} = {flow} ÷ {per_hole} = {z.holes:.2f},\n"
                              f"rounded to {holes}. Apply to motor uses {holes}.")
        self.injector.set("Liquid lasts", f"{lasts:.2f} s",
                          f"With {holes} {noun}s the flow is {u.text(holes * z.flow_per_hole, 'mass_flow', 3)}, so\n"
                          f"liquid oxidizer ÷ flow = {u.text(z.ox_liquid, 'mass')} ÷ {u.text(holes * z.flow_per_hole, 'mass_flow', 3)} = {lasts:.2f} s.\n"
                          "The real flow falls as the tank cools, so the full simulation runs a little longer.")
        bore = to_si(float(cfg["grn_OD"]), cfg["grn_OD_unit"], "length")
        if self._swirler:
            D_port = to_si(float(cfg["sw_D_port"]), cfg["sw_D_port_unit"], "length")
            R_in = to_si(float(cfg["sw_R_in"]), cfg["sw_R_in_unit"], "length")
            ports = int(cfg["sw_ports"])
            fill = swirl_fill(swirl_A(self._hole_D(cfg), ports, D_port, R_in))
            self.injector.sketch.show_data({
                "bore": bore,
                "swirler": {"exit": self._hole_D(cfg), "ports": ports, "port": D_port, "offset": R_in, "fill": fill},
                "caption": f"{ports} tangential ports, {hole} exit"})
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
        if t.grain_L:
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
                           "The wider port lowers the flux but adds burning wall, so the O/F drifts.")
            self.grain.set("Fuel burned", u.text(z.fuel_burned, "mass"),
                           "The fuel between the starting port and the port at liquid burnout.")
        else:
            for name in ("Grain length", "Port at liquid burnout", "O/F at liquid burnout", "Fuel burned"):
                self.grain.set(name, "needs a regression law (a > 0)")

        self.performance.set("Thrust", u.text(z.thrust, "force"))
        self.performance.set("Isp", f"{z.isp:.0f} s")
        self.performance.set("Impulse over the burn time", u.text(z.thrust * z.burn_time, "impulse"))

    def _show_throat(self):
        z, u = cast(Sizing, self._result), self._get_units()
        throat = self._picked_throat or z.throat_D
        t = self.targets()
        how = (f"Sized throat: the throat that holds {u.text(t.P_cmbr, 'pressure')} in the chamber at the starting flow.\n"
               f"throat area = total flow × C* ÷ (chamber pressure × throat Cd)\n"
               f"total flow = {u.text(z.mdot_o + z.mdot_f, 'mass_flow', 3)}, C* = {z.cstar:.0f} m/s, throat Cd = {self.noz_Cd.value():.3g}")
        self.nozzle.set("Throat diameter", u.text(throat, "length"), "Picked in the Cd check below." if self._picked_throat else how)
        self.nozzle.set("Sized throat", u.text(z.throat_D, "length"), how)
        self.nozzle.show_row("Sized throat", bool(self._picked_throat))
        self.nozzle.set("Exit diameter", u.text(throat * math.sqrt(z.ER), "length"))
        bore = to_si(float(self._cfg["grn_OD"]), self._cfg["grn_OD_unit"], "length")
        self.nozzle.sketch.show_data({"bore": bore, "throat": throat, "exit": throat * math.sqrt(z.ER),
                                      "caption": f"{u.text(throat, 'length')} throat"})
        v = self._values()
        parts = [f"Throat {u.text(v['throat_D'], 'length')}{' (picked)' if self._picked_throat else ''}",
                 f"expansion ratio {v['ER']:.2f}", f"{v['holes']} {'swirler' if self._swirler else 'hole'}{'' if v['holes'] == 1 else 's'}"]
        if math.isfinite(v["grain_L"]) and not self._by_length():
            parts.append(f"grain {u.text(v['grain_L'], 'length')}")
        parts.append(f"O/F {v['OF']:.2f}")
        self.apply_summary.setText("Applies: " + ", ".join(parts))

    @staticmethod
    def _hole_D(cfg: dict) -> float:
        return to_si(float(cfg["inj_D"]), cfg["inj_D_unit"], "length")

    def set_theme(self, name: str):
        for card in (self.injector, self.nozzle, self.grain):
            card.sketch.set_theme(name)

    def _on_pick(self, throat: float):
        self._picked_throat = throat
        self._show_throat()

    def _values(self) -> dict:
        z, t = cast(Sizing, self._result), self.targets()
        return {
            "throat_D": self._picked_throat or z.throat_D,
            "ER": z.ER,
            "holes": t.holes or max(1, round(z.holes)),
            "grain_L": z.grain_L,
            "OF": z.OF,
        }

    def sized_cfg(self) -> dict | None:
        """The motor with this sizing applied, for the sweep."""
        if self._result is None or self._cfg is None:
            return None
        v = self._values()
        cfg = dict(self._cfg)
        cfg.update(noz_thrt=v["throat_D"], noz_thrt_unit="m", noz_def="Nozzle Expansion Ratio", noz_ex=v["ER"],
                   inj_N=v["holes"], const_OF=v["OF"])
        if math.isfinite(v["grain_L"]):
            cfg.update(grn_L=v["grain_L"], grn_L_unit="m")
        return cfg

    def _on_apply(self):
        if self._result is None:
            return
        self._apply(self._values())
