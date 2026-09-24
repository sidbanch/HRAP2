"""Sizing page: injector, nozzle and grain for a target chamber pressure and burn time."""
from __future__ import annotations

import math
from typing import Callable, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
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
from hrap.gui.sweep import SweepPanel
from hrap.gui.widgets import PlainComboBox, PlainDoubleSpinBox, UnitRow
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

    def add(self, label: str, field: QWidget, unit: str = "", muted: bool = False):
        name = QLabel(label)
        if muted:
            name.setObjectName("cardLabel")
        self.addWidget(name, self._rows, 0)
        if isinstance(field, (UnitRow, PlainComboBox)):
            self.addWidget(field, self._rows, 1, 1, 2)
        else:
            self.addWidget(field, self._rows, 1)
            if unit:
                self.addWidget(QLabel(unit), self._rows, 2)
        self._rows += 1


class Card(QFrame):
    """A titled block of label / value rows."""

    def __init__(self, title: str, rows: list[str], tips: dict[str, str] | None = None):
        super().__init__()
        self.setObjectName("sizingCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        self.values: dict[str, QLabel] = {}
        for r, name in enumerate(rows):
            label = QLabel(name)
            label.setObjectName("cardLabel")
            value = QLabel("—")
            value.setObjectName("cardValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if tips and name in tips:
                label.setToolTip(tips[name])
                value.setToolTip(tips[name])
            grid.addWidget(label, r, 0)
            grid.addWidget(value, r, 1)
            self.values[name] = value
        grid.setColumnStretch(0, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

    def set(self, name: str, text: str):
        self.values[name].setText(text)

    def clear(self):
        for value in self.values.values():
            value.setText("—")


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

        self.P_cmbr = UnitRow(PRESSURE_ITEMS, "psi", 1)
        self.burn_time = PlainDoubleSpinBox(); self.burn_time.setRange(0.1, 120); self.burn_time.setDecimals(2)
        self.OF = PlainDoubleSpinBox(); self.OF.setRange(0.1, 50); self.OF.setDecimals(2)
        self.port_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.P_cmbr.setToolTip("Chamber pressure at the start of the burn (absolute). It falls as the tank cools.")
        self.burn_time.setToolTip("How long the liquid lasts at the starting oxidizer flow. The real flow falls\n"
                                  "during the burn, so the liquid lasts somewhat longer, then a short vapor tail follows.")
        self.OF.setToolTip("Oxidizer-to-fuel ratio at the start of the burn. With a regression law it drifts during the burn.")
        self.port_D.setToolTip("Starting port diameter. With the O/F target it sets the grain length.")

        targets, tl = card_frame("Targets")
        form = FieldGrid()
        form.add("Chamber pressure", self.P_cmbr)
        form.add("Burn time", self.burn_time, "s")
        form.add("O/F", self.OF)
        form.add("Starting port", self.port_D)
        tl.addLayout(form)

        # Motor inputs that drive sizing. They mirror the Simulation tab's fields (MainWindow keeps them in step).
        self.tank_V = UnitRow(VOLUME_ITEMS, "cm^3", 1)
        self.tank_T = UnitRow(TEMP_ITEMS, "C", 2)
        self.fill = PlainDoubleSpinBox(); self.fill.setRange(0, 100); self.fill.setDecimals(1)
        self.hole_D = UnitRow(LENGTH_ITEMS, "in", 4)
        self.inj_Cd = PlainDoubleSpinBox(); self.inj_Cd.setRange(0, 1); self.inj_Cd.setDecimals(3)
        self.inj_model = PlainComboBox(); self.inj_model.addItems(["SPI", "HEM", "Dyer"])
        self.propellant = PlainComboBox()
        for item in list_propellants():
            self.propellant.addItem(f"{item['name']} ({item['id']})", item["id"])
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
        mform.add("Injector hole", self.hole_D)
        mform.add("Injector Cd", self.inj_Cd)
        mform.add("Injector model", self.inj_model)
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
                             self.cstar, self.noz_Cd)
        for w in self.motor_fields:
            signal = w.currentIndexChanged if isinstance(w, PlainComboBox) else w.valueChanged
            signal.connect(self._on_motor_edited)
        self.injector = Card("Injector", ["Injector ΔP", "Oxidizer flow", "Flow per hole", "Holes", "Liquid lasts"],
                             {"Holes": "Holes needed for the oxidizer flow, rounded to a whole number.",
                              "Liquid lasts": "How long the liquid lasts at the starting flow with the rounded hole count."})
        self.nozzle = Card("Nozzle", ["Throat diameter", "Expansion ratio", "Exit diameter", "C*"],
                           {"Expansion ratio": "Sized so the exit pressure matches ambient pressure at the start of the burn."})
        self.grain = Card("Grain", ["Fuel flow", "Oxidizer flux", "Grain length", "Port at liquid burnout",
                                    "O/F at liquid burnout", "Fuel burned"])
        self.performance = Card("Performance at the start", ["Thrust", "Isp", "Impulse over the burn time"])

        self.error = QLabel("")
        self.error.setObjectName("sizingError")
        self.error.setWordWrap(True)
        self.error.hide()
        self.apply_btn = QPushButton("Apply to motor")
        self.apply_btn.setObjectName("runButton")
        self.apply_btn.setToolTip("Copy the throat, expansion ratio, rounded hole count, port, grain length and O/F into the motor.")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_summary = QLabel("")
        bar = QFrame()
        bar.setObjectName("applyBar")
        buttons = QHBoxLayout(bar)
        buttons.setContentsMargins(16, 10, 16, 10)
        buttons.addWidget(self.apply_summary, 1)
        buttons.addWidget(self.apply_btn)
        self.sweep = SweepPanel(self.sized_cfg, get_units, self._on_pick)

        intro = QLabel("Sizes the injector, nozzle and grain for conditions at the start of the burn, using the "
                       "simulation's own injector, combustion and nozzle equations. Apply the result, then run "
                       "the simulation to see the whole burn.")
        intro.setObjectName("cardLabel")
        intro.setWordWrap(True)

        inputs = QWidget()
        inputs.setFixedWidth(420)
        left = QVBoxLayout(inputs)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(targets)
        left.addWidget(motor)
        left.addStretch(1)
        results = QGridLayout()
        results.setSpacing(12)
        results.addWidget(self.injector, 0, 0)
        results.addWidget(self.nozzle, 0, 1)
        results.addWidget(self.grain, 1, 0)
        results.addWidget(self.performance, 1, 1)
        right = QVBoxLayout()
        right.setSpacing(12)
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

        for spin in (self.P_cmbr.spin, self.burn_time, self.OF, self.port_D.spin):
            spin.valueChanged.connect(self.refresh)

    def targets(self) -> SizingTargets:
        return SizingTargets(
            P_cmbr=to_si(self.P_cmbr.spin.value(), self.P_cmbr.unit.currentText(), "pressure"),
            burn_time=self.burn_time.value(),
            OF=self.OF.value(),
            port_D=to_si(self.port_D.spin.value(), self.port_D.unit.currentText(), "length"),
        )

    def targets_cfg(self) -> dict:
        t = self.targets()
        return {"P_cmbr": t.P_cmbr, "burn_time": t.burn_time, "OF": t.OF, "port_D": t.port_D}

    def set_targets(self, saved: dict, motor_cfg: dict):
        """Load saved targets, or start from the motor's own port and O/F."""
        self._loading = True
        port = saved.get("port_D") or to_si(float(motor_cfg.get("grn_ID") or 0.0), motor_cfg.get("grn_ID_unit") or "in", "length")
        self.P_cmbr.set_display(from_si(saved.get("P_cmbr") or to_si(400.0, "psi", "pressure"), self.P_cmbr.unit.currentText(), "pressure"))
        self.burn_time.setValue(float(saved.get("burn_time") or 5.0))
        self.OF.setValue(float(saved.get("OF") or motor_cfg.get("const_OF") or 6.0))
        self.port_D.set_display(from_si(port, self.port_D.unit.currentText(), "length"))
        self.sweep.set_cd_range(float(motor_cfg.get("inj_Cd") or 0.6))
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
        self.inj_model.setCurrentText(values["inj_model"])
        self.propellant.setCurrentIndex(max(self.propellant.findData(values["prop_id"]), 0))
        self.cstar.setValue(values["cstar"])
        self.noz_Cd.setValue(values["noz_Cd"])
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
        }

    def refresh(self):
        if self._loading:
            return
        u = self._get_units()
        cfg = self._get_cfg()
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
        holes = max(1, round(z.holes))
        lasts = z.ox_liquid / (holes * z.flow_per_hole)
        self.injector.set("Injector ΔP", f"{u.text(z.inj_dP, 'pressure')} ({100 * z.inj_dP / t.P_cmbr:.0f}% of chamber)")
        self.injector.set("Oxidizer flow", u.text(z.mdot_o, "mass_flow", 3))
        self.injector.set("Flow per hole", u.text(z.flow_per_hole, "mass_flow", 3))
        self.injector.set("Holes", f"{holes} ({z.holes:.2f} needed)")
        self.injector.set("Liquid lasts", f"{lasts:.2f} s")

        self._show_throat()
        self.nozzle.set("Expansion ratio", f"{z.ER:.2f}")
        self.nozzle.set("C*", f"{z.cstar:.0f} m/s")

        self.grain.set("Fuel flow", u.text(z.mdot_f, "mass_flow", 3))
        self.grain.set("Oxidizer flux", f"{z.ox_flux:.0f} kg/(m²·s)")
        if math.isfinite(z.grain_L):
            self.grain.set("Grain length", u.text(z.grain_L, "length"))
            self.grain.set("Port at liquid burnout", u.text(z.port_D_end, "length"))
            self.grain.set("O/F at liquid burnout", f"{z.OF_end:.2f}")
            self.grain.set("Fuel burned", u.text(z.fuel_burned, "mass"))
        else:
            for name in ("Grain length", "Port at liquid burnout", "O/F at liquid burnout", "Fuel burned"):
                self.grain.set(name, "needs a regression law (a > 0)")

        self.performance.set("Thrust", u.text(z.thrust, "force"))
        self.performance.set("Isp", f"{z.isp:.0f} s")
        self.performance.set("Impulse over the burn time", u.text(z.thrust * t.burn_time, "impulse"))

    def _show_throat(self):
        z, u = cast(Sizing, self._result), self._get_units()
        throat = self._picked_throat or z.throat_D
        if self._picked_throat:
            self.nozzle.set("Throat diameter", f"{u.text(throat, 'length')} (sized {u.text(z.throat_D, 'length')})")
        else:
            self.nozzle.set("Throat diameter", u.text(throat, "length"))
        self.nozzle.set("Exit diameter", u.text(throat * math.sqrt(z.ER), "length"))
        v = self._values()
        parts = [f"Throat {u.text(v['throat_D'], 'length')}{' (picked)' if self._picked_throat else ''}",
                 f"expansion ratio {v['ER']:.2f}", f"{v['holes']} holes", f"port {u.text(v['port_D'], 'length')}"]
        if math.isfinite(v["grain_L"]):
            parts.append(f"grain {u.text(v['grain_L'], 'length')}")
        self.apply_summary.setText("Applies: " + ", ".join(parts))

    def _on_pick(self, throat: float):
        self._picked_throat = throat
        self._show_throat()

    def _values(self) -> dict:
        z, t = cast(Sizing, self._result), self.targets()
        return {
            "throat_D": self._picked_throat or z.throat_D,
            "ER": z.ER,
            "holes": max(1, round(z.holes)),
            "port_D": t.port_D,
            "grain_L": z.grain_L,
            "OF": t.OF,
        }

    def sized_cfg(self) -> dict | None:
        """The motor with this sizing applied, for the sweep."""
        if self._result is None or self._cfg is None:
            return None
        v = self._values()
        cfg = dict(self._cfg)
        cfg.update(noz_thrt=v["throat_D"], noz_thrt_unit="m", noz_def="Nozzle Expansion Ratio", noz_ex=v["ER"],
                   inj_N=v["holes"], grn_ID=v["port_D"], grn_ID_unit="m", const_OF=v["OF"])
        if math.isfinite(v["grain_L"]):
            cfg.update(grn_L=v["grain_L"], grn_L_unit="m")
        return cfg

    def _on_apply(self):
        if self._result is None:
            return
        self._apply(self._values())
