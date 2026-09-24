"""Sizing page: injector, nozzle and grain for a target chamber pressure and burn time."""
from __future__ import annotations

import math
from typing import Callable, cast

from PySide6.QtCore import Qt
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
from hrap.gui.widgets import PlainDoubleSpinBox, UnitRow
from hrap.units import LENGTH_ITEMS, PRESSURE_ITEMS, DisplayUnits, from_si, to_si


class Card(QFrame):
    """A titled block of label / value rows."""

    def __init__(self, title: str, rows: list[str]):
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

        targets = QFrame()
        targets.setObjectName("sizingCard")
        tl = QVBoxLayout(targets)
        tl.setContentsMargins(14, 12, 14, 12)
        heading = QLabel("Targets")
        heading.setObjectName("cardTitle")
        tl.addWidget(heading)
        form = QFormLayout()
        form.addRow("Chamber pressure", self.P_cmbr)
        form.addRow("Burn time [s]", self.burn_time)
        form.addRow("O/F", self.OF)
        form.addRow("Starting port diameter", self.port_D)
        tl.addLayout(form)
        note = QLabel("Tank, injector hole, propellant and efficiencies come from the Simulation tab.")
        note.setObjectName("cardLabel")
        note.setWordWrap(True)
        tl.addWidget(note)

        self.motor = Card("From the motor", ["Tank", "Liquid oxidizer", "Injector", "Propellant"])
        self.injector = Card("Injector", ["Injector ΔP", "Oxidizer flow", "Flow per hole", "Holes needed", "Rounded"])
        self.nozzle = Card("Nozzle", ["Throat diameter", "Expansion ratio", "Exit diameter", "C*"])
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
        buttons = QHBoxLayout()
        buttons.addWidget(self.apply_btn)
        buttons.addStretch(1)
        self.sweep = SweepPanel(self.sized_cfg, get_units, self._on_pick)

        intro = QLabel("Sizes the injector, nozzle and grain for conditions at the start of the burn, using the "
                       "simulation's own injector, combustion and nozzle equations. Apply the result, then run "
                       "the simulation to see the whole burn.")
        intro.setWordWrap(True)

        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(targets)
        left.addWidget(self.motor)
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
        right.addLayout(buttons)
        right.addStretch(1)
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addLayout(left, 2)
        body.addLayout(right, 3)

        inner = QWidget()
        page = QVBoxLayout(inner)
        page.setContentsMargins(16, 16, 16, 16)
        page.setSpacing(12)
        page.addWidget(intro)
        page.addLayout(body)
        page.addWidget(self.sweep, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

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

        hole_D = to_si(float(cfg["inj_D"]), cfg["inj_D_unit"], "length")
        model = cfg.get("inj_model", "SPI")
        self.motor.set("Tank", f"{u.text(z.P_tnk, 'pressure')} at start")
        self.motor.set("Liquid oxidizer", u.text(z.ox_liquid, "mass"))
        self.motor.set("Injector", f"Ø{u.text(hole_D, 'length', 3)} holes, Cd {float(cfg['inj_Cd']):.2f}, {model}")
        self.motor.set("Propellant", str(cfg.get("prop_nm") or cfg.get("prop_id")))

        t = self.targets()
        holes = max(1, round(z.holes))
        lasts = z.ox_liquid / (holes * z.flow_per_hole)
        self.injector.set("Injector ΔP", f"{u.text(z.inj_dP, 'pressure')} ({100 * z.inj_dP / t.P_cmbr:.0f}% of chamber)")
        self.injector.set("Oxidizer flow", u.text(z.mdot_o, "mass_flow", 3))
        self.injector.set("Flow per hole", u.text(z.flow_per_hole, "mass_flow", 3))
        self.injector.set("Holes needed", f"{z.holes:.2f}")
        self.injector.set("Rounded", f"{holes} holes, liquid lasts {lasts:.2f} s")

        self._show_throat()
        self.nozzle.set("Expansion ratio", f"{z.ER:.2f} (exit at ambient pressure)")
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
            self.nozzle.set("Throat diameter", f"{u.text(throat, 'length')}, picked (sized {u.text(z.throat_D, 'length')})")
        else:
            self.nozzle.set("Throat diameter", u.text(throat, "length"))
        self.nozzle.set("Exit diameter", u.text(throat * math.sqrt(z.ER), "length"))

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
