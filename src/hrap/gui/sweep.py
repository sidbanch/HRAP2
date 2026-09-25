"""Throat × injector Cd sweep panel for the sizing page."""
from __future__ import annotations

import math
import traceback
from typing import Callable, cast

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hrap.engine.sweep import SweepCase, passing_throats, sweep, uses_spi
from hrap.units import DisplayUnits, from_si, to_si

OK = QColor("#2f7d4f")
OVER_LIMIT = QColor("#a8413b")
UNDER_LIMIT = QColor("#3b6ea8")
HIGH_DP = QColor("#a87a22")

METRICS = [
    ("Peak chamber pressure", "peak_P_cmbr", "pressure"),
    ("Average injector ΔP", "avg_inj_dP", "pressure"),
    ("Total impulse", "total_impulse", "impulse"),
    ("Peak thrust", "peak_thrust", "force"),
    ("Burn time", "burn_time", None),
]


class SweepWorker(QObject):
    case_done = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, cfg: dict, throats: list[float], cds: list[float]):
        super().__init__()
        self.cfg, self.throats, self.cds = cfg, throats, cds
        self.stop = False

    def run(self):
        try:
            for case in sweep(self.cfg, self.throats, self.cds):
                self.case_done.emit(case)
                if self.stop:
                    break
        except Exception:
            self.failed.emit(traceback.format_exc())
        self.finished.emit()


def _chip(color: QColor, text: str) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 12, 0)
    h.setSpacing(6)
    swatch = QLabel()
    swatch.setFixedSize(12, 12)
    swatch.setStyleSheet(f"background: {color.name()}; border-radius: 2px;")
    h.addWidget(swatch)
    h.addWidget(QLabel(text))
    return w


class SweepPanel(QFrame):
    """Full-simulation check of a sized motor across throat diameters and injector Cds."""

    finished = Signal()
    limit_edited = Signal()  # the chamber limit is a motor setting, shared with the Simulation tab

    def __init__(self, get_cfg: Callable[[], dict | None], get_units: Callable[[], DisplayUnits],
                 on_pick: Callable[[float], None]):
        super().__init__()
        self.setObjectName("sizingCard")
        self._get_cfg, self._get_units, self._on_pick = get_cfg, get_units, on_pick
        self.cases: list[SweepCase] = []
        self._throats: list[float] = []
        self._cells: dict[tuple[float, float], tuple[int, int]] = {}
        self._thread: QThread | None = None
        self._worker: SweepWorker | None = None
        self._error = ""
        self._picked: int | None = None
        self.spi = True
        self._length_unit = self._pressure_unit = ""

        # Throats step through round sizes in the length unit, so each one is a size you could machine.
        self.throat_lo, self.throat_hi = self._spin(0.0, 3), self._spin(0.0, 3)
        self.throat_step = self._spin(0.0, 3)
        self.throat_step.setMinimum(0.001)
        self.cd_lo, self.cd_hi = self._spin(0.3, 3), self._spin(0.9, 3)
        self.cd_n = self._count(6)
        self.min_chamber, self.max_chamber = self._spin(0.0, 0), self._spin(500.0, 0)
        self.max_dp = self._spin(300.0, 0)
        self.min_chamber.setToolTip("Lowest acceptable peak chamber pressure (absolute). Below it the throat is too big:\n"
                                    "the nozzle over-expands and the motor loses thrust and Isp.")
        self.max_chamber.setToolTip("Peak chamber pressure the chamber is designed for (absolute).\n"
                                    "Shared with the chamber pressure limit on the Simulation tab.")
        self.max_chamber.valueChanged.connect(self.limit_edited)
        self.max_dp.setToolTip("Above this burn-average ΔP, the SPI injector model overpredicts oxidizer flow.")
        self.sized_label = QLabel("")
        self.sized_label.setObjectName("cardLabel")
        self.model_label = QLabel("")
        self.model_label.setObjectName("cardLabel")

        heading = QLabel("Check across injector Cd")
        heading.setObjectName("cardTitle")
        intro = QLabel("Runs the full simulation of the sized motor for each throat and injector Cd, since the Cd isn't "
                       "known until the injector is flow tested. Click a column to use that throat.")
        intro.setObjectName("cardLabel")
        intro.setWordWrap(True)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        # Columns: label, from, "to", to, "step(s)" or unit, step or count, note.
        self.length_unit = QLabel("")
        self.pressure_unit = QLabel("")
        self.dp_label = QLabel("ΔP warning")
        self.dp_unit = QLabel("")
        rows = [
            ("Throat", self.throat_lo, self.throat_hi, self.length_unit, "step", self.throat_step, self.sized_label),
            ("Injector Cd", self.cd_lo, self.cd_hi, None, "steps", self.cd_n, None),
            ("Chamber pressure", self.min_chamber, self.max_chamber, self.pressure_unit, None, None, None),
        ]
        for r, (name, lo, hi, unit, step_name, step, note) in enumerate(rows):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(lo, r, 1)
            grid.addWidget(QLabel("to"), r, 2)
            grid.addWidget(hi, r, 3)
            if unit is not None:
                grid.addWidget(unit, r, 4, 1, 1 if step is not None else 4)
            if step is not None:
                grid.addWidget(QLabel(step_name), r, 5)
                grid.addWidget(step, r, 6)
            if note is not None:
                grid.addWidget(note, r, 7)
        grid.addWidget(self.dp_label, 3, 0)
        grid.addWidget(self.max_dp, 3, 1)
        grid.addWidget(self.dp_unit, 3, 2, 1, 5)
        grid.addWidget(self.model_label, 4, 0, 1, 8)
        grid.setColumnStretch(8, 1)

        self.run_btn = QPushButton("Run sweep")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.shown = QComboBox()
        self.shown.addItems([m[0] for m in METRICS])
        buttons = QHBoxLayout()
        for w in (self.run_btn, self.stop_btn, self.export_btn, self.progress):
            buttons.addWidget(w)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Show"))
        buttons.addWidget(self.shown)

        self.answer = QLabel("Run the sweep to see which throats stay in the chamber pressure range for every Cd.")
        self.answer.setWordWrap(True)
        font = QFont(self.answer.font())
        font.setBold(True)
        self.answer.setFont(font)

        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setMinimumHeight(260)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setHighlightSections(False)
        self.table.cellClicked.connect(lambda _row, col: self._pick(col))
        self.table.horizontalHeader().sectionClicked.connect(self._pick)

        self.dp_chip = _chip(HIGH_DP, "ΔP over the warning: the SPI injector model overpredicts flow")
        self.ok_chip_label = QLabel("")
        legend = QHBoxLayout()
        ok_chip = _chip(OK, "")
        ok_chip.layout().replaceWidget(ok_chip.layout().itemAt(1).widget(), self.ok_chip_label)
        legend.addWidget(ok_chip)
        legend.addWidget(_chip(OVER_LIMIT, "Over the chamber pressure range"))
        legend.addWidget(_chip(UNDER_LIMIT, "Under the chamber pressure range"))
        legend.addWidget(self.dp_chip)
        legend.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(intro)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.addWidget(self.answer)
        layout.addWidget(self.table, 1)
        self.legend = QWidget()
        legend.setContentsMargins(0, 0, 0, 0)
        self.legend.setLayout(legend)
        layout.addWidget(self.legend)
        self.table.hide()  # shown once a sweep runs
        self.legend.hide()

        for signal in (self.shown.currentIndexChanged, self.min_chamber.valueChanged, self.max_chamber.valueChanged,
                       self.max_dp.valueChanged):
            signal.connect(self._refresh)

    @staticmethod
    def _spin(value: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(0.0, 1e6)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setFixedWidth(96)
        return spin

    @staticmethod
    def _count(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 50)
        spin.setValue(value)
        spin.setFixedWidth(64)
        return spin

    def _throat_list(self) -> list[float]:
        u = self._get_units()
        lo, hi, step = self.throat_lo.value(), self.throat_hi.value(), self.throat_step.value()
        n = math.floor((hi - lo) / step + 1e-6) + 1
        return [to_si(round(lo + k * step, 6), u.length, "length") for k in range(max(n, 0))]

    def _throat_text(self, throat: float) -> str:
        """A throat in the length unit, with as many decimals as the step."""
        decimals = len(f"{self.throat_step.value():g}".partition(".")[2])
        return f"{from_si(throat, self._get_units().length, 'length'):.{decimals}f}"

    def _center_throats(self, throat: float):
        """Cover about 15% either side of the sized throat, rounded out to whole steps."""
        c, step = from_si(throat, self._get_units().length, "length"), self.throat_step.value()
        self.throat_lo.setValue(math.floor(0.85 * c / step) * step)
        self.throat_hi.setValue(math.ceil(1.15 * c / step) * step)

    def set_cd_range(self, cd: float):
        """Start the Cd range at half to one and a half times the motor's injector Cd."""
        self.cd_lo.setValue(0.5 * cd)
        self.cd_hi.setValue(min(1.0, 1.5 * cd))

    def update_motor(self, cfg: dict | None, throat: float):
        """Follow the sized motor: the throat range, the injector model and the units."""
        u = self._get_units()
        self.run_btn.setEnabled(cfg is not None and self._thread is None)
        self.sized_label.setText(f"sized {u.text(throat, 'length')}" if cfg is not None else "")
        self.sized_label.setToolTip("The throat from the Nozzle card above. Hover it there to see how it's worked out.")
        if u.length != self._length_unit:
            if self._length_unit:
                for spin in (self.throat_lo, self.throat_hi, self.throat_step):
                    spin.setValue(from_si(to_si(spin.value(), self._length_unit, "length"), u.length, "length"))
            else:
                self.throat_step.setValue(from_si(to_si(0.01, "in", "length"), u.length, "length"))
            self._length_unit = u.length
            self.length_unit.setText(u.length)
        if u.pressure != self._pressure_unit:
            old = self._pressure_unit or "psi"
            value = self.min_chamber.value() if self._pressure_unit else 400.0
            self.min_chamber.setValue(from_si(to_si(value, old, "pressure"), u.pressure, "pressure"))
            self._pressure_unit = u.pressure
        self.pressure_unit.setText(f"{u.pressure} (absolute)")
        for i, (label, _field, quantity) in enumerate(METRICS):  # the table cells are bare numbers
            self.shown.setItemText(i, f"{label}, {u.unit(quantity) if quantity else 's'}")
        self.dp_unit.setText(f"{u.pressure} (burn average)")
        if cfg is None:
            return
        throats = self._throat_list()
        if not throats or not throats[0] <= throat <= throats[-1]:
            self._center_throats(throat)
        self.spi = uses_spi(cfg)
        for w in (self.dp_label, self.max_dp, self.dp_unit, self.dp_chip):
            w.setVisible(self.spi)
        self.ok_chip_label.setText("In the chamber pressure range and under the ΔP warning" if self.spi
                                   else "In the chamber pressure range")
        if self.spi:
            self.model_label.setText("Injector model: SPI.")
        else:
            hem_cd = f"fixed at {cfg['inj_Cd_HEM']:.3g}" if cfg.get("inj_Cd_HEM") else "follows the swept Cd"
            model = f"Dyer (κ {cfg.get('dyer_kappa') or 1.0:.3g})" if cfg["inj_model"] == "Dyer" else cfg["inj_model"]
            self.model_label.setText(f"Injector model: {model}. HEM Cd {hem_cd}.")

    def busy(self) -> bool:
        return self._thread is not None

    def chamber_limit(self) -> float:
        return self._limits()[1]

    def set_chamber_limit(self, limit: float):
        self.max_chamber.blockSignals(True)
        self.max_chamber.setValue(from_si(limit, self._get_units().pressure, "pressure"))
        self.max_chamber.blockSignals(False)
        self._refresh()

    def _limits(self) -> tuple[float, float, float]:
        """Lowest and highest peak chamber pressure, and the ΔP warning (infinite outside SPI)."""
        p = self._get_units().pressure
        max_dp = to_si(self.max_dp.value(), p, "pressure") if self.spi else float("inf")
        return to_si(self.min_chamber.value(), p, "pressure"), to_si(self.max_chamber.value(), p, "pressure"), max_dp

    def _run(self):
        cfg = self._get_cfg()
        if cfg is None:
            return
        throats = self._throat_list()
        if not throats or throats[0] <= 0:
            self.answer.setText("The throat range has to start above zero and end above its start.")
            return
        if len(throats) > 40:
            self.answer.setText(f"That's {len(throats)} throats. Use a bigger step or a narrower range.")
            return
        cds = [float(c) for c in np.linspace(self.cd_lo.value(), self.cd_hi.value(), self.cd_n.value())]
        self._throats = throats
        self._cells = {(t, c): (j, i) for i, t in enumerate(throats) for j, c in enumerate(cds)}
        self.cases = []
        self._error = ""
        self._picked = None
        self.table.clear()
        self.table.setRowCount(len(cds))
        self.table.setColumnCount(len(throats))
        self._label_throats()
        self.table.setVerticalHeaderLabels([f"  Cd {c:.3g}  " for c in cds])
        self.table.show()
        self.legend.show()
        self.progress.setRange(0, len(self._cells))
        self.progress.setValue(0)
        self.progress.show()
        self.answer.setText(f"Running {len(self._cells)} simulations…")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = SweepWorker(cfg, throats, cds)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.case_done.connect(self._on_case)
        self._worker.failed.connect(lambda text: setattr(self, "_error", text.strip().splitlines()[-1]))
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _label_throats(self):
        u = self._get_units()
        self.table.setHorizontalHeaderLabels([
            f"{'▼ ' if i == self._picked else ''}{self._throat_text(t)} {u.length}"
            for i, t in enumerate(self._throats)
        ])

    def _pick(self, col: int):
        if not 0 <= col < len(self._throats):
            return
        self._picked = col
        self._label_throats()
        self._on_pick(self._throats[col])

    def clear_pick(self):
        if self._picked is not None:
            self._picked = None
            self._label_throats()

    def stop(self):
        if self._worker is not None:
            self._worker.stop = True
        self.stop_btn.setEnabled(False)

    _stop = stop

    def _on_case(self, case: SweepCase):
        self.cases.append(case)
        self.progress.setValue(len(self.cases))
        self._fill_cell(case)

    def _on_thread_finished(self):
        thread = cast(QThread, self._thread)
        thread.wait()
        thread.deleteLater()
        self._thread = self._worker = None
        self.progress.hide()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(bool(self.cases))
        self._update_answer()
        self.finished.emit()

    def _refresh(self):
        for case in self.cases:
            self._fill_cell(case)
        if self._thread is None and self._cells:
            self._update_answer()

    def _update_answer(self):
        if self._error:
            self.answer.setText(f"Sweep failed: {self._error}")
            return
        if len(self.cases) < len(self._cells):
            self.answer.setText(f"Stopped after {len(self.cases)} of {len(self._cells)} simulations.")
            return
        u = self._get_units()
        cd_range = f"Cd {self.cd_lo.value():.3g}–{self.cd_hi.value():.3g}"
        band = f"{self.min_chamber.value():.0f}–{self.max_chamber.value():.0f} {u.pressure}"
        ok = passing_throats(self.cases, *self._limits())
        if ok:
            listed = ", ".join(self._throat_text(t) for t in ok)
            dp = " and under the ΔP warning" if self.spi else ""
            self.answer.setText(f"Throats that stay in {band}{dp} for every {cd_range}: {listed} {u.length}. "
                                "Click one to use it.")
        else:
            self.answer.setText(f"No throat stays in {band} for every {cd_range}. "
                                "Read down a column to see which Cds that throat handles.")

    def _fill_cell(self, case: SweepCase):
        u = self._get_units()
        _label, field, quantity = METRICS[self.shown.currentIndex()]
        value = getattr(case, field)
        item = QTableWidgetItem(f"{u.value(value, quantity) if quantity else value:.4g}")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        min_P, max_P, max_dP = self._limits()
        color = (OVER_LIMIT if case.peak_P_cmbr > max_P else UNDER_LIMIT if case.peak_P_cmbr < min_P
                 else HIGH_DP if case.avg_inj_dP > max_dP else OK)
        item.setBackground(color)
        item.setForeground(QColor("white"))
        item.setToolTip("\n".join([
            f"Throat: {u.text(case.throat, 'length')}",
            f"Injector Cd: {case.inj_Cd:.3g}",
            f"Peak chamber pressure: {u.text(case.peak_P_cmbr, 'pressure')} (absolute)",
            f"Average injector ΔP: {u.text(case.avg_inj_dP, 'pressure')}",
            f"Total impulse: {u.text(case.total_impulse, 'impulse')}",
            f"Peak thrust: {u.text(case.peak_thrust, 'force')}",
            f"Burn time: {case.burn_time:.3g} s",
            f"End: {case.end_cond}",
        ]))
        self.table.setItem(*self._cells[(case.throat, case.inj_Cd)], item)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep", "HRAP_sweep.csv", "CSV (*.csv)")
        if not path:
            return
        u = self._get_units()
        header = (f"throat_{u.length},inj_Cd,peak_P_cmbr_{u.pressure},avg_inj_dP_{u.pressure},"
                  f"total_impulse_{u.force}s,peak_thrust_{u.force},burn_time_s,end_cond")
        rows = [
            f"{from_si(c.throat, u.length, 'length'):.6g},{c.inj_Cd:.6g},{u.value(c.peak_P_cmbr, 'pressure'):.6g},"
            f"{u.value(c.avg_inj_dP, 'pressure'):.6g},{u.value(c.total_impulse, 'impulse'):.6g},"
            f"{u.value(c.peak_thrust, 'force'):.6g},{c.burn_time:.6g},{c.end_cond}"
            for c in sorted(self.cases, key=lambda c: (c.throat, c.inj_Cd))
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join([header, *rows]) + "\n")
