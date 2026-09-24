"""Throat × injector Cd sweep panel for the sizing page."""
from __future__ import annotations

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
        self._center = 0.0
        self._length_unit = ""

        # The throat range is either ± a percentage of the sized throat, or a from / to range in a length unit.
        self.throat_mode = QComboBox()
        self.throat_mode.addItems(["± %", ""])
        self.throat_mode.setFixedWidth(72)
        self.throat_mode.setToolTip("Give the throat range as ± a percentage of the sized throat, or as a from / to range.")
        self.spread = self._spin(30.0, 0)
        self.throat_lo, self.throat_hi = self._spin(0.0, 3), self._spin(0.0, 3)
        self.throat_to = QLabel("to")
        self.throat_n = self._count(7)
        self.cd_lo, self.cd_hi = self._spin(0.3, 3), self._spin(0.9, 3)
        self.cd_n = self._count(6)
        self.max_chamber = self._spin(500.0, 0)
        self.max_dp = self._spin(300.0, 0)
        self.max_chamber.setToolTip("Peak chamber pressure the chamber is designed for (absolute).\n"
                                    "Shared with the chamber pressure limit on the Simulation tab.")
        self.max_chamber.valueChanged.connect(self.limit_edited)
        self.max_dp.setToolTip("Above this burn-average ΔP, the SPI injector model overpredicts oxidizer flow.")
        self.center_label = QLabel("")
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
        # Columns: label, range mode, from, "to", to, "steps", count. Rows without a mode span the first two.
        grid.addWidget(QLabel("Throat"), 0, 0)
        grid.addWidget(self.throat_mode, 0, 1)
        grid.addWidget(self.spread, 0, 2)
        grid.addWidget(self.center_label, 0, 3, 1, 2)
        grid.addWidget(self.throat_lo, 0, 2)
        grid.addWidget(self.throat_to, 0, 3)
        grid.addWidget(self.throat_hi, 0, 4)
        grid.addWidget(QLabel("steps"), 0, 5)
        grid.addWidget(self.throat_n, 0, 6)
        grid.addWidget(QLabel("Injector Cd"), 1, 0, 1, 2)
        grid.addWidget(self.cd_lo, 1, 2)
        grid.addWidget(QLabel("to"), 1, 3)
        grid.addWidget(self.cd_hi, 1, 4)
        grid.addWidget(QLabel("steps"), 1, 5)
        grid.addWidget(self.cd_n, 1, 6)
        grid.addWidget(QLabel("Chamber limit"), 2, 0, 1, 2)
        grid.addWidget(self.max_chamber, 2, 2)
        self.pressure_unit = QLabel("")
        grid.addWidget(self.pressure_unit, 2, 3, 1, 4)
        self.dp_label = QLabel("ΔP warning")
        self.dp_unit = QLabel("")
        grid.addWidget(self.dp_label, 3, 0, 1, 2)
        grid.addWidget(self.max_dp, 3, 2)
        grid.addWidget(self.dp_unit, 3, 3, 1, 4)
        grid.addWidget(self.model_label, 4, 0, 1, 7)
        grid.setColumnStretch(7, 1)
        self.throat_mode.currentIndexChanged.connect(self._on_throat_mode)
        self._on_throat_mode(0)

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

        self.answer = QLabel("Run the sweep to see which throats stay under the limit for every Cd.")
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
        legend.addWidget(_chip(OVER_LIMIT, "Chamber pressure over the limit"))
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

        for signal in (self.shown.currentIndexChanged, self.max_chamber.valueChanged, self.max_dp.valueChanged):
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

    def _on_throat_mode(self, index: int):
        percent = index == 0
        if not percent and self._center > 0:
            u, s = self._get_units(), self.spread.value() / 100.0
            self.throat_lo.setValue(from_si(self._center * (1 - s), u.length, "length"))
            self.throat_hi.setValue(from_si(self._center * (1 + s), u.length, "length"))
        for w in (self.spread, self.center_label):
            w.setVisible(percent)
        for w in (self.throat_lo, self.throat_to, self.throat_hi):
            w.setVisible(not percent)

    def _throat_range(self) -> tuple[float, float]:
        if self.throat_mode.currentIndex() == 0:
            s = self.spread.value() / 100.0
            return self._center * (1 - s), self._center * (1 + s)
        u = self._get_units()
        return to_si(self.throat_lo.value(), u.length, "length"), to_si(self.throat_hi.value(), u.length, "length")

    def set_cd_range(self, cd: float):
        """Start the Cd range at half to one and a half times the motor's injector Cd."""
        self.cd_lo.setValue(0.5 * cd)
        self.cd_hi.setValue(min(1.0, 1.5 * cd))

    def update_motor(self, cfg: dict | None, throat: float):
        """Follow the sized motor: the throat range centre, the injector model and the units."""
        u = self._get_units()
        self._center = throat
        self.run_btn.setEnabled(cfg is not None and self._thread is None)
        self.center_label.setText(f"of the sized {u.text(throat, 'length')} throat" if cfg is not None else "")
        self.center_label.setToolTip("The throat from the Nozzle card above. Hover it there to see how it's worked out.")
        if u.length != self._length_unit:
            if self._length_unit:
                for spin in (self.throat_lo, self.throat_hi):
                    spin.setValue(from_si(to_si(spin.value(), self._length_unit, "length"), u.length, "length"))
            self._length_unit = u.length
            self.throat_mode.setItemText(1, u.length)
        self.pressure_unit.setText(f"{u.pressure} (absolute)")
        self.dp_unit.setText(f"{u.pressure} (burn average)")
        if cfg is None:
            return
        self.spi = uses_spi(cfg)
        for w in (self.dp_label, self.max_dp, self.dp_unit, self.dp_chip):
            w.setVisible(self.spi)
        self.ok_chip_label.setText("Under both limits" if self.spi else "Under the chamber limit")
        if self.spi:
            self.model_label.setText("Injector model: SPI.")
        else:
            hem_cd = f"fixed at {cfg['inj_Cd_HEM']:.3g}" if cfg.get("inj_Cd_HEM") else "follows the swept Cd"
            model = f"Dyer (κ {cfg.get('dyer_kappa') or 1.0:.3g})" if cfg["inj_model"] == "Dyer" else cfg["inj_model"]
            self.model_label.setText(f"Injector model: {model}. HEM Cd {hem_cd}.")

    def busy(self) -> bool:
        return self._thread is not None

    def chamber_limit(self) -> float:
        return self._limits()[0]

    def set_chamber_limit(self, limit: float):
        self.max_chamber.blockSignals(True)
        self.max_chamber.setValue(from_si(limit, self._get_units().pressure, "pressure"))
        self.max_chamber.blockSignals(False)
        self._refresh()

    def _limits(self) -> tuple[float, float]:
        p = self._get_units().pressure
        max_dp = to_si(self.max_dp.value(), p, "pressure") if self.spi else float("inf")
        return to_si(self.max_chamber.value(), p, "pressure"), max_dp

    def _run(self):
        cfg = self._get_cfg()
        if cfg is None or self._center <= 0:
            return
        lo, hi = self._throat_range()
        if not 0 < lo <= hi:
            self.answer.setText("The throat range has to start above zero and end above its start.")
            return
        throats = [float(t) for t in np.linspace(lo, hi, self.throat_n.value())]
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
            f"{'▼ ' if i == self._picked else ''}{from_si(t, u.length, 'length'):.4g} {u.length}"
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
        max_P, max_dP = self._limits()
        cd_range = f"Cd {self.cd_lo.value():.3g}–{self.cd_hi.value():.3g}"
        ok = passing_throats(self.cases, max_P, max_dP)
        if ok:
            listed = ", ".join(f"{from_si(t, u.length, 'length'):.4g}" for t in ok)
            self.answer.setText(f"Throats under {'both limits' if self.spi else 'the chamber limit'} for every {cd_range}: {listed} {u.length}. Click one to use it.")
            return
        over_P = any(c.peak_P_cmbr > max_P for c in self.cases)
        over_dP = any(c.avg_inj_dP > max_dP for c in self.cases)
        why = ("small throats go over the chamber limit and large throats go over the ΔP warning"
               if over_P and over_dP else
               "every throat goes over the chamber limit somewhere" if over_P else
               "every throat goes over the ΔP warning at low Cd")
        self.answer.setText(f"No throat in this range works for every {cd_range}: {why}.")

    def _fill_cell(self, case: SweepCase):
        u = self._get_units()
        _label, field, quantity = METRICS[self.shown.currentIndex()]
        value = getattr(case, field)
        item = QTableWidgetItem(u.text(value, quantity) if quantity else f"{value:.3g} s")
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        max_P, max_dP = self._limits()
        color = OVER_LIMIT if case.peak_P_cmbr > max_P else HIGH_DP if case.avg_inj_dP > max_dP else OK
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
