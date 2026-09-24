"""Throat × injector Cd sweep window."""
from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
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


class SweepDialog(QDialog):
    def __init__(self, cfg: dict, units: DisplayUnits, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Throat / injector Cd sweep")
        self.resize(860, 620)
        self.cfg, self.units = cfg, units
        self.cases: list[SweepCase] = []
        self._cells: dict[tuple[float, float], tuple[int, int]] = {}
        self._thread: QThread | None = None
        self._worker: SweepWorker | None = None
        self._error = ""
        self.spi = uses_spi(cfg)
        self.limits = "both limits" if self.spi else "the chamber limit"

        length, pressure = units.length, units.pressure
        throat = from_si(to_si(cfg["noz_thrt"], cfg["noz_thrt_unit"], "length"), length, "length")
        cd = float(cfg["inj_Cd"])
        self.throat_lo, self.throat_hi = self._spin(0.7 * throat, 4), self._spin(1.3 * throat, 4)
        self.throat_n = self._count(7)
        self.cd_lo, self.cd_hi = self._spin(0.5 * cd, 3), self._spin(min(1.0, 1.5 * cd), 3)
        self.cd_n = self._count(6)
        self.max_chamber = self._spin(from_si(to_si(500.0, "psi", "pressure"), pressure, "pressure"), 0)
        self.max_dp = self._spin(from_si(to_si(300.0, "psi", "pressure"), pressure, "pressure"), 0)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        rows = [
            ("Throat diameter", self.throat_lo, self.throat_hi, length, self.throat_n),
            ("Injector Cd", self.cd_lo, self.cd_hi, "", self.cd_n),
        ]
        for r, (label, lo, hi, unit, n) in enumerate(rows):
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(lo, r, 1)
            grid.addWidget(QLabel("to"), r, 2)
            grid.addWidget(hi, r, 3)
            grid.addWidget(QLabel(unit), r, 4)
            grid.addWidget(QLabel("steps"), r, 5)
            grid.addWidget(n, r, 6)
        grid.addWidget(QLabel("Chamber pressure limit"), 2, 0)
        grid.addWidget(self.max_chamber, 2, 1)
        grid.addWidget(QLabel(f"{pressure} (absolute)"), 2, 2, 1, 3)
        if self.spi:
            grid.addWidget(QLabel("Injector ΔP warning"), 3, 0)
            grid.addWidget(self.max_dp, 3, 1)
            grid.addWidget(QLabel(f"{pressure} (burn average)"), 3, 2, 1, 3)
        else:
            hem_cd = f"fixed at {cfg['inj_Cd_HEM']:.3g}" if cfg.get("inj_Cd_HEM") else "follows the swept Cd"
            model = f"Dyer (κ {cfg.get('dyer_kappa') or 1.0:.3g})" if cfg["inj_model"] == "Dyer" else cfg["inj_model"]
            grid.addWidget(QLabel("Injector model"), 3, 0)
            grid.addWidget(QLabel(f"{model}. HEM Cd {hem_cd}."), 3, 1, 1, 6)
        grid.setColumnStretch(7, 1)

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

        self.answer = QLabel("Pick ranges, then run the sweep.")
        self.answer.setWordWrap(True)
        font = QFont(self.answer.font())
        font.setPointSizeF(font.pointSizeF() * 1.2)
        font.setBold(True)
        self.answer.setFont(font)

        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setCornerButtonEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setHighlightSections(False)

        legend = QHBoxLayout()
        legend.addWidget(_chip(OK, f"Under {self.limits}"))
        legend.addWidget(_chip(OVER_LIMIT, "Chamber pressure over the limit"))
        if self.spi:
            legend.addWidget(_chip(HIGH_DP, "ΔP over the warning: HRAP's liquid-only injector model overpredicts flow"))
        legend.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addLayout(grid)
        layout.addLayout(buttons)
        layout.addWidget(self.answer)
        layout.addWidget(self.table, 1)
        layout.addLayout(legend)

        for signal in (self.shown.currentIndexChanged, self.max_chamber.valueChanged, self.max_dp.valueChanged):
            signal.connect(self._refresh)

    @staticmethod
    def _spin(value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(0.0, 1e6)
        spin.setValue(value)
        spin.setFixedWidth(96)
        return spin

    @staticmethod
    def _count(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 50)
        spin.setValue(value)
        spin.setFixedWidth(64)
        return spin

    def _limits(self) -> tuple[float, float]:
        p = self.units.pressure
        max_dp = to_si(self.max_dp.value(), p, "pressure") if self.spi else float("inf")
        return to_si(self.max_chamber.value(), p, "pressure"), max_dp

    def _run(self):
        u = self.units
        throats = [to_si(t, u.length, "length")
                   for t in np.linspace(self.throat_lo.value(), self.throat_hi.value(), self.throat_n.value())]
        cds = [float(c) for c in np.linspace(self.cd_lo.value(), self.cd_hi.value(), self.cd_n.value())]
        self._cells = {(t, c): (i, j) for i, t in enumerate(throats) for j, c in enumerate(cds)}
        self.cases = []
        self._error = ""
        self.table.clear()
        self.table.setRowCount(len(throats))
        self.table.setColumnCount(len(cds))
        self.table.setVerticalHeaderLabels([f"  {from_si(t, u.length, 'length'):.4g} {u.length}  " for t in throats])
        self.table.setHorizontalHeaderLabels([f"Cd {c:.3g}" for c in cds])
        self.progress.setRange(0, len(self._cells))
        self.progress.setValue(0)
        self.progress.show()
        self.answer.setText(f"Running {len(self._cells)} simulations…")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = SweepWorker(self.cfg, throats, cds)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.case_done.connect(self._on_case)
        self._worker.failed.connect(lambda text: setattr(self, "_error", text.strip().splitlines()[-1]))
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.stop = True
        self.stop_btn.setEnabled(False)

    def _on_case(self, case: SweepCase):
        self.cases.append(case)
        self.progress.setValue(len(self.cases))
        self._fill_cell(case)

    def _on_thread_finished(self):
        thread = self._thread
        thread.wait()
        thread.deleteLater()
        self._thread = self._worker = None
        self.progress.hide()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(bool(self.cases))
        self._update_answer()

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
        u = self.units
        max_P, max_dP = self._limits()
        cd_range = f"Cd {self.cd_lo.value():.3g}–{self.cd_hi.value():.3g}"
        ok = passing_throats(self.cases, max_P, max_dP)
        if ok:
            listed = ", ".join(f"{from_si(t, u.length, 'length'):.4g}" for t in ok)
            self.answer.setText(f"Throats under {self.limits} for every {cd_range}: {listed} {u.length}")
            return
        over_P = any(c.peak_P_cmbr > max_P for c in self.cases)
        over_dP = any(c.avg_inj_dP > max_dP for c in self.cases)
        why = ("small throats go over the chamber limit and large throats go over the ΔP warning"
               if over_P and over_dP else
               "every throat goes over the chamber limit somewhere" if over_P else
               "every throat goes over the ΔP warning at low Cd")
        self.answer.setText(f"No throat in this range works for every {cd_range}: {why}.")

    def _fill_cell(self, case: SweepCase):
        u = self.units
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
        u = self.units
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

    def closeEvent(self, event):
        if self._thread is not None:
            self._stop()
            event.ignore()
            self.answer.setText("Stopping after the running simulations finish…")
            self._thread.finished.connect(self.close)
        else:
            event.accept()
