"""The injector's total CdA against time, for a motor whose injector flow changes over the burn."""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hrap.gui.widgets import FIELD_GAP, UNIT_W, UnitCombo
from hrap.units import AREA_ITEMS, convert, pretty_unit


def read_points(text: str) -> list[tuple[float, float]]:
    """(time, CdA) rows from spreadsheet cells or a CSV: time first, CdA second. Lines that aren't two numbers are
    skipped. Under a header with more columns, the first column named CdA is used."""
    col, points = 1, []
    for line in text.splitlines():
        cells = re.split(r"[\t,;]", line.strip())
        if len(cells) == 1:
            cells = line.split()
        try:
            points.append((float(cells[0]), float(cells[col])))
        except (ValueError, IndexError):
            if not points:
                col = next((i for i, cell in enumerate(cells) if "cda" in cell.lower()), col)
    return points


class CdaCurve(QWidget):
    """A small table of time (s) against total CdA: typed, pasted from a spreadsheet, or read from a CSV."""

    edited = Signal()

    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, 2)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFixedHeight(156)
        self.table.setToolTip("Paste two columns (time in s, then total CdA) from a spreadsheet or CSV with Ctrl+V / ⌘V.\n"
                              "Between points the CdA changes in a straight line; before the first and after the last\n"
                              "it holds their values.")
        self.table.itemChanged.connect(lambda _item: self.edited.emit())
        self.table.installEventFilter(self)
        self.unit = UnitCombo()
        self.unit.addItems(AREA_ITEMS)
        self.unit.setFixedWidth(UNIT_W)
        self.unit.setToolTip("Unit of the CdA column.")
        self._unit = self.unit.currentText()
        self.unit.currentIndexChanged.connect(self._on_unit)
        self._set_headers()

        add, remove, load = QPushButton("Add row"), QPushButton("Remove"), QPushButton("Import CSV…")
        add.clicked.connect(self._add_row)
        remove.clicked.connect(self._remove_rows)
        load.clicked.connect(self._import)
        load.setToolTip("A CSV of time (s) and total CdA. With more columns, the first one named CdA is used.")
        buttons = QHBoxLayout()
        buttons.setSpacing(FIELD_GAP)
        for button in (add, remove, load):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.unit)
        note = QLabel("For reproducing a fire that already happened: nobody knows ahead of time how an injector will change.")
        note.setObjectName("cardLabel")
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(FIELD_GAP)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(note)

    def points(self) -> list[list[float]]:
        """The rows with a time and a CdA of zero or more, in time order."""
        rows = []
        for r in range(self.table.rowCount()):
            try:
                t, cda = (float(self.table.item(r, c).text()) for c in (0, 1))
            except (AttributeError, ValueError):
                continue
            if cda >= 0:
                rows.append([t, cda])
        return sorted(rows)

    def set_points(self, points, unit: str):
        """Show a curve, without emitting an edit."""
        self.unit.blockSignals(True)
        self.unit.setCurrentText(unit)
        self.unit.blockSignals(False)
        self._unit = self.unit.currentText()
        self._set_headers()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._write(0, points)
        self.table.blockSignals(False)

    def _write(self, start: int, points):
        self.table.setRowCount(max(self.table.rowCount(), start + len(points)))
        for r, point in enumerate(points, start):
            for c, value in enumerate(point):
                self.table.setItem(r, c, QTableWidgetItem(f"{value:.10g}"))

    def _set_headers(self):
        self.table.setHorizontalHeaderLabels(["Time (s)", f"Total CdA ({pretty_unit(self._unit)})"])

    def _paste(self, points):
        """Pasted rows go in from the selected row down, as in a spreadsheet."""
        self.table.blockSignals(True)
        self._write(max(self.table.currentRow(), 0), points)
        self.table.blockSignals(False)
        self.edited.emit()

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.Type.KeyPress and event.matches(QKeySequence.StandardKey.Paste):
            if points := read_points(QApplication.clipboard().text()):
                self._paste(points)
                return True
        return super().eventFilter(obj, event)

    def _add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setCurrentCell(row, 0)
        self.table.setFocus()
        self.table.edit(self.table.model().index(row, 0))

    def _remove_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()} or {self.table.currentRow()}, reverse=True)
        for row in rows:
            if row >= 0:
                self.table.removeRow(row)
        self.edited.emit()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import CdA curve", "", "CSV (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            points = read_points(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError) as exc:
            QMessageBox.warning(self, "Import CSV", f"Couldn't read {Path(path).name}.\n\n{exc}")
            return
        if not points:
            QMessageBox.information(self, "Import CSV", f"{Path(path).name} has no rows of time and CdA.")
            return
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._write(0, points)
        self.table.blockSignals(False)
        self.edited.emit()

    def _on_unit(self, *_):
        """Convert the CdA column to the new unit, so the curve stays the same."""
        old, self._unit = self._unit, self.unit.currentText()
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            try:
                item.setText(f"{convert(float(item.text()), old, self._unit):.10g}")
            except (AttributeError, ValueError):
                continue
        self.table.blockSignals(False)
        self._set_headers()
        self.edited.emit()
