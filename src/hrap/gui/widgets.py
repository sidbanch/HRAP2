"""Widgets and spacing shared by every page: inputs, and cards of labelled rows."""
from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hrap.units import convert, from_si, pretty_unit, to_si

# One set of spacing for every page, in pixels.
PAGE_MARGIN = 16  # around a page
COLUMN_GAP = 16   # between columns of cards
CARD_GAP = 12     # between cards
CARD_PAD = 14, 12  # inside a card, across and down
ROW_GAP = 8       # between rows in a card
FIELD_GAP = 6     # between a field and its unit or checkbox
LABEL_W = 150     # the label column, so fields line up from card to card
UNIT_W = 72       # a UnitRow's unit dropdown


class _NoWheel:
    """Let the settings panel scroll instead of changing the focused control."""

    def wheelEvent(self, event):
        event.ignore()
        parent = cast(QWidget, self).parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                QApplication.sendEvent(parent.viewport(), event)
                return
            parent = parent.parentWidget()


class PlainSpinBox(_NoWheel, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(64)  # Qt otherwise reserves room for the largest allowed value


class PlainDoubleSpinBox(_NoWheel, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumWidth(64)  # Qt otherwise reserves room for the largest allowed value

    def textFromValue(self, value: float) -> str:
        """The shortest form: 750, 2, 0.198. It still holds its full number of decimals."""
        text = super().textFromValue(value)
        point = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(point) if point in text else text


class PlainComboBox(_NoWheel, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class UnitCombo(PlainComboBox):
    """A unit dropdown: it shows in² and holds the in^2 that motor files and conversions use."""

    def addItems(self, units: list[str]):
        for unit in units:
            self.addItem(pretty_unit(unit), unit)

    def currentText(self) -> str:
        return self.currentData() or ""

    def setCurrentText(self, unit: str):
        if (index := self.findData(unit)) >= 0:
            self.setCurrentIndex(index)


class ElidedLabel(QLabel):
    """A label that ends in … when it's too narrow for its text."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self._full = text
        self.setMinimumWidth(10)

    def set_full_text(self, text: str):
        self._full = text
        self.setText(self.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_full_text(self._full)


class HintList(QListWidget):
    """A list that says so in one grey line while it's empty."""

    def __init__(self, hint: str):
        super().__init__()
        self._hint = hint

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(self.viewport().rect().adjusted(8, 6, -8, -6), int(Qt.AlignmentFlag.AlignLeft), self._hint)


class UnitRow(QWidget):
    def __init__(self, items: list[str], unit: str, decimals: int = 4, maximum: float = 1e12):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(FIELD_GAP)
        self.spin = PlainDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(0.0, maximum)
        self.spin.setKeyboardTracking(False)
        # A value the app filled in (loaded, worked out, or converted to another unit) stays exact behind its
        # rounded display, so reading it back doesn't drift. Typing replaces it.
        self._exact: float | None = None
        self.spin.valueChanged.connect(self._on_typed)
        self.unit = UnitCombo()
        self.unit.addItems(items)
        if unit in items:
            self.unit.setCurrentText(unit)
        self.unit.setFixedWidth(UNIT_W)
        self._unit = self.unit.currentText()
        self.unit.currentIndexChanged.connect(lambda _: self._on_unit_changed(self.unit.currentText()))
        layout.addWidget(self.spin, 1)
        layout.addWidget(self.unit)

    def _on_unit_changed(self, new_unit: str):
        old = self._unit
        self._unit = new_unit
        if not old or old == new_unit:
            return
        try:
            new_val = convert(self.value(), old, new_unit)
        except Exception:
            return
        self.spin.blockSignals(True)
        self.spin.setValue(new_val)
        self.spin.blockSignals(False)
        self._exact = new_val

    def _on_typed(self, _value: float):
        self._exact = None

    def value(self) -> float:
        """The value in the row's unit."""
        return self.spin.value() if self._exact is None else self._exact

    def set_display(self, value: float, unit: str | None = None):
        if unit:
            self.unit.blockSignals(True)
            self.unit.setCurrentText(unit)
            self.unit.blockSignals(False)
            self._unit = self.unit.currentText()
        self.spin.blockSignals(True)
        self.spin.setValue(float(value or 0.0))
        self.spin.blockSignals(False)
        self._exact = float(value or 0.0)

    def si(self, quantity: str) -> float:
        return to_si(self.value(), self.unit.currentText(), quantity)

    def set_si(self, value: float, quantity: str):
        """Show an SI value in the row's current unit, without emitting a change."""
        self.set_display(from_si(value, self.unit.currentText(), quantity))


def sentence_case(text: str) -> str:
    """The engine's Title Case messages as the app writes them."""
    return text[:1] + text[1:].lower().replace("stl", "STL")


def beside(field: QWidget, box: QWidget) -> QWidget:
    """A field with a checkbox or button after it, on one row."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(FIELD_GAP)
    layout.addWidget(field, 1)
    layout.addWidget(box)
    return row


def page_layout(layout: QVBoxLayout | QHBoxLayout | QGridLayout):
    """A page's margins, with its cards a card gap apart."""
    layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    layout.setSpacing(CARD_GAP)
    return layout


def card_frame(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("sizingCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(CARD_PAD[0], CARD_PAD[1], CARD_PAD[0], CARD_PAD[1])
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("cardTitle")
    layout.addWidget(heading)
    return frame, layout


class FieldGrid(QGridLayout):
    """Label / field / unit rows with the same column widths in every card."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(FIELD_GAP)
        self.setVerticalSpacing(ROW_GAP)
        self.setColumnMinimumWidth(0, LABEL_W)
        self.setColumnMinimumWidth(2, UNIT_W)
        self.setColumnStretch(1, 1)
        self._rows = 0
        self.labels: list[QLabel] = []

    def add(self, label: str, field: QWidget, unit: str = "", muted: bool = False, span: bool = False) -> list[QWidget]:
        """Add a row and return its widgets, so the row can be hidden."""
        name = QLabel(label)
        if muted:
            name.setObjectName("cardLabel")
        self.labels.append(name)
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

    def add_line(self, widget: QWidget):
        """A line across the whole grid."""
        self.addWidget(widget, self._rows, 0, 1, 3)
        self._rows += 1


def line_up(*grids: FieldGrid):
    """One label column width for several cards' fields, so they line up down a page."""
    width = max(LABEL_W, *(label.sizeHint().width() for grid in grids for label in grid.labels))
    for grid in grids:
        grid.setColumnMinimumWidth(0, width)


class Card(QFrame):
    """A titled block of grey label / bold value rows, optionally with a sketch and the inputs that set them."""

    def __init__(self, title: str, rows: list[str], sketch: QWidget | None = None, inputs: QGridLayout | None = None,
                 sketch_over_results: bool = False):
        super().__init__()
        self.setObjectName("sizingCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(CARD_PAD[0], CARD_PAD[1], CARD_PAD[0], CARD_PAD[1])
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(ROW_GAP)
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
