"""Input widgets shared by the settings panel and the sizing page."""
from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hrap.units import convert


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


class PlainComboBox(_NoWheel, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class CollapsibleBox(QWidget):
    """Settings category that can be collapsed to just its title."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsibleBox")
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._btn = QPushButton(f"▾  {title}")
        self._btn.setObjectName("collapseHeader")
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setFlat(True)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._body = QWidget()
        self._body.setObjectName("collapseBody")
        self._form = QFormLayout(self._body)
        self._form.setContentsMargins(10, 6, 10, 10)
        outer.addWidget(self._btn)
        outer.addWidget(self._body)
        self._btn.toggled.connect(self._set_open)

    def form(self) -> QFormLayout:
        return self._form

    def _set_open(self, open_: bool) -> None:
        self._body.setVisible(open_)
        mark = "▾" if open_ else "▸"
        self._btn.setText(f"{mark}  {self._title}")


class UnitRow(QWidget):
    def __init__(self, items: list[str], unit: str, decimals: int = 4, maximum: float = 1e12):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.spin = PlainDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(0.0, maximum)
        self.spin.setKeyboardTracking(False)
        self.unit = PlainComboBox()
        self.unit.addItems(items)
        if unit in items:
            self.unit.setCurrentText(unit)
        self.unit.setFixedWidth(72)
        self._unit = self.unit.currentText()
        self.unit.currentTextChanged.connect(self._on_unit_changed)
        layout.addWidget(self.spin, 1)
        layout.addWidget(self.unit)

    def _on_unit_changed(self, new_unit: str):
        old = self._unit
        self._unit = new_unit
        if not old or old == new_unit:
            return
        try:
            new_val = convert(self.spin.value(), old, new_unit)
        except Exception:
            return
        self.spin.blockSignals(True)
        self.spin.setValue(new_val)
        self.spin.blockSignals(False)

    def set_display(self, value: float, unit: str | None = None):
        if unit:
            self.unit.blockSignals(True)
            self.unit.setCurrentText(unit)
            self.unit.blockSignals(False)
            self._unit = self.unit.currentText()
        self.spin.blockSignals(True)
        self.spin.setValue(float(value or 0.0))
        self.spin.blockSignals(False)
