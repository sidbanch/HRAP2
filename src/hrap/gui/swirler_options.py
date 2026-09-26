"""Swirler layouts that give the sized oxidizer flow, drilled with real number drills."""
from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from hrap.engine.swirl import nearest_drill, swirl_A, swirl_cd, swirl_port_D
from hrap.gui.widgets import PlainDoubleSpinBox, PlainSpinBox

IN = 0.0254
COLUMNS = ["Ports", "Drill", "Start O/F", "Flow vs target", "Liquid lasts"]
ON_TARGET = QColor("#3f9e62")  # flow within 3% of the target


@dataclass(frozen=True)
class Target:
    CdA: float         # m², total injector Cd × area for the sized flow
    mdot_o: float      # kg/s
    OF: float | None   # starting O/F, when the grain length is fixed so it follows the flow
    OF_exp: float      # with the grain length fixed, O/F grows as oxidizer flow^OF_exp
    ox_liquid: float   # kg
    swirlers: int
    R_in: float        # m, port offset from the swirler axis
    exit_D: float      # m, the injector's current exit, to mark the loaded layout
    ports: int         # the injector's current port count


@dataclass(frozen=True)
class Layout:
    exit_D: float  # m
    ports: int
    drill: int
    port_D: float  # m
    A: float
    cd: float
    CdA: float     # m², over all swirlers


def drilled_layout(t: Target, exit_D: float, ports: int) -> Layout:
    """The swirler with the number drill nearest the exact port size for the flow. Raises ValueError if none fits."""
    cd_needed = t.CdA / (t.swirlers * 0.25 * math.pi * exit_D ** 2)
    drill, port_D = nearest_drill(swirl_port_D(exit_D, ports, t.R_in, cd_needed))
    cd = swirl_cd(exit_D, ports, port_D, t.R_in)
    return Layout(exit_D, ports, drill, port_D, swirl_A(exit_D, ports, port_D, t.R_in), cd,
                  cd * t.swirlers * 0.25 * math.pi * exit_D ** 2)


def layouts(t: Target, exits: list[float], counts: range, min_drill: float) -> list[Layout]:
    found = []
    for exit_D in exits:
        for ports in counts:
            try:
                lay = drilled_layout(t, exit_D, ports)
            except ValueError:
                continue
            if lay.port_D >= min_drill and ports * lay.port_D <= 2.0 * math.pi * t.R_in:  # drillable, and the ports fit
                found.append(lay)
    return found


class SwirlerOptions(QFrame):
    """Drillable swirler layouts grouped by exit; clicking one loads its exit and port count into the injector."""

    picked = Signal(float, int)  # exit diameter (m), port count

    def __init__(self):
        super().__init__()
        self.setObjectName("sizingCard")
        self._target: Target | None = None
        self._rows: dict[int, Layout] = {}

        heading = QLabel("Swirler layouts")
        heading.setObjectName("cardTitle")
        intro = QLabel("Number-drill port sizes that give the target flow. Click a row to use it. The swirl theory has run "
                       "high on measured swirlers, so expect a little less flow and open the ports up a size after a flow test.")
        intro.setObjectName("cardLabel")
        intro.setWordWrap(True)

        self.exits = QLineEdit("0.188, 0.25")
        self.exits.setFixedWidth(120)
        self.exits.setToolTip("Exit diameters to try, in inches, separated by commas.\n"
                              "The exit is the narrowest point after the swirler, e.g. 0.188 for a stock 1/4\" PTC.")
        self.ports_lo, self.ports_hi = PlainSpinBox(), PlainSpinBox()
        for spin, value in ((self.ports_lo, 3), (self.ports_hi, 8)):
            spin.setRange(1, 24)
            spin.setValue(value)
            spin.setFixedWidth(56)
        self.min_drill = PlainDoubleSpinBox()
        self.min_drill.setDecimals(3)
        self.min_drill.setRange(0.0, 0.25)
        self.min_drill.setSingleStep(0.001)
        self.min_drill.setValue(0.030)
        self.min_drill.setFixedWidth(72)
        self.min_drill.setToolTip("Smallest port you're willing to drill. Tiny drills break and tiny ports clog.")
        inputs = QHBoxLayout()
        inputs.setSpacing(8)
        for label, widget in (("Exits (in)", self.exits), ("Ports", self.ports_lo), ("to", self.ports_hi),
                              ("Smallest drill (in)", self.min_drill)):
            inputs.addWidget(QLabel(label))
            inputs.addWidget(widget)
            inputs.addSpacing(8)
        inputs.addStretch(1)

        self.note = QLabel("")
        self.note.setObjectName("cardLabel")
        self.note.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.cellClicked.connect(self._pick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(intro)
        layout.addLayout(inputs)
        layout.addWidget(self.note)
        layout.addWidget(self.table)

        self.exits.editingFinished.connect(self._refresh)
        for spin in (self.ports_lo, self.ports_hi, self.min_drill):
            spin.valueChanged.connect(self._refresh)

    def update_target(self, target: Target | None):
        self._target = target
        self._refresh()

    def _refresh(self, *_):
        self.table.clearSpans()
        self.table.setRowCount(0)
        self._rows = {}
        t = self._target
        if t is None:
            return
        try:
            exits = [float(v) * IN for v in self.exits.text().replace(";", ",").split(",") if v.strip()]
        except ValueError:
            self.note.setText("Exits need to be numbers in inches, separated by commas.")
            return
        found = layouts(t, exits, range(self.ports_lo.value(), self.ports_hi.value() + 1), self.min_drill.value() * IN)
        self.note.setText(f"Target: {t.mdot_o:.3f} kg/s of oxidizer, total CdA {t.CdA / IN ** 2:.5f} in², "
                          f"ports {t.R_in / IN:.3f} in off the axis."
                          + ("" if found else " Nothing fits: try other exits, more ports or a smaller drill."))
        group_bg = self.table.palette().color(QPalette.ColorRole.AlternateBase)
        bold = QFont(self.table.font())
        bold.setBold(True)
        selected = None
        for exit_D in exits:
            group = [lay for lay in found if lay.exit_D == exit_D]
            if not group:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            cd_needed = t.CdA / (t.swirlers * 0.25 * math.pi * exit_D ** 2)
            header = QTableWidgetItem(f"{exit_D / IN:.3f} in exit  ·  needs Cd {cd_needed:.3f}  ·  swirl A ≈ {group[0].A:.1f}")
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header.setFont(bold)
            header.setBackground(group_bg)
            self.table.setItem(row, 0, header)
            self.table.setSpan(row, 0, 1, len(COLUMNS))
            for lay in group:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._rows[row] = lay
                ratio = lay.CdA / t.CdA  # the flow is proportional to CdA
                mdot = t.mdot_o * ratio
                error = round(100 * (ratio - 1))
                cells = [str(lay.ports), f"#{lay.drill}  ({lay.port_D / IN:.4f} in)",
                         "—" if t.OF is None else f"{t.OF * ratio ** t.OF_exp:.2f}",
                         "on target" if error == 0 else f"{error:+d}%", f"{t.ox_liquid / mdot:.2f} s"]
                tip = (f"{lay.ports} × #{lay.drill} ({lay.port_D / IN:.4f} in) ports in a {exit_D / IN:.3f} in exit\n"
                       f"swirl A {lay.A:.2f}, Cd {lay.cd:.3f}, total CdA {lay.CdA / IN ** 2:.5f} in², oxidizer flow {mdot:.3f} kg/s")
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setToolTip(tip)
                    if col == 3 and abs(ratio - 1) <= 0.03:
                        item.setForeground(ON_TARGET)
                    self.table.setItem(row, col, item)
                if abs(lay.exit_D - t.exit_D) < 1e-6 and lay.ports == t.ports:
                    selected = row
        if selected is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(selected)
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height() + 2
                                  + self.table.rowCount() * self.table.verticalHeader().defaultSectionSize())

    def _pick(self, row: int, _col: int):
        lay = self._rows.get(row)
        if lay is not None:
            self.picked.emit(lay.exit_D, lay.ports)
