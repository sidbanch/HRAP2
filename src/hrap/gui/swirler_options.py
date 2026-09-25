"""Swirler layouts that give the sized oxidizer flow, drilled with real number drills."""
from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
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
COLUMNS = ["Exit (in)", "Ports", "Drill (in)", "Swirl A", "Cd", "CdA (in²) vs target", "Oxidizer flow", "Start O/F", "Liquid lasts"]


@dataclass(frozen=True)
class Target:
    CdA: float         # m², total injector Cd × area for the sized flow
    mdot_o: float      # kg/s
    OF: float | None   # starting O/F, when the grain length is fixed so it follows the flow
    n: float           # regression exponent: O/F grows as oxidizer flow^(1 − n)
    ox_liquid: float   # kg
    swirlers: int
    R_in: float        # m, port offset from the swirler axis


@dataclass(frozen=True)
class Layout:
    exit_D: float  # m
    ports: int
    drill: int
    port_D: float  # m
    A: float
    cd: float
    CdA: float     # m², over all swirlers


def layouts(t: Target, exits: list[float], counts: range, min_drill: float) -> list[Layout]:
    """Each exit and port count, with the number drill nearest the exact port size for the flow."""
    found = []
    for exit_D in exits:
        cd_needed = t.CdA / (t.swirlers * 0.25 * math.pi * exit_D ** 2)
        for ports in counts:
            try:
                exact = swirl_port_D(exit_D, ports, t.R_in, cd_needed)
            except ValueError:
                continue
            drill, port_D = nearest_drill(exact)
            if port_D < min_drill or ports * port_D > 2.0 * math.pi * t.R_in or port_D > 2.0 * t.R_in:
                continue  # too small to drill, or the ports don't fit around the offset circle
            cd = swirl_cd(exit_D, ports, port_D, t.R_in)
            found.append(Layout(exit_D, ports, drill, port_D, swirl_A(exit_D, ports, port_D, t.R_in), cd,
                                cd * t.swirlers * 0.25 * math.pi * exit_D ** 2))
    return found


class SwirlerOptions(QFrame):
    """Table of drillable swirler layouts; clicking one loads it into the injector."""

    picked = Signal(float, int, float)  # exit diameter (m), port count, port diameter (m)

    def __init__(self):
        super().__init__()
        self.setObjectName("sizingCard")
        self._target: Target | None = None
        self._layouts: list[Layout] = []

        heading = QLabel("Swirler layouts")
        heading.setObjectName("cardTitle")
        intro = QLabel("Port sizes for the total CdA above at each exit and port count, rounded to the nearest number drill, "
                       "with the ports at the injector's port offset. The swirl theory has overpredicted measured swirlers, "
                       "so expect the drilled part to flow a little under this; open the ports up a drill size after a flow test. "
                       "Click a row to load it into the injector.")
        intro.setObjectName("cardLabel")
        intro.setWordWrap(True)

        self.exits = QLineEdit("0.188, 0.25")
        self.exits.setToolTip("Exit diameters to try, in inches, separated by commas.\n"
                              "The exit is the narrowest point after the swirler, e.g. 0.188 for a stock 1/4\" PTC.")
        self.ports_lo, self.ports_hi = PlainSpinBox(), PlainSpinBox()
        for spin, value in ((self.ports_lo, 3), (self.ports_hi, 8)):
            spin.setRange(1, 24)
            spin.setValue(value)
        self.min_drill = PlainDoubleSpinBox()
        self.min_drill.setDecimals(3)
        self.min_drill.setRange(0.0, 0.25)
        self.min_drill.setSingleStep(0.001)
        self.min_drill.setValue(0.030)
        self.min_drill.setToolTip("Smallest port you're willing to drill. Tiny drills break and tiny ports clog.")
        inputs = QHBoxLayout()
        for label, widget in (("Exits (in)", self.exits), ("Ports", self.ports_lo), ("to", self.ports_hi),
                              ("Smallest drill (in)", self.min_drill)):
            inputs.addWidget(QLabel(label))
            inputs.addWidget(widget)
        inputs.addStretch(1)

        self.note = QLabel("")
        self.note.setObjectName("cardLabel")
        self.note.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        self.table.setRowCount(0)
        t = self._target
        if t is None:
            return
        try:
            exits = [float(v) * IN for v in self.exits.text().replace(";", ",").split(",") if v.strip()]
        except ValueError:
            self.note.setText("Exits need to be numbers in inches, separated by commas.")
            return
        self._layouts = layouts(t, exits, range(self.ports_lo.value(), self.ports_hi.value() + 1), self.min_drill.value() * IN)
        self.note.setText(f"Target: total CdA {t.CdA / IN ** 2:.5f} in² over {t.swirlers} swirler{'' if t.swirlers == 1 else 's'}, "
                          f"ports {t.R_in / IN:.3f} in off the axis."
                          + ("" if self._layouts else " No layout fits: try other exits, more ports or a smaller drill."))
        self.table.setRowCount(len(self._layouts))
        for row, lay in enumerate(self._layouts):
            ratio = lay.CdA / t.CdA  # the flow is proportional to CdA
            mdot = t.mdot_o * ratio
            cells = [f"{lay.exit_D / IN:.3f}", str(lay.ports), f"#{lay.drill} ({lay.port_D / IN:.4f})", f"{lay.A:.2f}",
                     f"{lay.cd:.3f}", f"{lay.CdA / IN ** 2:.5f} ({round(100 * (ratio - 1)):+d}%)", f"{mdot:.3f} kg/s",
                     "—" if t.OF is None else f"{t.OF * ratio ** (1.0 - t.n):.2f}", f"{t.ox_liquid / mdot:.2f} s"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 4
                                  + sum(self.table.rowHeight(r) for r in range(self.table.rowCount())))

    def _pick(self, row: int, _col: int):
        lay = self._layouts[row]
        self.picked.emit(lay.exit_D, lay.ports, lay.port_D)
