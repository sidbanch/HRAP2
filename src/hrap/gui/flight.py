"""The Flight tab (experimental): fly a motor's runs on an OpenRocket rocket and compare apogee and the rest."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable

import pyqtgraph as pg
from PySide6.QtCore import QProcess, QSettings, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hrap.gui.simulation import DASHES, line_pixmap
from hrap.gui.study import CASE_COLORS
from hrap.gui.theme import colors_for
from hrap.gui.widgets import (
    CARD_GAP,
    COLUMN_GAP,
    ElidedLabel,
    FieldGrid,
    HintList,
    PlainComboBox,
    card_frame,
    page_layout,
)
from hrap.io import openrocket
from hrap.io.export import export_rse
from hrap.units import DisplayUnits

FOOT, G = 0.3048, 9.80665
COLUMNS = ("Apogee", "Max speed", "Max Mach", "Off the rail", "Stability off the rail", "Max acceleration", "Time to apogee")
TIPS = {
    "Off the rail": "How fast the rocket is going as it leaves the launch rail.",
    "Stability off the rail": "How far the center of pressure is behind the center of gravity as the rocket leaves the\n"
                              "rail, in body diameters (calibers). It includes that moment's gust.",
}
PLOTS = (("Altitude", "altitude"), ("Vertical speed", "speed"))


def _small(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("cardLabel")
    label.setWordWrap(True)
    return label


class FlightPage(QWidget):
    """Lists the shown motor's runs, flies the ticked ones in OpenRocket, and shows what each reached."""

    def __init__(self, prefs: QSettings, get_units: Callable[[], DisplayUnits], get_runs: Callable[[], list],
                 with_curves: Callable[..., None], varied_settings: Callable[[list[dict]], list[tuple[str, list[str]]]]):
        super().__init__()
        self._prefs, self._get_units, self._get_runs, self._with_curves = prefs, get_units, get_runs, with_curves
        self._varied_settings = varied_settings  # what differs between motors: each setting's name and values
        self._theme = "dark"
        self._openrocket = openrocket.find_openrocket(str(prefs.value("openRocketFolder") or "") or None)
        self._ork: openrocket.Ork | None = None
        self._ork_time = 0.0  # the file's modified time when it was read
        self._unticked: set[Path] = set()  # run files; a new run starts ticked
        self._flights: dict[Path, dict] = {}  # each flown run's reply from OpenRocket, by its file
        self._flying: list = []  # the runs being flown
        self._process: QProcess | None = None
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._output = b""
        self._stopped = False

        self.fly_btn = QPushButton("Fly")
        self.fly_btn.setObjectName("runButton")
        self.fly_btn.clicked.connect(self._fly_or_stop)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.status = _small()
        self.error = QLabel("")
        self.error.setObjectName("sizingError")
        self.error.setWordWrap(True)
        self.error.hide()

        rocket, rocket_l = card_frame("Rocket")
        title = rocket_l.itemAt(0).widget()
        rocket_l.removeWidget(title)
        head = QHBoxLayout()
        self.open_btn = QPushButton("Open .ork…")
        self.open_btn.clicked.connect(self._choose_ork)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.open_btn)
        rocket_l.addLayout(head)
        self.rocket_name, self.rocket_file = ElidedLabel("—"), ElidedLabel("—")
        self.rocket_name.setObjectName("cardValue")
        self.simulations = PlainComboBox()
        self.simulations.setToolTip("The launch rail, wind and launch site come from one of the rocket's own simulations.\n"
                                    "Set them up in OpenRocket. Every motor gets the same gusts, so flights compare fairly.")
        self.simulations.activated.connect(self._on_simulation)
        fields = FieldGrid()
        fields.add("Name", self.rocket_name, span=True)
        fields.add("File", self.rocket_file, span=True)
        fields.add("Launch as in", self.simulations)
        rocket_l.addLayout(fields)
        self.conditions = _small("Open the rocket's OpenRocket file. HRAP flies each run's motor on it in OpenRocket's own "
                                 "simulator, the same as exporting the run's RSE and flying it there.")
        rocket_l.addWidget(self.conditions)
        self.install = _small()
        self.find_btn = QPushButton("Find OpenRocket…")
        self.find_btn.clicked.connect(self._choose_openrocket)
        install_row = QHBoxLayout()
        install_row.addWidget(self.install, 1)
        install_row.addWidget(self.find_btn, 0, Qt.AlignmentFlag.AlignTop)
        rocket_l.addLayout(install_row)

        runs, runs_l = card_frame("Runs to fly")
        self.runs = HintList("Run this motor on the Simulation tab. Each run in its Runs list can be flown here.")
        self.runs.setMinimumHeight(90)
        self.runs.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.runs.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.runs.setToolTip("This motor's runs from the Simulation tab. Tick the ones to fly.")
        self.runs.itemChanged.connect(self._on_ticked)
        runs_l.addWidget(self.runs, 1)

        left = QWidget()
        left.setFixedWidth(400)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(CARD_GAP)
        for w in (self.fly_btn, self.progress, self.status, self.error, rocket):
            left_l.addWidget(w)
        left_l.addWidget(runs, 1)

        flights, flights_l = card_frame("Flights (experimental)")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet("QTableWidget { border: none; background: transparent; }")
        self.table.hide()
        flights_l.addWidget(self.table, 1)
        self.no_flights = _small("Open a rocket, tick the runs to fly, and press Fly.")
        self.no_flights.setAlignment(Qt.AlignmentFlag.AlignCenter)
        flights_l.addWidget(self.no_flights, 1)

        climb, climb_l = card_frame("Compare")
        title = climb_l.itemAt(0).widget()
        climb_l.removeWidget(title)
        head = QHBoxLayout()
        self.plotted = PlainComboBox()
        for name, key in PLOTS:
            self.plotted.addItem(name, key)
        self.plotted.activated.connect(lambda *_: self._show_flights())
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(QLabel("Plot"))
        head.addWidget(self.plotted)
        climb_l.addLayout(head)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(None)
        self.plot.setMinimumHeight(160)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        climb_l.addWidget(self.plot, 1)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(flights)
        split.addWidget(climb)
        split.setSizes([360, 520])
        root = page_layout(QHBoxLayout(self))
        root.setSpacing(COLUMN_GAP)
        root.addWidget(left)
        root.addWidget(split, 1)

        saved = str(prefs.value("flightOrk") or "")
        if saved and Path(saved).is_file():
            self._open_ork(saved, quiet=True)
        self._show_install()
        self.sync_runs()

    # What it flies ---------------------------------------------------------------------------------------------------

    def _choose_ork(self):
        start = str(self._ork.path.parent) if self._ork else str(self._prefs.value("lastFileDir") or "")
        path, _ = QFileDialog.getOpenFileName(self, "Open rocket", start, "OpenRocket (*.ork)")
        if path:
            self._open_ork(path)

    def _open_ork(self, path: str, quiet: bool = False):
        try:
            ork = openrocket.read_ork(path)
        except ValueError as exc:
            if not quiet:
                self._show_error(str(exc))
            return
        changed = self._ork is None or ork.path != self._ork.path
        if changed or ork.path.stat().st_mtime != self._ork_time:
            self._flights.clear()  # flown on another rocket, or on this one before it was saved again
        self._ork, self._ork_time = ork, ork.path.stat().st_mtime
        self._prefs.setValue("flightOrk", str(ork.path))
        self.rocket_name.set_full_text(ork.name)
        self.rocket_file.set_full_text(ork.path.name)
        self.rocket_file.setToolTip(str(ork.path))
        wanted = self.simulations.currentText() if not changed else str(self._prefs.value("flightSimulation") or "")
        self.simulations.clear()
        self.simulations.addItems([s.name for s in ork.simulations])
        self.simulations.setCurrentIndex(max(self.simulations.findText(wanted), 0))
        self.error.hide()
        self._show_conditions()
        self._show_flights()

    def _on_simulation(self, *_):
        self._prefs.setValue("flightSimulation", self.simulations.currentText())
        self._flights.clear()  # those were flown under the other simulation's conditions
        self._show_conditions()
        self._show_flights()

    def _show_conditions(self):
        ork, i = self._ork, self.simulations.currentIndex()
        if ork is None:
            return
        if not ork.simulations:
            self.conditions.setText(f"{ork.path.name} has no simulations. Add one in OpenRocket and save: it sets the launch "
                                    "rail, the wind and the launch site.")
            return
        s = ork.simulations[i]
        wind = "multi-level wind" if s.multi_level_wind else f"{self._speed(s.wind)} wind"
        text = f"{self._length(s.rod_length, 1)} rail, {s.rod_angle:g}° from vertical, {wind}."
        if s.apogee:
            text += (f" OpenRocket last saved an apogee of {self._length(s.apogee)}"
                     + (f" on its {s.motor} motor." if s.motor else "."))
        self.conditions.setText(text)

    def _choose_openrocket(self):
        start = "/Applications" if sys.platform == "darwin" else ""
        folder = QFileDialog.getExistingDirectory(self, "The folder OpenRocket is installed in", start)
        if not folder:
            return
        found = openrocket.find_openrocket(folder)
        if found is None:
            self._show_error(f"Couldn't find OpenRocket in {folder}. Pick the folder its installer made "
                             "(on a Mac, OpenRocket.app).")
            return
        self._openrocket = found
        self._prefs.setValue("openRocketFolder", folder)
        self.error.hide()
        self._show_install()

    def _problem(self) -> str:
        """Why nothing can be flown, in a sentence; empty when it can."""
        need = ".".join(str(n) for n in openrocket.OLDEST)
        if getattr(sys, "frozen", False):
            return "This build of HRAP can't fly rockets. Install HRAP with its installer script instead."
        if self._openrocket is None:
            return f"Flying needs OpenRocket {need} or newer installed. If it's somewhere unusual, point HRAP at it."
        if not self._openrocket.supported:
            return (f"Found OpenRocket {self._openrocket.version} in {self._openrocket.folder}, but flying needs "
                    f"{need} or newer. It opens older rocket files.")
        return ""

    def _show_install(self):
        problem = self._problem()
        self.install.setObjectName("notApplied" if problem else "cardLabel")
        self.install.style().unpolish(self.install)
        self.install.style().polish(self.install)
        self.install.setText(problem or f"Flown by OpenRocket {self._openrocket.version}.")
        self.find_btn.setVisible(bool(problem) and not getattr(sys, "frozen", False))

    def sync_runs(self):
        """List the shown motor's runs, newest first, keeping which are ticked."""
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in reversed(self._get_runs()):
            item = QListWidgetItem(run.label)
            item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Unchecked if run.file in self._unticked else Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, run.file)
            item.setToolTip("\n".join(run.changes))
            self.runs.addItem(item)
        self.runs.blockSignals(False)
        self._show_flights()

    def _on_ticked(self, item: QListWidgetItem):
        file = item.data(Qt.ItemDataRole.UserRole)
        (self._unticked.discard if item.checkState() == Qt.CheckState.Checked else self._unticked.add)(file)
        self._show_flights()

    def _ticked(self) -> list:
        return [r for r in reversed(self._get_runs()) if r.file not in self._unticked]

    def showEvent(self, event):
        super().showEvent(event)
        if (not self.busy() and self._ork is not None and self._ork.path.is_file()
                and self._ork.path.stat().st_mtime != self._ork_time):
            self._open_ork(str(self._ork.path), quiet=True)  # saved again in OpenRocket since
        self.sync_runs()

    # Flying ----------------------------------------------------------------------------------------------------------

    def busy(self) -> bool:
        return bool(self._flying)

    def _fly_or_stop(self):
        if self.busy():
            self.stop()
            return
        runs, problem = self._ticked(), self._problem()
        if not problem and self._ork is None:
            problem = "Open the rocket's .ork file first."
        elif not problem and not self._ork.simulations:
            problem = self.conditions.text()
        elif not problem and not runs:
            problem = "Tick at least one run to fly. Runs come from the Simulation tab."
        if problem:
            self._show_error(problem)
            return
        self.error.hide()
        self._flying, self._stopped = runs, False
        for run in runs:
            self._flights.pop(run.file, None)
        self._show_flights()
        for control in (self.open_btn, self.simulations, self.find_btn):
            control.setEnabled(False)
        self.fly_btn.setText("Stop")
        self.progress.setRange(0, 0)
        self.progress.show()
        waiting = [r for r in runs if r.output is None]  # read from their files: simulated again for their thrust
        self.status.setText(f"Simulating {len(waiting)} saved run{'s' if len(waiting) != 1 else ''} again…" if waiting
                            else "Starting OpenRocket…")
        for run in waiting:
            self._with_curves(run, lambda: self._launch_when_ready(runs), lambda: self._simulation_failed(runs))
        self._launch_when_ready(runs)

    def _simulation_failed(self, runs):
        if runs is self._flying and self._process is None:
            self._end("One of the saved runs couldn't be simulated again (see the status bar) or was deleted. "
                      "Untick it to fly the rest.")

    def _launch_when_ready(self, runs):
        if runs is not self._flying or self._process is not None or any(r.output is None for r in runs):
            return
        self._tmp = tempfile.TemporaryDirectory(prefix="hrap-flight-")
        files = []
        for i, run in enumerate(self._flying):
            files.append(Path(self._tmp.name) / f"{i}.rse")
            try:  # exactly the file Export RSE writes, so OpenRocket flies the same motor
                export_rse(files[-1], run.output, run.settings, OD=run.cfg.get("export_OD"), L=run.cfg.get("export_L"))
            except Exception as exc:  # the worker reports the missing file against this run
                self._flights[run.file] = {"error": f"Couldn't write its motor file: {exc}"}
        self.status.setText("Starting OpenRocket…")
        self._output = b""
        command = openrocket.worker_command()
        process = self._process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._on_process_error)
        process.start()
        process.write(openrocket.job(self._openrocket, self._ork.path, self.simulations.currentIndex(), files))
        process.closeWriteChannel()

    def _read(self):
        self._output += bytes(self._process.readAllStandardOutput().data())
        *lines, self._output = self._output.split(b"\n")
        for line in lines:
            reply = openrocket.reply(line.decode(errors="replace").strip())
            if reply is None:
                continue
            if "motor" not in reply:
                self._end(str(reply.get("error") or "OpenRocket couldn't fly the rocket."))
                return
            run = self._flying[int(reply["motor"])]
            if "error" not in self._flights.get(run.file, {}) or "error" not in reply:  # keep why its file wasn't written
                self._flights[run.file] = reply
            done = sum(r.file in self._flights for r in self._flying)
            self.progress.setRange(0, len(self._flying))
            self.progress.setValue(done)
            self.status.setText(f"Flew {done} of {len(self._flying)}…")
            self._show_flights()

    def _on_process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._end(f"Couldn't start the flight process ({sys.executable}).")

    def _finished(self, _code, _status):
        if not self.busy():
            return
        missing = [r for r in self._flying if r.file not in self._flights]
        if missing and not self._stopped:
            tail = bytes(self._process.readAllStandardError().data()).decode(errors="replace").strip().splitlines()
            self._end("OpenRocket stopped before every run was flown." + (f" It said: {tail[-1][:300]}" if tail else ""))
        else:
            n = len(self._flying) - len(missing)
            self._end(status="Stopped." if self._stopped else f"Flew {n} run{'s' if n != 1 else ''}.")

    def stop(self):
        """Stop flying. Runs already flown keep their results."""
        if not self.busy():
            return
        self._stopped = True
        if self._process is not None:
            self._process.kill()
            self._process.waitForFinished(2000)
        if self.busy():  # still waiting on saved runs being simulated again
            self._end(status="Stopped.")

    def _end(self, error: str = "", status: str = ""):
        process, self._process, self._flying = self._process, None, []
        if process is not None:
            process.blockSignals(True)
            process.kill()
            process.waitForFinished(2000)
            process.deleteLater()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        self.fly_btn.setText("Fly")
        for control in (self.open_btn, self.simulations, self.find_btn):
            control.setEnabled(True)
        self.progress.hide()
        self.status.setText(status)
        if error:
            self._show_error(error)
        self._show_flights()

    def _show_error(self, text: str):
        self.error.setText(text)
        self.error.show()

    # Results ---------------------------------------------------------------------------------------------------------

    def _imperial(self) -> bool:
        return self._get_units().length in ("in", "ft")

    def _length(self, metres: float, decimals: int = 0) -> str:
        return f"{metres / FOOT:,.{decimals}f} ft" if self._imperial() else f"{metres:,.{decimals}f} m"

    def _speed(self, value: float) -> str:
        return f"{value / FOOT:,.0f} ft/s" if self._imperial() else f"{value:,.1f} m/s"

    def set_theme(self, name: str):
        self._theme = name
        self._show_flights()

    def refresh(self):
        """Show everything again in the display units."""
        self._show_conditions()
        self._show_flights()

    def _show_flights(self):
        """The ticked runs in the table, and the flown ones on the plot. A run's line is the swatch beside its name.
        Up to three settings that differ between the runs get a column each; when it's one number, apogee can be
        plotted against it."""
        colors = colors_for(self._theme)
        runs = self._ticked()
        varied = self._varied_settings([r.cfg for r in runs]) if len(runs) > 1 else []
        varied = varied if len(varied) <= 3 else []
        try:
            swept = [float(v.split()[0]) for v in varied[0][1]] if len(varied) == 1 else None
        except (ValueError, IndexError):
            swept = None
        against = f"Apogee against {varied[0][0]}" if swept else ""
        if self.plotted.count() > len(PLOTS) and self.plotted.itemText(len(PLOTS)) != against:
            self.plotted.removeItem(len(PLOTS))
        if against and self.plotted.count() == len(PLOTS):
            self.plotted.addItem(against, "apogee")

        headers = ["Run", *(name[:1].upper() + name[1:] for name, _ in varied), *COLUMNS]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, name in enumerate(headers):
            header = self.table.horizontalHeaderItem(column)
            header.setToolTip(TIPS.get(name, ""))
            header.setTextAlignment((Qt.AlignmentFlag.AlignRight if column else Qt.AlignmentFlag.AlignLeft)
                                    | Qt.AlignmentFlag.AlignVCenter)
        self.table.setRowCount(len(runs))
        self.plot.clear()
        self.plot.enableAutoRange()
        key = self.plotted.currentData()
        scale = 1 / FOOT if self._imperial() else 1.0
        unit = "ft" if self._imperial() else "m"
        if key == "apogee":
            what = " ".join(varied[0][1][0].split()[1:])
            self.plot.setLabel("bottom", headers[1] + (f" ({what})" if what else ""))
            self.plot.setLabel("left", f"Apogee ({unit})")
            flown = sorted((x, self._flights[r.file]["apogee"] * scale) for x, r in zip(swept, runs)
                           if self._flights.get(r.file, {}).get("apogee") is not None)
            self.plot.plot([x for x, _ in flown], [y for _, y in flown], pen=pg.mkPen(colors["grey"], width=2))
        else:
            self.plot.setLabel("bottom", "Time (s)")
            self.plot.setLabel("left", f"{self.plotted.currentText()} ({unit}{'/s' if key == 'speed' else ''})")
        shows = (("apogee", self._length), ("max_speed", self._speed), ("max_mach", lambda v: f"{v:.2f}"),
                 ("rail_speed", self._speed), ("stability", lambda v: f"{v:.2f} cal"),
                 ("max_accel", lambda v: f"{v / G:.1f} g"), ("time_to_apogee", lambda v: f"{v:.1f} s"))
        for row, run in enumerate(runs):
            flight = self._flights.get(run.file)
            color, dash = colors[CASE_COLORS[row % len(CASE_COLORS)]], ([Qt.PenStyle.SolidLine, *DASHES])[
                row // len(CASE_COLORS) % (len(DASHES) + 1)]
            name = QTableWidgetItem(run.label)
            name.setToolTip(run.label)
            cells = ["—"] * len(COLUMNS)
            if flight is not None and "error" in flight:
                cells[0] = "Couldn't fly"
                name.setToolTip(str(flight["error"]))
            elif flight is not None:
                cells = [show(flight[k]) if flight.get(k) is not None else "—" for k, show in shows]
                name.setIcon(QIcon(line_pixmap(color, dash)))
                if flight.get("warnings"):
                    name.setText(f"{run.label} ⚠")
                    name.setToolTip("OpenRocket warned:\n" + "\n".join(flight["warnings"]))
                if key != "apogee":
                    self.plot.plot(flight["t"], [v * scale for v in flight[key]], pen=pg.mkPen(color, width=2, style=dash))
                elif flight.get("apogee") is not None:
                    self.plot.plot([swept[row]], [flight["apogee"] * scale], pen=None, symbol="o", symbolSize=10,
                                   symbolBrush=color, symbolPen=None)
            self.table.setItem(row, 0, name)
            for column, cell in enumerate([*(values[row] for _, values in varied), *cells], 1):
                item = QTableWidgetItem(cell)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(name.toolTip() if cell == "Couldn't fly" else "")
                self.table.setItem(row, column, item)
        self.table.setVisible(bool(runs))
        self.no_flights.setVisible(not runs)
