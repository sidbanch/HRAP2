"""Grain tab: an STL fuel grain in 3D, burned back by a chosen distance or to a time in the last run."""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hrap.gui.sizing import Card
from hrap.units import DisplayUnits

STEPS = 1000  # slider positions
PLAY_S = 6.0  # a burned-distance sweep plays over this long; a run plays in its own time
PREVIEW_CELLS = 5e5  # grid cells in the quick surface shown while the slider moves
THEMES = {
    "dark": {"bg": "#1a1d23", "fg": "#e6e8ee", "muted": "#9aa3b0", "fuel": (0.82, 0.86, 0.93, 0.16),
             "preview": (0.82, 0.86, 0.93, 0.5),
             "burned": "#f08a3c", "case": "#e06c75", "port_area": "#5b8def"},
    "light": {"bg": "#ffffff", "fg": "#1b1d21", "muted": "#5c6370", "fuel": (0.25, 0.30, 0.38, 0.16),
              "preview": (0.25, 0.30, 0.38, 0.5),
              "burned": "#e07020", "case": "#c8414c", "port_area": "#2f5fbf"},
}

_VERTEX = """
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec4 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_color = a_color;
        gl_Position = u_mvp * a_position;
    }
"""
# A key and a fill light that turn with the camera, lighting both sides of each face.
_SOLID = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        vec3 n = normalize(v_normal);
        if (n.z < 0.0) n = -n;
        vec3 key = normalize(vec3(-0.5, 0.7, 0.6));
        float light = 0.7 * max(dot(n, key), 0.0) + 0.25 * max(dot(n, normalize(vec3(0.6, -0.3, 0.4))), 0.0);
        float shine = pow(max(dot(n, normalize(key + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
        vec3 rgb = v_color.rgb * (0.3 + light) + 0.2 * shine + 0.25 * pow(1.0 - n.z, 3.0) * v_color.rgb;
        gl_FragColor = vec4(rgb, v_color.a);
    }
"""
# See-through, more solid where the surface turns away from the camera, so the outline and slots stand out.
_GLASS = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        float edge = pow(1.0 - abs(normalize(v_normal).z), 2.0);
        gl_FragColor = vec4(v_color.rgb * (0.6 + 0.4 * edge), min(v_color.a * (0.3 + 3.0 * edge), 1.0));
    }
"""


def _mesh_arrays(vertices: np.ndarray, faces: np.ndarray, smoothing: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertices turned so the grain's axis runs along the view's x, its faces, and vertex normals,
    averaged with their neighbours' `smoothing` times so the grid's steps don't show in the shading."""
    v = np.ascontiguousarray(vertices[:, [2, 0, 1]], dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)

    def gather(per_face):  # each vertex's sum over the faces around it
        return np.stack([sum(np.bincount(f[:, k], per_face[:, i], len(v)) for k in range(3)) for i in range(3)], axis=1)

    n = gather(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]))
    for _ in range(smoothing):
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        n = gather(n[f].sum(axis=1))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return v, f, n.astype(np.float32)


def _burned_arrays(geom, web: float, step: int, colors: dict) -> tuple:
    """The burned region's surface: the burning fuel, where it has burned through to the case, and the open ends."""
    from hrap.engine.grain_stl import port_surface

    vertices, faces = port_surface(geom, web, step=step)
    v, f, n = _mesh_arrays(vertices, faces, smoothing=2 if step == 1 else 1)
    # Each vertex sits on a grid edge; it's on the case when a cell at either end of that edge is outside the grain.
    grid = geom.distance[::step, ::step, ::step]
    g = (vertices - geom.origin) / (geom.spacing * step)
    at_case = np.zeros(len(g), dtype=bool)
    for ends in (np.floor(g + 1e-3), np.ceil(g - 1e-3)):
        i = ends.astype(int)
        inside = np.all((i >= 0) & (i < grid.shape), axis=1)
        at_case[inside] |= np.isinf(grid[i[inside, 0], i[inside, 1], i[inside, 2]])
    at_end = (g[:, 2] < 0) | (g[:, 2] > grid.shape[2] - 1)
    burning = np.array(pg.mkColor(colors["burned"]).getRgbF())
    rgba = np.where(at_case[:, None], pg.mkColor(colors["case"]).getRgbF(), burning)
    rgba[at_end] = burning * (0.5, 0.5, 0.5, 1)
    return v, f, n, rgba.astype(np.float32)


class _Surfaces(QObject):
    """Works out full-detail surfaces off the UI thread."""
    done = Signal(int, object)  # job number, mesh arrays

    def start(self, job: int, geom, web: float, colors: dict):
        def work():
            try:
                self.done.emit(job, _burned_arrays(geom, web, 1, colors))
            except Exception:
                pass  # the quick surface stays on screen
        threading.Thread(target=work, daemon=True).start()


class GrainView(QWidget):
    """The grain in 3D with its burned-back region, its burning and port areas, and a slider to burn it back.
    While a grain loads it shows its shape and the load's progress; without one, why not."""
    cancel_load = Signal()
    message_action = Signal()  # the button under a message was clicked

    def __init__(self, get_units: Callable[[], DisplayUnits], get_density: Callable[[], float]):
        super().__init__()
        self._get_units, self._get_density = get_units, get_density
        self._geom = None
        self._run = None  # the last run's output, when it burned this grain
        self._colors = THEMES["dark"]
        self._view = None  # the 3D view, made when the first grain loads
        self._job = 0  # the newest full-detail surface asked for
        self._play_start = (0.0, 0)  # when Play was pressed, and the slider then
        self._stale = False  # the 3D view is behind the slider, as it doesn't draw while hidden
        self._length = 0.0  # of the grain on screen, to frame it
        self._surfaces = _Surfaces()
        self._surfaces.done.connect(self._on_surface)

        self._message = QLabel("")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setFixedWidth(520)
        self._message_button = QPushButton()
        self._message_button.clicked.connect(self.message_action)
        waiting = QWidget()
        wl = QVBoxLayout(waiting)
        wl.setSpacing(14)
        wl.addStretch(1)
        wl.addWidget(self._message, 0, Qt.AlignmentFlag.AlignHCenter)
        wl.addWidget(self._message_button, 0, Qt.AlignmentFlag.AlignHCenter)
        wl.addStretch(1)
        self._stack = QStackedWidget()
        self._stack.addWidget(waiting)
        frame = QFrame()
        frame.setObjectName("motorPanel")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self._stack)

        self.play = QPushButton("▶  Play")
        self.play.setCheckable(True)
        self.play.setFixedWidth(96)
        self.play.setToolTip("Burn the grain back from the start.")
        self.play.toggled.connect(self._on_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self._show_full)
        self.position = QLabel("")
        self.position.setMinimumWidth(150)
        self.position.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.by_web = QPushButton("Burned distance")
        self.by_time = QPushButton("Burn time")
        self.by_web.setToolTip("Slide through how far the fuel has burned back from the starting port.")
        self.by_time.setToolTip("Slide through the last run's burn, with the burn-back it worked out at each time.")
        mode = QButtonGroup(self)
        for button in (self.by_web, self.by_time):
            button.setCheckable(True)
            mode.addButton(button)
        self.by_web.setChecked(True)
        self.by_time.toggled.connect(self._on_mode)
        self.legend = QLabel("")
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self._reset_camera())
        move = "⌘-drag" if sys.platform == "darwin" else "Ctrl-drag"
        hint = QLabel(f"Drag to turn, scroll to zoom, {move} to move.")
        hint.setObjectName("cardLabel")

        bar = QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(self.play)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.position)
        options = QHBoxLayout()
        options.setSpacing(6)
        options.addWidget(QLabel("Slide by"))
        options.addWidget(self.by_web)
        options.addWidget(self.by_time)
        options.addSpacing(16)
        options.addWidget(self.legend)
        options.addStretch(1)
        options.addWidget(hint)
        options.addSpacing(8)
        options.addWidget(reset)
        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)
        cl.addLayout(bar)
        cl.addLayout(options)
        self._progress_text = QLabel("")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(280)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_load)
        loading = QWidget()
        ll = QHBoxLayout(loading)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(12)
        ll.addWidget(self._progress_text, 1)
        ll.addWidget(self._progress_bar)
        ll.addWidget(cancel)
        self._bottom = QStackedWidget()  # the slider and its options, or the load's progress
        self._bottom.addWidget(controls)
        self._bottom.addWidget(loading)
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(frame, 1)
        left.addWidget(self._bottom)

        self.grain = Card("Grain", ["Length", "Outer diameter", "Starting port volume", "Same-area round port", "Fuel",
                                    "Case wall reached", "Fuel left then", "Last fuel gone at", "Grid"])
        self.now = Card("At the slider", ["Time", "Burned distance", "Burning area", "Exposed case wall", "Port area",
                                          "Fuel left", "O/F"])
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setMinimumHeight(300)
        self._area_plots = (self.plots.addPlot(row=0, col=0), self.plots.addPlot(row=1, col=0))
        self._markers = []
        self._case_lines = []
        self._area_plots[0].addLegend(offset=(-6, 6))
        for plot in self._area_plots:
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("left").setWidth(60)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            marker = pg.InfiniteLine(angle=90, movable=True)
            marker.setToolTip("Drag to burn the grain back to here.")
            marker.sigDragged.connect(self._on_marker)
            plot.addItem(marker, ignoreBounds=True)
            self._markers.append(marker)
            case_line = pg.InfiniteLine(angle=90, movable=False, label="Case wall reached" if not self._case_lines else None,
                                        labelOpts={"position": 0.12, "anchors": [(-0.04, 1), (-0.04, 1)]})
            plot.addItem(case_line, ignoreBounds=True)
            self._case_lines.append(case_line)
        self._area_plots[1].setXLink(self._area_plots[0])
        self._case_label = self._case_lines[0].label
        self._area_plots[0].getAxis("bottom").setStyle(showValues=False)
        right_column = self._right = QWidget()
        right_column.setFixedWidth(370)
        right = QVBoxLayout(right_column)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        right.addWidget(self.grain)
        self.grain_warning = QLabel("")  # the cubes are too coarse for the grain's thinnest walls
        self.grain_warning.setObjectName("notApplied")
        self.grain_warning.setWordWrap(True)
        self.grain_warning.hide()
        right.addWidget(self.grain_warning)
        right.addWidget(self.now)
        right.addWidget(self.plots, 1)

        page = QHBoxLayout(self)
        page.setContentsMargins(16, 16, 16, 16)
        page.setSpacing(16)
        page.addLayout(left, 1)
        page.addWidget(right_column)

        self._preview_timer = QTimer(self, singleShot=True, interval=0)
        self._preview_timer.timeout.connect(self._show_preview)
        self._settle_timer = QTimer(self, singleShot=True, interval=200)
        self._settle_timer.timeout.connect(self._show_full)
        self._play_timer = QTimer(self, interval=33)
        self._play_timer.timeout.connect(self._play_step)
        self.set_theme("dark")
        self.set_message("")

    def set_grain(self, geom):
        """Show a loaded grain, burned back from its start."""
        self._bottom.setCurrentIndex(0)
        self._bottom.show()
        for w in (self._right, self.now, self.plots):
            w.show()
        if geom is self._geom:
            return
        self._geom = geom
        self.play.setChecked(False)
        self._sync_mode()
        if self._view is None:
            self._make_view()
        self._stack.setCurrentIndex(1)
        self._fuel.setColor(self._colors["fuel"])
        self._burned.show()
        self._set_mesh(self._fuel, *_mesh_arrays(geom.mesh_vertices, geom.mesh_faces))
        self._reset_camera(geom.length)
        for name in self.grain.values:
            self.grain.show_row(name, True)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.refresh()
        self._show_preview()
        self._show_full()

    def set_loading(self, name: str, mesh: tuple | None):
        """While a grain loads: its shape as soon as the STL is read, and the load's progress below."""
        self._drop_grain()
        self._bottom.setCurrentIndex(1)
        self._bottom.show()
        if mesh is None:
            self._show_message(f"Loading {name}…")
            return
        vertices, faces = mesh
        if self._view is None:
            self._make_view()
        self._stack.setCurrentIndex(1)
        self._burned.hide()
        self._fuel.setColor(self._colors["preview"])
        self._set_mesh(self._fuel, *_mesh_arrays(vertices, faces))
        size = np.ptp(vertices, axis=0)
        self._reset_camera(float(size[2]))
        u = self._get_units()
        self._right.show()
        for w in (self.now, self.plots, self.grain_warning):
            w.hide()
        for row in self.grain.values:
            self.grain.show_row(row, row in ("Length", "Outer diameter"))
        self.grain.set("Length", u.text(float(size[2]), "length"))
        self.grain.set("Outer diameter", u.text(float(max(size[:2])), "length"))

    def set_progress(self, text: str, fraction: float):
        self._progress_text.setText(text)
        self._progress_bar.setValue(round(1000 * fraction))

    def set_message(self, text: str, action: str = ""):
        """No grain to show: why, and a button for the one thing to do about it."""
        self._drop_grain()
        self._bottom.hide()
        self._show_message(text, action)

    def _drop_grain(self):
        self._geom = None
        self.play.setChecked(False)
        self._sync_mode()
        self._right.hide()

    def _show_message(self, text: str, action: str = ""):
        self._stack.setCurrentIndex(0)
        self._message.setText(text)
        self._message.setFixedHeight(max(0, self._message.heightForWidth(self._message.width())))  # wrapped lines need room
        self._message_button.setText(action)
        self._message_button.setVisible(bool(action))

    def set_run(self, output):
        """The motor's current run, whose burn time the slider can follow; None when there isn't one."""
        web = getattr(output, "grn_web", None)
        self._run = output if web is not None and len(web) > 1 and float(np.max(web)) > 0 else None  # a round grain's is 0
        self._sync_mode()
        self.refresh()

    def _sync_mode(self):
        """Follow the run's burn time when there is one for this grain, otherwise slide by burned distance."""
        timed = self._run is not None and self._geom is not None
        self.by_time.setEnabled(timed)
        (self.by_time if timed else self.by_web).setChecked(True)

    def set_theme(self, name: str):
        self._colors = THEMES["light" if name == "light" else "dark"]
        if self._view is not None:
            self._view.setBackgroundColor(self._colors["bg"])
            self._fuel.setColor(self._colors["fuel" if self._geom is not None else "preview"])
        self.plots.setBackground(self._colors["bg"])
        self.legend.setText(f'<span style="color:{self._colors["burned"]}">■</span> Burning surface&nbsp;&nbsp;&nbsp;'
                            f'<span style="color:{self._colors["case"]}">■</span> Exposed case wall')
        for plot in self._area_plots:
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(self._colors["fg"])
                plot.getAxis(side).setTextPen(self._colors["fg"])
        for marker in self._markers:
            marker.setPen(pg.mkPen(self._colors["fg"], width=1, style=Qt.PenStyle.DashLine))
            marker.setHoverPen(pg.mkPen(self._colors["burned"], width=2))
        for line in self._case_lines:
            line.setPen(pg.mkPen(self._colors["case"], width=1.5, style=Qt.PenStyle.DashLine))
        self._case_label.setColor(self._colors["case"])
        for card, name in ((self.grain, "Case wall reached"), (self.now, "Exposed case wall")):
            card.values[name].setStyleSheet(f"color: {self._colors['case']}")
        self.refresh()
        if self._geom is not None:
            self._show_preview()
            self._show_full()

    def refresh(self):
        """Redraw the plots and numbers, for new units, a new run or a new theme."""
        g = self._geom
        if g is None:
            return
        u = self._get_units()
        rho = self._get_density()
        fuel = u.text(g.fuel_volume[0], "volume") + (f", {u.text(rho * g.fuel_volume[0], 'mass')}" if rho > 0 else "")
        self.grain.set("Length", u.text(g.length, "length"))
        self.grain.set("Outer diameter", u.text(g.outer_diameter, "length"))
        self.grain.set("Starting port volume", u.text(g.port_volume[0], "volume"), "The open volume inside the grain before it burns.")
        self.grain.set("Same-area round port", u.text(2 * np.sqrt(g.port_area[0] / np.pi), "length"),
                       "The round port with the same area as this one, averaged along the grain\n"
                       "(starting port volume ÷ length). The Motor tab uses it for its drawings.")
        self.grain.set("Fuel", fuel)
        reached = bool(np.isfinite(g.case_web))
        self.grain.show_row("Fuel left then", reached)
        if reached:
            when = f"{u.text(g.case_web, 'length', 3)} burned"
            t = self._run_time_at(g.case_web)
            left = float(np.interp(g.case_web, g.web, g.fuel_volume))
            self.grain.set("Case wall reached", when if t is None else f"{t:.2f} s, {when}",
                           "Where the burn first breaks through the fuel to the case wall. From here on the hot gas\n"
                           "touches the case, and the fuel left is in slivers that may break off.")
            self.grain.set("Fuel left then", self._fuel_text(left, rho, g))
        else:
            self.grain.set("Case wall reached", "never", "The fuel is gone before the burn reaches the case wall.")
        self.grain.set("Last fuel gone at", f"{u.text(g.web[-1], 'length', 3)} burned",
                       "How far the fuel burns back before none is left: the thickest part of the grain.")
        self.grain.set("Grid", f"{g.distance.size / 1e6:.1f} million cubes of {u.text(g.spacing, 'length', 3)}",
                       "The grain is split into cubes to work out how it burns back.\n"
                       "Set their size with Cube size on the Motor tab.")
        self.grain_warning.setText(g.warning)
        self.grain_warning.setVisible(bool(g.warning))
        x = u.value(g.web, "length")
        top, bottom = self._area_plots
        for plot in self._area_plots:
            for item in plot.listDataItems():
                plot.removeItem(item)
        top.legend.clear()
        top.plot(x, u.value(g.burn_area, "area"), pen=pg.mkPen(self._colors["burned"], width=2), name="Burning area")
        top.plot(x, u.value(g.case_area, "area"), pen=pg.mkPen(self._colors["case"], width=2), name="Exposed case wall")
        bottom.plot(x, u.value(g.port_area, "area"), pen=pg.mkPen(self._colors["port_area"], width=2))
        for plot, name in ((top, "Area"), (bottom, "Port area")):
            plot.setLabel("left", name, units=u.unit("area"), color=self._colors["fg"])
            plot.setXRange(0, float(x[-1]), padding=0)
        for line in self._case_lines:
            line.setVisible(reached)
            line.setValue(u.value(g.case_web, "length") if reached else 0.0)
        self._area_plots[1].setLabel("bottom", "Burned distance", units=u.unit("length"), color=self._colors["fg"])
        self._update_numbers()

    def _fuel_text(self, volume: float, rho: float, g) -> str:
        u = self._get_units()
        return (u.text(rho * volume, "mass") if rho > 0 else u.text(volume, "volume")) + f" ({100 * volume / g.fuel_volume[0]:.0f}%)"

    def _run_time_at(self, d: float) -> float | None:
        """When the run's burn reached a burned distance, if there's a run and it did."""
        if not self._time_mode():
            return None
        reached = np.flatnonzero(np.asarray(self._run.grn_web) >= d)
        return float(self._run.t[reached[0]]) if reached.size else None

    def follow_time(self, t: float):
        """Show the run at a time, as hovered on the Simulation tab's plots."""
        if self._time_mode():
            self.slider.setValue(round(float(np.clip(t / self._run.t[-1], 0, 1)) * STEPS))

    def showEvent(self, event):
        super().showEvent(event)
        if self._stale:
            self._show_preview()
            self._show_full()

    def _make_view(self):
        """The 3D view, made on first use so the app starts without OpenGL."""
        import pyqtgraph.opengl as gl
        from OpenGL import GL
        from pyqtgraph.opengl import shaders

        class Glass(gl.GLMeshItem):
            """See-through, without hiding its own far side."""
            def paint(self):
                GL.glDepthMask(GL.GL_FALSE)
                super().paint()
                GL.glDepthMask(GL.GL_TRUE)

        def shader(name, fragment):
            return shaders.ShaderProgram(name, [shaders.VertexShader(_VERTEX), shaders.FragmentShader(fragment)])

        self._gl = gl
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor(self._colors["bg"])
        self._burned = gl.GLMeshItem(shader=shader("hrapGrainSolid", _SOLID), smooth=True, glOptions="opaque")
        self._fuel = Glass(shader=shader("hrapGrainGlass", _GLASS), smooth=True, glOptions="translucent",
                           color=self._colors["fuel"])
        self._fuel.setDepthValue(1)  # drawn after the burned region, which shows through it
        self._view.addItem(self._burned)
        self._view.addItem(self._fuel)
        self._stack.addWidget(self._view)

    def _set_mesh(self, item, v, f, n, colors=None):
        md = self._gl.MeshData(vertexes=v, faces=f, vertexColors=colors)
        md._vertexNormals = n  # pyqtgraph works these out one vertex at a time in Python, far too slowly for these meshes
        item.setMeshData(meshdata=md)

    def _reset_camera(self, length: float | None = None):
        self._length = length or self._length
        if self._view is not None:
            self._view.setCameraParams(center=pg.Vector(0, 0, 0), distance=2.1 * self._length, elevation=20,
                                       azimuth=-68, fov=30)

    def _time_mode(self) -> bool:
        return self._run is not None and self._geom is not None and self.by_time.isChecked()

    def _web(self) -> float:
        """The burned distance at the slider."""
        f = self.slider.value() / STEPS
        if self._time_mode():
            o = self._run
            return float(np.interp(f * o.t[-1], o.t, o.grn_web))
        return f * float(self._geom.web[-1])

    def _on_mode(self, *_):
        """Keep the burn-back where it is when the slider changes between distance and time."""
        o = self._run
        if self._geom is None or o is None:
            return
        d = float(np.interp(self.slider.value() / STEPS * o.t[-1], o.t, o.grn_web)) if not self.by_time.isChecked() \
            else self.slider.value() / STEPS * float(self._geom.web[-1])
        self._set_web(d)
        self._update_numbers()

    def _set_web(self, d: float):
        """Move the slider to a burned distance."""
        if self._time_mode():
            o = self._run
            t = self._run_time_at(d)
            value = (float(o.t[-1]) if t is None else t) / float(o.t[-1])
        else:
            value = d / float(self._geom.web[-1])
        self.slider.setValue(round(float(np.clip(value, 0, 1)) * STEPS))

    def _on_slider(self, *_):
        self._update_numbers()
        self._preview_timer.start()
        if not self.play.isChecked():
            self._settle_timer.start()

    def _on_marker(self, marker):
        self._set_web(marker.value() / self._get_units().value(1.0, "length"))

    def _update_numbers(self):
        g = self._geom
        if g is None:
            return
        u = self._get_units()
        d = self._web()
        timed = self._time_mode()
        for marker in self._markers:
            marker.blockSignals(True)
            marker.setValue(u.value(d, "length"))
            marker.blockSignals(False)
        port = float(np.interp(d, g.web, g.port_area))
        case = float(np.interp(d, g.web, g.case_area))
        self.now.set("Burned distance", u.text(d, "length"))
        self.now.set("Burning area", u.text(float(np.interp(d, g.web, g.burn_area)), "area"),
                     "The fuel surface that's burning: how fast the open volume grows per burned distance.")
        self.now.show_row("Exposed case wall", case > 0)
        self.now.set("Exposed case wall", f"{u.text(case, 'area')} ({100 * case / (np.pi * g.outer_diameter * g.length):.0f}%)",
                     "Case wall the burn has reached, with no fuel left on it, and its share of the whole wall.")
        self.now.set("Port area", f"{u.text(port, 'area')}  ·  Ø{u.text(2 * np.sqrt(port / np.pi), 'length')}",
                     "The port's open area averaged along the grain, and the round port with that area.")
        self.now.set("Fuel left", self._fuel_text(float(np.interp(d, g.web, g.fuel_volume)), self._get_density(), g))
        self.now.show_row("Time", timed)
        self.now.show_row("O/F", timed)
        if timed:
            o = self._run
            t = self.slider.value() / STEPS * float(o.t[-1])
            self.now.set("Time", f"{t:.2f} s")
            self.now.set("O/F", f"{float(np.interp(t, o.t, o.OF)):.2f}")
            self.position.setText(f"{t:.2f} s  ·  {u.text(d, 'length', 3)}")
        else:
            self.position.setText(u.text(d, "length", 3))

    def _show_preview(self):
        """A quick, coarser surface while the slider moves."""
        g = self._geom
        if g is None or self._view is None:
            return
        self._stale = not self.isVisible()
        if self._stale:
            return
        self._job += 1  # a full-detail surface still being worked out is for an older position
        step = max(1, round((g.distance.size / PREVIEW_CELLS) ** (1 / 3)))
        self._set_mesh(self._burned, *_burned_arrays(g, self._web(), step, self._colors))

    def _show_full(self):
        if self._geom is not None and self._view is not None and not self.play.isChecked() and self.isVisible():
            self._job += 1
            self._surfaces.start(self._job, self._geom, self._web(), self._colors)

    def _on_surface(self, job: int, arrays):
        if job == self._job:
            self._set_mesh(self._burned, *arrays)

    def _on_play(self, playing: bool):
        self.play.setText("❚❚  Pause" if playing else "▶  Play")
        if playing:
            if self.slider.value() >= STEPS:
                self.slider.setValue(0)
            self._play_start = (time.monotonic(), self.slider.value())
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self._show_full()

    def _play_step(self):
        started, value = self._play_start
        span = float(self._run.t[-1]) if self._time_mode() else PLAY_S
        self.slider.setValue(min(round(value + STEPS * (time.monotonic() - started) / span), STEPS))
        if self.slider.value() >= STEPS:
            self.play.setChecked(False)
