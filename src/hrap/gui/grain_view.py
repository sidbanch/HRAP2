"""The STL fuel grain in 3D, burned back by a distance: on the Grain tab, to check the design, and on the
Simulation tab, at the time hovered on its plots."""
from __future__ import annotations

import math
import sys
import threading
import time
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
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

from hrap.gui.theme import colors_for
from hrap.gui.widgets import CARD_GAP, COLUMN_GAP, Card, page_layout
from hrap.units import DisplayUnits

STEPS = 1000  # slider positions
PLAY_S = 6.0  # Play burns the grain back over this long
PREVIEW_CELLS = 5e5  # grid cells in the quick surface shown while the burned distance changes
FOV = 30  # degrees across the 3D view
TAB_CAMERA = (20, -68)  # the Grain tab's camera elevation and azimuth, degrees
SIDE_CAMERA = (12, -80)  # the Simulation tab's wide, short panel: nearly side on


def _scene_colors(theme: str) -> dict:
    """The app theme's colours as the 3D view and the Grain tab's plots use them."""
    c = colors_for(theme)
    glass = pg.mkColor(c["glass"]).getRgbF()[:3]
    return {"bg": c["plot_bg"], "fg": c["plot_fg"], "fuel": (*glass, 0.16), "preview": (*glass, 0.5),
            "burned": c["fuel"], "case": c["limit"], "port_area": c["purple"]}

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



def _legend(colors: dict) -> str:
    return (f'<span style="color:{colors["burned"]}">■</span> Burning surface&nbsp;&nbsp;&nbsp;'
            f'<span style="color:{colors["case"]}">■</span> Exposed case wall')


def _holder() -> QWidget:
    """An empty spot for the 3D view."""
    box = QWidget()
    QVBoxLayout(box).setContentsMargins(0, 0, 0, 0)
    return box


class GrainScene(QObject):
    """The grain in 3D with the region burned back so far. Only one 3D view works per app here, so the Grain tab
    and the Simulation tab share this one: whichever is on screen holds it."""
    _surface_done = Signal(int, object)  # job number, full-detail mesh arrays

    def __init__(self):
        super().__init__()
        self.colors = _scene_colors("dark")
        self._view = None  # made when the first grain loads, so the app starts without OpenGL
        self._geom = None
        self._size = (1.0, 1.0)  # length and diameter of the grain on screen, to frame it
        self._holder = None  # the widget the view is in, and its camera's elevation and azimuth
        self._camera = TAB_CAMERA
        self._cameras = {}  # where the camera was left in the other holder
        self._web = 0.0  # the burned distance asked for
        self._shown = self._detailed = None  # the burned distance on screen, and asked for in full detail
        self._job = 0  # the newest full-detail surface asked for
        self._surface_done.connect(self._on_surface)
        self._preview_timer = QTimer(self, singleShot=True, interval=0)  # one quick surface per burst of moves
        self._preview_timer.timeout.connect(self._show_preview)
        self._settle_timer = QTimer(self, singleShot=True, interval=200)
        self._settle_timer.timeout.connect(self._show_full)

    def hold(self, holder: QWidget, camera: tuple[float, float]):
        """Move the view into a holder that's coming on screen, with the camera as it was left there."""
        if holder is self._holder:
            return
        if self._view is not None and self._holder is not None:
            o = self._view.opts
            self._cameras[self._holder] = {"center": pg.Vector(o["center"]), "distance": o["distance"],
                                           "elevation": o["elevation"], "azimuth": o["azimuth"]}
        self._holder, self._camera = holder, camera
        if self._view is not None:
            self._place()

    def _place(self):
        self._holder.layout().addWidget(self._view)
        self._view.show()
        camera = self._cameras.pop(self._holder, None)
        if camera is None:
            self.reset_camera()
        else:
            self._view.setCameraParams(**camera)

    def set_grain(self, geom):
        """Show a loaded grain, not burned yet."""
        self._geom, self._shown, self._detailed = geom, None, None
        self._show_fuel(geom.mesh_vertices, geom.mesh_faces, geom.length, geom.outer_diameter)
        self.burn_to(self._holder, 0.0)

    def set_shape(self, vertices: np.ndarray, faces: np.ndarray, length: float, diameter: float):
        """A grain still loading: just its shape."""
        self._geom = None
        self._show_fuel(vertices, faces, length, diameter)

    def _show_fuel(self, vertices: np.ndarray, faces: np.ndarray, length: float, diameter: float):
        first = self._view is None
        if first:
            self._make_view()
        self._burned.hide()  # until the grain's first burned surface is worked out
        self._fuel.setColor(self.colors["fuel" if self._geom is not None else "preview"])
        self._set_mesh(self._fuel, *_mesh_arrays(vertices, faces))
        self._size = (length, diameter)
        self._cameras.clear()
        if first and self._holder is not None:
            self._place()  # only now: showing the view paints it at once, and its items need their meshes
        else:
            self.reset_camera()

    def reset_camera(self):
        """Fit the whole grain in the view."""
        if self._view is None or self._holder is None:
            return
        length, diameter = self._size
        elevation, azimuth = (math.radians(a) for a in self._camera)
        across = length * abs(math.sin(azimuth)) + diameter * abs(math.cos(azimuth))
        up = diameter + length * abs(math.cos(azimuth)) * math.sin(elevation)
        aspect = self._holder.width() / max(self._holder.height(), 1)
        distance = max(across, up * aspect) / (0.85 * 2 * math.tan(math.radians(FOV / 2)))  # with 15% to spare
        self._view.setCameraParams(center=pg.Vector(0, 0, 0), distance=distance, elevation=self._camera[0],
                                   azimuth=self._camera[1], fov=FOV)

    def burn_to(self, holder: QWidget | None, web: float, settle: bool = True):
        """Show the grain burned back by a distance, if `holder` has the view: a quick surface at once, and the
        full-detail one once it stays put a moment (not with settle off, as while playing)."""
        if holder is None or holder is not self._holder:
            return
        self._web = web
        if web != self._shown:
            self._preview_timer.start()
        if not settle:
            self._settle_timer.stop()
        elif web != self._detailed:
            self._settle_timer.start()

    def set_theme(self, name: str):
        self.colors = _scene_colors(name)
        if self._view is not None:
            self._view.setBackgroundColor(self.colors["bg"])
            self._fuel.setColor(self.colors["fuel" if self._geom is not None else "preview"])
        self._shown = self._detailed = None
        self.burn_to(self._holder, self._web)

    def _can_draw(self) -> bool:
        return self._geom is not None and self._view is not None and self._holder is not None and self._holder.isVisible()

    def _show_preview(self):
        """A quick, coarser surface."""
        if not self._can_draw():
            return
        g = self._geom
        self._job += 1  # a full-detail surface still being worked out is for an older distance
        self._shown, self._detailed = self._web, None
        step = max(1, round((g.distance.size / PREVIEW_CELLS) ** (1 / 3)))
        self._set_mesh(self._burned, *_burned_arrays(g, self._web, step, self.colors))
        self._burned.show()

    def _show_full(self):
        """The full-detail surface, worked out off the UI thread."""
        if not self._can_draw() or self._web == self._detailed:
            return
        self._job += 1
        job, geom, web, colors = self._job, self._geom, self._web, self.colors
        self._detailed = web

        def work():
            try:
                self._surface_done.emit(job, _burned_arrays(geom, web, 1, colors))
            except Exception:
                pass  # the quick surface stays on screen
        threading.Thread(target=work, daemon=True).start()

    def _on_surface(self, job: int, arrays):
        if job == self._job:
            self._set_mesh(self._burned, *arrays)

    def _make_view(self):
        """The 3D view, made on first use so the app starts without OpenGL."""
        import pyqtgraph.opengl as gl
        from OpenGL import GL
        from pyqtgraph.opengl import shaders

        class Glass(gl.GLMeshItem):
            """See-through, without hiding its own far side."""
            def paint(self):
                GL.glDepthMask(GL.GL_FALSE)
                try:
                    super().paint()
                finally:
                    GL.glDepthMask(GL.GL_TRUE)  # or the depth buffer stops being cleared, and every frame after is wrong

        def shader(name, fragment):
            return shaders.ShaderProgram(name, [shaders.VertexShader(_VERTEX), shaders.FragmentShader(fragment)])

        self._gl = gl
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor(self.colors["bg"])
        self._burned = gl.GLMeshItem(shader=shader("hrapGrainSolid", _SOLID), smooth=True, glOptions="opaque")
        self._burned.hide()
        self._fuel = Glass(shader=shader("hrapGrainGlass", _GLASS), smooth=True, glOptions="translucent",
                           color=self.colors["fuel"])
        self._fuel.setDepthValue(1)  # drawn after the burned region, which shows through it
        self._view.addItem(self._burned)
        self._view.addItem(self._fuel)

    def _set_mesh(self, item, v, f, n, colors=None):
        md = self._gl.MeshData(vertexes=v, faces=f, vertexColors=colors)
        md._vertexNormals = n  # pyqtgraph works these out one vertex at a time in Python, far too slowly for these meshes
        item.setMeshData(meshdata=md)


class GrainView(QWidget):
    """Grain tab: the grain in 3D with a slider to burn it back, its burning and port areas, and its numbers.
    While a grain loads it shows its shape and the load's progress; without one, why not."""
    cancel_load = Signal()
    message_action = Signal()  # the button under a message was clicked

    def __init__(self, scene: GrainScene, get_units: Callable[[], DisplayUnits], get_density: Callable[[], float]):
        super().__init__()
        self._scene, self._get_units, self._get_density = scene, get_units, get_density
        self._geom = None
        self._play_start = (0.0, 0)  # when Play was pressed, and the slider then

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
        self._box = _holder()
        self._stack = QStackedWidget()
        self._stack.addWidget(waiting)
        self._stack.addWidget(self._box)
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
        self.position = QLabel("")
        self.position.setMinimumWidth(90)
        self.position.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.legend = QLabel("")
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self._scene.reset_camera())
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
        self.now = Card("At the slider", ["Burned distance", "Burning area", "Exposed case wall", "Port area",
                                          "Fuel left"])
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setMinimumHeight(180)  # the page fits the app's default 860 px window
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
        right.setSpacing(CARD_GAP)
        right.addWidget(self.grain)
        self.grain_warning = QLabel("")  # the cubes are too coarse for the grain's thinnest walls
        self.grain_warning.setObjectName("notApplied")
        self.grain_warning.setWordWrap(True)
        self.grain_warning.hide()
        right.addWidget(self.grain_warning)
        right.addWidget(self.now)
        right.addWidget(self.plots, 1)

        page = page_layout(QHBoxLayout(self))
        page.setSpacing(COLUMN_GAP)
        page.addLayout(left, 1)
        page.addWidget(right_column)

        self._play_timer = QTimer(self, interval=33)
        self._play_timer.timeout.connect(self._play_step)
        self.set_theme()
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
        self._stack.setCurrentWidget(self._box)
        for name in self.grain.values:
            self.grain.show_row(name, True)
        self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._scene.set_grain(geom)
        self.refresh()

    def set_loading(self, name: str, mesh: tuple | None):
        """While a grain loads: its shape as soon as the STL is read, and the load's progress below."""
        self._drop_grain()
        self._bottom.setCurrentIndex(1)
        self._bottom.show()
        if mesh is None:
            self._show_message(f"Loading {name}…")
            return
        vertices, faces = mesh
        size = np.ptp(vertices, axis=0)
        length, diameter = float(size[2]), float(max(size[:2]))
        self._stack.setCurrentWidget(self._box)
        self._scene.set_shape(vertices, faces, length, diameter)
        u = self._get_units()
        self._right.show()
        for w in (self.now, self.plots, self.grain_warning):
            w.hide()
        for row in self.grain.values:
            self.grain.show_row(row, row in ("Length", "Outer diameter"))
        self.grain.set("Length", u.text(length, "length"))
        self.grain.set("Outer diameter", u.text(diameter, "length"))

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
        self._right.hide()

    def _show_message(self, text: str, action: str = ""):
        self._stack.setCurrentIndex(0)
        self._message.setText(text)
        self._message.setFixedHeight(max(0, self._message.heightForWidth(self._message.width())))  # wrapped lines need room
        self._message_button.setText(action)
        self._message_button.setVisible(bool(action))

    def set_theme(self):
        """Take the scene's colours, which follow the app's theme."""
        colors = self._scene.colors
        self.plots.setBackground(colors["bg"])
        self.legend.setText(_legend(colors))
        for plot in self._area_plots:
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(colors["fg"])
                plot.getAxis(side).setTextPen(colors["fg"])
        for marker in self._markers:
            marker.setPen(pg.mkPen(colors["fg"], width=1, style=Qt.PenStyle.DashLine))
            marker.setHoverPen(pg.mkPen(colors["burned"], width=2))
        for line in self._case_lines:
            line.setPen(pg.mkPen(colors["case"], width=1.5, style=Qt.PenStyle.DashLine))
        self._case_label.setColor(colors["case"])
        for card, name in ((self.grain, "Case wall reached"), (self.now, "Exposed case wall")):
            card.values[name].setStyleSheet(f"color: {colors['case']}")
        self.refresh()

    def refresh(self):
        """Redraw the plots and numbers, for new units or a new theme."""
        g = self._geom
        if g is None:
            return
        u = self._get_units()
        colors = self._scene.colors
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
            self.grain.set("Case wall reached", f"{u.text(g.case_web, 'length', 3)} burned",
                           "Where the burn first breaks through the fuel to the case wall. From here on the hot gas\n"
                           "touches the case, and the fuel left is in slivers that may break off.")
            self.grain.set("Fuel left then", self._fuel_text(float(np.interp(g.case_web, g.web, g.fuel_volume)), rho, g))
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
        top.plot(x, u.value(g.burn_area, "area"), pen=pg.mkPen(colors["burned"], width=2), name="Burning area")
        top.plot(x, u.value(g.case_area, "area"), pen=pg.mkPen(colors["case"], width=2), name="Exposed case wall")
        bottom.plot(x, u.value(g.port_area, "area"), pen=pg.mkPen(colors["port_area"], width=2))
        for plot, name in ((top, "Area"), (bottom, "Port area")):
            plot.setLabel("left", name, units=u.unit("area"), color=colors["fg"])
            plot.setXRange(0, float(x[-1]), padding=0)
        for line in self._case_lines:
            line.setVisible(reached)
            line.setValue(u.value(g.case_web, "length") if reached else 0.0)
        self._area_plots[1].setLabel("bottom", "Burned distance", units=u.unit("length"), color=colors["fg"])
        self._update_numbers()

    def _fuel_text(self, volume: float, rho: float, g) -> str:
        u = self._get_units()
        return (u.text(rho * volume, "mass") if rho > 0 else u.text(volume, "volume")) + f" ({100 * volume / g.fuel_volume[0]:.0f}%)"

    def showEvent(self, event):
        super().showEvent(event)
        self._scene.hold(self._box, TAB_CAMERA)
        if self._geom is not None:
            self._scene.burn_to(self._box, self._web())

    def _web(self) -> float:
        """The burned distance at the slider."""
        return self.slider.value() / STEPS * float(self._geom.web[-1])

    def _on_slider(self, *_):
        self._update_numbers()
        if self._geom is not None:
            self._scene.burn_to(self._box, self._web(), settle=not self.play.isChecked())

    def _on_marker(self, marker):
        d = marker.value() / self._get_units().value(1.0, "length")
        self.slider.setValue(round(float(np.clip(d / float(self._geom.web[-1]), 0, 1)) * STEPS))

    def _update_numbers(self):
        g = self._geom
        if g is None:
            return
        u = self._get_units()
        d = self._web()
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
        self.position.setText(u.text(d, "length", 3))

    def _on_play(self, playing: bool):
        self.play.setText("❚❚  Pause" if playing else "▶  Play")
        if playing:
            if self.slider.value() >= STEPS:
                self.slider.setValue(0)
            self._play_start = (time.monotonic(), self.slider.value())
            self._play_timer.start()
        else:
            self._play_timer.stop()
            if self._geom is not None:
                self._scene.burn_to(self._box, self._web())

    def _play_step(self):
        started, value = self._play_start
        self.slider.setValue(min(round(value + STEPS * (time.monotonic() - started) / PLAY_S), STEPS))
        if self.slider.value() >= STEPS:
            self.play.setChecked(False)


class RunGrainView(QFrame):
    """The Simulation tab's grain in 3D: as far burned back as it was at the time last hovered on the plots."""

    def __init__(self, scene: GrainScene, get_units: Callable[[], DisplayUnits]):
        super().__init__()
        self.setObjectName("motorPanel")
        self._scene, self._get_units = scene, get_units
        self._at = (None, 0.0, False)  # the time shown, its burned distance, and whether there's a run
        self._box = _holder()
        self._caption = QLabel("")
        self._legend = QLabel("")
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self._scene.reset_camera())
        row = QHBoxLayout()
        row.setContentsMargins(10, 4, 6, 6)
        row.setSpacing(6)
        row.addWidget(self._caption)
        row.addSpacing(18)
        row.addWidget(self._legend)
        row.addStretch(1)
        row.addWidget(reset)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._box, 1)
        layout.addLayout(row)
        self.set_theme()
        self.set_run(False)

    def set_run(self, ran: bool):
        """Back to the starting grain, for a new run or none."""
        self._at = (None, 0.0, ran)
        self.refresh()

    def show_at(self, t: float, web: float):
        """The grain at a time in the run, which had burned it back by `web`."""
        self._at = (t, web, True)
        self.refresh()

    def refresh(self):
        t, web, ran = self._at
        self._caption.setText(f"At {t:.2f} s: {self._get_units().text(web, 'length', 3)} burned" if t is not None else
                              "Starting grain. Hover a plot to see it later in the burn." if ran else "Starting grain")
        self._scene.burn_to(self._box, web)

    def set_theme(self):
        self._legend.setText(_legend(self._scene.colors))

    def showEvent(self, event):
        super().showEvent(event)
        self._scene.hold(self._box, SIDE_CAMERA)
        self._scene.burn_to(self._box, self._at[1])
