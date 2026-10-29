"""Uniform normal burn-back of a watertight STL fuel solid, in SI units.

The case is a cylinder and inhibits the outside. Disconnected voids are filled;
the largest void component is the port. Geometry alone does not model swirl,
heat transfer, detached slivers, or structural failure after wall contact.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Callable
from zipfile import BadZipFile

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

_progress: ContextVar[Callable[[str, float], None] | None] = ContextVar("grain_progress", default=None)


def _report(stage: str, fraction: float):
    callback = _progress.get()
    if callback is not None:
        callback(stage, fraction)


@dataclass
class GrainTables:
    web: np.ndarray
    port_volume: np.ndarray
    burn_area: np.ndarray
    port_area: np.ndarray
    fuel_volume: np.ndarray
    length: float
    outer_diameter: float
    case_area: np.ndarray
    case_web: float
    resolution_warning: str


@dataclass
class GrainGeometry(GrainTables):
    distance: np.ndarray
    spacing: float
    origin: np.ndarray
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray

    def tables(self) -> GrainTables:
        return GrainTables(**{f.name: getattr(self, f.name) for f in fields(GrainTables)})


def load_grain_stl(path, *, units: str = "in", voxel: float | None = None,
                   ends_burn: bool = False,
                   progress: Callable[[str, float], None] | None = None) -> GrainGeometry:
    """Load fuel, with voxel size expressed in the chosen STL units.

    The axis with matching transverse dimensions becomes z. Supply an axis-aligned STL; a
    general oblique orientation cannot be inferred reliably from its bounds.
    Progress receives a stage name and overall fraction; raising cancels the load.
    Memory and disk cache hits report only ("Ready", 1.0).
    """
    path = _stl_path(path)
    token = _progress.set(progress)
    try:
        result = _load(str(path), path.stat().st_mtime_ns, units, voxel, ends_burn)
        _report("Ready", 1.0)
        return result
    finally:
        _progress.reset(token)


def _stl_path(path):
    if not path:
        raise ValueError("No STL file chosen")
    if not isinstance(path, (str, Path)):
        raise ValueError("STL path must be a file name")
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=2)
def _load(path, mtime, units, voxel, ends_burn):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(json.dumps([9, units, voxel, ends_burn]).encode())
    cache = _cache_dir() / (digest.hexdigest() + ".npz")
    try:
        with np.load(cache, allow_pickle=False) as saved:
            values = {f.name: saved[f.name] for f in fields(GrainGeometry)}
            for key in ("length", "outer_diameter", "case_web", "spacing"):
                values[key] = float(values[key])
            values["resolution_warning"] = str(values["resolution_warning"])
            result = GrainGeometry(**values)
        cache.touch()
        return result
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        pass
    result = _build(path, units, voxel, ends_burn)
    _report("Saving geometry", 0.98)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **{f.name: getattr(result, f.name) for f in fields(GrainGeometry)})
        os.replace(temporary, cache)
        size = 0
        for entry in sorted(cache.parent.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
            size += entry.stat().st_size
            if size > 1024 ** 3:
                entry.unlink(missing_ok=True)
    except OSError:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)
    return result


def _cache_dir():
    from PySide6.QtCore import QCoreApplication, QStandardPaths

    from hrap import APP_NAME
    if QCoreApplication.instance() is not None:
        root = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    else:
        root = Path(QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation)) / "HCAT" / APP_NAME
    return root / "grain-cache"


def read_grain_mesh(path, *, units: str = "in") -> tuple[np.ndarray, np.ndarray]:
    """Read the STL quickly; vertices are centred, oriented along z and in metres."""
    path = _stl_path(path)
    if units not in ("in", "mm"):
        raise ValueError("STL units must be in or mm")
    scale = 0.0254 if units == "in" else 0.001
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("STL must contain triangular fuel solids")
    counts = np.bincount(mesh.edges_unique_inverse)
    if np.any(counts % 2):
        raise ValueError("STL fuel solids have open edges; export closed solids")
    mesh.apply_scale(scale)
    extents = mesh.extents
    mismatch = [abs(extents[(i + 1) % 3] - extents[(i + 2) % 3]) / max(extents[(i + 1) % 3], extents[(i + 2) % 3])
                for i in range(3)]
    axis = int(np.argmin(mismatch))
    order = [(axis + 1) % 3, (axis + 2) % 3, axis]
    mesh.vertices = mesh.vertices[:, order]
    mesh.vertices -= mesh.bounds.mean(axis=0)
    width, depth, length = mesh.extents
    if abs(width - depth) > 0.05 * max(width, depth):
        raise ValueError("Align the grain axis with X, Y or Z and use a cylindrical outer case")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _build(path, units, voxel, ends_burn):
    _report("Reading STL", 0.0)
    vertices, faces = read_grain_mesh(path, units=units)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    width, depth, length = mesh.extents
    scale = 0.0254 if units == "in" else 0.001
    _report("Filling cubes", 0.04)
    radius = max(width, depth) / 2
    spacing = max(0.00076, (width * depth * length / 24_000_000) ** (1/3)) if voxel is None else voxel * scale
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Voxel size must be positive")
    # Use an integer number of axial cells, preserving the exact grain length.
    spacing = length / max(1, round(length / spacing))
    nxy = int(np.ceil(2 * radius / spacing)) + 2
    nz = int(round(length / spacing))
    if nxy * nxy * nz > 35_000_000:
        raise ValueError("Voxel size is too small; choose a coarser grid (maximum 35 million cells)")
    origin = np.array([-(nxy - 1) * spacing / 2] * 2 + [-length / 2 + spacing / 2])
    coords = origin[0] + np.arange(nxy) * spacing
    case = coords[:, None] ** 2 + coords[None, :] ** 2 <= radius ** 2
    fuel = np.zeros((nxy, nxy, nz), dtype=bool)
    for k in range(nz):
        lines, face_ids = trimesh.intersections.mesh_plane(
            mesh, plane_origin=[0, 0, origin[2] + k * spacing], plane_normal=[0, 0, 1], return_faces=True)
        fuel[:, :, k] = _scanlines(lines, mesh.face_normals[face_ids, 1], coords)
        if k % 16 == 0:
            _report("Filling cubes", 0.04 + 0.18 * (k + 1) / nz)
    # Treat sub-cell assembly gaps as bonded joints, not additional burning faces.
    fuel = ndimage.binary_closing(fuel, structure=np.ones((1, 1, 3)), border_value=1)
    _report("Finding the port", 0.24)
    void = case[:, :, None] & ~fuel
    labels, count = ndimage.label(void)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if count == 0:
        raise ValueError("STL has no open port inside its outer diameter")
    port = labels == sizes.argmax()
    if not port[:, :, 0].any() or not port[:, :, -1].any():
        raise ValueError("The largest port must open at both ends of the grain")
    del labels, void
    fuel = case[:, :, None] & ~port
    if not fuel.any():
        raise ValueError("No fuel is resolved at this cube size; choose smaller cubes")
    centres = mesh.triangles_center
    normals = mesh.face_normals
    near = np.zeros(len(centres), dtype=bool)
    for sign in (-1, 1):
        indices = ((centres + sign * normals * spacing - origin) / spacing).T
        near ^= ndimage.map_coordinates(port.astype(np.uint8), indices, order=0, mode="nearest") > 0
    caps = (np.abs(centres[:, 2]) >= length / 2 - spacing / 10) & (np.abs(normals[:, 2]) > 1 - 1e-8)
    outer = np.all(np.linalg.norm(mesh.triangles[:, :, :2], axis=2) > radius - spacing / 10, axis=1)
    # A port face separates fuel from the port. Matching assembly faces have
    # the same material on both sides and must not seed the burn distance.
    surface = mesh.triangles[near & ~caps & ~outer]
    if not len(surface):
        raise ValueError("Could not identify the grain's open port surface")
    _report("Measuring burn distances", 0.28)
    tree, triangle_ids = _surface_tree(surface, spacing)
    # Distance is smooth away from the initial interface. Sample that region on
    # a two-cell grid, then evaluate the interface band directly on STL triangles.
    # Retain actual negative distances near the surface; unmeasured deep-port
    # cells stay NaN here and are counted separately, never put in the histogram.
    stride = 2
    cx = np.unique(np.r_[np.arange(0, nxy, stride), nxy - 1])
    cz = np.unique(np.r_[np.arange(0, nz, stride), nz - 1])
    filter_width = min(spacing, 0.00054)
    # Beyond the filter's four-sigma support, port cells contribute only volume.
    # Five extra cells cover interpolation corners and the rasterized interface.
    margin = int(np.ceil(4 * filter_width / spacing)) + 5
    deep_port = ndimage.binary_erosion(port, structure=np.ones((3, 3, 3)),
                                      iterations=margin, border_value=1)
    coarse = np.full((len(cx), len(cx), len(cz)), np.nan, dtype=np.float32)
    coarse_xy = np.array(np.meshgrid(cx, cx, indexing="ij")).reshape(2, -1).T
    for j, k in enumerate(cz):
        points = np.c_[origin[:2] + coarse_xy * spacing, np.full(len(coarse_xy), origin[2] + k * spacing)]
        evaluate = ~deep_port[coarse_xy[:, 0], coarse_xy[:, 1], k]
        dist = np.full(len(points), np.nan)
        dist[evaluate] = _surface_distance(points[evaluate], tree, triangle_ids, surface)
        dist[port[coarse_xy[:, 0], coarse_xy[:, 1], k]] *= -1
        coarse[:, :, j] = dist.reshape(len(cx), len(cx))
        if j % 16 == 0:
            _report("Measuring burn distances", 0.38 + 0.15 * (j + 1) / len(cz))
    distance = np.full(port.shape, np.inf, dtype=np.float32)
    xy = np.column_stack(np.nonzero(case))
    uv = np.c_[np.interp(xy[:, 0], cx, np.arange(len(cx))), np.interp(xy[:, 1], cx, np.arange(len(cx)))]
    for k in range(nz):
        w = np.interp(k, cz, np.arange(len(cz)))
        dist = ndimage.map_coordinates(coarse, np.c_[uv, np.full(len(xy), w)].T, order=1)
        near = (np.abs(dist) < 2 * spacing) | (~np.isfinite(dist) & ~port[xy[:, 0], xy[:, 1], k])
        points = np.c_[origin[:2] + xy[near] * spacing, np.full(near.sum(), origin[2] + k * spacing)]
        dist[near] = _surface_distance(points, tree, triangle_ids, surface)
        dist = np.abs(dist)
        dist[port[xy[:, 0], xy[:, 1], k]] *= -1
        distance[xy[:, 0], xy[:, 1], k] = dist
        if k % 16 == 0:
            _report("Measuring burn distances", 0.53 + 0.25 * (k + 1) / nz)
    angle = np.arange(max(16, int(np.ceil(2 * np.pi * radius / spacing))))
    angle = angle * 2 * np.pi / len(angle)
    wall_points = radius * np.c_[np.cos(angle), np.sin(angle)]
    _report("Measuring case exposure", 0.78)
    wall_distances = []
    side_end = 0.0
    for k in range(nz):
        points = np.c_[wall_points, np.full(len(wall_points), origin[2] + k * spacing)]
        dist = _surface_distance(points, tree, triangle_ids, surface)
        side_end = max(side_end, float(dist.max()))
        if ends_burn:
            dist = np.minimum(dist, (min(k, nz - 1 - k) + 0.5) * spacing)
        wall_distances.append(dist)
        if k % 16 == 0:
            _report("Measuring case exposure", 0.78 + 0.10 * (k + 1) / nz)
    wall_distances = np.sort(np.concatenate(wall_distances))
    case_web = 0.0 if ends_burn else float(wall_distances[0])
    # Smooth lattice noise on at most one cell, so refinement also shrinks the
    # filter. Reflect at the last occupied bin instead of extending the fuel.
    _report("Building tables", 0.90)
    values = distance[np.isfinite(distance)]
    step = min(spacing / 4, 0.000135)
    low, high = float(values.min()), float(values.max())
    edges = np.linspace(low, high, int(np.ceil((high - low) / step)) + 1)
    step = edges[1] - edges[0]
    counts = np.histogram(values, edges)[0].astype(float)
    sigma = filter_width / step
    smoothed = ndimage.gaussian_filter1d(counts, sigma=sigma, mode="reflect")
    cumulative = np.r_[0, np.cumsum(smoothed)] * spacing ** 3
    omitted = np.count_nonzero(np.isnan(distance))
    port_volume = float(np.interp(0, edges, cumulative)) + omitted * spacing ** 3
    total = np.pi * radius ** 2 * length
    end = max(high, side_end)
    web = np.r_[np.arange(0, end, step), end]
    area = np.interp(web, (edges[1:] + edges[:-1]) / 2, smoothed * spacing ** 3 / step)
    burned = np.r_[0, np.cumsum((area[1:] + area[:-1]) * np.diff(web) / 2)]
    scale_area = (total - port_volume) / burned[-1]
    area *= scale_area
    volume = port_volume + burned * scale_area
    volume[-1] = total
    if ends_burn:
        web, remaining = _burning_ends(distance, case, edges, sigma, web, total - volume, spacing, length)
        volume = total - remaining
        area = np.maximum(0, np.gradient(volume, web, edge_order=2))
        for k in range(nz):
            distance[:, :, k][case] = np.minimum(distance[:, :, k][case],
                                               (min(k, nz - 1 - k) + 0.5) * spacing)
    case_area = np.searchsorted(wall_distances, web, side="right") / len(wall_distances) * 2 * np.pi * radius * length
    area[-1] = 0.0
    _report("Building tables", 0.96)
    distance[port] = 0
    cells = web[-1] / spacing
    warning = f"Walls are only {cells:.1f} cubes thick; set a smaller cube size." if cells < 8 else ""
    return GrainGeometry(web, volume, area, volume / length, total - volume,
                         float(length), float(2 * radius), case_area, case_web, warning, distance, spacing, origin,
                         np.asarray(mesh.vertices), np.asarray(mesh.faces))


def _burning_ends(distance, case, edges, sigma, web, fuel_volume, spacing, length):
    """Retain only the fraction of each axial cell between the receding ends.

    End burning does not change the initial port volume. Keeping the end motion
    out of the side-distance histogram avoids smoothing end fuel into the port.
    """
    end = min(float(web[-1]), length / 2)
    shortened_web = np.r_[web[web < end], end]
    fuel = np.interp(shortened_web, web, fuel_volume)
    weights = np.zeros(len(shortened_web))
    surviving = np.zeros(len(shortened_web))
    nz = distance.shape[2]
    for k in range(nz):
        values = distance[:, :, k][case]
        measured = values[np.isfinite(values)]
        counts = np.histogram(measured, edges)[0].astype(float)
        smoothed = ndimage.gaussian_filter1d(counts, sigma=sigma, mode="reflect")
        cumulative = np.r_[0, np.cumsum(smoothed)]
        local_fuel = np.maximum(0, len(measured) - np.interp(shortened_web, edges, cumulative))
        overlap = np.clip(np.minimum((k + 1) * spacing, length - shortened_web)
                          - np.maximum(k * spacing, shortened_web), 0, spacing) / spacing
        weights += local_fuel
        surviving += local_fuel * overlap
        if k % 16 == 0:
            _report("Building tables", 0.90 + 0.04 * (k + 1) / nz)
    fraction = np.divide(surviving, weights, out=np.zeros_like(weights), where=weights > 0)
    remaining = fuel * fraction
    remaining[-1] = 0
    return shortened_web, remaining


def _scanlines(lines, normals_y, coords):
    """Vectorized winding fill: overlapping bodies union instead of XOR."""
    layer = np.zeros((len(coords), len(coords)), dtype=bool)
    if not len(lines):
        return layer
    a, b = lines[:, 0, :2], lines[:, 1, :2]
    for i, x in enumerate(coords):
        hit = ((a[:, 0] <= x) & (b[:, 0] > x)) | ((b[:, 0] <= x) & (a[:, 0] > x))
        if not hit.any():
            continue
        y = a[hit, 1] + (x - a[hit, 0]) * (b[hit, 1] - a[hit, 1]) / (b[hit, 0] - a[hit, 0])
        order = np.argsort(y)
        winding = np.cumsum(-np.sign(normals_y[hit][order]))
        indices = np.searchsorted(y[order], coords, side="right") - 1
        layer[i] = (indices >= 0) & (winding[np.maximum(indices, 0)] != 0)
    return layer


def _surface_tree(triangles, spacing):
    # Sample rows across each triangle without recursively subdividing skinny STL
    # facets: long cylindrical facets otherwise generate millions of tiny faces.
    samples, sample_ids = [], []
    for index, tri in enumerate(triangles):
        if index % 512 == 0:
            _report("Measuring burn distances", 0.28 + 0.10 * index / len(triangles))
        lengths = np.linalg.norm(tri - np.roll(tri, -1, axis=0), axis=1)
        i = int(lengths.argmax())
        a, b, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
        height = np.linalg.norm(np.cross(b - a, c - a)) / lengths[i]
        rows = max(1, int(np.ceil(height / spacing)))
        for row in range(rows):
            fraction = (row + 0.5) / rows
            count = max(1, int(np.ceil((1 - fraction) * lengths[i] / spacing)))
            along = (np.arange(count) + 0.5) / count
            samples.append((1 - fraction) * (a + along[:, None] * (b - a)) + fraction * c)
            sample_ids.append(np.full(count, index, dtype=np.int32))
    return cKDTree(np.concatenate(samples)), np.concatenate(sample_ids)


def _surface_distance(points, tree, ids, triangles):
    _, nearest = tree.query(points, k=4)
    candidates = triangles[ids[nearest]].reshape(-1, 3, 3)
    repeated = np.repeat(points, 4, axis=0)
    closest = trimesh.triangles.closest_point(candidates, repeated)
    return np.linalg.norm(closest - repeated, axis=1).reshape(-1, 4).min(axis=1)


def port_surface(geom: GrainGeometry, web: float, *, step: int = 1):
    """Return the burned-region boundary, including its open-end caps."""
    field = geom.distance[::step, ::step, ::step].copy()
    field[~np.isfinite(field)] = geom.web[-1] + geom.spacing * 2
    field = np.pad(field, 1, constant_values=geom.web[-1] + geom.spacing * 2)
    level = max(float(web), geom.spacing * 1e-4)
    vertices, faces, _, _ = marching_cubes(field, level=level, spacing=(geom.spacing * step,) * 3)
    return vertices + geom.origin - geom.spacing * step, faces
