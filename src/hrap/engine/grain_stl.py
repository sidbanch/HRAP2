"""Uniform normal burn-back of a watertight STL fuel solid, in SI units.

The case is a cylinder and inhibits the outside. Disconnected voids are filled;
the largest void component is the port. Geometry alone does not model swirl,
heat transfer, detached slivers, or structural failure after wall contact.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.draw import polygon
from skimage.measure import marching_cubes


@dataclass
class GrainGeometry:
    web: np.ndarray
    port_volume: np.ndarray
    burn_area: np.ndarray
    port_area: np.ndarray
    fuel_volume: np.ndarray
    length: float
    outer_diameter: float
    distance: np.ndarray
    spacing: float
    origin: np.ndarray
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray


def load_grain_stl(path, *, units: str = "in", voxel: float | None = None,
                   ends_burn: bool = False) -> GrainGeometry:
    """Load fuel, with voxel size expressed in the chosen STL units.

    The longest bounding-box axis becomes z. Supply an axis-aligned STL; a
    general oblique orientation cannot be inferred reliably from its bounds.
    """
    path = Path(path).expanduser().resolve()
    return _load(str(path), path.stat().st_mtime_ns, units, voxel, ends_burn)


@lru_cache(maxsize=2)
def _load(path, mtime, units, voxel, ends_burn):
    if units not in ("in", "mm"):
        raise ValueError("STL units must be in or mm")
    scale = 0.0254 if units == "in" else 0.001
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError("STL must contain closed, watertight fuel solids")
    mesh.apply_scale(scale)
    axis = int(np.argmax(mesh.extents))
    order = [i for i in range(3) if i != axis] + [axis]
    mesh.vertices = mesh.vertices[:, order]
    mesh.vertices -= mesh.bounds.mean(axis=0)
    width, depth, length = mesh.extents
    if abs(width - depth) > 0.05 * max(width, depth) or length < max(width, depth):
        raise ValueError("Align the grain axis with X, Y or Z and use a cylindrical outer case")
    radius = max(width, depth) / 2
    spacing = max(0.000508, (width * depth * length / 24_000_000) ** (1/3)) if voxel is None else voxel * scale
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
    for part in mesh.split(only_watertight=True):
        for k in range(nz):
            section = part.section(plane_origin=[0, 0, origin[2] + k * spacing], plane_normal=[0, 0, 1])
            if section is None:
                continue
            layer = np.zeros((nxy, nxy), dtype=bool)
            for loop in section.discrete:
                ij = (loop[:, :2] - origin[:2]) / spacing
                rows, cols = polygon(ij[:, 0], ij[:, 1], layer.shape)
                layer[rows, cols] ^= True
            fuel[:, :, k] |= layer
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
    # Ask EDT for nearest-port indices, then form distances one layer at a time.
    # SciPy's full-grid coordinate differences otherwise peak near 2 GB here.
    nearest = ndimage.distance_transform_edt(~port, return_distances=False, return_indices=True)
    distance = np.empty(port.shape, dtype=np.float32)
    xx, yy = np.ogrid[:nxy, :nxy]
    for k in range(nz):
        squared = (nearest[0, :, :, k] - xx) ** 2 + (nearest[1, :, :, k] - yy) ** 2
        squared += (nearest[2, :, :, k] - k) ** 2
        distance[:, :, k] = np.sqrt(squared) * spacing
    del nearest
    # Distances between cell centres overestimate the interface by half a cell.
    distance[fuel] = np.maximum(distance[fuel] - spacing / 2, spacing / 2)
    if ends_burn:
        axial = (np.minimum(np.arange(nz), np.arange(nz)[::-1]) + 0.5) * spacing
        np.minimum(distance, axial[None, None, :], out=distance)
    distance[~np.broadcast_to(case[:, :, None], fuel.shape)] = np.inf
    centres = mesh.triangles_center
    ijk = ((centres - origin) / spacing).T
    near_port = ndimage.map_coordinates(distance, ijk, order=0, mode="nearest") <= spacing
    end_face = np.abs(centres[:, 2]) > length / 2 - spacing / 10
    initial_area = float(mesh.area_faces[near_port & ~end_face].sum())
    if ends_burn:
        initial_area += float(mesh.area_faces[end_face].sum())
    values = distance[fuel]
    web = np.linspace(0, float(values.max()), max(24, int(values.max() / spacing * 2) + 2))
    hist = np.histogram(values, bins=web)[0]
    burned = np.r_[0, np.cumsum(hist)].astype(float) * spacing ** 3
    total = burned[-1]
    # A quadratic volume curve removes lattice noise for smoothly evolving ports.
    # Only use it when its residual is below a quarter cell of interface motion;
    # mergers and slivers otherwise retain the local histogram smoothing.
    fit = np.polynomial.Polynomial.fit(web[2:-2], burned[2:-2], 2).convert()
    tolerance = total / web[-1] * spacing / 4
    if (np.max(np.abs(fit(web[2:-2]) - burned[2:-2])) < tolerance
            and initial_area > 0 and np.all(fit.deriv()(web) > 0)):
        # The STL gives an accurate initial surface area without differentiating
        # voxel counts. Constrain the smooth fit to that measured initial slope.
        design = np.c_[np.ones(len(web) - 4), web[2:-2] ** 2]
        _, quadratic = np.linalg.lstsq(design, burned[2:-2] - initial_area * web[2:-2], rcond=None)[0]
        fit = np.polynomial.Polynomial([0, initial_area, quadratic])
        roots = (fit - total).roots()
        positive = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        end = min(positive, key=lambda r: abs(r - web[-1])) if positive else 0
        if abs(end - web[-1]) < spacing:
            web = np.linspace(0, end, len(web))
            burned = fit(web)
        else:
            burned = _smooth_volume(burned)
    else:
        burned = _smooth_volume(burned)
    burned = np.maximum.accumulate(np.clip(burned, 0, total))
    burned[0], burned[-1] = 0, total
    volume = port.sum() * spacing ** 3 + burned
    area = np.maximum(0, np.gradient(volume, web, edge_order=2))
    return GrainGeometry(web, volume, area, volume / length, burned[-1] - burned,
                         float(length), float(2 * radius), distance, spacing, origin,
                         np.asarray(mesh.vertices), np.asarray(mesh.faces))


def _smooth_volume(burned):
    smoothed = savgol_filter(burned, min(11, len(burned) // 2 * 2 - 1), 2)
    smoothed -= smoothed[0]
    return smoothed * burned[-1] / smoothed[-1]


def port_surface(geom: GrainGeometry, web: float, *, step: int = 1):
    """Return the burned-region boundary, including its open-end caps."""
    field = geom.distance[::step, ::step, ::step].copy()
    field[~np.isfinite(field)] = geom.web[-1] + geom.spacing * 2
    field = np.pad(field, 1, constant_values=geom.web[-1] + geom.spacing * 2)
    level = max(float(web), geom.spacing * 1e-4)
    vertices, faces, _, _ = marching_cubes(field, level=level, spacing=(geom.spacing * step,) * 3)
    return vertices + geom.origin - geom.spacing * step, faces
