"""Analytic cylinder check and generated geometry for manual STL comparisons."""
import json
import pickle

import numpy as np
import pytest
import trimesh

from hrap.engine.grain_stl import _load, load_grain_stl, port_surface, read_grain_mesh
from hrap.engine.sim import run
from hrap.engine.study import case_cfg, run_case, study
from hrap.engine.summary import summarize
from hrap.io.config import bundled_motor, load_json, resolve, save_json


def twisted_grain(path):
    """Six radial slots with one full twist, dimensions in inches."""
    count, layers = 192, 81
    theta = np.arange(count) * 2 * np.pi / count
    inner = np.where(np.cos(6 * theta) > 0.7, 1.45, 1.05)
    vertices = []
    for z in np.linspace(-10, 10, layers):
        angle = theta + (z + 10) / 20 * 2 * np.pi
        for radius in (np.full(count, 1.695), inner):
            vertices.extend(np.c_[radius * np.cos(angle), radius * np.sin(angle), np.full(count, z)])
    faces = []
    def quad(a, b, c, d):
        faces.extend([(a, b, c), (a, c, d)])
    for k in range(layers - 1):
        for ring in (0, 1):
            for j in range(count):
                a = k * count * 2 + ring * count + j
                b = k * count * 2 + ring * count + (j + 1) % count
                quad(a, b, b + 2 * count, a + 2 * count)
    for k in (0, layers - 1):
        for j in range(count):
            a = k * count * 2 + j
            b = k * count * 2 + (j + 1) % count
            quad(a, b, b + count, a + count)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    mesh.export(path)


def base_motor():
    cfg = bundled_motor("example_98mm")
    cfg.update(grn_OD=3.39, grn_ID=2.5, grn_L=20.0, reg_model="Shifting OF")
    return cfg


def test_stl_geometry_and_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr("hrap.engine.grain_stl._cache_dir", lambda: tmp_path / "cache")
    path = tmp_path / 'cylinder.stl'
    trimesh.creation.annulus(r_min=1.25, r_max=1.695, height=20, sections=256).export(path)
    events = []
    g = load_grain_stl(path, progress=lambda *event: events.append(event))
    assert events[0] == ("Reading STL", 0.0) and events[-1] == ("Ready", 1.0)
    assert len(events) > 10 and np.all(np.diff([fraction for _, fraction in events]) >= 0)
    vertices, faces = read_grain_mesh(path)
    np.testing.assert_array_equal(vertices, g.mesh_vertices)
    np.testing.assert_array_equal(faces, g.mesh_faces)
    before = set((tmp_path / "cache").iterdir())
    def cancel(stage, fraction):
        if fraction > 0.25:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        load_grain_stl(path, voxel=0.1, progress=cancel)
    assert set((tmp_path / "cache").iterdir()) == before
    assert load_grain_stl(path) is g
    radius = 1.25 * 0.0254 + g.web
    burning = g.web < 0.445 * 0.0254 - g.spacing / 2
    np.testing.assert_allclose(g.port_area[burning], np.pi * radius[burning]**2, rtol=0.02)
    np.testing.assert_allclose(g.burn_area[burning], 2 * np.pi * radius[burning] * g.length, rtol=0.02)
    vertices, faces = port_surface(g, 0, step=4)
    assert np.isfinite(vertices).all() and len(faces) > 0
    cfg = base_motor()
    s, x = resolve(cfg)
    x, o = run(s, x)
    reference = summarize(s, x, o)
    cfg.update(grn_shape='stl', grn_stl=str(path))
    save_json(tmp_path / "motor.json", cfg)
    assert json.loads((tmp_path / "motor.json").read_text())["grn_stl"] == path.name
    cfg = load_json(tmp_path / "motor.json")
    assert cfg["grn_stl"] == str(path)
    s, x = resolve(cfg)
    initial_mass = x.m_f
    x, o = run(s, x)
    actual = summarize(s, x, o)
    for name in ('total_impulse', 'avg_OF'):
        assert actual[name] == pytest.approx(reference[name], rel=0.01)
    assert initial_mass - x.m_f == pytest.approx(np.sum(o.mdot_f[1:]) * s.dt)
    np.testing.assert_allclose(o.m_f, s.prop_Rho * np.interp(o.grn_web, g.web, g.fuel_volume))
    assert np.all(np.diff(o.grn_web) >= 0)
    cfg['reg_model'] = 'Constant OF'
    with pytest.raises(ValueError, match='Shifting OF'):
        resolve(cfg)

    twist_path = tmp_path / "twisted.stl"
    twisted_grain(twist_path)
    fine = load_grain_stl(twist_path, voxel=0.54 / 25.4)
    coarse = load_grain_stl(twist_path, voxel=0.76 / 25.4)
    mesh = trimesh.Trimesh(fine.mesh_vertices, fine.mesh_faces, process=False)
    centres = mesh.triangles_center
    port_faces = ((abs(centres[:, 2]) < fine.length / 2 - 1e-6)
                  & (np.linalg.norm(centres[:, :2], axis=1) < fine.outer_diameter / 2 - 0.001))
    initial_area = mesh.area_faces[port_faces].sum()
    assert fine.burn_area[0] == pytest.approx(initial_area, rel=0.02)
    assert coarse.burn_area[0] == pytest.approx(initial_area, rel=0.02)
    comparison = np.interp(coarse.web, fine.web, fine.burn_area)
    assert np.max(abs(coarse.burn_area - comparison)) / initial_area < 0.02
    assert fine.burn_area[-1] == 0 and fine.fuel_volume[-1] == 0
    assert np.all(np.diff(fine.case_area) >= 0)
    assert not fine.case_area[fine.web < fine.case_web].any()
    assert fine.case_web < fine.web[-1]
    assert np.all(fine.distance[np.isfinite(fine.distance)] >= 0)
    assert len(pickle.dumps(fine.tables())) < 20000
    _load.cache_clear()
    events = []
    cached = load_grain_stl(twist_path, voxel=0.54 / 25.4, progress=lambda *event: events.append(event))
    assert events == [("Ready", 1.0)]
    np.testing.assert_array_equal(cached.distance, fine.distance)
    cfg.update(reg_model="Shifting OF", grn_stl=str(twist_path), t_max=0.02)
    settings, _ = resolve(cfg)
    assert not hasattr(settings.grain_geometry, "distance")
    for key in ("grain_L", "port_D"):
        with pytest.raises(ValueError, match="cannot be swept"):
            case_cfg(cfg, [(key, 0.1)], None)
    sized = run_case(cfg, [("start_OF", 6.0)], None, throat_P=2e6, grain_tables=fine.tables())
    assert sized.inj_CdA > 0
    cases = list(study(cfg, [("grain_file", [str(path), str(twist_path)])]))
    assert {case.grain_label for case in cases} == {path.name, twist_path.name}
    assert all(case.grain_file for case in cases)

    # Shared and sub-cell-gapped mating faces must not become burning slots.
    half = trimesh.creation.annulus(r_min=1.25, r_max=1.695, height=10, sections=128)
    for gap in (0, 0.01):
        a, b = half.copy(), half.copy()
        a.apply_translation([0, 0, -5 - gap / 2])
        b.apply_translation([0, 0, 5 + gap / 2])
        joined = tmp_path / f"joined-{gap}.stl"
        trimesh.util.concatenate([a, b]).export(joined)
        joined_geom = load_grain_stl(joined, voxel=1 / 25.4)
        assert joined_geom.burn_area[0] == pytest.approx(2 * np.pi * 0.03175 * joined_geom.length, rel=0.02)
        assert joined_geom.case_web == pytest.approx(0.445 * 0.0254, abs=joined_geom.spacing / 4)
