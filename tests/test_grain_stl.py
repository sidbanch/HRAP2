"""Analytic cylinder check and generated geometry for manual STL comparisons."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from hrap.engine.grain_stl import load_grain_stl, port_surface
from hrap.engine.sim import run
from hrap.engine.summary import summarize
from hrap.io.config import load_json, resolve, save_json


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
    path = Path(__file__).resolve().parents[1] / 'HCAT motor files/Development/HPS01-2/HPS01-2_24in_1x.json'
    cfg = load_json(path)
    cfg.update(grn_OD=3.39, grn_ID=2.5, grn_L=20.0, reg_model="Shifting OF")
    return cfg


def test_cylinder_stl_matches_round(tmp_path):
    path = tmp_path / 'cylinder.stl'
    trimesh.creation.annulus(r_min=1.25, r_max=1.695, height=20, sections=256).export(path)
    g = load_grain_stl(path)
    assert load_grain_stl(path) is g
    radius = 1.25 * 0.0254 + g.web
    np.testing.assert_allclose(g.port_area, np.pi * radius**2, rtol=0.02)
    np.testing.assert_allclose(g.burn_area, 2 * np.pi * radius * g.length, rtol=0.025)
    vertices, faces = port_surface(g, 0, step=4)
    assert np.isfinite(vertices).all() and len(faces) > 0
    cfg = base_motor()
    s, x = resolve(cfg)
    x, o = run(s, x)
    reference = summarize(s, x, o)
    cfg.update(grn_shape='stl', grn_stl=str(path))
    cfg["grn_stl"] = path.name
    save_json(tmp_path / "motor.json", cfg)
    cfg = load_json(tmp_path / "motor.json")
    assert cfg["grn_stl"] == str(path)
    s, x = resolve(cfg)
    initial_mass = x.m_f
    x, o = run(s, x)
    actual = summarize(s, x, o)
    print('maximum port-area error', np.max(abs(g.port_area / (np.pi * radius**2) - 1)),
          'maximum burn-area error', np.max(abs(g.burn_area / (2 * np.pi * radius * g.length) - 1)))
    print({key: (reference[key], actual[key]) for key in ('burn_time', 'total_impulse', 'avg_OF')})
    for name in ('total_impulse', 'avg_OF'):
        assert actual[name] == pytest.approx(reference[name], rel=0.01)
    assert initial_mass - x.m_f == pytest.approx(np.sum(o.mdot_f[1:]) * s.dt)
    np.testing.assert_allclose(o.m_f, s.prop_Rho * np.interp(o.grn_web, g.web, g.fuel_volume))
    assert np.all(np.diff(o.grn_web) >= 0)
    cfg['reg_model'] = 'Constant OF'
    with pytest.raises(ValueError, match='Shifting OF'):
        resolve(cfg)
