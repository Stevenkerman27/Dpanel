import json
from pathlib import Path

import numpy as np
import pytest
import torch

from panel.p1_cst_gradient import grass_section_geometry
from tools.web import geometry, params, pipeline

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "GRASS-plane" / "cst_airfoil_codec.py").is_file(),
    reason="GRASS-plane CST codec is not installed alongside this repository")


def test_grass_sections_drive_display_te_features_and_fixed_topology(tmp_path):
    p, _ = params.normalise({"grass_sections": params.GRASS_DEFAULTS})
    curves = geometry.section_data(p, count=5)
    data = p["grass_sections"]
    model = {name: torch.tensor(value, dtype=torch.float64) for name, value in data.items()}
    wing, xyz, _, area = grass_section_geometry(model, chord_panels=4, span_panels_per_segment=2)
    vertices = xyz.detach().numpy()
    positive = curves[len(data["chord"]) - 1:]
    for section in positive:
        for point in section["upper"] + section["lower"]:
            assert np.linalg.norm(vertices - point, axis=1).min() < 1e-12
    display = geometry.display_mesh(p, count=5)
    for section in curves:
        for point in (section["upper"][0], section["upper"][-1]):
            assert np.linalg.norm(np.asarray(display["vertices"]) - point, axis=1).min() < 1e-12
    features = geometry.write_mesh_features(tmp_path / "features.json", p)
    trailing = geometry.write_trailing_edge(tmp_path / "te.json", p)
    np.testing.assert_allclose(features["features"][0]["points"],
                               [s["upper"][0] for s in geometry.section_data(p)])
    np.testing.assert_allclose(trailing["points"],
                               [s["upper"][-1] for s in geometry.section_data(p)])
    assert trailing["reference_area"] == pytest.approx(float(area))
    assert len(wing.faces) > 0


@pytest.mark.parametrize("three_sections", [False, True])
def test_grass_step_surface_and_te_match_section_geometry(tmp_path, three_sections):
    gmsh = pytest.importorskip("gmsh")
    data = json.loads(json.dumps(params.GRASS_DEFAULTS))
    if three_sections:
        data["leading_edge_xyz"].insert(1, [.09, .9, .12])
        data["chord"].insert(1, .92)
        data["twist_radians"].insert(1, -.03)
        data["cst"].insert(1, [.19] * 8 + [-.14] * 8 + [.01, 0.])
    p, _ = params.normalise({"grass_sections": data})
    step = tmp_path / "wing.step"
    info = geometry.write_step(step, p)
    assert info["volume"] > 0
    curves = geometry.section_data(p)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(step), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        faces = gmsh.model.getEntities(2)
        for section in curves:
            for point in (section["upper"][0], section["upper"][10],
                          section["upper"][20], section["upper"][-1],
                          section["lower"][10], section["lower"][20]):
                distances = []
                for _, tag in faces:
                    closest, _ = gmsh.model.getClosestPoint(2, tag, point)
                    distances.append(np.linalg.norm(np.asarray(closest) - point))
                assert min(distances) < .0015
    finally:
        gmsh.finalize()


def test_grass_web_mesh_keeps_declared_te_conforming(tmp_path):
    job = pipeline.Job(tmp_path)
    built = pipeline.build(job, {"params": {"grass_sections": params.GRASS_DEFAULTS}})
    path, info = pipeline.mesh_surface(job, built["params"], .25, 4., ncpu=1)
    surface = json.loads(path.read_text(encoding="utf-8"))
    from panel.grass_mesh import reconstruct_grass_mesh
    _, wing, xyz, wake, area = reconstruct_grass_mesh(surface)
    assert info["panels"] > 0
    assert float(area.detach()) == pytest.approx(3.6)
    np.testing.assert_allclose(xyz.detach().numpy(), surface["coordinates"], atol=0, rtol=0)
    np.testing.assert_allclose(wake.detach().numpy()[:len(wing.te_geo)],
                               xyz.detach().numpy()[wing.te_geo], atol=0, rtol=0)
    surface["coordinates"][0][0] += .01
    with pytest.raises(ValueError, match="显示网格"):
        reconstruct_grass_mesh(surface)


def test_invalid_sections_rejected():
    data = json.loads(json.dumps(params.GRASS_DEFAULTS))
    data["cst"][0][-1] = .01
    with pytest.raises(ValueError, match="尾缘闭合"):
        params.normalise({"grass_sections": data})
