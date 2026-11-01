import numpy as np
import pytest

from panel.p1_adaptive_area import (adaptive_area_terms, composite_duffy_terms,
                                    _midpoint_split, _subdivide)
from panel.p1_projection_duffy import area_terms
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def sheet():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                    [2., 0., 0.], [2., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    values = np.array([[0.4, -0.2, 0.1], [0.4, 0.1, 0.7], [-0.2, 0.3, -0.2],
                       [-0.2, -0.2, 0.5]])
    gamma = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4],
                      [0.3, -0.3, 0.1], [0.2, 0.2, -0.1]])
    return xyz, faces, values, gamma


def test_midpoint_split_preserves_total_area():
    tri = np.stack([np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])])
    pieces = _midpoint_split(tri)
    assert pieces.shape == (4, 3, 3)
    parent = 0.5*np.linalg.norm(np.cross(tri[0, 1]-tri[0, 0], tri[0, 2]-tri[0, 0]))
    total = sum(0.5*np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0])) for p in pieces)
    assert abs(total-parent) < 1e-14
    assert _subdivide(tri, 2).shape == (16, 3, 3)


def test_composite_duffy_matches_the_projection_fan_far_from_the_sheet():
    xyz, faces, _values, _gamma = sheet()
    tri = xyz[faces]
    point = np.array([0.5, 0.5, 4.0])
    reference = area_terms(tri, point, 0.01, order=48, strips=8)
    composite = composite_duffy_terms(tri, point, 0.01, levels=2, order=12)
    scale = np.linalg.norm(reference.sum(axis=0))
    assert np.linalg.norm((composite-reference).sum(axis=0)) < 1e-5*scale


def test_adaptive_partition_counts_every_face_once():
    xyz, faces, _values, gamma = sheet()
    tri = xyz[faces]
    terms, diagnostics = adaptive_area_terms(tri, gamma, np.array([0.5, 0.5, 0.01]),
                                             0.01, near_ratio=2.0, budget=2e-4)
    assert diagnostics["near_faces"]+diagnostics["far_faces"] == len(tri)
    assert terms.shape == (len(tri), 3)
    assert np.isfinite(terms).all()
    _t, far_diag = adaptive_area_terms(tri, gamma, np.array([0.5, 0.5, 40.0]), 0.01,
                                       near_ratio=2.0, budget=2e-4)
    assert far_diag["near_faces"] == 0
    assert far_diag["far_faces"] == len(tri)


def test_tighter_budget_escalates_more_faces():
    xyz, faces, _values, gamma = sheet()
    tri = xyz[faces]
    point = np.array([0.5, 0.5, 0.004])
    _loose, loose = adaptive_area_terms(tri, gamma, point, 0.01, near_ratio=2.0,
                                        budget=2e-2, max_escalations=2)
    _tight, tight = adaptive_area_terms(tri, gamma, point, 0.01, near_ratio=2.0,
                                        budget=1e-14, max_escalations=2)
    assert tight["relative_estimate"] <= loose["relative_estimate"]+1e-18
    assert tight["escalated_faces"] >= loose["escalated_faces"]
    assert tight["escalations"] >= 1


def test_adaptive_velocity_agrees_with_the_projection_fan():
    xyz, faces, values, _gamma = sheet()
    point = np.array([0.5, 0.5, 0.02])
    fan = RegularizedP1SheetField([(xyz, faces, values)], freestream=(1., 0., 0.),
                                  core_radius=0.01, area_quadrature="projection_fan",
                                  fan_order=28, fan_strips=8)
    adaptive = RegularizedP1SheetField([(xyz, faces, values)], freestream=(1., 0., 0.),
                                       core_radius=0.01,
                                       area_quadrature="projection_fan_adaptive",
                                       fan_order=28, fan_strips=8,
                                       fan_refinement_order=48, fan_refinement_strips=8,
                                       near_ratio=2.0, adaptive_budget=2e-4)
    reference = fan(point)
    value = adaptive(point)
    assert np.linalg.norm(value-reference) < 1e-4*np.linalg.norm(reference)
    assert adaptive.last_adaptive["near_faces"] >= 1
    assert adaptive.pop_adaptive_estimates() != []


def test_internal_gate_reports_estimates_and_refuses_external_refinement():
    xyz, faces, values, _gamma = sheet()
    field = RegularizedP1SheetField([(xyz, faces, values)], freestream=(1., 0., 0.),
                                    core_radius=0.01,
                                    area_quadrature="projection_fan_adaptive",
                                    fan_order=16, fan_strips=8,
                                    fan_refinement_order=24, fan_refinement_strips=8)
    assert field.quadrature_is_internal
    with pytest.raises(ValueError, match="gates internally"):
        field.quadrature_arguments(48)
    plan = field.quadrature_plan(48)
    assert plan["internal"] and plan["method"] == "projection_fan_adaptive"
    field.evaluate_many(np.array([[0.5, 0.5, 0.02], [1.5, 0.5, 0.02]]))
    estimates = field.pop_adaptive_estimates()
    assert len(estimates) == 2 and all(0.0 <= e < 1.0 for e in estimates)
    assert field.pop_adaptive_estimates() == []


def test_adaptive_scalar_and_batch_paths_agree():
    xyz, faces, values, _gamma = sheet()
    field = RegularizedP1SheetField([(xyz, faces, values)], freestream=(1., 0., 0.),
                                    core_radius=0.01,
                                    area_quadrature="projection_fan_adaptive",
                                    fan_order=16, fan_strips=8,
                                    fan_refinement_order=24, fan_refinement_strips=8)
    points = np.array([[0.5, 0.5, 0.02], [1.5, 0.4, 0.05], [1.0, 0.5, 0.5]])
    batch = field.evaluate_many(points)
    field.pop_adaptive_estimates()
    scalar = np.stack([field(p) for p in points])
    field.pop_adaptive_estimates()
    assert np.allclose(batch, scalar, rtol=0.0, atol=1e-15)

def test_subdivision_rung_triggers_and_equals_its_own_fan_aggregation():
    xyz, faces, values, _gamma = sheet()
    tri = xyz[faces[:1]]
    gamma = np.array([[0.1, 0.2, 0.3]])
    point = np.array([1.0/3.0, 1.0/3.0, 0.0])   # on the sheet plane: the near set
    _loose, without = adaptive_area_terms(tri, gamma, point, 0.01, budget=2e-4)
    terms, rung = adaptive_area_terms(tri, gamma, point, 0.01, budget=1e-14,
                                      max_escalations=0, subdivision_levels=2,
                                      subdivision_fan=(48, 8))
    assert without["subdivision_faces"] == 0 and without["subdivision_pieces"] == 0
    # max_escalations=0 keeps every near face over budget, so the rung must fire.
    assert rung["escalations"] == 0
    assert rung["subdivision_levels"] == 2
    assert rung["subdivision_faces"] == 1
    assert rung["subdivision_pieces"] == 4**2
    assert rung["subdivision_seconds"] > 0.0
    pieces = _subdivide(tri, 2)
    aggregate = area_terms(pieces, point, 0.01, order=96, strips=16, chunk=1024).sum(axis=0)
    assert np.allclose(terms[0], aggregate, rtol=0.0, atol=1e-18)
    assert np.isfinite(terms).all()


def test_subdivision_rung_rejects_invalid_configuration():
    xyz, faces, _values, _gamma = sheet()
    tri = xyz[faces[:1]]
    gamma = np.array([[0.1, 0.2, 0.3]])
    point = np.array([1.0/3.0, 1.0/3.0, 0.0])
    for kwargs in (dict(subdivision_levels=5), dict(subdivision_levels=1.5),
                   dict(subdivision_levels=2, subdivision_fan=(2, 8)),
                   dict(subdivision_levels=2, subdivision_fan=(48,))):
        with pytest.raises(ValueError):
            adaptive_area_terms(tri, gamma, point, 0.01, budget=1e-14, **kwargs)


def test_aggregate_estimate_cancels_where_the_per_face_sum_does_not():
    # Two coincident faces with opposite strength: every per-face difference is
    # nonzero, but the velocity they contribute is exactly zero, so the gate must
    # report no aggregate error while the per-face sum stays large.
    tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.stack([tri, tri])
    gamma = np.array([[0.2, -0.1, 0.05], [-0.2, 0.1, -0.05]])
    _terms, diagnostics = adaptive_area_terms(triangles, gamma,
                                              np.array([0.3, 0.3, 0.004]), 0.01,
                                              budget=2e-4)
    assert diagnostics["estimated_error"] > 0.0
    assert diagnostics["aggregate_estimated_error"] < 1e-12
    assert diagnostics["relative_estimate"] > 0.0
    assert diagnostics["aggregate_relative_estimate"] < 1e-12
    assert np.asarray(diagnostics["aggregate_vector"]).shape == (3,)
