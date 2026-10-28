"""Contract tests for panel.wake_nodes (issue #65 Phase B1).

Every assertion is a measured property of the module, not a preference: the
tolerances are the values observed on the reference case, with room for
floating-point drift, except where an exact identity is claimed.
"""
import numpy as np
import pytest

from panel.wake_nodes import (StepGates, advance, boundary_nodes, face_geometry,
                             grid_sheet, node_velocity,
                             probe_velocity_face_centers, shape_gradients,
                             sheet_diagnostics, sheet_from_nodal)
from panel.wake_velocity import sheet_velocity_sum


def _flat_sheet(nx=3, ny=2):
    return grid_sheet(nx, ny, loading="linear", mu0=0.3, mu_y=0.2, camber=0.0)


def test_shape_gradients_reproduce_linear_field():
    s = _flat_sheet()
    grad = shape_gradients(s.vertices, s.faces)
    a = np.array([0.31, -0.22, 0.0])
    b = 0.11
    f = s.vertices @ a + b
    got = np.einsum("fa,fai->fi", f[s.faces], grad)
    assert np.allclose(got, np.broadcast_to(a, got.shape), atol=1e-12)
    # the gradients are in-plane, so a normal component cannot be represented
    assert np.allclose(grad @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-15)


def test_nodal_strength_gradient_matches_analytic():
    s = _flat_sheet()
    assert np.allclose(s.grad_face, np.broadcast_to([0.0, 0.2, 0.0], s.grad_face.shape),
                       atol=1e-12)
    assert np.allclose(s.mu_face, 0.3 + 0.2 * s.vertices[s.faces][:, :, 1].mean(axis=1),
                       atol=1e-12)


def test_pv_is_the_mean_of_the_two_one_sided_limits():
    s = _flat_sheet(2, 2)
    vel = node_velocity(s)
    assert np.allclose(vel.pv, 0.5 * (vel.plus + vel.minus), atol=1e-15)
    tri = s.triangles()
    normals = vel.nodal_normal
    for i in (0, 5, s.n_nodes - 1):
        h = vel.eps[i]
        up = sheet_velocity_sum(s.vertices[i] + h * normals[i], tri, s.mu_face,
                                s.grad_face) + s.freestream
        assert np.allclose(vel.plus[i], up, atol=1e-13)


def test_node_velocity_equals_freestream_when_strength_is_zero():
    s = grid_sheet(2, 2, mu0=0.0)
    vel = node_velocity(s)
    assert np.allclose(vel.pv, np.broadcast_to(s.freestream, vel.pv.shape), atol=1e-13)


def test_core_convergence_and_tip_divergence_are_ordered():
    s = grid_sheet(6, 6, mu0=0.3)
    vel = node_velocity(s)
    n = s.n_faces and s.n_nodes
    i = np.arange(n) % 7
    eta = np.abs(s.vertices[:, 1])
    core = (i > 0) & (i < 6) & (eta < 0.8)
    tip = eta >= 0.8
    conv = vel.convergence_array
    assert np.median(conv[core]) < 5e-2
    assert np.median(conv[tip]) > np.median(conv[core])


def test_probe_estimator_is_not_the_same_operator():
    s = grid_sheet(6, 6, mu0=0.3)
    vel = node_velocity(s)
    probe = probe_velocity_face_centers(s)
    pv_face = vel.pv[s.faces].mean(axis=1)
    rel = np.max(np.linalg.norm(probe - pv_face, axis=1)) / np.linalg.norm(s.freestream)
    assert rel > 0.05


def test_roots_are_never_moved():
    s = grid_sheet(3, 2, mu0=0.3)
    vel = node_velocity(s)
    res = advance(s, vel.pv, 0.05)
    assert res.reasons["root"] == len(s.roots)
    assert np.allclose(res.sheet.vertices[s.roots], s.vertices[s.roots], atol=0.0)
    assert not res.accepted[s.roots].any()


def test_speed_gate_refuses_stagnant_nodes():
    s = grid_sheet(3, 2, mu0=0.3)
    res = advance(s, np.zeros((s.n_nodes, 3)), 0.05)
    assert res.reasons["speed"] == s.n_nodes - len(s.roots)
    assert np.allclose(res.sheet.vertices, s.vertices, atol=0.0)


def test_step_clamp_limits_displacement():
    s = grid_sheet(3, 2, mu0=0.3)
    vel = node_velocity(s)
    edge = np.median(face_geometry(s.vertices, s.faces)[3].ravel())
    res = advance(s, vel.pv, 10.0, gates=StepGates(max_step_fraction=0.5))
    assert res.stats["step_max"] <= 0.5 * edge + 1e-9
    assert res.reasons["step"] > 0


def test_area_gate_rejects_and_restores():
    s = grid_sheet(3, 2, mu0=0.3)
    vel = np.zeros((s.n_nodes, 3))
    vel[:, 2] = 0.0
    vel[:, 1] = -1.0
    res = advance(s, vel, 0.4, gates=StepGates(max_area_ratio=1.05,
                                               max_step_fraction=1e9,
                                               check_intersections=False))
    assert res.reasons["area"] > 0
    rejected = ~res.accepted
    assert np.allclose(res.sheet.vertices[rejected], s.vertices[rejected], atol=0.0)


def test_wall_gate_refuses_a_step_through_the_wall():
    s = grid_sheet(2, 2, mu0=0.3)
    wall = np.array([[[-5.0, -5.0, 0.5], [5.0, -5.0, 0.5], [5.0, 5.0, 0.5]],
                     [[-5.0, -5.0, 0.5], [5.0, 5.0, 0.5], [-5.0, 5.0, 0.5]]])
    vel = np.zeros((s.n_nodes, 3))
    vel[:, 2] = 1.0
    res = advance(s, vel, 1.0, gates=StepGates(wall_triangles=wall,
                                               max_step_fraction=1e9,
                                               max_area_ratio=1e9,
                                               check_intersections=False))
    assert res.reasons["wall"] > 0
    assert not res.accepted.any()


def test_intersection_gate_refuses_a_self_crossing_step():
    v = np.array([
        [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 2.0],
        [0.0, 1.0, 1.0], [2.0, 1.0, 1.0], [0.5, 1.4, 1.4],
    ])
    f = np.array([[0, 1, 2], [0, 1, 4], [0, 4, 3], [3, 4, 5]])
    mu = np.linspace(0.4, 0.0, len(v))
    s = sheet_from_nodal(v, f, mu, roots=(0, 1), freestream=(1.0, 0.0, 0.0))
    vel = np.zeros((s.n_nodes, 3))
    vel[5] = (0.0, -1.0, 0.0)
    res = advance(s, vel, 2.5, gates=StepGates(max_area_ratio=1e9, max_fold_deg=180.0,
                                               max_step_fraction=1e9,
                                               intersection_pairs_limit=10 ** 6))
    assert res.reasons["intersection"] == 1
    assert not res.accepted[5]


def test_boundary_nodes_flags_the_sheet_edge():
    s = grid_sheet(2, 2)
    mask = boundary_nodes(s)
    assert mask.sum() == 8
    assert mask[0] and mask[-1]


def test_dcp_identity_and_no_normal_jump():
    s = grid_sheet(5, 4, mu0=0.3)
    vel = node_velocity(s)
    diag = sheet_diagnostics(s, vel, bands=2)
    assert diag["dcp_identity_residual"] < 1e-10
    assert diag["overall"]["normal_jump_max"] < 1e-12


def test_diagnostics_bands_cover_the_sheet():
    s = grid_sheet(5, 4, mu0=0.3)
    diag = sheet_diagnostics(s, node_velocity(s), bands=3)
    assert sum(b["panels"] for b in diag["bands"]) == s.n_faces
    assert diag["overall"]["panels"] == s.n_faces
    assert diag["quality"]["aspect_p90"] >= 1.0


def test_advance_modes_share_direction_and_differ_in_speed():
    s = grid_sheet(3, 2, mu0=0.4)
    vel = node_velocity(s)
    arc = advance(s, vel.pv, 0.05, mode="arc_length")
    pt = advance(s, vel.pv, 0.05, mode="pseudo_time")
    moved = arc.accepted & pt.accepted
    assert moved.sum() > 0
    d_arc = arc.sheet.vertices[moved] - s.vertices[moved]
    d_pt = pt.sheet.vertices[moved] - s.vertices[moved]
    cos = np.einsum("ij,ij->i", d_arc, d_pt) / (
        np.linalg.norm(d_arc, axis=1) * np.linalg.norm(d_pt, axis=1))
    assert np.degrees(np.max(np.arccos(np.clip(cos, -1, 1)))) < 1e-5
    assert pt.stats["step_max"] != arc.stats["step_max"]


def test_eps_convergence_gate_reports_and_refuses():
    s = grid_sheet(4, 3, mu0=0.3)
    vel = node_velocity(s)
    gates = StepGates(max_eps_convergence=1e-9, max_step_fraction=1e9)
    res = advance(s, vel.pv, 0.03, gates=gates, eps_convergence=vel.convergence_array)
    assert res.reasons["convergence"] > 0
    assert np.allclose(res.sheet.vertices, s.vertices, atol=0.0)
