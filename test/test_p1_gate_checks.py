"""Gate checks for the P1 doublet kernel (issue #49 priority d)."""
import numpy as np
import pytest

from panel.p1_doublet_moment import triangle_p1_doublet_potentials
from panel.p1_gate_checks import (
    near_plane_gate,
    node_dof_audit,
    polar_doublet_quadrature,
    self_trace_gate,
    solid_angle_fraction,
    assembled_free_terms,
)


TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_near_plane_gate_passes_at_one_em12():
    report = near_plane_gate(TRIANGLE)
    assert report["passed"] is True
    assert report["worst_relative_error"] < 1e-12
    ratios = {row["ratio"] for row in report["rows"]}
    assert min(ratios) <= 1e-4 and 1e-2 in ratios


def test_near_plane_gate_matches_oracle_row_by_row():
    report = near_plane_gate(TRIANGLE, ratios=(1e-3, 1e-2), offsets=((0.0, 0.0),))
    for row in report["rows"]:
        assert row["relative_error"] < 1e-12
        assert np.allclose(row["analytic"], row["quadrature"], rtol=0.0, atol=1e-13)


def test_quadrature_rejects_on_plane_and_outside_targets():
    with pytest.raises(ValueError, match="panel plane"):
        polar_doublet_quadrature(TRIANGLE.mean(axis=0), TRIANGLE)
    with pytest.raises(ValueError, match="outside the triangle"):
        polar_doublet_quadrature(np.array([3.0, 3.0, 1e-3]), TRIANGLE)
    with pytest.raises(ValueError, match="Degenerate triangle"):
        polar_doublet_quadrature(np.array([0.0, 0.0, 1e-3]), np.zeros((3, 3)))


def test_solid_angle_fraction_geometric_free_terms():
    assert solid_angle_fraction(TRIANGLE, TRIANGLE.mean(axis=0)) == 0.5
    assert solid_angle_fraction(TRIANGLE, 0.5 * (TRIANGLE[0] + TRIANGLE[1])) == 0.25
    corner = solid_angle_fraction(TRIANGLE, TRIANGLE[0])
    assert abs(corner - 0.125) < 1e-12


def test_self_trace_interior_coefficient_matches_one_half():
    report = self_trace_gate(TRIANGLE, point="centroid")
    assert report["predicted_coefficient"] == 0.5
    assert abs(report["extrapolated_coefficient"] - 0.5) < 1e-6
    assert abs(report["measured_coefficient"] - 0.5) < 1e-5
    assert report["principal_value_vanishes"] is True
    assert report["oracle_coefficient_error"] == 0.0


def test_self_trace_vertex_coefficient_is_the_corner_angle():
    report = self_trace_gate(TRIANGLE, point="vertex:0")
    assert abs(report["predicted_coefficient"] - 0.125) < 1e-12
    assert abs(report["extrapolated_coefficient"] - 0.125) < 1e-5
    assert abs(report["measured_coefficient"] - 0.125) < 1e-4
    assert abs(report["oracle_coefficient_error"] - 0.375) < 1e-12


def test_self_trace_edge_coefficient_is_one_quarter():
    report = self_trace_gate(TRIANGLE, point="edge")
    assert report["predicted_coefficient"] == 0.25
    assert abs(report["extrapolated_coefficient"] - 0.25) < 1e-5
    assert abs(report["measured_coefficient"] - 0.25) < 1e-4
    assert abs(report["oracle_coefficient_error"] - 0.25) < 1e-12


def test_self_trace_oracle_self_is_the_minus_half_jump():
    oracle = triangle_p1_doublet_potentials(
        TRIANGLE[2][None, :], TRIANGLE[None, :, :], self_ids=np.array([0]))[0, 0]
    assert np.allclose(oracle, np.array([0.0, 0.0, -0.5]))
    report = self_trace_gate(TRIANGLE, point="vertex:2")
    assert np.allclose(report["oracle_self"], oracle)

def test_assembled_free_terms_flat_pair_is_not_uniform():
    verts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    report = assembled_free_terms(verts, faces)
    assert report["n_non_smooth"] >= 1
    assert report["max"] <= 0.5 + 1e-12


def test_assembled_free_terms_closed_cube_stays_below_one_half():
    cube = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris.append([a, b, c])
        tris.append([a, c, d])
    report = assembled_free_terms(cube, np.array(tris))
    assert report["n_non_smooth"] > 0
    assert report["max"] <= 0.5 + 1e-12


def test_node_audit_detects_split_shared_nodes():
    verts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    shared = node_dof_audit(verts, faces)
    assert shared["n_unique_vertices"] == 4
    assert shared["n_unknowns_raw"] == 4
    assert shared["n_boundary_edges"] == 4
    assert shared["gauge_required"] is True
    assert shared["split_nodes_present"] is False
    split_verts = np.vstack([verts, verts[[1, 2]] + 1e-12])
    split_faces = np.array([[0, 1, 2], [4, 3, 5]])
    split = node_dof_audit(split_verts, split_faces)
    assert split["n_unknowns_raw"] == 6
    assert split["n_unique_vertices"] == 4
    assert split["n_duplicated_groups"] == 2
    assert split["n_boundary_edges_raw"] == 6
    assert split["n_boundary_edges"] == 4
    assert split["split_nodes_present"] is True


def test_node_audit_closed_body_needs_no_gauge():
    verts = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0],
                      [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    report = node_dof_audit(verts, faces)
    assert report["closed"] is True
    assert report["gauge_required"] is False
    assert report["n_boundary_edges"] == 0
    assert report["nonmanifold_edges"] == []


def test_node_audit_rejects_bad_inputs():
    good = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    with pytest.raises(ValueError, match="outside the array"):
        node_dof_audit(good, np.array([[0, 1, 9]]))
    with pytest.raises(ValueError, match="non-empty"):
        node_dof_audit(good, np.zeros((0, 3), dtype=int))
    with pytest.raises(ValueError, match="tolerance"):
        node_dof_audit(good, np.array([[0, 1, 2]]), tolerance=0.0)