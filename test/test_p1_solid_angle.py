"""Contract tests for 3D solid-angle free terms (issue #56, research only)."""

from __future__ import annotations

import numpy as np
import pytest

from panel.p1_solid_angle import (
    assembled_row_sum,
    cone_solid_angle,
    dihedral_angles,
    flat_sheet,
    icosphere,
    incident_fan_sum,
    outward_orient,
    sector_angle_free_term,
    tetrahedron,
    triangle_solid_angle,
)

FOUR_PI = 4.0 * np.pi


def test_triangle_vertex_and_edge_oracles():
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        triangle_solid_angle(triangle, triangle[0])
    edge = abs(triangle_solid_angle(triangle, np.array([0.5, 0.0, 1e-6])))
    assert abs(edge / FOUR_PI - 0.25) < 1e-5
    vertex = abs(triangle_solid_angle(triangle, np.array([0.0, 0.0, 1e-6])))
    assert abs(vertex / FOUR_PI - 0.125) < 1e-5  # right angle / 4 pi
    assert abs(vertex - np.pi / 2.0) < 1e-4


def test_outward_orient_flips_inward_faces():
    vertices, faces = tetrahedron()
    flipped = outward_orient(vertices, faces)
    centre = vertices.mean(axis=0)
    for face in flipped:
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        assert float(np.dot(normal, triangle.mean(axis=0) - centre)) > 0.0


def test_orthogonal_corner_coefficients_are_one_eighth_and_seven_eighths():
    vertices, faces = tetrahedron()
    faces = outward_orient(vertices, faces)
    angles = dihedral_angles(vertices, faces, 0)
    assert np.allclose(angles, np.pi / 2.0, atol=1e-12)
    corner = cone_solid_angle(vertices, faces, 0)
    assert abs(corner["omega"] - np.pi / 2.0) < 1e-12
    assert abs(corner["interior_coefficient"] - 0.125) < 1e-12
    assert abs(corner["exterior_coefficient"] - 0.875) < 1e-12


def test_sector_rule_disagrees_with_the_corner_coefficient():
    vertices, faces = tetrahedron()
    faces = outward_orient(vertices, faces)
    sector = sector_angle_free_term(vertices, faces, 0)
    exterior = cone_solid_angle(vertices, faces, 0)["exterior_coefficient"]
    assert abs(sector - 0.375) < 1e-12
    assert abs(sector - exterior) > 0.49  # 0.375 vs 0.875: a 133 percent error


def test_all_panel_sum_is_the_constant_density_identity():
    vertices, faces = tetrahedron()
    faces = outward_orient(vertices, faces)
    assert abs(assembled_row_sum(vertices, faces, 0, side="interior") - 1.0) < 1e-6
    assert abs(assembled_row_sum(vertices, faces, 0, side="exterior")) < 1e-6
    level = icosphere(2)
    nodes = (0, 5, 17)
    for node in nodes:
        assert abs(assembled_row_sum(level[0], level[1], node, side="interior") - 1.0) < 1e-6
        assert abs(assembled_row_sum(level[0], level[1], node, side="exterior")) < 1e-6


def test_incident_fan_sum_is_not_the_corner_coefficient():
    """A generic (non-orthogonal) corner: the fan sum misses the coefficient by 0.9."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.35, 1.1, 0.0], [0.2, 0.25, 0.9]]
    )
    faces = outward_orient(vertices, np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    corner = cone_solid_angle(vertices, faces, 0)
    fan = incident_fan_sum(vertices, faces, 0, side="exterior")
    assert abs(fan - corner["exterior_coefficient"]) > 0.5
    # the fan sum is still a valid signed solid-angle sum, just not a free term
    assert abs(fan) < 0.2


def test_smooth_sphere_sector_sum_is_not_the_flat_limit():
    """On a triangulated sphere both incident-fan rules are biased; only the identity holds."""
    vertices, faces = icosphere(2)
    sector = np.array([sector_angle_free_term(vertices, faces, int(n)) for n in range(8)])
    assert np.all(np.abs(sector - 0.5) < 0.02)
    fan = np.array([incident_fan_sum(vertices, faces, int(n)) for n in range(8)])
    assert np.all(np.abs(fan) < 0.5)
    # signed fan sums carry opposite signs on different faces of the same node ring
    assert np.all(np.abs(np.abs(fan) - 0.42) < 0.05)


def test_open_sheet_has_no_constant_density_identity():
    vertices, faces = flat_sheet()
    shared = 1
    assert abs(sector_angle_free_term(vertices, faces, shared) - 0.125) < 1e-12
    assert abs(incident_fan_sum(vertices, faces, shared) + 0.125) < 1e-6
    value = assembled_row_sum(vertices, faces, shared)
    assert abs(value - 0.125) < 1e-6  # not 1: the sheet does not enclose the probe

