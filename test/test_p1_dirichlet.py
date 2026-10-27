"""Tests for the G0 P1 doublet Dirichlet solve (issue #49, priority d)."""

import numpy as np
import pytest

from panel.p1_dirichlet import (
    assemble_p1_doublet,
    face_cp_from_mu,
    icosphere,
    panel_doublet_coefficients,
    self_free_row,
    solve_p1_dirichlet,
    sphere_case,
)
from panel.p1_doublet_moment import triangle_p1_doublet_potentials


def test_icosphere_sizes_and_radius():
    for level, (n_vertices, n_faces) in enumerate([(12, 20), (42, 80), (162, 320), (642, 1280)]):
        vertices, faces = icosphere(level)
        assert len(vertices) == n_vertices
        assert len(faces) == n_faces
        assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12)


def test_panel_coefficients_match_the_oracle():
    vertices, faces = icosphere(2)
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    rows = [0, 1, 5, 17, 64, 200]
    for target_row in rows:
        for panel in (3, 21, 77, 150, 301):
            if target_row in faces[panel]:
                continue
            mine = panel_doublet_coefficients(centroids[target_row][None, :], triangles[panel])[0]
            reference = triangle_p1_doublet_potentials(
                centroids[target_row][None, :], triangles[panel][None, :])[0, 0]
            assert np.allclose(mine, reference, atol=1e-12, rtol=0.0)


def test_self_free_term_is_the_minus_half_jump():
    row = self_free_row()
    assert row.shape == (3,)
    assert abs(row.sum() + 0.5) < 1e-15
    assert np.allclose(row, -1.0 / 6.0, atol=0.0)


def test_on_plane_kernel_evaluation_is_the_degenerate_half_jump():
    # Exactly at the centroid the arctan2 branch is ill-conditioned: the kernel
    # returns a half jump whose sign is decided by roundoff, which is exactly why
    # the assembly replaces that row with self_free_row().
    vertices, faces = icosphere(1)
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    for panel in (0, 7, 19, 55):
        coefficients = panel_doublet_coefficients(centroids[panel][None, :], triangles[panel])[0]
        assert abs(abs(coefficients.sum()) - 0.5) < 1e-8


@pytest.mark.parametrize("level", [1, 2, 3])
def test_closed_sphere_row_sums_are_minus_one(level):
    vertices, faces = icosphere(level)
    state = assemble_p1_doublet(vertices, faces)
    row_sums = state["row_sums"]
    assert abs(row_sums.mean() + 1.0) < 1e-12
    assert np.ptp(row_sums) < 1e-12


def test_matrix_entry_uses_the_self_free_term():
    vertices, faces = icosphere(1)
    state = assemble_p1_doublet(vertices, faces)
    matrix = state["matrix"]
    diagonal = np.array([matrix[j, faces[j][0]] for j in range(len(faces))])
    assert np.all(np.abs(diagonal) > 0.0)
    assert np.all(np.isfinite(matrix))


def test_sphere_mu_amplitude_converges_to_three_halves():
    alphas = [sphere_case(level)["mu_alpha"] for level in (1, 2, 3)]
    assert alphas[0] > alphas[1] > alphas[2]
    assert abs(alphas[2] - 1.5) < 2e-3
    assert all(alpha > 1.5 for alpha in alphas)


def test_sphere_cp_errors_decrease_with_refinement():
    reports = [sphere_case(level) for level in (1, 2, 3)]
    rms = [report["cp_rms"] for report in reports]
    assert rms[0] > rms[1] > rms[2]
    assert rms[2] < 0.02
    nose = [report["cp_nose"] for report in reports]
    assert all(0.6 < value <= 1.0 for value in nose)
    assert nose[0] < nose[1] < nose[2]
    assert abs(reports[2]["cp_nose"] - 1.0) < 0.03
    assert abs(reports[2]["cp_tail"] - reports[2]["cp_nose"]) < 1e-9


def test_sphere_residual_decreases_with_refinement():
    residuals = [sphere_case(level)["least_squares_residual"] for level in (1, 2, 3)]
    assert residuals[0] > residuals[1] > residuals[2]


def test_source_included_variant_is_rejected():
    # A single layer with density n.U_inf already carries one third of the
    # freestream potential on a sphere; subtracting it from the RHS rescales the
    # boundary value, so the doublet amplitude moves from 1.5 to 4/3 * 1.5 = 2 and
    # the pressures become O(1) wrong.  The source term is redundant for a closed
    # body and is not part of the validated formulation.
    report = sphere_case(2, sigma_sign=1.0)
    assert abs(report["mu_alpha"] - 2.0) < 0.05
    assert report["cp_rms"] > 0.5


def test_face_cp_uses_the_tangential_gradient():
    # The P1 gradient of a linear field is exact but tangential: the normal
    # component is lost, so the expected Cp is built from the projection.
    vertices, faces = icosphere(1)
    field = np.array([0.3, -0.2, 0.1])
    mu = vertices @ field
    cp = face_cp_from_mu(vertices, faces, mu, speed=1.0)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normal_part = normals @ field
    expected = 1.0 - (field @ field - normal_part * normal_part)
    assert np.allclose(cp, expected, atol=1e-12)


def test_invalid_inputs_are_rejected():
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        panel_doublet_coefficients(np.array([[0.0, 0.0, 1.0]]), flat)
    vertices, faces = icosphere(1)
    with pytest.raises(ValueError):
        solve_p1_dirichlet(vertices, faces, np.array([1.0, 0.0, 0.0]), sigma_sign=2.0)
    with pytest.raises(ValueError):
        face_cp_from_mu(vertices, faces, np.zeros(len(vertices) + 1))
    with pytest.raises(ValueError):
        assemble_p1_doublet(vertices, faces + 1)
