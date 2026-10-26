"""Reference audit: statistics, residual reconstruction and the sphere anchor."""
import numpy as np
import pytest

from panel.analytic_sphere import (
    audit_sphere, exact_cp, icosphere, sphere_solution,
)
from panel.audit_reference import _describe, linear_residual


def test_describe_reports_signed_and_absolute_statistics():
    stats = _describe(np.array([-3.0, -1.0, 1.0, 3.0]))
    assert stats["rms"] == pytest.approx(np.sqrt(5.0))
    assert stats["mean"] == 0.0
    assert stats["median_abs"] == 2.0
    assert stats["max_abs"] == 3.0
    assert 2.0 < stats["p90_abs"] <= 3.0


def test_linear_residual_vanishes_on_a_consistent_system():
    matrix = np.array([[2.0, 1.0], [1.0, 4.0]])
    state = np.array([1.0, -2.0])
    result = {"matrix_half": matrix, "rhs_half": matrix @ state,
              "mu_half": state[:1], "gamma_half": state[1:]}
    residual = linear_residual(result)
    assert residual["unknowns"] == 2
    assert residual["absolute"] < 1e-15
    assert residual["relative"] < 1e-15
    result["rhs_half"] = result["rhs_half"] + np.array([1.0, 0.0])
    assert linear_residual(result)["relative"] > 1e-3


def test_exact_sphere_pressure_at_stagnation_and_equator():
    flow = (1.0, 0.0, 0.0)
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cp = exact_cp(points, freestream=flow, radius=1.0)
    assert cp[0] == pytest.approx(1.0)
    assert cp[1] == pytest.approx(1.0)
    assert cp[2] == pytest.approx(-1.25)


def test_icosphere_is_watertight_and_monotone():
    counts = []
    for subdivisions in (1, 2, 3):
        vertices, faces = icosphere(subdivisions)
        counts.append(len(faces))
        assert len(faces) == 20 * 4 ** subdivisions
        triangles = vertices[faces]
        area = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0]), axis=1)
        assert area.min() > 0.0
        assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)
        # Closed surface: the area-weighted normals cancel to round-off.
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0]) / 2
        assert np.linalg.norm(normals.sum(axis=0)) < 1e-12
    assert counts[0] < counts[1] < counts[2]


def test_sphere_solution_has_zero_net_source():
    vertices, faces = icosphere(2)
    state = sphere_solution(faces, vertices, (1.0, 0.0, 0.0))
    assert abs(state["source_total"]) < 1e-12
    assert np.all(np.isfinite(state["mu"]))


def test_sphere_audit_converges_away_from_the_surface():
    errors = []
    for subdivisions in (1, 2):
        vertices, faces = icosphere(subdivisions)
        audit = audit_sphere(faces, vertices)
        far = audit["samples"]["0.2"]["cp_rms_error"]
        near = audit["samples"]["0.01"]["cp_rms_error"]
        assert far < near
        errors.append(far)
        # A closed body in potential flow carries no net force, and the
        # discrete pressure integral cancels far below the per-face scale.
        surface = audit["surface"]
        assert abs(surface["drag_over_qA"]) < 1e-6
        assert surface["sum_abs_face_drag_over_qA"] > 0.1
    assert errors[1] < errors[0]