"""Tests for the P1 wake sheet and Kutta rows (issue #49, priority d, G1 building blocks)."""

import numpy as np
import pytest

from panel.p1_wake import (
    dof_audit,
    jump_across_sheet,
    kutta_rows,
    mirror_nodes,
    sheet_potential,
    wake_transport_rows,
)


def rectangle(n_chord=2, n_span=2, chord=1.0, span=1.0):
    xs = np.linspace(0.0, chord, n_chord + 1)
    ys = np.linspace(-0.5 * span, 0.5 * span, n_span + 1)
    vertices = np.array([(x, y, 0.0) for x in xs for y in ys])
    index = lambda i, j: i * (n_span + 1) + j
    faces = []
    for i in range(n_chord):
        for j in range(n_span):
            a, b = index(i, j), index(i, j + 1)
            c, d = index(i + 1, j + 1), index(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return vertices, np.array(faces)


def test_uniform_density_gives_a_unit_potential_jump():
    vertices, faces = rectangle()
    result = jump_across_sheet(vertices, faces, np.ones(len(vertices)), heights=(1e-2, 1e-4, 1e-6))
    deficits = [float(np.max(np.abs(row["jump"] - 1.0))) for row in result["rows"]]
    assert deficits[0] > deficits[1] > deficits[2]
    # Floor: close to the surface the arctan2 branch loses absolute precision, so the
    # measured jump saturates a few 1e-6 away from the exact density.
    assert deficits[2] < 1e-5


def test_jump_equals_the_interpolated_density_not_its_average():
    vertices, faces = rectangle()
    density = vertices[:, 1] * 2.0 + 0.5
    result = jump_across_sheet(vertices, faces, density, heights=(1e-6,))
    centroids = result["centroids"]
    expected = 2.0 * centroids[:, 1] + 0.5
    assert np.allclose(result["rows"][0]["jump"], expected, atol=2e-6)


def test_jump_collapses_with_height_to_the_sheet_density():
    vertices, faces = rectangle()
    density = np.linspace(0.0, 1.0, len(vertices))
    result = jump_across_sheet(vertices, faces, density, heights=(1e-2, 1e-4, 1e-6))
    spreads = [float(np.ptp(row["jump"] - result["rows"][-1]["jump"])) for row in result["rows"]]
    assert spreads[0] > spreads[1] > spreads[2]
    assert spreads[2] < 1e-5


def test_far_field_decay_of_a_compact_sheet_is_quadrupole_like():
    vertices, faces = rectangle()
    density = np.ones(len(vertices))
    # Targets must leave the sheet plane: a constant-density sheet seen from a point in
    # its own plane subtends zero solid angle and the potential is identically zero.
    distances = np.array([10.0, 20.0, 40.0])
    targets = np.stack([distances, np.zeros(3), np.ones(3)]).T
    values = np.abs(sheet_potential(vertices, faces, density, targets))
    assert values[0] > values[1] > values[2]
    order = np.log(values[:-1] / values[1:]) / np.log(2.0)
    assert np.all(order > 1.5)


def test_mirroring_flips_the_doublet_sign():
    vertices, faces = rectangle()
    density = np.linspace(0.2, 0.9, len(vertices))
    mirrored, mirrored_faces = mirror_nodes(vertices, faces)
    targets = np.array([[0.3, 0.11, 0.4], [-0.2, -0.05, 0.9], [0.7, 0.2, 2.0]])
    plain = sheet_potential(vertices, faces, density, targets)
    mirrored_targets = targets.copy()
    mirrored_targets[:, 1] *= -1.0
    same = sheet_potential(mirrored, mirrored_faces, density, mirrored_targets)
    flipped = sheet_potential(mirrored, mirrored_faces, -density, mirrored_targets)
    assert np.allclose(same, -plain, atol=1e-12)
    assert np.allclose(flipped, plain, atol=1e-12)


def test_kutta_rows_are_one_per_station():
    rows = kutta_rows([1, 2, 3], [4, 5, 6], [7, 8, 9])
    assert len(rows["upper"]) == 3
    with pytest.raises(ValueError):
        kutta_rows([1, 2], [3], [4, 5])


def test_wake_transport_needs_matching_stations():
    transport = wake_transport_rows([[0, 1, 2], [3, 4, 5]])
    assert transport["n_stations"] == 2
    with pytest.raises(ValueError):
        wake_transport_rows([[0, 1]])
    with pytest.raises(ValueError):
        wake_transport_rows([[0, 1], [2]])


def test_dof_audit_of_a_closed_body_with_wake_is_square():
    audit = dof_audit(n_wall_nodes=100, n_wake_nodes=12, n_stations=4, n_wall_rows=100)
    assert audit["square"]
    assert audit["unknowns"] == 112
    assert audit["equations"] == 100 + 4 + 8
    with pytest.raises(ValueError):
        dof_audit(n_wall_nodes=100, n_wake_nodes=12, n_stations=4, n_wall_rows=90)


def test_sheet_potential_rejects_bad_shapes():
    vertices, faces = rectangle()
    with pytest.raises(ValueError):
        sheet_potential(vertices, faces, np.ones(len(vertices) + 1), np.array([[0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError):
        sheet_potential(vertices, faces, np.ones(len(vertices)), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        sheet_potential(vertices, faces + 99, np.ones(len(vertices)), np.array([[0.0, 0.0, 1.0]]))
