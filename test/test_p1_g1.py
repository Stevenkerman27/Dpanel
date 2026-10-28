"""Contract tests for the P1 G1 assembly (research only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panel.p1_g1 import (
    assemble_g1,
    hat_weights,
    load_case,
    solve_g1,
    triangle_centroids,
    wall_node_index,
)

MESH = Path("D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json")


@pytest.fixture(scope="module")
def case():
    if not MESH.exists():
        pytest.skip("wing/body half mesh is not available")
    return load_case(MESH)


@pytest.fixture(scope="module")
def system(case):
    return assemble_g1(case)


def test_hat_weights_partition_of_unity():
    knots = np.linspace(0.0, 2.0, 5)
    values = np.array([0.0, 0.3, 0.5, 1.2, 1.999, 2.0])
    weights = hat_weights(values, knots)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-15)
    outside = hat_weights(np.array([-0.1, 2.1]), knots)
    assert np.all(outside == 0.0)
    single = hat_weights(np.array([0.55]), knots)[0]
    at_knot = hat_weights(np.array([0.5]), knots)[0]
    assert at_knot[1] == 1.0
    assert np.count_nonzero(single) == 2


def test_wall_node_index_is_a_bijection_on_used_nodes():
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4]])
    nodes, local = wall_node_index(faces)
    assert np.array_equal(nodes, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(np.unique(local), np.arange(len(nodes)))
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    centroids = triangle_centroids(vertices, local)
    assert np.allclose(centroids[0], [1.0 / 3.0, 1.0 / 3.0, 0.0])


def test_constant_layer_identity_needs_the_mirror_block(system):
    own = system["wall_block"].sum(axis=1)
    mirror = system["mirror_block"].sum(axis=1)
    assert own.mean() < -0.5  # open half sheet: the identity cannot hold
    assert mirror.mean() > 0.0  # the mirrored layer enters with the opposite sign
    combined = own - mirror
    assert abs(float(combined.mean()) + 1.0) < 1e-12
    assert float(np.ptp(combined)) < 1e-12


def test_kutta_rows_are_unit_jumps_against_the_wake_basis(case, system):
    kutta = system["kutta"]
    assert kutta.shape[0] == len(case["te_pairs"])
    nodes = system["nodes"]
    lookup = {int(node): index for index, node in enumerate(nodes)}
    n_wall = len(nodes)
    for row, (upper, lower) in enumerate(case["te_pairs"]):
        assert kutta[row, lookup[int(upper)]] == 1.0
        assert kutta[row, lookup[int(lower)]] == -1.0
        assert abs(float(kutta[row, n_wall:].sum()) + 1.0) < 1e-12


def test_first_system_is_overdetermined_and_full_rank(system):
    matrix = system["matrix"]
    stacked = np.concatenate([matrix, system["kutta"]], axis=0)
    assert matrix.shape == (396, 239)
    assert stacked.shape == (440, 239)
    singular = np.linalg.svd(stacked, compute_uv=False)
    assert singular[-1] / singular[0] > 1e-6


def test_first_solution_is_non_circulatory(case):
    """The linear Kutta row is a definition: the least-squares answer has no circulation."""
    result = solve_g1(case)
    assert abs(result["cl_circulation"]) < 0.02
    assert abs(result["sum_gamma_dy"]) < 0.02
    assert 0.40 < result["cl_pressure"] < 0.45  # cambered closed-body value, not 0.531
    assert abs(result["row_sum_mean"] + 1.0) < 1e-12

