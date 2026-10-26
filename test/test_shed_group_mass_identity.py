"""Pure-NumPy regression for the circulation mass row of the shed-group v2 contract.

The v2 contract (docs/ISSUE39_SHED_GROUPS_V2.md, section 4) defines

    mass[i, j]     = sum_{e,q} w_eq * B_i(y_eq) * B_j(y_eq)
    integrals[e,i] = sum_q     w_eq * B_i(y_eq)
    measure[i]     = sum_e integrals[e,i]
    M              = mass / measure[:, None]

A draft of the contract wrote M = (integrals.T @ integrals) / measure, which
multiplies two separately summed basis integrals instead of the product of the
two basis functions under the quadrature weight. On a non-uniform trailing edge
the two differ by 78%, and for ng=1 by a factor measure, so this file pins the
correct algebra without importing JAX or any solver code.
"""

from __future__ import annotations

import numpy as np

GAUSS = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])

NONUNIFORM_Y = np.array([0.05, 0.15, 0.35, 0.70, 0.95])
NONUNIFORM_L = np.array([0.20, 0.05, 0.30, 0.10, 0.25])


def _basis(y):
    """Piecewise-linear P1 basis (1-y, y) evaluated at the quadrature points."""
    return np.stack([1.0 - y, y], axis=-1)


def _constant_basis(y):
    return np.ones(y.shape + (1,))


def _quadrature(edges_y, lengths, basis):
    yq = np.stack([edges_y * (1.0 - g) + edges_y * g for g in GAUSS], axis=1)
    w = np.broadcast_to((lengths / 2.0)[:, None], yq.shape)
    b = basis(yq)
    mass = np.einsum("eq,eqi,eqj->ij", w, b, b)
    integrals = np.einsum("eq,eqi->ei", w, b)
    measure = integrals.sum(axis=0)
    return dict(w=w, b=b, mass=mass, integrals=integrals, measure=measure)


def test_correct_mass_rows_sum_to_one():
    for edges_y, lengths, basis in (
        (NONUNIFORM_Y, NONUNIFORM_L, _basis),
        (np.array([0.25, 0.75]), np.array([0.5, 0.5]), _basis),
    ):
        qy = _quadrature(edges_y, lengths, basis)
        m = qy["mass"] / qy["measure"][:, None]
        # partition of unity: sum_i B_i = 1 -> every row of M integrates to measure_i
        assert np.allclose(m.sum(axis=1), 1.0, rtol=0.0, atol=1e-15), m.sum(axis=1)


def test_documented_nonuniform_counterexample():
    qy = _quadrature(NONUNIFORM_Y, NONUNIFORM_L, _basis)
    assert np.allclose(qy["measure"], [0.47, 0.43], rtol=0.0, atol=1e-15)
    correct = qy["mass"] / qy["measure"][:, None]
    assert np.allclose(
        correct, [[0.75106383, 0.24893617], [0.27209302, 0.72790698]], rtol=0.0, atol=1e-8
    )
    naive = (qy["integrals"].T @ qy["integrals"]) / qy["measure"][:, None]
    assert np.allclose(
        naive, [[0.16380319, 0.05906915], [0.06456395, 0.16857558]], rtol=0.0, atol=1e-8
    )
    gap = np.abs(correct - naive).max()
    assert abs(gap - 0.5872606382978723) < 1e-9
    assert gap > 0.5 * np.abs(correct).max()


def test_single_constant_basis_is_a_unit_scale_error_only():
    qy = _quadrature(np.array([0.10, 0.60]), np.array([0.5, 0.3]), _constant_basis)
    assert np.allclose(qy["measure"], [0.8], rtol=0.0, atol=1e-15)
    correct = qy["mass"] / qy["measure"][:, None]
    naive = (qy["integrals"].T @ qy["integrals"]) / qy["measure"][:, None]
    assert np.allclose(correct, [[1.0]], rtol=0.0, atol=1e-15)
    assert np.allclose(naive, [[0.425]], rtol=0.0, atol=1e-15)


def test_weak_kutta_row_uses_the_same_per_dof_measure():
    qy = _quadrature(NONUNIFORM_Y, NONUNIFORM_L, _basis)
    edges = NONUNIFORM_Y.size
    incidence = np.zeros((edges, 2))
    incidence[0, 0] = 1.0
    incidence[0, 1] = -1.0
    t = (qy["integrals"].T @ incidence) / qy["measure"][:, None]
    # only edge 0 carries a jump: integrate B_i over that edge's two Gauss points
    y_edge = np.full(GAUSS.size, NONUNIFORM_Y[0])
    expected = ((NONUNIFORM_L[0] / 2.0) * _basis(y_edge)).sum(axis=0) / qy["measure"]
    assert np.allclose(t[:, 0], expected, rtol=0.0, atol=1e-15)
    assert np.allclose(t[:, 1], -expected, rtol=0.0, atol=1e-15)
