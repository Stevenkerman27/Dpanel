"""Rank and gauge evidence for the P1 doublet operators (issue #56, research only).

This module does not assemble physics: it takes an operator that was already built and
answers three questions numerically, because the previous claim ("an open surface has a
boundary edge, therefore a gauge is required") was not derived from any operator.

  1. What is the numerical rank of the matrix, and at which singular value does it drop?
  2. Does the constant density vector lie in the null space (``||A 1||``)?
  3. What does an explicit constraint do: leave the fit unchanged (a gauge) or worsen it
     (a change of the discrete problem)?
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "operator_stats",
    "constraint_effect",
    "flat_sheet",
    "tetrahedron",
    "topology_report",
]


def operator_stats(matrix, *, rank_tolerance=None):
    """Singular-value summary plus the constant-density response of ``matrix``."""
    operator = np.asarray(matrix, dtype=float)
    if operator.ndim != 2 or not operator.size:
        raise ValueError("matrix must be a non-empty 2D array")
    if not np.isfinite(operator).all():
        raise ValueError("matrix must be finite")
    singular = np.linalg.svd(operator, compute_uv=False)
    scale = float(singular[0]) if singular.size else 0.0
    if rank_tolerance is None:
        rank_tolerance = float(max(operator.shape)) * np.finfo(float).eps * max(scale, 1.0)
    rank = int(np.count_nonzero(singular > rank_tolerance))
    n_rows, n_cols = operator.shape
    ones = np.ones(n_cols)
    response = operator @ ones
    norm_inf = float(np.abs(operator).sum(axis=1).max())
    row_sums = operator.sum(axis=1)
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "sigma_max": float(scale),
        "sigma_min": float(singular[-1]) if singular.size else 0.0,
        "condition_number": float(scale / singular[-1]) if singular.size and singular[-1] > 0 else float("inf"),
        "rank": rank,
        "nullity": int(n_cols - rank),
        "rank_tolerance": float(rank_tolerance),
        "constant_response": float(np.linalg.norm(response)),
        "constant_response_relative": float(np.linalg.norm(response) / max(norm_inf, 1e-300)),
        "constant_is_null_vector": bool(np.linalg.norm(response) <= 1e-8 * max(norm_inf, 1.0)),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_spread": float(np.ptp(row_sums)),
        "overdetermined": bool(n_rows > n_cols),
        "square": bool(n_rows == n_cols),
    }


def constraint_effect(matrix, rhs, index, *, value=0.0):
    """Least-squares fit with and without pinning one unknown to ``value``."""
    operator = np.asarray(matrix, dtype=float)
    target = np.asarray(rhs, dtype=float)
    if operator.ndim != 2 or target.shape != (operator.shape[0],):
        raise ValueError("rhs must have one entry per matrix row")
    if not 0 <= int(index) < operator.shape[1]:
        raise ValueError("index outside the unknown range")
    free = np.linalg.lstsq(operator, target, rcond=None)
    residual_free = float(np.linalg.norm(operator @ free[0] - target))
    keep = [column for column in range(operator.shape[1]) if column != int(index)]
    pin = float(value)
    reduced = operator[:, keep]
    shifted = target - pin * operator[:, int(index)]
    pinned = np.linalg.lstsq(reduced, shifted, rcond=None)
    residual_pinned = float(np.linalg.norm(reduced @ pinned[0] - shifted))
    singular_free = np.linalg.svd(operator, compute_uv=False)
    singular_pinned = np.linalg.svd(reduced, compute_uv=False)
    return {
        "index": int(index),
        "value": pin,
        "residual_free": residual_free,
        "residual_pinned": residual_pinned,
        "residual_change": float(residual_pinned - residual_free),
        "residual_change_relative": float((residual_pinned - residual_free) / max(residual_free, 1e-300)),
        "sigma_min_free": float(singular_free[-1]),
        "sigma_min_pinned": float(singular_pinned[-1]) if singular_pinned.size else 0.0,
        "is_gauge": bool(abs(residual_pinned - residual_free) <= 1e-10 * max(residual_free, 1.0)),
    }


def flat_sheet():
    """Two unit triangles sharing an edge: an open, flat, non-lifting sheet."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    return vertices, faces


def tetrahedron():
    """Unit right tetrahedron with outward winding."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def topology_report():
    """Rank and gauge evidence for the topologies that issue #56 asks about."""
    from panel.p1_dirichlet import assemble_p1_doublet, icosphere

    cases = {}
    for level in (1, 2):
        vertices, faces = icosphere(level)
        matrix = assemble_p1_doublet(vertices, faces)["matrix"]
        cases["icosphere_%d" % level] = operator_stats(matrix)
    vertices, faces = tetrahedron()
    cases["tetrahedron"] = operator_stats(assemble_p1_doublet(vertices, faces)["matrix"])
    vertices, faces = flat_sheet()
    sheet = assemble_p1_doublet(vertices, faces)["matrix"]
    cases["open_sheet"] = operator_stats(sheet)
    return cases
