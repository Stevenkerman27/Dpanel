"""Rank and gauge contract for the P1 doublet operators (issue #56, part C)."""

from __future__ import annotations

import numpy as np
import pytest

from panel.p1_dirichlet import assemble_p1_doublet, icosphere
from panel.p1_rank_gauge import (
    constraint_effect,
    flat_sheet,
    operator_stats,
    tetrahedron,
    topology_report,
)


def test_closed_sphere_operator_has_full_column_rank():
    for level in (1, 2):
        vertices, faces = icosphere(level)
        stats = operator_stats(assemble_p1_doublet(vertices, faces)["matrix"])
        assert stats["overdetermined"] is True
        assert stats["nullity"] == 0
        assert stats["rank"] == stats["n_cols"]
        assert stats["sigma_min"] > 1e-3
        assert stats["condition_number"] < 1e3


def test_constant_density_is_not_a_null_vector_of_the_pure_doublet_operator():
    vertices, faces = icosphere(2)
    stats = operator_stats(assemble_p1_doublet(vertices, faces)["matrix"])
    assert stats["constant_is_null_vector"] is False
    assert abs(stats["row_sum_mean"] + 1.0) < 1e-12
    assert abs(stats["constant_response"] - np.sqrt(len(faces))) < 1e-9


def test_tetrahedron_operator_matches_the_closed_sphere():
    vertices, faces = tetrahedron()
    stats = operator_stats(assemble_p1_doublet(vertices, faces)["matrix"])
    assert stats["square"] is True
    assert stats["nullity"] == 0
    assert abs(stats["row_sum_mean"] + 1.0) < 1e-12


def test_open_sheet_is_underdetermined_not_gauge_free():
    vertices, faces = flat_sheet()
    stats = operator_stats(assemble_p1_doublet(vertices, faces)["matrix"])
    assert stats["n_rows"] < stats["n_cols"]
    assert stats["nullity"] == stats["n_cols"] - stats["rank"]
    assert stats["nullity"] >= 2  # underdetermination, not a physical gauge
    assert stats["constant_is_null_vector"] is False


def test_pinning_is_not_a_gauge_on_the_pure_doublet_operator():
    vertices, faces = icosphere(2)
    matrix = assemble_p1_doublet(vertices, faces)["matrix"]
    rhs = -matrix.sum(axis=1) * 0.5 + 0.123
    effect = constraint_effect(matrix, rhs, 0)
    assert effect["residual_free"] < 1e-9
    assert effect["is_gauge"] is False
    assert effect["residual_pinned"] > effect["residual_free"]


def test_topology_report_covers_the_four_topologies():
    report = topology_report()
    assert set(report) == {"icosphere_1", "icosphere_2", "tetrahedron", "open_sheet"}
    for name, stats in report.items():
        assert stats["rank"] <= stats["n_cols"]
        assert stats["sigma_max"] > 0.0
