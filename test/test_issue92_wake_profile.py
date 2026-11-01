"""Contract tests for the issue #92 wake-evaluator profiler (no heavyweight fixtures)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.issue92 import wake_evaluator_profile as profiler


def _sheet():
    """Two-triangle sheet with one root node; the free triangle spans x in [1,2]."""
    xyz = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    group = SimpleNamespace(name="g", wake_xyz=xyz, wake_faces=faces)
    material = SimpleNamespace(roots=np.array([0]))
    return group, material


def test_free_targets_drops_roots_and_offsets_along_the_normal():
    group, material = _sheet()
    mesh = SimpleNamespace(groups=(group,))
    targets = profiler._free_targets(mesh, {"g": material}, eps_fraction=1e-3)
    assert targets.shape == (3, 3)
    # every free node of a flat sheet is moved off the plane along +z
    assert np.all(targets[:, 2] > 0.0)
    # the offset is bounded by the node scale, not by the sheet size
    assert np.all(targets[:, 2] < 1e-2)
    # the root node (node 0) is never a sampler target
    assert not np.any(np.all(np.isclose(targets[:, :2], 0.0), axis=1))


def test_free_targets_skips_nodes_without_faces():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    group = SimpleNamespace(name="g", wake_xyz=xyz, wake_faces=faces)
    mesh = SimpleNamespace(groups=(group,))
    targets = profiler._free_targets(mesh, {"g": SimpleNamespace(roots=np.array([0]))})
    assert len(targets) == 2   # nodes 0, 3 have no face normal; 0 is also a root


def test_stages_split_the_evaluator_from_the_fan_internals():
    snapshot = {"seconds": {"field.edge_velocity": 0.5, "area_terms.kernel": 2.0,
                            "fan.calls": 0.25, "limits.total": 1.0},
                "counters": {}}
    stages = profiler._stages(snapshot)
    assert stages["evaluator"] == {"field.edge_velocity": 0.5, "limits.total": 1.0}
    assert stages["fan"] == {"area_terms.kernel": 2.0, "fan.calls": 0.25}


def test_measure_reports_per_target_cost_and_counters():
    class Field:
        def __init__(self):
            self.seen = 0

        def __call__(self, point):
            self.seen += 1
            return np.zeros(3)

    field = Field()
    record = profiler.measure(field, np.zeros((4, 3)), label="unit", repeat=1)
    assert record["label"] == "unit"
    assert record["targets"] == 4
    assert field.seen == 4
    assert record["ms_per_target"] >= 0.0
    assert set(record["stages_seconds"]) == {"evaluator", "fan"}
    assert isinstance(record["counters"], dict)


def test_counters_report_the_sheet_sizes_it_profiled():
    group, material = _sheet()
    mesh = SimpleNamespace(faces=np.zeros((5, 3), int), groups=(group,))
    counters = profiler._counters(mesh, {"g": material})
    assert counters == {"n_wall_faces": 5, "n_wake_triangles": 2,
                        "n_wake_nodes": 4, "n_free_targets": 3}
