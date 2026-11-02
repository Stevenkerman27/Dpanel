"""Contract tests for the far-field distance-band probe (no heavyweight fixtures)."""
from __future__ import annotations

import numpy as np

from tools.issue92 import far_field_probe as probe


def _square():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mu = np.array([0.0, 1e-3, 2e-3, 1e-3])
    return xyz, faces, mu


def test_monopole_is_the_point_source_form_with_the_core_in_the_denominator():
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    terms = probe._monopole_terms(tri, np.array([0.3, 0.3, 2.0]), 0.1)
    area = 0.5
    r = np.array([0.3, 0.3, 2.0]) - np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    d2 = float(r @ r) + 0.01
    expected = (area / (d2 * np.sqrt(d2))) * r
    assert np.allclose(terms[0], expected, rtol=1e-14)


def test_velocity_is_the_face_sum_of_gamma_cross_term():
    gamma = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    terms = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    got = probe._velocity(gamma, terms)
    expected = np.cross(gamma[0], terms[0]) + np.cross(gamma[1], terms[1])
    assert np.allclose(got, expected)


def test_measure_reports_every_candidate_and_the_near_face_share():
    xyz, faces, mu = _square()
    records = probe.measure(xyz, faces, mu, (50.0,), core=0.01,
                            reference_order=8, reference_strips=1)
    assert len(records) == 3            # off-plane, below, downstream
    row = records[0]
    for key in ("monopole", "fan16x1", "fan24x1", "fan32x1",
                "legacy_duffy8", "legacy_duffy16", "legacy_duffy24"):
        assert key in row and np.isfinite(row[key])
    # a point half a length unit off a unit square keeps no face inside 10 core
    assert row["face_count"] == 2
    assert 0 <= row["faces_within_10_core"] <= 2
    assert row["faces_within_30_core"] >= row["faces_within_10_core"]
    # the reference is the fan itself, so the fan at the reference setting is exact
    assert row["fan32x1"] < 1e-6
