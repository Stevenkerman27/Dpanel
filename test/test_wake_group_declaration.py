"""Contract tests for the gmsh-free wake-group declaration parser."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel.wake_group_declaration import (  # noqa: E402
    SCHEMA, merged_stations, positive_trailing_edge, read_wake_groups)

SCRATCH = ROOT / "runs" / "wake_declaration_tests"


def _write(name, payload):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _te(n=5, span=1.6, z=0.0):
    y = np.linspace(-span, span, n)
    x = 0.9 + 0.1*np.abs(y)/span
    return np.stack((x, y, np.full(n, z)), axis=1)


def test_merged_stations_removes_float_near_duplicates():
    # The uniform grid already contains 0.4; a TE station at 0.40000000000001
    # would otherwise survive np.unique as a second station, creating a
    # zero-width wake strip and an OCC addLine failure.
    te = np.asarray([0.0, 0.40000000000001, 1.0])
    merged = merged_stations(te, 5)
    assert np.all(np.diff(merged) > 1e-9)
    assert np.allclose(merged, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-12)


def test_merged_stations_rejects_nonmonotone_input():
    with pytest.raises(ValueError, match="strictly increasing"):
        merged_stations(np.asarray([0.0, 0.5, 0.4]), 2)


def test_positive_trailing_edge_pins_the_root_and_drops_negative_y():
    te = _te()
    positive = positive_trailing_edge(te)
    assert positive[0, 1] == 0.0
    assert positive[0, 0] == pytest.approx(np.interp(0., te[:, 1], te[:, 0]))
    assert np.all(positive[1:, 1] > 0)
    assert np.all(np.diff(positive[:, 1]) > 0)


def test_positive_trailing_edge_requires_positive_y_stations():
    with pytest.raises(ValueError, match="span positive and negative y|positive-y"):
        positive_trailing_edge(np.asarray([[0.9, -0.4, 0.0], [1.0, 0.0, 0.0]]))


def test_read_wake_groups_normalizes_and_keeps_physical_coordinates():
    payload = dict(schema=SCHEMA, groups=[
        dict(group_id="lower", points=_te().tolist(), z=0.0),
        dict(group_id="upper", points=_te(z=0.24).tolist(), z=0.24),
    ])
    path = _write("two_groups.json", payload)
    declaration, groups = read_wake_groups(path, chord=0.8, origin=np.zeros(3))
    assert [g["group_id"] for g in groups] == ["lower", "upper"]
    lower, upper = groups
    assert np.allclose(lower["normalized_te"], lower["physical_te"]/0.8)
    assert lower["normalized_z"] == 0.0
    assert upper["normalized_z"] == pytest.approx(0.24/0.8)
    # The sheet-relative shift must survive the round trip: the normalized TE
    # of a raised group sits at its own normalized z, not at zero.
    assert np.allclose(upper["normalized_te"][:, 2], upper["normalized_z"])
    assert np.allclose(upper["physical_te"][:, 2], 0.24)
    assert declaration["schema"] == SCHEMA


def test_read_wake_groups_rejects_duplicate_identifiers():
    payload = dict(schema=SCHEMA, groups=[
        dict(group_id="wing", points=_te().tolist(), z=0.0),
        dict(group_id="wing", points=_te(z=0.2).tolist(), z=0.2),
    ])
    with pytest.raises(ValueError, match="Duplicate wake group_id"):
        read_wake_groups(_write("dup.json", payload), chord=1.0, origin=np.zeros(3))


def test_read_wake_groups_rejects_shared_plane():
    payload = dict(schema=SCHEMA, groups=[
        dict(group_id="a", points=_te().tolist(), z=0.2),
        dict(group_id="b", points=_te().tolist(), z=0.2),
    ])
    with pytest.raises(ValueError, match="share the same z plane"):
        read_wake_groups(_write("plane.json", payload), chord=1.0, origin=np.zeros(3))


def test_read_wake_groups_rejects_unknown_schema_and_missing_keys():
    with pytest.raises(ValueError, match="Unsupported wake-group declaration schema"):
        read_wake_groups(_write("schema.json", dict(schema="dpanel.other", groups=[
            dict(group_id="a", points=_te().tolist(), z=0.0)])),
            chord=1.0, origin=np.zeros(3))
    with pytest.raises(ValueError, match="missing"):
        read_wake_groups(_write("missing.json", dict(schema=SCHEMA, groups=[
            dict(group_id="a", points=_te().tolist())])),
            chord=1.0, origin=np.zeros(3))


def test_read_wake_groups_rejects_asymmetric_origin_and_bad_chord():
    path = _write("ok.json", dict(schema=SCHEMA, groups=[
        dict(group_id="a", points=_te().tolist(), z=0.0)]))
    with pytest.raises(ValueError, match="symmetric and finite"):
        read_wake_groups(path, chord=1.0, origin=np.asarray([0.0, 0.1, 0.0]))
    with pytest.raises(ValueError, match="positive and finite"):
        read_wake_groups(path, chord=0.0, origin=np.zeros(3))