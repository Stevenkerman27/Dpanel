"""Pure NumPy regression of a real half-mesh extractor.

The fixture was prepared from the strictly conforming half-wing reference
already accepted in PR #21. This test is a mesh *export* acceptance, NOT proof
that OCC produces such a mesh on actual STEP wing-body inputs.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from panel.half_surface_mesh import (
    build_half_surface, half_contact_station_map,
)


SOURCE = Path(__file__).resolve().parents[1] / "cases" / "panel_half" / "thick_wing_two_strip.json"


def surface_from_fixture():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    points = np.asarray(source["coordinates"], dtype=float)
    return source, dict(
        points=points, wall_faces=source["faces"],
        wall_tags=source["component_ids"], wake_faces=source["wake_faces"],
        te_polyline=[[1., 0., 0.], [1., .5, 0.], [1., 1., 0.]],
        stations=source["gamma_stations"], contact_intervals=[],
        reference_area=source["reference_area"],
        reference_chord=source["reference_chord"],
        reference_point=source["moment_reference"],
        freestream=source["freestream"], density=source["density"],
        wake_end_x=5.,
    )


def test_half_export_keeps_only_real_positive_wall_and_wake():
    src, config = surface_from_fixture()
    out = build_half_surface(**config)
    assert out["schema"] == "dpanel.half_planar_cut.v1"
    assert len(out["faces"]) == len(src["faces"])
    assert len(out["wake_faces"]) == len(src["wake_faces"])+4
    assert out["gamma_station_dof"] == [0, 1, -1]
    assert out["cut_kind"] == ["trailing_edge", "trailing_edge"]
    assert {tuple(e) for e in out["cut_edges"]} == {(2, 6), (6, 10)}
    assert out["audit"]["no_artificial_symmetry_wall"]
    assert out["audit"]["symmetry_seam_edges"] > 0
    assert out["audit"]["wake_wall_contact_edges"] == 0
    x = np.asarray(out["coordinates"])
    assert np.all(x[:, 1] >= 0)
    assert not np.any(np.all(x[np.asarray(out["faces"]), 1] == 0, axis=1))
    # Optional JAX+half-native integration must validate the exported schema
    # and preserve free root circulation; do not silently skip a missing backend.
    from panel.run_half_native import load_half_surface
    from panel.half_native import prepare_half_topology
    t = prepare_half_topology(
        x, out["faces"], out["wake_faces"], out["trailing_edges"],
        out["upper_te_faces"], out["lower_te_faces"],
        out["gamma_stations"], out["gamma_station_dof"],
        out["component_ids"], cut_edges=out["cut_edges"])
    assert len(t.faces) == len(out["faces"])


def test_half_contact_map_preserves_root_dof_and_supports_body_contact():
    y = np.array([0., .2, .4, 1.])
    np.testing.assert_array_equal(
        half_contact_station_map(y, [], tolerance=1e-9),
        [0, 1, 2, -1])
    np.testing.assert_array_equal(
        half_contact_station_map(y, [[0., .2]], tolerance=1e-9),
        [0, 0, 1, -1])
    with pytest.raises(ValueError, match="interior"):
        half_contact_station_map(y, [[.4, 1.]], tolerance=1e-9)


def test_negative_y_or_crossing_wall_cannot_be_clipped_silently():
    _, options = surface_from_fixture()
    options["points"] = np.array(options["points"], copy=True)
    options["points"][4, 1] = -.1
    with pytest.raises(ValueError, match="negative-y|cross"):
        build_half_surface(**options)


def test_artificial_symmetry_plane_cap_is_rejected():
    _, options = surface_from_fixture()
    options["wall_faces"] = list(options["wall_faces"]) + [[0, 1, 2]]
    options["wall_tags"] = list(options["wall_tags"]) + [10]
    with pytest.raises(ValueError, match="cap|Non-manifold"):
        build_half_surface(**options)


def test_nonconforming_wall_edge_cannot_be_welded_by_distance():
    _, options = surface_from_fixture()
    faces = np.asarray(options["wall_faces"], dtype=int).copy()
    vertices = np.asarray(options["points"], dtype=float)
    faces[0, 0] = len(vertices)
    vertices = np.vstack((vertices, vertices[0]))
    options["points"], options["wall_faces"] = vertices, faces
    with pytest.raises(ValueError, match="open edge|Non-manifold|seam"):
        build_half_surface(**options)


def test_unsupported_wake_extension_rejected():
    _, options = surface_from_fixture()
    options["wake_end_x"] = 4.
    with pytest.raises(ValueError, match="farther downstream"):
        build_half_surface(**options)
