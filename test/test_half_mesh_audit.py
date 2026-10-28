"""Synthetic, fixed-topology tests for separate mirror failure classes."""
import numpy as np
import pytest
from panel.audit_half_mesh import audit_half_surface


def symmetric_fixture():
    # Two wing surface banks meet at a shared TE root+tip edge on each side.
    # The reflected left triangles have reversed winding to preserve normals.
    x = [
        [0.,0.,0.], [1.,0.,0.], [1.,1.,0.],
        [.5,.7,.2], [.5,.7,-.2], [1.,-1.,0.],
        [.5,-.7,.2], [.5,-.7,-.2], [2.,0.,0.],
        [2.,1.,0.], [2.,-1.,0.],
    ]
    faces = [
        [1,2,3], [1,4,2],        # right upper, lower
        [1,6,5], [1,5,7],        # left upper, lower
    ]
    wake_faces = [
        [1,8,9], [1,9,2], [1,10,8], [1,5,10],
    ]
    return dict(schema="dpanel.planar_cut.v1", coordinates=x, faces=faces,
                wake_faces=wake_faces,
                component_ids=[10,10,10,10],
                trailing_edges=[[1,2],[1,5]],
                upper_te_faces=[0,2], lower_te_faces=[1,3],
                gamma_stations=[-1.,0.,1.], gamma_station_dof=[-1,0,-1])


def test_perfect_reflection_matches_vertices_wall_wake_te_and_banks():
    report = audit_half_surface(symmetric_fixture())
    assert report["status"] == "topology_symmetric", report["reasons"]
    assert report["vertices"]["paired"] == 11
    assert report["wall"]["exact_face_pairs"] == 4
    assert report["wake"]["exact_face_pairs"] == 4
    assert report["wall"]["plane_face_count"] == 0
    assert report["trailing_edges"]["paired_with_same_upper_lower_banks"] == 2
    assert report["wall"]["normals_mirror_max_error"] < 1e-15
    assert report["wake"]["normals_mirror_max_error"] < 1e-15


def test_geometry_asymmetry_is_detected_before_topology():
    f = symmetric_fixture()
    f["coordinates"][6][1] -= .01
    report = audit_half_surface(f)
    assert report["status"] == "blocked"
    assert not report["vertices"]["bijective"] or report["vertices"]["paired"] < 11
    assert report["wall"]["exact_face_pairs"] is None


def test_mirror_nodes_but_different_triangle_connectivity_is_detected():
    f = symmetric_fixture()
    # Mirror vertices remain, but one left-side face uses the wrong corner.
    f["faces"][2] = [1,5,8]
    report = audit_half_surface(f)
    assert report["vertices"]["paired"] == 11
    assert report["wall"]["exact_face_pairs"] < 4
    assert report["status"] == "blocked"


def test_wrong_te_bank_mapping_is_detected_independently():
    f = symmetric_fixture()
    f["lower_te_faces"][1] = 2
    report = audit_half_surface(f)
    assert report["wall"]["exact_face_pairs"] == 4
    assert report["trailing_edges"]["paired_with_same_upper_lower_banks"] < 2
    assert report["status"] == "blocked"


def test_plane_face_and_crossing_face_distinguished_from_self_nearest():
    f = symmetric_fixture()
    f["coordinates"].append([.3, 0., .4])
    f["faces"].append([1,8,len(f["coordinates"])-1])  # entirely on y=0
    f["component_ids"].append(10)
    result = audit_half_surface(f)
    assert result["wall"]["plane_face_count"] == 1
    assert result["status"] == "blocked"


def test_unordered_stations_rejected():
    f = symmetric_fixture()
    f["gamma_stations"] = [0,1,-1]
    with pytest.raises(ValueError, match="unordered"):
        audit_half_surface(f)


def test_circulation_stations_can_mirror_while_dof_labels_do_not():
    f = symmetric_fixture()
    f["gamma_station_dof"] = [-1,0,1]
    result = audit_half_surface(f)
    assert result["stations"]["mirror_compatible"]
    assert not result["stations"]["circulation_dof_mirror_compatible"]
    assert result["status"] == "blocked"
