"""Contract tests for the topology-aware wake/wall contact classifier (issue 84 T5-A)."""

import numpy as np
import pytest

from panel.p1_geometry_contact import (
    DECLARED_EDGE,
    DECLARED_VERTEX,
    classify_new_triangle,
    classify_swept_wall,
    coplanar_overlap_area,
    declared_contact_pairs,
    swept_hull,
    triangle_triangle_intersection,
)

WALL = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])


def test_declared_pairs_use_vertex_ids_not_coordinates():
    wall_faces = np.array([[0, 1, 2], [2, 3, 4]])
    wake_faces = [np.array([[0, 1, 5], [1, 6, 7]])]
    pairs = declared_contact_pairs(wake_faces, wall_faces, group_labels=["main"])
    assert pairs[("main", 0, 0)][0] == DECLARED_EDGE
    assert set(pairs[("main", 0, 0)][1]) == {0, 1}
    # Wake face 1 shares vertex 1 with wall face 0: a declared vertex contact.
    assert pairs[("main", 1, 0)] == (DECLARED_VERTEX, (1,))


def test_shared_vertex_tangential_motion_is_declared_contact():
    old = np.array([[0., 0., 0.], [.2, 0., .4], [0., .2, .4]])
    new = old + np.array([.5, 0., 0.])
    cls = classify_swept_wall(swept_hull(old, new), WALL, tol=1e-10, declared=DECLARED_VERTEX)
    assert cls.kind == "contact"
    assert cls.declared == DECLARED_VERTEX
    assert not cls.violation


def test_shared_vertex_pushed_through_the_wall_is_a_violation():
    old = np.array([[0., 0., 0.], [.2, 0., .4], [0., .2, .4]])
    new = old + np.array([0., 0., -.5])
    cls = classify_swept_wall(swept_hull(old, new), WALL, tol=1e-10, declared=DECLARED_VERTEX)
    assert cls.kind in ("penetration", "transverse")
    assert cls.depth > 0.4
    assert cls.violation


def test_shared_edge_tangential_motion_is_declared_edge_contact():
    old = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .3, .5]])
    new = old + np.array([.2, .1, 0.])
    cls = classify_swept_wall(swept_hull(old, new), WALL, tol=1e-10, declared=DECLARED_EDGE)
    assert cls.kind == "contact"
    assert cls.declared == DECLARED_EDGE
    assert not cls.violation


def test_declared_edge_passed_through_the_wall_is_still_a_violation():
    old = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .3, .5]])
    new = old.copy()
    new[2] = (.3, .3, -.5)
    cls = classify_swept_wall(swept_hull(old, new), WALL, tol=1e-10, declared=DECLARED_EDGE)
    assert cls.kind in ("penetration", "transverse")
    assert cls.violation


def test_coplanar_overlap_reports_positive_area():
    tri = np.array([[.1, .1, 0.], [.7, .1, 0.], [.1, .7, 0.]])
    area = coplanar_overlap_area(tri, WALL, 1e-10)
    assert area == pytest.approx(.18, rel=1e-6)
    cls = classify_swept_wall(swept_hull(tri, tri), WALL, tol=1e-10)
    assert cls.kind == "coplanar_overlap"
    assert cls.area == pytest.approx(.18, rel=1e-6)
    assert cls.violation


def test_nonadjacent_collision_is_detected_without_declaration():
    old = np.array([[.2, .2, .3], [.6, .2, .3], [.2, .6, .3]])
    new = old.copy()
    new[:, 2] = -.3
    cls = classify_swept_wall(swept_hull(old, new), WALL, tol=1e-10)
    assert cls.kind in ("penetration", "transverse")
    assert cls.violation


def test_contact_that_separates_leaves_the_hull_touching_only():
    old = np.array([[0., 0., 0.], [.2, 0., .4], [0., .2, .4]])
    new = old + np.array([0., 0., .3])
    hull = swept_hull(old, new)
    declared = classify_swept_wall(hull, WALL, tol=1e-10, declared=DECLARED_VERTEX)
    assert not declared.violation
    undeclared = classify_swept_wall(hull, WALL, tol=1e-10)
    assert undeclared.violation
    assert classify_new_triangle(new, WALL, tol=1e-10).kind == "disjoint"


def test_declared_pair_that_penetrates_after_the_first_update_is_caught():
    old = np.array([[0., 0., 0.], [.2, 0., .4], [0., .2, .4]])
    first = old + np.array([.3, .1, 0.])
    assert not classify_swept_wall(swept_hull(old, first), WALL, tol=1e-10,
                                   declared=DECLARED_VERTEX).violation
    second = first + np.array([0., 0., -.4])
    cls = classify_swept_wall(swept_hull(first, second), WALL, tol=1e-10,
                              declared=DECLARED_VERTEX)
    assert cls.violation


def test_near_miss_is_not_a_contact():
    tri = np.array([[.2, .2, .4], [.6, .2, .4], [.2, .6, .4]])
    cls = classify_swept_wall(swept_hull(tri, tri), WALL, tol=1e-10)
    assert cls.kind == "disjoint"
    assert not cls.violation


def test_triangle_triangle_intersection_covers_point_segment_coplanar():
    q = np.array([[.2, .2, -.5], [.2, .2, .5], [.6, .6, .5]])
    kind, _pts = triangle_triangle_intersection(q, WALL, 1e-12)
    assert kind in ("point", "segment")
    coplanar = np.array([[.1, .1, 0.], [.3, .1, 0.], [.1, .3, 0.]])
    assert triangle_triangle_intersection(coplanar, WALL, 1e-12)[0] == "coplanar"
    far = np.array([[3., 3., 0.], [4., 3., 0.], [3., 4., 0.]])
    assert triangle_triangle_intersection(far, WALL, 1e-12)[0] == "none"


def test_scale_aware_tolerance_admits_shallow_touch_and_refuses_deep_overlap():
    tri = np.array([[0., 0., 0.], [.2, 0., .4], [0., .2, .4]])
    shallow = tri + np.array([0., 0., -1e-12])
    assert classify_swept_wall(swept_hull(shallow, shallow), WALL, tol=1e-10,
                               declared=DECLARED_VERTEX).kind == "contact"
    deep = tri + np.array([0., 0., -0.4])
    assert classify_swept_wall(swept_hull(deep, deep), WALL, tol=1e-10,
                               declared=DECLARED_VERTEX).violation
