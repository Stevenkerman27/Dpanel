"""Falsifiable tests for the projection-aware Duffy area quadrature (issue #84).

Each test states the property it protects; none of them depend on the production
field class except the cross-validation ones, which compare against the shipped
regularized evaluator at a high order.
"""
from __future__ import annotations

import numpy as np
import pytest

from panel.p1_projection_duffy import (
    _duffy_nodes,
    _sub_area2,
    area_terms,
    area_velocity,
    closest_point_on_triangle,
    projection_fan,
)
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField

TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _sliver(aspect, length=1.0):
    """Right triangle whose aspect ratio (longest edge / shortest height) is aspect."""
    height = length / aspect
    return np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, height, 0.0]])


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@pytest.mark.parametrize("point", [
    np.array([0.25, 0.25, 0.4]),      # interior projection
    np.array([0.5, -0.3, 0.2]),       # projection on edge 01
    np.array([-0.4, -0.4, 0.3]),      # projection at vertex 0
    np.array([2.0, 2.0, 0.5]),        # projection outside, still vertex 0 region
])
@pytest.mark.parametrize("strips", [1, 3])
@pytest.mark.parametrize("swap", [False, True])
def test_partition_area_equals_parent(point, strips, swap):
    sub, meta = projection_fan(TRI, point, strips=strips, swap=swap)
    parent = 0.5 * _sub_area2(TRI[None])[0]
    total = 0.5 * float(np.sum(_sub_area2(sub)))
    assert meta["n_sub"] == len(sub)
    assert abs(total - parent) <= 1e-14 * parent


@pytest.mark.parametrize("point", [
    np.array([0.2, 0.2, 0.7]),
    np.array([0.3, -0.6, 0.1]),
    np.array([0.9, -0.9, 0.1]),
])
def test_partition_has_no_overlap(point):
    sub, _ = projection_fan(TRI, point, strips=2)
    rng = np.random.default_rng(7)
    n = 400
    uv = rng.random((n, 2))
    keep = uv.sum(axis=1) <= 1.0
    uv = uv[keep]
    samples = TRI[0] + uv[:, :1] * (TRI[1] - TRI[0]) + uv[:, 1:] * (TRI[2] - TRI[0])
    hits = np.zeros(len(samples), dtype=int)
    for piece in sub:
        hits += _inside(piece, samples).astype(int)
    assert int(hits.min()) >= 1
    assert int(hits.max()) == 1


def _inside(tri, points):
    """Barycentric containment with a small tolerance, in the triangle plane."""
    a, b, c = tri
    v0, v1 = b - a, c - a
    normal = np.cross(v0, v1)
    scale = np.linalg.norm(normal)
    normal = normal / scale
    rel = points - a
    out_of_plane = np.abs(rel @ normal) > 1e-10 * max(scale, 1.0)
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    denom = d00 * d11 - d01 * d01
    dp = rel @ np.stack([v0, v1]).T
    u = (d11 * dp[:, 0] - d01 * dp[:, 1]) / denom
    v = (d00 * dp[:, 1] - d01 * dp[:, 0]) / denom
    tol = 1e-10
    inside = (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol) & (~out_of_plane)
    return inside


def test_constant_integrand_reproduces_area_exactly():
    sub, _ = projection_fan(TRI, np.array([0.3, 0.3, 0.2]), strips=4)
    u, v, w = _duffy_nodes(6)
    total = 0.0
    for piece in sub:
        a2 = float(_sub_area2(piece[None])[0])
        total += a2 * float(np.sum(w))
    assert abs(total - 0.5) <= 1e-14


def test_duffy_map_covers_its_sub_triangle():
    sub, _ = projection_fan(TRI, np.array([0.2, 0.4, 0.1]), strips=1)
    u, v, _ = _duffy_nodes(5)
    for piece in sub:
        q, a, b = piece
        source = q + u[:, None] * (a - q) + (u * v)[:, None] * (b - a)
        assert _inside(piece, source).all()
        assert np.all(source[:, 2] == 0.0)


def _reference(point, core, tri=TRI, *, order=48, strips=16):
    # the fan at order 48 is converged to ~1e-9 for these cases; strips saturate at 8
    return area_terms(tri[None], point, core, order=order, strips=strips)[0]


@pytest.mark.parametrize("distance", [0.3, 1.0, 3.0, 10.0])
def test_matches_reference_below_onee_6(distance):
    point = np.array([0.25, 0.25, distance])
    core = 0.01
    ref = _reference(point, core)
    got = area_terms(TRI[None], point, core, order=32, strips=8)[0]
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 5e-7, rel


def test_translation_rotation_scale_covariance():
    point = np.array([0.2, 0.35, 0.05])
    core = 0.02
    base = area_terms(TRI[None], point, core, order=24, strips=4)[0]
    rot = _rotation([1.0, 2.0, 3.0], 0.7)
    shift = np.array([3.0, -2.0, 1.5])
    scale = 2.5
    tri2 = (TRI @ rot.T) * scale + shift
    point2 = rot @ (point * scale) + shift
    moved = area_terms(tri2[None], point2, core * scale, order=24, strips=4)[0]
    # the integral scales like length, so int r/|r|^3 dS is invariant
    rel = np.linalg.norm(moved - rot @ base) / np.linalg.norm(base)
    assert rel < 1e-12, rel


@pytest.mark.parametrize("point,kind", [
    (np.array([0.2, 0.3, 0.6]), "interior"),
    (np.array([0.5, -0.2, 0.6]), "edge"),
    (np.array([-0.3, -0.3, 0.6]), "vertex"),
])
def test_projection_branch_is_reported(point, kind):
    _, region = closest_point_on_triangle(TRI, point)
    assert region["kind"] == kind


def test_edge_branch_picks_the_right_edge():
    _, region = closest_point_on_triangle(TRI, np.array([0.5, -0.1, 0.0]))
    assert region == {"kind": "edge", "index": 0}
    _, region = closest_point_on_triangle(TRI, np.array([0.6, 0.5, 0.0]))
    assert region == {"kind": "edge", "index": 1}
    _, region = closest_point_on_triangle(TRI, np.array([-0.1, 0.5, 0.0]))
    assert region == {"kind": "edge", "index": 2}


@pytest.mark.parametrize("aspect", [1.0, 10.0, 50.0, 260.0, 520.0, 1405.7])
def test_sliver_error_shrinks_with_strips(aspect):
    tri = _sliver(aspect)
    point = np.array([0.5, 0.5 / aspect, 0.01])
    core = 0.01
    ref = _reference(point, core, tri=tri, order=48, strips=16)
    coarse = area_terms(tri[None], point, core, order=16, strips=1)[0]
    fine = area_terms(tri[None], point, core, order=16, strips=8)[0]
    e_coarse = np.linalg.norm(coarse - ref) / np.linalg.norm(ref)
    e_fine = np.linalg.norm(fine - ref) / np.linalg.norm(ref)
    # strips must never make a sliver worse, and the fan must reach the reference
    assert e_fine < 5e-3, (aspect, e_fine)
    if aspect >= 50.0:
        assert e_fine < 0.5 * e_coarse, (aspect, e_coarse, e_fine)


def test_error_drops_with_order_at_fixed_strips():
    """Once the peak is at a fan apex the error is order limited, not strips limited."""
    tri = _sliver(520.0)
    point = np.array([0.5, 0.5 / 520.0, 0.01])
    core = 0.01
    ref = _reference(point, core, tri=tri, order=48, strips=16)
    e16 = np.linalg.norm(area_terms(tri[None], point, core, order=16, strips=8)[0] - ref)
    e32 = np.linalg.norm(area_terms(tri[None], point, core, order=32, strips=8)[0] - ref)
    e48 = np.linalg.norm(area_terms(tri[None], point, core, order=48, strips=8)[0] - ref)
    assert e32 < 0.01 * e16, (e16, e32)
    assert e48 < 0.01 * e32, (e32, e48)


def test_strips_saturate_beyond_eight():
    tri = _sliver(1405.7)
    point = np.array([0.5, 0.5 / 1405.7, 0.01])
    core = 0.01
    a = area_terms(tri[None], point, core, order=16, strips=8)[0]
    b = area_terms(tri[None], point, core, order=16, strips=64)[0]
    assert np.linalg.norm(a - b) <= 1e-6 * np.linalg.norm(b)


def test_legacy_quadrature_is_not_converged_near_a_sliver():
    """The shipped single-Duffy map stays far from the reference where the fan converges."""
    tri = _sliver(20.0)
    point = np.array([0.4, 0.01, 0.02])
    core = 0.01
    field = RegularizedP1SheetField([(tri, np.array([[0, 1, 2]]), np.array([[0.0, 0.3, 0.0]]))],
                                     core_radius=core)
    gamma = field.gamma[0:1]
    ref = area_velocity(tri[None], gamma, point, core, order=48, strips=16)
    legacy96 = field._area_velocity(point, 96)
    fan32 = area_velocity(tri[None], gamma, point, core, order=32, strips=8)
    e_legacy = np.linalg.norm(legacy96 - ref) / np.linalg.norm(ref)
    e_fan = np.linalg.norm(fan32 - ref) / np.linalg.norm(ref)
    assert e_fan < 1e-7, e_fan
    assert e_legacy > 1e-4, e_legacy


def test_agrees_with_production_in_the_smooth_far_field():
    tri = _sliver(20.0)
    point = np.array([0.4, 0.01, 3.0])
    core = 0.01
    field = RegularizedP1SheetField([(tri, np.array([[0, 1, 2]]), np.array([[0.0, 0.3, 0.0]]))],
                                     core_radius=core)
    gamma = field.gamma[0:1]
    production = field._area_velocity(point, 96)
    fan = area_velocity(tri[None], gamma, point, core, order=24, strips=4)
    rel = np.linalg.norm(fan - production) / np.linalg.norm(production)
    assert rel < 1e-6, rel
