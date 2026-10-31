"""Adaptive near/far area quadrature for the regularized P1 sheet kernel (issue #84 T2).

The projection fan is accurate but pays for every triangle; the single-collapse Duffy
map is cheap but has no error control. This module splits each target's triangles into
a near set (target close to the triangle relative to its own size) and a far set, uses
an a-posteriori difference estimator on both, and escalates only the faces whose
estimated contribution exceeds their share of a global error budget.

Everything here is opt-in and works on the same kernel as the projection fan:

    I_f(p) = int_{T_f} (p - x) / (|p - x|^2 + core^2)^{3/2} dS

Near faces use the projection fan (base -> refined fan difference is the estimator).
Far faces use a composite Duffy map (uniform midpoint subdivision, then one collapsed
Duffy map per sub-triangle); the estimator is the difference between two subdivision
or order levels. Neither estimator is an a-priori bound: both are differences between
two computed values, so they are a-posteriori estimates only.
"""
from __future__ import annotations

import time

import numpy as np

from . import velocity_profile as _profile
from .p1_projection_duffy import _duffy_nodes, _sub_area2, area_terms, closest_point_on_triangle


def _midpoint_split(triangles):
    """Split each (3,3) triangle into four by connecting its edge midpoints."""
    tri = np.asarray(triangles, float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
    out = np.empty((len(tri), 4, 3, 3))
    out[:, 0] = np.stack((a, ab, ca), axis=1)
    out[:, 1] = np.stack((ab, b, bc), axis=1)
    out[:, 2] = np.stack((ca, bc, c), axis=1)
    out[:, 3] = np.stack((ab, bc, ca), axis=1)
    return out.reshape(-1, 3, 3)


def _subdivide(triangles, levels):
    tri = np.asarray(triangles, float)
    for _ in range(int(levels)):
        tri = _midpoint_split(tri)
    return tri


def composite_duffy_terms(triangles, point, core, *, levels=1, order=8, chunk=4096):
    """Far-field area kernel with midpoint subdivision and one collapsed Duffy per piece.

    Returns (N,3) per-parent-triangle contributions. The collapse vertex of every piece
    is the one closest to the target, so the peak stays near a parametric corner even
    without the full projection machinery.
    """
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must be (N,3,3)")
    if not np.isfinite(core) or core <= 0.0:
        raise ValueError("core must be a positive finite radius")
    if int(levels) < 0:
        raise ValueError("levels must be >= 0")
    _profile.add("adaptive.far_calls")
    pieces = _subdivide(tri, levels)
    owner = np.repeat(np.arange(len(tri)), 4 ** int(levels))
    distance = np.linalg.norm(pieces - p[None, None, :], axis=2)
    collapse = np.argmin(distance, axis=1)
    index = np.arange(len(pieces))
    v0 = pieces[index, collapse]
    v1 = pieces[index, (collapse + 1) % 3]
    v2 = pieces[index, (collapse + 2) % 3]
    a2 = _sub_area2(np.stack((v0, v1, v2), axis=1))
    u, v, w = _duffy_nodes(int(order))
    uu = u[None, :]
    vv = v[None, :]
    out = np.zeros((len(tri), 3))
    step = max(1, int(chunk))
    for first in range(0, len(pieces), step):
        last = first + step
        source = (v0[first:last, None, :] + uu[..., None] * (v1[first:last, None, :] - v0[first:last, None, :])
                  + (uu * vv)[..., None] * (v2[first:last, None, :] - v1[first:last, None, :]))
        r = p[None, None, :] - source
        den = (np.einsum("sqi,sqi->sq", r, r) + core * core) ** 1.5
        quad = np.einsum("q,sqi->si", w, r / den[..., None])
        np.add.at(out, owner[first:last], quad * a2[first:last, None])
    return out


def _face_geometry(triangles, point):
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    distance = np.empty(len(tri))
    for index in range(len(tri)):
        q = closest_point_on_triangle(tri[index], p)[0]
        distance[index] = float(np.linalg.norm(p - q))
    edges = np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2)
    area2 = _sub_area2(tri)
    with np.errstate(divide="ignore", invalid="ignore"):
        circumradius = np.where(area2 > 0.0, np.prod(edges, axis=1) / (2.0 * area2), np.inf)
    return distance, area2, edges.min(axis=1), circumradius


def adaptive_area_terms(
    triangles, gamma, point, core, *, near_ratio=2.0, near_fan=(28, 8),
    near_refinement=(48, 8), far_order=8, far_levels=1, budget=2e-4,
    max_escalations=2, subdivision_levels=0, subdivision_fan=(48, 8), chunk=4096,
):
    """Near/far adaptive area kernel with a global relative error budget.

    The budget is relative to the norm of the base area-velocity sum and is allocated to
    each face by |gamma_f| * area_f / max(distance_f, core)^2, its share of the local
    kernel magnitude. A face whose a-posteriori estimate exceeds its share is refined
    (near: fan strips or order up; far: one more subdivision level and a higher order)
    for at most max_escalations rounds.

    Order escalation alone stalls on near-singular faces: on a 1405 aspect sliver the fan
    stays within 1e-2 of the converged value however many rounds it is granted, because
    base and refined fail in the same direction. ``subdivision_levels`` adds a final rung:
    every near face still over budget after the last round is split in the parameter
    domain (``4**levels`` pieces) and each piece is integrated with ``subdivision_fan``.
    Set it to 0 (the default) to keep the order-only ladder.
    """
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    values = np.asarray(gamma, float)
    if values.shape == (len(tri),):
        magnitude = np.abs(values)
    elif values.shape == (len(tri), 3):
        magnitude = np.linalg.norm(values, axis=1)
    else:
        raise ValueError("gamma must be one scalar or one vector per triangle")
    started = time.perf_counter()
    distance, area2, _minimum_edge, circumradius = _face_geometry(tri, p)
    near = (distance <= near_ratio * circumradius) | (distance <= core)
    near_count = int(near.sum())
    far_count = int(len(tri) - near_count)

    def near_terms(order, strips, mask):
        if not mask.any():
            return np.zeros((len(tri), 3))
        out = np.zeros((len(tri), 3))
        out[mask] = area_terms(tri[mask], p, core, order=order, strips=strips, chunk=1024)
        return out

    def far_terms(order, levels, mask):
        if not mask.any():
            return np.zeros((len(tri), 3))
        out = np.zeros((len(tri), 3))
        out[mask] = composite_duffy_terms(tri[mask], p, core, levels=levels,
                                          order=order, chunk=chunk)
        return out

    def weighted_vector(block, strengths):
        """The velocity error proxy of a per-face difference: sum_f gamma_f x diff_f."""
        if len(block) == 0:
            return np.zeros(3)
        if values.ndim == 2:
            return np.sum(np.cross(strengths, block), axis=0)
        return np.sum(strengths[:, None]*block, axis=0)

    base = near_terms(near_fan[0], near_fan[1], near) + far_terms(far_order, far_levels, ~near)
    refined = (near_terms(near_refinement[0], near_refinement[1], near)
               + far_terms(far_order + 4, far_levels + 1, ~near))
    difference = refined - base
    aggregate = weighted_vector(difference, values)
    estimate = np.linalg.norm(difference, axis=1)
    weight = magnitude * (area2 / 2.0) / np.maximum(distance, core) ** 2
    scale = max(float(np.linalg.norm(base.sum(axis=0))), 1e-30)
    allowed_total = float(budget) * scale
    if weight.sum() > 0.0:
        allowed = allowed_total * weight / weight.sum()
    else:
        allowed = np.full(len(tri), allowed_total / max(len(tri), 1))

    if type(subdivision_levels) is not int or subdivision_levels < 0:
        raise ValueError("subdivision_levels must be an integer >= 0")
    if subdivision_levels > 4:
        raise ValueError("subdivision_levels must be <= 4")
    try:
        subdivision_order, subdivision_strips = (int(v) for v in subdivision_fan)
    except (TypeError, ValueError):
        raise ValueError("subdivision_fan must be a (order, strips) pair") from None
    if subdivision_order < 4 or subdivision_strips < 1:
        raise ValueError("subdivision_fan order must be >= 4 and strips >= 1")

    total = base.copy()
    escalations = 0
    escalated_faces = 0
    for _round in range(int(max_escalations)):
        over = (estimate > allowed) & (estimate > 0.0)
        if not over.any():
            break
        escalations += 1
        escalated_faces += int(over.sum())
        near_over = over & near
        far_over = over & ~near
        fan_order = min(near_fan[0] * 2 ** escalations, 96)
        strips = near_fan[1] * 2 ** escalations
        refined_fan = (min(near_refinement[0] * 2 ** escalations, 192),
                       near_refinement[1] * 2 ** escalations)
        if near_over.any():
            coarse = near_terms(fan_order, strips, near_over)
            fine = near_terms(refined_fan[0], refined_fan[1], near_over)
            delta = fine - coarse - difference
            aggregate += weighted_vector(delta[near_over], values[near_over])
            difference = np.where(near_over[:, None], fine - coarse, difference)
            estimate[near_over] = np.linalg.norm(fine[near_over] - coarse[near_over], axis=1)
            total[near_over] = fine[near_over]
        if far_over.any():
            coarse = far_terms(far_order + 4 * escalations, far_levels + escalations, far_over)
            fine = far_terms(far_order + 4 * escalations + 4, far_levels + escalations + 1,
                             far_over)
            delta = fine - coarse - difference
            aggregate += weighted_vector(delta[far_over], values[far_over])
            difference = np.where(far_over[:, None], fine - coarse, difference)
            estimate[far_over] = np.linalg.norm(fine[far_over] - coarse[far_over], axis=1)
            total[far_over] = fine[far_over]

    subdivision_faces = 0
    subdivision_pieces = 0
    division_seconds = 0.0
    if int(subdivision_levels) > 0 and (estimate > allowed).any():
        stuck = np.flatnonzero((estimate > allowed) & (estimate > 0.0) & near)
        if len(stuck):
            subdivision_faces = int(len(stuck))
            begin = time.perf_counter()
            pieces = _subdivide(tri[stuck], subdivision_levels)
            per_face = 4 ** int(subdivision_levels)
            subdivision_pieces = int(len(pieces))
            local = np.repeat(np.arange(len(stuck)), per_face)
            coarse = area_terms(pieces, p, core, order=subdivision_order,
                               strips=subdivision_strips, chunk=1024)
            fine = area_terms(pieces, p, core, order=min(2 * subdivision_order, 96),
                              strips=2 * subdivision_strips, chunk=1024)
            aggregated_coarse = np.zeros((len(stuck), 3))
            aggregated_fine = np.zeros((len(stuck), 3))
            np.add.at(aggregated_coarse, local, coarse)
            np.add.at(aggregated_fine, local, fine)
            piece_difference = aggregated_fine - aggregated_coarse
            aggregate += weighted_vector(piece_difference - difference[stuck], values[stuck])
            estimate[stuck] = np.linalg.norm(piece_difference, axis=1)
            total[stuck] = aggregated_fine
            division_seconds = time.perf_counter() - begin

    diagnostics = dict(
        near_faces=near_count, far_faces=far_count,
        near_fan_order=int(near_fan[0]), near_fan_strips=int(near_fan[1]),
        near_refinement_order=int(near_refinement[0]),
        near_refinement_strips=int(near_refinement[1]),
        far_order=int(far_order), far_levels=int(far_levels),
        budget=float(budget), allowed_total=allowed_total, velocity_scale=scale,
        estimated_error=float(estimate.sum()),
        relative_estimate=float(estimate.sum() / scale),
        aggregate_estimated_error=float(np.linalg.norm(aggregate)),
        aggregate_relative_estimate=float(np.linalg.norm(aggregate) / scale),
        aggregate_vector=aggregate,
        escalations=escalations, escalated_faces=escalated_faces,
        subdivision_levels=int(subdivision_levels),
        subdivision_order=int(subdivision_order),
        subdivision_strips=int(subdivision_strips),
        subdivision_faces=subdivision_faces,
        subdivision_pieces=subdivision_pieces,
        subdivision_seconds=float(division_seconds),
        faces_over_budget=int((estimate > allowed).sum()),
        seconds=time.perf_counter() - started)
    if _profile.enabled():
        _profile.add("adaptive.near_faces", near_count)
        _profile.add("adaptive.far_faces", far_count)
        _profile.add("adaptive.escalations", escalations)
        _profile.add("adaptive.subdivision_faces", subdivision_faces)
        _profile.add("adaptive.subdivision_pieces", subdivision_pieces)
    return total, diagnostics
