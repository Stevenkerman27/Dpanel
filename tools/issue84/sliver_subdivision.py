"""Does subdividing a degenerate near-singular wake face fix the TE sliver? (issue #84)

The adaptive evaluator escalates only the fan order on whole faces. On the one
trailing-edge sliver that failed the declared gate (xyz=(22.333, 0.238, 0), face
scale 0.0185) two escalation rounds stall at 1.9e-3, because the fan maps inside a
triangle whose aspect ratio at the projection point is already degenerate.

This tool answers the question the receipt asked the reviewer: if the near faces are
split in the parameter domain (midpoint 4-way, same piecewise-constant gamma per
piece, each piece integrated with its own projection fan), does the same cost bring
the error below 2e-4?

The area kernel is linear in gamma, so splitting a triangle and summing the pieces is
an exact re-partition of the same integral; only the quadrature changes. The
reference is a deeper subdivision at a higher strip count, and the reference's own
self-convergence is reported.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import panel.p1_projection_duffy as duffy
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from tools.issue84.fan_quadrature_gate import _jsonable, _wake_geometry
from tools.issue84.wake_node_rejection import build_case


def split_faces(triangles, depth):
    """Midpoint 4-way subdivision, repeated depth times. Returns (M,3,3).

    The pieces of face i are the block [4*i : 4*i+4], and this stays true at every
    level, so owner = repeat(arange(N), 4**depth) maps pieces back to their parent.
    """
    tri = np.asarray(triangles, float)
    for _ in range(int(depth)):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab = 0.5 * (a + b)
        bc = 0.5 * (b + c)
        ca = 0.5 * (c + a)
        blocks = np.stack([
            np.stack([a, ab, ca], axis=1),
            np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1),
            np.stack([ab, bc, ca], axis=1),
        ], axis=1)  # (N, 4, 3, 3)
        tri = blocks.reshape(-1, 3, 3)
    return tri


def aspect(triangles):
    tri = np.asarray(triangles, float)
    edges = np.stack([np.linalg.norm(tri[:, (i + 1) % 3] - tri[:, i], axis=1)
                      for i in range(3)], axis=1)
    return edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-30)


def point_triangle_distance(triangle, point):
    """(distance from point to the closed triangle, face scale = max centroid radius)."""
    centre = triangle.mean(axis=0)
    scale = max(float(np.linalg.norm(triangle - centre, axis=1).max()), 1e-12)
    q = _closest(triangle, point)
    return float(np.linalg.norm(np.asarray(point, float) - q)), scale


def _closest(triangle, point):
    """Closest point using the barycentric region returned by closest_point_on_triangle."""
    tri = np.asarray(triangle, float)
    p = np.asarray(point, float)
    a, b, c = tri
    ab, ac = b - a, c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * bc
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _gauss_triangle(rule=7):
    """Dunavant degree-5 (7 point) barycentric nodes and area-normalised weights."""
    if int(rule) != 7:
        raise ValueError("only the 7 point rule is implemented")
    a, b, w = 0.059715871789770, 0.470142064105115, 0.225
    c, d, v = 0.797426985353087, 0.101286507323456, 0.132394152788506
    nodes = np.array([
        [a, b, b], [b, a, b], [b, b, a],
        [c, d, d], [d, c, d], [d, d, c],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    weights = np.array([w, w, w, v, v, v, 0.125939180544827])
    return nodes, weights / weights.sum()


def reference_integral(triangles, gamma, point, core, *, depth=6, rule=7, batch=16):
    """Uniform subdivision plus a degree-5 rule: an independent oracle for the fan.

    The fan is a collapsing-map quadrature; this second family shares no nodes or
    weights with it, so agreement between the two cannot be an aliasing accident.
    """
    nodes, weights = _gauss_triangle(rule)
    p = np.asarray(point, float)
    total = np.zeros(3)
    tri = np.asarray(triangles, float)
    gam = np.asarray(gamma, float)
    for first in range(0, len(tri), batch):
        block = split_faces(tri[first:first + batch], depth)
        source = np.einsum("qv,svi->sqi", nodes, block)
        r = p[None, None, :] - source
        den = (np.einsum("sqi,sqi->sq", r, r) + core * core) ** 1.5
        # area_terms uses the doubled area with the u-weighted Duffy Jacobian, so the
        # matching physical area here is |cross|/2.
        area = 0.5 * np.linalg.norm(np.cross(block[:, 1] - block[:, 0],
                                             block[:, 2] - block[:, 0]), axis=1)[:, None]
        integral = np.einsum("q,sqi->si", weights, r / den[:, :, None]) * area
        per_face = integral.reshape(-1, 4 ** int(depth), 3).sum(axis=1)
        total += np.sum(np.cross(gam[first:first + batch], per_face), axis=0)
    return total, int(len(tri) * 4 ** int(depth))


def area_velocity_split(triangles, gamma, point, core, *, depth, order, strips):
    pieces = split_faces(triangles, depth)
    owner = np.repeat(np.arange(len(triangles)), 4 ** int(depth))
    terms = duffy.area_terms(pieces, point, core, order=int(order), strips=int(strips))
    return np.sum(np.cross(np.asarray(gamma, float)[owner], terms), axis=0), len(pieces)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--near-ratio", type=float, default=2.0)
    parser.add_argument("--point", type=float, nargs=3, default=[22.333, 0.238, 0.0])
    parser.add_argument("--truth-depth", type=int, default=4)
    parser.add_argument("--truth-order", type=int, default=24)
    parser.add_argument("--truth-check-order", type=int, default=32)
    parser.add_argument("--oracle-depth", type=int, default=6)
    parser.add_argument("--configs", default="0:28:8,0:96:8,2:28:8,2:48:8,3:28:8,3:48:8,4:28:8",
                        help="depth:order:strips candidates, comma separated")
    parser.add_argument("--out", type=Path, default=Path("docs/issue84/sliver_subdivision.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    groups = [g for g in mesh.groups if len(g.wake_faces)]
    geometry = _wake_geometry(groups)
    solve_seconds = time.perf_counter() - started

    total = TotalVelocity(mesh, state, method="rm_regularized_p1",
                          core_radius=float(args.core_radius))
    field = total._regularized
    triangles = np.asarray(field.triangles, float)
    gamma = np.asarray(field.gamma, float)
    point = np.asarray(args.point, float)
    core = float(args.core_radius)

    distance = np.empty(len(triangles))
    scale = np.empty(len(triangles))
    for index, triangle in enumerate(triangles):
        distance[index], scale[index] = point_triangle_distance(triangle, point)
    near = (distance <= args.near_ratio * scale) | (distance <= core)
    near_index = np.nonzero(near)[0]
    near_tri = triangles[near_index]
    near_gamma = gamma[near_index]

    rows = []

    def record(label, depth, order, strips):
        begin = time.perf_counter()
        value, pieces = area_velocity_split(near_tri, near_gamma, point, core,
                                           depth=depth, order=order, strips=strips)
        elapsed = time.perf_counter() - begin
        rows.append(dict(label=label, depth=int(depth), order=int(order), strips=int(strips),
                         pieces=int(pieces), value=[float(v) for v in value],
                         seconds=float(elapsed)))
        print("%-22s depth %d fan %d:%d pieces %6d  |v| %.6e  %.3f s"
              % (label, depth, order, strips, pieces, float(np.linalg.norm(value)), elapsed),
              flush=True)
        return value

    truth = record("fan_reference", args.truth_depth, args.truth_order, 8)
    truth_check = record("fan_reference_check", args.truth_depth, args.truth_check_order, 8)
    begin = time.perf_counter()
    oracle, nodes_used = reference_integral(near_tri, near_gamma, point, core,
                                            depth=args.oracle_depth)
    oracle_seconds = time.perf_counter() - begin
    rows.append(dict(label="uniform_family", depth=int(args.oracle_depth), order=0, strips=0,
                     pieces=int(nodes_used), value=[float(v) for v in oracle],
                     seconds=float(oracle_seconds)))
    print("%-22s depth %d 7 point nodes %6d  |v| %.6e  %.3f s"
          % ("uniform_family", args.oracle_depth, nodes_used,
             float(np.linalg.norm(oracle)), oracle_seconds), flush=True)
    for spec in args.configs.split(","):
        depth, order, strips = (int(v) for v in spec.split(":"))
        record("depth%d_%d_%d" % (depth, order, strips), depth, order, strips)

    truth_norm = max(float(np.linalg.norm(truth)), 1e-30)
    by_label = {row["label"]: row for row in rows}
    for row in rows:
        value = np.asarray(row["value"], float)
        row["relative_error"] = float(np.linalg.norm(value - truth) / truth_norm)
        row["error_vs_uniform_family"] = float(np.linalg.norm(value - oracle) / truth_norm)
        row["relative_to_production"] = float(
            np.linalg.norm(value - np.asarray(by_label["depth0_28_8"]["value"], float))
            / truth_norm)
    truth_self = float(np.linalg.norm(truth_check - truth) / truth_norm)
    fan_vs_oracle = float(np.linalg.norm(truth - oracle) / truth_norm)

    inside = aspect(near_tri)
    deepest = split_faces(near_tri, 2)
    payload = dict(
        schema="dpanel.issue84.sliver_subdivision.v1",
        stage="T2-sliver",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=core,
        point=[float(v) for v in point],
        service=dict(wall_faces=int(len(mesh.faces)), wake_triangles=int(len(triangles)),
                     near_faces=int(near_index.size),
                     near_ratio=args.near_ratio,
                     near_aspect_median=float(np.median(inside)),
                     near_aspect_max=float(inside.max()),
                     piece_aspect_max_depth2=float(aspect(deepest).max())),
        truth_self_convergence=truth_self,
        fan_truth_vs_independent=fan_vs_oracle,
        truth=dict(depth=int(args.truth_depth), order=int(args.truth_order),
                   check_order=int(args.truth_check_order)),
        rows=[{k: (_jsonable(v) if k != "value" else v) for k, v in row.items()} for row in rows],
        solve_seconds=solve_seconds,
        wall_seconds=time.perf_counter() - started,
        gate_threshold=2e-4,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("truth self-convergence %.3e" % truth_self)
    for row in rows:
        print("%-22s err vs fan reference %.3e  uniformity %.3e  vs production %.3e" %
              (row["label"], row["relative_error"], row["error_vs_uniform_family"],
               row["relative_to_production"]))
    print("fan reference vs uniform family %.3e (uniform family is not converged on the"
          " sheet; see oracle_offset.json)" % fan_vs_oracle)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
