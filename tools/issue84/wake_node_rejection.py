"""Per-node rejection diagnostics for the coupled material wake (issue #84, B).

The coupled free-wake update in panel/p1_coupled_material_wake.py decides
whether each free wake node may move.  A node is rejected when

  * epsilon_error   > epsilon_tol   (successive ladder means do not settle)
  * quadrature_error> quadrature_tol (same mean, doubled quadrature order)
  * |V| <= 1e-6                      (no usable tangent)
  * |v[node] + v[predecessor]| <= 2e-6 (opposed endpoints)

This tool reproduces exactly that classification for one case at iteration 0
(the frozen wake geometry) and reports, for every free node, the sampled
velocity, the ladder error, the offset actually used, the local mesh metrics
and the nearest-neighbour distances.  It is read-only: it does not relax,
does not change defaults, and never reclassifies a rejected node as eligible.

Usage (from the repository root):

    python -m tools.issue84.wake_node_rejection \
        --main-half-surface runs/half_cad_wingbody_v1/surface.json \
        --out runs/issue84/wake_rejection_g2.json \
        --csv runs/issue84/wake_rejection_g2.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from panel.p1_coupled_material_wake import (
    TotalVelocity, triangle_data, limits, material_paths,
)
from panel.p1_lifting_g1 import from_dpanel_half_surface, make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.p1_steady_free_wake import prepare_wake_material

REASONS = ("sampling_epsilon", "sampling_quadrature", "zero_speed",
           "opposed_tangent")


def build_case(main_surface, *, case="g2", chord_panels=4, span_panels=4,
               wake_panels=3, alpha_deg=5.):
    """Replicate the G1/G2 assembly of panel/run_p1_coupled_material_wake.py."""
    paths = list(main_surface) if isinstance(main_surface, (list, tuple)) else [main_surface]
    main = from_dpanel_half_surface(Path(paths[0]))
    wings, names, origins = [main], ["main_wing"], [(0., 0., 0.)]
    if case == "g2":
        wings.append(make_wing(chord_panels=chord_panels, span_panels=span_panels,
                               span=1.8, chord=.55, camber=0., wake_panels=wake_panels))
        names.append("raised_tail")
        origins.append((1.6, 0., .35))
    mesh = combine_wings(wings, names=names, origins=origins,
                         reference_chord=main.chord, reference_point=main.reference_point)
    return mesh, alpha_deg


def point_triangle_distance(points, tri):
    """Exact unsigned distance from each point to each triangle (Ericson, RTCD 5.1.5).

    points (P,3), tri (F,3,3) -> (P,F).  Vectorised over both axes; used to
    measure how far a wake node sits from the wall surface, which is what the
    geometry gates would test, not from the nearest wall *vertex*.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b-a, c-a
    ap = points[:, None, :]-a[None, :, :]
    d1 = np.einsum("pfi,fi->pf", ap, ab)
    d2 = np.einsum("pfi,fi->pf", ap, ac)
    bp = points[:, None, :]-b[None, :, :]
    d3 = np.einsum("pfi,fi->pf", bp, ab)
    d4 = np.einsum("pfi,fi->pf", bp, ac)
    cp = points[:, None, :]-c[None, :, :]
    d5 = np.einsum("pfi,fi->pf", cp, ab)
    d6 = np.einsum("pfi,fi->pf", cp, ac)
    va = d3*d6-d5*d4
    vb = d5*d2-d1*d6
    vc = d1*d4-d3*d2
    denom = va+vb+vc
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    v = vb/denom
    w = vc/denom
    u = 1.0-v-w

    def seg(p, x, y):
        d = y-x
        dd = np.einsum("fi,fi->f", d, d)
        t = np.clip(np.einsum("pfi,fi->pf", p[:, None, :]-x[None], d) /
                    np.maximum(dd, 1e-300), 0., 1.)
        q = x[None]+t[..., None]*d[None]
        return np.linalg.norm(p[:, None, :]-q, axis=2)

    inside = (u >= 0) & (v >= 0) & (w >= 0)
    plane = np.abs(np.einsum("pfi,fi->pf",
                             points[:, None, :]-a[None, :, :],
                             np.cross(ab, ac)) / np.linalg.norm(np.cross(ab, ac), axis=1)[None])
    edge = np.minimum(np.minimum(seg(points, a, b), seg(points, b, c)), seg(points, c, a))
    return np.where(inside, plane, edge)


def ladder_trace(field, point, normal, scale, *, eps_fraction, ladder, order):
    """Repeat the production ladder for one node and keep every intermediate mean."""
    uref = float(np.linalg.norm(field.free))
    means, offsets, errors = [], [], []
    previous = None
    for factor in ladder:
        h = eps_fraction*scale*factor
        up, um = field(point+h*normal, order=order), field(point-h*normal, order=order)
        mean = (up+um)/2
        means.append(mean)
        offsets.append(h)
        errors.append(0. if previous is None else
                      float(np.linalg.norm(mean-previous)/uref))
        previous = mean
    return dict(offsets=[float(v) for v in offsets],
                means=[float(v) for v in np.asarray(means).ravel()],
                increments=errors, uref=uref)


def node_table(mesh, state, group, field, *, eps_fraction, ladder, order,
               epsilon_tol, quadrature_tol, wall_tri, other_xyz):
    tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
    mat = prepare_wake_material(group)
    _s, roots = material_paths(mat)
    normals = np.zeros_like(group.wake_xyz)
    scales = np.full(len(group.wake_xyz), np.inf)
    face_count = np.zeros(len(group.wake_xyz), int)
    for k in range(3):
        np.add.at(normals, group.wake_faces[:, k], normal*area[:, None])
        np.minimum.at(scales, group.wake_faces[:, k], edges.min(axis=1))
        np.add.at(face_count, group.wake_faces[:, k], 1)
    free = np.setdiff1d(mat.used, mat.roots)
    lengths = np.linalg.norm(normals[free], axis=1)
    samples = limits(field, group.wake_xyz[free], normals[free]/lengths[:, None],
                     scales[free], eps_fraction=eps_fraction, ladder=ladder, order=order)
    velocity = np.zeros_like(group.wake_xyz)
    velocity[free] = samples["mean"]
    speed = np.linalg.norm(samples["mean"], axis=1)
    eps_error = samples["epsilon_error"]
    quad_error = samples["quadrature_error"]
    finite_limit = (eps_error <= epsilon_tol) & (quad_error <= quadrature_tol)
    fast = speed > 1e-6
    eligible = np.zeros(len(velocity), bool)
    eligible[free] = finite_limit & fast
    opposed = np.zeros(len(velocity), bool)
    material_order = np.full(len(velocity), -1, int)
    material_order[mat.order] = np.arange(len(mat.order))
    for node in mat.order:
        pred = int(mat.predecessor[node])
        if (pred >= 0 and mat.predecessor[pred] >= 0 and eligible[node]
                and eligible[pred] and np.linalg.norm(velocity[node]+velocity[pred]) <= 2e-6):
            eligible[node] = False
            opposed[node] = True

    # Sheet boundary: an undirected edge carried by exactly one wake face.
    edge_use = {}
    for tri_ids in group.wake_faces:
        for a, b in ((tri_ids[0], tri_ids[1]), (tri_ids[1], tri_ids[2]),
                     (tri_ids[2], tri_ids[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_use[key] = edge_use.get(key, 0) + 1
    boundary_node = np.zeros(len(velocity), bool)
    boundary_segments = []
    for (a, b), count in edge_use.items():
        if count == 1:
            boundary_node[a] = boundary_node[b] = True
            boundary_segments.append((group.wake_xyz[a], group.wake_xyz[b]))

    def segment_distance(points):
        """Distance from each point to the nearest sheet-boundary edge segment."""
        if not boundary_segments:
            return np.full(len(points), np.inf)
        best = np.full(len(points), np.inf)
        for x, y in boundary_segments:
            d = y-x
            dd = float(d@d)
            if dd <= 0:
                continue
            t = np.clip((points-x)@d/max(dd, 1e-300), 0., 1.)
            q = x+t[:, None]*d
            best = np.minimum(best, np.linalg.norm(points-q, axis=1))
        return best

    # Local face metrics per node.
    node_area_min = np.full(len(velocity), np.inf)
    node_aspect_max = np.zeros(len(velocity))
    face_aspect = edges.max(axis=1)/np.maximum(edges.min(axis=1), 1e-300)
    for f, tri_ids in enumerate(group.wake_faces):
        for v in tri_ids:
            node_area_min[v] = min(node_area_min[v], area[f])
            node_aspect_max[v] = max(node_aspect_max[v], face_aspect[f])

    tip = float(np.max(np.abs(group.wake_xyz[:, 1]))) if len(velocity) else 0.
    # Nearest other node on the same sheet (k=2 skips the node itself).
    same_tree = cKDTree(group.wake_xyz)
    wall_dist = np.min(point_triangle_distance(group.wake_xyz[free], wall_tri), axis=1)
    edge_dist = segment_distance(group.wake_xyz[free])
    rows = []
    for k, node in enumerate(free):
        reasons = []
        if eps_error[k] > epsilon_tol:
            reasons.append("sampling_epsilon")
        if quad_error[k] > quadrature_tol:
            reasons.append("sampling_quadrature")
        if not fast[k]:
            reasons.append("zero_speed")
        if opposed[node]:
            reasons.append("opposed_tangent")
        if not reasons:
            reasons.append("eligible")
        rows.append(dict(
            group=group.name, node=int(node),
            is_root=bool(np.isin(node, mat.roots)),
            material_order=int(material_order[node]),
            predecessor=int(mat.predecessor[node]),
            root=int(roots[node]),
            xyz=[float(v) for v in group.wake_xyz[node]],
            normal=[float(v) for v in normals[node]/max(lengths[k], 1e-300)],
            scale=float(scales[node]),
            incident_faces=int(face_count[node]),
            local_area_min=float(node_area_min[node]),
            local_aspect_max=float(node_aspect_max[node]),
            on_sheet_boundary=bool(boundary_node[node]),
            near_tip=bool(abs(group.wake_xyz[node, 1]) >= 0.99*tip) if tip else False,
            dist_wall=float(wall_dist[k]),
            dist_sheet_edge=float(edge_dist[k]),
            h_over_edge=(float(edge_dist[k]/samples["eps"][k])
                         if samples["eps"][k] > 0 else None),
            dist_wake_same=float(same_tree.query(group.wake_xyz[node], k=2)[0][1]),
            dist_wake_other=(float(np.min(np.linalg.norm(other_xyz-group.wake_xyz[node], axis=1)))
                             if other_xyz is not None and len(other_xyz) else None),
            velocity=[float(v) for v in velocity[node]],
            speed=float(speed[k]),
            sample_offset=float(samples["eps"][k]),
            h_over_scale=float(samples["eps"][k]/max(scales[node], 1e-300)),
            order=int(order),
            epsilon_error=float(eps_error[k]),
            quadrature_error=float(quad_error[k]),
            eligible=bool(eligible[node]),
            reasons=reasons,
        ))
    table = dict(rows=rows, samples=samples, free=free, mat=mat,
                 velocity=velocity, scales=scales, normals=normals)
    return table


def _finite_or_none(value):
    """JSON has no NaN/Infinity: report 'not applicable' as null instead."""
    value = float(value)
    return value if np.isfinite(value) else None


def classify(rows):
    counts = {name: 0 for name in REASONS}
    counts["eligible"] = 0
    for row in rows:
        primary = row["reasons"][0]
        counts[primary] = counts.get(primary, 0) + 1
    worst = {}
    for name in REASONS:
        members = [r for r in rows if name in r["reasons"] and r["reasons"][0] == name]
        members.sort(key=lambda r: r["epsilon_error"], reverse=True)
        worst[name] = [dict(node=r["node"], group=r["group"], eps=r["epsilon_error"],
                            quad=r["quadrature_error"], speed=r["speed"],
                            dist_wall=r["dist_wall"], dist_sheet_edge=r["dist_sheet_edge"],
                            h_over_edge=r["h_over_edge"], on_sheet_boundary=r["on_sheet_boundary"],
                            near_tip=r["near_tip"], local_aspect_max=r["local_aspect_max"],
                            h_over_scale=r["h_over_scale"])
                       for r in members[:8]]
    return counts, worst


def write_csv(path, rows):
    columns = ["group", "node", "is_root", "material_order", "predecessor", "root",
               "xyz_x", "xyz_y", "xyz_z", "nx", "ny", "nz", "scale", "incident_faces",
               "local_area_min", "local_aspect_max", "on_sheet_boundary", "near_tip",
               "dist_wall", "dist_sheet_edge", "h_over_edge", "dist_wake_same",
               "dist_wake_other", "vx", "vy", "vz",
               "speed", "sample_offset", "h_over_scale", "order", "epsilon_error",
               "quadrature_error", "eligible", "reasons"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for name in columns:
                if name in ("xyz_x", "xyz_y", "xyz_z"):
                    value = row["xyz"][int(name[-1] == "y") * 1 + int(name[-1] == "z") * 2]
                elif name in ("nx", "ny", "nz"):
                    value = row["normal"][{"nx": 0, "ny": 1, "nz": 2}[name]]
                elif name in ("vx", "vy", "vz"):
                    value = row["velocity"][{"vx": 0, "vy": 1, "vz": 2}[name]]
                elif name == "reasons":
                    value = "|".join(row["reasons"])
                elif name in ("dist_wake_other", "h_over_edge"):
                    value = "" if row[name] is None else row[name]
                else:
                    value = row[name]
                values.append(str(value))
            handle.write(",".join(values) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.)
    parser.add_argument("--order", type=int, default=48)
    parser.add_argument("--velocity-method", default="analytic_edges",
                        choices=("analytic_edges", "rm_regularized_p1"))
    parser.add_argument("--core-radius", type=float, default=None)
    parser.add_argument("--eps-fraction", type=float, default=.05)
    parser.add_argument("--ladder", type=float, nargs="+", default=(1., .5, .25, .125))
    parser.add_argument("--epsilon-tol", type=float, default=.02)
    parser.add_argument("--quadrature-tol", type=float, default=.002)
    parser.add_argument("--out", type=Path, default=Path("runs/issue84/wake_rejection.json"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    field = TotalVelocity(mesh, state, method=args.velocity_method,
                          core_radius=args.core_radius)

    groups = list(mesh.groups)
    payload = dict(
        schema="dpanel.issue84.wake_node_rejection.v1",
        python=platform.python_version(), numpy=np.__version__,
        main_half_surface=[str(Path(p)) for p in args.main_half_surface],
        input_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.main_half_surface],
        case=args.case, alpha_deg=alpha_deg, kutta="pressure",
        velocity_method=args.velocity_method, core_radius=args.core_radius,
        quadrature_order=args.order, eps_fraction=args.eps_fraction, ladder=list(args.ladder),
        epsilon_tol=args.epsilon_tol, quadrature_tol=args.quadrature_tol,
        wall_faces=int(len(mesh.faces)), wake_faces=[int(len(g.wake_faces)) for g in groups],
        loads=dict(CL=float(state["CL"]), CL_gamma=float(state["CL_gamma"]),
                   CD_pressure=float(state["CD_pressure"]), CM_y=float(state["CM_y"]),
                   wall_rms=float(state["wall_rms"]),
                   pressure_kutta_rms=float(state["pressure_kutta_rms"]),
                   signed_kutta_rms=float(state["signed_kutta_rms"])),
        nonlinear=state.get("nonlinear"),
        groups={}, classes={}, worst={}, repro=[], notes=[
            "Read-only diagnostic. The production gates are unchanged.",
            "analytic_edges has no finite core, so h/core is not applicable.",
            "quadrature_error is 0 (NOT APPLICABLE) for analytic_edges and active for"
            " rm_regularized_p1.",
        ])
    all_rows = []
    for gi, group in enumerate(groups):
        other = np.concatenate([g.wake_xyz for j, g in enumerate(groups) if j != gi]) \
            if len(groups) > 1 else np.empty((0, 3))
        wall_tri = np.asarray(mesh.xyz, float)[np.asarray(mesh.faces, int)]
        table = node_table(mesh, state, group, field, eps_fraction=args.eps_fraction,
                           ladder=tuple(args.ladder), order=args.order,
                           epsilon_tol=args.epsilon_tol, quadrature_tol=args.quadrature_tol,
                           wall_tri=wall_tri, other_xyz=other)
        rows = table["rows"]
        counts, worst = classify(rows)
        summary = dict(rows=len(rows), roots=int(len(table["mat"].roots)),
                       eligible=int(sum(r["eligible"] for r in rows)),
                       classes=counts,
                       epsilon_error_max=float(max(r["epsilon_error"] for r in rows)),
                       epsilon_error_median=float(np.median([r["epsilon_error"] for r in rows])),
                       quadrature_error_max=float(max(r["quadrature_error"] for r in rows)),
                       rejected_on_boundary=int(sum(1 for r in rows if not r["eligible"]
                                                    and r["on_sheet_boundary"])),
                       rejected_within_1_h_of_edge=int(sum(
                           1 for r in rows if not r["eligible"]
                           and r["dist_sheet_edge"] <= r["sample_offset"])),
                       rejected_within_1_face_of_edge=int(sum(
                           1 for r in rows if not r["eligible"]
                           and r["dist_sheet_edge"] <= 1.5*r["scale"])),
                       rejected_near_tip=int(sum(1 for r in rows if not r["eligible"]
                                                 and r["near_tip"])),
                       rejected_h_over_scale_min=_finite_or_none(min([r["h_over_scale"] for r in rows
                                                            if not r["eligible"]] or [np.nan])),
                       rejected_dist_wall_min=_finite_or_none(min([r["dist_wall"] for r in rows
                                                         if not r["eligible"]] or [np.nan])))
        payload["groups"][group.name] = summary
        payload["classes"][group.name] = counts
        payload["worst"][group.name] = worst
        for row in rows:
            row["group"] = group.name
        all_rows.extend(rows)
        rejected = [r for r in rows if not r["eligible"]]
        rejected.sort(key=lambda r: r["epsilon_error"], reverse=True)
        for row in rejected[:2]:
            node = row["node"]
            payload["repro"].append(dict(
                group=group.name, node=node, epsilon_error=row["epsilon_error"],
                quadrature_error=row["quadrature_error"], reasons=row["reasons"],
                sample_offset=row["sample_offset"], scale=row["scale"],
                xyz=row["xyz"], normal=row["normal"],
                ladder_trace=ladder_trace(field, np.asarray(row["xyz"], float),
                                          np.asarray(row["normal"], float), row["scale"],
                                          eps_fraction=args.eps_fraction,
                                          ladder=tuple(args.ladder), order=args.order)))
    payload["totals"] = {name: sum(c.get(name, 0) for c in payload["classes"].values())
                         for name in list(REASONS) + ["eligible"]}
    payload["seconds"] = time.perf_counter() - started
    payload["csv"] = str(args.csv) if args.csv else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv:
        write_csv(args.csv, all_rows)
    print(json.dumps({k: payload[k] for k in ("totals", "groups", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
