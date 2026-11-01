"""Target-set composition and measured per-target cost of a B2 sampling stage (P0).

The B2 solver samples two target sets per group and per iteration: the wake-face
centroids and the free wake nodes, each through limits() with the epsilon ladder.
This tool rebuilds exactly those targets from the same mesh/state and measures, with
the opt-in profiling counters, how many evaluations the stage really costs:

  * composition: face samples, free-node samples, ladder steps, plus/minus targets,
    and the extra quadrature-refinement pass;
  * cost: measured wall clock for the first N points of each set, and the linear
    scaling to the full set (labelled as an extrapolation, not a measurement).

It never changes the solver and never relaxes a gate. Run it with --max-points small
first: one fan target can cost a second.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel import velocity_profile
from panel.p1_coupled_material_wake import TotalVelocity, limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_steady_free_wake import prepare_wake_material
from tools.issue84.wake_node_rejection import build_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--method", default="analytic_edges",
                        choices=("analytic_edges", "legacy_gauss", "rm_regularized_p1"))
    parser.add_argument("--core-radius", type=float, default=None)
    parser.add_argument("--area-quadrature", default="legacy_duffy",
                        choices=("legacy_duffy", "projection_fan"))
    parser.add_argument("--fan-order", type=int, default=28)
    parser.add_argument("--fan-strips", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--eps-fraction", type=float, default=.05)
    parser.add_argument("--ladder", type=float, nargs="+", default=[1., .5, .25, .125])
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("docs/issue84/b2_target_profile.json"))
    args = parser.parse_args(argv)
    if args.method == "rm_regularized_p1" and (args.core_radius is None
                                               or args.core_radius <= 0):
        raise SystemExit("rm_regularized_p1 needs --core-radius")

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    field = TotalVelocity(mesh, state, method=args.method, core_radius=args.core_radius,
                          area_quadrature=args.area_quadrature,
                          fan_order=args.fan_order, fan_strips=args.fan_strips)
    materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
    solve_seconds = time.perf_counter() - started

    payload = dict(schema="dpanel.issue84.b2_target_profile.v1", stage="P0",
                   python=platform.python_version(), numpy=np.__version__,
                   case=args.case, alpha_deg=alpha_deg, method=args.method,
                   core_radius=args.core_radius, area_quadrature=args.area_quadrature,
                   fan_order=args.fan_order, fan_strips=args.fan_strips,
                   quadrature_order=args.quadrature_order,
                   eps_fraction=args.eps_fraction, ladder=list(args.ladder),
                   max_points=args.max_points, wall_faces=int(len(mesh.faces)),
                   groups={}, solve_seconds=solve_seconds, status="running")
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter() - started
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for group in mesh.groups:
        mat = materials[group.name]
        tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
        normals = np.zeros_like(group.wake_xyz)
        scales = np.full(len(normals), np.inf)
        for k in range(3):
            np.add.at(normals, group.wake_faces[:, k], normal*area[:, None])
            np.minimum.at(scales, group.wake_faces[:, k], edges.min(axis=1))
        free_nodes = np.setdiff1d(mat.used, mat.roots)
        lengths = np.linalg.norm(normals[free_nodes], axis=1)
        node_normals = normals[free_nodes]/lengths[:, None]
        node_scales = scales[free_nodes]
        sets = {"wake_face": (tri.mean(axis=1), normal, edges.min(axis=1)),
                "free_node": (group.wake_xyz[free_nodes], node_normals, node_scales)}
        block = dict(wake_faces=int(len(tri)), free_nodes=int(len(free_nodes)),
                     roots=int(len(mat.roots)), samples={})
        for tag, (points, set_normals, set_scales) in sets.items():
            take = min(args.max_points, len(points))
            velocity_profile.reset()
            velocity_profile.enable(True)
            t0 = time.perf_counter()
            samples = limits(field, points[:take], set_normals[:take], set_scales[:take],
                             eps_fraction=args.eps_fraction, ladder=args.ladder,
                             order=args.quadrature_order, tag=tag)
            elapsed = time.perf_counter() - t0
            velocity_profile.enable(False)
            snap = velocity_profile.snapshot()
            counters = snap["counters"]
            seconds = snap["seconds"]
            full_targets = 2*len(points)*(len(args.ladder) + (0 if args.method == "analytic_edges" else 1))
            per_point = elapsed/take
            row = dict(sampled_points=int(take), points_total=int(len(points)),
                       measured_seconds=round(elapsed, 3),
                       seconds_per_point=round(per_point, 6),
                       extrapolated_seconds=round(per_point*len(points), 3),
                       ladder_steps=int(counters.get("limits.tag."+tag+".targets", 0)//max(1, 2*take)),
                       refined_calls=int(counters.get("limits.tag."+tag+".refined_targets", 0)),
                       counters=counters, seconds={k: round(v, 6) for k, v in seconds.items()},
                       targets_full_set=int(full_targets),
                       finite=bool(np.isfinite(samples["mean"]).all()))
            block["samples"][tag] = row
            print("%s %s %d/%d pts %.3f s -> extrapolated %.1f s" %
                  (group.name, tag, take, len(points), elapsed, row["extrapolated_seconds"]),
                  flush=True)
            flush()
        payload["groups"][group.name] = block
        flush()

    total = 0.0
    for group in payload["groups"].values():
        for row in group["samples"].values():
            total += row["extrapolated_seconds"]
    payload["extrapolated_iteration_seconds"] = round(total, 1)
    payload["note"] = ("extrapolated_seconds is a linear scaling of the measured first "
                       "max_points samples within the real ladder; it is a model, not a "
                       "measurement of the whole iteration")
    payload["status"] = "completed"
    flush()
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
