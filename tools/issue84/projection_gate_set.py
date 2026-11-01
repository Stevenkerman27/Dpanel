"""Fixed acceptance set for the projection-aware Duffy integrator (issue #84, C3 step 6).

The G2 wake-node rejection table provides two disjoint sets of nodes where the
shipped single-Duffy quadrature is suspected of being the binding error:

* sampling_quadrature rejects, which appear only with the regularized kernel
  (core = 0.01) and are exactly the aliasing symptom;
* sampling_epsilon rejects, the 102 nodes the unregularized analytic validator
  aborts on, which the regularized kernel is supposed to cover.

For every node this tool evaluates the whole RM field with the production
single-Duffy map and with the projection fan, and reports the difference of the
total velocity vector, the projection class and the aspect ratio of the triangle
that dominates the area term, plus wall clock per point. It is a read-only probe:
the production gate and the default velocity method are untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_projection_duffy import closest_point_on_triangle, field_area_velocity, field_velocity
from tools.issue84.wake_node_rejection import build_case


def _aspect(tri):
    e = np.stack([tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2]])
    length = np.linalg.norm(e, axis=1)
    area2 = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    if area2 <= 0.0:
        return float("inf")
    return float(length.max() / (area2 / length.max()))


def load_points(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("eligible", "").strip().lower() in ("true", "1"):
                continue
            reasons = row.get("reasons", "")
            why = [r for r in ("sampling_epsilon", "sampling_quadrature") if r in reasons]
            if not why:
                continue
            rows.append(dict(
                group=row["group"], node=int(row["node"]), reasons=why,
                xyz=[float(row["xyz_x"]), float(row["xyz_y"]), float(row["xyz_z"])],
                epsilon_error=float(row["epsilon_error"]),
                quadrature_error=float(row["quadrature_error"]),
                aspect_incident=float(row["local_aspect_max"]),
                on_sheet_boundary=row.get("on_sheet_boundary", "") == "True",
            ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--csv", action="append", required=True,
                        help="label:path to a wake_node_rejection CSV")
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--legacy-order", type=int, default=48)
    parser.add_argument("--fan-orders", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--strips", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/issue84/projection_gate_set.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    base = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=args.core_radius)
    field = base._regularized
    solve_seconds = time.perf_counter() - started

    sets = {}
    for spec in args.csv:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit("--csv wants label:path")
        rows = load_points(path)
        if args.limit:
            rows = rows[:args.limit]
        sets[label] = rows

    fan_hi = max(args.fan_orders)
    results = {}
    for label, rows in sets.items():
        out_rows = []
        for row in rows:
            p = np.asarray(row["xyz"], float)
            t0 = time.perf_counter()
            legacy = np.asarray(base(p, order=args.legacy_order), float)
            t_legacy = time.perf_counter() - t0
            fan = {}
            seconds = {}
            for order in args.fan_orders:
                t0 = time.perf_counter()
                fan[order] = np.asarray(field_velocity(field, p, order=order,
                                               strips=args.strips, face_chunk=32), float)
                seconds[order] = time.perf_counter() - t0
            ref = fan[fan_hi]
            value, terms = field_area_velocity(field, p, order=fan_hi,
                                               strips=args.strips, face_chunk=32,
                                               return_face_terms=True)
            dominant = int(np.argmax(np.linalg.norm(terms, axis=1)))
            tri = field.triangles[dominant]
            _q, region = closest_point_on_triangle(tri, p)
            entry = dict(row)
            entry.update(
                legacy_order=args.legacy_order, legacy_seconds=t_legacy,
                fan_seconds={str(o): seconds[o] for o in args.fan_orders},
                legacy_vs_fan_rel=float(np.linalg.norm(legacy - ref) / np.linalg.norm(ref)),
                fan_rel={str(o): float(np.linalg.norm(fan[o] - ref) / np.linalg.norm(ref))
                          for o in args.fan_orders if o != fan_hi},
                dominant_face=dominant, projection_kind=region["kind"],
                projection_index=region["index"], dominant_aspect=_aspect(tri),
            )
            out_rows.append(entry)
        results[label] = out_rows
        kinds = {}
        for entry in out_rows:
            kinds[entry["projection_kind"]] = kinds.get(entry["projection_kind"], 0) + 1
        results[label + "__summary"] = dict(
            points=len(out_rows), projection_kinds=kinds,
            legacy_vs_fan_rel_max=max(e["legacy_vs_fan_rel"] for e in out_rows),
            legacy_vs_fan_rel_median=float(np.median([e["legacy_vs_fan_rel"] for e in out_rows])),
            aspect_max=max(e["dominant_aspect"] for e in out_rows),
            legacy_seconds_median=float(np.median([e["legacy_seconds"] for e in out_rows])),
            fan_seconds_median=float(np.median([e["fan_seconds"][str(fan_hi)] for e in out_rows])),
        )

    payload = dict(
        schema="dpanel.issue84.projection_gate_set.v1",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
        legacy_order=args.legacy_order, fan_orders=list(args.fan_orders), strips=args.strips,
        wall_faces=int(len(mesh.faces)), wake_faces=[int(len(g.wake_faces)) for g in mesh.groups],
        field_faces=int(len(field.triangles)), solve_seconds=solve_seconds,
        total_seconds=time.perf_counter() - started, sets=results,
        notes=[
            "Read-only probe: production gates and the default velocity method are unchanged.",
            "fan_rel is measured against the highest requested fan order, not against an analytic value.",
            "projection_kind describes the triangle that dominates the area term at that target.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for label, summary in results.items():
        if label.endswith("__summary"):
            print(label, json.dumps(summary, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()