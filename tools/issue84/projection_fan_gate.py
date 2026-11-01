"""Projection-fan integration gate on the fixed 109-point G2 set (issue #84, stage A5/A6).

This tool exercises the *integrated* evaluator (TotalVelocity with
area_quadrature="projection_fan") rather than calling the projection module
directly, so it measures what production would actually run: P1 edge term,
freestream and face-gradient strength are shared with the legacy path and only
the area integral is swapped.

For every frozen point it reports the total velocity vector difference against a
saturated fan reference, legacy-vs-reference, projection class and dominant-face
aspect, plus per-configuration wall clock. Rows are flushed to disk as they are
produced so an interrupted run still leaves usable data.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_projection_duffy import closest_point_on_triangle
from tools.issue84.projection_gate_set import _aspect, load_points
from tools.issue84.wake_node_rejection import build_case


def _rel(value, reference):
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denominator)


def _stats(values):
    if not values:
        return dict(count=0)
    arr = np.asarray(values, float)
    return dict(count=int(arr.size), min=float(arr.min()), median=float(np.median(arr)),
                p90=float(np.percentile(arr, 90)), max=float(arr.max()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--csv", action="append", required=True, help="label:path")
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--legacy-order", type=int, default=48)
    parser.add_argument("--orders", type=int, nargs="+", default=(16, 32, 48))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="explicit order:strips list; overrides --orders/--strips")
    parser.add_argument("--strips", type=int, nargs="+", default=(8,))
    parser.add_argument("--reference-order", type=int, default=48)
    parser.add_argument("--reference-strips", type=int, default=16)
    parser.add_argument("--stability-order", type=int, default=48)
    parser.add_argument("--stability-strips", type=int, default=64)
    parser.add_argument("--stability-points", type=int, default=8)
    parser.add_argument("--batch-config", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/issue84/projection_fan_gate.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    if args.configs:
        configs = [tuple(int(part) for part in item.split(":")) for item in args.configs]
    else:
        configs = [(o, s) for o in args.orders for s in args.strips]
    fields = {}
    for order, strips in configs:
        fields[(order, strips)] = TotalVelocity(
            mesh, state, method="rm_regularized_p1", core_radius=args.core_radius,
            area_quadrature="projection_fan", fan_order=order, fan_strips=strips)
    reference_field = fields.get((args.reference_order, args.reference_strips))
    if reference_field is None:
        reference_field = TotalVelocity(
            mesh, state, method="rm_regularized_p1", core_radius=args.core_radius,
            area_quadrature="projection_fan", fan_order=args.reference_order,
            fan_strips=args.reference_strips)
    legacy_field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                                 core_radius=args.core_radius)
    field_faces = int(len(legacy_field.triangles))
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

    payload = dict(
        schema="dpanel.issue84.projection_fan_gate.v1", stage="A5_A6",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
        legacy_order=args.legacy_order, orders=list(args.orders), strips=list(args.strips),
        reference=dict(order=args.reference_order, strips=args.reference_strips),
        stability=dict(order=args.stability_order, strips=args.stability_strips,
                       points=args.stability_points),
        wall_faces=int(len(mesh.faces)),
        wake_faces=[int(len(g.wake_faces)) for g in mesh.groups],
        field_faces=field_faces, solve_seconds=solve_seconds,
        error_metric="||v_cfg - v_ref|| / ||v_ref|| on the total velocity vector",
        sets={}, status="running")
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter() - started
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for label, rows in sets.items():
        out_rows = []
        for index, row in enumerate(rows):
            p = np.asarray(row["xyz"], float)
            t0 = time.perf_counter()
            reuse = configs and (args.reference_order, args.reference_strips) in fields
            reference = (np.asarray(fields[(args.reference_order, args.reference_strips)](p), float)
                         if reuse else np.asarray(reference_field(p), float))
            t_ref = time.perf_counter() - t0
            entry = dict(row)
            entry["reference_seconds"] = t_ref
            entry["fan_rel"] = {}
            entry["fan_seconds"] = {}
            for config in configs:
                t0 = time.perf_counter()
                value = np.asarray(fields[config](p), float)
                entry["fan_seconds"]["%d/%d" % config] = time.perf_counter() - t0
                entry["fan_rel"]["%d/%d" % config] = _rel(value, reference)
            t0 = time.perf_counter()
            legacy = np.asarray(legacy_field(p, order=args.legacy_order), float)
            entry["legacy_seconds"] = time.perf_counter() - t0
            entry["legacy_vs_reference_rel"] = _rel(legacy, reference)
            entry["reference_norm"] = float(np.linalg.norm(reference))
            tri_all = legacy_field.triangles
            dominant = _dominant(tri_all, p, legacy_field)
            _q, region = closest_point_on_triangle(tri_all[dominant], p)
            entry["dominant_face"] = dominant
            entry["projection_kind"] = region["kind"]
            entry["projection_index"] = region["index"]
            entry["dominant_aspect"] = _aspect(tri_all[dominant])
            out_rows.append(entry)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                payload["sets"][label] = list(out_rows)
                print("%s %d/%d" % (label, index + 1, len(rows)), flush=True)
                flush()
        payload["sets"][label] = out_rows
        summary = dict(points=len(out_rows),
                       legacy_vs_reference_rel=_stats([e["legacy_vs_reference_rel"] for e in out_rows]),
                       legacy_seconds=_stats([e["legacy_seconds"] for e in out_rows]),
                       reference_seconds=_stats([e["reference_seconds"] for e in out_rows]),
                       fan={}, kinds={}, aspect=_stats([e["dominant_aspect"] for e in out_rows]))
        for config in configs:
            key = "%d/%d" % config
            summary["fan"][key] = dict(
                rel=_stats([e["fan_rel"][key] for e in out_rows]),
                seconds=_stats([e["fan_seconds"][key] for e in out_rows]))
        for entry in out_rows:
            summary["kinds"][entry["projection_kind"]] = summary["kinds"].get(entry["projection_kind"], 0) + 1
        payload["sets"][label + "__summary"] = summary
        flush()

    batch_config = configs[0]
    all_points = np.asarray([r["xyz"] for rows in sets.values() for r in rows], float)
    batch_field = fields.get(batch_config) or reference_field
    t0 = time.perf_counter()
    batch_values = np.asarray(batch_field.evaluate_many(all_points), float)
    batch_seconds = time.perf_counter() - t0
    scalar_values = np.stack([np.asarray(batch_field(point), float) for point in all_points])
    parity = float(np.max(np.abs(batch_values - scalar_values))) if len(all_points) else 0.0
    chunk_parity = None
    regularized = getattr(batch_field, "_regularized", None)
    if regularized is not None:
        one = np.asarray(regularized.evaluate_many(all_points, chunk_size=1), float)
        many = np.asarray(regularized.evaluate_many(all_points, chunk_size=8), float)
        chunk_parity = float(np.max(np.abs(one - many))) if len(all_points) else 0.0
        parity_field = float(np.max(np.abs(one - scalar_values))) if len(all_points) else 0.0
    else:
        parity_field = parity
    payload["batch_parity"] = dict(points=int(len(all_points)),
                                   config="%d/%d" % batch_config,
                                   total_velocity_max_abs_difference=parity,
                                   regularized_field_max_abs_difference=parity_field,
                                   chunk_size_1_vs_8_max_abs_difference=chunk_parity,
                                   batch_seconds=batch_seconds)

    sample = [row for rows in sets.values() for row in rows]
    if sample and args.stability_points:
        indices = np.linspace(0, len(sample) - 1, min(args.stability_points, len(sample))).astype(int)
        stability = []
        stability_field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                                        core_radius=args.core_radius,
                                        area_quadrature="projection_fan",
                                        fan_order=args.stability_order,
                                        fan_strips=args.stability_strips)
        for index in indices:
            p = np.asarray(sample[int(index)]["xyz"], float)
            fine = np.asarray(stability_field(p), float)
            coarse = np.asarray(reference_field(p), float)
            stability.append(dict(set_index=int(index), rel=_rel(coarse, fine),
                                  norm=float(np.linalg.norm(fine))))
        payload["reference_stability"] = dict(
            reference="%d/%d" % (args.reference_order, args.reference_strips),
            finer="%d/%d" % (args.stability_order, args.stability_strips),
            rows=stability,
            rel_max=max(row["rel"] for row in stability))

    payload["status"] = "completed"
    flush()
    print("wrote", out_path)


def _dominant(triangles, point, field):
    """Dominant area-term triangle index (largest |cross(gamma, terms)| weight)."""
    from panel.p1_projection_duffy import area_terms
    terms = area_terms(triangles, np.asarray(point, float), field.core_radius,
                       order=16, strips=1, face_chunk=32)
    return int(np.argmax(np.linalg.norm(np.cross(field._regularized.gamma, terms), axis=1)))


if __name__ == "__main__":
    main()
