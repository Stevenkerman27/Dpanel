"""Fan quadrature-gate acceptance run (issue #84, task 1).

The fan area integrator used to ignore the gauss order handed to limits(), so the
gate compared a configuration with itself and returned quadrature_error == 0.
This tool verifies the repaired gate on the frozen G2 geometry:

* the 109 frozen points (7 legacy-quadrature rejects + 102 epsilon rejects) get
  limits() with the production fan (28:8 / 48:8), an artificially coarse fan
  (8:8 / 48:8) and the legacy Duffy path (48 vs 96);
* a stratified subset is compared against a saturated reference (48:64) to show
  whether the gate flags the points whose true error exceeds the threshold;
* analytic_edges and legacy_duffy regressions are recorded as booleans;
* a stratified sample of the real B2 target set (face centroids + free nodes)
  is measured the same way.

Rows are flushed after every point so an interrupted run still leaves data.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity, limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_projection_duffy import closest_point_on_triangle
from panel.p1_steady_free_wake import prepare_wake_material
from tools.issue84.projection_gate_set import load_points
from tools.issue84.wake_node_rejection import build_case

TOLERANCE = 0.002


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _stats(values):
    if not len(values):
        return dict(count=0)
    arr = np.asarray(values, float)
    return dict(count=int(arr.size), min=float(arr.min()), median=float(np.median(arr)),
                p90=float(np.percentile(arr, 90)), max=float(arr.max()))


def _field(mesh, state, core, order=None, strips=None, refinement=None,
           refinement_strips=None):
    kwargs = {}
    if order is not None:
        kwargs = dict(area_quadrature="projection_fan", fan_order=int(order),
                      fan_strips=int(strips),
                      fan_refinement_order=int(refinement),
                      fan_refinement_strips=int(refinement_strips))
    return TotalVelocity(mesh, state, method="rm_regularized_p1",
                         core_radius=float(core), **kwargs)


def _wake_geometry(groups):
    geometry = []
    for group in groups:
        tri, normal, _area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
        geometry.append(dict(name=group.name, tri=tri, normal=normal,
                             scale=edges.min(axis=1)))
    return geometry


def _nearest_wake_triangle(geometry, point):
    best = None
    for part in geometry:
        for index in range(len(part["tri"])):
            q = closest_point_on_triangle(part["tri"][index], point)[0]
            distance = float(np.linalg.norm(point - q))
            if best is None or distance < best[0]:
                best = (distance, part["name"], index,
                        np.asarray(part["normal"][index], float),
                        float(part["scale"][index]))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--quad-csv", type=Path,
                        default=Path("runs/issue84/projection_gate_quad.csv"))
    parser.add_argument("--eps-csv", type=Path,
                        default=Path("runs/issue84/projection_gate_eps.csv"))
    parser.add_argument("--production", default="28:8")
    parser.add_argument("--refinement", default="48:8")
    parser.add_argument("--coarse", default="8:8")
    parser.add_argument("--reference", default="48:64")
    parser.add_argument("--reference-points", type=int, default=12)
    parser.add_argument("--target-points", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/issue84/fan_quadrature_gate.json"))
    args = parser.parse_args(argv)

    def pair(text):
        first, _, second = text.partition(":")
        return int(first), int(second)

    production, refinement = pair(args.production), pair(args.refinement)
    coarse = pair(args.coarse)
    reference = pair(args.reference)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    groups = [g for g in mesh.groups if len(g.wake_faces)]
    solve_seconds = time.perf_counter() - started

    prod_field = _field(mesh, state, args.core_radius, *production, refinement=refinement[0],
                        refinement_strips=refinement[1])
    coarse_field = _field(mesh, state, args.core_radius, *coarse, refinement=refinement[0],
                          refinement_strips=refinement[1])
    legacy_field = _field(mesh, state, args.core_radius)
    reference_field = _field(mesh, state, args.core_radius, *reference, refinement=refinement[0],
                             refinement_strips=refinement[1])

    rows = []
    for label, path in (("quadrature", args.quad_csv), ("epsilon", args.eps_csv)):
        for row in load_points(path):
            rows.append(dict(set=label, xyz=np.asarray(row["xyz"], float),
                             reasons=row.get("reasons")))
    if args.limit:
        rows = rows[:args.limit]

    payload = dict(
        schema="dpanel.issue84.fan_quadrature_gate.v1", stage="T1",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
        production=dict(order=production[0], strips=production[1]),
        refinement=dict(order=refinement[0], strips=refinement[1]),
        coarse=dict(order=coarse[0], strips=coarse[1]),
        reference=dict(order=reference[0], strips=reference[1]),
        tolerance=TOLERANCE, wall_faces=int(len(mesh.faces)),
        wake_faces=[int(len(g.wake_faces)) for g in groups],
        field_faces=int(len(prod_field.triangles)), solve_seconds=solve_seconds,
        error_metric="||v - v_reference|| / ||v_reference|| on the mean total velocity",
        points=[], targets=[], status="running")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter() - started
        args.out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    geometry = _wake_geometry(groups)
    ladder = np.array([1.0, .5, .25, .125])
    kwargs = dict(eps_fraction=.05, ladder=ladder, order=48)

    for index, row in enumerate(rows):
        p = row["xyz"]
        t0 = time.perf_counter()
        distance, group, face, normal, scale = _nearest_wake_triangle(geometry, p)
        points = p[None, :]
        normals = normal[None, :]/np.linalg.norm(normal)
        scales = np.array([max(scale, 1e-9)])
        prod = limits(prod_field, points, normals, scales, tag="prod", **kwargs)
        coarse_sample = limits(coarse_field, points, normals, scales, tag="coarse", **kwargs)
        legacy = limits(legacy_field, points, normals, scales, tag="legacy", **kwargs)

        row.update(dict(
            index=index, group=group, face=int(face), distance=float(distance),
            scale=float(scales[0]),
            prod_quadrature_error=float(prod["quadrature_error"][0]),
            prod_epsilon_error=float(prod["epsilon_error"][0]),
            coarse_quadrature_error=float(coarse_sample["quadrature_error"][0]),
            legacy_quadrature_error=float(legacy["quadrature_error"][0]),
            seconds=time.perf_counter()-t0))
        payload["points"].append(row)
        flush()
        print("%s %d/%d prod=%.3e coarse=%.3e legacy=%.3e" % (
            row["set"], index+1, len(rows), row["prod_quadrature_error"],
            row["coarse_quadrature_error"], row["legacy_quadrature_error"]), flush=True)

    regressions = dict(
        analytic_quadrature_error_max=0.0,
        legacy_gate_matches_direct_difference=0.0)
    analytic_velocity = TotalVelocity(mesh, state, method="analytic_edges")
    probe = payload["points"][0] if payload["points"] else None
    if probe is not None:
        p = probe["xyz"]
        _d, _g, _f, normal, scale = _nearest_wake_triangle(geometry, p)
        normals = normal[None, :]/np.linalg.norm(normal)
        scales = np.array([max(scale, 1e-9)])
        analytic = limits(analytic_velocity, p[None, :], normals, scales, tag="analytic",
                          **kwargs)
        regressions["analytic_quadrature_error_max"] = float(
            np.max(np.abs(analytic["quadrature_error"])))
        legacy = limits(legacy_field, p[None, :], normals, scales, tag="legacy_check",
                        **kwargs)
        h = kwargs["eps_fraction"]*scales*ladder[-1]
        targets_direct = np.concatenate((p[None, :]+h[:, None]*normals,
                                         p[None, :]-h[:, None]*normals))
        base_direct = legacy_field.evaluate_many(targets_direct, order=48)
        refined_direct = legacy_field.evaluate_many(targets_direct, order=96)
        mean_direct = (base_direct[:1]+base_direct[1:])/2
        refined_mean = (refined_direct[:1]+refined_direct[1:])/2
        reference_speed = getattr(legacy_field, "freestream", None)
        if reference_speed is None:
            reference_speed = legacy_field.free
        direct = np.linalg.norm(refined_mean-mean_direct, axis=1)/np.linalg.norm(
            reference_speed)
        regressions["legacy_gate_matches_direct_difference"] = float(
            np.max(np.abs(legacy["quadrature_error"]-direct)))
    payload["regressions"] = regressions
    flush()

    truth = []
    if args.reference_points:
        order = np.argsort([row["prod_quadrature_error"] for row in payload["points"]])
        chosen = sorted(set(np.linspace(0, len(order)-1, args.reference_points).astype(int)))
        for position in chosen:
            row = payload["points"][order[position]]
            p = row["xyz"]
            distance, group, face, normal, scale = _nearest_wake_triangle(geometry, p)
            normals = normal[None, :]/np.linalg.norm(normal)
            scales = np.array([max(scale, 1e-9)])
            base = limits(prod_field, p[None, :], normals, scales, tag="truth", **kwargs)
            ref = limits(reference_field, p[None, :], normals, scales, tag="truth", **kwargs)
            error = float(np.linalg.norm(base["mean"][0]-ref["mean"][0])
                          / max(np.linalg.norm(ref["mean"][0]), 1e-30))
            truth.append(dict(index=int(row["index"]), prod_quadrature_error=row["prod_quadrature_error"],
                              coarse_quadrature_error=row["coarse_quadrature_error"],
                              true_error_vs_reference=error))
            payload["truth"] = truth
            flush()
            print("truth %d/%d true=%.3e gate=%.3e" % (len(truth), len(chosen), error,
                  row["prod_quadrature_error"]), flush=True)

    targets = []
    if args.target_points:
        rng = np.random.default_rng(args.seed)
        for group in groups:
            mat = prepare_wake_material(group)
            tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
            free = np.setdiff1d(mat.used, mat.roots)
            picked_faces = rng.choice(len(tri), size=min(args.target_points//2, len(tri)),
                                      replace=False)
            for index in picked_faces:
                targets.append(dict(kind="wake_face", xyz=tri[index].mean(axis=0),
                                    normal=normal[index], scale=float(edges[index].min())))
            node_normals = np.zeros_like(group.wake_xyz)
            node_scale = np.full(len(node_normals), np.inf)
            for k in range(3):
                np.add.at(node_normals, group.wake_faces[:, k], normal*area[:, None])
                np.minimum.at(node_scale, group.wake_faces[:, k], edges.min(axis=1))
            picked_nodes = rng.choice(free, size=min(args.target_points//2, len(free)),
                                      replace=False)
            for node in picked_nodes:
                vector = node_normals[node]
                targets.append(dict(kind="free_node", xyz=group.wake_xyz[node],
                                    normal=vector/np.linalg.norm(vector),
                                    scale=float(node_scale[node])))
        for index, target in enumerate(targets):
            t0 = time.perf_counter()
            prod = limits(prod_field, target["xyz"][None, :],
                          np.asarray(target["normal"], float)[None, :],
                          np.array([target["scale"]]), tag="target", **kwargs)
            record = dict(index=index, kind=target["kind"],
                          prod_quadrature_error=float(prod["quadrature_error"][0]),
                          prod_epsilon_error=float(prod["epsilon_error"][0]),
                          scale=float(target["scale"]),
                          seconds=time.perf_counter()-t0)
            payload["targets"].append(record)
            flush()
            print("target %d/%d q=%.3e (%.2fs)" % (index+1, len(targets),
                  record["prod_quadrature_error"], record["seconds"]), flush=True)

    points = payload["points"]
    payload["summary"] = dict(
        point_count=len(points),
        prod=_stats([row["prod_quadrature_error"] for row in points]),
        coarse=_stats([row["coarse_quadrature_error"] for row in points]),
        legacy=_stats([row["legacy_quadrature_error"] for row in points]),
        prod_flagged=int(sum(row["prod_quadrature_error"] > TOLERANCE for row in points)),
        coarse_flagged=int(sum(row["coarse_quadrature_error"] > TOLERANCE for row in points)),
        gate_flags_every_point_with_true_error=int(sum(
            row["true_error_vs_reference"] > 2e-4 and row["prod_quadrature_error"] > 2e-4
            for row in truth)) if truth else 0,
        truth_count=len(truth),
        truth=_stats([row["true_error_vs_reference"] for row in truth]) if truth else dict(count=0),
        target_count=len(payload["targets"]),
        targets=_stats([row["prod_quadrature_error"] for row in payload["targets"]]),
        targets_flagged=int(sum(row["prod_quadrature_error"] > TOLERANCE
                                for row in payload["targets"])))
    payload["status"] = "completed"
    payload["total_seconds"] = time.perf_counter() - started
    args.out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
