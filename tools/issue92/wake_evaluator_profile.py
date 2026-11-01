"""Per-target profile of the production wake-velocity evaluator (issue #92, Phase 1 step 3).

The Phase 0 profile established that one coupled wake iteration costs 57.7 s for
18992 target evaluations (3.0 ms/point). This tool decomposes that number into the
stages the evaluator actually runs, per configuration, on the frozen CAD case:

  * analytic_edges      -- method 1, the production default;
  * rm_regularized_p1   -- method 2, with each area quadrature backend.

It changes no production default and writes one JSON file.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from panel import velocity_profile

ROOT = Path(__file__).resolve().parents[1]

def _threads():
    return {"numpy": np.__version__}


def _free_targets(mesh, materials, *, eps_fraction=1e-3):
    """Off-sheet sample points of the free wake nodes.

    The free nodes themselves sit on the sheet, where both the unregularized edge
    integral and the fan projection are singular by construction, so the sampler never
    evaluates them directly: it samples at p + h*n and p - h*n. This mirrors that, with
    an area-weighted sheet normal and the node's own edge scale.
    """
    from panel.p1_coupled_material_wake import node_step_scale

    rows = []
    for group in mesh.groups:
        material = materials[group.name]
        roots = set(int(index) for index in material.roots.ravel())
        xyz = np.asarray(group.wake_xyz, float)
        faces = np.asarray(group.wake_faces, int)
        tri = xyz[faces]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        nodal = np.zeros_like(xyz)
        for corner in range(3):
            np.add.at(nodal, faces[:, corner], normals)
        length = np.linalg.norm(nodal, axis=1)
        nodal[length > 0] /= length[length > 0, None]
        scale = node_step_scale(xyz, faces)
        for node in range(len(xyz)):
            if node in roots or length[node] <= 0 or not np.isfinite(scale[node]):
                continue
            rows.append(xyz[node] + eps_fraction*float(scale[node])*nodal[node])
    return np.asarray(rows, float) if rows else np.empty((0, 3))


def _counters(mesh, materials):
    return {"n_wall_faces": int(len(mesh.faces)),
            "n_wake_triangles": int(sum(len(g.wake_faces) for g in mesh.groups)),
            "n_wake_nodes": int(sum(len(g.wake_xyz) for g in mesh.groups)),
            "n_free_targets": int(len(_free_targets(mesh, materials)))}


def _stages(snapshot):
    """Stage seconds, split into the evaluator's own stages and the fan internals."""
    times = snapshot.get("seconds") or {}
    out = {"evaluator": {}, "fan": {}}
    for key, value in times.items():
        bucket = "fan" if key.startswith(("fan.", "area_terms.")) else "evaluator"
        out[bucket][key] = round(float(value), 6)
    return out


def measure(field, targets, *, label, repeat=1):
    """Evaluate every target one at a time and attribute the time to stages."""
    best = None
    for _ in range(max(1, int(repeat))):
        velocity_profile.reset()
        velocity_profile.enable(True)
        t0 = time.perf_counter()
        for point in targets:
            field(point)
        seconds = time.perf_counter() - t0
        velocity_profile.enable(False)
        snapshot = velocity_profile.snapshot()
        if best is None or seconds < best[0]:
            best = (seconds, snapshot)
    seconds, snapshot = best
    counters = snapshot.get("counters") or {}
    total = int(len(targets))
    return {"label": label,
            "targets": total,
            "repeat": int(repeat),
            "seconds": round(seconds, 6),
            "ms_per_target": round(1000.0 * seconds / max(1, total), 6),
            "stages_seconds": _stages(snapshot),
            "counters": {key: value for key, value in counters.items()
                         if key.startswith(("field.", "fan.", "area_terms.", "total."))},
            "peak_rss_mb": round(velocity_profile.peak_rss_mb(), 1)}


def run(*, targets_limit=None, fan_targets=24, repeat=1, out=None):
    from panel.p1_steady_free_wake import prepare_wake_material
    from tools.issue84.wake_node_rejection import build_case

    from tools.issue92 import fixtures

    source = fixtures._first_existing("runs/half_cad_wingbody_v1/surface.json")
    if source is None:
        raise SystemExit("half_cad_wingbody_v1/surface.json not found")

    mesh, alpha = build_case([str(source)])
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
    materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
    targets = _free_targets(mesh, materials)
    if targets_limit:
        targets = targets[:int(targets_limit)]
    fan_target_set = targets[:min(len(targets), int(fan_targets))]

    records = []
    configurations = [
        ("analytic_edges", dict(method="analytic_edges"), targets),
        ("rm_regularized_p1/legacy_duffy/order48",
         dict(method="rm_regularized_p1", core_radius=0.01,
              area_quadrature="legacy_duffy"), targets),
        ("rm_regularized_p1/projection_fan/16x8",
         dict(method="rm_regularized_p1", core_radius=0.01,
              area_quadrature="projection_fan", fan_order=16, fan_strips=8),
         fan_target_set),
        ("rm_regularized_p1/projection_fan/48x8",
         dict(method="rm_regularized_p1", core_radius=0.01,
              area_quadrature="projection_fan", fan_order=48, fan_strips=8),
         fan_target_set),
        ("rm_regularized_p1/projection_fan_adaptive/16x8",
         dict(method="rm_regularized_p1", core_radius=0.01,
              area_quadrature="projection_fan_adaptive", fan_order=16, fan_strips=8),
         fan_target_set),
    ]
    for label, kwargs, subset in configurations:
        field = TotalVelocity(mesh, state, **kwargs)
        record = measure(field, subset, label=label, repeat=repeat)
        record["config"] = {key: value for key, value in kwargs.items()}
        records.append(record)
        print("%-52s %6d targets  %8.3f s  %7.3f ms/pt" % (
            label, record["targets"], record["seconds"], record["ms_per_target"]))

    payload = {"meta": {"source": str(source).replace("\\", "/"),
                        "alpha_deg": float(alpha),
                        "wall_faces": int(len(mesh.faces)),
                        "wake_triangles": int(sum(len(g.wake_faces) for g in mesh.groups)),
                        "fan_targets": int(len(fan_target_set)),
                        "repeat": int(repeat),
                        "env": _threads()},
               "targets": _counters(mesh, materials),
               "records": records}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("wrote", out)
    return payload


def sweep_strips(mesh, state, targets, *, pairs, reference=(48, 8), out=None):
    """Fan work versus accuracy: (fan_order, strips) against the finest configuration.

    The fan's cost is set by how many quadrature nodes the configuration asks for, not
    by Python overhead, so this sweep reports both the node count and the relative
    velocity error against a fixed reference. It is the evidence behind any claim that
    a cheaper fan is acceptable.
    """
    def evaluate(order, strips):
        field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.01,
                              area_quadrature="projection_fan", fan_order=order,
                              fan_strips=strips)
        velocity_profile.reset()
        velocity_profile.enable(True)
        t0 = time.perf_counter()
        values = np.stack([field(point) for point in targets])
        seconds = time.perf_counter() - t0
        velocity_profile.enable(False)
        snapshot = velocity_profile.snapshot()
        counters = snapshot["counters"]
        return {"velocity": values, "seconds": seconds,
                "subtriangles": int(counters.get("area_terms.subtriangles", 0)),
                "kernel_seconds": float((snapshot["seconds"] or {})
                                        .get("area_terms.kernel", 0.0))}

    ref = evaluate(*reference)
    records = []
    for order, strips in pairs:
        got = evaluate(order, strips)
        delta = np.linalg.norm(got["velocity"] - ref["velocity"], axis=1)
        scale = np.maximum(np.linalg.norm(ref["velocity"], axis=1), 1e-300)
        record = {"fan_order": int(order), "fan_strips": int(strips),
                  "targets": int(len(targets)),
                  "seconds": round(got["seconds"], 6),
                  "ms_per_target": round(1000.0*got["seconds"]/max(1, len(targets)), 6),
                  "subtriangles_per_target":
                      int(round(got["subtriangles"]/max(1, len(targets)))),
                  "kernel_seconds": round(got["kernel_seconds"], 6),
                  "max_relative_error": float(np.max(delta/scale)),
                  "median_relative_error": float(np.median(delta/scale))}
        records.append(record)
        print("fan %2dx%-2d  %8.3f ms/pt  %7d sub/target  max_rel_err %.3e" % (
            order, strips, record["ms_per_target"],
            record["subtriangles_per_target"], record["max_relative_error"]))
    payload = {"reference": {"fan_order": int(reference[0]),
                             "fan_strips": int(reference[1]),
                             "seconds": round(ref["seconds"], 6),
                             "ms_per_target": round(
                                 1000.0*ref["seconds"]/max(1, len(targets)), 6)},
               "targets": int(len(targets)),
               "records": records}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("wrote", out)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/issue92/wake_evaluator_profile.json")
    parser.add_argument("--targets-limit", type=int, default=None)
    parser.add_argument("--fan-targets", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sweep-out", type=Path, default=None)
    parser.add_argument("--sweep-targets", type=int, default=16)
    args = parser.parse_args(argv)
    payload = run(targets_limit=args.targets_limit, fan_targets=args.fan_targets,
                  repeat=args.repeat, out=args.out)
    if args.sweep_out is not None:
        from panel.p1_steady_free_wake import prepare_wake_material
        from tools.issue84.wake_node_rejection import build_case

        mesh, alpha = build_case([payload["meta"]["source"]])
        ops = assemble_multi(mesh)
        state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
        materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
        targets = _free_targets(mesh, materials)[:int(args.sweep_targets)]
        sweep_strips(mesh, state, targets,
                     pairs=[(16, 1), (16, 2), (16, 8), (32, 1), (32, 2), (48, 1),
                            (48, 2), (48, 8)],
                     out=args.sweep_out)


if __name__ == "__main__":
    main()
