"""Issue #92 S7: parity and cost of the torch regularized-sheet evaluator.

The regularized P1 sheet (RM-P1) is the only velocity method in the coupled free-wake loop
that Route B had not moved off per-target NumPy. This tool measures the port on one real
sheet and one real coupled iteration:

* field parity on the G2 CAD wake (794 triangles, 64 targets) for both area quadratures,
  batched and single-target;
* one coupled wake iteration on the small G2 fixture with `velocity_method='rm_regularized_p1'`
  and every decision the loop records (status, acceptance, rejected sampling nodes);
* wall-clock for both.

Order matters: every NumPy reference is computed BEFORE torch is imported in this process,
because a NumPy LAPACK call after a torch kernel crashes the interpreter (section 6f of
docs/ISSUE92_TORCH_PORT_MAP.md).

  python -m tools.issue92.s7_rm_parity --device cpu

Writes docs/issue92/s7_rm_parity.json and exits non-zero when a declared threshold fails.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

THRESHOLDS = dict(
    declared_before_measurement=True,
    field_relative=1e-12,
    # The load gate covers the frozen-geometry solve only (iteration 0). Later
    # iterations feed the difference through the loop's geometry update and backtracking,
    # which is an amplifier, not a port property; those rows are reported uncalibrated and
    # the first measurement is disclosed in the JSON rather than used to pick a bound.
    loads_relative=1e-9,
    gated_iterations=(0,),
    # Decisions are the stop classification, never the iteration count: section 10.5 shows
    # that which rule fires (and hence nfev) flips under a 1e-13 initial-state perturbation.
    decision_keys=("status", "success", "message"),
    criteria_equal=True,
    speedup_reported_only=True,
)
LOADS = ("CL", "CL_gamma", "CD_pressure", "CM_y")
FRACTION = 0.05


def _parts(mesh, phase=0.7):
    """Synthetic P1 nodal strength on the real wake geometry."""
    parts = []
    for index, group in enumerate(mesh.groups):
        nodal = 0.02 * np.cos(np.arange(group.wake_xyz.shape[0]) * phase + index)
        parts.append((group.wake_xyz, group.wake_faces, nodal[group.wake_faces]))
    return parts


def _targets(mesh, count=64):
    xyz = mesh.groups[0].wake_xyz
    step = max(1, len(xyz) // count)
    pick = xyz[::step][:count]
    offset = np.array([0.0, 0.0, FRACTION])
    return pick + offset


def _field_row(base, other, points, tag):
    reference = base.evaluate_many(points, chunk_size=2)
    value = other.evaluate_many(points, chunk_size=2)
    scale = float(np.abs(reference).max())
    return dict(tag=tag, targets=int(len(points)), scale=scale,
                batch_max_relative=float(np.abs(value - reference).max()/scale))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT/"docs/issue92/s7_rm_parity.json")
    args = parser.parse_args(argv)

    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField
    from tools.issue92.s5_loop_parity import _mesh

    cad = _mesh("g2_cad")
    small = _mesh("g2_conventional")
    parts = _parts(cad)
    points = _targets(cad, args.targets)
    freestream = (0.9961946980917455, 0.0, 0.08715574274765817)
    configs = (("legacy_duffy", dict(quadrature_order=12)),
               ("projection_fan", dict(quadrature_order=48, fan_order=8, fan_strips=1)))

    # -- NumPy references, before torch is imported -----------------------------
    numpy_fields = {}
    numpy_seconds = {}
    for quadrature, keywords in configs:
        field = RegularizedP1SheetField(parts, freestream=freestream, core_radius=0.01,
                                        area_quadrature=quadrature, **keywords)
        numpy_fields[quadrature] = field.evaluate_many(points, chunk_size=2)
        start = time.perf_counter()
        field.evaluate_many(points, chunk_size=2)
        numpy_seconds[quadrature] = time.perf_counter() - start

    # Both loop runs use the torch potential solve, not because it is faster but
    # because the loop calls the linear solver every iteration: a NumPy LAPACK call after
    # any torch kernel aborts the process (exit 0xC0000005, section 6f). Using the same
    # solve backend on both sides also keeps the rm evaluator the only difference.
    loop_arguments = dict(
        iterations=1, velocity_method="rm_regularized_p1", core_radius=0.01,
        area_quadrature="legacy_duffy", fan_order=8, fan_strips=1, gate_mode="audit",
        solve_backend="torch", solve_device="cpu")

    import torch  # noqa: F401  (imported only after every pure NumPy field reference)

    from panel.p1_regularized_sheet_velocity_torch import TorchRegularizedP1SheetField

    report = dict(thresholds=THRESHOLDS, device=args.device,
                  sheet_triangles=int(len(parts) and sum(len(p[1]) for p in parts)),
                  targets=int(len(points)), fields={}, loop={})
    failures = []
    for quadrature, keywords in configs:
        field = TorchRegularizedP1SheetField(parts, freestream=freestream, core_radius=0.01,
                                             area_quadrature=quadrature,
                                             device=args.device, **keywords)
        value = field.evaluate_many(points, chunk_size=2)
        reference = numpy_fields[quadrature]
        scale = float(np.abs(reference).max())
        start = time.perf_counter()
        field.evaluate_many(points, chunk_size=2)
        torch_seconds = time.perf_counter() - start
        row = dict(quadrature=quadrature, scale=scale,
                   batch_max_relative=float(np.abs(value - reference).max()/scale),
                   single_max_relative=float(max(
                       np.abs(field(point) - reference[i]).max()
                       for i, point in enumerate(points))/scale),
                   numpy_seconds=numpy_seconds[quadrature],
                   torch_seconds=torch_seconds,
                   speedup=numpy_seconds[quadrature]/torch_seconds,
                   chunk_budget=int(field._chunk(
                       field.quadrature_order, 16, quadrature == "projection_fan")))
        if row["batch_max_relative"] > THRESHOLDS["field_relative"]:
            failures.append("%s field %.3e" % (quadrature, row["batch_max_relative"]))
        report["fields"][quadrature] = row
        print("%-15s field rel %.3e single %.3e numpy %.3f s torch %.3f s (%.2fx)"
              % (quadrature, row["batch_max_relative"], row["single_max_relative"],
                 row["numpy_seconds"], row["torch_seconds"], row["speedup"]))

    numpy_loop = solve_coupled_material_wake(small, rm_backend="numpy",
                                            **loop_arguments)
    torch_loop = solve_coupled_material_wake(small, rm_backend="torch",
                                             rm_device=args.device, **loop_arguments)
    loop = dict(status_numpy=numpy_loop["status"], status_torch=torch_loop["status"],
                records_numpy=len(numpy_loop["history"]),
                records_torch=len(torch_loop["history"]), rows=[])
    for index, (left, right) in enumerate(zip(numpy_loop["history"], torch_loop["history"])):
        row = dict(iteration=index)
        for key in LOADS:
            row[key] = dict(numpy=float(left[key]), torch=float(right[key]),
                            relative=float(abs(right[key] - left[key])
                                           / max(abs(left[key]), 1e-30)))
            if (index in THRESHOLDS["gated_iterations"]
                    and row[key]["relative"] > THRESHOLDS["loads_relative"]):
                failures.append("it%d %s %.3e" % (index, key, row[key]["relative"]))
        row["recorded_backend"] = dict(numpy=left.get("rm_backend"),
                                       torch=right.get("rm_backend"))
        for key in ("kinematic_candidate", "velocity_method", "accepted"):
            row[key] = dict(numpy=left.get(key), torch=right.get(key))
            if left.get(key) != right.get(key):
                failures.append("it%d decision %s" % (index, key))
        left_nonlinear = left.get("nonlinear") or {}
        right_nonlinear = right.get("nonlinear") or {}
        row["nonlinear"] = dict(
            numpy={key: left_nonlinear.get(key)
                   for key in THRESHOLDS["decision_keys"] + ("nfev", "criteria")},
            torch={key: right_nonlinear.get(key)
                   for key in THRESHOLDS["decision_keys"] + ("nfev", "criteria")})
        for key in THRESHOLDS["decision_keys"]:
            if left_nonlinear.get(key) != right_nonlinear.get(key):
                failures.append("it%d nonlinear %s" % (index, key))
        if THRESHOLDS["criteria_equal"]:
            if left_nonlinear.get("criteria") != right_nonlinear.get("criteria"):
                failures.append("it%d nonlinear criteria" % index)
        loop["rows"].append(row)
        print("iteration %d CL %.12f / %.12f  rm_backend %r / %r"
              % (index, left["CL"], right["CL"], left.get("rm_backend"),
                 right.get("rm_backend")))
    report["loop"] = loop
    report["failures"] = failures
    report["passed"] = not failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote %s passed=%r failures=%r" % (args.out, report["passed"], failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
