"""Compare frozen iteration-0 velocity-kernel runs for Issue #84.

The B2 solver re-solves the same unregularized P1 strengths in every run.
This utility verifies that geometry/strengths/Sref/alpha/Kutta are identical,
then reports changes in the reconstructed wake velocity samples and slip.
It does not select a core radius or claim physical accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LOADS = ("CL", "CL_gamma", "CD_pressure", "CM_y")


def read_run(path):
    path = Path(path)
    with (path/"receipt.json").open(encoding="utf-8") as f:
        receipt = json.load(f)
    with (path/"iteration_000.json").open(encoding="utf-8") as f:
        iteration = json.load(f)
    with np.load(path/"iteration_000.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return receipt, iteration, arrays


def compare_runs(reference, candidate, *, state_atol=1e-10, state_rtol=1e-10):
    ref_receipt, ref_iter, ref = read_run(reference)
    cand_receipt, cand_iter, cand = read_run(candidate)
    ref_params = ref_receipt.get("parameters", {})
    cand_params = cand_receipt.get("parameters", {})
    invariant_parameters = ("case", "alpha", "kutta", "chord_panels",
                            "span_panels", "wake_panels", "main_half_surface")
    parameter_differences = {
        key: [ref_params.get(key), cand_params.get(key)]
        for key in invariant_parameters if ref_params.get(key) != cand_params.get(key)
    }
    for key in ("reference_area", "reference_chord"):
        if ref_receipt.get(key) != cand_receipt.get(key):
            parameter_differences[key] = [ref_receipt.get(key), cand_receipt.get(key)]
    geometry_keys = [k for k in ref if k.endswith(("_xyz", "_faces", "_dof_face",
                                                    "_gamma_hat", "_roots",
                                                    "_predecessor", "_used"))]
    missing = sorted(set(ref)-set(cand))
    geometry_differences = {}
    for key in geometry_keys:
        if key not in cand or not np.array_equal(ref[key], cand[key]):
            geometry_differences[key] = None if key not in cand else float(
                np.max(np.abs(np.asarray(ref[key], float)-np.asarray(cand[key], float))))
    state_differences = {}
    for key in ("mu", "gamma", "cp"):
        if key not in cand:
            state_differences[key] = None
            continue
        if not np.allclose(ref[key], cand[key], atol=state_atol, rtol=state_rtol):
            state_differences[key] = float(np.max(np.abs(ref[key]-cand[key])))
    load_differences = {
        key: float(cand_iter[key]-ref_iter[key])
        for key in LOADS if key in ref_iter and key in cand_iter
    }
    velocity_differences = {}
    for key in sorted(ref):
        if key.endswith(("_node_mean", "_face_mean")) and key in cand:
            delta = np.asarray(cand[key])-np.asarray(ref[key])
            velocity_differences[key] = dict(
                max=float(np.max(np.linalg.norm(delta, axis=-1))),
                rms=float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1)))))
    loads_match = all(abs(value) <= 1e-9 for value in load_differences.values())
    slip = {}
    for group, ref_diag in ref_iter.get("wakes", {}).items():
        cand_diag = cand_iter.get("wakes", {}).get(group, {})
        slip[group] = dict(
            reference=ref_diag.get("overall", {}).get("slip_rms"),
            candidate=cand_diag.get("overall", {}).get("slip_rms"))
    return dict(
        schema="dpanel.issue84.iteration0_ab.v1",
        reference=str(reference), candidate=str(candidate),
        reference_method=ref_iter.get("velocity_method"),
        candidate_method=cand_iter.get("velocity_method"),
        reference_core_radius=ref_iter.get("core_radius"),
        candidate_core_radius=cand_iter.get("core_radius"),
        parameter_differences=parameter_differences,
        missing_candidate_arrays=missing,
        geometry_differences=geometry_differences,
        state_differences=state_differences,
        load_differences=load_differences,
        loads_match=loads_match,
        velocity_differences=velocity_differences,
        slip_rms=slip,
        same_frozen_case=(not parameter_differences and not geometry_differences
                          and not state_differences and not missing and loads_match),
        physical_accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="iteration-0 analytic_edges run directory")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="iteration-0 RM-regularized P1 run directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.reference, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["same_frozen_case"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
