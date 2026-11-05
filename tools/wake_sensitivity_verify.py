"""Verify frozen B3 evidence and replay iteration zero and the first update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import _one_wake_target, prepare_wake_material, solve_steady_free_wake
from tools.issue92.b3_dpanel import _wing

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--evidence", type=Path, default=root / "runs/verification/issue102_wake_sensitivity")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError("Preserve previous verification: " + str(args.out))
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
started = time.time()
manifest = json.loads((args.evidence / "manifest.json").read_text(encoding="utf-8"))
report = dict(command=[sys.executable, *sys.argv], head=subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    os=platform.platform(), python=sys.version, numpy=np.__version__, torch=torch.__version__,
    threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    evidence_manifest_sha256=hashlib.sha256((args.evidence / "manifest.json").read_bytes()).hexdigest(),
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope="hashes, recorded arithmetic, uniform axial counterexample and fresh 0/1-step solves; not a full E1-E4 rerun",
    physical_accuracy_accepted=False, cases={}, status="running")
assert manifest["execution_complete"] and not manifest["missing_completed_runs_or_audits"]
assert not manifest["process_failures"] and not manifest["physical_accuracy_accepted"]
hash_count = 0
for name, receipt in manifest["files"].items():
    path = args.evidence / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"], name
    hash_count += 1
report["manifest_files_verified"] = hash_count
for case in ("rect", "swept30", "elliptic"):
    folder = args.evidence / case
    baseline = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "baseline_geometry_audit.json").read_text(encoding="utf-8"))
    assert len(baseline["rows"]) == len(audit["rows"]) == 61
    assert hashlib.sha256((folder / "initial_state.npz").read_bytes()).hexdigest() == baseline["initial_state_sha256"]
    for row in baseline["rows"]:
        path = folder / row["snapshot"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["snapshot_sha256"], str(path)
    spec = baseline["spec"]
    if "surface" in spec:
        assert hashlib.sha256((root / spec["surface"]).read_bytes()).hexdigest() == spec["surface_sha256"]
    wing = _wing(spec)
    mesh = combine_wings([wing], reference_chord=spec["cref"], reference_area=spec["sref"],
                         reference_point=wing.reference_point)
    with np.load(folder / "initial_state.npz") as frozen:
        for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
            np.testing.assert_array_equal(getattr(wing, key), frozen[key])
    group = mesh.groups[0]
    used = np.unique(group.wake_faces)
    # The CAD TE declaration excludes two tip wall contacts; check both sets explicitly.
    te = wing.xyz[wing.te_geo]
    separation = np.linalg.norm(group.wake_xyz[used, None] - te[None], axis=2)
    declared = used[separation.min(axis=1) <= 1e-10 * mesh.reference_chord]
    wall_vertices = mesh.xyz[np.unique(mesh.faces)]
    wall_distance = np.linalg.norm(group.wake_xyz[used, None] - wall_vertices[None], axis=2)
    attached = used[wall_distance.min(axis=1) <= 1e-10 * mesh.reference_chord]
    assert np.isin(declared, attached).all()
    extra = np.setdiff1d(attached, declared)
    saved = json.loads((args.evidence / "uniform_axial_kinematics.json").read_text(encoding="utf-8"))["cases"][case]
    np.testing.assert_array_equal(attached, saved["attached_ids"])
    material = prepare_wake_material(group)
    velocity = np.zeros_like(group.wake_xyz)
    velocity[:, 0] = 1.
    target, frozen_nodes = _one_wake_target(group, material, velocity, min_axial=.08, freeze=True)
    displacement = np.linalg.norm(target - group.wake_xyz, axis=1) / mesh.reference_chord
    moving = attached[displacement[attached] > 1e-10]
    np.testing.assert_array_equal(moving, saved["moving_attachment_ids"])
    assert not frozen_nodes
    tri = group.wake_xyz[group.wake_faces]
    cross0 = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tri = target[group.wake_faces]
    cross1 = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    degenerate = int(np.count_nonzero(np.linalg.norm(cross1, axis=1) <= 1e-14 * mesh.reference_chord**2))
    assert degenerate == saved["unrelaxed_target_degenerate_faces"]
    for row, checked in zip(baseline["rows"], audit["rows"]):
        assert row["iteration"] == checked["iteration"]
        with np.load(folder / row["snapshot"]) as snapshot:
            tri = snapshot["wake_xyz"][group.wake_faces]
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            signed = np.einsum("fi,fi->f", cross, cross0) / np.sum(cross0**2, axis=1)
            np.testing.assert_allclose(signed.min(), checked["geometry"]["min_reference_signed_jacobian"], atol=1e-12)
    e1 = json.loads((folder / "e1_position_sensitivity.json").read_text(encoding="utf-8"))
    for derivative in e1["derivatives"]:
        pair = sorted([r for r in e1["rows"] if r.get("region") == derivative["region"]
                       and r.get("direction") == derivative["direction"]
                       and r.get("epsilon_chord") == derivative["epsilon_chord"]
                       and r["status"] == "completed"], key=lambda r: r["sign"])
        if len(pair) == 2:
            for name in ("CL", "CL_gamma", "CM_y"):
                value = (pair[1][name] - pair[0][name]) / (2 * derivative["epsilon_chord"])
                np.testing.assert_allclose(value, derivative["d" + name + "_d_epsilon_chord"], atol=1e-12)
    e4 = json.loads((folder / "e4_conditioning.json").read_text(encoding="utf-8"))
    residuals = []
    for row in e4["rows"]:
        assert row["status"] == "completed"
        for operator in row["operators"].values():
            spectrum = operator["spectrum"]
            np.testing.assert_allclose(spectrum["condition_2_estimate"], spectrum["sigma_max"] / spectrum["sigma_min"])
            residuals.extend([spectrum["largest_eigenpair_relative_residual"],
                              spectrum["inverse_largest_eigenpair_relative_residual"]])
    for phase in ("e2", "e3"):
        for path in sorted(folder.glob(phase + "_*/history.json")):
            if path.parent.name.endswith("_failure"):
                continue
            intervention = json.loads(path.read_text(encoding="utf-8"))
            assert intervention["status"] == "completed"
            initial = path.parent / "initial_mesh.npz"
            assert hashlib.sha256(initial.read_bytes()).hexdigest() == intervention["initial_mesh_sha256"]
            with np.load(initial) as arrays:
                np.testing.assert_array_equal(arrays["wall_xyz"], mesh.xyz)
                np.testing.assert_array_equal(arrays["wall_faces"], mesh.faces)
                assert arrays["gamma_hat"].shape[1] == group.gamma_hat.shape[1]
    result = dict(snapshot_hashes_verified=len(baseline["rows"]), wall_coincident_attachments=len(attached),
                  declared_TE_matches=len(declared), extra_wall_contact_ids=extra.tolist(),
                  extra_wall_contact_coordinates=group.wake_xyz[extra].tolist(),
                  extra_wall_contact_gamma_hat_max=(float(abs(group.gamma_hat[extra]).max()) if len(extra) else None),
                  moving_attachments_in_uniform_axial_flow=len(moving),
                  moving_declared_TE_matches=int(np.count_nonzero(displacement[declared] > 1e-10)),
                  attachment_max_raw_displacement_chord=float(displacement[attached].max()),
                  unrelaxed_degenerate_faces=degenerate, E4_max_recorded_eigenpair_residual=max(residuals),
                  note="E4 arithmetic checked; QR factors and spectra not independently recomputed", replay=[])
    report["cases"][case] = result
    print(case, "input/snapshot hashes and recorded arithmetic verified; replay starting", flush=True)
    samples = []
    replay = solve_steady_free_wake(mesh, alpha_deg=4., kutta="pressure", iterations=1, relaxation=.3,
                                    save_callback=lambda i, current, state, record: samples.append((current, state)))
    for iteration, (current, state) in enumerate(samples):
        original = baseline["rows"][iteration]
        with np.load(folder / original["snapshot"]) as snapshot:
            delta_xyz = float(np.max(np.abs(current.groups[0].wake_xyz - snapshot["wake_xyz"])))
            delta_gamma = float(np.max(np.abs(state["gamma"] - snapshot["gamma"])))
            delta_cp = float(np.max(np.abs(state["cp"] - snapshot["cp"])))
        delta_CL = float(state["CL"] - original["CL"])
        passed = abs(delta_CL) <= 1e-8 and delta_xyz <= 1e-8 and delta_gamma <= 1e-8 and delta_cp <= 1e-6
        result["replay"].append(dict(iteration=iteration, CL=state["CL"], CL_difference=delta_CL,
                                    max_xyz_difference=delta_xyz, max_gamma_difference=delta_gamma,
                                    max_cp_difference=delta_cp, passed=passed))
        print(case, "iteration", iteration, "CL delta=%.3g passed=%s" % (delta_CL, passed), flush=True)
    result["replay_passed"] = all(r["passed"] for r in result["replay"])
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
report["status"] = "completed"
report["seconds"] = time.time() - started
report["replay_checks_passed"] = all(v["replay_passed"] for v in report["cases"].values())
args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
if not report["replay_checks_passed"]:
    raise AssertionError("Replay differs from the saved evidence; inspect receipt, preserve the original evidence")
print("evidence checks and six fresh solves passed; physical accuracy remains unaccepted", flush=True)
