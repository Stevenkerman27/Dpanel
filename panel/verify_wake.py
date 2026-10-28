"""Paired wake-direction experiments: identical walls, circulation DOFs and flow."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--audit-python", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists()
args.out.mkdir(parents=True)
root = Path(__file__).resolve().parents[1]
rows, checks = [], []
(args.out / "verify_wake.py.snapshot").write_bytes(Path(__file__).read_bytes())
for case, alphas, moved_mode in (
        ("single_wing", (0., 2.5, 5.), "freestream"),
        ("forward_body", (0., 5.), "freestream"),
        ("aft_body", (0., 5.), "far-freestream"),
        ("swept_tapered", (5.,), "freestream"),
        ("thick_body", (5.,), "far-freestream")):
    for alpha in alphas:
        paired = []
        for mode in ("fixed", moved_mode):
            name = f"{case}_a{alpha:g}_{mode}"
            directory = args.out / name
            surface = root / "cases/panel_cut" / (case + ".json")
            command = [sys.executable, str(root / "panel/solve.py"), "--surface", str(surface),
                       "--out", str(directory), "--alpha", str(alpha), "--wake-mode", mode, "--ncpu", "8"]
            derivatives = alpha == 5 and mode != "fixed" and case in ("single_wing", "swept_tapered", "thick_body")
            if derivatives:
                command.append("--derivatives")
            process = subprocess.run(command, cwd=root, capture_output=True, text=True)
            (args.out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
            assert process.returncode == 0, name
            result = json.loads((directory / "result.json").read_text())
            field = np.load(directory / "fields.npz")
            command_audit = [str(args.audit_python), str(root / "panel/audit.py"), str(directory),
                             "--wake-samples", "64", "--ncpu", "8"]
            process_audit = subprocess.run(command_audit, cwd=root, capture_output=True, text=True)
            (args.out / (name + ".audit.log")).write_text(process_audit.stdout + process_audit.stderr, encoding="utf-8")
            assert process_audit.returncode == 0, name + " independent audit"
            audit = json.loads((directory / "independent_audit.json").read_text())
            row = dict(case=case, alpha=alpha, mode=mode, CL=result["CL"], CD_pressure=result["CD_pressure"],
                       CM_y=result["objectives"]["CM_y"], wall_normal_rms=audit["independent_field"]["wall_normal_rms"],
                       wake_tangency=audit["wake_tangency"], wake_model=result["wake_model"],
                       surface_sha256=result["surface_sha256"], directory=str(directory),
                       solve_command=command, audit_command=command_audit, derivatives_passed=result.get("derivatives_passed"))
            if derivatives:
                checks.append(dict(test=name + "_AD_FD_45_scalars", passed=result["derivatives_passed"]))
            checks.append(dict(test=name + "_independent_kernels", passed=max(audit["kernel_max_abs_errors"].values()) < 1e-10))
            if mode == "freestream":
                checks.append(dict(test=name + "_freestream_tangency", passed=result["wake_model"]["freestream_normal_max"] < 1e-10))
            paired.append(field)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.out / "summary.json").write_text(json.dumps(dict(runs=rows, checks=checks, status="running"), indent=2) + "\n", encoding="utf-8")
        wall_error = float(np.max(abs(paired[0]["coordinates"][paired[0]["faces"]] - paired[1]["coordinates"][paired[1]["faces"]])))
        checks.append(dict(test=f"{case}_a{alpha:g}_same_wall", max_error=wall_error, passed=wall_error == 0.))
        if alpha == 0:
            error = max(float(np.max(abs(paired[0][key] - paired[1][key]))) for key in ("mu", "gamma", "Cp_raw", "coordinates"))
            checks.append(dict(test=case + "_zero_alpha_same_problem", max_error=error, passed=error < 1e-10))
        historical = root / "runs/panel_robust_matrix_v1" / case / "fields.npz"
        if alpha == 5 and historical.exists():
            old = np.load(historical)
            error = max(float(np.max(abs(paired[0][key] - old[key]))) for key in ("mu", "gamma", "Cp_raw"))
            checks.append(dict(test=case + "_historical_fixed_regression", max_error=error, passed=error < 1e-10))
# A contact sheet cannot be rotated while retaining its old wall intersection.
command = [sys.executable, str(root / "panel/solve.py"), "--surface", str(root / "cases/panel_cut/aft_body.json"),
           "--out", str(args.out / "contact_full_rotation_rejected"), "--wake-mode", "freestream", "--ncpu", "8"]
process = subprocess.run(command, cwd=root, capture_output=True, text=True)
(args.out / "contact_full_rotation_rejected.log").write_text(process.stdout + process.stderr, encoding="utf-8")
failure = json.loads((args.out / "contact_full_rotation_rejected/failure.json").read_text())
checks.append(dict(test="contact_requires_new_fluid_cut", passed=process.returncode != 0 and failure["status"] == "input_rejected"
                   and "new CAD fluid cut" in failure["message"]))
summary = dict(status="checks_passed" if all(row["passed"] for row in checks) else "check_failure",
               runs=rows, checks=checks, physical_accuracy_accepted=False,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
assert all(row["passed"] for row in checks)
