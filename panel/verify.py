"""Run the public geometry/flow/AD/invariance/rejection matrix, preserving logs."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--audit-python", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.out = args.out.resolve()
assert not args.out.exists(), "Preserve previous experiment matrices"
args.out.mkdir(parents=True)
started = time.perf_counter()
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np

(args.out / "verify.py.snapshot").write_bytes(Path(__file__).read_bytes())
checks, runs = [], []
fixtures = root / "cases" / "panel_cut"
baseline_names = ["single_wing", "forward_body", "aft_body", "cambered_body",
                  "swept_tapered", "thick_body", "two_bodies_coarse"]
experiments = [(name, name, [], name in ("single_wing", "aft_body", "two_bodies_coarse"))
               for name in baseline_names]
experiments += [
    ("aft_negative_alpha", "aft_body", ["--alpha", "-5"], False),
    ("aft_zero_alpha", "aft_body", ["--alpha", "0"], False),
    ("aft_bulge_span", "aft_body", ["--bulge", ".4", "--span", ".7"], True),
    ("aft_thin_wide", "aft_body", ["--bulge", "-.5", "--span", "1.4"], False),
    ("thick_negative_alpha", "thick_body", ["--alpha", "-8"], False),
    ("two_bodies_zero_alpha", "two_bodies_coarse", ["--alpha", "0"], False),
]
if (fixtures / "two_bodies_fine.json").exists():
    experiments.append(("two_bodies_fine", "two_bodies_fine", [], False))
for name, fixture, options, derivatives in experiments:
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface",
               str(fixtures / (fixture + ".json")), "--out", str(args.out / name), "--ncpu", "8", *options]
    if derivatives:
        command.append("--derivatives")
    process = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    result_path = args.out / name / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    passed = process.returncode == 0 and result.get("geometry_robustness_accepted", False)
    checks.append(dict(test="valid_" + name, passed=passed))
    if derivatives:
        checks.append(dict(test="AD_FD_" + name, scalar_comparisons=45, passed=result.get("derivatives_passed", False)))
    runs.append(dict(name=name, command=command, returncode=process.returncode, result=result))
    print(json.dumps(dict(case=name, passed=passed, CL=result.get("CL"), Cp_min=result.get("cp_min"))), flush=True)

base_path = args.out / "aft_body"
base = json.loads((base_path / "result.json").read_text())
base_fields = np.load(base_path / "fields.npz")
base_fixture = json.loads((fixtures / "aft_body.json").read_text())
for name, scale, shift, reference_offset in [
    ("scale_small_translate", .001, [2.1, -3.3, 1.7], [0., 0., 0.]),
    ("scale_large_translate", 1000., [-210., 330., -170.], [0., 0., 0.]),
    ("moment_reference", 1., [0., 0., 0.], [.2, -.3, .4]),
]:
    fixture = copy.deepcopy(base_fixture)
    fixture["coordinates"] = (np.asarray(fixture["coordinates"]) * scale + shift).tolist()
    fixture["origin"] = (np.asarray(fixture["origin"]) * scale + shift).tolist()
    fixture["reference_chord"] *= scale
    fixture["reference_area"] *= scale**2
    fixture["moment_reference"] = (np.asarray(fixture["moment_reference"]) * scale + shift + np.asarray(reference_offset)).tolist()
    fixture["gamma_stations"] = (np.asarray(fixture["gamma_stations"]) * scale + shift[1]).tolist()
    fixture_path = args.out / (name + ".json")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface", str(fixture_path), "--out", str(args.out / name)]
    process = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        checks.append(dict(test=name, passed=False, returncode=process.returncode))
        continue
    result = json.loads((args.out / name / "result.json").read_text())
    fields = np.load(args.out / name / "fields.npz")
    cp_error = float(np.max(abs(fields["Cp_raw"] - base_fields["Cp_raw"])))
    force_error = float(np.max(abs(np.asarray(result["force"]) / scale**2 - base["force"])))
    expected_moment = np.asarray(base["moment"]) - np.cross(np.asarray(reference_offset) / scale, base["force"])
    moment_error = float(np.max(abs(np.asarray(result["moment"]) / scale**3 - expected_moment)))
    checks.append(dict(test=name, max_cp_error=cp_error, max_force_scaling_error=force_error,
                       max_moment_transform_error=moment_error,
                       passed=bool(max(cp_error, force_error, moment_error) < 1e-8)))

# Each mutation changes the actual input. No expected diagnostic is injected
# into the solver, and failures must be classified before the linear solve.
bad_names = ["open_wall", "duplicate_wall", "flipped_wall", "wrong_te_banks", "missing_cut",
             "wrong_te_list", "nonplanar_wake", "wake_inside_solid", "missing_contact_constraint",
             "unsupported_gamma", "collapsed_panel", "collapsed_span", "nonfinite_parameter"]
rejections = []
for name in bad_names:
    fixture = copy.deepcopy(base_fixture)
    options = []
    if name == "open_wall":
        fixture["faces"].pop()
        fixture["component_ids"].pop()
    elif name == "duplicate_wall":
        fixture["faces"].append(fixture["faces"][0])
        fixture["component_ids"].append(fixture["component_ids"][0])
    elif name == "flipped_wall":
        fixture["faces"][0].reverse()
    elif name == "wrong_te_banks":
        fixture["upper_te_faces"][0] = fixture["lower_te_faces"][0]
    elif name == "missing_cut":
        fixture["cut_edges"].pop()
        fixture["cut_face_pairs"].pop()
        fixture["cut_kind"].pop()
    elif name == "wrong_te_list":
        fixture["trailing_edges"][0] = fixture["trailing_edges"][1]
    elif name == "nonplanar_wake":
        wall_nodes = set(np.asarray(fixture["faces"]).ravel().tolist())
        wake_node = next(node for node in np.asarray(fixture["wake_faces"]).ravel().tolist() if node not in wall_nodes)
        fixture["coordinates"][wake_node][2] += .01
    elif name == "wake_inside_solid":
        # This closed extra triangular sheet has no shared wall edge; only the
        # independent solid-angle test detects its location inside the body.
        first = len(fixture["coordinates"])
        fixture["coordinates"] += [[.9, -.01, 0.], [.92, -.01, 0.], [.91, .01, 0.]]
        fixture["wake_faces"].append([first, first + 1, first + 2])
    elif name == "missing_contact_constraint":
        fixture["audit"]["wall_contact_intervals"] = []
        values = fixture["gamma_station_dof"]
        merged = next(value for value in values if value >= 0 and values.count(value) > 1)
        station = values.index(merged) + 1
        values[station] = max(values) + 1
    elif name == "unsupported_gamma":
        values = fixture["gamma_station_dof"]
        merged = next(value for value in values if value >= 0 and values.count(value) > 2)
        indices = [i for i, value in enumerate(values) if value == merged]
        values[indices[len(indices) // 2]] = max(values) + 1
    elif name == "collapsed_panel":
        fixture["faces"][0][1] = fixture["faces"][0][0]
    elif name == "collapsed_span":
        options = ["--span", "0"]
    elif name == "nonfinite_parameter":
        options = ["--alpha", "nan"]
    fixture_path = args.out / ("invalid_" + name + ".json")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    output = args.out / ("invalid_" + name)
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface", str(fixture_path),
               "--out", str(output), "--validate-only", *options]
    process = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.out / ("invalid_" + name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    failure = json.loads((output / "failure.json").read_text()) if (output / "failure.json").exists() else {}
    passed = process.returncode != 0 and failure.get("status") == "input_rejected"
    checks.append(dict(test="reject_" + name, passed=passed))
    rejections.append(dict(name=name, diagnostic=failure, command=command))
    print(json.dumps(dict(rejection=name, passed=passed, diagnostic=failure.get("message"))), flush=True)

if args.audit_python:
    for name in ("aft_body", "thick_body", "two_bodies_coarse"):
        command = [str(args.audit_python), str(root / "panel" / "audit.py"), str(args.out / name), "--off-surface"]
        process = subprocess.run(command, capture_output=True, text=True, cwd=root)
        (args.out / (name + ".audit.log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        checks.append(dict(test="independent_audit_" + name, passed=process.returncode == 0))
        print(json.dumps(dict(audit=name, completed=process.returncode == 0)), flush=True)
summary = dict(status="passed" if all(row["passed"] for row in checks) else "failed",
               checks=checks, runs=runs, rejections=rejections,
               elapsed_seconds=time.perf_counter() - started,
               interpretation="Geometry and implementation checks only. Independent wall-field errors retained; physical accuracy not accepted.")
(args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(dict(status=summary["status"], passed=sum(row["passed"] for row in checks), total=len(checks))), flush=True)
assert summary["status"] == "passed"
