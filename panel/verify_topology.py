"""Recheck fixtures and two adversarial wake geometries after slice validation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.out = args.out.resolve()
assert not args.out.exists()
args.out.mkdir(parents=True)
checks = []
fixtures = root / "cases" / "panel_cut"
for name in ("single_wing", "forward_body", "aft_body", "cambered_body", "swept_tapered",
             "thick_body", "two_bodies_coarse", "two_bodies_fine"):
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface",
               str(fixtures / (name + ".json")), "--out", str(args.out / name), "--validate-only"]
    process = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    checks.append(dict(test=name, expected="accept", passed=process.returncode == 0, command=command))
    print(name, process.returncode, flush=True)
base = json.loads((fixtures / "aft_body.json").read_text())
for name in ("small_solid_piercing_wake", "unclassified_free_wake_boundary"):
    fixture = copy.deepcopy(base)
    points = np.asarray(fixture["coordinates"])
    wake = points[np.asarray(fixture["wake_faces"])]
    i = np.flatnonzero((wake.mean(axis=1)[:, 0] > 4) & (wake.mean(axis=1)[:, 0] < 8))[0]
    center = wake[i].T @ [.6, .2, .2]
    size = np.min(np.linalg.norm(wake[i] - wake[i, [1, 2, 0]], axis=1)) / 1000
    first = len(points)
    if name == "small_solid_piercing_wake":
        fixture["coordinates"] += (center + size * np.array([[-1, -1, -1], [1, -1, -1], [0, 1, -1], [0, 0, 1]])).tolist()
        fixture["faces"] += (first + np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])).tolist()
        fixture["component_ids"] += [20] * 4
        expected = "Wake triangle interior intersects the solid wall slice"
    else:
        fixture["coordinates"] += (center + size * np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]])).tolist()
        fixture["wake_faces"].append([first, first + 1, first + 2])
        expected = "Unclassified free wake boundary"
    path = args.out / (name + ".json")
    path.write_text(json.dumps(fixture), encoding="utf-8")
    output = args.out / name
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface", str(path),
               "--out", str(output), "--validate-only"]
    process = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
    failure = json.loads((output / "failure.json").read_text()) if (output / "failure.json").exists() else {}
    passed = process.returncode != 0 and failure.get("status") == "input_rejected" and expected in failure.get("message", "")
    checks.append(dict(test=name, expected=expected, passed=passed, failure=failure, command=command))
    print(name, passed, failure.get("message"), flush=True)
(args.out / "summary.json").write_text(json.dumps(dict(checks=checks, passed=all(row["passed"] for row in checks)), indent=2), encoding="utf-8")
assert all(row["passed"] for row in checks)
