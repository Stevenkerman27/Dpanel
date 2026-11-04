"""Verify early, recorded rejection of invalid declared trailing edges.

A non-planar declaration is no longer rejected by a planarity gate: it must
still be rejected because the declared curve has to lie on the CAD trailing
edge, which keeps this negative case meaningful for the 3-D TE contract.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous checks"
args.out.mkdir(parents=True)
cases = root/"results"/"cad_inputs_v1"
inputs = cases/"naca4412_wing"
if not (inputs/"trailing_edge.json").exists():
    inputs = cases/"naca4412_wing_body"  # wing-only case is not tracked in this checkout
declaration = json.loads((inputs/"trailing_edge.json").read_text(encoding="utf-8"))
records = []
for name, expected in [
    ("offset", "trimmed wing boundary curve"),
    ("nonplanar", "trimmed wing boundary curve"),
    ("unsorted", "strictly increasing"),
    ("outside_span", "trimmed wing boundary curve"),
]:
    case = json.loads(json.dumps(declaration))
    if name == "offset":
        for point in case["points"]:
            point[0] += .01
    if name == "nonplanar":
        case["points"][len(case["points"])//2][2] = .01
    if name == "unsorted":
        case["points"][2], case["points"][3] = case["points"][3], case["points"][2]
    if name == "outside_span":
        case["points"][0][1] -= .05
    te_path = args.out/(name+".json")
    te_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    target = args.out/name
    command = [sys.executable, str(root/"build_cad_mesh.py"), "--wing-cad",
               str(inputs/"wing.step"), "--trailing-edge", str(te_path), "--out", str(target),
               "--size", "0.4"]
    run = subprocess.run(command, capture_output=True, text=True)
    (args.out/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    failure = json.loads((target/"failure.json").read_text(encoding="utf-8"))
    passed = run.returncode != 0 and expected in failure["message"] and not (target/"mesh.npz").exists()
    records.append(dict(test=name, passed=passed, command=command, returncode=run.returncode,
                        message=failure["message"]))
(args.out/"results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
(args.out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
