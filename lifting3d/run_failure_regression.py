"""Reject a schema-invalid wake even when its separate traces have valid topology."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=root/"results"/"failure_regression_v2")
args = parser.parse_args()
source = root/"meshes"/"wing03"/"mesh.npz"
out = args.out
assert not out.exists(), "Preserve previous regression outputs"
out.mkdir(parents=True)
mesh = dict(np.load(source))
metadata = json.loads(source.with_name("mesh.json").read_text())
permutation = np.arange(len(mesh["wake_lower"]))[::-1]
mesh["wake_lower"] = mesh["wake_lower"][permutation]
mesh["wake_lower_tet"] = mesh["wake_lower_tet"][permutation]
np.savez_compressed(out/"mesh.npz", **mesh)
(out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
command = [sys.executable, str(root/"solve.py"), "--mesh", str(out/"mesh.npz"),
           "--alpha", "5", "--ncpu", "8", "--out", str(out/"solve")]
run = subprocess.run(command, capture_output=True, text=True)
(out/"run.log").write_text(run.stdout+run.stderr, encoding="utf-8")
result = json.loads((out/"solve"/"result.json").read_text())
records = [
    dict(test="nonzero_failure_exit", passed=run.returncode == 2),
    dict(test="failure_clears_converged_flag", passed=result["converged"] is False),
    dict(test="explicit_wake_pairing_failure",
         passed="Wake face rows do not form physical pairs" in result.get("message", "")),
    dict(test="schema_failure_rejected_before_linear_solve", passed=not result["history"]),
]
summary = dict(command=command, returncode=run.returncode, records=records,
               passed=sum(row["passed"] for row in records), total=len(records),
               source_mesh_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               scope="Early schema rejection; previous v1 separately verifies failure state after algebraic convergence")
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
assert summary["passed"] == summary["total"]
