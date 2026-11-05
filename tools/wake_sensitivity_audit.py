"""Post-audit saved geometry without repeating or replacing numerical solves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import with_wake_xyz
from tools.issue92.b3_dpanel import _wing
from tools.wake_sensitivity_geometry import audit_geometry, geometry_context

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("rect", "swept30", "elliptic"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out / args.case
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
spec = manifest["spec"]
wing = _wing(spec)
mesh = combine_wings([wing], reference_chord=spec["cref"], reference_area=spec["sref"],
                     reference_point=wing.reference_point)
with np.load(out / "initial_state.npz") as data:
    for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
        np.testing.assert_array_equal(getattr(wing, key), data[key])
    xyz0 = data["state_wake_xyz"].copy()
with np.load(out / "iteration_001.npz") as data:
    actual = data["wake_xyz"]-xyz0
context = geometry_context(mesh)
geometry0 = audit_geometry(mesh, context)
root = Path(__file__).resolve().parents[1]
source_hash = hashlib.sha256((root / "tools/wake_sensitivity_geometry.py").read_bytes()).hexdigest()
receipt = dict(command=[sys.executable, *sys.argv], geometry_source_sha256=source_hash,
               frozen_initial_state_sha256=manifest["initial_state_sha256"],
               original_numerical_outputs_preserved=True,
               correction="actual triangle intersections; exclude only declared shared-point/edge contact; no tolerance change",
               initial_geometry=geometry0, rows=[], physical_accuracy_accepted=False)
for row in manifest["rows"]:
    path = out / row["snapshot"]
    with np.load(path) as data:
        changed = with_wake_xyz(mesh, {mesh.groups[0].name: data["wake_xyz"]})
        finite = bool(np.isfinite(data["gamma"]).all())
    quality = audit_geometry(changed, context, geometry0)
    production_inner = bool(row["nonlinear"]["success"] and row["nonlinear"]["criteria"]["stationary"]
                            and row["jump_rms"] <= 1e-10 and row["pressure_kutta_rms"] <= 1e-4)
    strict_floor = bool(row["nonlinear"]["wall_over_floor"] <= 1.05)
    geometry_valid = bool(quality["min_area2"] > 0 and quality["min_reference_signed_jacobian"] > 0
                          and quality["new_or_deeper_violation_count"] == 0)
    receipt["rows"].append(dict(iteration=row["iteration"], CL=row["CL"], geometry=quality,
                                production_inner_converged=production_inner, wall_at_floor=strict_floor,
                                finite_gamma=finite, geometry_valid=geometry_valid,
                                strict_admissible=production_inner and strict_floor and finite and geometry_valid))
receipt["baseline_status"] = manifest["status"]
(out / "baseline_geometry_audit.json").write_text(json.dumps(receipt, indent=2, allow_nan=False),
                                                 encoding="utf-8")
print(args.case, "baseline audited", len(receipt["rows"]), flush=True)
path = out / "e1_position_sensitivity.json"
if not path.exists():
    sys.exit(0)
e1 = json.loads(path.read_text(encoding="utf-8"))
if e1["status"] != "completed":
    print(args.case, "E1 still running; numerical outputs left untouched", flush=True)
    sys.exit(0)
normals = np.zeros_like(xyz0)
tri = xyz0[mesh.groups[0].wake_faces]
cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
for k in range(3):
    np.add.at(normals, mesh.groups[0].wake_faces[:, k], cross)
used = context["topology"]["used"]
normals[used] /= np.linalg.norm(normals[used], axis=1)[:, None]
streamwise = np.zeros_like(xyz0)
streamwise[used, 0] = 1.
directions = dict(normal=normals, streamwise=streamwise, actual=actual)
receipt.update(rows=[], numerical_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               acceptance_note="production nonlinear success and wall-floor diagnostic reported separately; strict admissibility also requires original 1.05 floor ratio")
for row in e1["rows"]:
    if row["status"] != "completed":
        continue
    ids = np.asarray(e1["regions"][row["region"]], dtype=int)
    direction = directions[row["direction"]]
    scale = np.linalg.norm(direction[ids], axis=1).max()
    xyz = xyz0.copy()
    xyz[ids] += row["sign"]*row["epsilon_chord"]*mesh.reference_chord*direction[ids]/scale
    changed = with_wake_xyz(mesh, {mesh.groups[0].name: xyz})
    quality = audit_geometry(changed, context, geometry0)
    geometry_valid = bool(quality["min_area2"] > 0 and quality["min_reference_signed_jacobian"] > 0
                          and quality["new_or_deeper_violation_count"] == 0)
    production_inner = bool(row["nonlinear"]["success"] and row["nonlinear"]["criteria"]["stationary"]
                            and row["jump_rms"] <= 1e-10 and row["pressure_kutta_rms"] <= 1e-4)
    receipt["rows"].append(dict(name=row["name"], geometry=quality, geometry_valid=geometry_valid,
                                production_inner_converged=production_inner,
                                wall_at_floor=row["nonlinear"]["wall_over_floor"] <= 1.05,
                                strict_admissible=bool(row["inner_converged"] and row["finite_gamma"]
                                                       and geometry_valid)))
receipt["derivatives"] = []
for derivative in e1["derivatives"]:
    matching = [r for r in e1["rows"] if r.get("region") == derivative["region"]
                and r.get("direction") == derivative["direction"]
                and r.get("epsilon_chord") == derivative["epsilon_chord"]]
    names = {r["name"] for r in matching}
    pair = [r for r in receipt["rows"] if r["name"] in names]
    receipt["derivatives"].append(dict(derivative, both_admissible=bool(
        len(pair) == 2 and all(r["strict_admissible"] for r in pair)),
        both_geometry_valid=bool(len(pair) == 2 and all(r["geometry_valid"] for r in pair))))
(out / "e1_geometry_audit.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
print(args.case, "E1 audited", len(receipt["rows"]), flush=True)
