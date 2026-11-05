"""Split the saved first B3 displacement into attached and free-node motion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.p1_steady_free_wake import reassemble_wake_only, with_wake_xyz
from tools.issue92.b3_dpanel import _wing
from tools.wake_sensitivity_geometry import audit_geometry, geometry_context

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("rect", "swept30", "elliptic"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
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
    movement = data["wake_xyz"]-xyz0
context = geometry_context(mesh)
geometry0 = audit_geometry(mesh, context)
top = context["topology"]
operators0 = assemble_multi(mesh)
report = dict(case=args.case, schema="dpanel.wake_sensitivity.first_step.v1",
              command=[sys.executable, *sys.argv], frozen_initial_state_sha256=manifest["initial_state_sha256"],
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              attached_ids=top["attached"].tolist(), graph_root_ids=top["graph_roots"].tolist(),
              rows=[], physical_accuracy_accepted=False)
for name, ids in (("attachment_only", top["attached"]),
                   ("free_only", np.setdiff1d(top["used"], top["attached"])),
                   ("all", top["used"])):
    xyz = xyz0.copy()
    xyz[ids] += movement[ids]
    changed = with_wake_xyz(mesh, {mesh.groups[0].name: xyz})
    operators = reassemble_wake_only(changed, operators0)
    state = solve_multi(changed, 4., kutta="pressure", max_nfev=80, operators=operators)
    quality = audit_geometry(changed, context, geometry0)
    row = {k: state[k] for k in ("CL", "CL_gamma", "CM_y", "wall_rms", "jump_rms",
                                 "pressure_kutta_rms", "nonlinear")}
    row.update(name=name, max_displacement_chord=float(
        np.linalg.norm(movement[ids], axis=1).max()/mesh.reference_chord),
        gamma=np.asarray(state["gamma"]).tolist(), geometry=quality,
        W_relative_change=float(np.linalg.norm(operators["W"]-operators0["W"])
                                / np.linalg.norm(operators0["W"])))
    report["rows"].append(row)
    print("%s %s CL=%.9f step=%.6gc Jmin=%.5g" % (
        args.case, name, row["CL"], row["max_displacement_chord"],
        quality["min_reference_signed_jacobian"]), flush=True)
report["all_step_CL_difference"] = report["rows"][-1]["CL"]-manifest["rows"][1]["CL"]
report["status"] = "completed"
(out / "first_step_decomposition.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                  encoding="utf-8")
