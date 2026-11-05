"""Check the imported wake-map revision against frozen B3 inputs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from panel.p1_lifting_g1 import from_dpanel_half_surface, make_wing
from panel.p1_lifting_g2 import combine_wings
import panel.p1_steady_free_wake as wake

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
started = time.time()
revision = Path(wake.__file__).resolve().parents[1]
report = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=revision,
              text=True).strip(), imported_solver=str(Path(wake.__file__).resolve()),
              numpy=np.__version__, torch=torch.__version__, cases={},
              physical_accuracy_accepted=False)
specs = json.loads((args.inputs / "docs/issue92/b3_cases.json").read_text(encoding="utf-8"))["cases"]
for case in ("rect", "swept30", "elliptic"):
    spec = specs[case]
    if spec["kind"] == "rect":
        wing = make_wing(**spec["make_wing"])
    else:
        surface = args.inputs / spec["surface"]
        assert hashlib.sha256(surface.read_bytes()).hexdigest() == spec["surface_sha256"]
        wing = from_dpanel_half_surface(surface)
    wing = replace(wing, reference_point=np.asarray(spec["moment_reference"], float))
    frozen_folder = args.inputs / "runs/verification/issue102_wake_sensitivity" / case
    with np.load(frozen_folder / "initial_state.npz") as frozen:
        for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
            np.testing.assert_array_equal(getattr(wing, key), frozen[key])
    mesh = combine_wings([wing], reference_chord=spec["cref"],
                         reference_area=spec["sref"], reference_point=wing.reference_point)
    group = mesh.groups[0]
    material = wake.prepare_wake_material(group)
    result = dict(frozen_arrays_equal=True, declared_attachments=len(material.attachments),
                  graph_roots=len(material.roots), checks={}, replay=[])
    report["cases"][case] = result
    for label, vector in (("axial", [1., 0., 0.]), ("oblique", [1., .12, -.07])):
        try:
            target, frozen_nodes = wake._one_wake_target(
                group, material, np.tile(vector, (len(group.wake_xyz), 1)), freeze=True)
            result["checks"][label] = dict(completed=True, frozen_nodes=frozen_nodes,
                max_displacement=float(np.linalg.norm(target-group.wake_xyz, axis=1).max()))
        except Exception:
            result["checks"][label] = dict(completed=False, traceback=traceback.format_exc())

    def save(iteration, current, state, record):
        row = dict(iteration=iteration, CL=float(state["CL"]))
        with np.load(frozen_folder / ("iteration_%03d.npz" % iteration)) as frozen:
            row.update(max_gamma_difference=float(np.max(abs(state["gamma"]-frozen["gamma"]))),
                       max_cp_difference=float(np.max(abs(state["cp"]-frozen["cp"]))),
                       max_wake_difference=float(np.max(abs(current.groups[0].wake_xyz-frozen["wake_xyz"]))))
        result["replay"].append(row)

    try:
        solved = wake.solve_steady_free_wake(mesh, 4., kutta="pressure", iterations=1,
                                            relaxation=.3, save_callback=save)
        result["first_step_completed"] = True
    except Exception:
        result.update(first_step_completed=False, traceback=traceback.format_exc())
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(case, "first_step_completed=", result["first_step_completed"], flush=True)
report["seconds"] = time.time()-started
report["status"] = "completed" if all(r["first_step_completed"] for r in report["cases"].values()) else "blocked"
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("review_status=", report["status"], flush=True)
