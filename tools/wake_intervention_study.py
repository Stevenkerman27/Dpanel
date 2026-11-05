"""Run one E2 mesh or E3 row intervention on the unchanged B3 P1 equations.

python -m tools.wake_intervention_study --phase e3 --case swept30 --variant freeze_1 --iterations 60 --out runs/verification/issue102_wake_sensitivity
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.p1_steady_free_wake import (
    _node_velocities, _one_wake_target, mean_wake_velocity, prepare_wake_material,
    reassemble_wake_only, with_wake_xyz)
from tools.issue92.b3_dpanel import _wing
from tools.wake_mesh_interventions import clip_wake, refine_wake
from tools.wake_sensitivity_geometry import audit_geometry, geometry_context, spacing


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("e2", "e3"), required=True)
parser.add_argument("--case", choices=("rect", "swept30", "elliptic"), required=True)
parser.add_argument("--variant", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=8)
args = parser.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
case_out = args.out / args.case
out = case_out / (args.phase + "_" + args.variant)
out.mkdir(parents=True, exist_ok=True)
if (out / "history.json").exists():
    raise FileExistsError("Preserve previous run: " + str(out))
baseline_manifest = json.loads((case_out / "manifest.json").read_text(encoding="utf-8"))
spec = baseline_manifest["spec"]
wing = _wing(spec)
original = combine_wings([wing], reference_chord=spec["cref"], reference_area=spec["sref"],
                         reference_point=wing.reference_point)
with np.load(case_out / "initial_state.npz") as data:
    for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
        np.testing.assert_array_equal(getattr(wing, key), data[key])
mesh = original
description = {}
if args.phase == "e2":
    if args.variant.startswith("length_"):
        length = float(args.variant.split("_")[1])
        if length != 30:
            mesh, description = clip_wake(mesh, length)
        else:
            description = dict(length_chord=30, rule="unchanged original B3 wake")
    elif args.variant in ("streamwise", "spanwise"):
        mesh, description = refine_wake(mesh, args.variant)
    elif args.variant == "rect_96" and args.case == "rect":
        fine = make_wing(**dict(spec["make_wing"], wake_panels=8))
        fine = replace(fine, reference_point=wing.reference_point)
        mesh = combine_wings([fine], reference_chord=spec["cref"], reference_area=spec["sref"],
                             reference_point=wing.reference_point)
        description = dict(rule="original rectangular control; wake_panels=8 instead of 2; same 5c sheet and wall")
    else:
        raise ValueError("Unknown E2 variant")
context = geometry_context(mesh)
geometry0 = audit_geometry(mesh, context)
top = context["topology"]
allowance = np.ones(len(mesh.groups[0].wake_xyz))
if args.phase == "e3":
    if args.variant.startswith("freeze_"):
        number = int(args.variant.split("_")[1])
        allowance[(top["rows"] >= 0) & (top["rows"] < number)] = 0.
        description = dict(constrained_rows=list(range(number)),
                           diagnostic_only=True, rule="pin listed initial attachment/BFS rows to initial coordinates")
    elif args.variant == "smooth":
        t = np.clip(top["rows"] / 3., 0., 1.)
        allowance = t*t*(3-2*t)
        description = dict(diagnostic_only=True, transition="smoothstep(row / 3), row 0 fixed and rows >=3 free")
    else:
        raise ValueError("Unknown E3 variant")
assert np.array_equal(mesh.xyz, original.xyz)
assert np.array_equal(mesh.faces, original.faces)
materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
static = assemble_multi(mesh)
report = dict(
    schema="dpanel.wake_sensitivity." + args.phase + ".v1", case=args.case,
    variant=args.variant, description=description,
    frozen_initial_state_sha256=baseline_manifest["initial_state_sha256"],
    settings=baseline_manifest["settings"], iteration_limit=args.iterations,
    spacing=spacing(mesh), initial_geometry=geometry0,
    attachment_ids=top["attached"].tolist(), prepared_root_ids=top["graph_roots"].tolist(),
    row_ids={str(k): np.flatnonzero(top["rows"] == k).tolist()
             for k in sorted(set(top["rows"][top["used"]].tolist()))},
    constrained_ids=np.flatnonzero(allowance == 0).tolist(),
    displacement_allowance=allowance.tolist(), rows=[], physical_accuracy_accepted=False,
    status="running", command=[sys.executable, *sys.argv],
    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__), root / "tools/wake_sensitivity_geometry.py",
                             root / "tools/wake_mesh_interventions.py")},
    criterion="simultaneous inner convergence, positive geometry, no new penetration, change<.002c and slip<.01",
    exit_code=None)
np.savez_compressed(out / "initial_mesh.npz", wall_xyz=mesh.xyz, wall_faces=mesh.faces,
                    wake_xyz=mesh.groups[0].wake_xyz, wake_faces=mesh.groups[0].wake_faces,
                    gamma_hat=mesh.groups[0].gamma_hat)
report["initial_mesh_sha256"] = hashlib.sha256((out / "initial_mesh.npz").read_bytes()).hexdigest()
current = mesh
started = time.time()
stop_next = False
previous_change = None
try:
    for iteration in range(args.iterations + 1):
        ops = static if iteration == 0 else reassemble_wake_only(current, static)
        state = solve_multi(current, 4., kutta="pressure", max_nfev=80, operators=ops)
        panel_velocity, slip = mean_wake_velocity(current, state, probe_fraction=.055,
                                                 step_fraction=.009, block=24)
        quality = audit_geometry(current, context, geometry0)
        nonlinear = state["nonlinear"]
        inner_ok = bool(nonlinear["success"] and nonlinear["criteria"]["stationary"]
                        and nonlinear["wall_over_floor"] <= 1.05
                        and state["jump_rms"] <= 1e-10 and state["pressure_kutta_rms"] <= 1e-4)
        admissible = bool(inner_ok and np.isfinite(state["gamma"]).all()
                          and quality["min_area2"] > 0
                          and quality["min_reference_signed_jacobian"] > 0
                          and quality["new_or_deeper_violation_count"] == 0)
        record = {k: state[k] for k in ("CL", "CL_gamma", "CM_y", "CD_pressure",
                                       "wall_rms", "jump_rms", "pressure_kutta_rms", "nonlinear")}
        record.update(iteration=iteration, gamma=np.asarray(state["gamma"]).tolist(),
                      wakes=slip, geometry=quality, inner_converged=inner_ok,
                      admissible=admissible, incoming_displacement_chord=previous_change,
                      raw_target_displacement_chord=None, accepted_displacement_chord=None)
        path = out / ("iteration_%03d.npz" % iteration)
        np.savez_compressed(path, mu=state["mu"], gamma=state["gamma"],
                            cp=state["cp"], wake_xyz=current.groups[0].wake_xyz)
        record["snapshot"] = path.name
        report["rows"].append(record)
        if iteration == args.iterations or stop_next:
            record["elapsed"] = time.time()-started
            break
        target = {}
        raw_max, accepted_max = 0., 0.
        for group in current.groups:
            velocity = _node_velocities(group, panel_velocity[group.name])
            candidate, frozen = _one_wake_target(group, materials[group.name], velocity,
                                                 min_axial=.08, freeze=True)
            raw = candidate-group.wake_xyz
            new = group.wake_xyz + .3*allowance[:, None]*raw
            fixed = allowance == 0
            new[fixed] = mesh.groups[0].wake_xyz[fixed]
            new[materials[group.name].roots] = materials[group.name].reference_xyz[materials[group.name].roots]
            target[group.name] = new
            raw_max = max(raw_max, float(np.linalg.norm(raw[top["used"]], axis=1).max()
                                         / mesh.reference_chord))
            accepted_max = max(accepted_max, float(np.linalg.norm(
                new[top["used"]]-group.wake_xyz[top["used"]], axis=1).max() / mesh.reference_chord))
            record["axial_guard_frozen_ids"] = frozen
        record.update(raw_target_displacement_chord=raw_max,
                      accepted_displacement_chord=accepted_max, elapsed=time.time()-started)
        report["status"] = "running"
        (out / "history.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("%s %s %s iter=%d CL=%.9f raw=%.4g step=%.4g slip=%.3e valid=%s"
              % (args.case, args.phase, args.variant, iteration, state["CL"], raw_max, accepted_max,
                 max(v["normal_slip_rms"] for v in slip.values()), admissible), flush=True)
        current = with_wake_xyz(current, target)
        previous_change = accepted_max
        stop_next = bool(iteration > 1 and accepted_max < .002
                         and max(v["normal_slip_rms"] for v in slip.values()) < .01)
    last = report["rows"][-1]
    slip_max = max(v["normal_slip_rms"] for v in last["wakes"].values())
    report.update(status="completed", exit_code=0, seconds=time.time()-started,
                  converged=bool(last["admissible"] and previous_change is not None
                                 and previous_change < .002 and slip_max < .01),
                  geometry_stable=bool(previous_change is not None and previous_change < .002),
                  all_states_admissible=all(r["admissible"] for r in report["rows"]))
except Exception as error:
    report.update(status="failed", exit_code=1, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    (out / "history.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
