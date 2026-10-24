"""Sequential, pre-specified reference suite. Failures remain in the result table."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"closed_v1"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", NUMBA_NUM_THREADS="8")
cases = ["sphere_L2", "sphere_L3", "sphere_L4", "prolate_ellipsoid_L2", "prolate_ellipsoid_L3",
         "prolate_ellipsoid_L4", "sphere_L3_scale_0p001", "sphere_L3_scale_1000",
         "sphere_L3_translate", "sphere_L3_rotate_axis_1_2_3_deg37"]
reference = json.loads((root/"reference"/"analytic_reference.json").read_text())
records = []
for name in cases:
    target = out/name
    target.mkdir(exist_ok=True)
    command = [sys.executable, str(root/"solve.py"), str(root/"reference"/f"{name}.npz"),
               "--out", str(target), "--ncpu", "8"]
    if name.endswith("L4"):
        command += ["--off-surface"]
    if "rotate" in name:
        command += ["--rotate-flow"]
    print(f"Running {name}", flush=True)
    with (target/"run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
    if process.returncode:
        records.append(dict(case=name, passed=False, exit_code=process.returncode))
        continue
    result = json.loads((target/"result.json").read_text())
    surface = np.load(target/"surface.npz")
    mesh = np.load(root/"reference"/f"{name}.npz")
    # Radial projection of flat-face centroids to the smooth reference surface.
    local = (surface["centers"]-mesh["center"]) @ mesh["rotation"] / mesh["scale"]
    local /= np.linalg.norm(local/mesh["axes"], axis=1)[:,None]
    normal = local/mesh["axes"]**2
    normal /= np.linalg.norm(normal, axis=1)[:,None]
    factors = np.array(reference["depolarization_factors"]) if name.startswith("prolate") else np.ones(3)/3.
    analytic_flow = surface["flow"] @ mesh["rotation"]
    potential_gradient = analytic_flow/(1.-factors)
    exact_v = potential_gradient - (normal@potential_gradient)[:,None]*normal
    exact_cp = 1.-np.sum(exact_v**2, axis=1)
    rmse = np.sqrt(np.average((surface["cp"]-exact_cp)**2, weights=surface["area"]))
    gates = dict(source_flux=max(abs(np.array(result["normalized_source_flux"]))) <= 1e-12,
                 centroid_rms=result["collocation_rms"] <= 1e-6,
                 centroid_max=result["collocation_max"] <= 1e-5,
                 pressure_force=np.linalg.norm(result["pressure_force_coefficient"]) < 1e-8)
    if name.endswith("L4"):
        gates["cp_rmse"] = bool(rmse < .08)
        gates["off_collocation"] = result["off_collocation_rms"] < .1
    if name.endswith(("L3","L4")):
        previous = next(row for row in records if row["case"] == name[:-1]+str(int(name[-1])-1))
        ratio = float(rmse/previous["cp_rmse"])
        gates["refinement_ratio"] = ratio < .8
        result["refinement_ratio"] = ratio
    if name.startswith("sphere_L3_"):
        baseline = np.load(out/"sphere_L3"/"surface.npz")
        errors = dict(cp=float(np.max(abs(surface["cp"]-baseline["cp"]))),
                      velocity=float(np.max(abs(surface["velocity"]@mesh["rotation"]-baseline["velocity"]))),
                      normal_residual=float(np.max(abs(surface["normal_residual"]-baseline["normal_residual"]))),
                      force=float(np.max(abs(surface["panel_force_coefficient"]@mesh["rotation"]
                                             -baseline["panel_force_coefficient"]))))
        base_result = json.loads((out/"sphere_L3"/"result.json").read_text())
        errors["flux"] = float(np.max(abs(np.array(result["normalized_source_flux"])
                                         -base_result["normalized_source_flux"])))
        result["transform_errors"] = errors
        gates["transform"] = max(errors.values()) <= 1e-9
    gates = {key:bool(value) for key,value in gates.items()}
    result.update(case=name, cp_rmse=float(rmse), gates=gates, passed=all(gates.values()))
    records.append(result)
    print(f"{name}: Cp RMSE={rmse:.6g}, gates={gates}", flush=True)
(out/"summary.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
assert all(row["passed"] for row in records), "Reference gate failed; inspect summary.json"
