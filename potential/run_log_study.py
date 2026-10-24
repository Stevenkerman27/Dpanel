"""Fixed log-coordinate study; preserve every failed gate and out-of-scope state."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"log_v1"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
# boundary, Mach, alpha, epsilon, eta, Nr, Nt, R, angular order, wall degree
specs = [("exact", 0., 5., .08, 0., nr, 4*nr, 20., 4, 2) for nr in [32, 64, 128]]
specs += [("exact", 0., 5., .08, 0., 128, 512, 20., order, degree)
          for order, degree in [(2, 2), (4, 3), (4, 4)]]
specs += [("exact", 0., alpha, .08, 0., 128, 512, 20., 4, 2) for alpha in [-5., 0.]]
specs += [("exact", 0., 5., epsilon, 0., 128, 512, 20., 4, 2) for epsilon in [.04, .16, .30]]
specs += [("exact", 0., alpha, .08, .04, 128, 512, 20., 4, 2) for alpha in [0., 5.]]
specs += [("exact", 0., 5., .08, 0., nr, nt, 20., 4, 2) for nr, nt in [(128, 1024), (256, 512)]]
specs += [("exact", .001, 5., .08, 0., 128, 512, 20., 4, 2)]
specs += [("uniform", mach, 5., .08, 0., 64, 256, 20., 4, 2) for mach in [.1, .2, .3]]
specs += [("uniform", .2, 5., .08, 0., 128, 512, 20., 4, 2)]
specs += [("uniform", .2, 5., .08, 0., round(128*np.log(radius)/np.log(20.)), 512, radius, 4, 2)
          for radius in [10., 40.]]
specs += [("uniform", .6, 5., .08, 0., 64, 256, 20., 4, 2)]
records = []
for boundary, mach, alpha, epsilon, eta, nr, nt, radius, order, degree in specs:
    case = f"{boundary}_M{mach:g}_a{alpha:g}_eps{epsilon:g}_eta{eta:g}_{nr}x{nt}_r{radius:g}_o{order}d{degree}"
    target = out/case
    target.mkdir(exist_ok=True)
    command = [sys.executable, str(root/"solve_airfoil_log.py"), "--nr", str(nr), "--nt", str(nt),
               "--mach", str(mach), "--alpha", str(alpha), "--epsilon", str(epsilon), "--eta", str(eta),
               "--radius", str(radius), "--boundary", boundary, "--surface-order", str(order),
               "--wall-degree", str(degree), "--ncpu", "8", "--out", str(target)]
    print(f"Running {case}", flush=True)
    with (target/"run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    row = json.loads((target/"result.json").read_text())
    row.update(case=case, command=command, exit_code=process.returncode)
    gates = {}
    if row["converged"]:
        final = row["history"][-1]
        gates.update(residual_rms=final["residual_divergence_rms"] <= 1e-9,
                     residual_max=final["residual_divergence_max"] <= 1e-7,
                     mass=abs(final["normalized_net_mass_flux"]) < 1e-9,
                     kutta=abs(row["kutta_residual"]) <= 1e-10)
        if boundary == "exact" and mach == 0. and alpha == 5. and epsilon == .08 and eta == 0.:
            if order == 4 and degree == 2 and nr in [64, 128] and nt == 4*nr:
                coarse = next(r for r in records if r["nr"] == nr//2 and r["nt"] == 2*nr)
                gates["cp_refinement"] = row["cp_rmse_exact"] < coarse["cp_rmse_exact"]
                gates["wall_normal_refinement"] = (row["reconstructed_wall_normal_max"]
                                                   < coarse["reconstructed_wall_normal_max"])
            if order == 4 and degree == 2 and nr == 128 and nt == 512:
                gates.update(cp_rmse=row["cp_rmse_exact"] < .002,
                             cl_relative=abs(row["cl_pressure"]/row["cl_exact"]-1.) < .002,
                             pressure_drag=abs(row["cd_pressure"]) < .0005,
                             wall_normal_max=row["reconstructed_wall_normal_max"] < .01)
        if boundary == "exact" and mach == 0. and (alpha != 5. or epsilon != .08 or eta != 0.):
            gates.update(cp_rmse=row["cp_rmse_exact"] < .005,
                         cl_absolute=abs(row["cl_pressure"]-row["cl_exact"]) < .005)
            if alpha == 0. and eta == 0.:
                gates["zero_lift"] = abs(row["cl_pressure"]) < 1e-8
            if alpha == -5. and eta == 0.:
                positive = next(r for r in records if r["nr"] == 128 and r["alpha_deg"] == 5.
                                and r["surface_order"] == 4 and r["wall_degree"] == 2)
                row["cl_antisymmetry_error"] = abs(row["cl_pressure"]+positive["cl_pressure"])
                gates["alpha_antisymmetry"] = row["cl_antisymmetry_error"] < 1e-8
        if mach == .001:
            zero = next(r for r in records if r["mach"] == 0. and r["nr"] == nr and r["nt"] == nt
                        and r["alpha_deg"] == alpha and r["surface_order"] == order and r["wall_degree"] == degree)
            field = np.load(target/"field.npz")
            zero_field = np.load(out/zero["case"]/"field.npz")
            difference = float(np.max(abs(field["cp"]-zero_field["cp"])))
            row["low_mach_max_cp_difference"] = difference
            gates["low_mach"] = difference < 1e-4
    row["gates"] = {key: bool(value) for key, value in gates.items()}
    row["passed_specified_gates"] = bool(row["converged"] and all(gates.values()))
    records.append(row)
    summary = dict(records=records, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{case}: {row['status']}, passed={row['passed_specified_gates']}, "
          f"Cp_RMSE={row.get('cp_rmse_exact')}, vnmax={row.get('reconstructed_wall_normal_max')}", flush=True)
print(f"{sum(row['passed_specified_gates'] for row in records)}/{len(records)} specified gate sets passed.",
      flush=True)
