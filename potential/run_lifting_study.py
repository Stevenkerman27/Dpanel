"""Run the fixed lifting protocol, retaining convergence and physical failures separately."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"lifting_v1"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
specs = [("exact", 0., 5., 0., nr, 4*nr, 20.) for nr in [32, 64, 128]]
specs += [("exact", 0., alpha, 0., 128, 512, 20.) for alpha in [-5., 0.]]
specs += [("exact", 0., alpha, .04, 128, 512, 20.) for alpha in [0., 5.]]
specs += [("exact", .001, 5., 0., 128, 512, 20.)]
specs += [("uniform", mach, 5., 0., 64, 256, 20.) for mach in [0., .1, .2, .3]]
specs += [("uniform", .2, 5., 0., 128, 512, 20.)]
specs += [("uniform", .2, 5., 0., round(128*np.log(radius)/np.log(20.)), 512, radius)
          for radius in [10., 40.]]
specs += [("uniform", .6, 5., 0., 64, 256, 20.)]
source_hash = hashlib.sha256((root/"solve_airfoil.py").read_bytes()).hexdigest()
protocol_hash = hashlib.sha256((root/"LIFTING_PROTOCOL.md").read_bytes()).hexdigest()
records = []
for boundary, mach, alpha, eta, nr, nt, radius in specs:
    name = f"{boundary}_M{mach:g}_a{alpha:g}_e{eta:g}_{nr}x{nt}_r{radius:g}"
    target = out/name
    target.mkdir(exist_ok=True)
    command = [sys.executable, str(root/"solve_airfoil.py"), "--nr", str(nr), "--nt", str(nt),
               "--radius", str(radius), "--mach", str(mach), "--alpha", str(alpha),
               "--eta", str(eta), "--boundary", boundary, "--ncpu", "8", "--out", str(target)]
    print(f"Running {name}", flush=True)
    with (target/"run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    result = json.loads((target/"result.json").read_text())
    result.update(case=name, command=command, exit_code=process.returncode)
    assert result["source_sha256"] == source_hash and result["protocol_sha256"] == protocol_hash
    gates = {}
    if result["converged"]:
        final = result["history"][-1]
        gates.update(residual_rms=final["residual_divergence_rms"] <= 1e-9,
                     residual_max=final["residual_divergence_max"] <= 1e-7,
                     mass_flux=abs(final["normalized_net_mass_flux"]) < 1e-9,
                     kutta=abs(result["kutta_residual"]) <= 1e-10)
        field = np.load(target/"field.npz")
        if boundary == "exact" and mach == 0. and alpha == 5. and eta == 0.:
            if nr > 32:
                coarse = next(row for row in records if row["boundary"] == "exact" and row["mach"] == 0.
                              and row["alpha_deg"] == 5. and row["eta"] == 0. and row["nr"] == nr//2)
                ratio = result["cp_rmse_exact"]/coarse["cp_rmse_exact"]
                result["cp_refinement_ratio"] = ratio
                gates["cp_refinement"] = ratio < .8
            if nr == 128:
                gates.update(cp_rmse=result["cp_rmse_exact"] < .02,
                             circulation_relative=abs(result["circulation_ccw"]
                                                      /result["circulation_ccw_exact_incompressible"]-1.) < .005,
                             cl_relative=result["cl_error_exact"]/abs(result["cl_exact_incompressible"]) < .005,
                             cl_pressure_circulation=abs(result["cl_pressure_circulation_difference"])
                                                     /abs(result["cl_exact_incompressible"]) < .005,
                             pressure_drag=abs(result["cd_pressure"]) < .002)
        if boundary == "exact" and mach == 0. and not (alpha == 5. and eta == 0.):
            gates.update(cl_absolute=result["cl_error_exact"] < .005, cp_rmse=result["cp_rmse_exact"] < .02)
        if boundary == "exact" and mach == 0. and eta == 0. and alpha == -5.:
            plus = next(row for row in records if row["boundary"] == "exact" and row["mach"] == 0.
                        and row["alpha_deg"] == 5. and row["nr"] == 128)
            result["cl_antisymmetry_error"] = abs(result["cl_pressure"]+plus["cl_pressure"])
            gates["alpha_antisymmetry"] = result["cl_antisymmetry_error"] < 1e-8
        if boundary == "exact" and mach == 0. and eta == 0. and alpha == 0.:
            gates["symmetric_zero_lift"] = abs(result["cl_pressure"]) < 1e-8
        if mach == .001:
            baseline = next(row for row in records if row["boundary"] == "exact" and row["mach"] == 0.
                            and row["alpha_deg"] == 5. and row["eta"] == 0. and row["nr"] == nr)
            zero = np.load(out/baseline["case"]/"field.npz")
            difference = float(np.max(abs(field["cp"]-zero["cp"])))
            result["low_mach_max_cp_difference"] = difference
            gates["low_mach_limit"] = difference < 1e-4
    result["gates"] = {key: bool(value) for key, value in gates.items()}
    result["passed_specified_gates"] = bool(result["converged"] and all(gates.values()))
    records.append(result)
    (out/"summary.json").write_text(json.dumps(dict(full_potential=records, source_sha256=source_hash,
                                                   protocol_sha256=protocol_hash), indent=2, allow_nan=False),
                                    encoding="utf-8")
    print(f"{name}: {result['status']}, gates={result['gates']}, "
          f"CL={result.get('cl_pressure')}, time={result['elapsed_seconds']:.3f}s", flush=True)
print(f"Accepted {sum(row['passed_specified_gates'] for row in records)}/{len(records)} configurations.",
      flush=True)
