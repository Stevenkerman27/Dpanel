"""Separate solution error and wall-reconstruction error without changing accepted fields."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"lifting_cusp_v1"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
cases = [root/"results"/"lifting_v1"/f"exact_M0_a5_e0_{nr}x{nt}_r20"
         for nr, nt in [(32, 128), (64, 256), (128, 512)]]
for nr, nt in [(256, 512), (128, 1024), (256, 1024), (512, 512)]:
    target = out/f"exact_M0_a5_{nr}x{nt}"
    target.mkdir(exist_ok=True)
    command = [sys.executable, str(root/"solve_airfoil.py"), "--nr", str(nr), "--nt", str(nt),
               "--mach", "0", "--alpha", "5", "--boundary", "exact", "--ncpu", "8", "--out", str(target)]
    print(f"Running cusp diagnostic {nr}x{nt}", flush=True)
    with (target/"run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    result = json.loads((target/"result.json").read_text())
    result.update(command=command, exit_code=process.returncode)
    (target/"command.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cases.append(target)
records = []
for target in cases:
    row = json.loads((target/"result.json").read_text())
    if not row["converged"]:
        records.append(dict(case=str(target), status=row["status"]))
        continue
    field = np.load(target/"field.npz")
    rc, theta, psi = [field[key] for key in ["rc", "theta", "psi"]]
    dt = 2.*np.pi/row["nt"]
    c = complex(-row["epsilon"], row["eta"])
    a = abs(1.-c)
    alpha = np.deg2rad(row["alpha_deg"])
    wall_zeta = c+a*np.exp(1j*theta)
    metric = abs(1.-1./wall_zeta**2)
    k = float(field["k"])
    exact_psi = np.real(np.exp(-1j*alpha)*a*rc[:, None]*np.exp(1j*theta)
                        +a*np.exp(1j*alpha)/rc[:, None]/np.exp(1j*theta))
    dzdt = (1.-1./wall_zeta**2)*1j*a*np.exp(1j*theta)
    normal_ds = np.column_stack([dzdt.imag, -dzdt.real])*dt
    for degree in [2, 3, 4]:
        x = rc[:degree+1]-1.
        vandermonde = np.vander(x, degree+1, increasing=True)
        value_rhs = np.zeros(degree+1)
        value_rhs[0] = 1.
        derivative_rhs = np.zeros(degree+1)
        derivative_rhs[1] = 1.
        value_weight = np.linalg.solve(vandermonde.T, value_rhs)
        derivative_weight = np.linalg.solve(vandermonde.T, derivative_rhs)
        wall_psi = value_weight@psi[:degree+1]
        wall_vn = (derivative_weight@psi[:degree+1])/(a*metric)
        exact_sample_vn = (derivative_weight@exact_psi[:degree+1])/(a*metric)
        wall_vt = ((np.roll(wall_psi, -1)-np.roll(wall_psi, 1))/(2.*dt)+k)/(a*metric)
        cp = 1.-wall_vn**2-wall_vt**2
        force = -np.sum(cp[:, None]*normal_ds, axis=0)/row["chord"]
        peak_index = int(np.argmax(abs(wall_vn)))
        records.append(dict(case=target.name, nr=row["nr"], nt=row["nt"], wall_polynomial_degree=degree,
                            original_solver_source_sha256=row["source_sha256"],
                            wall_normal_rms=float(np.sqrt(np.average(wall_vn**2, weights=field["wall_ds"]))),
                            wall_normal_max=float(np.max(abs(wall_vn))),
                            wall_normal_peak_x_chord=float((field["wall_z"].real[peak_index]
                                                           -field["wall_z"].real.min())/row["chord"]),
                            exact_sample_wall_normal_max=float(np.max(abs(exact_sample_vn))),
                            solution_contribution_wall_normal_max=float(np.max(abs(wall_vn-exact_sample_vn))),
                            cp_rmse=float(np.sqrt(np.average((cp-field["exact_cp"])**2, weights=field["wall_ds"]))),
                            cp_max_error=float(np.max(abs(cp-field["exact_cp"]))),
                            cl_pressure=float(force@np.array([-np.sin(alpha), np.cos(alpha)])),
                            cd_pressure=float(force@np.array([np.cos(alpha), np.sin(alpha)])),
                            diagnostic_only=True))
summary = dict(protocol_sha256=hashlib.sha256((root/"CUSP_PROTOCOL.md").read_bytes()).hexdigest(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records)
(out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
for row in records:
    if "wall_normal_max" in row:
        print(f"{row['nr']}x{row['nt']} degree {row['wall_polynomial_degree']}: "
              f"vn_max={row['wall_normal_max']:.5g}, "
              f"exact_sample={row['exact_sample_wall_normal_max']:.5g}, "
              f"solution_part={row['solution_contribution_wall_normal_max']:.5g}, "
              f"Cp_RMSE={row['cp_rmse']:.5g}")
