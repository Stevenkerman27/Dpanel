"""Sequential pre-specified cylinder study; keep rejected states and all commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"cylinder_v1"
out.mkdir(parents=True,exist_ok=True)
env = dict(os.environ,OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",MKL_NUM_THREADS="8")
specs = [(0.,nr,2*nr,20.) for nr in [32,64,128]]
specs += [(mach,nr,2*nr,20.) for mach in [.001,.1,.2,.3] for nr in [64,128]]
specs += [(.3,round(128*np.log(radius)/np.log(20.)),256,radius) for radius in [10.,40.]]
specs += [(.45,64,128,20.)]
records = []
for mach,nr,nt,radius in specs:
    name = f"M{mach:g}_r{radius:g}_{nr}x{nt}"
    target = out/name
    target.mkdir(exist_ok=True)
    command = [sys.executable,str(root/"solve_cylinder.py"),"--nr",str(nr),"--nt",str(nt),
               "--radius",str(radius),"--mach",str(mach),"--ncpu","8","--out",str(target)]
    print(f"Running {name}",flush=True)
    with (target/"run.log").open("w",encoding="utf-8") as log:
        p = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    result = json.loads((target/"result.json").read_text())
    result.update(case=name,command=command,exit_code=p.returncode)
    if result["converged"]:
        final = result["history"][-1]
        gates = dict(residual_rms=final["residual_divergence_rms"] <= 1e-9,
                     residual_max=final["residual_divergence_max"] <= 1e-7,
                     mass_flux=abs(final["normalized_net_mass_flux"]) < 1e-10,
                     force=np.linalg.norm(result["pressure_force_coefficient"]) < 1e-8)
        field = np.load(target/"field.npz")
        if mach == 0.:
            gates["unit_density"] = bool(np.max(abs(field["rho"]-1.)) == 0.)
            if nr > 32:
                previous = next(row for row in records if row["mach"] == 0. and row["nr"] == nr//2)
                ratio = result["cp_rmse_finite_annulus_exact"]/previous["cp_rmse_finite_annulus_exact"]
                result["cp_refinement_ratio"] = ratio
                gates["refinement"] = ratio < .65
            if nr == 128:
                gates["cp_rmse"] = result["cp_rmse_finite_annulus_exact"] < .01
        if mach == .001:
            baseline = np.load(out/f"M0_r20_{nr}x{nt}"/"field.npz")
            result["low_mach_max_cp_difference"] = float(np.max(abs(field["cp"]-baseline["cp"])))
            gates["low_mach_limit"] = result["low_mach_max_cp_difference"] < 1e-4
        if mach in [.1,.2,.3] and nr == 128 and radius == 20.:
            coarse = np.load(out/f"M{mach:g}_r20_64x128"/"field.npz")
            coarse_theta = coarse["theta"]
            coarse_cp = np.interp(field["theta"],coarse_theta,coarse["cp"],period=2.*np.pi)
            result["coarse_to_fine_cp_rms_difference"] = float(np.sqrt(np.mean((field["cp"]-coarse_cp)**2)))
        result["gates"] = {key:bool(value) for key,value in gates.items()}
        result["passed_pilot_gates"] = all(gates.values())
    else:
        result["passed_pilot_gates"] = False
    records.append(result)
    print(f"{name}: {result['status']}, {len(result['history'])-1} steps, "
          f"{result['elapsed_seconds']:.3g} s, Cp RMSE {result.get('cp_rmse_finite_annulus_exact')}",flush=True)
panels = []
for n in [32,64,128,256]:
    target = out/f"panel_{n}"
    target.mkdir(exist_ok=True)
    command = [sys.executable,str(root/"panel_cylinder.py"),"--panels",str(n),"--out",str(target)]
    with (target/"run.log").open("w",encoding="utf-8") as log:
        p = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    assert p.returncode == 0
    row = json.loads((target/"result.json").read_text())
    row["command"] = command
    panels.append(row)
(out/"summary.json").write_text(json.dumps(dict(full_potential=records,panel=panels),indent=2,allow_nan=False),encoding="utf-8")
assert all(row["converged"] and row["passed_pilot_gates"] for row in records if row["mach"] <= .3)
assert records[-1]["status"] == "local_sonic_state_outside_elliptic_pilot"
