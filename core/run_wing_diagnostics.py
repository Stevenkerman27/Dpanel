"""Falsification sweep for the first source/fixed-wake lifting formulation."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
out = root/"results"/"wing_diagnostics_v1"
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", NUMBA_NUM_THREADS="8")
for mesh_name, geometry in [("wing_t17","audit/cases/wing_t17_a5/model.vspgeom"),
                            ("wing_t33","audit/cases/wing_t33_a5_w20/model.vspgeom")]:
    subprocess.run([sys.executable, str(root/"import_vspgeom.py"), str(root.parent/geometry),
                    "--out", str(root/"meshes"/f"{mesh_name}.npz")], check=True, timeout=30)
cases = [("wing_t33",alpha,50,mass) for alpha in [-5,0,5] for mass in ["constrain","report"]]
cases += [("wing_t33",5,wake,"constrain") for wake in [10,200]]
cases += [("wing_t17",alpha,50,"constrain") for alpha in [0,5]]
records = []
for mesh, alpha, wake, mass in cases:
    name = f"{mesh}_a{alpha}_w{wake}_{mass}"
    target = out/name
    target.mkdir(exist_ok=True)
    command = [sys.executable, str(root/"solve.py"), str(root/"meshes"/f"{mesh}.npz"),
               "--out", str(target), "--alpha", str(alpha), "--wake-length",str(wake),
               "--mass",mass,"--off-surface","--ncpu","8"]
    print(f"Running {name}",flush=True)
    with (target/"run.log").open("w",encoding="utf-8") as log:
        p = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if p.returncode:
        records.append(dict(case=name,exit_code=p.returncode))
        continue
    result = json.loads((target/"result.json").read_text())
    result.update(case=name, alpha=alpha, command=command)
    records.append(result)
    print(f"CL pressure/wake {result['pressure_CL']:.6g}/{result['wake_CL']:.6g}, "
          f"normal RMS {result['collocation_rms']:.4g}, flux {result['normalized_source_flux']}",flush=True)
(out/"summary.json").write_text(json.dumps(records,indent=2,allow_nan=False),encoding="utf-8")
