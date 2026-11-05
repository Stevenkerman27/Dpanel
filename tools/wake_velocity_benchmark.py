"""Frozen-state B3 potential and analytic-velocity backend timing and parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import _potentials_at, _wake_mesh
from panel.p1_coupled_material_wake import TotalVelocity
from tools.issue92.b3_dpanel import _wing

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("rect","swept30","elliptic"), default="swept30")
parser.add_argument("--targets", type=int, default=512)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root = Path(__file__).resolve().parents[1]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")))
spec = json.loads((root/"docs/issue92/b3_cases.json").read_text())["cases"][args.case]
wing = _wing(spec)
mesh = combine_wings([wing],reference_chord=spec["cref"],reference_area=spec["sref"])
snapshot = root/"runs/verification/issue102_wake_sensitivity"/args.case/"iteration_000.npz"
with np.load(snapshot) as data:
    state = dict(mu=data["mu"].copy(), gamma=data["gamma"].copy(), alpha_deg=4.)
    np.testing.assert_array_equal(mesh.groups[0].wake_xyz, data["wake_xyz"])
wm = _wake_mesh(mesh.groups[0])
ids = np.linspace(0,len(wm.faces)-1,min(len(wm.faces),max(1,args.targets//12)),dtype=int)
step = .009*np.sqrt(wm.areas[ids])
center = wm.centers[ids,None,None,:] + np.array([-1.,1.])[None,:,None,None]*(
    .055*np.sqrt(wm.areas[ids]))[:,None,None,None]*wm.normals[ids,None,None,:]
points = (center[:,:,:,None,:]+np.array([-1.,1.])[None,None,None,:,None]*(
    step[:,None,None,None,None])*np.eye(3)[None,None,:,None,:]).reshape(-1,3)
report = dict(case=args.case,head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,
    threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
    cuda=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    input_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),targets=len(points),
    repeats=args.repeats,statistic="median of warmed complete calls including source setup and transfers",
    potential=[],analytic_velocity=[],physical_accuracy_accepted=False)
configs=[("numpy/24","numpy","cpu",24),("numpy/96","numpy","cpu",96),
         ("torch/cpu/256","torch","cpu",256)]
if torch.cuda.is_available():
    configs.append(("torch/cuda/512","torch","cuda",512))
reference=None
for label,backend,device,block in configs:
    _potentials_at(mesh,state,points,block=block,backend=backend,device=device)
    seconds=[]
    for repeat in range(args.repeats):
        started=time.perf_counter()
        values=_potentials_at(mesh,state,points,block=block,backend=backend,device=device)
        seconds.append(time.perf_counter()-started)
    velocity=((values.reshape(-1,2,3,2)[:,:,:,1]-values.reshape(-1,2,3,2)[:,:,:,0])
              /(2*step[:,None,None])).mean(axis=1)
    if reference is None:
        reference=values.copy();vref=velocity.copy()
    error=float(np.max(abs(velocity-vref)))
    scale=max(1.,float(np.max(abs(vref))))
    row=dict(config=label,seconds=float(np.median(seconds)),samples=seconds,
        max_potential_error=float(np.max(abs(values-reference))),max_velocity_error=error,
        parity=error<=1e-8*scale)
    report["potential"].append(row)
    print("potential",label,row["seconds"],"velocity error",error,flush=True)
reference=None
configs=[("loop","cpu",None),("numpy","cpu",256),("torch","cpu",256)]
if torch.cuda.is_available():
    configs.append(("torch","cuda",256))
for backend,device,chunk in configs:
    field=TotalVelocity(mesh,state,analytic_backend=backend,analytic_device=device,
                        **({} if chunk is None else dict(analytic_chunk=chunk)))
    field.evaluate_many(points)
    seconds=[]
    for repeat in range(args.repeats):
        started=time.perf_counter()
        values=field.evaluate_many(points)
        seconds.append(time.perf_counter()-started)
    if reference is None:
        reference=values.copy()
    error=float(np.max(abs(values-reference)))
    scale=max(1.,float(np.max(abs(reference))))
    row=dict(config=backend+"/"+device,seconds=float(np.median(seconds)),samples=seconds,
             max_velocity_error=error,parity=error<=1e-11*scale)
    report["analytic_velocity"].append(row)
    print("analytic",row["config"],row["seconds"],"error",error,flush=True)
for key in ("potential","analytic_velocity"):
    for row in report[key]:
        row["speedup"]=report[key][0]["seconds"]/row["seconds"]
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
