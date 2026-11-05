"""Frozen-input B3 old / attachment-only / material-only / combined experiment."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from panel.p1_lifting_g2 import combine_wings
import panel.p1_steady_free_wake as wake
from tools.issue92.b3_dpanel import _wing
from tools.wake_sensitivity_geometry import audit_geometry, geometry_context


def legacy_target(group, material, velocity, *, min_axial=.08, freeze=False):
    original=material.reference_xyz
    current=group.wake_xyz
    target=current.copy();frozen=[]
    for node in material.order:
        parent=int(material.predecessor[node])
        if parent<0 or node in material.attachments:
            target[node]=original[node]
            continue
        v0,v1=velocity[parent],velocity[node]
        if min(v0[0],v1[0])<min_axial:
            if not freeze:
                raise ValueError("Reverse flow in legacy map")
            frozen.append(int(node));target[node]=current[node]
            continue
        target[node,0]=original[node,0]
        target[node,1:]=target[parent,1:]+.5*(original[node,0]-original[parent,0])*(
            v0[1:]/v0[0]+v1[1:]/v1[0])
    return target,frozen


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case",choices=("rect","swept30","elliptic"),required=True)
parser.add_argument("--mode",choices=("old","attachment","material","both"),required=True)
parser.add_argument("--iterations",type=int,default=60)
parser.add_argument("--potential-backend",choices=("numpy","torch"),default="numpy")
parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
parser.add_argument("--block",type=int,default=24)
parser.add_argument("--unguarded",action="store_true")
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
args.out.mkdir(parents=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")))
root=Path(__file__).resolve().parents[1]
spec=json.loads((root/"docs/issue92/b3_cases.json").read_text())["cases"][args.case]
wing=_wing(spec)
mesh=combine_wings([wing],reference_chord=spec["cref"],reference_area=spec["sref"],
                   reference_point=wing.reference_point)
frozen=root/"runs/verification/issue102_wake_sensitivity"/args.case/"initial_state.npz"
with np.load(frozen) as data:
    for key in ("xyz","faces","wake_xyz","wake_faces","gamma_hat"):
        np.testing.assert_array_equal(getattr(wing,key),data[key])
group=mesh.groups[0]
declared=group.wake_attachment_nodes
context=geometry_context(mesh)
initial_audit=audit_geometry(mesh,context,distance=False)
prepare=wake.prepare_wake_material


def material_for_mode(group, *, streamwise=False):
    material=prepare(group,streamwise=streamwise and args.mode in ("material","both"))
    if args.mode in ("old","material"):
        material=replace(material,attachments=material.roots)
    return material


wake.prepare_wake_material=material_for_mode
if args.mode in ("old","attachment"):
    wake._one_wake_target=legacy_target
started=time.time()
report=dict(case=args.case,mode=args.mode,command=[sys.executable,*sys.argv],
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                   ("panel/p1_steady_free_wake.py","panel/p1_lifting_g2.py","tools/wake_material_ablation.py")},
    frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),numpy=np.__version__,
    torch=torch.__version__,python=sys.version,relaxation=.3,alpha_deg=4.,kutta="pressure",
    potential_backend=args.potential_backend,device=args.device,block=args.block,
    orientation_backtracking=not args.unguarded,
    iterations_requested=args.iterations,status="running",rows=[],physical_accuracy_accepted=False)
records=[]
initial_cross=context["cross"]


def save(iteration,current,state,record):
    xyz=current.groups[0].wake_xyz
    tri=xyz[group.wake_faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    signed=np.einsum("fi,fi->f",cross,initial_cross)/context["area2"]**2
    path=args.out/("iteration_%03d.npz"%iteration)
    np.savez_compressed(path,wake_xyz=xyz,gamma=state["gamma"],mu=state["mu"],cp=state["cp"])
    row=dict(iteration=iteration,CL=float(state["CL"]),CL_gamma=float(state["CL_gamma"]),
        wall_rms=float(state["wall_rms"]),jump_rms=float(state["jump_rms"]),
        pressure_kutta_rms=float(state["pressure_kutta_rms"]),nonlinear=state["nonlinear"],
        wakes=record["wakes"],min_signed_jacobian=float(signed.min()),
        nonpositive_signed_faces=int(np.sum(signed<=0)),finite_gamma=bool(np.isfinite(state["gamma"]).all()),
        declared_attachment_max=float(np.linalg.norm(xyz[declared]-group.wake_xyz[declared],axis=1).max()),
        wall_contact_max=float(np.linalg.norm(xyz[context["topology"]["attached"]]-
            group.wake_xyz[context["topology"]["attached"]],axis=1).max()),
        snapshot=path.name,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        seconds=time.time()-started)
    if iteration in (0,1,8,60) or iteration==args.iterations:
        row["geometry"]=audit_geometry(current,context,initial_audit,distance=False)
    records.append(record)
    report["rows"].append(row)
    for entry,previous in zip(report["rows"],records):
        for key in ("raw_map_residual","wake_geometry_change","attachment_target_displacement",
                    "relaxed_nodes","frozen_nodes","frozen_node_ids","clipped_nodes","orientation_backtracking"):
            if key in previous:
                entry[key]=previous[key]
    (args.out/"history.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(args.case,args.mode,iteration,"CL",row["CL"],"signed",row["min_signed_jacobian"],
          "slip",max(v["normal_slip_rms"] for v in row["wakes"].values()),flush=True)


try:
    value=wake.solve_steady_free_wake(mesh,4.,kutta="pressure",iterations=args.iterations,
        relaxation=.3,wake_velocity_block=args.block,wake_potential_backend=args.potential_backend,
        wake_device=args.device,orientation_backtracking=not args.unguarded,save_callback=save)
    for entry,previous in zip(report["rows"],records):
        for key in ("raw_map_residual","wake_geometry_change","orientation_backtracking"):
            if key in previous:
                entry[key]=previous[key]
    report.update(status="completed",solver_status=value["status"],converged=value["converged"],
                  geometry_stable=value["geometry_stable"])
except Exception:
    report.update(status="failed",traceback=traceback.format_exc())
    raise
finally:
    report["seconds"]=time.time()-started
    (args.out/"history.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
