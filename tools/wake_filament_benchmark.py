"""Compare wake-velocity evaluators on the same saved wing state (CPU)."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np

from panel.p1_filament_free_wake import wake_velocity
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import mean_wake_velocity, with_wake_xyz


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--chord-panels",type=int,default=30)
parser.add_argument("--span-panels",type=int,default=32)
parser.add_argument("--wake-panels",type=int,default=8)
parser.add_argument("--length",type=float,default=10.)
parser.add_argument("--core",type=float,default=.05)
parser.add_argument("--subdivisions",type=int,default=2)
parser.add_argument("--repeat",type=int,default=3)
args=parser.parse_args()
if args.out.exists() or args.repeat<1:
    raise ValueError("new output and positive repeat are required")
wing=make_wing(chord_panels=args.chord_panels,span_panels=args.span_panels,
               wake_panels=args.wake_panels,wake_length=args.length)
mesh=combine_wings([wing],reference_chord=1.,reference_area=4.,
                   reference_point=np.array([.25,0.,0.]))
with np.load(args.snapshot) as data:
    if not np.array_equal(data["faces"],mesh.groups[0].wake_faces):
        raise ValueError("snapshot topology does not match wing arguments")
    if not np.array_equal(data["gamma_hat"],mesh.groups[0].gamma_hat):
        raise ValueError("snapshot circulation basis does not match wing arguments")
    mesh=with_wake_xyz(mesh,{mesh.groups[0].name:data["xyz"]})
    state=dict(alpha_deg=4.,mu=data["mu"],gamma=data["gamma"])
group=mesh.groups[0]
points=group.wake_xyz[group.wake_faces].mean(axis=1)
times={"filament":[],"sheet":[]}
outputs={}
for iteration in range(args.repeat+1):
    t=time.perf_counter()
    values=wake_velocity(mesh,state,points,args.core,args.subdivisions)
    elapsed=time.perf_counter()-t
    t=time.perf_counter()
    other,_=mean_wake_velocity(mesh,state)
    other=other[group.name]
    sheet_time=time.perf_counter()-t
    if iteration:
        times["filament"].append(elapsed)
        times["sheet"].append(sheet_time)
    outputs={"filament":values,"sheet":other}
delta=outputs["filament"]-outputs["sheet"]
report=dict(command=[sys.executable,*sys.argv],platform=platform.platform(),
            python=sys.version,numpy=np.__version__,
            threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
            targets=len(points),core_radius_c=args.core,subdivisions=args.subdivisions,
            timings=times,median_seconds={k:float(np.median(v)) for k,v in times.items()},
            velocity_difference_rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
            velocity_difference_max=float(np.linalg.norm(delta,axis=1).max()),
            note="Different regularization models; same geometry/state/targets. Timings compare only velocity evaluation, not full solver or equal numerical accuracy.")
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report["median_seconds"]),flush=True)
print("velocity rms difference %.8g" % report["velocity_difference_rms"],flush=True)
