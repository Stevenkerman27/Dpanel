"""Run one structured P1 weak-wall / regularized vortex-ribbon wake case."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from panel.p1_filament_free_wake import audit_nonadjacent_intersections, solve_filament_free_wake
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--snapshot",type=Path)
parser.add_argument("--chord-panels",type=int,default=10)
parser.add_argument("--span-panels",type=int,default=12)
parser.add_argument("--wake-panels",type=int,default=8)
parser.add_argument("--length",type=float,default=10.)
parser.add_argument("--iterations",type=int,default=16)
parser.add_argument("--relaxation",type=float,default=.25)
parser.add_argument("--core",type=float,required=True)
parser.add_argument("--subdivisions",type=int,default=2)
parser.add_argument("--order",type=int,default=3)
parser.add_argument("--intersection-guard",action="store_true")
args=parser.parse_args()
if args.out.exists() or (args.snapshot and args.snapshot.exists()):
    raise FileExistsError("report and snapshot must be new paths")
args.out.parent.mkdir(parents=True,exist_ok=True)
if args.snapshot:
    args.snapshot.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")))
report=dict(command=[sys.executable,*sys.argv],platform=platform.platform(),
            python=sys.version,numpy=np.__version__,torch=torch.__version__,
            chord_panels=args.chord_panels,span_panels=args.span_panels,
            wake_panels=args.wake_panels,length_c=args.length,alpha_deg=4.,
            order=args.order,relaxation=args.relaxation,core_radius_c=args.core,
            subdivisions=args.subdivisions,wall_formulation="galerkin",
            intersection_guard=args.intersection_guard,
            kinematics="regularized midpoint vortex ribbons; unregularized P1 wall and P1 wake in wall solve",
            warning="Experimental structured wake; core and ribbon discretization alter velocity only; optional self-intersection guard skips shared-node pairs and wall crossings; no physical accuracy acceptance.",
            status="running",history=[])
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
try:
    wing=make_wing(chord_panels=args.chord_panels,span_panels=args.span_panels,
                   wake_panels=args.wake_panels,wake_length=args.length)
    mesh=combine_wings([wing],reference_chord=1.,reference_area=4.,
                       reference_point=np.array([.25,0.,0.]))
    def save(iteration,current,state,record):
        report["history"].append(record)
        args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("iteration %d CL %.7f map %.5g direction %.5g slip %.5g" %
              (iteration,record["CL"],record["raw_map_residual"],
               record["direction_residual"],record["normal_slip_rms"]),flush=True)
    result=solve_filament_free_wake(mesh,4.,iterations=args.iterations,
                relaxation=args.relaxation,core_radius=args.core,
                subdivisions=args.subdivisions,galerkin_order=args.order,save_callback=save,
                intersection_guard=args.intersection_guard)
    report.update(status=result["status"],converged=result["converged"],
                  seconds=result["seconds"],history=result["history"],
                  physical_accuracy_accepted=False)
    group=result["mesh"].groups[0]
    report["nonadjacent_self_intersections"]=audit_nonadjacent_intersections(
        group.wake_xyz,group.wake_faces)
    if args.snapshot:
        np.savez_compressed(args.snapshot,xyz=group.wake_xyz,faces=group.wake_faces,
                            gamma_hat=group.gamma_hat,gamma=result["state"]["gamma"],
                            mu=result["state"]["mu"])
except Exception:
    report.update(status="failed",traceback=traceback.format_exc())
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print("status=%s" % report["status"],flush=True)
if report["status"]=="failed":
    sys.exit(1)
