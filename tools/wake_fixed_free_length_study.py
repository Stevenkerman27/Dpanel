"""Compare fixed wake lengths and a free wake on one prescribed P1 grid."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_steady_free_wake import reassemble_wake_only, solve_steady_free_wake
from tools.issue92.trefftz_full import trefftz_cd_full


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--chord-panels",type=int,default=30)
parser.add_argument("--span-panels",type=int,default=32)
parser.add_argument("--wake-panels",type=int,default=2)
parser.add_argument("--lengths",default="2,5,10,20")
parser.add_argument("--free-length",type=float,default=5.)
parser.add_argument("--iterations",type=int,default=4)
parser.add_argument("--order",type=int,default=3)
parser.add_argument("--relaxation",type=float,default=.25)
parser.add_argument("--max-nfev",type=int,default=80)
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
args.out.parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")))
report=dict(command=[sys.executable,*sys.argv],platform=platform.platform(),
            python=sys.version,numpy=np.__version__,torch=torch.__version__,
            alpha_deg=4.,chord_panels=args.chord_panels,span_panels=args.span_panels,
            wake_panels=args.wake_panels,order=args.order,iterations=args.iterations,
            relaxation=args.relaxation,wall_formulation="galerkin",
            trefftz_note="Gamma-only straight-sheet diagnostic; not a far-field calculation on the relaxed wake geometry.",
            warning="Joint wall/Gamma grid screen at 5c is not independent refinement or physical acceptance; wake streamwise resolution, free-wake convergence and length convergence are separate.",
            physical_accuracy_accepted=False,records=[])
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
static=None
for length in [float(value) for value in args.lengths.split(",")]:
    start=time.perf_counter()
    row=dict(mode="fixed",length_c=length,status="running")
    report["records"].append(row)
    try:
        wing=make_wing(chord_panels=args.chord_panels,span_panels=args.span_panels,
                       wake_length=length,wake_panels=args.wake_panels)
        mesh=combine_wings([wing],reference_chord=1.,reference_area=4.,
                           reference_point=np.array([.25,0.,0.]))
        if static is None:
            static=assemble_galerkin_multi(mesh,order=args.order)
            ops=static
        else:
            ops=reassemble_wake_only(mesh,static)
        state=solve_multi(mesh,4.,kutta="pressure",operators=ops,max_nfev=args.max_nfev)
        cd,_=trefftz_cd_full(np.asarray(wing.gamma_y),np.r_[0.,state["gamma"],0.],area=4.)
        row.update(status="completed",wall_faces=len(wing.faces),wall_dof=wing.ndof_mu,
                   wake_faces=len(wing.wake_faces),CL=state["CL"],CL_gamma=state["CL_gamma"],
                   CD_pressure=state["CD_pressure"],CD_trefftz=float(cd),CM_y=state["CM_y"],
                   wall_dual_rms=state["wall_dual_rms"],jump_rms=state["jump_rms"],
                   pressure_kutta_rms=state["pressure_kutta_rms"],nonlinear=state["nonlinear"])
    except Exception:
        row.update(status="failed",traceback=traceback.format_exc())
    row["seconds"]=time.perf_counter()-start
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:row.get(k) for k in ("mode","length_c","status","CL","CL_gamma","CD_pressure","CD_trefftz","CM_y","seconds")}),flush=True)

start=time.perf_counter()
row=dict(mode="free",length_c=args.free_length,status="running")
report["records"].append(row)
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
try:
    wing=make_wing(chord_panels=args.chord_panels,span_panels=args.span_panels,
                   wake_length=args.free_length,wake_panels=args.wake_panels)
    mesh=combine_wings([wing],reference_chord=1.,reference_area=4.,
                       reference_point=np.array([.25,0.,0.]))
    def save(iteration,current,state,record):
        row["history"]=row.get("history",[])+[dict(record)]
        args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("free iteration %d CL %.8f jump %.3g" %
              (iteration,state["CL"],state["jump_rms"]),flush=True)
    result=solve_steady_free_wake(mesh,4.,kutta="pressure",iterations=args.iterations,
                                  relaxation=args.relaxation,galerkin_order=args.order,
                                  max_nfev=args.max_nfev,save_callback=save)
    state=result["state"]
    cd,_=trefftz_cd_full(np.asarray(wing.gamma_y),np.r_[0.,state["gamma"],0.],area=4.)
    row.update(status=result["status"],converged=result["converged"],
               geometry_stable=result["geometry_stable"],CL=state["CL"],
               CL_gamma=state["CL_gamma"],CD_pressure=state["CD_pressure"],
               CD_trefftz=float(cd),CM_y=state["CM_y"],wall_dual_rms=state["wall_dual_rms"],
               jump_rms=state["jump_rms"],pressure_kutta_rms=state["pressure_kutta_rms"],
               nonlinear=state["nonlinear"],history=result["history"])
except Exception:
    row.update(status="failed",traceback=traceback.format_exc())
row["seconds"]=time.perf_counter()-start
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:row.get(k) for k in ("mode","length_c","status","converged","CL","CL_gamma","CD_pressure","CD_trefftz","CM_y","seconds")}),flush=True)
if any(item["status"]=="failed" for item in report["records"]):
    sys.exit(1)
