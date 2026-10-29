"""Reproducible collocation-vs-Galerkin A/B on frozen generated G1/G2 geometries."""
from __future__ import annotations
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
import scipy

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, assemble_multi, solve_multi
from panel.p1_galerkin_solver import assemble_galerkin_multi, solve_galerkin_multi
from panel.p1_te_jump_integral import audit_g2_group_te_jump


def clean(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,np.integer,np.bool_)): return x.item()
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    return x


def build(c,span,wake,camber,g2):
    main=make_wing(chord_panels=c,span_panels=span,wake_panels=wake,camber=camber)
    wings=[main];names=["main_wing"];origins=[(0.,0.,0.)]
    if g2:
        tail=make_wing(chord_panels=max(3,c//2),span_panels=max(3,span//2),
                       span=1.8,chord=.55,camber=0.,wake_panels=wake)
        wings.append(tail);names.append("raised_tail");origins.append((1.6,0.,.35))
    return combine_wings(wings,names=names,origins=origins,
                         reference_chord=main.chord,reference_point=main.reference_point)


def run(a):
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha=None
    report=dict(schema="dpanel.issue71.galerkin_ab.v1",physical_accuracy_accepted=False,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                         scipy=scipy.__version__,git_head=sha,
                         threads={k:os.environ.get(k) for k in
                                  ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        options=dict(chord_panels=a.chord_panels,span_panels=a.span_panels,
                     wake_panels=a.wake_panels,alpha=a.alpha,camber=a.camber,
                     g2=a.g2,kutta=a.methods,order=a.order,max_nfev=a.max_nfev),
        cases=[])
    t0=time.perf_counter()
    for c in a.chord_panels:
        mesh=build(c,a.span_panels,a.wake_panels,a.camber,a.g2)
        coll=assemble_multi(mesh)
        gal=assemble_galerkin_multi(mesh,order=a.order)
        case=dict(chord_panels=c,wall_faces=len(mesh.faces),wall_dofs=mesh.ndof_mu,
                  gamma_dofs=coll["W"].shape[1],results=[])
        for method in a.methods:
            row=dict(kutta=method)
            try:
                old=solve_multi(mesh,a.alpha,kutta=method,max_nfev=a.max_nfev,operators=coll)
                row["collocation"]=dict(CL=old["CL"],CL_gamma=old["CL_gamma"],
                    CD_pressure=old["CD_pressure"],CM_y=old["CM_y"],
                    wall_rms=old["wall_rms"],jump_rms=old["jump_rms"],
                    signed_kutta_rms=old["signed_kutta_rms"],
                    pressure_kutta_rms=old["pressure_kutta_rms"],
                    nonlinear=old["nonlinear"],
                    te_jump={g.name:audit_g2_group_te_jump(mesh,old,coll,g.name) for g in mesh.groups})
            except Exception as e:
                row["collocation_error"]=repr(e)
            try:
                new=solve_galerkin_multi(mesh,a.alpha,kutta=method,max_nfev=a.max_nfev,
                                         operators=gal)
                row["galerkin"]=dict(CL=new["CL"],CL_gamma=new["CL_gamma"],
                    CD_pressure=new["CD_pressure"],CM_y=new["CM_y"],
                    wall_rms=new["wall_rms"],wall_dual_rms=new["wall_dual_rms"],
                    jump_rms=new["jump_rms"],signed_kutta_rms=new["signed_kutta_rms"],
                    pressure_kutta_rms=new["pressure_kutta_rms"],
                    nonlinear=new["nonlinear"],quadrature_order=new["order"],
                    te_jump={g.name:audit_g2_group_te_jump(mesh,new,gal,g.name) for g in mesh.groups})
                if "mu" in new:
                    row["state_delta"]={"mu_l2":float(np.linalg.norm(new["mu"]-old["mu"]))
                                        if "collocation" in row else None,
                                        "gamma_l2":float(np.linalg.norm(new["gamma"]-old["gamma"]))
                                        if "collocation" in row else None}
            except Exception as e:
                row["galerkin_error"]=repr(e)
            case["results"].append(row)
        report["cases"].append(case)
    report["seconds"]=time.perf_counter()-t0
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chord-panels",type=int,nargs="+",default=[4,6,8])
    p.add_argument("--span-panels",type=int,default=4)
    p.add_argument("--wake-panels",type=int,default=2)
    p.add_argument("--alpha",type=float,default=5.)
    p.add_argument("--camber",type=float,default=.02)
    p.add_argument("--g2",action="store_true")
    p.add_argument("--methods",nargs="+",choices=("signed_speed","pressure"),
                   default=["signed_speed","pressure"])
    p.add_argument("--order",type=int,default=3)
    p.add_argument("--max-nfev",type=int,default=80)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():
        p.error("output path already exists; choose a new path to preserve prior receipts")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    report=run(a)
    a.out.write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"out":str(a.out),"seconds":report["seconds"],
                      "cases":len(report["cases"]),"physical_accuracy_accepted":False}))
    return 0 if all(not any(k.endswith("_error") for r in c["results"] for k in r)
                    for c in report["cases"]) else 2


if __name__=="__main__":
    raise SystemExit(main())
