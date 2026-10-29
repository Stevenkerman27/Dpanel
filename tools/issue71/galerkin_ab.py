"""Reproducible collocation-vs-Galerkin A/B on frozen generated G1/G2 geometries."""
from __future__ import annotations
import argparse
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
    if isinstance(x,np.ndarray): return [clean(v) for v in x.tolist()]
    if isinstance(x,(np.floating,np.integer,np.bool_)): return x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
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



def fixed_point_velocity(mesh,state,points,h):
    """Independent fourth-order FD of the complete off-surface total potential."""
    from panel.p1_steady_free_wake import _potentials_at
    p=np.asarray(points,float)
    grad=np.empty_like(p)
    for k in range(3):
        d=np.zeros_like(p);d[:,k]=float(h)
        values=_potentials_at(mesh,state,np.concatenate((p-2*d,p-d,p+d,p+2*d),axis=0))
        fm2,fm1,fp1,fp2=np.split(np.asarray(values,float),4)
        grad[:,k]=(fm2-8*fm1+8*fp1-fp2)/(12*float(h))
    return grad

def solve_status(state, cap):
    nonlinear=state.get("nonlinear")
    if nonlinear is None or nonlinear.get("success",True):
        return "linear_constrained_least_squares_solved"
    if int(nonlinear.get("nfev",0))>=cap:
        return "iteration_limit_not_converged"
    return "nonlinear_failure_not_converged"


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
        case=dict(chord_panels=c,results=[],errors=[])
        try:
            mesh=build(c,a.span_panels,a.wake_panels,a.camber,a.g2)
            case.update(wall_faces=len(mesh.faces),wall_dofs=mesh.ndof_mu)
        except Exception as exc:
            case["errors"].append(dict(stage="geometry",error=repr(exc)))
            report["cases"].append(case)
            continue
        coll=gal=None
        try:
            coll=assemble_multi(mesh)
            case["gamma_dofs"]=int(coll["W"].shape[1])
        except Exception as exc:
            case["errors"].append(dict(stage="collocation_assembly",error=repr(exc)))
        try:
            gal=assemble_galerkin_multi(mesh,order=a.order)
            case["galerkin_quadrature_order"]=a.order
        except Exception as exc:
            case["errors"].append(dict(stage="galerkin_assembly",error=repr(exc)))
        for method in a.methods:
            row=dict(kutta=method)
            old_state=new_state=None
            if coll is not None:
                try:
                    old_state=solve_multi(mesh,a.alpha,kutta=method,max_nfev=a.max_nfev,operators=coll)
                    row["collocation"]=dict(status=solve_status(old_state,a.max_nfev),
                        CL=old_state["CL"],CL_gamma=old_state["CL_gamma"],
                        CD_pressure=old_state["CD_pressure"],CM_y=old_state["CM_y"],
                        wall_rms=old_state["wall_rms"],jump_rms=old_state["jump_rms"],
                        signed_kutta_rms=old_state["signed_kutta_rms"],
                        pressure_kutta_rms=old_state["pressure_kutta_rms"],
                        nonlinear=old_state["nonlinear"],
                        te_jump={g.name:audit_g2_group_te_jump(mesh,old_state,coll,g.name)
                                 for g in mesh.groups})
                except Exception as exc:
                    row["collocation_error"]=repr(exc)
                    case["errors"].append(dict(stage="collocation_solve",method=method,error=repr(exc)))
            if gal is not None:
                try:
                    new_state=solve_galerkin_multi(mesh,a.alpha,kutta=method,max_nfev=a.max_nfev,
                                                   operators=gal)
                    row["galerkin"]=dict(status=solve_status(new_state,a.max_nfev),
                        CL=new_state["CL"],CL_gamma=new_state["CL_gamma"],
                        CD_pressure=new_state["CD_pressure"],CM_y=new_state["CM_y"],
                        wall_rms=new_state["wall_rms"],wall_dual_rms=new_state["wall_dual_rms"],
                        jump_rms=new_state["jump_rms"],signed_kutta_rms=new_state["signed_kutta_rms"],
                        pressure_kutta_rms=new_state["pressure_kutta_rms"],
                        nonlinear=new_state["nonlinear"],quadrature_order=new_state["order"],
                        te_jump={g.name:audit_g2_group_te_jump(mesh,new_state,gal,g.name)
                                 for g in mesh.groups})
                except Exception as exc:
                    row["galerkin_error"]=repr(exc)
                    case["errors"].append(dict(stage="galerkin_solve",method=method,error=repr(exc)))
            if old_state is not None and new_state is not None:
                row["state_delta"]={"mu_l2":float(np.linalg.norm(new_state["mu"]-old_state["mu"])),
                                    "gamma_l2":float(np.linalg.norm(new_state["gamma"]-old_state["gamma"]))}
                probes=np.array([[.25,0.,.12],[.50,0.,.12],[.75,0.,.12]])
                row["fixed_point_velocity"]={"points":probes.tolist(),"step_sensitivity":{}}
                for h in (5e-4,2.5e-4):
                    try:
                        vc=fixed_point_velocity(mesh,old_state,probes,h)
                        vg=fixed_point_velocity(mesh,new_state,probes,h)
                        row["fixed_point_velocity"]["step_sensitivity"][str(h)]={
                            "collocation":vc.tolist(),"galerkin":vg.tolist(),
                            "delta_l2":float(np.linalg.norm(vg-vc)),
                            "delta_linf":float(np.max(np.abs(vg-vc)))}
                    except Exception as exc:
                        row["fixed_point_velocity"]["step_sensitivity"][str(h)]={"error":repr(exc)}
            case["results"].append(row)
        report["cases"].append(case)
    report["seconds"]=time.perf_counter()-t0
    report["execution_errors"]=sum(len(c.get("errors",[])) for c in report["cases"])
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
                      "cases":len(report["cases"]),"execution_errors":report["execution_errors"],
                      "physical_accuracy_accepted":False}))
    # Nonlinear nonconvergence is data and is retained in the report; exit 2 is
    # reserved for execution/assembly exceptions, not a failed physical test.
    return 0 if report["execution_errors"]==0 else 2


if __name__=="__main__":
    raise SystemExit(main())
