"""Measure TE jump compatibility and pressure-Kutta residual separately.

This script is deliberately diagnostic. It does not use the TE jump integral
as a replacement for pressure Kutta: integral(mu_upper-mu_lower)=integral(Gamma)
is the integrated form of the existing jump compatibility, not a condition
that selects the physical circulation by itself.
"""
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
from panel.p1_te_jump_integral import audit_g2_group_te_jump, audit_te_jump_integral


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,np.integer)): return x.item()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [jsonable(v) for v in x]
    return x


def run_case(c, span, alpha, camber, wake_panels, methods):
    wing=make_wing(chord_panels=c,span_panels=span,camber=camber,wake_panels=wake_panels)
    mesh=combine_wings([wing],names=["main_wing"],origins=[(0.,0.,0.)],
                       reference_chord=wing.chord,reference_point=wing.reference_point)
    ops=assemble_multi(mesh)
    rows=[]
    for method in methods:
        state=solve_multi(mesh,alpha,kutta=method,operators=ops,max_nfev=200)
        jump=audit_g2_group_te_jump(mesh,state,ops,"main_wing")
        # Deliberate perturbation: holding mu fixed while changing one Gamma
        # must fail the jump compatibility test; this is not a new solve.
        perturbed=dict(state)
        perturbed["gamma"]=np.asarray(state["gamma"]).copy()
        if len(perturbed["gamma"]):
            perturbed["gamma"][len(perturbed["gamma"])//2]+=0.1
            pert=audit_g2_group_te_jump(mesh,perturbed,ops,"main_wing")
        else:
            pert=None
        rows.append(dict(
            method=method,CL=float(state["CL"]),CL_gamma=float(state["CL_gamma"]),
            gamma_sum=float(np.sum(state["gamma"])),wall_rms=float(state["wall_rms"]),
            jump_rms=float(state["jump_rms"]),signed_kutta_rms=float(state["signed_kutta_rms"]),
            pressure_kutta_rms=float(state["pressure_kutta_rms"]),
            nonlinear=state["nonlinear"],te_jump=jump,
            deliberate_gamma_perturbation=(
                None if pert is None else dict(
                    nodal_residual_max=pert["nodal_residual_max"],
                    integral_residual=pert["integral_residual"],
                    l2_rms=pert["l2_rms"],max_abs_residual=pert["max_abs_residual"]))))
    return dict(chord_panels=c,span_panels=span,wall_faces=len(mesh.faces),
                wall_dofs=mesh.ndof_mu,gamma_dofs=ops["W"].shape[1],solutions=rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chord-panels",type=int,nargs="+",default=[4,8,12,16,24])
    p.add_argument("--span-panels",type=int,default=8)
    p.add_argument("--alpha",type=float,default=5.)
    p.add_argument("--camber",type=float,default=.02)
    p.add_argument("--wake-panels",type=int,default=3)
    p.add_argument("--methods",nargs="+",choices=("pressure","signed_speed"),
                   default=["pressure","signed_speed"])
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha=None
    report=dict(schema="dpanel.issue71.te_jump_integral.v1",
        physical_accuracy_accepted=False,
        interpretation="TE jump compatibility is not an independent pressure-Kutta closure",
        environment=dict(python=sys.version,platform=platform.platform(),
                         numpy=np.__version__,scipy=scipy.__version__,git_head=sha,
                         thread_env={k:os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        options=dict(chord_panels=a.chord_panels,span_panels=a.span_panels,alpha=a.alpha,
                     camber=a.camber,wake_panels=a.wake_panels,methods=a.methods),
        cases=[])
    t=time.perf_counter()
    for c in a.chord_panels:
        report["cases"].append(run_case(c,a.span_panels,a.alpha,a.camber,a.wake_panels,a.methods))
    report["seconds"]=time.perf_counter()-t
    rendered=json.dumps(jsonable(report),indent=2,allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(rendered,encoding="utf-8")
    print(rendered)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
