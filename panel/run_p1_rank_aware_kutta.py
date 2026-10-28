"""Run Issue #69 rank-aware Kutta research, preserving existing G1/G2 defaults."""
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
import traceback

import numpy as np
from .p1_rank_aware_kutta import RankControls, solve_g2_rank_aware


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.floating,np.integer)):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=("g1","g2"),default="g1")
    p.add_argument("--main-half-surface",type=Path)
    p.add_argument("--alpha",type=float,default=5.)
    p.add_argument("--chord-panels",type=int,default=4)
    p.add_argument("--span-panels",type=int,default=4)
    p.add_argument("--wake-panels",type=int,default=3)
    p.add_argument("--max-nfev",type=int,default=80)
    p.add_argument("--wall-rcond",type=float,default=1e-11)
    p.add_argument("--min-singular-gap",type=float,default=100.)
    p.add_argument("--pressure-rms-tol",type=float,default=1e-6)
    p.add_argument("--compare-legacy-pressure",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True,
                                    cwd=Path(__file__).resolve().parents[1],
                                    stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        sha=None
    report=dict(schema="dpanel.issue69.rank_aware_kutta.v1",status="started",
        physical_accuracy_accepted=False,
        environment=dict(python=sys.version,platform=platform.platform(),
                         numpy=np.__version__,git_head=sha,
                         thread_env={k:os.environ.get(k) for k in
                                     ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    def flush():
        (args.out/"report.json").write_text(
            json.dumps(jsonable(report),indent=2,allow_nan=False),encoding="utf-8")
    try:
        from .p1_lifting_g1 import make_wing,from_dpanel_half_surface
        from .p1_lifting_g2 import combine_wings,assemble_multi,solve_multi
        import scipy
        report["environment"]["scipy"]=scipy.__version__
        if args.main_half_surface is None:
            main_wing=make_wing(chord_panels=args.chord_panels,
                               span_panels=args.span_panels,wake_panels=args.wake_panels,
                               span=4.,chord=1.,camber=.02)
        else:
            report["input_sha256"]=hashlib.sha256(
                args.main_half_surface.read_bytes()).hexdigest()
            main_wing=from_dpanel_half_surface(args.main_half_surface)
        wings=[main_wing]; names=["main_wing"];origins=[(0.,0.,0.)]
        if args.case=="g2":
            wings.append(make_wing(chord_panels=args.chord_panels,
                                   span_panels=args.span_panels,wake_panels=args.wake_panels,
                                   span=1.8,chord=.55,camber=0.))
            names.append("raised_tail");origins.append((1.6,0.,.35))
        mesh=combine_wings(wings,names=names,origins=origins,
                           reference_chord=main_wing.chord,
                           reference_point=main_wing.reference_point)
        ops=assemble_multi(mesh)
        report["mesh"]=dict(wall_faces=len(mesh.faces),mu_dofs=mesh.ndof_mu,
                             gamma_dofs=ops["W"].shape[1],
                             jump_rows=len(ops["J"]),kutta_rows=len(ops["K"]),
                             reference_area=mesh.reference_area)
        controls=RankControls(wall_rcond=args.wall_rcond,
                              min_singular_gap=args.min_singular_gap,
                              pressure_rms_tol=args.pressure_rms_tol,
                              max_nfev=args.max_nfev)
        value=solve_g2_rank_aware(mesh,args.alpha,operators=ops,controls=controls)
        report["status"]=value["status"]
        report["rank_experiment"]={k:v for k,v in value.items()
                                   if k not in ("candidate","wall_minimizer","candidate_state")}
        if "candidate_state" in value:
            s=value["candidate_state"]
            np.savez_compressed(args.out/"candidate.npz",
                mu=s["mu"],gamma=s["gamma"],cp=s["cp"],
                wall_xyz=mesh.xyz,wall_faces=mesh.faces,
                wall_dof_face=mesh.dof_face)
            report["candidate_loads"]={k:v for k,v in s.items()
                                        if k not in ("mu","gamma","cp")}
        if args.compare_legacy_pressure:
            old=solve_multi(mesh,args.alpha,kutta="pressure",
                            max_nfev=args.max_nfev,operators=ops)
            report["legacy_pressure"]={k:old[k] for k in
                ("CL","CL_gamma","CD_pressure","CM_y","wall_rms",
                 "jump_rms","signed_kutta_rms","pressure_kutta_rms","nonlinear")}
        report["exit_code"]=(0 if report["status"]==
            "pressure_kutta_in_weak_wall_modes_not_physical_acceptance" else 2)
    except Exception as error:
        report.update(status="execution_error",exit_code=1,error=repr(error))
        (args.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        report["seconds"]=time.perf_counter()-started
        flush()
    print(report["status"]+"; physical_accuracy_accepted=false")
    return report["exit_code"]


if __name__=="__main__":
    raise SystemExit(main())
