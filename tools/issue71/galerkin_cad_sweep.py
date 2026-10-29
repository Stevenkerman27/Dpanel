"""Run Galerkin vs collocation on multiple committed CAD-derived half meshes.

This runner uses the repository's CAD-derived dpanel.half_planar_cut.v1
surface.json fixtures. It mirrors each half mesh through the existing
validated adapter, then solves both methods on identical full-span geometry.
Every model writes a status and partial results even when a solver fails.
"""
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
import scipy

ROOT=Path(__file__).resolve().parents[2]
DEFAULTS={
 "elliptic_s075":ROOT/"cases/geom_robust/surfaces/elliptic_s075/surface.json",
 "conv_s075":ROOT/"cases/geom_robust/surfaces/conv_s075/surface.json",
 "swept30_s075":ROOT/"cases/geom_robust/surfaces/swept30_s075/surface.json",
 "bwb_s055":ROOT/"cases/geom_robust/surfaces/bwb_s055/surface.json",
}


def clean(x):
    if isinstance(x,np.ndarray): return [clean(v) for v in x.tolist()]
    if isinstance(x,(np.floating,np.integer,np.bool_)): return x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    return x


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def status(state,cap):
    from panel.p1_solution_status import assess_solution
    nonlinear=state.get('nonlinear')
    if nonlinear is not None and not nonlinear.get('success',False):
        if int(nonlinear.get('nfev',0))>=cap:
            return 'iteration_limit_not_converged'
        return 'nonlinear_failure_not_converged'
    # Preserve an explicit invalidity verdict from other backends too.
    if state.get('solution_status')=='implausible_solution':
        return 'implausible_solution'
    return assess_solution(state)['solution_status']


def component_pressure_loads(path,state,ms,geometry,alpha_deg):
    """Split pressure force by exported CAD physical component tag."""
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    half=np.asarray(raw.get("component_ids",[]),int)
    if not len(half):
        return {}
    ids=np.r_[half,half]
    if len(ids)!=len(ms.triangles) or len(state.get("cp",[]))!=len(ms.triangles):
        return {"error":"component ID count does not match mirrored wall faces"}
    role_by_tag={}
    provenance=raw.get("provenance",{})
    for pas in provenance.get("CAD_passes",[]):
        for item in pas.get("component_classification",[]):
            tag=item.get("physical_tag")
            role=item.get("selected_role")
            if tag is not None and role:
                role_by_tag[int(tag)]=str(role)
    cp=np.asarray(state["cp"],float)
    face_force=-(cp*ms.areas)[:,None]*ms.normals
    alpha=np.deg2rad(float(alpha_deg))
    U=np.array([np.cos(alpha),0.,np.sin(alpha)])
    lift=np.array([-np.sin(alpha),0.,np.cos(alpha)])
    cd_face=face_force@U/geometry.reference_area
    cl_face=face_force@lift/geometry.reference_area
    result={}
    for tag in np.unique(ids):
        mask=ids==tag
        force=face_force[mask].sum(axis=0)
        moment=np.cross(ms.centers[mask]-np.asarray(geometry.reference_point),face_force[mask]).sum(axis=0)
        cdi=cd_face[mask]
        local_ids=np.flatnonzero(mask)
        high_cp=local_ids[np.argsort(np.abs(cp[mask]))[-10:][::-1]]
        high_drag=local_ids[np.argsort(cd_face[mask])[-10:][::-1]]
        result[str(role_by_tag.get(int(tag),f"component_{int(tag)}"))]=dict(
            physical_tag=int(tag),faces=int(mask.sum()),
            CL_pressure=float(force@lift/geometry.reference_area),
            CD_pressure=float(force@U/geometry.reference_area),
            CM_y=float(moment[1]/(geometry.reference_area*geometry.reference_chord)),
            drag_positive_sum=float(cdi[cdi>0].sum()),
            drag_negative_sum=float(cdi[cdi<0].sum()),
            drag_abs_sum=float(np.abs(cdi).sum()),
            drag_cancellation_ratio=(float(np.abs(cdi).sum()/abs(float(cdi.sum())))
                if abs(float(cdi.sum()))>1e-15 else None),
            drag_cancellation_status=('defined' if abs(float(cdi.sum()))>1e-15
                else ('no_drag_contributions' if np.abs(cdi).sum()==0. else 'complete_cancellation')),
            cp_abs_max=float(np.max(np.abs(cp[mask]))),
            cp_min=float(np.min(cp[mask])),cp_max=float(np.max(cp[mask])),
            top_abs_cp_faces=[dict(face=int(i),center=ms.centers[i].tolist(),
                cp=float(cp[i]),area=float(ms.areas[i]),CD_face=float(cd_face[i]))
                for i in high_cp],
            top_positive_drag_faces=[dict(face=int(i),center=ms.centers[i].tolist(),
                cp=float(cp[i]),area=float(ms.areas[i]),CD_face=float(cd_face[i]))
                for i in high_drag]
        )
    return result


def trace_audit_summary(mesh):
    """Keep topology acceptance evidence beside every CAD solve receipt."""
    fields=("wall_geometric_nodes","wall_potential_dofs","split_nodes",
            "te_jump_rows","contact_jump_rows","unsplit_zero_gamma_tips",
            "contact_edges","trailing_edges","contact_constraints_are_not_kutta")
    out={}
    for group in mesh.groups:
        audit=getattr(group,"trace_audit",None)
        if audit is not None:
            out[group.name]={key:clean(audit[key]) for key in fields if key in audit}
    return out


def contact_jump_diagnostics(mesh,state,ops):
    """Evaluate the wall trace jump for either Galerkin or collocation states."""
    from panel.p1_wake_contact import contact_jump_report
    return contact_jump_report(mesh,state["mu"],state["gamma"],ops["slices"])

def run_model(name,path,args):
    from panel.p1_lifting_g1 import from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings,assemble_multi,solve_multi
    from panel.p1_galerkin_solver import assemble_galerkin_multi,solve_galerkin_multi
    from panel.p1_te_jump_integral import audit_g2_group_te_jump
    start=time.perf_counter()
    row=dict(name=name,input=str(path),input_sha256=sha(path),status="started",
             physical_accuracy_accepted=False,stages={})
    try:
        t=time.perf_counter()
        wing=from_dpanel_half_surface(path)
        row["mesh_input"]=dict(half_vertices=int(len(json.loads(path.read_text())["coordinates"])),
            half_faces=int(len(json.loads(path.read_text())["faces"])),
            full_faces=int(len(wing.faces)),full_vertices=int(len(wing.xyz)),
            mu_dofs=int(wing.ndof_mu),gamma_dofs=int(wing.gamma_hat.shape[1]),
            te_pairs=int(len(wing.te_pairs)),wake_faces=int(len(wing.wake_faces)),
            trace_audit=clean(wing.trace_audit or {}))
        mesh=combine_wings([wing],names=[name],origins=[(0.,0.,0.)],
            reference_area=wing.reference_area,reference_chord=wing.chord,
            reference_point=wing.reference_point)
        row["stages"]["geometry_seconds"]=time.perf_counter()-t
        t=time.perf_counter()
        coll=assemble_multi(mesh)
        row["stages"]["collocation_assembly_seconds"]=time.perf_counter()-t
        t=time.perf_counter()
        gal=assemble_galerkin_multi(mesh,order=args.order,block=args.kernel_block,
                                    target_face_batch=args.target_face_batch)
        row["stages"]["galerkin_assembly_seconds"]=time.perf_counter()-t
        row["operators"]=dict(wall_faces=int(len(mesh.faces)),mu_dofs=int(mesh.ndof_mu),
            gamma_dofs=int(coll["W"].shape[1]),galerkin_W_dofs=int(gal["W"].shape[1]),
            galerkin_order=int(args.order),target_face_batch=int(args.target_face_batch),
            kernel_block=int(args.kernel_block),
            gamma_dimensions_match=bool(coll["W"].shape[1]==gal["W"].shape[1]))
        if not row["operators"]["gamma_dimensions_match"]:
            raise ValueError("Galerkin/collocation Gamma basis dimensions differ")
        if args.assembly_only:
            row["status"]="assembly_only_success"
            row["stages"]["total_seconds"]=time.perf_counter()-start
            return row
        row["solves"]=[]
        for method in args.methods:
            result=dict(kutta=method)
            for label,fn,ops in (
                ("collocation",solve_multi,coll),
                ("galerkin",solve_galerkin_multi,gal)):
                t=time.perf_counter()
                try:
                    state=fn(mesh,args.alpha,kutta=method,max_nfev=args.max_nfev,operators=ops)
                    result[label]=dict(status=status(state,args.max_nfev),
                        seconds=time.perf_counter()-t,CL=float(state["CL"]),
                        CL_gamma=float(state["CL_gamma"]),CD_pressure=float(state["CD_pressure"]),
                        CM_y=float(state["CM_y"]),wall_rms=float(state["wall_rms"]),
                        wall_dual_rms=(float(state["wall_dual_rms"]) if "wall_dual_rms" in state else None),
                        jump_rms=float(state["jump_rms"]),
                        signed_kutta_rms=float(state["signed_kutta_rms"]),
                        pressure_kutta_rms=(None if state["pressure_kutta_rms"] is None
                                            else float(state["pressure_kutta_rms"])),
                        nonlinear=state["nonlinear"],
                        group_results=state["group_results"] if "group_results" in state else None,
                        mu_norm=float(np.linalg.norm(state.get("mu",[]))),
                        gamma_norm=float(np.linalg.norm(state.get("gamma",[]))),
                        constrained_condition=state.get("constrained_condition"),
                        linear_singular_min=state.get("linear_singular_min"),
                        gamma_column_norm_min=state.get("gamma_column_norm_min"),
                        gamma_column_norm_median=state.get("gamma_column_norm_median"),
                        solution_status=state.get("solution_status"),
                        contact_jump=contact_jump_diagnostics(mesh,state,ops),
                        diagnostic_warnings=state.get("diagnostic_warnings",[]),
                        implausible_reasons=state.get("implausible_reasons",[]),
                        te_jump=audit_g2_group_te_jump(mesh,state,ops,name),
                        component_pressure_loads=component_pressure_loads(path,state,ops["mesh"],mesh,args.alpha))
                except Exception as exc:
                    result[label+"_error"]=repr(exc)
                    result[label+"_traceback"]=traceback.format_exc()
                result.setdefault("timing_seconds",{})[label]=time.perf_counter()-t
            row["solves"].append(result)
        row["status"]="completed_with_solver_failures" if any(
            any(k.endswith("_error") for k in sol) or any(
                (sol.get(k,{}).get("status","").endswith("not_converged")
                 or sol.get(k,{}).get("status","")=="implausible_solution")
                for k in ("collocation","galerkin")) for sol in row["solves"]) else "completed"
    except Exception as exc:
        row["status"]="execution_error";row["error"]=repr(exc)
        row["traceback"]=traceback.format_exc()
    row["stages"]["total_seconds"]=time.perf_counter()-start
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",nargs="+",choices=tuple(DEFAULTS),default=list(DEFAULTS))
    p.add_argument("--methods",nargs="+",choices=("signed_speed","pressure"),default=["signed_speed"])
    p.add_argument("--alpha",type=float,default=5.)
    p.add_argument("--order",type=int,default=2)
    p.add_argument("--kernel-block",type=int,default=2048)
    p.add_argument("--target-face-batch",type=int,default=64)
    p.add_argument("--max-nfev",type=int,default=80)
    p.add_argument("--assembly-only",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():p.error("output exists; use a new path to preserve previous receipts")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    try: head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: head=None
    report=dict(schema="dpanel.issue71.galerkin_cad_sweep.v1",physical_accuracy_accepted=False,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
            scipy=scipy.__version__,git_head=head,threads={k:os.environ.get(k) for k in
            ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        options=dict(models=a.models,methods=a.methods,alpha=a.alpha,order=a.order,
            kernel_block=a.kernel_block,target_face_batch=a.target_face_batch,
            max_nfev=a.max_nfev,assembly_only=a.assembly_only),models=[])
    for name in a.models:
        report["models"].append(run_model(name,DEFAULTS[name],a))
        a.out.write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({"model":name,"status":report["models"][-1]["status"],
                          "seconds":report["models"][-1]["stages"].get("total_seconds"),
                          "out":str(a.out)}),flush=True)
    return 0 if all(m["status"] in ("completed","assembly_only_success") for m in report["models"]) else 2


if __name__=="__main__":
    raise SystemExit(main())
