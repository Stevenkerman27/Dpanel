"""Benchmark old per-target-face versus batched Galerkin assembly.

The reference reproduces the pre-batching loop and is intentionally slow.
Both paths use the same analytic source kernel, Duffy quadrature, geometry,
Gamma basis and floating-point precision. Reports wall/wake/RHS equivalence
and assembly speedup; it is not a physical accuracy test.
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

from panel.p1_lifting_g1 import make_wing,_doublet_matrix,_mesh_as_kernel
from panel.p1_galerkin_wall import _duffy_rule,assemble_galerkin_wall


def reference(wing,order,block=32):
    m=_mesh_as_kernel(wing);tris=m.triangles
    bary,qw=_duffy_rule(order);n=wing.ndof_mu
    A=np.zeros((n,n));W=np.zeros((n,wing.gamma_hat.shape[1]));b=np.zeros(n)
    wake=wing.wake_xyz[wing.wake_faces]
    U=np.array([1.,0.,0.])
    for f in range(len(wing.faces)):
        points=bary@tris[f];weights=2*m.areas[f]*qw
        D=_doublet_matrix(tris,points,self_faces=np.full(len(points),f),
                          face_dofs=wing.dof_face,ndof=n,block=block)
        Dw=_doublet_matrix(wake,points,face_dofs=wing.wake_faces,
                           ndof=len(wing.wake_xyz),block=block)@wing.gamma_hat
        for k,row in enumerate(wing.dof_face[f]):
            wk=weights*bary[:,k]
            A[row]+=wk@D;W[row]+=wk@Dw;b[row]-=np.dot(wk,points@U)
    return A,W,b


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chord-panels",type=int,nargs="+",default=[4,8,12])
    p.add_argument("--span-panels",type=int,default=8)
    p.add_argument("--order",type=int,default=3)
    p.add_argument("--target-face-batch",type=int,default=64)
    p.add_argument("--kernel-block",type=int,default=2048)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.out.exists():p.error("output exists; use a new path")
    try:head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception:head=None
    report=dict(schema="dpanel.issue71.galerkin_batch_benchmark.v1",
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                         scipy=scipy.__version__,git_head=head,
                         threads={k:os.environ.get(k) for k in
                         ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        options=dict(chord_panels=a.chord_panels,span_panels=a.span_panels,order=a.order,
                     target_face_batch=a.target_face_batch,kernel_block=a.kernel_block),
        cases=[],physical_accuracy_accepted=False)
    for c in a.chord_panels:
        wing=make_wing(chord_panels=c,span_panels=a.span_panels,wake_panels=2,camber=.02)
        t=time.perf_counter();ref=reference(wing,a.order);tref=time.perf_counter()-t
        t=time.perf_counter()
        fast=assemble_galerkin_wall(wing,order=a.order,block=a.kernel_block,
            target_face_batch=a.target_face_batch)
        tfast=time.perf_counter()-t
        deltas={}
        for name,x,y in (("A",ref[0],fast["A"]),("W",ref[1],fast["W"]),("b",ref[2],fast["b"])):
            deltas[name]=dict(max_abs=float(np.max(np.abs(x-y))),
                relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-300)))
        equivalent=all(v["relative_l2"]<1e-11 for v in deltas.values())
        report["cases"].append(dict(chord_panels=c,wall_faces=len(wing.faces),
            wall_dofs=int(wing.ndof_mu),gamma_dofs=int(wing.gamma_hat.shape[1]),
            reference_seconds=tref,batched_seconds=tfast,
            speedup=float(tref/max(tfast,1e-30)),equivalent=bool(equivalent),deltas=deltas))
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({"chord_panels":c,"reference_seconds":tref,
                          "batched_seconds":tfast,"speedup":tref/max(tfast,1e-30),
                          "equivalent":equivalent}),flush=True)
        if not equivalent:return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
