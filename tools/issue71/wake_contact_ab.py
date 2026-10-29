"""Reproduce Issue77 contact-trace fix on the same frozen CAD geometry.

No clipping/remeshing/BC switching. Baseline is the prior TE-only trace space;
contact-only keeps its Kutta sampling; fixed uses the declared true TE banks.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import scipy

from panel.p1_lifting_g1 import from_dpanel_half_surface, _from_dpanel_half_surface_te_only
from panel.p1_lifting_g2 import combine_wings, assemble_multi
from panel.p1_galerkin_solver import assemble_galerkin_multi, solve_galerkin_multi
from tools.issue71.galerkin_cad_sweep import clean, component_pressure_loads


def geometry(wing):
    return combine_wings([wing],names=['cad'],origins=[(0.,0.,0.)],
        reference_area=wing.reference_area,reference_chord=wing.chord,
        reference_point=wing.reference_point)


def evaluate(path,mesh,ops,method,alpha,cap):
    start=time.perf_counter()
    s=solve_galerkin_multi(mesh,alpha,kutta=method,max_nfev=cap,operators=ops)
    fields=('CL','CL_gamma','CD_pressure','CM_y','wall_rms','wall_dual_rms',
            'jump_rms','signed_kutta_rms','pressure_kutta_rms','constrained_condition',
            'mu_norm','gamma_norm','nonlinear','contact_jump','solution_status',
            'diagnostic_warnings')
    result={k:s[k] for k in fields}
    result.update(seconds=time.perf_counter()-start,mu_dofs=mesh.ndof_mu,
        gamma_dofs=len(s['gamma']),wall_faces=len(mesh.faces),
        jump_rows=len(ops['J']),kutta_rows=len(ops['K']),
        components=component_pressure_loads(path,s,ops['mesh'],mesh,alpha))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface',type=Path,default=Path('cases/geom_robust/surfaces/conv_s075/surface.json'))
    p.add_argument('--variants',nargs='+',choices=('te_only','contact_only','fixed'),default=['te_only','contact_only','fixed'])
    p.add_argument('--methods',nargs='+',choices=('signed_speed','pressure'),default=['signed_speed'])
    p.add_argument('--alpha',type=float,default=5.)
    p.add_argument('--order',type=int,default=2)
    p.add_argument('--max-nfev',type=int,default=80)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    if args.out.exists(): p.error('Preserve prior receipts; output already exists')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    try: head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): head=None
    receipt=dict(schema='dpanel.issue77.wake_contact_ab.v1',git_head=head,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__),
        input_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        physical_accuracy_accepted=False,results=[],errors=[])
    def save(): args.out.write_text(json.dumps(clean(receipt),indent=2,allow_nan=False),encoding='utf-8')
    old=_from_dpanel_half_surface_te_only(args.surface)
    fixed=from_dpanel_half_surface(args.surface)
    for key in ('xyz','faces','wake_xyz','wake_faces','gamma_y','gamma_hat'):
        np.testing.assert_array_equal(getattr(old,key),getattr(fixed,key))
    receipt['geometry_and_gamma_basis_unchanged']=True
    receipt['contact_topology']=fixed.trace_audit
    fixed_mesh=fixed_ops=None
    for variant in args.variants:
        try:
            t=time.perf_counter()
            if variant=='te_only':
                mesh=geometry(old);ops=assemble_galerkin_multi(mesh,order=args.order)
            else:
                if fixed_ops is None:
                    fixed_mesh=geometry(fixed)
                    fixed_ops=assemble_galerkin_multi(fixed_mesh,order=args.order)
                mesh=fixed_mesh;ops=fixed_ops
                if variant=='contact_only':
                    # Same split trace space and weak matrices. Only K and its
                    # face pairs use the previous geometric-neighbour selector.
                    mesh=geometry(replace(fixed,te_face_pairs=None))
                    legacy=assemble_multi(mesh)
                    np.testing.assert_allclose(legacy['J'],fixed_ops['J'],atol=0,rtol=0)
                    ops=dict(fixed_ops,K=legacy['K'],
                             te_face_pairs=[p for g in legacy['te_face_pairs'] for p in g])
            assembly_seconds=time.perf_counter()-t
            for method in args.methods:
                result=evaluate(args.surface,mesh,ops,method,args.alpha,args.max_nfev)
                receipt['results'].append(dict(variant=variant,method=method,
                                               assembly_or_reuse_seconds=assembly_seconds,**result))
                save()
                print(json.dumps({k:result[k] for k in ('CL','CD_pressure','solution_status')},allow_nan=False),flush=True)
        except Exception as error:
            receipt['errors'].append(dict(variant=variant,error=repr(error),traceback=traceback.format_exc()))
            save()
    save()
    return 2 if receipt['errors'] or any(r['solution_status']!='solver_success' for r in receipt['results']) else 0


if __name__=='__main__':
    raise SystemExit(main())
