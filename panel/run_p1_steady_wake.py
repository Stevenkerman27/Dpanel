"""Run the steady (NOT time-dependent) coupled P1/G2 free-wake research solver.

Example:
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_steady_wake \
   --alpha 0 5 --iterations 16 --wake-panels 6 --out runs/g2_steady_free_wake
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .p1_lifting_g1 import make_wing, from_dpanel_half_surface
from .p1_lifting_g2 import combine_wings
from .p1_steady_free_wake import solve_steady_free_wake


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--main-half-surface',type=Path,default=None,
                   help='Existing DPanel half surface.json, requires corrected TE-bank adapter')
    p.add_argument('--main-span',type=float,default=4.)
    p.add_argument('--main-chord',type=float,default=1.)
    p.add_argument('--main-camber',type=float,default=.02)
    p.add_argument('--main-chord-panels',type=int,default=4)
    p.add_argument('--main-span-panels',type=int,default=6)
    p.add_argument('--tail-span',type=float,default=1.8)
    p.add_argument('--tail-chord',type=float,default=.55)
    p.add_argument('--tail-camber',type=float,default=0.)
    p.add_argument('--tail-chord-panels',type=int,default=4)
    p.add_argument('--tail-span-panels',type=int,default=4)
    p.add_argument('--tail-x',type=float,default=1.6)
    p.add_argument('--tail-z',type=float,default=.35)
    p.add_argument('--wake-panels',type=int,default=6)
    p.add_argument('--alpha',type=float,nargs='+',default=[5.])
    p.add_argument('--kutta',choices=['pressure','signed_speed'],default='pressure')
    p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--relaxation',type=float,default=.25)
    p.add_argument('--probe-fraction',type=float,default=.055)
    p.add_argument('--step-fraction',type=float,default=.009)
    p.add_argument('--probe-floor-fraction',type=float,default=0.,
                   help='Floor the two-sided probe height by this fraction of the short edge')
    p.add_argument('--min-axial',type=float,default=.08,
                   help='Minimum axial speed for x-marching a wake node')
    p.add_argument('--slip-gate',type=float,default=0.,
                   help='Relax only nodes whose incident faces have |normal slip| below this')
    p.add_argument('--relax-min-y',type=float,default=0.,
                   help='Freeze wake nodes closer than this |y| to the symmetry plane')
    p.add_argument('--max-step-chord',type=float,default=0.,
                   help='Cap per-iteration node displacement in reference chords')
    p.add_argument('--strict-marching',action='store_true',
                   help='Raise on axial stagnation instead of freezing the node')
    p.add_argument('--geometry-tol',type=float,default=.002,
                   help='Maximum per-iteration displacement / reference chord')
    p.add_argument('--normal-slip-tol',type=float,default=.01,
                   help='Wake normal-flow RMS / Uinf')
    p.add_argument('--reference-area',type=float,default=None)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    start=time.perf_counter()
    if args.main_half_surface:
        main=from_dpanel_half_surface(args.main_half_surface)
        source_sha=hashlib.sha256(args.main_half_surface.read_bytes()).hexdigest()
    else:
        main=make_wing(chord_panels=args.main_chord_panels,
            span_panels=args.main_span_panels,span=args.main_span,
            chord=args.main_chord,camber=args.main_camber,wake_panels=args.wake_panels)
        source_sha=None
    tail=make_wing(chord_panels=args.tail_chord_panels,
        span_panels=args.tail_span_panels,span=args.tail_span,
        chord=args.tail_chord,camber=args.tail_camber,wake_panels=args.wake_panels)
    model=combine_wings([main,tail],origins=[(0,0,0),(args.tail_x,0,args.tail_z)],
        names=['main_wing','raised_tail'],reference_chord=main.chord,
        reference_area=args.reference_area if args.reference_area is not None else
                       main.reference_area+tail.reference_area,
        reference_point=main.reference_point)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'wet_wall.npz',xyz=model.xyz,
       faces=model.faces,dof_face=model.dof_face,face_bank=model.face_bank,
       face_group=model.face_group)
    reports=[]
    for alpha in args.alpha:
        tag=f'alpha_{alpha:+08.3f}'.replace('+','p').replace('-','m')
        subdir=args.out/tag;subdir.mkdir()
        def save(it,m,state,record):
            arrays=dict(mu=state['mu'],gamma=state['gamma'],cp=state['cp'])
            for gi,g in enumerate(m.groups):
                arrays.update({f'wake_xyz_{gi}':g.wake_xyz,
                               f'wake_faces_{gi}':g.wake_faces,
                               f'gamma_hat_{gi}':g.gamma_hat,
                               f'gamma_y_{gi}':g.gamma_y})
            np.savez_compressed(subdir/f'iteration_{it:03d}.npz',**arrays)
        value=solve_steady_free_wake(model,alpha,kutta=args.kutta,
            iterations=args.iterations,relaxation=args.relaxation,
            probe_fraction=args.probe_fraction,step_fraction=args.step_fraction,
            probe_floor_fraction=args.probe_floor_fraction,
            min_axial=args.min_axial,frozen_guard=not args.strict_marching,
            relax_min_abs_y=args.relax_min_y,max_step_chord=args.max_step_chord,
            slip_gate=args.slip_gate,
            geometry_tol=args.geometry_tol,slip_tol=args.normal_slip_tol,
            save_callback=save)
        log=[]
        for item in value['history']:
            row=dict(item)
            row['field_npz']=str(Path(tag)/f'iteration_{item["iteration"]:03d}.npz')
            log.append(row)
        final=value['state']
        report=dict(alpha_deg=alpha,iterations=value['iterations'],
            converged=value['converged'],geometry_stable=value['geometry_stable'],
            geometry_tol=value['geometry_tol'],normal_slip_tol=value['normal_slip_tol'],
            seconds=value['seconds'],initial=log[0],final=log[-1],history=log,
            component_names=[g.name for g in value['mesh'].groups],
            interpretation='simultaneous mu/Gamma at every geometry update; steady free wake only')
        reports.append(report)
        print(f'alpha={alpha:+.2f} deg: CL {log[0]["CL"]:+.6f} -> {log[-1]["CL"]:+.6f}; '
              f'CLgamma {log[0]["CL_gamma"]:+.6f} -> {log[-1]["CL_gamma"]:+.6f}; '
              f'geometry_stable={value["geometry_stable"]}, normal-flow-converged={value["converged"]}')
        for name in report['component_names']:
            print(' ',name,'normal slip RMS',
                  round(log[0]['wakes'][name]['normal_slip_rms'],5),'->',
                  round(log[-1]['wakes'][name]['normal_slip_rms'],5))
    payload=dict(schema='dpanel.p1.g2.steady_free_wake.v1',
        source_half_surface_sha256=source_sha,
        physical_accuracy_accepted=False,
        model='steady material-coordinate streamline-marched doublet wake',
        wake_panels=[len(g.wake_faces) for g in model.groups],
        n_wall_faces=len(model.faces),n_wall_dof=model.ndof_mu,
        reference_area=model.reference_area,reference_chord=model.reference_chord,
        parameters=dict(alpha=args.alpha,kutta=args.kutta,iterations=args.iterations,
            relaxation=args.relaxation,probe_fraction=args.probe_fraction,
            step_fraction=args.step_fraction,geometry_tol=args.geometry_tol,
            normal_slip_tol=args.normal_slip_tol,
            probe_floor_fraction=args.probe_floor_fraction,min_axial=args.min_axial,
            relax_min_abs_y=args.relax_min_y,max_step_chord=args.max_step_chord,
            slip_gate=args.slip_gate),
        reports=reports,elapsed_seconds=time.perf_counter()-start,
        limitations=['fixed mesh connectivity and finite wake length',
                     'velocity principal value via explicit two-sided finite probes',
                     'steady only: no time stepping or Kelvin-history shedding',
                     'geometry-stable is distinct from converged normal-slip',
                     'coarse wake root/tip and local Cp errors remain',
                     'no JAX shape VJP or wake roll-up core yet'])
    (args.out/'report.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload

if __name__=='__main__':main()
