"""Run a full-span continuous-P1 lifting G1 prototype on generated or DPanel CAD.

Usage:
  python -m panel.run_p1_lifting_g1 --alpha 0 5 --kutta pressure --out runs/p1-g1
  python -m panel.run_p1_lifting_g1 --half-surface path/to/surface.json --alpha 0 5 --out runs/g1-cad
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .p1_lifting_g1 import make_wing,from_dpanel_half_surface,solve,alpha_sensitivity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--half-surface',type=Path,help='Positive-y DPanel surface.json; mirror to full wing')
    p.add_argument('--chord-panels',type=int,default=4)
    p.add_argument('--span-panels',type=int,default=6)
    p.add_argument('--wake-panels',type=int,default=2)
    p.add_argument('--thickness',type=float,default=.12)
    p.add_argument('--camber',type=float,default=0.)
    p.add_argument('--span',type=float,default=4.)
    p.add_argument('--chord',type=float,default=1.)
    p.add_argument('--alpha',type=float,nargs='+',default=[0.,5.])
    p.add_argument('--kutta',choices=('signed_speed','pressure'),default='pressure')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    began=time.perf_counter()
    if args.half_surface:
        wing=from_dpanel_half_surface(args.half_surface)
        input_sha=hashlib.sha256(args.half_surface.read_bytes()).hexdigest()
    else:
        wing=make_wing(chord_panels=args.chord_panels,span_panels=args.span_panels,
                       thickness=args.thickness,camber=args.camber,span=args.span,
                       chord=args.chord,wake_panels=args.wake_panels)
        input_sha=None
    # Keep artifacts independent: preexisting source and output are not modified.
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'mesh.npz',xyz=wing.xyz,faces=wing.faces,
                        dof_face=wing.dof_face,wake_xyz=wing.wake_xyz,
                        wake_faces=wing.wake_faces,gamma_y=wing.gamma_y,
                        te_pairs=wing.te_pairs,te_pair_y=wing.te_pair_y)
    entries=[]
    for alpha in args.alpha:
        r=solve(wing,float(alpha),kutta=args.kutta)
        tag=f'alpha_{alpha:+08.3f}'.replace('+','p').replace('-','m')
        np.savez_compressed(args.out/f'{tag}.npz',mu=r['mu'],gamma=r['gamma'],cp=r['cp'])
        if args.kutta=='signed_speed': derivative=alpha_sensitivity(wing,float(alpha))
        else: derivative=None
        entries.append(dict(alpha_deg=alpha,CL=r['CL'],CL_gamma=r['CL_gamma'],
                            CD_pressure=r['CD_pressure'],CM_y=r['CM_y'],
                            wall_rms=r['wall_rms'],kutta_signed_speed_rms=r['signed_kutta_rms'],
                            kutta_pressure_rms=r['pressure_kutta_rms'],
                            jump_rms=r['jump_rms'],gamma=r['gamma'].tolist(),
                            cp_min=float(r['cp'].min()),cp_max=float(r['cp'].max()),
                            derivative=derivative,field_npz=f'{tag}.npz'))
        print(f'alpha {alpha:+.2f}: CL={r["CL"]:+.6f} CLgamma={r["CL_gamma"]:+.6f} '
              f'CDp={r["CD_pressure"]:+.6f} jump={r["jump_rms"]:.2e}')
    report=dict(schema='dpanel.p1_lifting_g1.research.v1',
                physical_accuracy_accepted=False,source=wing.section,
                source_sha256=input_sha,method=args.kutta,
                n_wall_faces=len(wing.faces),n_wall_nodes=wing.ndof_mu,
                n_wake_triangles=len(wing.wake_faces),n_gamma=len(wing.gamma_y)-2,
                n_kutta_pairs=len(wing.te_pairs),reference_area=wing.reference_area,
                reference_chord=wing.chord,alpha=entries,
                method_note='P1 doublet Dirichlet + P1 wake jump + independent velocity/pressure Kutta',
                limitations=['research-only full-span reference','no viscosity','no shape VJP in this NumPy path',
                             'blunt-base CAD adapter is experimental','pressure drag not accepted'],
                elapsed_seconds=time.perf_counter()-began)
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('alpha','limitations')},indent=2))
    return report

if __name__=='__main__':main()
