"""Run an entire coupled P1 G2 main-wing / raised-tail research configuration.

python -m panel.run_p1_lifting_g2 --alpha 0 5 --out runs/g2_generated
python -m panel.run_p1_lifting_g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 0 5 --out runs/g2_cad
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .p1_lifting_g1 import make_wing,from_dpanel_half_surface
from .p1_lifting_g2 import combine_wings,assemble_multi,solve_multi,alpha_sensitivity_multi


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--main-half-surface',type=Path,default=None,
                        help='Optional real DPanel positive-y surface.json (already has physical TE face pairing)')
    parser.add_argument('--main-span',type=float,default=4.)
    parser.add_argument('--main-chord',type=float,default=1.)
    parser.add_argument('--main-camber',type=float,default=.02)
    parser.add_argument('--main-chord-panels',type=int,default=4)
    parser.add_argument('--main-span-panels',type=int,default=6)
    parser.add_argument('--tail-span',type=float,default=1.8)
    parser.add_argument('--tail-chord',type=float,default=.55)
    parser.add_argument('--tail-camber',type=float,default=0.)
    parser.add_argument('--tail-chord-panels',type=int,default=4)
    parser.add_argument('--tail-span-panels',type=int,default=4)
    parser.add_argument('--tail-x',type=float,default=1.6)
    parser.add_argument('--tail-z',type=float,default=.35)
    parser.add_argument('--wake-panels',type=int,default=2)
    parser.add_argument('--alpha',type=float,nargs='+',default=[0.,5.])
    parser.add_argument('--kutta',choices=('signed_speed','pressure'),default='pressure')
    parser.add_argument('--reference-area',type=float,default=None,
                        help='Optional fixed aircraft Sref; default is main + tail reference areas')
    parser.add_argument('--out',type=Path,required=True)
    arg=parser.parse_args(argv)
    started=time.perf_counter()
    if arg.main_half_surface:
        main_wing=from_dpanel_half_surface(arg.main_half_surface)
        source_sha=hashlib.sha256(arg.main_half_surface.read_bytes()).hexdigest()
    else:
        main_wing=make_wing(chord_panels=arg.main_chord_panels,
            span_panels=arg.main_span_panels,span=arg.main_span,
            chord=arg.main_chord,camber=arg.main_camber,wake_panels=arg.wake_panels)
        source_sha=None
    tail=make_wing(chord_panels=arg.tail_chord_panels,
        span_panels=arg.tail_span_panels,span=arg.tail_span,
        chord=arg.tail_chord,camber=arg.tail_camber,wake_panels=arg.wake_panels)
    model=combine_wings([main_wing,tail],
        origins=[(0,0,0),(arg.tail_x,0,arg.tail_z)],
        names=['main_wing','raised_tail'],reference_chord=main_wing.chord,
        reference_area=(arg.reference_area if arg.reference_area is not None
                        else main_wing.reference_area+tail.reference_area),
        reference_point=main_wing.reference_point)
    ops=assemble_multi(model)
    arg.out.mkdir(parents=True,exist_ok=False)
    meshes=dict(xyz=model.xyz,faces=model.faces,dof_face=model.dof_face,
        face_bank=model.face_bank,face_group=model.face_group)
    for i,g in enumerate(model.groups):
        meshes.update({f'wake_xyz_{i}':g.wake_xyz,f'wake_faces_{i}':g.wake_faces,
                       f'gamma_stations_{i}':g.gamma_y,
                       f'te_pairs_{i}':g.te_pairs})
    np.savez_compressed(arg.out/'mesh.npz',**meshes)
    entries=[]
    for alpha in arg.alpha:
        result=solve_multi(model,alpha,kutta=arg.kutta,operators=ops)
        tag=f'alpha_{alpha:+08.3f}'.replace('+','p').replace('-','m')
        np.savez_compressed(arg.out/f'{tag}.npz',mu=result['mu'],gamma=result['gamma'],cp=result['cp'])
        derivative=(alpha_sensitivity_multi(model,alpha,operators=ops)
                    if arg.kutta=='signed_speed' else None)
        entry={k:v for k,v in result.items() if k not in ('mu','gamma','cp','force')}
        entry['derivative']=derivative;entry['field_npz']=f'{tag}.npz'
        entries.append(entry)
        groups=', '.join(f'{r["name"]}: CL={r["CL_pressure"]:+.4f}, '
                         f'CLg={r["CL_gamma"]:+.4f}' for r in result['group_results'])
        print(f'alpha {alpha:+.2f}° total CL={result["CL"]:+.6f} '
              f'CLg={result["CL_gamma"]:+.6f} | {groups}')
    report=dict(schema='dpanel.p1_lifting_g2.research.v1',
        physical_accuracy_accepted=False,source_sha256=source_sha,
        components=[dict(name=g.name,wall_faces=len(g.face_ids),
                         wake_triangles=len(g.wake_faces),gamma_dofs=g.gamma_hat.shape[1],
                         reference_area=g.reference_area) for g in model.groups],
        n_wall_faces=len(model.faces),n_wall_dofs=model.ndof_mu,
        reference_area=model.reference_area,reference_chord=model.reference_chord,
        kutta=arg.kutta,alpha=entries,elapsed_seconds=time.perf_counter()-started,
        interpretation='coupled P1 doublet wall, independent Gamma sheet/Kutta per lifting component',
        limitations=['research mode','no viscous drag','fixed planar wakes',
                     'G2 is not an experimentally validated aircraft','no JAX shape VJP yet'])
    (arg.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    main()
