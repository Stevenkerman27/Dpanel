"""Local acceptance driver for the vertex-input evaluator on saved real runs.

python -m panel.verify_differentiable --run runs/CASE/solution --out runs/NEW_CHECK
No CAD, source fields or production solver files are changed. Run directories
must be new. Pass --derivatives for a local-load VJP/FD test on a compatible
geometry direction, including a nonuniform movement of spanwise stations.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .differentiable import load_prepared_run, solve_fields, validate_configuration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--derivatives',action='store_true')
    p.add_argument('--max-panels',type=int,default=2000)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory; preserve earlier evidence')
    args.out.mkdir(parents=True)
    report={'status':'started','physical_accuracy_accepted':False,'source_run':str(args.run)}
    try:
        jax.config.update('jax_enable_x64',True)
        inp,meta=load_prepared_run(args.run)
        n=len(inp['topology'].faces)
        if n>args.max_panels:
            raise ValueError(f'Refusing {n} panels; explicit memory budget is {args.max_panels}')
        report.update(wall_panels=n,jax=jax.__version__,backend=jax.default_backend(),
                      source_sha256={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest()
                                     for f in ('differentiable.py','kernel.py')},
                      fixture_sha256=hashlib.sha256((args.run/'surface_input.json').read_bytes()).hexdigest())
        simulate=jax.jit(solve_fields)
        t=time.perf_counter();out=jax.tree.map(np.asarray,simulate(**inp))
        cold=time.perf_counter()-t
        t=time.perf_counter();out=jax.tree.map(np.asarray,simulate(**inp))
        warm=time.perf_counter()-t
        chord=float(inp['reference_chord']);original=meta['result']
        with np.load(args.run/'fields.npz',allow_pickle=False) as f:
            comparisons={
                'Cp_raw':(out['Cp_raw'],f['Cp_raw']),
                'panel_forces':(out['panel_forces'],f['panel_forces']*chord**2),
                'panel_moments':(out['panel_moments'],f['panel_moments']*chord**3),
                'mu':(out['mu'],f['mu']*chord),
                'gamma':(out['gamma'],f['gamma']*chord),
                'CL':(out['CL'],np.asarray(original['CL'])),
                'CM':(out['CM'],np.asarray(original['CM'])),
            }
            report['regression']={}
            for name,(value,reference) in comparisons.items():
                delta=np.asarray(value)-np.asarray(reference)
                report['regression'][name]={'max_abs':float(np.max(np.abs(delta))),
                    'relative_L2':float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30))}
                np.testing.assert_allclose(value,reference,rtol=2e-8,atol=1e-9,err_msg=name)
        rhs=out['rhs']; residual=float(np.linalg.norm(out['linear_residual'])/max(np.linalg.norm(rhs),1e-30))
        if not np.isfinite(residual) or residual>1e-10:
            raise ValueError(f'True linear residual {residual} exceeds tolerance')
        np.testing.assert_allclose(out['component_forces'].sum(axis=0),out['force'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(out['component_moments'].sum(axis=0),out['moment'],atol=1e-10,rtol=1e-10)
        report.update(compile_and_first_seconds=cold,warm_fields_seconds=warm,
                      true_relative_residual=residual,CL=float(out['CL']),
                      cp_min=float(out['Cp_raw'].min()),cp_max=float(out['Cp_raw'].max()))
        if args.derivatives:
            rng=np.random.default_rng(20261025)
            wf=jnp.asarray(rng.standard_normal((n,3)));wm=jnp.asarray(rng.standard_normal((n,3)))
            def objective(x,y):
                fields=solve_fields(**{**inp,'vertices':x,'station_y':y})
                return jnp.sum(wf*fields['panel_forces'])+jnp.sum(wm*fields['panel_moments'])
            x,y=inp['vertices'],inp['station_y']
            t=time.perf_counter()
            val,(gx,gy)=jax.jit(jax.value_and_grad(objective,argnums=(0,1)))(x,y)
            gx,gy=np.asarray(gx),np.asarray(gy);val=float(val)
            report['vjp_compile_and_first_seconds']=time.perf_counter()-t
            if not np.isfinite(gx).all() or not np.isfinite(gy).all():
                raise ValueError('Nonfinite geometry VJP')
            # Global compatible geometry map, not independent invalid TE/wake motion.
            origin=np.asarray(meta['surface']['origin'])
            xn=(np.asarray(x)-origin)/chord;yn=(np.asarray(y)-origin[1])/chord
            dx=chord*np.column_stack((.02*xn[:,0]+.005*xn[:,1]**2,
                                      .01*xn[:,1]**3/(1+float(np.max(yn**2))),
                                      .02*xn[:,2]))
            dy=chord*.01*yn**3/(1+float(np.max(yn**2)))
            ad=float(np.sum(gx*dx)+np.sum(gy*dy));derivatives=[]
            f=jax.jit(objective)
            for h in (1e-3,1e-4,1e-5):
                validate_configuration(x+h*dx,y+h*dy,inp['topology'])
                validate_configuration(x-h*dx,y-h*dy,inp['topology'])
                fd=float((f(x+h*dx,y+h*dy)-f(x-h*dx,y-h*dy))/(2*h))
                derivatives.append(dict(h=h,ad=ad,fd=fd,abs_error=abs(fd-ad)))
                np.testing.assert_allclose(fd,ad,rtol=3e-4,atol=2e-7)
            report.update(local_seed_objective=val,geometry_vjp_norm=float(np.linalg.norm(gx)),
                          station_vjp_norm=float(np.linalg.norm(gy)),directional_checks=derivatives)
        report['status']='numerical_regression_passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        raise
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
