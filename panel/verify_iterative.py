"""Run ONE solver mode per process on immutable saved fields; compare memory.

Example:
 python -m panel.verify_iterative --run runs/CASE/solve --out runs/NEW \
   --mode gmres --pc block --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10
Use --half only for exactly mirrored triangular meshes; otherwise it rejects.
--derivatives verifies one legal joint geometry direction; --count separately
counts SciPy GMRES iterations using the SAME analytic matvec (not saved dense A).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .differentiable import load_prepared_run, solve_fields
from .iterative import make_plan, validate_plan, PanelOperator, solve_fields_iterative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('lu','gmres'),required=True)
    p.add_argument('--pc',choices=('none','block'),required=True)
    p.add_argument('--block',type=int,required=True)
    p.add_argument('--near',type=int,required=True)
    p.add_argument('--restart',type=int,required=True)
    p.add_argument('--cycles',type=int,required=True)
    p.add_argument('--tol',type=float,required=True)
    p.add_argument('--half',action='store_true')
    p.add_argument('--derivatives',action='store_true')
    p.add_argument('--count',action='store_true')
    p.add_argument('--max-panels',type=int,default=2000)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve existing runs; choose a new output directory')
    args.out.mkdir(parents=True)
    report=dict(status='started',physical_accuracy_accepted=False,mode=args.mode,
                preconditioner=args.pc,half=args.half,jax=jax.__version__,
                python=platform.python_version(),source_run=str(args.run))
    try:
        jax.config.update('jax_enable_x64',True)
        inp,meta=load_prepared_run(args.run)
        nf=len(inp['topology'].faces)
        if nf>args.max_panels:raise ValueError('Panel budget exceeded')
        if args.mode=='lu' and args.half:raise ValueError('LU reference is full model; half is the iterative RAE path')
        plan=make_plan(inp['vertices'],inp['station_y'],inp['topology'],inp['freestream'],
            block_size=args.block,preconditioner_block_size=args.near,symmetry=args.half)
        report.update(full_wall_panels=nf,full_unknowns=nf+inp['topology'].hat_map.shape[1],
            solve_unknowns=plan.shape[0],target_block=args.block,near_block=args.near,
            tolerance=args.tol,restart=args.restart,max_restart_cycles=args.cycles)
        def evaluate(x,y):
            data={**inp,'vertices':x,'station_y':y}
            if args.mode=='lu':return solve_fields(**data)
            return solve_fields_iterative(**data,plan=plan,tolerance=args.tol,
                absolute_tolerance=1e-13,restart=args.restart,max_cycles=args.cycles,
                preconditioned=args.pc=='block')
        keys=('Cp_raw','panel_forces','panel_moments','CL','CM','mu','gamma','rhs','linear_residual')
        def selected(x,y):
            out=evaluate(x,y)
            return {k:out[k] for k in keys}
        forward=jax.jit(selected)
        x,y=inp['vertices'],inp['station_y']
        t=time.perf_counter();out=jax.tree.map(np.asarray,forward(x,y));report['cold_forward_seconds']=time.perf_counter()-t
        t=time.perf_counter();out=jax.tree.map(np.asarray,forward(x,y));report['warm_forward_seconds']=time.perf_counter()-t
        if not all(np.isfinite(a).all() for a in out.values()):
            raise FloatingPointError('GMRES/adjoint or physical field nonfinite; no fallback to LU')
        chord=float(inp['reference_chord'])
        with np.load(args.run/'fields.npz',allow_pickle=False) as f:
            refs={k:np.asarray(f[k])*scale for k,scale in (
                ('Cp_raw',1.),('panel_forces',chord**2),('panel_moments',chord**3),
                ('mu',chord),('gamma',chord))}
        refs.update(CL=np.asarray(meta['result']['CL']),CM=np.asarray(meta['result']['CM']))
        report['field_errors']={}
        for name,reference in refs.items():
            delta=out[name]-reference
            report['field_errors'][name]=dict(max_abs=float(np.max(np.abs(delta))),
                relative_L2=float(np.linalg.norm(delta)/max(np.linalg.norm(reference),1e-30)))
            np.testing.assert_allclose(out[name],reference,rtol=3e-6,atol=2e-8,err_msg=name)
        report['true_residual']=float(np.linalg.norm(out['linear_residual'])/max(np.linalg.norm(out['rhs']),1e-30))
        if args.derivatives:
            rng=np.random.default_rng(20261025)
            w=jnp.asarray(rng.normal(size=(nf,3)));v=jnp.asarray(rng.normal(size=(nf,3)))
            def objective(x,y):
                out=evaluate(x,y)
                return jnp.sum(w*out['panel_forces'])+.2*jnp.sum(v*out['panel_moments'])
            f=jax.jit(objective);back=jax.jit(jax.value_and_grad(objective,argnums=(0,1)))
            t=time.perf_counter();value,gradient=jax.tree.map(np.asarray,back(x,y));report['cold_vjp_seconds']=time.perf_counter()-t
            t=time.perf_counter();value,(gx,gy)=jax.tree.map(np.asarray,back(x,y));report['warm_vjp_seconds']=time.perf_counter()-t
            if not np.isfinite(gx).all() or not np.isfinite(gy).all():
                raise FloatingPointError('Transpose solve failed or geometry VJP nonfinite')
            origin=np.asarray(meta['surface']['origin']);xn=(np.asarray(x)-origin)/chord;yn=(np.asarray(y)-origin[1])/chord
            dx=chord*np.column_stack((.02*xn[:,0]+.005*xn[:,1]**2,
                .01*xn[:,1]**3/(1+np.max(yn**2)),.02*xn[:,2]))
            dy=chord*.01*yn**3/(1+np.max(yn**2))
            ad=float(np.sum(gx*dx)+np.sum(gy*dy));checks=[]
            for h in (1e-3,1e-4,1e-5):
                validate_plan(x+h*dx,y+h*dy,inp['topology'],inp['freestream'],plan)
                validate_plan(x-h*dx,y-h*dy,inp['topology'],inp['freestream'],plan)
                fd=float((f(x+h*dx,y+h*dy)-f(x-h*dx,y-h*dy))/(2*h))
                checks.append(dict(h=h,ad=ad,fd=fd,abs_error=abs(ad-fd)))
                np.testing.assert_allclose(fd,ad,rtol=1e-3,atol=2e-6)
            report['derivative_checks']=checks
        if args.count:
            # Separate counting experiment; not the differentiated solver timing.
            from scipy.sparse.linalg import LinearOperator,gmres
            op=PanelOperator(x,y,inp['topology'],inp['freestream'],plan)
            pc,pct=op.preconditioner() if args.pc=='block' else (None,None)
            report['scipy_iteration_diagnostics']=[]
            for transpose,mv,pre in ((False,op.matvec,pc),(True,op.rmatvec,pct)):
                fn=jax.jit(mv);b=np.asarray(op.rhs()) if not transpose else np.random.default_rng(7).normal(size=plan.shape[0])
                fn(jnp.asarray(b)).block_until_ready()
                apply=None if pre is None else jax.jit(pre)
                if apply is not None:apply(jnp.asarray(b)).block_until_ready()
                calls=[0];history=[]
                def product(v):
                    calls[0]+=1
                    return np.array(fn(jnp.asarray(v)), copy=True)
                A=LinearOperator(plan.shape,matvec=product,dtype=np.float64)
                M=None if apply is None else LinearOperator(plan.shape,matvec=lambda v:np.array(apply(jnp.asarray(v)), copy=True),dtype=np.float64)
                t=time.perf_counter();state,info=gmres(A,b,M=M,rtol=args.tol,atol=1e-13,
                    restart=args.restart,maxiter=args.cycles,callback=history.append,callback_type='pr_norm')
                duration=time.perf_counter()-t
                residual=float(np.linalg.norm(product(state)-b)/np.linalg.norm(b))
                report['scipy_iteration_diagnostics'].append(dict(transpose=transpose,
                    iterations=len(history),matvec_calls=calls[0],info=int(info),
                    true_relative_residual=residual,seconds=duration))
                if info!=0:raise ValueError('Counting solve did not converge')
        report['status']='numerical_regression_passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        raise
    try:
        import resource
        # Linux ru_maxrss is KiB. On Windows use a parent PeakWorkingSet sampler.
        report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    except ImportError:
        report['process_peak_rss_bytes']=None
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
