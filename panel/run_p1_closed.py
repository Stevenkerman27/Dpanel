"""Executable P0/P1 sphere benchmark: solves unknowns, then queries true fields.

python -m panel.run_p1_closed --levels 1 2 3 --samples 96 --out NEW_DIRECTORY
No VSPAERO truth values. Sphere potential/velocity are analytic references.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from .p1_closed_body import solve, surface_values, velocity_fd


def icosphere(level=1, radius=1.):
    if not isinstance(level,int) or not 0<=level<=4 or not np.isfinite(radius) or radius<=0:
        raise ValueError('Sphere level must be 0..4 and radius positive')
    t=(1+np.sqrt(5))/2
    v=np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
                [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]],float)
    f=np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
                [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
                [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]],int)
    v/=np.linalg.norm(v,axis=1)[:,None]
    for _ in range(level):
        points=v.tolist();memo={}
        def middle(a,b):
            key=tuple(sorted((a,b)))
            if key not in memo:
                p=(v[a]+v[b])/2;p/=np.linalg.norm(p)
                memo[key]=len(points);points.append(p.tolist())
            return memo[key]
        rows=[]
        for a,b,c in f:
            ab,bc,ca=middle(a,b),middle(b,c),middle(c,a)
            rows += [[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]
        v=np.asarray(points);f=np.asarray(rows)
    return v*radius,f


def sphere_velocity(points, flow=(1.,0.,0.), radius=1.):
    """Exact total exterior velocity in body-centered Cartesian coordinates."""
    x=np.asarray(points,float);u=np.asarray(flow,float)
    r=np.linalg.norm(x,axis=1);d=x/r[:,None]
    if np.any(r<radius*(1-1e-12)):raise ValueError('Exact exterior sphere requires r>=a')
    return u+.5*(radius/r)[:,None]**3*(u-3*d*(d@u)[:,None])


def directions(count):
    if count<12:raise ValueError('At least 12 independent sphere probes')
    k=np.arange(count);z=1-2*(k+.5)/count
    theta=k*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
    return np.vstack((np.stack((r*np.cos(theta),r*np.sin(theta),z),axis=1),np.eye(3),-np.eye(3)))


def benchmark(levels=(1,2,3), *, samples=96, offsets=(.01,.02,.05,.2),
              derivative='fd4', samples_per_face=1):
    probe_dirs=directions(samples);flow=np.array([1.,0.,0.]);report=[]
    for level in levels:
        v,f=icosphere(level);mesh_hash=hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()
        entry=dict(level=level,faces=len(f),nodes=len(v),mesh_sha256=mesh_hash,methods={})
        for order in (0,1):
            tick=time.perf_counter()
            # Fixed P0 source on identical triangles. P0 baseline matches the
            # old face-center equation; P1 is *not* an interpolated P0 solution.
            state=solve(v,f,flow,order=order,samples_per_face=(samples_per_face if order else 1))
            data=dict(solve_seconds=time.perf_counter()-tick,diagnostics=state['diagnostics'],offsets={})
            if derivative=='jax':
                import jax
                jax.config.update('jax_enable_x64',True)
                from .p1_closed_jax import make_velocity_evaluator
                velocity=make_velocity_evaluator(state)
            else:
                velocity=lambda p:velocity_fd(state,p,step=1e-4)
            for delta in offsets:
                if not np.isfinite(delta) or delta<=0:raise ValueError('Positive exterior offsets required')
                points=(1+delta)*probe_dirs
                actual=np.asarray(velocity(points));exact=sphere_velocity(points,flow)
                cp=1-np.sum(actual**2,axis=1);true=1-np.sum(exact**2,axis=1)
                # At r>a the exact normal velocity is NOT zero. Subtract it.
                normal_error=np.sum((actual-exact)*probe_dirs,axis=1)
                row=dict(cp_rms_error=float(np.sqrt(np.mean((cp-true)**2))),
                         cp_max_error=float(np.max(abs(cp-true))),
                         velocity_vector_rms_error=float(np.sqrt(np.mean(np.sum((actual-exact)**2,axis=1)))),
                         radial_velocity_error_rms=float(np.sqrt(np.mean(normal_error**2))))
                if derivative=='fd4':
                    half=velocity_fd(state,points,step=5e-5)
                    row['fd_step_halving_max_velocity_delta']=float(abs(actual-half).max())
                data['offsets'][str(delta)]=row
            # Facet-center normal probes, each scaled by sqrt(local area):
            # do NOT use projected tangent velocity to assert impermeability.
            ids=np.linspace(0,len(f)-1,min(len(f),samples)).astype(int)
            h=1e-3*np.sqrt(state['mesh'].areas[ids]);nc=state['mesh'].normals[ids]
            p=state['mesh'].centers[ids]+h[:,None]*nc
            field=np.asarray(velocity(p) if derivative=='jax' else
                             velocity_fd(state,p,step=min(1e-5,float(h.min())/8)))
            data['facet_probe_normal_velocity_rms']=float(np.sqrt(np.mean(np.sum(field*nc,axis=1)**2)))
            data['facet_probe_offset_over_sqrt_area']=1e-3
            if order==1:
                surf=surface_values(state,reference_area=np.pi)
                rays=state['mesh'].centers/np.linalg.norm(state['mesh'].centers,axis=1)[:,None]
                true=1-2.25*(1-rays[:,0]**2)
                data['surface_trace_cp_rms_vs_radially_matched_sphere']=float(np.sqrt(np.mean((surf['cp']-true)**2)))
                data['surface_force_coefficient']=surf['force_coefficient'].tolist()
                data['node_mu_rms_vs_sphere']=float(np.sqrt(np.mean((state['mu']-.5*v[:,0])**2)))
            data['total_seconds']=time.perf_counter()-tick
            entry['methods'][f'P{order}']=data
        report.append(entry)
    return dict(schema='dpanel.p1_closed_sphere.v1',implementation='continuous_P1_area_weighted_face_interior_QR',
                physical_accuracy_accepted=False,probe_count=len(probe_dirs),derivative=derivative,
                independent_reference='analytic_sphere_not_VSPAERO',
                geometry_caveat='polyhedral wall vs smooth sphere geometry; common P0/P1 mesh',cases=report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--samples',type=int,default=96)
    p.add_argument('--offsets',type=float,nargs='+',default=[.01,.02,.05,.2])
    p.add_argument('--derivative',choices=['fd4','jax'],default='fd4')
    p.add_argument('--samples-per-face',type=int,choices=[1,3],default=1)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    # Fail before computation, never overwrite a previous receipt.
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        report=benchmark(a.levels,samples=a.samples,offsets=a.offsets,derivative=a.derivative,
                         samples_per_face=a.samples_per_face)
        report['environment']=dict(python=platform.python_version(),numpy=np.__version__)
        report['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__),Path(__file__).with_name('p1_closed_body.py'),
                                             Path(__file__).with_name('p1_closed_jax.py'))}
        (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        for c in report['cases']:
            print(json.dumps(dict(faces=c['faces'],p0=c['methods']['P0']['offsets'],p1=c['methods']['P1']['offsets'])))
    except Exception as exc:
        (a.out/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,message=str(exc)),indent=2),encoding='utf-8')
        raise


if __name__=='__main__':main()
