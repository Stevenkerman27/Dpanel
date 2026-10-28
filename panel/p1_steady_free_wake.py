"""Steady free-wake relaxation for coupled continuous-P1 lifting surfaces.

This is an *aerodynamic equilibrium* iteration, not an unsteady wake model:
(1) solve all wall P1 potentials and distinct wake Gamma vectors on current
    wake triangles, including TE jumps and a genuine Kutta selection;
(2) compute the two-sided-average total velocity along each wake, using the
    same analytical P1 double-layer potential as the wall solve;
(3) integrate dy/dx=Vy/Vx, dz/dx=Vz/Vx along fixed wake material columns;
(4) relax the wake geometry while retaining the original material Gamma hats.

No time stepping, no Kelvin-history shedding, no VLM substitution.

The numerical +normal/-normal velocity probes are a *principal-value
regularization* for a discrete doublet sheet, not a physical vortex core or
viscous diffusion model. Report sensitivity to probe offset when interpreting
roll-up and far wake. Wall P0 production code is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from .p1_closed_body import Mesh, influence_block
from .p1_lifting_g2 import (MultiLiftingMesh, LiftingGroup, _mesh, assemble_multi,
                            solve_multi)


@dataclass(frozen=True)
class WakeMaterial:
    """Frozen triangulation, original material coordinate and convection paths."""
    group: str
    used: np.ndarray
    roots: np.ndarray
    order: np.ndarray
    predecessor: np.ndarray
    reference_xyz: np.ndarray
    panel_area: np.ndarray


def _wake_mesh(g: LiftingGroup) -> Mesh:
    """Use exactly the same analytic P1 triangle potential as the G1/G2 wall."""
    x=np.asarray(g.wake_xyz,dtype=float)
    f=np.asarray(g.wake_faces,dtype=np.int64)
    tri=x[f]
    av=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);a2=np.linalg.norm(av,axis=1)
    if np.any(a2<=1e-14):
        raise ValueError(f'{g.name}: zero-area or folded wake triangle')
    normal=av/a2[:,None]
    g1=np.cross(tri[:,2]-tri[:,0],normal)/a2[:,None]
    g2=np.cross(normal,tri[:,1]-tri[:,0])/a2[:,None]
    grad=np.stack((-g1-g2,g1,g2),axis=1)
    return Mesh(x,f,np.arange(len(x)),tri,tri.mean(axis=1),normal,a2/2,grad,
                dict(diameter=float(np.linalg.norm(np.ptp(x,axis=0)))))


def prepare_wake_material(g: LiftingGroup) -> WakeMaterial:
    """Find upstream wake-node predecessors by ORIGINAL connectivity/material y.

    This supports regular and locally irregular triangular wake sheets. A node
    without an upstream connected neighbor is a TE attachment: it is fixed.
    It deliberately does not move underlying CAD/wet-wall vertices.
    """
    x=np.asarray(g.wake_xyz,float);f=np.asarray(g.wake_faces,int)
    if x.ndim!=2 or x.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
        raise ValueError(f'{g.name}: invalid wake geometry')
    if not np.isfinite(x).all() or not np.isfinite(g.gamma_hat).all():
        raise ValueError(f'{g.name}: nonfinite wake/material data')
    used=np.unique(f)
    if f.min()<0 or f.max()>=len(x):
        raise ValueError(f'{g.name}: wake face index out of bounds')
    adj={int(i):set() for i in used}
    for a,b,c in f:
        for i,j in ((a,b),(b,c),(c,a)):
            adj[int(i)].add(int(j));adj[int(j)].add(int(i))
    ref=x.copy();dx=max(float(np.ptp(x[used,0])),1.)
    tol=2e-11*dx
    pred=np.full(len(x),-1,int)
    for j in used:
        # Same frozen original material-y preferred. The original wake's
        # material label must NOT be recomputed from its deformed global y.
        upstream=[i for i in adj[int(j)] if ref[i,0]<ref[j,0]-tol]
        if upstream:
            pred[j]=min(upstream,key=lambda i:(abs(ref[j,1]-ref[i,1]),
                                              ref[j,0]-ref[i,0]))
    roots=used[pred[used]<0]
    if not len(roots):
        raise ValueError(f'{g.name}: no attached upstream wake vertices')
    # A legitimate wake must have at least one far-downstream vertex.
    if not np.any(pred[used]>=0):
        raise ValueError(f'{g.name}: no streamwise wake path')
    # Reachability: predecessor chain of every convected vertex reaches a root.
    for j in used:
        k=int(j)
        for _ in range(len(used)+1):
            if pred[k]<0:break
            k=int(pred[k])
        else:
            raise ValueError(f'{g.name}: cyclic or invalid wake predecessor path')
    return WakeMaterial(g.name,used,roots,used[np.argsort(ref[used,0])],pred,
                        ref,np.linalg.norm(np.cross(ref[f[:,1]]-ref[f[:,0]],
                                                    ref[f[:,2]]-ref[f[:,0]]),axis=1)/2)


def _shortest_edges(mesh: Mesh) -> np.ndarray:
    """Shortest triangle edge per face; the length scale that bounds a near-sheet probe."""
    p=mesh.triangles
    return np.minimum(np.minimum(np.linalg.norm(p[:,1]-p[:,0],axis=1),
                                  np.linalg.norm(p[:,2]-p[:,1],axis=1)),
                      np.linalg.norm(p[:,0]-p[:,2],axis=1))


def _aspect_ratio(mesh: Mesh) -> np.ndarray:
    p=mesh.triangles
    e=np.column_stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),
                       np.linalg.norm(p[:,2]-p[:,1],axis=1),
                       np.linalg.norm(p[:,0]-p[:,2],axis=1)))
    return e.max(axis=1)/e.min(axis=1)


def _potentials_at(mesh, result, targets, *, block=20, cached_wall=None):
    """Total uniform + all P1 wall + ALL P1 wake potential at off-sheet points."""
    points=np.asarray(targets,float)
    if points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Off-surface targets must be (N,3)')
    wall=_mesh(mesh) if cached_wall is None else cached_wall
    theta=np.deg2rad(result['alpha_deg'])
    U=np.array([np.cos(theta),0.,np.sin(theta)])
    values=points@U
    mu_faces=np.asarray(result['mu'])[mesh.dof_face]
    gamma=np.asarray(result['gamma'])
    slices=[];n0=0
    for g in mesh.groups:
        ng=g.gamma_hat.shape[1]
        gm=_wake_mesh(g)
        gf=g.gamma_hat[g.wake_faces]@gamma[n0:n0+ng]
        slices.append((gm,gf));n0+=ng
    for st in range(0,len(points),block):
        ed=min(st+block,len(points));p=points[st:ed]
        _,_,D=influence_block(p,wall)
        values[st:ed]+=np.einsum('bfa,fa->b',D,mu_faces,optimize=True)
        for wake,gf in slices:
            _,_,D=influence_block(p,wake)
            values[st:ed]+=np.einsum('bfa,fa->b',D,gf,optimize=True)
    return values


def mean_wake_velocity(mesh: MultiLiftingMesh, state, *,
                       probe_fraction=0.055, step_fraction=0.009,
                       probe_floor_fraction=0.0, block=24):
    """PV approximate velocity at EVERY wake-face centroid from both sheet sides.

    Evaluate the complete P1 potential on each side of each sheet then take
    centered spatial gradients separately, before averaging the two traces.
    This includes the freestream, wall-induced, all wake-induced and self
    (principal-value) contributions. No artificial omission of self-wake.
    """
    if not (0<step_fraction<probe_fraction/3<0.15):
        raise ValueError('Require positive gradient step << wake-side probe height')
    wall=_mesh(mesh)
    output={};summary={}
    for g in mesh.groups:
        wm=_wake_mesh(g)
        length=np.sqrt(wm.areas)
        h=probe_fraction*length
        shortest=_shortest_edges(wm)
        if probe_floor_fraction>0:
            # On a strongly anisotropic wake sheet sqrt(area) is set by the long
            # edge, so a probe scaled only by it sits inside the face's own near
            # field. Floor the height by the short edge instead.
            h=np.maximum(h,probe_floor_fraction*shortest)
        step=step_fraction*length
        e=np.eye(3)
        centers=wm.centers[:,None,None,:]+(np.array([-1.,1.])[None,:,None,None]
                               *h[:,None,None,None]*wm.normals[:,None,None,:])
        pts=centers[:,:,:,None,:]+np.array([-1.,1.])[None,None,None,:,None]*step[:,None,None,None,None]*e[None,None,:,None,:]
        # pts shape (face,side,axis,+/- sign,xyz), and pairwise gradient
        f=wm.centers.shape[0]
        v=_potentials_at(mesh,state,pts.reshape(-1,3),block=block,cached_wall=wall)
        v=v.reshape(f,2,3,2)
        gradients=(v[:,:,:,1]-v[:,:,:,0])/(2*step[:,None,None])
        vavg=gradients.mean(axis=1)
        slip=np.einsum('fi,fi->f',vavg,wm.normals)
        # Pressure continuity on a vortex sheet is a separate *dynamic*
        # condition; kinematic streamline relaxation does not guarantee it.
        cp_difference=np.sum(gradients[:,0]**2,axis=1)-np.sum(gradients[:,1]**2,axis=1)
        if not np.isfinite(vavg).all():
            raise FloatingPointError(f'{g.name}: nonfinite wake velocity')
        output[g.name]=vavg
        summary[g.name]=dict(n_panels=f,
                velocity_max=float(np.linalg.norm(vavg,axis=1).max()),
                velocity_min_x=float(vavg[:,0].min()),
                normal_slip_rms=float(np.sqrt(np.mean((slip/np.linalg.norm(
                    [np.cos(np.deg2rad(state['alpha_deg'])),0,
                     np.sin(np.deg2rad(state['alpha_deg']))]))**2))),
                normal_slip_max=float(np.max(np.abs(slip))),
                wake_cp_jump_rms=float(np.sqrt(np.mean(cp_difference**2))),
                wake_cp_jump_max=float(np.max(np.abs(cp_difference))),
                shortest_edge_median=float(np.median(shortest)),
                aspect_p90=float(np.percentile(_aspect_ratio(wm),90)),
                probe_over_shortest_max=float(np.max(h/shortest)))
    return output,summary


def _node_velocities(g, face_velocity):
    w=_wake_mesh(g);nv=len(g.wake_xyz)
    total=np.zeros((nv,3));weight=np.zeros(nv)
    for j,f in enumerate(g.wake_faces):
        total[f]+=w.areas[j]*face_velocity[j]
        weight[f]+=w.areas[j]
    used=weight>0
    total[used]/=weight[used,None]
    return total


def _one_wake_target(g, mat, vel, *, min_axial=0.08, freeze=False):
    """Streamline-march frozen x-stations, including spanwise drift and rollup.

    Returns (target, frozen_nodes). A node whose upstream axial speed falls below
    min_axial cannot be marched in x. With freeze=True it is held at its current
    position and reported; with freeze=False this is an error, because silently
    marching through a stagnation point would fabricate a streamline.
    """
    current=np.asarray(g.wake_xyz,float)
    target=current.copy()
    original=mat.reference_xyz
    frozen=[]
    for node in mat.order:
        predecessor=int(mat.predecessor[node])
        if predecessor<0:
            # True TE/attached row (based on ORIGINAL wake connectivity).
            target[node]=original[node]
            continue
        v0=vel[predecessor];v1=vel[node]
        if min(v0[0],v1[0])<min_axial:
            if not freeze:
                raise ValueError(f'{g.name}: wake streamwise marching requires Vx>0; '
                                 'reverse flow needs pseudo-time propagation')
            frozen.append(int(node))
            target[node]=current[node]
            continue
        dx=original[node,0]-original[predecessor,0]
        target[node,0]=original[node,0]
        target[node,1:]=target[predecessor,1:]+0.5*dx*(
            v0[1:]/v0[0]+v1[1:]/v1[0])
    return target,frozen


def with_wake_xyz(mesh: MultiLiftingMesh, wake_coordinates: dict[str,np.ndarray]):
    """Update wake coordinates ONLY; preserve material gamma_hat, wall, TE and faces."""
    groups=[]
    for g in mesh.groups:
        x=wake_coordinates.get(g.name,g.wake_xyz)
        if x.shape!=g.wake_xyz.shape or not np.isfinite(x).all():
            raise ValueError(f'Invalid {g.name} wake coordinates')
        groups.append(replace(g,wake_xyz=np.asarray(x,float).copy()))
    return replace(mesh,groups=tuple(groups))


def reassemble_wake_only(mesh: MultiLiftingMesh, static):
    """Cache A/J/K/grad/face geometry; replace ONLY wake influence columns W."""
    centers=static['mesh'].centers
    parts=[]
    for g in mesh.groups:
        ng=g.gamma_hat.shape[1]
        wg=np.zeros((len(centers),ng))
        gm=_wake_mesh(g)
        for st in range(0,len(centers),32):
            ed=min(st+32,len(centers))
            _,_,D=influence_block(centers[st:ed],gm)
            wg[st:ed]=np.einsum('bfa,fan->bn',D,g.gamma_hat[g.wake_faces],optimize=True)
        parts.append(wg)
    new=dict(static)
    new['W']=np.column_stack(parts)
    return new


def solve_steady_free_wake(mesh: MultiLiftingMesh, alpha_deg=5., *,
                           kutta='pressure', iterations=6, relaxation=.25,
                           probe_fraction=.055, step_fraction=.009,
                           geometry_tol=2e-3, slip_tol=1e-2, max_nfev=80,
                           wake_velocity_block=24, save_callback=None,
                           min_axial=0.08, frozen_guard=True,
                           probe_floor_fraction=0.0, relax_min_abs_y=0.0,
                           max_step_chord=0.0, slip_gate=0.0):
    """Return final state, relaxed wake mesh, and complete iteration history.

    Iteration 0 is ALWAYS the original fixed-wake solution; downstream wake
    vertices are then convected in x as a material sheet (y and z drift).
    Gamma is recomputed in the current wake geometry at EVERY iteration.
    """
    if not (isinstance(iterations,int) and 0<=iterations<=100
            and 0<relaxation<=1 and geometry_tol>0 and slip_tol>0):
        raise ValueError('Invalid relaxation/iteration count')
    t0=time.perf_counter();mat={g.name:prepare_wake_material(g) for g in mesh.groups}
    fixed=assemble_multi(mesh)
    current=mesh
    history=[]
    state=None
    for i in range(iterations+1):
        ops=reassemble_wake_only(current,fixed) if i else fixed
        state=solve_multi(current,alpha_deg,kutta=kutta,max_nfev=max_nfev,operators=ops)
        panel_velocity,slip=mean_wake_velocity(current,state,
                    probe_fraction=probe_fraction,step_fraction=step_fraction,
                    probe_floor_fraction=probe_floor_fraction,
                    block=wake_velocity_block)
        record=dict(iteration=i,CL=state['CL'],CL_gamma=state['CL_gamma'],
                CD_pressure=state['CD_pressure'],CM_y=state['CM_y'],
                wall_rms=state['wall_rms'],jump_rms=state['jump_rms'],
                pressure_kutta_rms=state['pressure_kutta_rms'],
                wakes=slip,groups=state['group_results'],
                wake_geometry_change=None)
        if save_callback:
            save_callback(i,current,state,record)
        history.append(record)
        if i==iterations:
            if i>0:
                record['wake_geometry_change']=history[-2]['wake_geometry_change']
                record['relaxed_nodes']=history[-2]['relaxed_nodes']
                record['frozen_nodes']=history[-2]['frozen_nodes']
            break
        target={};displacement=[];frozen={}
        for g in current.groups:
            m=mat[g.name]
            v=_node_velocities(g,panel_velocity[g.name])
            t,frozen_nodes=_one_wake_target(g,m,v,min_axial=min_axial,
                                            freeze=frozen_guard)
            frozen[g.name]=len(frozen_nodes)
            x_new=g.wake_xyz.copy()
            movable=np.zeros(len(x_new),bool);movable[m.used]=True
            if slip_gate>0:
                # Relax only where the sheet is already close to a stream
                # surface. Elsewhere the loop would chase a near-singular
                # velocity that the panel discretisation cannot represent.
                ns=np.zeros(len(x_new))
                cnt=np.zeros(len(x_new))
                face_slip=np.abs(np.einsum('fi,fi->f',panel_velocity[g.name],
                                            _wake_mesh(g).normals))
                np.add.at(ns,g.wake_faces,face_slip[:,None].repeat(3,1))
                np.add.at(cnt,g.wake_faces,1)
                movable&=((ns/np.maximum(cnt,1))<slip_gate)
            if relax_min_abs_y>0:
                # A wake strip whose spanwise Gamma basis changes sign cannot be
                # relaxed there: the junction acts as a concentrated vortex and
                # the kinematic stream-surface condition has no meaning across it.
                movable&=np.abs(g.wake_xyz[:,1])>=relax_min_abs_y
            step=relaxation*(t[movable]-g.wake_xyz[movable])
            if max_step_chord>0:
                limit=max_step_chord*mesh.reference_chord
                length=np.linalg.norm(step,axis=1)
                scale=np.minimum(1.,limit/np.maximum(length,1e-30))
                step=step*scale[:,None]
            x_new[movable]=g.wake_xyz[movable]+step
            # The x-stations and material Gamma basis never move/reinterpolate.
            x_new[m.used,0]=m.reference_xyz[m.used,0]
            x_new[m.roots]=m.reference_xyz[m.roots]
            d=np.linalg.norm(x_new[movable]-g.wake_xyz[movable],axis=1)
            displacement.append(float(d.max()/mesh.reference_chord) if len(d) else 0.)
            record.setdefault('relaxed_nodes',{})[g.name]=int(movable.sum())
            target[g.name]=x_new
        record['wake_geometry_change']=max(displacement)
        record['frozen_nodes']=frozen
        current=with_wake_xyz(current,target)
        if (record['wake_geometry_change']<geometry_tol and i>1
                and max(item['normal_slip_rms'] for item in slip.values()) < slip_tol):
            # A separate *final* equilibrium solve on the newly relaxed shape.
            ops=reassemble_wake_only(current,fixed)
            state=solve_multi(current,alpha_deg,kutta=kutta,max_nfev=max_nfev,operators=ops)
            _,slip=mean_wake_velocity(current,state,probe_fraction=probe_fraction,
                                      step_fraction=step_fraction,
                                      probe_floor_fraction=probe_floor_fraction,
                                      block=wake_velocity_block)
            final=dict(iteration=i+1,CL=state['CL'],CL_gamma=state['CL_gamma'],
                       CD_pressure=state['CD_pressure'],CM_y=state['CM_y'],
                       wall_rms=state['wall_rms'],jump_rms=state['jump_rms'],
                       pressure_kutta_rms=state['pressure_kutta_rms'],wakes=slip,
                       groups=state['group_results'],
                       wake_geometry_change=record['wake_geometry_change'])
            history.append(final)
            if save_callback:save_callback(i+1,current,state,final)
            break
    return dict(mesh=current,state=state,history=history,
                iterations=len(history)-1,seconds=time.perf_counter()-t0,
                converged=bool(len(history)>1 and history[-1]['wakes']
                    and history[-1]['wake_geometry_change'] is not None
                    and history[-1]['wake_geometry_change']<geometry_tol
                    and max(x['normal_slip_rms'] for x in history[-1]['wakes'].values())<slip_tol),
                geometry_stable=bool(len(history)>1
                    and history[-1]['wake_geometry_change'] is not None
                    and history[-1]['wake_geometry_change']<geometry_tol),
                geometry_tol=geometry_tol,normal_slip_tol=slip_tol,
                method='steady-material-streamline-free-wake-P1-G2',
                physical_accuracy_accepted=False)
