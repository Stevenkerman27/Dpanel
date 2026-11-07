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
roll-up and far wake. Galerkin is the default; collocation is an explicit
comparison option.
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
    attachments: object = None
    streamwise_integral: object = None


@dataclass(frozen=True)
class MaterialLineIntegral:
    """Sparse nodal integration weights without a dense node-by-node array."""
    rows: np.ndarray
    columns: np.ndarray
    weights: np.ndarray
    nodes: int

    def __matmul__(self, values):
        values=np.asarray(values,float)
        out=np.zeros((self.nodes,)+values.shape[1:])
        weight=self.weights.reshape((-1,)+(1,)*(values.ndim-1))
        np.add.at(out,self.rows,weight*values[self.columns])
        return out


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


def prepare_wake_material(g: LiftingGroup, *, streamwise=False) -> WakeMaterial:
    """Find upstream wake-node predecessors by ORIGINAL connectivity/material y.

    Graph roots are integration diagnostics, distinct from declared attachments.
    B3 optionally caches line integrals on the frozen (x, material-y) mesh.
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
    declared=getattr(g,"wake_attachment_nodes",None)
    if declared is None:
        # Compatibility for hand-built legacy groups; combine_wings supplies
        # explicit declared TE-node correspondence on production paths.
        attachments=roots.copy()
    else:
        attachments=np.asarray(declared,int).reshape(-1)
        if len(np.unique(attachments)) != len(attachments) or not np.isin(attachments,used).all():
            raise ValueError(f"{g.name}: invalid declared wake attachment node map")
    integral=_material_line_integrals(g) if streamwise else None
    return WakeMaterial(g.name,used,roots,used[np.argsort(ref[used,0])],pred,
                        ref,np.linalg.norm(np.cross(ref[f[:,1]]-ref[f[:,0]],
                                                    ref[f[:,2]]-ref[f[:,0]]),axis=1)/2,
                        attachments,integral)


def _material_line_integrals(g):
    """Exact P1 integrals along constant reference-y rays, not mesh edges."""
    xyz=np.asarray(g.wake_xyz,float)
    faces=np.asarray(g.wake_faces,int)
    tri=xyz[faces]
    area_xy=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,2]
    scale=max(float(np.ptp(xyz[np.unique(faces),0])),1.)
    tol=2e-11*scale
    if np.any(np.abs(area_xy) <= 1e-14*scale**2):
        raise ValueError(f'{g.name}: B3 material coordinates require nondegenerate (x,y) triangles')
    # Use local differences so horizontal edges have exactly zero x coefficient.
    slope=(tri[:,[1,2,0],1]-tri[:,[2,0,1],1])/area_xy[:,None]
    y_gradient=(tri[:,[2,0,1],0]-tri[:,[1,2,0],0])/area_xy[:,None]
    rows=[];columns=[];values=[]
    for node in np.unique(faces):
        intercept=y_gradient*(xyz[node,1]-tri[:,0,1])[:,None]+np.array([1.,0.,0.])
        nonzero=np.abs(slope)>1e-13/scale
        cross=tri[:,0,0,None]+np.divide(-intercept,slope,out=np.zeros_like(slope),where=nonzero)
        lower=np.max(np.where(nonzero & (slope>0),cross,-np.inf),axis=1)
        upper=np.min(np.where(nonzero & (slope<0),cross,np.inf),axis=1)
        upper=np.minimum(upper,xyz[node,0])
        valid=(upper-lower>tol)&~np.any(~nonzero & (intercept < -1e-10),axis=1)
        ids=np.flatnonzero(valid)
        if not len(ids):
            continue
        knots=np.sort(np.concatenate((lower[ids],upper[ids])))
        knots=knots[np.r_[True,np.diff(knots)>tol]]
        mid=(knots[1:]+knots[:-1])/2
        length=np.diff(knots)
        cover=(mid[:,None]>=lower[ids][None]-tol)&(mid[:,None]<=upper[ids][None]+tol)
        if abs(knots[-1]-xyz[node,0])>tol:
            # The node itself is a new upstream entry; an earlier disjoint
            # interval must not supply its displacement.
            continue
        # A nonconvex TE can intersect the ray more than once. Start at the
        # upstream boundary of the connected interval containing this node.
        gaps=np.flatnonzero(~cover.any(axis=1))
        first=int(gaps[-1]+1) if len(gaps) else 0
        cover=cover[first:];mid=mid[first:];length=length[first:]
        # A ray on a shared triangle edge is integrated once, never twice.
        chosen=ids[np.argmax(cover,axis=1)]
        weights=(slope[chosen]*(mid-tri[chosen,0,0])[:,None]+intercept[chosen])*length[:,None]
        rows.extend(np.full(weights.size,int(node)))
        columns.extend(faces[chosen].ravel())
        values.extend(weights.ravel())
    return MaterialLineIntegral(np.asarray(rows,int),np.asarray(columns,int),
                                np.asarray(values,float),len(xyz))


def wake_material_graph_audit(material: WakeMaterial) -> dict:
    """Audit predecessor links and rooted paths without changing the graph.

    Reports direct topological edge lengths separately from cumulative path
    arclength. A long direct link is not silently reclassified as a boundary:
    the audit preserves its node IDs, coordinates, path tail and x-monotonicity
    evidence so the mesh/graph construction can be corrected at its source.
    """
    xyz = np.asarray(material.reference_xyz, float)
    used = np.asarray(material.used, int)
    pred = np.asarray(material.predecessor, int)
    valid = used[pred[used] >= 0]
    direct = np.linalg.norm(xyz[valid] - xyz[pred[valid]], axis=1)
    positive = direct[direct > 0.]
    median = float(np.median(positive)) if len(positive) else 0.
    order = np.argsort(direct)[::-1] if len(direct) else np.empty(0, int)
    longest = []
    for index in order[:20]:
        node = int(valid[index])
        parent = int(pred[node])
        longest.append(dict(node=node, predecessor=parent,
                            length=float(direct[index]),
                            dx=float(xyz[node, 0]-xyz[parent, 0]),
                            dy=float(xyz[node, 1]-xyz[parent, 1]),
                            dz=float(xyz[node, 2]-xyz[parent, 2]),
                            node_xyz=xyz[node].tolist(),
                            predecessor_xyz=xyz[parent].tolist(),
                            ratio_to_median=(float(direct[index]/median) if median else None)))
    path_records = []
    cycle_nodes = []
    nonmonotone_edges = {}
    max_path_length = -1.
    max_path = []
    for start in used:
        node = int(start)
        seen = {}
        chain = []
        length = 0.
        while node >= 0:
            if node in seen:
                cycle_nodes.append(dict(start=int(start), cycle=chain[seen[node]:] + [node]))
                break
            seen[node] = len(chain)
            chain.append(node)
            parent = int(pred[node])
            if parent < 0:
                break
            dx = float(xyz[node, 0]-xyz[parent, 0])
            if dx <= 0.:
                nonmonotone_edges[(node, parent)] = dict(node=node, predecessor=parent, dx=dx)
            length += float(np.linalg.norm(xyz[node]-xyz[parent]))
            node = parent
        else:
            pass
        if node >= 0 and int(pred[node]) >= 0 and node in seen:
            continue
        if length > max_path_length:
            max_path_length, max_path = length, chain
    bins = np.array([0., .5, 1., 2., 4., 8., 16., 32., 64., np.inf])
    hist, _ = np.histogram(direct, bins=bins)
    return dict(
        used_nodes=int(len(used)), predecessor_edges=int(len(valid)),
        root_nodes=int(np.count_nonzero(pred[used] < 0)),
        direct_link_median=float(median),
        direct_link_max=float(direct.max()) if len(direct) else 0.,
        direct_link_p95=float(np.percentile(direct, 95)) if len(direct) else 0.,
        direct_link_histogram=[dict(lower=float(a), upper=(None if not np.isfinite(b) else float(b)),
                                    count=int(n)) for a, b, n in zip(bins[:-1], bins[1:], hist)],
        longest_direct_links=longest,
        max_root_to_tail_path_length=float(max_path_length),
        max_root_to_tail_path_nodes=[int(v) for v in max_path],
        max_path_root_xyz=(xyz[max_path[-1]].tolist() if max_path else None),
        max_path_tail_xyz=(xyz[max_path[0]].tolist() if max_path else None),
        predecessor_cycle_count=int(len(cycle_nodes)),
        predecessor_cycles=cycle_nodes[:20],
        nonmonotone_x_edge_count=int(len(nonmonotone_edges)),
        nonmonotone_x_edges=list(nonmonotone_edges.values())[:20],
        chain_x_span=(float(xyz[max_path[0], 0]-xyz[max_path[-1], 0])
                      if max_path else 0.),
        path_to_x_span_ratio=(float(max_path_length /
            max(abs(xyz[max_path[0], 0]-xyz[max_path[-1], 0]), 1e-30))
            if max_path else None),
    )


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


def _potentials_at(mesh, result, targets, *, block=20, cached_wall=None,
                   backend='numpy', device='cpu'):
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
    if backend == 'torch':
        import torch
        from .p1_batched_influence import source_invariants, block_rows
        triangles=np.concatenate([wall.triangles]+[gm.triangles for gm,_ in slices])
        strength=np.concatenate([mu_faces]+[gf for _,gf in slices])
        source=source_invariants(triangles,device=device)
        density=torch.as_tensor(strength,dtype=torch.float64,device=device)
        for st in range(0,len(points),block):
            p=torch.as_tensor(points[st:st+block],dtype=torch.float64,device=device)
            total=torch.zeros(len(p),dtype=torch.float64,device=device)
            for s in range(0,len(triangles),256):
                index=torch.arange(s,min(s+256,len(triangles)),device=device)
                rows=block_rows(p,source,index,torch=torch)
                total+=torch.einsum('bsa,sa->b',rows,density[index])
            values[st:st+len(p)]+=total.detach().cpu().numpy()
        return values
    if backend != 'numpy' or device != 'cpu':
        raise ValueError('Potential backend must be numpy/cpu or torch on an explicit device')
    for st in range(0,len(points),block):
        ed=min(st+block,len(points));p=points[st:ed]
        _,_,D=influence_block(p,wall)
        values[st:ed]+=np.einsum('bfa,fa->b',D,mu_faces,optimize=True)
        for wake,gf in slices:
            _,_,D=influence_block(p,wake)
            values[st:ed]+=np.einsum('bfa,fa->b',D,gf,optimize=True)
    return values


def _probe_segment_crosses_triangle(start, end, tri):
    edge1=tri[1]-tri[0]; edge2=tri[2]-tri[0]; direction=end-start
    cross=np.cross(direction,edge2)
    determinant=float(np.dot(edge1,cross))
    scale=np.linalg.norm(edge1)*np.linalg.norm(edge2)*np.linalg.norm(direction)
    if abs(determinant)<=32*np.finfo(float).eps*scale:
        return False
    offset=start-tri[0]
    u=float(np.dot(offset,cross)/determinant)
    other=np.cross(offset,edge1)
    v=float(np.dot(direction,other)/determinant)
    t=float(np.dot(edge2,other)/determinant)
    return bool(u>=-1e-10 and v>=-1e-10 and u+v<=1+1e-10
                and -1e-10<=t<=1+1e-10)


def _safe_probe_scales(centers, normals, heights, steps, triangles, own_faces,
                       reference_chord, max_halvings=16):
    """Shrink stencils that cross another sheet or wall triangle."""
    heights=np.asarray(heights,float).copy()
    steps=np.asarray(steps,float).copy()
    low=triangles.min(axis=1); high=triangles.max(axis=1)
    halvings=np.zeros(len(centers),dtype=int)
    tol=1e-10*reference_chord
    eye=np.eye(3)
    for face,center in enumerate(centers):
        for attempt in range(max_halvings+1):
            h=heights[face]; step=steps[face]
            extent=h*np.abs(normals[face])+step
            candidates=np.flatnonzero(np.all(low<=center+extent+tol,axis=1)
                                       & np.all(high>=center-extent-tol,axis=1))
            candidates=candidates[candidates!=own_faces[face]]
            blocked=False
            for side in (-1.,1.):
                probe=center+side*h*normals[face]
                segments=[(center,probe)]
                segments.extend((probe-step*axis,probe+step*axis) for axis in eye)
                for start,end in segments:
                    segment_candidates=candidates[
                        np.all(low[candidates]<=np.maximum(start,end)+tol,axis=1)
                        & np.all(high[candidates]>=np.minimum(start,end)-tol,axis=1)]
                    for index in segment_candidates:
                        if _probe_segment_crosses_triangle(start,end,triangles[index]):
                            blocked=True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if not blocked:
                break
            if attempt==max_halvings:
                raise ValueError(f'wake face {face}: probe intersects another surface after {max_halvings} halvings')
            heights[face]*=.5
            steps[face]*=.5
            halvings[face]+=1
    return heights,steps,halvings


def mean_wake_velocity(mesh: MultiLiftingMesh, state, *,
                       probe_fraction=0.055, step_fraction=0.009,
                       probe_floor_fraction=0.0, block=24,
                       potential_backend='numpy', device='cpu'):
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
    triangles=np.concatenate([wall.triangles]+[_wake_mesh(g).triangles for g in mesh.groups])
    wake_start=len(wall.triangles)
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
        requested_h=h.copy()
        h,step,halvings=_safe_probe_scales(wm.centers,wm.normals,h,step,triangles,
                                           wake_start+np.arange(len(wm.faces)),
                                           mesh.reference_chord)
        wake_start+=len(wm.faces)
        e=np.eye(3)
        centers=wm.centers[:,None,None,:]+(np.array([-1.,1.])[None,:,None,None]
                               *h[:,None,None,None]*wm.normals[:,None,None,:])
        pts=centers[:,:,:,None,:]+np.array([-1.,1.])[None,None,None,:,None]*step[:,None,None,None,None]*e[None,None,:,None,:]
        # pts shape (face,side,axis,+/- sign,xyz), and pairwise gradient
        f=wm.centers.shape[0]
        v=_potentials_at(mesh,state,pts.reshape(-1,3),block=block,cached_wall=wall,
                         backend=potential_backend,device=device)
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
                probe_over_shortest_max=float(np.max(h/shortest)),
                requested_probe_over_shortest_max=float(np.max(requested_h/shortest)),
                probe_adjusted_faces=int(np.count_nonzero(halvings)),
                probe_max_halvings=int(np.max(halvings)))
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


def _one_wake_target(g, mat, vel, *, min_axial=0.08, freeze=False, fixed_span=False):
    """Integrate nodal velocity slopes along frozen material-coordinate rays."""
    current=np.asarray(g.wake_xyz,float)
    original=mat.reference_xyz
    velocity=np.asarray(vel,float).copy()
    if velocity.shape != original.shape or not np.isfinite(velocity).all():
        raise ValueError(f'{g.name}: invalid wake-node velocity field')
    if fixed_span:
        normals=_node_velocities(g,_wake_mesh(g).normals)
        if np.any(np.abs(normals[mat.used,2])<1e-8):
            raise ValueError(f'{g.name}: fixed-span wake requires a single-valued z(x,y) surface')
        velocity[:,2]+=velocity[:,1]*normals[:,1]/np.where(np.abs(normals[:,2])<1e-8,1.,normals[:,2])
        velocity[:,1]=0.
    integral=mat.streamwise_integral
    if integral is None:
        integral=_material_line_integrals(replace(g,wake_xyz=original))
    target=original.copy()
    attachment_mask=np.zeros(len(original),bool)
    attachments=mat.roots if mat.attachments is None else mat.attachments
    attachment_mask[np.asarray(attachments,int)]=True
    stalled=velocity[:,0]<min_axial
    blocked=(integral @ stalled.astype(float)>1e-12)|stalled
    free=np.zeros(len(original),bool);free[mat.used]=True
    free&=~attachment_mask
    frozen=np.flatnonzero(free & blocked)
    if len(frozen) and not freeze:
        raise ValueError(f'{g.name}: wake streamwise marching requires Vx>0; '
                         'reverse flow needs pseudo-time propagation')
    denominator=np.where(stalled,1.,velocity[:,0])
    target[:,1:]+=integral @ (velocity[:,1:]/denominator[:,None])
    target[frozen]=current[frozen]
    target[attachment_mask]=original[attachment_mask]
    return target,frozen.tolist()

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
    if static.get('wall_formulation')=='galerkin':
        from .p1_galerkin_wall import assemble_galerkin_multi
        return assemble_galerkin_multi(mesh,order=static['order'],block=static['block'],
                                       target_face_batch=static['target_face_batch'],static=static)
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
                           max_step_chord=0.0, slip_gate=0.0,
                           wake_potential_backend='numpy', wake_device='cpu',
                           orientation_backtracking=True, wall_formulation='galerkin',
                           galerkin_order=3, fixed_span=False):
    """Return final state, relaxed wake mesh, and complete iteration history.

    Iteration 0 is ALWAYS the original fixed-wake solution; downstream wake
    vertices are then convected in x as a material sheet (y and z drift).
    Gamma is recomputed in the current wake geometry at EVERY iteration.
    Run the requested number of updates unless a geometry guard blocks one;
    convergence is reported on the final state and does not stop updates.
    fixed_span is experimental: it holds y fixed and corrects the z slope
    using the sheet normal, but does not transport Gamma along drifting rays.
    """
    if not (isinstance(iterations,int) and 0<=iterations<=100
            and 0<relaxation<=1 and geometry_tol>0 and slip_tol>0):
        raise ValueError('Invalid relaxation/iteration count')
    t0=time.perf_counter();mat={g.name:prepare_wake_material(g,streamwise=True) for g in mesh.groups}
    if wall_formulation=='galerkin':
        from .p1_galerkin_wall import assemble_galerkin_multi
        fixed=assemble_galerkin_multi(mesh,order=galerkin_order)
    elif wall_formulation=='collocation':
        fixed=assemble_multi(mesh)
    else:
        raise ValueError("wall_formulation must be collocation or galerkin")
    current=mesh
    history=[]
    state=None
    blocked_geometry=False
    for i in range(iterations+1):
        ops=reassemble_wake_only(current,fixed) if i else fixed
        state=solve_multi(current,alpha_deg,kutta=kutta,max_nfev=max_nfev,operators=ops,
                          wall_formulation=wall_formulation)
        panel_velocity,slip=mean_wake_velocity(current,state,
                    probe_fraction=probe_fraction,step_fraction=step_fraction,
                    probe_floor_fraction=probe_floor_fraction,
                    block=wake_velocity_block,potential_backend=wake_potential_backend,
                    device=wake_device)
        record=dict(iteration=i,CL=state['CL'],CL_gamma=state['CL_gamma'],
                CD_pressure=state['CD_pressure'],CM_y=state['CM_y'],
                elapsed_seconds=time.perf_counter()-t0,
                wall_rms=state['wall_rms'],jump_rms=state['jump_rms'],
                wall_formulation=state['wall_formulation'],wall_dual_rms=state['wall_dual_rms'],
                pressure_kutta_rms=state['pressure_kutta_rms'],
                wakes=slip,groups=state['group_results'],
                wake_geometry_change=None)
        if save_callback:
            save_callback(i,current,state,record)
        history.append(record)
        target={};displacement=[];frozen={}
        for g in current.groups:
            m=mat[g.name]
            v=_node_velocities(g,panel_velocity[g.name])
            t,frozen_nodes=_one_wake_target(g,m,v,min_axial=min_axial,
                                            freeze=frozen_guard,fixed_span=fixed_span)
            frozen[g.name]=len(frozen_nodes)
            record.setdefault('frozen_node_ids',{})[g.name]=frozen_nodes
            x_new=g.wake_xyz.copy()
            movable=np.zeros(len(x_new),bool);movable[m.used]=True
            movable[m.attachments]=False
            record.setdefault('raw_map_residual',{})[g.name]=float(
                np.linalg.norm(t[m.used]-g.wake_xyz[m.used],axis=1).max()/mesh.reference_chord)
            record.setdefault('attachment_target_displacement',{})[g.name]=float(
                np.linalg.norm(t[m.attachments]-m.reference_xyz[m.attachments],axis=1).max())
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
                record.setdefault('clipped_nodes',{})[g.name]=np.flatnonzero(movable)[scale<1.].tolist()
            x_new[movable]=g.wake_xyz[movable]+step
            # The x-stations and material Gamma basis never move/reinterpolate.
            x_new[m.used,0]=m.reference_xyz[m.used,0]
            x_new[m.attachments]=m.reference_xyz[m.attachments]
            d=np.linalg.norm(x_new[movable]-g.wake_xyz[movable],axis=1)
            displacement.append(float(d.max()/mesh.reference_chord) if len(d) else 0.)
            record.setdefault('relaxed_nodes',{})[g.name]=int(movable.sum())
            target[g.name]=x_new
        record['wake_geometry_change']=max(displacement)
        record['frozen_nodes']=frozen
        if i==iterations:
            record['proposed_geometry_change']=record['wake_geometry_change']
            record['wake_geometry_change']=history[-2]['wake_geometry_change'] if i>0 else None
            break
        if orientation_backtracking:
            record['orientation_diagnostics']={}
            for g in current.groups:
                faces=g.wake_faces
                reference=mat[g.name].reference_xyz[faces]
                ref_cross=np.cross(reference[:,1]-reference[:,0],reference[:,2]-reference[:,0])
                tri=g.wake_xyz[faces]
                cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                trial=target[g.name][faces]
                trial_cross=np.cross(trial[:,1]-trial[:,0],trial[:,2]-trial[:,0])
                area2=np.einsum('fi,fi->f',ref_cross,ref_cross)
                ratio=np.einsum('fi,fi->f',cross,ref_cross)/area2
                trial_ratio=np.einsum('fi,fi->f',trial_cross,ref_cross)/area2
                face=int(np.argmin(trial_ratio))
                record['orientation_diagnostics'][g.name]=dict(
                    face=face,nodes=faces[face].tolist(),
                    centroid=tri[face].mean(axis=0).tolist(),
                    velocity=panel_velocity[g.name][face].tolist(),
                    current_ratio=float(ratio[face]),trial_ratio=float(trial_ratio[face]),
                    minimum_current_ratio=float(ratio.min()))
            for attempt in range(25):
                factor=2.**(-attempt)
                candidate={}
                admissible=True
                for g in current.groups:
                    xyz=g.wake_xyz+factor*(target[g.name]-g.wake_xyz)
                    tri=xyz[g.wake_faces]
                    reference=mat[g.name].reference_xyz[g.wake_faces]
                    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    ref_cross=np.cross(reference[:,1]-reference[:,0],reference[:,2]-reference[:,0])
                    admissible &= bool(np.all(np.einsum('fi,fi->f',cross,ref_cross)>0.)
                                       and np.all(np.linalg.norm(cross,axis=1)>1e-14))
                    candidate[g.name]=xyz
                if admissible:
                    break
            record['orientation_backtracking']=dict(attempts=attempt+1,factor=factor,
                                                     accepted=admissible)
            if not admissible:
                blocked_geometry=True
                record['wake_geometry_change']=0.
                break
            target=candidate
            record['wake_geometry_change']*=factor
        current=with_wake_xyz(current,target)
    converged=bool(not blocked_geometry and len(history)>1 and history[-1]['wakes']
                    and history[-1]['wake_geometry_change'] is not None
                    and history[-1]['wake_geometry_change']<geometry_tol
                    and max(history[-1].get('raw_map_residual',{'none':float('inf')}).values())<geometry_tol
                    and not any(history[-1].get('frozen_nodes',{}).values())
                    and max(x['normal_slip_rms'] for x in history[-1]['wakes'].values())<slip_tol)
    return dict(mesh=current,state=state,history=history,
                iterations=len(history)-1,seconds=time.perf_counter()-t0,
                converged=converged,
                status='blocked_geometry' if blocked_geometry else 'completed',
                geometry_stable=bool(not blocked_geometry and len(history)>1
                    and history[-1]['wake_geometry_change'] is not None
                    and history[-1]['wake_geometry_change']<geometry_tol),
                geometry_tol=geometry_tol,normal_slip_tol=slip_tol,
                method='steady-material-streamline-free-wake-P1-G2',
                physical_accuracy_accepted=False)
