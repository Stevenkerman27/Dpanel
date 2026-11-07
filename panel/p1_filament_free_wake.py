"""Structured material-line wake coupled to the P1 weak wall."""
from dataclasses import replace
import time

import numpy as np

from .p1_closed_body import influence_block
from .p1_galerkin_wall import assemble_galerkin_multi
from .p1_lifting_g1 import _interpolation_basis
from .p1_lifting_g2 import _mesh, solve_multi
from .p1_steady_free_wake import reassemble_wake_only, with_wake_xyz
from .p1_geometry_contact import coplanar_overlap_area
from .shed_groups import _segment_triangle_t


def structured_wake(group):
    """Validate the generated rectangular wing's tensor-product wake topology."""
    count=len(group.wake_attachment_nodes)
    if count < 2 or len(group.wake_xyz) % count:
        raise ValueError("filament experiment requires a structured wake")
    grid=np.arange(len(group.wake_xyz)).reshape(-1,count)
    if len(grid)<2 or not np.array_equal(np.sort(group.wake_attachment_nodes),grid[0]):
        raise ValueError("filament experiment requires an attached first wake row")
    expected=[]
    for k in range(len(grid)-1):
        for j in range(count-1):
            a,b,c,d=grid[k,j],grid[k+1,j],grid[k+1,j+1],grid[k,j+1]
            expected.extend(((a,b,c),(a,c,d)))
    if not np.array_equal(group.wake_faces,np.asarray(expected)):
        raise ValueError("filament experiment requires the generated structured triangles")
    basis=group.gamma_hat.reshape(len(grid),count,-1)
    if not np.allclose(basis,basis[0],atol=1e-13,rtol=0):
        raise ValueError("Gamma basis must be constant along material lines")
    return grid


def resample_structured_wake(mesh, wake_panels=8):
    """Refine an already structured straight-extruded wake along each line."""
    if not isinstance(wake_panels, int) or wake_panels < 1:
        raise ValueError("wake_panels must be a positive integer")
    groups = []
    for group in mesh.groups:
        grid = structured_wake(group)
        points = np.asarray(group.wake_xyz)[grid]
        fraction = np.linspace(0., 1., len(grid))[:, None, None]
        if not np.allclose(points, points[0] + fraction * (points[-1] - points[0]),
                           atol=1e-10, rtol=1e-10):
            raise ValueError("structured wake is not linearly extruded")
        count = len(grid[0])
        rows = np.linspace(0., 1., wake_panels + 1)[:, None, None]
        xyz = (points[0] + rows * (points[-1] - points[0])).reshape(-1, 3)
        faces = []
        for k in range(wake_panels):
            for j in range(count - 1):
                a = k * count + j
                b = a + count
                faces.extend(((a, b, b + 1), (a, b + 1, a + 1)))
        basis = np.asarray(group.gamma_hat)[grid[0]]
        groups.append(replace(group, wake_xyz=xyz, wake_faces=np.asarray(faces, dtype=int),
                              gamma_hat=np.tile(basis, (wake_panels + 1, 1)),
                              wake_attachment_nodes=np.arange(count)))
    return replace(mesh, groups=tuple(groups))


def structure_cad_wake(mesh, wake_panels=8):
    """Retessellate a contact-free, x-extruded wake without changing its sheet."""
    if not isinstance(wake_panels,int) or wake_panels<1:
        raise ValueError("wake_panels must be a positive integer")
    groups=[]
    for group in mesh.groups:
        if group.contact_pairs is not None and len(group.contact_pairs):
            raise ValueError("涡线尾迹暂不支持尾迹与机身的接触约束")
        xyz=np.asarray(group.wake_xyz,float)
        faces=np.asarray(group.wake_faces,int)
        used=np.unique(faces)
        triangles=xyz[faces]
        scale=max(float(mesh.reference_chord),float(np.ptp(xyz[used],axis=0).max()),1.)
        tol=1e-9*scale
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        edges,counts=np.unique(edges,axis=0,return_counts=True)
        if np.any(counts>2):
            raise ValueError("涡线尾迹不支持非流形尾迹")
        boundary=edges[counts==1]
        vertices=np.unique(boundary)
        stations=np.sort(np.r_[xyz[vertices,1],xyz[group.wake_attachment_nodes,1],group.gamma_y])
        stations=stations[np.r_[True,np.diff(stations)>tol]]
        if len(stations)<2:
            raise ValueError("涡线尾迹需要展向单值的尾迹边界")
        expected=_interpolation_basis(xyz[used,1],group.gamma_y)[:,1:-1]
        if expected.shape!=group.gamma_hat[used].shape or not np.allclose(
                expected,group.gamma_hat[used],atol=1e-10,rtol=0):
            raise ValueError("涡线尾迹需要沿流向不变的环量基函数")
        start=[];end=[]
        for y in stations:
            points=[]
            for a,b in boundary:
                p=xyz[a];q=xyz[b];dy=q[1]-p[1]
                if abs(dy)<=tol:
                    if abs(y-p[1])<=tol:
                        points.extend((p,q))
                elif -tol<=y-min(p[1],q[1]) and y-max(p[1],q[1])<=tol:
                    t=np.clip((y-p[1])/dy,0.,1.)
                    points.append(p+t*(q-p))
            if len(points)<2:
                raise ValueError("涡线尾迹无法重建完整的展向边界")
            points=np.asarray(points)
            start.append(points[np.argmin(points[:,0])])
            end.append(points[np.argmax(points[:,0])])
        start=np.asarray(start);end=np.asarray(end)
        if (np.any(end[:,0]-start[:,0]<=tol)
                or np.ptp(end[:,0])>tol
                or np.max(np.abs(start[:,2]-end[:,2]))>tol
                or np.max(np.abs(xyz[used,2]-np.interp(xyz[used,1],stations,start[:,2])))>tol):
            raise ValueError("涡线尾迹仅支持平直流向拉伸且下游边界平齐的尾迹")
        for node in group.wake_attachment_nodes:
            if np.min(np.linalg.norm(start-xyz[node],axis=1))>tol:
                raise ValueError("涡线尾迹无法保持后缘附着位置")
        rows=[]
        for k in range(wake_panels+1):
            rows.append(start+(end-start)*(k/wake_panels))
        wake_xyz=np.concatenate(rows)
        count=len(stations);wake_faces=[]
        for k in range(wake_panels):
            for j in range(count-1):
                a=k*count+j;b=(k+1)*count+j
                c=b+1;d=a+1
                wake_faces.extend(((a,b,c),(a,c,d)))
        wake_faces=np.asarray(wake_faces,int)
        rebuilt=wake_xyz[wake_faces]
        old_area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                         triangles[:,2]-triangles[:,0]),axis=1).sum()
        new_area=np.linalg.norm(np.cross(rebuilt[:,1]-rebuilt[:,0],
                                         rebuilt[:,2]-rebuilt[:,0]),axis=1).sum()
        if not np.isclose(old_area,new_area,rtol=1e-8,atol=tol**2):
            raise ValueError("涡线尾迹重网格改变了原尾迹范围")
        basis=_interpolation_basis(stations,group.gamma_y)[:,1:-1]
        groups.append(replace(group,wake_xyz=wake_xyz,wake_faces=wake_faces,
                              wake_attachment_nodes=np.arange(count),
                              gamma_hat=np.tile(basis,(wake_panels+1,1))))
    return replace(mesh,groups=tuple(groups))


def wake_ring_segments(group, gamma, subdivisions=1):
    """Approximate a P1 wake by midpoint-strength vortex ribbons."""
    if not isinstance(subdivisions,int) or subdivisions<1:
        raise ValueError("subdivisions must be a positive integer")
    grid=structured_wake(group)
    xyz=group.wake_xyz[grid]
    mu=(group.gamma_hat @ np.asarray(gamma,float))[grid]
    starts=[];ends=[];strength=[]
    for j in range(grid.shape[1]-1):
        for q in range(subdivisions):
            left=q/subdivisions
            right=(q+1)/subdivisions
            a=(1-left)*xyz[:,j]+left*xyz[:,j+1]
            b=(1-right)*xyz[:,j]+right*xyz[:,j+1]
            value=(1-(left+right)/2)*mu[0,j]+(left+right)/2*mu[0,j+1]
            for k in range(len(grid)-1):
                # Ring sign matches influence_block's doublet-potential convention.
                vertices=(a[k],a[k+1],b[k+1],b[k])
                for i in range(4):
                    starts.append(vertices[i]);ends.append(vertices[(i+1)%4])
                    strength.append(-value)
    return np.asarray(starts),np.asarray(ends),np.asarray(strength)


def filament_velocity(points, start, end, strength, core_radius, block=32):
    """Vectorized Rosenhead-Moore segment integral, including self segments."""
    points=np.asarray(points,float)
    start=np.asarray(start,float);end=np.asarray(end,float)
    strength=np.asarray(strength,float)
    if core_radius<=0 or not np.isfinite(core_radius):
        raise ValueError("a finite positive vortex core radius is required")
    delta=end-start
    length=np.linalg.norm(delta,axis=1)
    if np.any(length<=1e-14):
        raise ValueError("zero-length filament")
    tangent=delta/length[:,None]
    out=np.zeros_like(points)
    for first in range(0,len(points),block):
        r=points[first:first+block,None,:]-start[None,:,:]
        axial=np.einsum("bsi,si->bs",r,tangent)
        rho=r-axial[:,:,None]*tangent[None,:,:]
        effective=np.einsum("bsi,bsi->bs",rho,rho)+core_radius**2
        integral=(axial/np.sqrt(effective+axial**2)
                  -(axial-length)/np.sqrt(effective+(axial-length)**2))/effective
        out[first:first+block]=np.sum(np.cross(tangent[None,:,:],rho)
                                     *(integral*strength[None,:])[:,:,None],axis=1)/(4*np.pi)
    return out


def wake_velocity(mesh, state, points, core_radius, subdivisions, step_fraction=1e-4):
    """Freestream + exact P1 wall finite difference + regularized ribbon wake."""
    points=np.asarray(points,float)
    wall=_mesh(mesh)
    mu=np.asarray(state["mu"])[mesh.dof_face]
    theta=np.deg2rad(state["alpha_deg"])
    velocity=np.broadcast_to([np.cos(theta),0.,np.sin(theta)],points.shape).copy()
    step=step_fraction*mesh.reference_chord
    for axis in range(3):
        offset=np.eye(3)[axis]*step
        for first in range(0,len(points),32):
            p=points[first:first+32]
            _,_,plus=influence_block(p+offset,wall)
            _,_,minus=influence_block(p-offset,wall)
            velocity[first:first+len(p),axis]+=np.einsum(
                "bfa,fa->b",plus-minus,mu)/(2*step)
    index=0
    for group in mesh.groups:
        count=group.gamma_hat.shape[1]
        a,b,gamma=wake_ring_segments(group,state["gamma"][index:index+count],subdivisions)
        velocity+=filament_velocity(points,a,b,gamma,core_radius)
        index+=count
    return velocity


def line_target(xyz, velocity, lengths):
    """Integrate segment directions downstream while preserving material lengths."""
    speed=np.linalg.norm(velocity,axis=2)
    if np.any(speed<=1e-12) or not np.isfinite(speed).all():
        raise ValueError("zero or nonfinite wake velocity")
    direction=velocity/speed[:,:,None]
    target=xyz.copy()
    for k in range(len(xyz)-1):
        target[k+1]=target[k]+lengths[k,:,None]*direction[k]
    return target


def audit_nonadjacent_intersections(xyz, faces, tol=1e-9):
    """Detect nonadjacent wake-face crossings and coplanar overlaps."""
    tri=np.asarray(xyz,float)[np.asarray(faces,int)]
    low=tri.min(axis=1);high=tri.max(axis=1);pairs=[]
    for i in range(len(faces)):
        candidates=np.flatnonzero(np.all(low[:i]<=high[i]+tol,axis=1)
                                  &np.all(high[:i]>=low[i]-tol,axis=1))
        for j in candidates:
            if np.intersect1d(faces[i],faces[j]).size:
                continue
            overlap=coplanar_overlap_area(tri[i],tri[j],tol)
            if overlap>tol**2:
                pairs.append([int(i),int(j),"coplanar"])
                continue
            found=False
            for a,b in ((tri[i],tri[j]),(tri[j],tri[i])):
                for start,end in zip(a,np.roll(a,-1,axis=0)):
                    t=_segment_triangle_t(start,end,b,tol)
                    if t is not None and tol<t<1-tol:
                        found=True
                        break
                if found:
                    break
            if found:
                pairs.append([int(i),int(j),"transverse"])
    return pairs


def solve_filament_free_wake(mesh, alpha_deg=4., *, iterations=16, relaxation=.25,
                             core_radius=.05, subdivisions=2, galerkin_order=3,
                             geometry_tol=.002, direction_tol=.01, slip_tol=.01,
                             max_nfev=80, save_callback=None, intersection_guard=False):
    """Run a fixed update budget unless a geometry guard blocks the wake."""
    if iterations<0 or not 0<relaxation<=1:
        raise ValueError("invalid iteration count or relaxation")
    start=time.perf_counter()
    grids={g.name:structured_wake(g) for g in mesh.groups}
    lengths={g.name:np.linalg.norm(np.diff(g.wake_xyz[grids[g.name]],axis=0),axis=2)
             for g in mesh.groups}
    static=assemble_galerkin_multi(mesh,order=galerkin_order)
    current=mesh;history=[];status="completed"
    for iteration in range(iterations+1):
        ops=static if iteration==0 else reassemble_wake_only(current,static)
        state=solve_multi(current,alpha_deg,kutta="pressure",max_nfev=max_nfev,
                          operators=ops,wall_formulation="galerkin")
        midpoints=[(g.wake_xyz[grids[g.name]][:-1]
                    +g.wake_xyz[grids[g.name]][1:]).reshape(-1,3)/2
                   for g in current.groups]
        centers=[g.wake_xyz[g.wake_faces].mean(axis=1) for g in current.groups]
        points=np.concatenate(midpoints+centers)
        velocity=wake_velocity(current,state,points,core_radius,subdivisions)
        targets={};offset=0;direction_error=0.;map_error=0.;updates=0.
        face_offset=sum(len(p) for p in midpoints)
        slips=[];min_area=float("inf");min_projection=float("inf")
        for group in current.groups:
            grid=grids[group.name]
            xyz=group.wake_xyz[grid]
            size=(len(grid)-1)*grid.shape[1]
            v=velocity[offset:offset+size].reshape(len(grid)-1,grid.shape[1],3)
            offset+=size
            direction=np.diff(xyz,axis=0)
            direction/=np.linalg.norm(direction,axis=2)[:,:,None]
            unit=v/np.linalg.norm(v,axis=2)[:,:,None]
            direction_error=max(direction_error,float(np.linalg.norm(direction-unit,axis=2).max()))
            target=line_target(xyz,v,lengths[group.name])
            map_error=max(map_error,float(np.linalg.norm(target-xyz,axis=2).max()/mesh.reference_chord))
            new=group.wake_xyz.copy()
            new[grid]=(1-relaxation)*xyz+relaxation*target
            new[grid[0]]=group.wake_xyz[grid[0]]
            updates=max(updates,float(np.linalg.norm(new-group.wake_xyz,axis=1).max()/mesh.reference_chord))
            targets[group.name]=new
            tri=group.wake_xyz[group.wake_faces]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            area=np.linalg.norm(cross,axis=1)
            normal=cross/area[:,None]
            face_velocity=velocity[face_offset:face_offset+len(tri)]
            face_offset+=len(tri)
            slips.extend(np.einsum("fi,fi->f",normal,face_velocity).tolist())
            original=next(g for g in mesh.groups if g.name==group.name)
            ref=original.wake_xyz[group.wake_faces]
            ref_n=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0])
            min_area=min(min_area,float(np.min(area/np.linalg.norm(ref_n,axis=1))))
            min_projection=min(min_projection,float(np.min(
                np.einsum("fi,fi->f",cross,ref_n)/np.einsum("fi,fi->f",ref_n,ref_n))))
        slip_rms=float(np.sqrt(np.mean(np.asarray(slips)**2)))
        record=dict(iteration=iteration,CL=state["CL"],CL_gamma=state["CL_gamma"],
                    CD_pressure=state["CD_pressure"],CM_y=state["CM_y"],
                    elapsed_seconds=time.perf_counter()-start,
                    wall_dual_rms=state["wall_dual_rms"],jump_rms=state["jump_rms"],
                    pressure_kutta_rms=state["pressure_kutta_rms"],
                    raw_map_residual=map_error,direction_residual=direction_error,
                    normal_slip_rms=slip_rms,minimum_area_ratio=min_area,
                    minimum_projection_ratio=min_projection,
                    proposed_geometry_change=updates)
        history.append(record)
        if save_callback:
            save_callback(iteration,current,state,record)
        if iteration==iterations:
            break
        factor=1.
        reason="geometry"
        for attempt in range(20):
            candidate={};valid=True
            for group in current.groups:
                xyz=group.wake_xyz+factor*(targets[group.name]-group.wake_xyz)
                old=group.wake_xyz[group.wake_faces]
                new=xyz[group.wake_faces]
                old_n=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
                new_n=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
                valid &= bool(np.all(np.linalg.norm(new_n,axis=1)>1e-12*mesh.reference_chord**2)
                              and np.all(np.einsum("fi,fi->f",old_n,new_n)>0))
                candidate[group.name]=xyz
            reason="geometry"
            if valid and intersection_guard:
                for group in current.groups:
                    if audit_nonadjacent_intersections(candidate[group.name],group.wake_faces):
                        valid=False
                        reason="intersection"
                        break
            if valid:
                break
            if attempt<19:
                factor*=.5
        record["backtracking_factor"]=factor
        if not valid:
            status="blocked_"+reason
            break
        current=with_wake_xyz(current,candidate)
    converged=bool(status=="completed" and map_error<geometry_tol
                   and direction_error<direction_tol and slip_rms<slip_tol)
    if converged:
        for group in current.groups:
            if audit_nonadjacent_intersections(group.wake_xyz,group.wake_faces):
                status="invalid_self_intersection"
                converged=False
                break
    return dict(mesh=current,state=state,history=history,status=status,
                converged=converged,seconds=time.perf_counter()-start,
                core_radius=core_radius,subdivisions=subdivisions,
                physical_accuracy_accepted=False)
