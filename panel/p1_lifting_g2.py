"""Fully coupled independent continuous-P1 lifting groups on triangle meshes.

Research engine on top of P1 G1: each independent lifting component contributes
its OWN physical TE potential-cut constraints, wake doublet sheet, circulation
interpolation/DOFs and Kutta equations. All components interact via ONE wall
potential system. No group is added by superposing separately solved CLs.

Scope: disconnected, full-span generated wings / declared CAD wing group;
synthetic biplane or raised tail. No Gmsh or automatic CAD TE guessing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.optimize import least_squares

from .p1_closed_body import Mesh
from .p1_lifting_g1 import Wing, make_wing, _doublet_matrix, _gradient_map, _interpolation_basis


@dataclass(frozen=True)
class LiftingGroup:
    name: str
    face_ids: np.ndarray
    te_pairs: np.ndarray
    te_xyz: np.ndarray
    te_pair_y: np.ndarray
    wake_xyz: np.ndarray
    wake_faces: np.ndarray
    gamma_y: np.ndarray
    gamma_hat: np.ndarray
    chord: float
    reference_area: float


@dataclass(frozen=True)
class MultiLiftingMesh:
    xyz: np.ndarray
    faces: np.ndarray
    face_bank: np.ndarray
    face_group: np.ndarray
    dof_face: np.ndarray
    ndof_mu: int
    groups: tuple[LiftingGroup, ...]
    reference_area: float
    reference_chord: float
    reference_point: np.ndarray


def combine_wings(wings: Sequence[Wing], *, origins=None, names=None,
                  reference_area=None, reference_chord=None, reference_point=(0,0,0)):
    """Combine independent full-span lifting components into one BEM solve.

    Origins translate each component including its physical wake and TE, but
    never transform an individual group into a fake merged Gamma space.
    """
    n=len(wings)
    if n < 1:
        raise ValueError('At least one lifting surface required')
    if origins is None:
        origins=[(0.,0.,0.)]*n
    if names is None:
        names=[f'lifting_{j}' for j in range(n)]
    if len(origins)!=n or len(names)!=n or len(set(names))!=n:
        raise ValueError('Each lifting surface needs unique name and an origin')
    all_xyz=[];all_faces=[];all_dface=[];all_bank=[];all_owner=[];groups=[]
    node_offset=0;dof_offset=0;face_offset=0
    for k,(wing,trans,name) in enumerate(zip(wings,origins,names)):
        origin=np.asarray(trans,float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('Invalid lifting surface translation')
        xyz=np.asarray(wing.xyz,float)+origin
        faces=np.asarray(wing.faces,int)+node_offset
        dface=np.asarray(wing.dof_face,int)+dof_offset
        all_xyz.append(xyz);all_faces.append(faces)
        all_dface.append(dface);all_bank.append(np.asarray(wing.face_bank,int))
        all_owner.extend([k]*len(faces))
        # Resolve each actual paired TE station to its physical 3-D point;
        # wing.te_geo may additionally contain the two tip stations.
        te_line=xyz[np.asarray(wing.te_geo,int)]
        te_y=np.asarray(wing.te_pair_y,float)+origin[1]
        match=np.argmin(abs(te_line[:,1,None]-te_y[None,:]),axis=0)
        te_xyz=te_line[match]
        groups.append(LiftingGroup(
            name=str(name),face_ids=np.arange(face_offset,face_offset+len(faces)),
            te_pairs=np.asarray(wing.te_pairs,int)+dof_offset,
            te_xyz=te_xyz,
            te_pair_y=np.asarray(wing.te_pair_y,float)+origin[1],
            wake_xyz=np.asarray(wing.wake_xyz,float)+origin,
            wake_faces=np.asarray(wing.wake_faces,int),
            gamma_y=np.asarray(wing.gamma_y,float)+origin[1],
            gamma_hat=np.asarray(wing.gamma_hat,float),
            chord=float(wing.chord),reference_area=float(wing.reference_area)))
        node_offset+=len(xyz);dof_offset+=int(wing.ndof_mu);face_offset+=len(faces)
    if reference_area is None:
        reference_area=sum(g.reference_area for g in groups)
    if reference_chord is None:
        reference_chord=groups[0].chord
    point=np.asarray(reference_point,float)
    if not (np.isfinite(reference_area) and reference_area>0
            and np.isfinite(reference_chord) and reference_chord>0
            and point.shape==(3,) and np.isfinite(point).all()):
        raise ValueError('Invalid aerodynamic reference')
    return MultiLiftingMesh(
        xyz=np.concatenate(all_xyz),faces=np.concatenate(all_faces),
        face_bank=np.concatenate(all_bank),face_group=np.asarray(all_owner,int),
        dof_face=np.concatenate(all_dface),ndof_mu=dof_offset,
        groups=tuple(groups),reference_area=float(reference_area),
        reference_chord=float(reference_chord),reference_point=point)


def make_conventional_fixture(*, main=(4,6,4.,1.,.02),
                              tail=(4,4,1.8,.55,0.),
                              tail_origin=(1.6,0.,.35),wake_panels=2):
    """A real 3-D lifting main wing + raised horizontal tail, each with own wake."""
    def create(a):
        chord_panels,span_panels,span,chord,camber=a
        return make_wing(chord_panels=chord_panels,span_panels=span_panels,
                         span=span,chord=chord,camber=camber,
                         wake_panels=wake_panels)
    return combine_wings([create(main),create(tail)],
                         origins=[(0.,0.,0.),tail_origin],
                         names=['main_wing','raised_tail'])


def _mesh(mesh):
    tri=mesh.xyz[mesh.faces]
    edge1=tri[:,1]-tri[:,0];edge2=tri[:,2]-tri[:,0]
    av=np.cross(edge1,edge2);norm=np.linalg.norm(av,axis=1)
    if np.any(norm<=0):raise ValueError('Degenerate wet wall triangle')
    normal=av/norm[:,None]
    g1=np.cross(edge2,normal)/norm[:,None];g2=np.cross(normal,edge1)/norm[:,None]
    grad=np.stack((-g1-g2,g1,g2),axis=1)
    return Mesh(mesh.xyz,mesh.faces,np.arange(len(mesh.xyz)),tri,tri.mean(axis=1),
                normal,norm/2,grad,dict(diameter=float(np.linalg.norm(np.ptp(mesh.xyz,axis=0)))))


def assemble_multi(mesh: MultiLiftingMesh):
    """Assemble all wall/wake interactions in ONE coupled potential system.

    No superposition of separate wing solutions; cross-component influence
    appears in A and all wake groups act on every wall collocation target.
    """
    ms=_mesh(mesh);nf=len(mesh.faces);nm=mesh.ndof_mu
    A=_doublet_matrix(ms.triangles,ms.centers,self_faces=np.arange(nf),
                      face_dofs=mesh.dof_face,ndof=nm)
    grad=_gradient_map(mesh,ms)
    W_parts=[];J_parts=[];K_parts=[];TEface=[];slices={};offset=0;groups_receipt=[]
    for gi,group in enumerate(mesh.groups):
        ng=group.gamma_hat.shape[1]
        wake_tris=np.asarray(group.wake_xyz,float)[np.asarray(group.wake_faces,int)]
        # Assemble all wake source triangles in one analytic-kernel traversal.
        # face_dofs maps each source triangle back to the shared wake-node basis,
        # so this is algebraically the sum of the former per-triangle calls.
        coefficients=_doublet_matrix(wake_tris,ms.centers,
            face_dofs=np.asarray(group.wake_faces,int),
            ndof=len(group.wake_xyz))
        Wg=coefficients@np.asarray(group.gamma_hat,float)
        W_parts.append(Wg)
        slices[group.name]=slice(offset,offset+ng)
        offset+=ng
        c=np.zeros((len(group.te_pairs),nm))
        for i,(up,down) in enumerate(group.te_pairs):
            c[i,up]+=1.;c[i,down]-=1.
        H=_interpolation_basis(group.te_pair_y,group.gamma_y)[:,1:-1]
        J_parts.append((c,H))
        kg=np.zeros((len(group.te_pairs),nm));pairs=[]
        for i,(up,down) in enumerate(group.te_pairs):
            y=float(group.te_pair_y[i]); pos=group.te_xyz[i]
            found=[]
            for bank,dof in ((1,up),(-1,down)):
                cand=group.face_ids[(mesh.face_bank[group.face_ids]==bank)&
                    np.any(mesh.dof_face[group.face_ids]==dof,axis=1)]
                if not len(cand):
                    cand=group.face_ids[mesh.face_bank[group.face_ids]==bank]
                if not len(cand):raise ValueError(f'Missing {group.name} TE face bank')
                metric=4*(ms.centers[cand,0]-pos[0])**2+(ms.centers[cand,1]-y)**2
                found.append(cand[np.argsort(metric)[:2]])
            top,bot=found;pairs.append((top,bot))
            kg[i]=(np.mean(grad[top,0,:],axis=0)-np.mean(grad[bot,0,:],axis=0))
        K_parts.append(kg);TEface.append(pairs)
        groups_receipt.append(dict(name=group.name,n_gamma=ng,n_kutta=len(c),
                                   n_wall_faces=len(group.face_ids)))
    W=np.column_stack(W_parts)
    J=[]
    for gi,(c,h) in enumerate(J_parts):
        part=np.zeros((len(c),W.shape[1]));part[:,slices[mesh.groups[gi].name]]=-h
        J.append(np.column_stack((c,part)))
    J=np.concatenate(J,axis=0)
    K=np.column_stack((np.concatenate(K_parts,axis=0),np.zeros((J.shape[0],W.shape[1]))))
    return dict(A=A,W=W,J=J,K=K,grad=grad,mesh=ms,
                slices=slices,te_face_pairs=TEface,group_info=groups_receipt)


def solve_multi(mesh, alpha_deg=5., *, kutta='pressure',kut_weight=5.,
                max_nfev=80,operators=None):
    """Solve wall P1 mu and ALL independent wake Gamma simultaneously."""
    ops=assemble_multi(mesh) if operators is None else operators
    A,W,J,K,grad=ops['A'],ops['W'],ops['J'],ops['K'],ops['grad'];ms=ops['mesh']
    nm=A.shape[1];ng=W.shape[1];alpha=np.deg2rad(alpha_deg)
    U=np.array([np.cos(alpha),0.,np.sin(alpha)])
    rhs=-ms.centers@U
    weights=np.sqrt(ms.areas/ms.areas.mean())
    B=weights[:,None]*np.column_stack((A,W))
    bw=weights*rhs
    def constrained(cons):
        N=null_space(cons,rcond=1e-11)
        if not len(N.T):raise ValueError('All independent mu/Gamma DOFs consumed by constraints')
        z,_,rank,sv=lstsq(B@N,bw,lapack_driver='gelsd')
        return N@z,N,int(rank),sv
    x,Nall,rank,sv=constrained(np.vstack((J,K)))
    pairs=[p for g in ops['te_face_pairs'] for p in g]
    def pressure(vec):
        v=np.einsum('fij,j->fi',grad,vec[:nm])
        return np.array([np.mean(np.sum(v[top]**2,axis=1))-
                         np.mean(np.sum(v[bot]**2,axis=1)) for top,bot in pairs])
    def jac(vec):
        v=np.einsum('fij,j->fi',grad,vec[:nm])
        d=np.array([2*np.mean(np.einsum('fi,fij->fj',v[t],grad[t]),axis=0)-
                    2*np.mean(np.einsum('fi,fij->fj',v[b],grad[b]),axis=0) for t,b in pairs])
        return np.column_stack((d,np.zeros((len(d),ng))))
    if kutta=='pressure':
        Nj=null_space(J,rcond=1e-11)
        def fun(z):
            state=Nj@z
            return np.r_[B@state-bw,kut_weight*pressure(state)]
        def d_fun(z):
            state=Nj@z
            return np.vstack((B,kut_weight*jac(state)))@Nj
        rr=least_squares(fun,Nj.T@x,jac=d_fun,max_nfev=max_nfev,
                         xtol=1e-11,ftol=1e-11,gtol=1e-11)
        x=Nj@rr.x
        nonlinear=dict(success=bool(rr.success),message=rr.message,nfev=rr.nfev)
    elif kutta!='signed_speed':
        raise ValueError('kutta=pressure or signed_speed')
    else:
        nonlinear=None
    mu=x[:nm];gamma=x[nm:];v=np.einsum('fij,j->fi',grad,mu)
    cp=1-np.sum(v**2,axis=1)
    faceforces=-(cp*ms.areas)[:,None]*ms.normals
    net=faceforces.sum(axis=0)
    moment=np.cross(ms.centers-mesh.reference_point,faceforces).sum(axis=0)
    lift=np.array([-np.sin(alpha),0.,np.cos(alpha)])
    totalCL=float(net@lift/mesh.reference_area)
    per=[]
    for gi,g in enumerate(mesh.groups):
        c=faceforces[g.face_ids].sum(axis=0)
        G=gamma[ops['slices'][g.name]]
        Gfull=np.r_[0.,G,0.]
        per.append(dict(name=g.name,CL_pressure=float(c@lift/mesh.reference_area),
                        CL_gamma=float(2*np.trapezoid(Gfull,g.gamma_y)/mesh.reference_area),
                        gamma=G.tolist(),gamma_y=g.gamma_y.tolist(),
                        Gamma_integral=float(np.trapezoid(Gfull,g.gamma_y)),
                        wall_faces=len(g.face_ids),wake_triangles=len(g.wake_faces)))
    return dict(alpha_deg=float(alpha_deg),kutta=kutta,CL=totalCL,
                CL_gamma=float(sum(x['CL_gamma'] for x in per)),
                CD_pressure=float(net@U/mesh.reference_area),
                CM_y=float(moment[1]/(mesh.reference_area*mesh.reference_chord)),
                force=net.tolist(),mu=mu,cp=cp,gamma=gamma,
                group_results=per,
                jump_rms=float(np.linalg.norm(J@x)/np.sqrt(J.shape[0])),
                signed_kutta_rms=float(np.linalg.norm(K@x)/np.sqrt(len(K))),
                pressure_kutta_rms=float(np.linalg.norm(pressure(x))/np.sqrt(len(K))),
                wall_rms=float(np.linalg.norm(A@mu+W@gamma-rhs)/np.sqrt(len(rhs))),
                linear_constrained_rank=rank,linear_singular_min=float(sv[-1]),
                nonlinear=nonlinear,physical_accuracy_accepted=False)


def alpha_sensitivity_multi(mesh,alpha_deg=5.,*,operators=None):
    """Exact fixed-geometry alpha derivative for the coupled G1/G2 signed Kutta.

    Linear state equations have the same constrained operator for all alpha;
    solve both freestream directions in a single factorization. This is not
    geometry VJP or the derivative of nonlinear pressure Kutta.
    """
    ops=assemble_multi(mesh) if operators is None else operators
    A,W,J,K=ops['A'],ops['W'],ops['J'],ops['K'];ms=ops['mesh']
    nm=A.shape[1];theta=np.deg2rad(alpha_deg)
    U=np.array([np.cos(theta),0.,np.sin(theta)])
    dU=np.array([-np.sin(theta),0.,np.cos(theta)])
    weights=np.sqrt(ms.areas/ms.areas.mean())
    N=null_space(np.vstack((J,K)),rcond=1e-11)
    B=(weights[:,None]*np.column_stack((A,W)))@N
    RHS=-weights[:,None]*ms.centers
    coeff,_,_,_=lstsq(B,np.column_stack((RHS@U,RHS@dU)),lapack_driver='gelsd')
    X=N@coeff;mu,dmu=X[:nm,0],X[:nm,1]
    dg=X[nm:,1]
    grad=ops['grad']
    v=np.einsum('fij,j->fi',grad,mu)
    dv=np.einsum('fij,j->fi',grad,dmu)
    cp=1-np.sum(v**2,axis=1)
    dcp=-2*np.einsum('fi,fi->f',v,dv)
    force=-np.einsum('f,f,fi->i',cp,ms.areas,ms.normals)
    dforce=-np.einsum('f,f,fi->i',dcp,ms.areas,ms.normals)
    lift=np.array([-np.sin(theta),0.,np.cos(theta)])
    dlift=np.array([-np.cos(theta),0.,-np.sin(theta)])
    sensitivity=dict(alpha_deg=float(alpha_deg),
        dCL_dalpha_rad=float((dforce@lift+force@dlift)/mesh.reference_area),
        dCM_y_dalpha_rad=float(np.sum(np.cross(ms.centers-mesh.reference_point,
                        -(dcp*ms.areas)[:,None]*ms.normals),axis=0)[1]
                         /(mesh.reference_area*mesh.reference_chord)))
    group_derivatives={}
    for group in mesh.groups:
        dG=dg[ops['slices'][group.name]]
        group_derivatives[group.name]=float(2*np.trapezoid(np.r_[0.,dG,0.],group.gamma_y)
                                           /mesh.reference_area)
    sensitivity['dCL_gamma_dalpha_rad']=float(sum(group_derivatives.values()))
    sensitivity['dCL_gamma_groups_dalpha_rad']=group_derivatives
    return sensitivity
