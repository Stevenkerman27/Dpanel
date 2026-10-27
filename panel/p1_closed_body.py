"""Research G0 continuous-P1 doublet solver, not a lifting-body production path.

For one closed convex oriented polyhedron, solve D^- mu_h = S sigma,
 sigma=-n.U, with continuous nodal P1 mu_h and constant source per triangle.
Face-interior collocation gives an overdetermined system, solved by full-rank
area-weighted QR. This intentionally avoids unverified vertex free terms.
The exterior perturbation is D mu_h - S sigma. No gauge is imposed: A@1=-1.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    used_vertex_ids: np.ndarray
    triangles: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    basis_gradients: np.ndarray
    audit: dict


def prepare_mesh(vertices, faces, *, max_faces=2560, max_nodes=1400) -> Mesh:
    """Compact actual wall nodes; reject open/nonmanifold/inward/nonconvex inputs.

    Convexity is an explicit initial scope restriction, not a mesh repair.
    Unreferenced coordinates (e.g. wake-only nodes) are not wall unknowns.
    """
    v, raw = np.asarray(vertices, float), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
            or raw.ndim != 2 or raw.shape[1:] != (3,)
            or not np.issubdtype(raw.dtype, np.integer) or len(raw) < 4
            or not isinstance(max_faces, int) or isinstance(max_faces, bool)
            or not isinstance(max_nodes, int) or isinstance(max_nodes, bool)
            or max_faces < 4 or max_nodes < 4):
        raise ValueError('Require finite (Nv,3), integer (Nf,3) and positive budgets')
    if len(raw) > max_faces:
        raise MemoryError('G0 dense-reference face budget exceeded')
    f = raw.astype(np.int64, copy=True)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('Face index outside vertex table')
    used, inverse = np.unique(f, return_inverse=True)
    v = v[used].copy(); f = inverse.reshape(f.shape)
    if len(v) > max_nodes:
        raise MemoryError('G0 dense-reference node budget exceeded')
    if len(np.unique(v, axis=0)) != len(v):
        raise ValueError('Coincident wall nodes: physical trace merging is not automatic')
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError('Duplicate physical faces')
    tri = v[f]; centers = tri.mean(axis=1)
    e1, e2 = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    cross = np.cross(e1,e2); area2 = np.linalg.norm(cross,axis=1)
    diameter = float(np.linalg.norm(np.ptp(v,axis=0)))
    if diameter <= 0 or np.any(area2 <= 1e-14*diameter**2):
        raise ValueError('Degenerate or excessively small face')
    normals = cross/area2[:,None]
    edge_table = {}
    for row in f:
        for a,b in ((row[0],row[1]),(row[1],row[2]),(row[2],row[0])):
            edge_table.setdefault((min(a,b),max(a,b)),[]).append((a,b))
    for pair in edge_table.values():
        if len(pair) != 2:
            raise ValueError('G0 requires closed two-manifold wall; no open sheet/half-mesh/wake')
        if pair[0] != pair[1][::-1]:
            raise ValueError('Inconsistent outward winding')
    if len(v)-len(edge_table)+len(f) != 2:
        raise ValueError('Require a single genus-zero G0 boundary')
    # Every vertex must be in every oriented face's inner half-space. This
    # proves the convex-scope geometry instead of trusting a filename.
    for start in range(0,len(f),32):
        side = np.einsum('fvi,fi->fv',v[None]-tri[start:start+32,0,None],
                         normals[start:start+32])
        if np.max(side) > 1e-10*diameter:
            raise ValueError('Require outward convex boundary (no nonconvex support yet)')
    volume = float(np.sum(np.einsum('fi,fi->f',centers-v.mean(axis=0),cross))/6)
    if volume <= 0:
        raise ValueError('Nonpositive enclosed volume')
    g1 = np.cross(e2,normals)/area2[:,None]
    g2 = np.cross(normals,e1)/area2[:,None]
    gradients = np.stack((-g1-g2,g1,g2),axis=1)
    audit = dict(input_vertices=len(vertices), wall_nodes=len(v), wall_faces=len(f),
                 ignored_unreferenced_vertices=len(vertices)-len(v), volume=volume,
                 closed=True, convex=True, euler_characteristic=2,
                 physical_cut_traces=0, diameter=diameter)
    return Mesh(v,f,used,tri,centers,normals,area2/2,gradients,audit)


def influence_block(points, mesh: Mesh, *, self_faces=None):
    """Exact constant-source, P0-doublet and P1-doublet coefficients.

    self_faces are valid ONLY for strict face-interior targets. Every other
    point must be off-surface, or in a disjoint face's plane. No displaced
    collocation, vortex core or clipped denominator is introduced.
    """
    x=np.asarray(points,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or not np.isfinite(x).all():
        raise ValueError('Finite target points required')
    tri,n = mesh.triangles, mesh.normals
    r=tri[None]-x[:,None,None,:]; radius=np.linalg.norm(r,axis=-1)
    if np.any(radius==0):
        raise ValueError('Vertex targets unsupported: use strictly interior collocation')
    r0,r1,r2=r[:,:,0],r[:,:,1],r[:,:,2]
    det=np.einsum('bfi,bfi->bf',r0,np.cross(r1,r2))
    dot=lambda a,b: np.einsum('bfi,bfi->bf',a,b)
    den=(radius.prod(axis=2)+dot(r0,r1)*radius[:,:,2]
         +dot(r1,r2)*radius[:,:,0]+dot(r2,r0)*radius[:,:,1])
    diag=np.zeros(det.shape,dtype=bool)
    if self_faces is not None:
        ids=np.asarray(self_faces)
        if (ids.shape!=(len(x),) or not np.issubdtype(ids.dtype,np.integer)
                or np.any(ids<0) or np.any(ids>=len(tri))):
            raise ValueError('Invalid collocation face IDs')
        b=1/3+np.einsum('bai,bi->ba',mesh.basis_gradients[ids],x-mesh.centers[ids])
        h=np.einsum('bi,bi->b',x-tri[ids,0],n[ids])
        if np.any(b <= 1e-8) or np.any(abs(h)>1e-11*mesh.audit['diameter']):
            raise ValueError('Self trace requires strictly face-interior coplanar targets')
        diag[np.arange(len(x)),ids]=True
        det=np.where(diag,0,det);den=np.where(diag,1,den)
    omega=-2*np.arctan2(det,den)
    D0=np.where(diag,-.5,omega/(4*np.pi))
    height=-np.einsum('bfi,fi->bf',r0,n)
    height=np.where(diag,0.,height)
    edge_sum=np.zeros_like(height);grad_S_t=np.zeros((*height.shape,3))
    for k,l in ((0,1),(1,2),(2,0)):
        edge=tri[:,l]-tri[:,k];length=np.linalg.norm(edge,axis=1)
        outward=np.cross(edge/length[:,None],n)
        rk,rl=r[:,:,k],r[:,:,l]
        d=dot(rk,rl);prod=radius[:,:,k]*radius[:,:,l]
        opposite=d<0
        plus=np.where(opposite,dot(np.cross(rk,rl),np.cross(rk,rl)) /
                      np.where(opposite,prod-d,1.),prod+d)
        gap=2*plus/(radius[:,:,k]+radius[:,:,l]+length[None])
        if np.any(gap <= 0) or not np.isfinite(gap).all():
            raise ValueError('Singular edge interaction in analytic kernel')
        log=np.log1p(2*length[None]/gap)
        edge_sum+=np.einsum('bfi,fi->bf',rk,outward)*log
        grad_S_t-=outward[None]*log[:,:,None]
    S=(edge_sum-height*omega)/(4*np.pi)
    J=(x[:,None]-mesh.centers[None])*D0[:,:,None]+height[:,:,None]*grad_S_t/(4*np.pi)
    D1=D0[:,:,None]/3+np.einsum('fai,bfi->bfa',mesh.basis_gradients,J)
    if not all(np.isfinite(a).all() for a in (S,D0,D1)):
        raise FloatingPointError('Nonfinite analytic influence')
    return S,D0,D1


def collocation(mesh: Mesh, samples_per_face=1):
    if samples_per_face == 1:
        bary=np.array([[1/3,1/3,1/3]])
    elif samples_per_face == 3:
        bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    else:
        raise ValueError('samples_per_face must be 1 or 3')
    points=np.einsum('qa,fai->fqi',bary,mesh.triangles).reshape(-1,3)
    ids=np.repeat(np.arange(len(mesh.faces)),len(bary))
    weights=np.repeat(mesh.areas/len(bary),len(bary))
    return points,ids,weights


def assemble(mesh: Mesh, *, order=1, samples_per_face=1, block_size=32,
             max_matrix_entries=8_000_000):
    """Only P1 trial-space vs P0 trial-space differs; same physical source RHS."""
    if order not in (0,1) or not isinstance(block_size,int) or block_size<1:
        raise ValueError('Invalid order/block size')
    x,ids,w=collocation(mesh,samples_per_face)
    nd=len(mesh.vertices) if order==1 else len(mesh.faces)
    if len(x)*(nd+len(mesh.faces)) > max_matrix_entries:
        raise MemoryError('Dense matrix budget exceeded before assembly')
    A=np.zeros((len(x),nd));S=np.empty((len(x),len(mesh.faces)))
    for start in range(0,len(x),block_size):
        stop=min(start+block_size,len(x))
        sb,d0,d1=influence_block(x[start:stop],mesh,self_faces=ids[start:stop])
        S[start:stop]=sb
        if order==0: A[start:stop]=d0
        else:
            # Scatter all incident-panel basis contributions into shared nodes.
            target=A[start:stop]
            for a in range(3):
                np.add.at(target,(np.arange(stop-start)[:,None],mesh.faces[None,:,a]),d1[:,:,a])
    weights=np.sqrt(w/w.sum())
    identity_error=float(np.max(abs(A@np.ones(nd)+1)))
    if identity_error>1e-9:
        raise ValueError('Closed outward interior constant-density identity failed')
    return dict(A=A,S=S,points=x,self_faces=ids,weights=weights,
                constant_density_identity_error=identity_error)


def solve(vertices, faces, freestream=(1.,0.,0.), *, order=1, samples_per_face=1,
          min_singular_ratio=1e-11, **mesh_options):
    """Actual full-rank solve for unknown nodal mu; no prescribed analytic mu.

    Untruncated weighted QR, not normal equations or a regularized inverse.
    Residual in the overdetermined BIE is NOT expected to be machine zero.
    Report it separately from the QR optimality/stationarity residual.
    """
    mesh=prepare_mesh(vertices,faces,**mesh_options)
    flow=np.asarray(freestream,float)
    if flow.shape!=(3,) or not np.isfinite(flow).all() or np.linalg.norm(flow)==0:
        raise ValueError('Nonzero finite freestream required')
    if not (0<min_singular_ratio<1):
        raise ValueError('Invalid rank threshold')
    ops=assemble(mesh,order=order,samples_per_face=samples_per_face)
    A,S,w=ops['A'],ops['S'],ops['weights'];sigma=-mesh.normals@flow
    rhs=S@sigma;Aw=w[:,None]*A;bw=w*rhs
    Q,R=np.linalg.qr(Aw,mode='reduced')
    singular=np.linalg.svd(R,compute_uv=False)
    ratio=float(singular[-1]/singular[0])
    if len(singular)!=A.shape[1] or ratio<min_singular_ratio:
        raise ValueError('Rank deficient P1 trial space: no hidden gauge or pseudoinverse')
    mu=np.linalg.solve(R,Q.T@bw)
    residual=A@mu-rhs;weighted=w*residual
    norm_rhs=max(float(np.linalg.norm(bw)),np.finfo(float).tiny)
    stationarity=np.linalg.norm(Aw.T@weighted)/(np.linalg.norm(Aw)*norm_rhs)
    report=dict(order=order,n_equations=A.shape[0],n_unknowns=A.shape[1],
                matrix_rank=len(singular),sigma_max=float(singular[0]),
                sigma_min=float(singular[-1]),singular_ratio=ratio,
                constant_density_identity_error=ops['constant_density_identity_error'],
                constant_mode_response_norm=float(np.linalg.norm(A@np.ones(A.shape[1]))),
                gauge_applied=False,weighted_bie_relative_residual=float(np.linalg.norm(weighted)/norm_rhs),
                max_bie_residual=float(abs(residual).max()),
                qr_stationarity_relative_residual=float(stationarity),
                source_flux=float(sigma@mesh.areas),
                physical_accuracy_accepted=False)
    return dict(mesh=mesh,mu=mu,sigma=sigma,freestream=flow,order=order,
                operators=ops,diagnostics=report)


def surface_values(state, *, density=1., reference_area=1., reference_point=(0,0,0)):
    """P1 tangent derivative, NOT a claim of pointwise zero exterior flux."""
    if state['order'] != 1:
        raise ValueError('No P0 recovery substitute: use independent exterior-field A/B')
    refs=np.asarray([density,reference_area],float)
    origin=np.asarray(reference_point,float)
    if not np.isfinite(refs).all() or np.any(refs<=0) or origin.shape!=(3,) or not np.isfinite(origin).all():
        raise ValueError('Positive density/area and finite moment origin required')
    mesh=state['mesh'];mu=state['mu'];flow=state['freestream']
    grad=np.einsum('fa,fai->fi',mu[mesh.faces],mesh.basis_gradients)
    velocity=flow-mesh.normals*(mesh.normals@flow)[:,None]+grad
    cp=1-np.sum(velocity**2,axis=1)/(flow@flow)
    q=.5*density*(flow@flow)
    forces=-q*(cp*mesh.areas)[:,None]*mesh.normals
    moment=np.sum(np.cross(mesh.centers-np.asarray(reference_point),forces),axis=0)
    return dict(cp=cp,velocity=velocity,force=forces.sum(axis=0),moment=moment,
                force_coefficient=forces.sum(axis=0)/(q*reference_area))


def potential(state, points, *, block_size=32):
    """Total exterior potential; callers choose physical off-surface probes."""
    x=np.asarray(points,float);mesh=state['mesh'];out=[]
    for start in range(0,len(x),block_size):
        p=x[start:start+block_size];S,D0,D1=influence_block(p,mesh)
        double=(D0@state['mu'] if state['order']==0 else
                np.einsum('bfa,fa->b',D1,state['mu'][mesh.faces]))
        out.append(p@state['freestream']+double-S@state['sigma'])
    return np.concatenate(out) if out else np.empty(0)


def velocity_fd(state, points, *, step):
    """Fourth-order centered diagnostic derivative. Test step and step/2.

    Not the AD implementation and not used to assemble the P1 solve.
    Every shifted point must remain in the physical exterior.
    """
    x=np.asarray(points,float)
    if not np.isfinite(step) or step<=0:
        raise ValueError('Positive finite diagnostic derivative step required')
    columns=[]
    for k in range(3):
        d=np.eye(3)[k]*step
        columns.append((-potential(state,x+2*d)+8*potential(state,x+d)
                        -8*potential(state,x-d)+potential(state,x-2*d))/(12*step))
    return np.stack(columns,axis=1)
