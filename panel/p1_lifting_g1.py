"""Experimental end-to-end continuous P1 lifting-wing framework on triangular panels.

Full-span sharp-TE wing with explicit independent upper/lower potential traces,
linear P1 wake doublet sheet, wake jump constraint and a *real* signed tangential
speed Kutta condition. No source sheet for the closed G0 Dirichlet extension.
This is research code; it is not the production P0 half-wing solver.

The fast reference follows the planar P1 doublet analytic moments in
p1_closed_body.influence_block.  The wake is a real triangulated doublet sheet,
not a fitted far-field load. Supports multiple alpha without remeshing.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.optimize import least_squares
from .p1_closed_body import Mesh, influence_block


@dataclass(frozen=True)
class Wing:
    xyz: np.ndarray
    faces: np.ndarray
    face_bank: np.ndarray  # upper=+1, lower=-1, end cap=0
    te_geo: np.ndarray     # geometric node per span station
    te_up: np.ndarray      # P1 potential DOF per span station
    te_down: np.ndarray
    te_pairs: np.ndarray  # (N_TE,2) physical upper/lower wall P1 DOFs
    te_pair_y: np.ndarray  # TE row span locations
    dof_face: np.ndarray
    ndof_mu: int
    wake_xyz: np.ndarray
    wake_faces: np.ndarray
    gamma_y: np.ndarray
    gamma_hat: np.ndarray
    chord: float
    span: float
    section: str
    reference_area: float
    reference_point: np.ndarray


def _airfoil(x, *, thickness=.12, camber=.0):
    """Closed-TE NACA-00XX thickness; optional smooth symmetric-zero-end camber."""
    t=5*thickness*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
    return t, 4*camber*x*(1-x)


def make_wing(*, chord_panels=6, span_panels=6, thickness=.12,
              camber=0.0, span=4.0, chord=1.0, wake_length=5.,
              wake_panels=3):
    """Closed full-span thick wing: identical geometric TE nodes, split potential traces.

    The two end caps enforce a closed physical wall. Gamma=0 at the tips;
    the interior span TE nodes have duplicated *potential*, not geometry.
    """
    if chord_panels<3 or span_panels<3 or wake_panels<1 or not (0<thickness<.3):
        raise ValueError('Need small positive thickness and sufficient wing resolution')
    x=.5*(1-np.cos(np.linspace(0,np.pi,chord_panels+1)))
    y=np.linspace(-span/2,span/2,span_panels+1)
    t,zcam=_airfoil(x,thickness=thickness,camber=camber)
    vertices=[]; ids={}
    def index(i,j,bank):
        tag=(i,j,0) if i in (0,chord_panels) else (i,j,bank)
        if tag not in ids:
            ids[tag]=len(vertices)
            vertices.append([chord*x[i],y[j],chord*(zcam[i]+(t[i] if bank==1 else -t[i]))])
        return ids[tag]
    up=np.array([[index(i,j,1) for j in range(len(y))] for i in range(len(x))],int)
    lo=np.array([[index(i,j,-1) for j in range(len(y))] for i in range(len(x))],int)
    faces=[];bank=[]
    for side,grid in ((1,up),(-1,lo)):
        for i in range(chord_panels):
            for j in range(span_panels):
                a,b,c,d=grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]
                pair=((a,b,c),(a,c,d)) if side==1 else ((a,c,b),(a,d,c))
                for f in pair:faces.append(f);bank.append(side)
    # End caps: fan from an interior centroid; orient exactly +/-y.
    for j,desired in ((0,-1),(span_panels,+1)):
        perimeter=[int(up[i,j]) for i in range(chord_panels+1)] + [int(lo[i,j]) for i in range(chord_panels-1,0,-1)]
        center=np.array([.42*chord,y[j],chord*np.interp(.42,x,zcam)])
        centerid=len(vertices);vertices.append(center.tolist())
        for k in range(len(perimeter)):
            a,b=perimeter[k],perimeter[(k+1)%len(perimeter)]
            trial=np.array([vertices[a],vertices[b],vertices[centerid]])
            norm=np.cross(trial[1]-trial[0],trial[2]-trial[0])
            if norm[1]*desired<0:faces.append((b,a,centerid))
            else:faces.append((a,b,centerid))
            bank.append(0)
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);bank=np.asarray(bank,int)
    # New unknown for the lower-face trace at interior TE stations only.
    geom_te=up[-1].copy(); dface=faces.copy();down=np.copy(geom_te);up_d=np.copy(geom_te)
    extra={int(geom_te[j]):len(vertices)+j-1 for j in range(1,span_panels)}
    for facei in range(len(faces)):
        if bank[facei]==-1:
            dface[facei]=[extra.get(int(v),int(v)) for v in faces[facei]]
    for j in range(1,span_panels):down[j]=extra[int(geom_te[j])]
    # Wake, oriented +z: x cross y = +z; same span stations as trailing edge.
    wake_verts=[]
    for k in range(wake_panels+1):
        X=chord*(1+wake_length*k/wake_panels)
        for j,yy in enumerate(y):
            wake_verts.append([X,yy,float(vertices[geom_te[j],2])])
    wake_verts=np.array(wake_verts,float)
    wake_faces=[]; stride=len(y)
    for k in range(wake_panels):
        for j in range(span_panels):
            a,b,c,d=k*stride+j,(k+1)*stride+j,(k+1)*stride+j+1,k*stride+j+1
            wake_faces.extend([(a,b,c),(a,c,d)])
    gamma_hat=np.zeros((len(wake_verts),span_panels-1))
    for k in range(wake_panels+1):
        for j in range(1,span_panels):gamma_hat[k*stride+j,j-1]=1.
    pair=np.stack([up_d[1:-1],down[1:-1]],axis=1)
    return Wing(vertices,faces,bank,geom_te,up_d,down,pair,y[1:-1],dface,len(vertices)+span_panels-1,
                wake_verts,np.asarray(wake_faces,int),y,gamma_hat,chord,span,
                'NACA00 closed TE' if camber==0 else 'smooth cambered closed TE',
                chord*span,np.zeros(3))


def _interpolation_basis(values, stations):
    values=np.asarray(values,float).reshape(-1)
    stations=np.asarray(stations,float)
    out=np.zeros((len(values),len(stations)))
    for k in range(len(stations)-1):
        hit=(values>=stations[k]-1e-12)&(values<=stations[k+1]+1e-12)
        t=np.clip((values[hit]-stations[k])/(stations[k+1]-stations[k]),0,1)
        out[hit,k]=1-t;out[hit,k+1]=t
    if np.any(abs(out.sum(axis=1)-1)>1e-10):raise ValueError('Stations do not cover wake/TE y')
    return out


def _from_dpanel_half_surface_edges(path):
    """Legacy adapter for surfaces that only declare trailing_edges vertex pairs.

    Keep for schemas whose TE banks really are distinct vertex pairs. Surfaces
    that declare upper_te_faces/lower_te_faces go through the fixed adapter
    below, which pairs the shared TE edge and splits only the potential trace.
    """

    data=json.loads(Path(path).read_text(encoding='utf-8'))
    xyz=np.asarray(data['coordinates'],float)
    faces=np.asarray(data['faces'],int)
    wake=np.asarray(data['wake_faces'],int)
    te=np.asarray(data['trailing_edges'],int)
    half_y=np.asarray(data['gamma_stations'],float)
    tol=1e-10*max(1.,float(np.ptp(xyz,axis=0).max()))
    if np.any(xyz[:,1]<-tol) or half_y[0]<-tol or np.any(np.diff(half_y)<=0):
        raise ValueError('Half surface must have nonnegative y and increasing stations')
    mirror_map=np.arange(len(xyz),dtype=int)
    others=np.flatnonzero(xyz[:,1]>tol)
    mirror_map[others]=len(xyz)+np.arange(len(others))
    reflected=xyz[others]*np.array([1.,-1.,1.])
    vertices=np.vstack([xyz,reflected])
    full_faces=np.vstack([faces,mirror_map[faces][:,::-1]])
    full_wake=np.vstack([wake,mirror_map[wake][:,::-1]])
    # Per-face bank is used only at declared TE locations, never to decide
    # which unrelated CAD part gets an independent circulation unknown.
    tri=vertices[full_faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    face_bank=np.where(normal[:,2]>1e-12,1,np.where(normal[:,2]<-1e-12,-1,0))
    full_te=np.unique(np.vstack([te,mirror_map[te]]),axis=0)
    te_y=np.mean(vertices[full_te][:,:,1],axis=1)
    order=np.argsort(te_y)
    full_te=full_te[order];te_y=te_y[order]
    ys=np.r_[-half_y[:0:-1],half_y]
    inner=(te_y>ys[0]+tol)&(te_y<ys[-1]-tol)
    full_te=full_te[inner];te_y=te_y[inner]
    # Some meshes have multiple geometric TE face pairs at a given station.
    # Keep each as a row in the Kutta functional, while Gamma(y) retains the
    # original prescribed independent station space.
    used_wall=np.unique(full_faces)
    compact=np.full(len(vertices),-1,dtype=int)
    compact[used_wall]=np.arange(len(used_wall))
    dface=compact[full_faces].copy();nd=len(used_wall)
    tips=[float(ys[0]),float(ys[-1])]
    duplicated={}
    for u,l in full_te:
        if u==l and all(abs(float(vertices[u,1])-tip)>tol for tip in tips):
            if int(u) not in duplicated:
                duplicated[int(u)]=nd;nd+=1
    for f in range(len(full_faces)):
        if face_bank[f]<0:
            dface[f]=[duplicated.get(int(v),int(compact[v])) for v in full_faces[f]]
    p1_te=compact[full_te].copy()
    p1_te[:,1]=np.asarray([duplicated.get(int(k),int(compact[k])) for k in full_te[:,1]])
    # Pin the wake tips to gamma=0, but give the root/mid-span its own DOF.
    H=_interpolation_basis(vertices[:,1][np.unique(full_wake)],ys)
    wids=np.unique(full_wake)
    Ghat=np.zeros((len(vertices),len(ys)-2))
    Ghat[wids]=H[:,1:-1]
    ref_area=float(data.get('reference_area',float(ys[-1]-ys[0])*data['reference_chord']))
    chord=float(data.get('reference_chord',1.))
    rp=np.asarray(data.get('reference_point',[0.,0.,0.]),float)
    return Wing(vertices,full_faces,face_bank,full_te[:,0],full_te[:,0],full_te[:,1],
                p1_te,te_y,dface,nd,vertices,full_wake,ys,Ghat,chord,
                float(ys[-1]-ys[0]),f'half CAD mirrored: {Path(path).name}',ref_area,rp)



def from_dpanel_half_surface(path):
    """Load a true positive-y DPanel half-surface and create a FULL mirrored G1.

    The cross-TE pairing comes from the declared ``upper_te_faces`` /
    ``lower_te_faces`` arrays: each pair shares the physical TE edge, so a sharp
    TE has the same geometric node on both banks and only the P1 POTENTIAL trace
    is split. ``trailing_edges`` is the TE polyline (a chain of TE edges), not a
    bank pairing; mistaking the chain for bank pairs leaves the model without a
    circulation unknown and forces Gamma to its minimum-norm value of zero.

    Surfaces without upper/lower TE face arrays fall back to the legacy
    trailing_edges adapter.
    """
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if 'upper_te_faces' not in data or 'lower_te_faces' not in data:
        return _from_dpanel_half_surface_edges(path)
    xyz=np.asarray(data['coordinates'],float)
    faces=np.asarray(data['faces'],int)
    wake=np.asarray(data['wake_faces'],int)
    up_te=np.asarray(data['upper_te_faces'],int)
    lo_te=np.asarray(data['lower_te_faces'],int)
    half_y=np.asarray(data['gamma_stations'],float)
    tol=1e-10*max(1.,float(np.ptp(xyz,axis=0).max()))
    if np.any(xyz[:,1]<-tol) or half_y[0]<-tol or np.any(np.diff(half_y)<=0):
        raise ValueError('Half surface must have nonnegative y and increasing stations')
    mirror_map=np.arange(len(xyz),dtype=int)
    others=np.flatnonzero(xyz[:,1]>tol)
    mirror_map[others]=len(xyz)+np.arange(len(others))
    reflected=xyz[others]*np.array([1.,-1.,1.])
    vertices=np.vstack([xyz,reflected])
    n_faces=len(faces)
    full_faces=np.vstack([faces,mirror_map[faces][:,::-1]])
    full_wake=np.vstack([wake,mirror_map[wake][:,::-1]])
    tri=vertices[full_faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    face_bank=np.where(normal[:,2]>1e-12,1,np.where(normal[:,2]<-1e-12,-1,0))
    # Mirrored face k is row n_faces+k with reversed winding.
    up_full=np.concatenate([up_te,up_te+n_faces])
    lo_full=np.concatenate([lo_te,lo_te+n_faces])
    nodes=[]
    for fu,fl in zip(up_full,lo_full):
        shared=np.intersect1d(full_faces[fu],full_faces[fl])
        if not len(shared):
            raise ValueError('upper/lower TE faces share no node at row %d; '
                             'refusing to guess a TE bank pairing'%int(fu))
        nodes.extend(int(v) for v in shared)
    te_nodes=np.unique(np.asarray(nodes,int))
    te_y=vertices[te_nodes,1]
    order=np.argsort(te_y,kind='stable')
    te_nodes=te_nodes[order];te_y=te_y[order]
    # The exported wake stations may include a fuselage-contact interval
    # inboard of the actual wing TE (e.g. y=0, 0.2, 0.249996 at a T-tail
    # root). Those stations are needed to mesh the wake/contact topology but
    # must not become independent circulation DOFs: their W columns have
    # near-zero wall support while the TE/Kutta constraints still act on them.
    # Build the solver's Gamma basis only on the actual positive-y TE span.
    positive_te=te_y[te_y>=-tol]
    if not len(positive_te):
        raise ValueError('No nonnegative TE stations found in half-surface')
    te_root=float(np.min(positive_te))
    te_tip=float(np.max(positive_te))
    station_map=np.asarray(data.get('gamma_station_dof',np.arange(len(half_y))),int)
    if station_map.shape!=(len(half_y),):
        raise ValueError('gamma_station_dof length must match gamma_stations')
    keep=(half_y>=te_root-tol)&(half_y<=te_tip+tol)
    excluded=np.flatnonzero(~keep)
    root_candidates=np.flatnonzero(np.abs(half_y-te_root)<=tol)
    if len(root_candidates):
        root_label=int(station_map[root_candidates[0]])
        # Excluded contact stations may be removed only if the exporter
        # explicitly tied them to the physical TE-root circulation station.
        unrelated=[int(i) for i in excluded
                   if station_map[i]!=root_label and station_map[i]>=0]
        if unrelated:
            i=unrelated[0]
            raise ValueError(
                f'Gamma station y={half_y[i]:.6f} lies inboard of the TE trace '
                f'(root {te_root:.6f}) but is not tied to the TE-root DOF; '
                'refusing a zero-support circulation basis')
    else:
        raise ValueError('Gamma stations must include the actual TE-root y station')
    clipped=half_y[keep]
    if len(clipped)<2 or abs(clipped[-1]-te_tip)>tol:
        raise ValueError('Gamma stations do not cover the actual TE root-to-tip span')
    if abs(clipped[0]-te_root)>tol:
        clipped=np.r_[te_root,clipped]
    if abs(clipped[-1]-te_tip)>tol:
        clipped=np.r_[clipped,te_tip]
    half_y=np.unique(clipped)
    ys=np.r_[-half_y[:0:-1],half_y]
    if te_y[0]<ys[0]-tol or te_y[-1]>ys[-1]+tol:
        raise ValueError('TE nodes fall outside the clipped gamma station span')
    used_wall=np.unique(full_faces)
    compact=np.full(len(vertices),-1,dtype=int)
    compact[used_wall]=np.arange(len(used_wall))
    dface=compact[full_faces].copy();nd=len(used_wall)
    # Gamma is pinned to zero at the ACTUAL TE tips (the extremes of the TE
    # node set), never at a station extrapolated outside the mesh.
    tips=[float(ys[0]),float(ys[-1])]
    inner=(te_y>tips[0]+tol)&(te_y<tips[1]-tol)
    te_nodes=te_nodes[inner];te_y=te_y[inner]
    duplicated={}
    for node in te_nodes:
        if int(node) not in duplicated:
            duplicated[int(node)]=nd;nd+=1
    for f in range(len(full_faces)):
        if face_bank[f]<0:
            dface[f]=[duplicated.get(int(v),int(compact[v])) for v in full_faces[f]]
    te_up=compact[te_nodes]
    te_down=np.asarray([duplicated[int(v)] for v in te_nodes],int)
    pair=np.stack([te_up,te_down],axis=1)
    H=_interpolation_basis(vertices[:,1][np.unique(full_wake)],ys)
    wids=np.unique(full_wake)
    Ghat=np.zeros((len(vertices),len(ys)-2))
    Ghat[wids]=H[:,1:-1]
    ref_area=float(data.get('reference_area',float(ys[-1]-ys[0])*data['reference_chord']))
    chord=float(data.get('reference_chord',1.))
    rp=np.asarray(data.get('reference_point',[0.,0.,0.]),float)
    return Wing(vertices,full_faces,face_bank,te_nodes,te_up,te_down,pair,te_y,dface,nd,
                vertices,full_wake,ys,Ghat,chord,float(ys[-1]-ys[0]),
                f'half CAD mirrored: {Path(path).name}',ref_area,rp)

def _mesh_as_kernel(wing):
    tri=wing.xyz[wing.faces]
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if np.any(area2<1e-12):raise ValueError('Degenerate wing triangle')
    norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])/area2[:,None]
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    g1=np.cross(e2,norm)/area2[:,None];g2=np.cross(norm,e1)/area2[:,None]
    gradients=np.stack((-g1-g2,g1,g2),axis=1)
    return Mesh(wing.xyz,wing.faces,np.arange(len(wing.xyz)),tri,tri.mean(axis=1),norm,area2/2,gradients,
               dict(diameter=float(np.linalg.norm(np.ptp(wing.xyz,axis=0)))))


def _doublet_matrix(tris, targets, *, self_faces=None, face_dofs=None, ndof=None,
                    block=512):
    # Build from the single triangle analytic source moments; exact direct kernel.
    tris=np.asarray(tris,float);targets=np.asarray(targets,float)
    if face_dofs is None: face_dofs=np.arange(len(tris)*3).reshape(-1,3)
    if ndof is None:ndof=int(np.max(face_dofs))+1
    A=np.zeros((len(targets),ndof))
    for j,tri in enumerate(tris):
        # Same formula as p1_closed_body.influence_block, but vectorized only over targets.
        e1=tri[1]-tri[0];e2=tri[2]-tri[0];n=np.cross(e1,e2)
        a2=np.linalg.norm(n)
        if a2<=0: raise ValueError('Zero-area source triangle')
        n/=a2;g1=np.cross(e2,n)/a2;g2=np.cross(n,e1)/a2;G=np.stack((-g1-g2,g1,g2));ctr=tri.mean(axis=0)
        for start in range(0,len(targets),block):
            stop=min(len(targets),start+block);X=targets[start:stop]
            r=tri[None]-X[:,None,:];L=np.linalg.norm(r,axis=2)
            if np.any(L==0):raise ValueError('Source vertex target')
            a,b,c=r[:,0],r[:,1],r[:,2]
            dot=lambda u,v:np.einsum('bi,bi->b',u,v)
            den=L.prod(axis=1)+dot(a,b)*L[:,2]+dot(b,c)*L[:,0]+dot(c,a)*L[:,1]
            D=-2*np.arctan2(dot(a,np.cross(b,c)),den)/(4*np.pi)
            h=(X-tri[0])@n;gd=np.zeros((len(X),3))
            for k,l in ((0,1),(1,2),(2,0)):
                edge=tri[l]-tri[k];ll=np.linalg.norm(edge);out=np.cross(edge/ll,n)
                dotr=dot(r[:,k],r[:,l]);product=L[:,k]*L[:,l];op=dotr<0
                cr=np.cross(r[:,k],r[:,l]);plus=np.where(op,np.einsum('bi,bi->b',cr,cr)/np.where(op,product-dotr,1.),product+dotr)
                gap=2*plus/(L[:,k]+L[:,l]+ll)
                if np.any(gap<=0):raise FloatingPointError('Near-singular edge geometry')
                gd-=out[None]*np.log1p(2*ll/gap)[:,None]
            M=(X-ctr)*D[:,None]+h[:,None]*gd/(4*np.pi)
            rows=D[:,None]/3+M@G.T
            if self_faces is not None:
                indices=np.flatnonzero(np.asarray(self_faces[start:stop])==j)
                if len(indices):
                    B=1/3+(X[indices]-ctr)@G.T
                    rows[indices]=-.5*B
            for k in range(3):A[start:stop,face_dofs[j,k]]+=rows[:,k]
    return A


def _gradient_map(wing,mesh):
    grad=np.zeros((len(wing.faces),3,wing.ndof_mu))
    for i in range(len(wing.faces)):
        for k in range(3):grad[i,:,wing.dof_face[i,k]]+=mesh.basis_gradients[i,k]
    return grad


def assemble(wing):
    mesh=_mesh_as_kernel(wing)
    nface=len(wing.faces);nt=wing.ndof_mu;ng=wing.gamma_hat.shape[1]
    A=_doublet_matrix(mesh.triangles,mesh.centers,self_faces=np.arange(nface),face_dofs=wing.dof_face,ndof=nt)
    W=np.zeros((nface,ng))
    for j,f in enumerate(wing.wake_faces):
        D=_doublet_matrix(wing.wake_xyz[f][None],mesh.centers,face_dofs=np.arange(3)[None],ndof=3)
        W+=D@wing.gamma_hat[f]
    # mu_upper - mu_lower = Gamma at every inboard TE station.
    C=np.zeros((len(wing.te_pairs),nt));
    for i,(u,l) in enumerate(wing.te_pairs):
        C[i,u]+=1.;C[i,l]-=1.
    H=_interpolation_basis(wing.te_pair_y,wing.gamma_y)
    # Tip Gamma degrees are prescribed zero and are not unknowns.
    GH=H[:,1:-1]

    # Define face-bank signed tangent speed near TE, at spanwise source strips.
    grad=_gradient_map(wing,mesh)
    Vface=np.array([1.,0.,0.]);K=np.zeros((len(C),nt));pairs=[]
    for k, (uid,lid) in enumerate(wing.te_pairs):
        yy=wing.te_pair_y[k]; idx=[]
        for side,node in ((1,uid),(-1,lid)):
            adjacent=np.flatnonzero(np.any(wing.dof_face==node,axis=1)&(wing.face_bank==side))
            if not len(adjacent):
                # A bank may contain only non-TE-adjacent triangles in a generic
                # CAD fixture; inspect geometry rather than guessing a physical cut.
                adjacent=np.flatnonzero(wing.face_bank==side)
            if not len(adjacent): raise ValueError('Missing upper/lower TE bank faces')
            pair_xyz=wing.xyz[wing.te_geo[k]] if len(wing.te_geo)==len(wing.te_pairs) else np.array([wing.chord,yy,0.])
            metric=4*(mesh.centers[adjacent,0]-pair_xyz[0])**2+(mesh.centers[adjacent,1]-yy)**2
            inds=adjacent[np.argsort(metric)[:2]]
            idx.append(inds)
        top,bot=idx
        K[k]=np.mean(np.einsum('i,fij->fj',Vface,grad[top]),axis=0)-np.mean(np.einsum('i,fij->fj',Vface,grad[bot]),axis=0)
        pairs.append((top,bot))
    return dict(A=A,W=W,C=C,GH=GH,K=K,mesh=mesh,grad=grad,te_pairs=pairs)


def solve(wing,alpha_deg=5., *, kutta='signed_speed', wall_weight=1.,
          jump_weight=6.,kutta_weight=5.,seed=None, max_nfev=100):
    ops=assemble(wing);A,W,C,K,GH=ops['A'],ops['W'],ops['C'],ops['K'],ops['GH'];ng=W.shape[1];nm=len(A[0]);nk=len(C)
    alpha=np.deg2rad(alpha_deg);U=np.array([np.cos(alpha),0,np.sin(alpha)])
    rhs=-ops['mesh'].centers@U
    scale=np.sqrt(ops['mesh'].areas/np.mean(ops['mesh'].areas))
    # Two independent aerodynamic constraints, with no penalty factors:
    # (i) wake jump is the TE potential discontinuity;
    # (ii) Kutta chooses circulation through TE velocity, not through definition (i).
    J=np.block([C,-GH])
    E=np.block([K,np.zeros((nk,ng))])
    B=wall_weight*scale[:,None]*np.column_stack([A,W])
    bw=wall_weight*scale*rhs
    def exact_constrained_solve(constraints):
        N=null_space(constraints, rcond=1e-11)
        if N.shape[1]==0: raise ValueError('No degrees of freedom after Kutta constraints')
        t,_,rank,sv=lstsq(B@N,bw,lapack_driver='gelsd')
        return N@t,rank,sv,N
    initial,rank,sv,N_lin=exact_constrained_solve(np.vstack((J,E)))
    mu_gamma=initial.copy();mu,gamma=mu_gamma[:nm],mu_gamma[nm:]
    def pressure_residual(vec):
        mu_=vec[:nm];vel=np.einsum('fij,j->fi',ops['grad'],mu_)
        return np.array([np.mean(np.sum(vel[top]**2,axis=1))-np.mean(np.sum(vel[bot]**2,axis=1))
                         for top,bot in ops['te_pairs']])
    def pressure_jac(vec):
        mu_=vec[:nm];vel=np.einsum('fij,j->fi',ops['grad'],mu_)
        mat=np.zeros((nk,nm))
        for i,(top,bot) in enumerate(ops['te_pairs']):
            mat[i]=2*np.mean(np.einsum('fi,fij->fj',vel[top],ops['grad'][top]),axis=0) - 2*np.mean(np.einsum('fi,fij->fj',vel[bot],ops['grad'][bot]),axis=0)
        return np.c_[mat,np.zeros((nk,ng))]
    if kutta=='pressure':
        N=null_space(J,rcond=1e-11)
        # Jump equality is exact in this parameterisation for every iterate.
        def f(t):
            v=N@t
            return np.r_[B@v-bw,kutta_weight*pressure_residual(v)]
        def jac(t):
            v=N@t
            return np.vstack([B,kutta_weight*pressure_jac(v)])@N
        z0=N.T@initial
        result=least_squares(f,z0,jac=jac,max_nfev=max_nfev,
                             ftol=1e-11,xtol=1e-11,gtol=1e-11)
        mu_gamma=N@result.x;mu,gamma=mu_gamma[:nm],mu_gamma[nm:]
    elif kutta!='signed_speed':raise ValueError('kutta must be signed_speed or pressure')
    vel=np.einsum('fij,j->fi',ops['grad'],mu)
    cp=1-np.sum(vel**2,axis=1)
    force=-np.einsum('f,f,fi->i',cp,ops['mesh'].areas,ops['mesh'].normals)
    eh=np.array([-np.sin(alpha),0,np.cos(alpha)]);ed=U
    CL=float(force@eh/wing.reference_area); CD=float(force@ed/wing.reference_area)
    gfull=np.r_[0,gamma,0];CLg=float(2*np.trapezoid(gfull,wing.gamma_y)/wing.reference_area)
    moment=np.sum(np.cross(ops['mesh'].centers-wing.reference_point,
                           -(cp*ops['mesh'].areas)[:,None]*ops['mesh'].normals),axis=0)
    CM_y=float(moment[1]/(wing.reference_area*wing.chord))
    return dict(CL=CL,CD_pressure=CD,CL_gamma=CLg,CM_y=CM_y,force=force,cp=cp,mu=mu,gamma=gamma,
                constrained_rank=int(rank),singular=sv,
                pressure_kutta_rms=float(np.sqrt(np.mean(pressure_residual(mu_gamma)**2))) if kutta=='pressure' else None,
                signed_kutta_rms=float(np.sqrt(np.mean((K@mu)**2))),jump_rms=float(np.sqrt(np.mean((J@mu_gamma)**2))),
                wall_rms=float(np.sqrt(np.mean((A@mu+W@gamma-rhs)**2))),
                alpha_deg=alpha_deg,method=kutta,ops=ops,
                diagnostics=dict(n_wall_faces=len(wing.faces),n_wall_dofs=nm,
                                 n_gamma_dofs=ng,n_kutta_rows=nk,
                                 cp_min=float(cp.min()),cp_max=float(cp.max()),
                                 reference_area=wing.reference_area,
                                 source=wing.section,physical_accuracy_accepted=False))


def alpha_sensitivity(wing, alpha_deg=5.):
    """Analytic d(CL, CL_gamma, CM_y)/d(alpha_rad) for linear signed-speed Kutta.

    Fixed geometry, fixed wake, no finite-difference step or autodiff magic.
    Shape VJP and nonlinear pressure-Kutta implicit differentiation are separate.
    """
    ops=assemble(wing)
    A,W,C,K,GH=ops['A'],ops['W'],ops['C'],ops['K'],ops['GH']
    ng=W.shape[1];nm=A.shape[1]
    J=np.c_[C,-GH];E=np.c_[K,np.zeros((len(K),ng))]
    N=null_space(np.vstack([J,E]),rcond=1e-11)
    weights=np.sqrt(ops['mesh'].areas/np.mean(ops['mesh'].areas))
    B=weights[:,None]*np.c_[A,W]
    rad=np.deg2rad(alpha_deg)
    U=np.array([np.cos(rad),0.,np.sin(rad)])
    dU=np.array([-np.sin(rad),0.,np.cos(rad)])
    RHS=-weights[:,None]*ops['mesh'].centers
    # Same QR geometry/operator for both alpha RHS and its exact derivative.
    BN=B@N
    T,_,_,_=lstsq(BN,np.c_[RHS@U,RHS@dU],lapack_driver='gelsd')
    X=N@T
    mu,gamma=X[:nm,0],X[nm:,0]
    dmu,dgamma=X[:nm,1],X[nm:,1]
    vel=np.einsum('fij,j->fi',ops['grad'],mu)
    dvel=np.einsum('fij,j->fi',ops['grad'],dmu)
    cp=1-np.sum(vel**2,axis=1)
    dcp=-2*np.sum(vel*dvel,axis=1)
    force=-np.einsum('f,f,fi->i',cp,ops['mesh'].areas,ops['mesh'].normals)
    dforce=-np.einsum('f,f,fi->i',dcp,ops['mesh'].areas,ops['mesh'].normals)
    el=np.array([-np.sin(rad),0,np.cos(rad)])
    del_=np.array([-np.cos(rad),0,-np.sin(rad)])
    dcl=float((dforce@el+force@del_)/wing.reference_area)
    dg=np.r_[0,dgamma,0]
    dcl_gamma=float(2*np.trapezoid(dg,wing.gamma_y)/wing.reference_area)
    dmom=np.sum(np.cross(ops['mesh'].centers-wing.reference_point,
                          -(dcp*ops['mesh'].areas)[:,None]*ops['mesh'].normals),axis=0)
    return dict(alpha_deg=alpha_deg,dCL_dalpha_rad=dcl,
                dCL_gamma_dalpha_rad=dcl_gamma,
                dCM_y_dalpha_rad=float(dmom[1]/(wing.reference_area*wing.chord)),
                meaning='fixed-geometry exact linear-system sensitivity',
                physical_accuracy_accepted=False)
