"""Small differentiable counterpart of p1_closed_body (fixed legal topology).

Call the NumPy prepare_mesh/solve rank audit BEFORE differentiating. This core
must not infer connectivity or repair geometry inside an AD trace. No
rank truncation or gauge is introduced. Dense reference budgets are explicit.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def _geometry(vertices, faces):
    tri=vertices[faces]
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    av=jnp.cross(e1,e2);area2=jnp.linalg.norm(av,axis=1)
    n=av/area2[:,None]
    g1=jnp.cross(e2,n)/area2[:,None];g2=jnp.cross(n,e1)/area2[:,None]
    G=jnp.stack((-g1-g2,g1,g2),axis=1)
    return tri,n,area2/2,G


def _influence(points,tri,n,G,self_faces=None):
    r=tri[None]-points[:,None,None,:];radius=jnp.linalg.norm(r,axis=-1)
    r0,r1,r2=r[:,:,0],r[:,:,1],r[:,:,2]
    dot=lambda a,b:jnp.einsum('bfi,bfi->bf',a,b)
    det=dot(r0,jnp.cross(r1,r2))
    den=(radius.prod(axis=2)+dot(r0,r1)*radius[:,:,2]
         +dot(r1,r2)*radius[:,:,0]+dot(r2,r0)*radius[:,:,1])
    if self_faces is not None:
        diag=self_faces[:,None]==jnp.arange(len(tri))[None]
        det=jnp.where(diag,0.,det);den=jnp.where(diag,1.,den)
    omega=-2*jnp.arctan2(det,den)
    D=omega/(4*jnp.pi)
    h=-jnp.einsum('bfi,fi->bf',r0,n)
    if self_faces is not None:
        D=jnp.where(diag,-.5,D);h=jnp.where(diag,0.,h)
    edge_sum=jnp.zeros_like(h);grad=jnp.zeros((*h.shape,3),dtype=points.dtype)
    for k,l in ((0,1),(1,2),(2,0)):
        edge=tri[:,l]-tri[:,k];length=jnp.linalg.norm(edge,axis=1)
        out=jnp.cross(edge/length[:,None],n)
        rk,rl=r[:,:,k],r[:,:,l]
        d=dot(rk,rl);prod=radius[:,:,k]*radius[:,:,l];opposite=d<0
        cr=jnp.cross(rk,rl)
        plus=jnp.where(opposite,dot(cr,cr)/jnp.where(opposite,prod-d,1.),prod+d)
        gap=2*plus/(radius[:,:,k]+radius[:,:,l]+length[None])
        log=jnp.log1p(2*length[None]/gap)
        edge_sum+=jnp.einsum('bfi,fi->bf',rk,out)*log
        grad-=out[None]*log[:,:,None]
    S=(edge_sum-h*omega)/(4*jnp.pi)
    J=(points[:,None]-tri.mean(axis=1)[None])*D[:,:,None]+h[:,:,None]*grad/(4*jnp.pi)
    D1=D[:,:,None]/3+jnp.einsum('fai,bfi->bfa',G,J)
    return S,D,D1


def solve_nodal(vertices, faces, freestream, *, samples_per_face=1):
    """AD core for a prevalidated closed convex G0 mesh, <=320 faces.

    No call to an analytic sphere solution occurs here. Nodal mu is solved
    from freestream data. Geometry and freestream VJPs differentiate the full
    source/doublet assembly, area weighting, QR solve and P1 surface gradient.
    """
    if not jax.config.x64_enabled:
        raise ValueError('Explicit JAX float64 required')
    if len(faces)>320:
        raise MemoryError('Small differentiable reference limited to 320 faces')
    v=jnp.asarray(vertices,dtype=jnp.float64);f=jnp.asarray(faces,dtype=jnp.int32)
    u=jnp.asarray(freestream,dtype=jnp.float64)
    tri,n,area,G=_geometry(v,f)
    if samples_per_face==1:
        bary=jnp.array([[1/3,1/3,1/3]],dtype=v.dtype)
    elif samples_per_face==3:
        bary=jnp.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],dtype=v.dtype)
    else:raise ValueError('samples_per_face must be 1 or 3')
    x=jnp.einsum('qa,fai->fqi',bary,tri).reshape(-1,3)
    ids=jnp.repeat(jnp.arange(len(f)),len(bary))
    S,D,D1=_influence(x,tri,n,G,ids)
    A=jnp.zeros((len(x),len(v)),dtype=v.dtype)
    for a in range(3):A=A.at[:,f[:,a]].add(D1[:,:,a])
    sigma=-n@u;rhs=S@sigma
    weights=jnp.sqrt(jnp.repeat(area/len(bary),len(bary))/area.sum())
    Q,R=jnp.linalg.qr(A*weights[:,None],mode='reduced')
    mu=jnp.linalg.solve(R,Q.T@(weights*rhs))
    grad=jnp.einsum('fa,fai->fi',mu[f],G)
    velocity=u-n*(n@u)[:,None]+grad
    cp=1-jnp.sum(velocity**2,axis=1)/jnp.dot(u,u)
    return dict(mu=mu,sigma=sigma,cp=cp,velocity=velocity,A=A,rhs=rhs,areas=area)


def make_velocity_evaluator(state):
    """JIT/vmap spatial AD of the solved exterior potential, P0 or P1 state.

    Uses the same source/doublet definitions but *not* tangent projection.
    Comparing its actual normal component tests the wall boundary condition.
    """
    if not jax.config.x64_enabled:raise ValueError('Explicit JAX float64 required')
    mesh=state['mesh']
    v=jnp.asarray(mesh.vertices);f=jnp.asarray(mesh.faces)
    tri,n,area,G=_geometry(v,f)
    mu=jnp.asarray(state['mu']);sigma=jnp.asarray(state['sigma'])
    u=jnp.asarray(state['freestream']);order=state['order']
    def total_potential(point):
        S,D,D1=_influence(point[None],tri,n,G)
        double=(D@mu if order==0 else jnp.einsum('bfa,fa->b',D1,mu[f]))
        return (point@u+double-S@sigma).reshape(())
    return jax.jit(jax.vmap(jax.grad(total_potential)))
