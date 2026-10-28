"""Small NumPy oracle for linear-doublet influence on planar triangles.

This is NOT a production solver, pressure recovery or differentiable kernel.
"""
import numpy as np


def triangle_p1_doublet_potentials(targets, triangles, *, self_ids=None,
                                   max_pairs=20000):
    """Exact analytic [target, source triangle, 3 barycentric coefficients]."""
    X,T=np.asarray(targets,float),np.asarray(triangles,float)
    if (X.ndim!=2 or X.shape[1]!=3 or T.ndim!=3 or T.shape[1:]!=(3,3)
            or not len(T) or not np.isfinite(X).all() or not np.isfinite(T).all()):
        raise ValueError('Invalid finite triangle and target arrays')
    if len(X)*len(T)>max_pairs:raise MemoryError('P1 oracle pair budget')
    if self_ids is not None:
        ids=np.asarray(self_ids)
        if (ids.shape!=(len(X),) or not np.issubdtype(ids.dtype,np.integer)
                or np.any(ids<0) or np.any(ids>=len(T))):
            raise ValueError('Invalid self trace IDs')
    result=np.empty((len(X),len(T),3))
    for j,tri in enumerate(T):
        v,w=tri[1]-tri[0],tri[2]-tri[0]
        n=np.cross(v,w);area2=np.linalg.norm(n)
        if area2<=0:raise ValueError('Degenerate triangle')
        n/=area2
        g1=np.cross(w,n)/area2;g2=np.cross(n,v)/area2
        G=np.stack((-g1-g2,g1,g2));center=tri.mean(axis=0)
        for i,x in enumerate(X):
            if self_ids is not None and ids[i]==j:
                B=1/3+G@(x-center)
                if (B.min()<-1e-12 or B.max()>1+1e-12
                        or abs((x-tri[0])@n)>1e-12*np.sqrt(area2)):
                    raise ValueError('Self target not inside triangle plane')
                result[i,j]=-.5*B;continue
            r=tri-x;L=np.linalg.norm(r,axis=1)
            if np.any(L==0):raise ValueError('Vertex trace undefined')
            a,b,c=r
            den=L.prod()+(a@b)*L[2]+(b@c)*L[0]+(c@a)*L[1]
            D=-2*np.arctan2(a@np.cross(b,c),den)/(4*np.pi)
            h=(x-tri[0])@n;edge_grad=np.zeros(3)
            for k,l in ((0,1),(1,2),(2,0)):
                edge=tri[l]-tri[k];length=np.linalg.norm(edge)
                outward=np.cross(edge/length,n)
                dot=r[k]@r[l];prod=L[k]*L[l]
                plus=(np.linalg.norm(np.cross(r[k],r[l]))**2/(prod-dot)
                      if dot<0 else prod+dot)
                gap=2*plus/(L[k]+L[l]+length)
                if gap<=0:raise ValueError('Singular edge trace')
                edge_grad-=outward*np.log1p(2*length/gap)
            J=(x-center)*D+h*edge_grad/(4*np.pi)
            result[i,j]=D/3+G@J
    return result
