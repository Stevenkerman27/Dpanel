"""Read-only 3D affine gradient fit on the FROZEN cut-aware face stencil.

Unsupported curved/rank-poor patches are masked, never pseudo-inverted.
"""
import numpy as np


def recover_tangent_gradient(centers, normals, mu, neighbor_ids, *,
        min_ratio_3d=1e-4,min_ratio_2d=1e-6,planar_tol=1e-8,
        max_normal_angle_deg=45.,strict=False):
    x,n,z,ids=map(np.asarray,(centers,normals,mu,neighbor_ids))
    if (x.ndim!=2 or x.shape[1]!=3 or n.shape!=x.shape
        or z.shape!=(len(x),) or ids.ndim!=2 or not len(ids)
        or ids.shape[1]<3 or len(ids)>len(x)
        or not np.issubdtype(ids.dtype,np.integer)
        or ids.min()<0 or ids.max()>=len(x)
        or not all(np.isfinite(a).all() for a in (x,n,z))
        or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-9)
        or not (0<min_ratio_3d<1 and 0<min_ratio_2d<1
                and 0<planar_tol<1 and 0<max_normal_angle_deg<180)):
        raise ValueError('Invalid frozen affine inputs')
    grad=np.full((len(ids),3),np.nan);audit=[]
    for i,js in enumerate(ids):
        N=n[i];axis=np.eye(3)[np.argmin(abs(N))]
        t=np.cross(N,axis);t/=np.linalg.norm(t)
        s=np.cross(N,t);d=x[js]-x[i]
        h=np.linalg.norm(d,axis=1).max()
        ang=np.degrees(np.arccos(np.clip(n[js]@N,-1,1))).max()
        item=dict(face=i,mode='unsupported',ratio3=0.,
                  max_normal_angle_deg=float(ang))
        if h<=0 or ang>max_normal_angle_deg:
            audit.append(item);continue
        A=np.stack((d@t,d@s,d@N),axis=1)/h
        sv=np.linalg.svd(A,compute_uv=False)
        ratio=float(sv[-1]/sv[0]) if sv[0] else 0.
        item['ratio3']=ratio
        if ratio>=min_ratio_3d:
            C=A;item['mode']='ambient_3d_qr'
        elif (np.max(np.linalg.norm(n[js]-N,axis=1))<=planar_tol
              and np.max(abs(d@N))/h<=planar_tol):
            C=A[:,:2]
            sv=np.linalg.svd(C,compute_uv=False)
            if not sv[0] or sv[-1]/sv[0]<min_ratio_2d:
                audit.append(item);continue
            item['mode']='verified_planar_2d_qr'
        else:
            audit.append(item);continue
        Q,R=np.linalg.qr(C,mode='reduced')
        coeff=np.linalg.solve(R,Q.T@(z[js]-z[i]))/h
        grad[i]=coeff[0]*t+coeff[1]*s
        audit.append(item)
    good=np.isfinite(grad).all(axis=1)
    if strict and not good.all():
        raise ValueError(f'Unsupported curved/rank-poor faces: {np.where(~good)[0].tolist()}')
    return dict(gradient=grad,valid=good,rows=audit,
        count_3d=sum(v['mode']=='ambient_3d_qr' for v in audit),
        count_planar=sum(v['mode']=='verified_planar_2d_qr' for v in audit),
        count_unsupported=int((~good).sum()),all_valid=bool(good.all()))
