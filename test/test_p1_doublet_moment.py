import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from panel.p1_doublet_moment import triangle_p1_doublet_potentials as kernel


def tri():return np.array([[.1,-.2,.3],[1.3,-.1,.1],[.2,1.,.5]])


def duffy(x,T,order=44):
    g,w=leggauss(order);g=(1+g)/2;w=w/2
    e1,e2=T[1]-T[0],T[2]-T[0]
    area=np.linalg.norm(np.cross(e1,e2))
    n=np.cross(e1,e2)/area;out=np.zeros(3)
    for u,wu in zip(g,w):
        for s,ws in zip(g,w):
            v=(1-u)*s;r=x-T[0]-u*e1-v*e2
            out+=wu*ws*area*(1-u)*(n@r)/(4*np.pi*np.linalg.norm(r)**3)*np.array([1-u-v,u,v])
    return out


def test_independent_quadrature_and_partition():
    T=tri();x=np.array([.2,.13,.9]);z=kernel(x[None],T[None])[0,0]
    q=duffy(x,T)
    assert np.max(abs(z-q))<1e-9
    assert abs(z.sum()-q.sum())<1e-9


def test_rigid_rotation_translation():
    T=tri();x=np.array([.43,.32,1.2]);u=kernel(x[None],T[None])[0,0]
    axis=np.array([1.2,-.8,.6]);axis/=np.linalg.norm(axis)
    K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    R=np.eye(3)+np.sin(.9)*K+(1-np.cos(.9))*(K@K);t=[4.2,-1.3,2.1]
    v=kernel((x@R.T+t)[None],(T@R.T+t)[None])[0,0]
    assert np.allclose(u,v,rtol=1e-10,atol=1e-12)


def test_reversed_source_winding():
    T=tri();x=np.array([.2,.13,.9]);u=kernel(x[None],T[None])[0,0]
    v=kernel(x[None],T[[0,2,1]][None])[0,0]
    assert np.allclose(u,-v[[0,2,1]],rtol=1e-12,atol=1e-13)


def test_interior_self_trace_and_centroid():
    T=tri();B=np.array([.2,.3,.5]);x=B@T
    assert np.allclose(kernel(x[None],T[None],self_ids=[0])[0,0],-.5*B)
    assert np.allclose(kernel(T.mean(axis=0)[None],T[None],self_ids=[0])[0,0],[-1/6]*3)


def test_multiple_rows_against_independent_duffy():
    T=tri();P=np.array([[.4,.2,.8],[.4,.1,1.3]])
    A=np.stack([T,T+[1.4,0,0]])
    D=kernel(P,A);assert D.shape==(2,2,3)
    for i in range(2):
        for j in range(2):assert np.max(abs(D[i,j]-duffy(P[i],A[j])))<1e-8


@pytest.mark.parametrize('bad',['nan','shape','zero','badid','badself','budget'])
def test_reject_invalid_geometry_or_budget(bad):
    T=tri();x=T.mean(axis=0)+[0.,0.,.8];kw={}
    if bad=='nan':x[0]=np.nan
    if bad=='shape':T=T[:2]
    if bad=='zero':T[2]=T[1]
    if bad=='badid':kw['self_ids']=[1]
    if bad=='badself':kw['self_ids']=[0]
    if bad=='budget':kw['max_pairs']=0
    with pytest.raises((ValueError,MemoryError)):
        kernel(x[None],T[None],**kw)
