"""Independent 4-D-ready source-moment oracle for DPanel's existing P1 kernel.

This tests the ACTUAL production G1 function, using an independently
Jacobian-weighted Duffy area integral. It does not test physical lift
convergence or infer wake/Kutta correctness.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from panel.p1_lifting_g1 import _doublet_matrix


def independent_duffy_p1(tri, x, order=90):
    """Duffy y=a+u(b-a)+(1-u)v(c-a), Jacobian 2A(1-u)."""
    q,w=leggauss(order)
    q=(q+1)/2
    w=w/2
    u,v=np.meshgrid(q,q,indexing="ij")
    wt=w[:,None]*w[None,:]*(1-u)
    N=np.stack(((1-u)*(1-v),u,(1-u)*v),axis=-1)
    y=N@tri
    r=x-y
    double_area_normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    kernel=(r@double_area_normal)/(4*np.pi*np.linalg.norm(r,axis=-1)**3)
    return np.einsum("uv,uv,uvk->k",wt,kernel,N)


def test_actual_P1_doublet_three_linear_coefficients_match_Duffy():
    tri=np.array([[0.,0.,0.],[1.3,.1,.13],[.2,1.1,-.03]])
    faces=np.arange(3)[None]
    for p in (np.array([.25,.12,.8]),np.array([.6,-.3,1.7]),
              np.array([.7,.4,-.6])):
        got=_doublet_matrix(tri[None],p[None],face_dofs=faces,ndof=3)[0]
        oracle=independent_duffy_p1(tri,p,order=90)
        np.testing.assert_allclose(got,oracle,atol=2e-12,rtol=2e-11)


def test_actual_P1_column_sum_agrees_with_constant_doublet():
    tri=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    p=np.array([.24,.31,.3])
    actual=_doublet_matrix(tri[None],p[None],face_dofs=np.arange(3)[None],
                           ndof=3)[0]
    independent=independent_duffy_p1(tri,p,order=120)
    assert abs(actual.sum()-independent.sum())<1e-12


def test_actual_P1_self_face_trace_has_full_linear_barycentric_part():
    tri=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bary=np.array([.2,.3,.5])
    p=bary@tri
    actual=_doublet_matrix(tri[None],p[None],self_faces=np.array([0]),
                           face_dofs=np.arange(3)[None],ndof=3)[0]
    np.testing.assert_allclose(actual,-.5*bary,atol=1e-14,rtol=1e-14)
