"""Independent velocity oracles for the exact P1 edge-integration research kernel."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from panel.p1_analytic_sheet_velocity import AnalyticSheetField, segment_moments


def area_gradient(point, tri, values, order=60):
    """Independent Duffy area integral of the P1 doublet potential gradient."""
    z, w = leggauss(order)
    z, w = (z+1)/2, w/2
    u, v = np.meshgrid(z, z, indexing="ij")
    weights = w[:, None]*w[None, :]*(1-u)
    bary = np.stack(((1-u)*(1-v), u, (1-u)*v), axis=-1)
    source, mu = bary@tri, bary@values
    r = point-source
    rn = np.linalg.norm(r, axis=-1)
    av = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    integrand = av/rn[..., None]**3 - 3*(r@av)[..., None]*r/rn[..., None]**5
    return np.sum((weights*mu)[..., None]*integrand, axis=(0, 1))/(4*np.pi)


def test_segment_moments_vs_independent_adaptive_scipy():
    p = np.array([.42, .08, .03])
    a = np.array([[0.,0.,0.],[-4.,0.,0.],[3.,2.,0.]])
    b = np.array([[1.,0.,0.],[-3.,0.,0.],[3.,3.,0.]])
    i1, i3, il3, t, _rho = segment_moments(p,a,b)
    for j in range(3):
        L = np.linalg.norm(b[j]-a[j])
        def radius(l):
            return np.linalg.norm(p-a[j]-l*t[j])
        expected = [
            quad(lambda l:1/radius(l),0,L,epsabs=1e-12)[0],
            quad(lambda l:1/radius(l)**3,0,L,epsabs=1e-12)[0],
            quad(lambda l:l/radius(l)**3,0,L,epsabs=1e-12)[0],
        ]
        np.testing.assert_allclose([i1[j],i3[j],il3[j]],expected,rtol=2e-11,atol=2e-11)


def test_collinear_outside_segment_is_finite_but_on_segment_is_refused():
    import pytest
    a,b=np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]])
    for p in ([-1.,0.,0.],[2.,0.,0.]):
        i1,i3,il3,*_=segment_moments(np.array(p),a,b)
        assert i1[0]>0 and i3[0]>0 and il3[0]>0
    with pytest.raises(FloatingPointError):
        segment_moments(np.array([.5,0.,0.]),a,b)


def test_linear_P1_velocity_vs_independent_area_oracle():
    tri=np.array([[0.,0.,0.],[1.,.1,.2],[.2,.9,-.1]])
    mu=np.array([.2,-.1,.4])
    shift=np.array([1.4,-.1,.3])
    gamma=np.array([.3,-.2,.4])
    field=AnalyticSheetField([
        (tri,np.array([[0,1,2]]),mu[None,:]),
        (tri+shift,np.array([[0,1,2]]),gamma[None,:])
    ])
    for p in [np.array([.3,-.4,1.7]),np.array([-.1,.2,-1.])]:
        expected=np.array([1.,0.,0.])+area_gradient(p,tri,mu)+area_gradient(p,tri+shift,gamma)
        np.testing.assert_allclose(field(p),expected,rtol=1e-12,atol=1e-12)


def test_shared_edge_P1_traces_cancel_but_physical_cut_does_not():
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    f=np.array([[0,1,2],[0,2,3]])
    values=np.array([.5,.6,.8,.7])[f]
    merged=AnalyticSheetField([(xyz,f,values)])
    separate=AnalyticSheetField([(xyz,f[:1],values[:1]),(xyz,f[1:],values[1:])])
    assert merged.n_boundary_edges==4
    assert separate.n_boundary_edges==6
    for p in [np.array([.2,.3,1.]),np.array([.2,.3,-1.])]:
        np.testing.assert_allclose(merged(p),separate(p),rtol=1e-12,atol=1e-12)
    cut_values=values.copy()
    cut_values[1,0]+=.1
    cut=AnalyticSheetField([(xyz,f,cut_values)])
    assert cut.n_boundary_edges==5


def test_aspect_260_near_sheet_remains_finite_without_core_or_quad():
    tri=np.array([[0.,0.,0.],[26.,0.,0.],[26.,.1,0.]])
    f=np.array([[0,1,2]])
    values=np.array([[.1,.2,.3]])
    p=tri.mean(axis=0)+np.array([0.,0.,.000625])
    field=AnalyticSheetField([(tri,f,values)],freestream=(0.,0.,0.))
    bigger=AnalyticSheetField([(tri*7,f,values)],freestream=(0.,0.,0.))
    assert np.isfinite(field(p)).all()
    np.testing.assert_allclose(bigger(p*7),field(p)/7,rtol=4e-11,atol=4e-11)


def test_winding_and_linear_strength_equivariance():
    tri=np.array([[0.,0.,0.],[1.,.2,0.],[.2,1.,0.]])
    mu=np.array([[.3,.6,-.1]])
    p=np.array([.5,.2,.3])
    original=AnalyticSheetField([(tri,np.array([[0,1,2]]),mu)],freestream=(0,0,0))
    double=AnalyticSheetField([(tri,np.array([[0,1,2]]),2*mu)],freestream=(0,0,0))
    reversed_sheet=AnalyticSheetField([(tri,np.array([[0,2,1]]),mu[:,[0,2,1]])],freestream=(0,0,0))
    np.testing.assert_allclose(double(p),2*original(p),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(reversed_sheet(p),-original(p),rtol=1e-13,atol=1e-13)
