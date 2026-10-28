"""Mathematical and analytic-kernel checks; no CAD or production-state solve."""
import json
from pathlib import Path
import numpy as np
import pytest
from panel.gamma_refinement import (
    refine_space, basis_at, split_wake, wake_basis, weak_te,
    wake_operator, nested_checks, solve_augmented,
)


def rectangle():
    return np.array([[[1.,0.,0.],[3.,0.,0.],[3.,1.,0.]],
                     [[1.,0.,0.],[3.,1.,0.],[1.,1.,0.]]])


def te_data():
    return np.array([[[1.,0.,0.],[1.,.3,0.]], [[1.,.3,0.],[1.,1.,0.]]]), [0,2], [1,3]


@pytest.mark.parametrize('factor', [1,2,4])
def test_nested_contact_space(factor):
    y, d = np.array([0.,.1,.2,.6,1.]), np.array([0,0,0,1,-1])
    old, fine = refine_space(y,d), refine_space(y,d,factor)
    assert fine.hat.shape[1] == 2+2*(factor-1)
    sample = np.linspace(0,1,131)
    np.testing.assert_allclose(basis_at(sample, fine)@fine.prolongation,
                               basis_at(sample, old), atol=2e-15)
    np.testing.assert_array_equal(fine.dofs[fine.stations <= .2], 0)
    np.testing.assert_array_equal(fine.hat[-1], 0.)
    assert fine.dofs[0] == 0
    np.testing.assert_array_equal(d, [0,0,0,1,-1])


@pytest.mark.parametrize('factor', [1,2,4])
def test_partition_conserves_each_parent_and_centroid(factor):
    s = refine_space([0.,.3,1.],[0,1,-1],factor)
    tri = rectangle(); original = tri.copy()
    sub, parent, interval, audit = split_wake(tri,s.stations)
    assert audit['area_relative_max'] < 1e-14
    assert audit['first_moment_scaled_max'] < 1e-14
    assert np.all(sub[:,:,1].min(axis=1) >= s.stations[interval])
    assert np.all(sub[:,:,1].max(axis=1) <= s.stations[interval+1])
    assert set(parent) == {0,1}
    np.testing.assert_array_equal(tri, original)


def test_exact_p1_mass_and_reversed_edge():
    s = refine_space([0.,.5,1.],[0,1,-1])
    edge = np.array([[[1.,0.,0.],[1.,1.,0.]]])
    t = weak_te(edge,[0],[1],2,s)
    np.testing.assert_allclose(t['M'], [[1/6,1/12],[1/12,1/3]], atol=1e-15)
    np.testing.assert_allclose(t['measure'], [.25,.5], atol=1e-15)
    backwards = weak_te(edge[:,::-1],[0],[1],2,s)
    np.testing.assert_allclose(t['K'], backwards['K'], atol=1e-15)
    np.testing.assert_allclose(t['M'], backwards['M'], atol=1e-15)


@pytest.mark.parametrize('factor',[2,4])
def test_galerkin_mass_and_jump_exact_with_contact(factor):
    y,d=[0.,.1,.2,.6,1.],[0,0,0,1,-1]
    old,fine=refine_space(y,d),refine_space(y,d,factor)
    edges=np.array([[[1.,.2,0.],[1.2,.6,0.]],[[1.2,.6,0.],[1.4,1.,0.]]])
    a,b=weak_te(edges,[0,2],[1,3],4,old),weak_te(edges,[0,2],[1,3],4,fine)
    p=fine.prolongation
    np.testing.assert_allclose(a['M'],p.T@b['M']@p,atol=1e-15)
    np.testing.assert_allclose(a['K'],p.T@b['K'],atol=1e-15)
    assert np.linalg.eigvalsh(b['M']).min()>0


def test_no_exposed_te_support_is_rejected():
    s=refine_space([0.,.3,.6,1.],[0,1,2,-1],4)
    with pytest.raises(ValueError,match='support'):
        weak_te(np.array([[[1.,.61,0.],[1.,1.,0.]]]),[0],[1],2,s)


def test_geometry_scale_translation_and_moment():
    tri=rectangle(); s=refine_space([0.,.3,1.],[0,1,-1],4)
    a,_,_,_=split_wake(tri,s.stations)
    shifted=tri*100+np.array([1000.,0.,12.])
    b,_,_,_=split_wake(shifted,100*s.stations)
    np.testing.assert_allclose(a*100+[1000,0,12],b,atol=1e-12)


@pytest.mark.parametrize('bad', [0,-2,True,1.5])
def test_invalid_factor(bad):
    with pytest.raises(ValueError): refine_space([0.,1.],[0,-1],bad)


def test_invalid_contact_and_wake():
    with pytest.raises(ValueError): refine_space([0.,.2,.4,.6,1.],[0,1,0,2,-1])
    with pytest.raises(ValueError): refine_space([0.,.3,1.],[0,-1,-1])
    with pytest.raises(ValueError): split_wake(rectangle()[:,::-1], [0.,1.])
    with pytest.raises(ValueError): split_wake(rectangle(), [0.,.9])
    with pytest.raises(MemoryError): split_wake(rectangle(), np.linspace(0,1,30),max_pieces=2)


def test_linear_wake_field_integrals_not_just_midpoint_resampling():
    s=refine_space([0.,.3,1.],[0,1,-1],4)
    p,_,i,_=split_wake(rectangle(),s.stations)
    b,sl= wake_basis(p,i,s)
    gamma=np.arange(s.hat.shape[1],dtype=float)
    vertices=p[:,:,1]
    reconstructed=(b@gamma)[:,None]+(vertices-p.mean(axis=1)[:,None,1])*(sl@gamma)[:,None]
    np.testing.assert_allclose(reconstructed,basis_at(vertices,s)@gamma,atol=1e-14)


def test_dense_augmented_solve_no_regularization():
    e,up,lo=te_data();s=refine_space([0.,.3,1.],[0,1,-1],2)
    t=weak_te(e,up,lo,4,s)
    D=np.array([[-1.,.1,0,0],[.2,-1,0,0],[0,.1,-1,.2],[.1,0,0,-1.]])
    W=np.arange(4*s.hat.shape[1]).reshape(4,-1)*.0001
    mu,g,r=solve_augmented(D,np.arange(4.)+.1,W,t)
    np.testing.assert_allclose(D@mu+W@g,np.arange(4.)+.1,atol=1e-14)
    np.testing.assert_allclose(t['K']@mu,t['M']@g,atol=1e-14)
    assert r['true_relative_residual']<1e-14


def test_cli_rejects_resource_and_overwrite_before_backend_import(tmp_path):
    from panel.run_gamma_refinement import run
    p=tmp_path/'input.json'
    p.write_text(json.dumps({'schema':'dpanel.half_planar_cut.v1','faces':[[0,1,2]]*3}))
    with pytest.raises(MemoryError): run(p,tmp_path/'new',max_half_panels=2)
    with pytest.raises(FileExistsError): run(p,tmp_path)
    assert not (tmp_path/'new').exists()


def _jax():
    jax=pytest.importorskip('jax')
    jax.config.update('jax_enable_x64',True)
    return jax


@pytest.mark.parametrize('factor',[2,4])
def test_actual_kernel_old_space_embedding_and_mirror(factor):
    _jax()
    old=refine_space([0.,.3,1.],[0,1,-1])
    new=refine_space([0.,.3,1.],[0,1,-1],factor)
    points=np.array([[.1,.1,.2],[.2,.8,-.3],[4.,.3,.2],[1.5,1.2,.3]])
    a,_,ai,_=split_wake(rectangle(),old.stations)
    b,_,bi,_=split_wake(rectangle(),new.stations)
    Wa=wake_operator(points,a,ai,old,block=2)
    Wb=wake_operator(points,b,bi,new,block=3)
    e,up,lo=te_data()
    ta,tb=weak_te(e,up,lo,4,old),weak_te(e,up,lo,4,new)
    result=nested_checks(Wa,ta,Wb,tb,new.prolongation)
    assert result['wake_action']['max_abs']<2e-14
    mirror=wake_operator(points*[1.,-1.,1.],b,bi,new,block=2)
    np.testing.assert_allclose(Wb,mirror,atol=2e-14)


def test_analytic_wake_against_independent_area_quadrature():
    _jax()
    s=refine_space([0.,.3,1.],[0,1,-1],2)
    tri,_,i,_=split_wake(rectangle(),s.stations)
    targets=np.array([[.1,.2,.7],[3.5,1.4,-.8]])
    W=wake_operator(targets,tri,i,s,block=2)
    # Independent integration of n.(target-source)/(4*pi*R^3).
    v,w=np.polynomial.legendre.leggauss(32);v=(v+1)/2;w=w/2
    u,vv=np.meshgrid(v,v,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij')
    integral=np.zeros_like(W)
    for face in tri:
        q=face[0]+u[...,None]*(face[1]-face[0])+(1-u[...,None])*vv[...,None]*(face[2]-face[0])
        weight=(wu*wv*(1-u)*np.linalg.norm(np.cross(face[1]-face[0],face[2]-face[0]))).ravel()
        pts=q.reshape(-1,3);hat=basis_at(pts[:,1],s)
        for reflect in (1.,-1.):
            image=pts*np.array([1.,reflect,1.])
            r=targets[:,None]-image
            d=r[:,:,2]/(4*np.pi*np.linalg.norm(r,axis=2)**3)
            integral+=(d*weight)@hat
    np.testing.assert_allclose(W,integral,rtol=1e-11,atol=1e-13)


def test_kernel_resource_guard():
    _jax()
    s=refine_space([0.,1.],[0,-1],2)
    p,_,i,_=split_wake(rectangle(),s.stations)
    with pytest.raises(MemoryError):
        wake_operator(np.ones((10,3)),p,i,s,max_pairs=1)
