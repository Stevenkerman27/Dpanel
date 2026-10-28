import numpy as np
import pytest
from panel.affine_consistent_recovery import recover_tangent_gradient as fit


def cloud():
    points=np.array([[0,0,0],[1,0,.1],[0,1,.2],[1,1,.4],[-1,0,.3],[0,-1,-.1]],float)
    n=np.tile([0,0,1.],(len(points),1))
    n[1]=[.1,0,.99498743710662];n[2]=[0,.1,.99498743710662]
    n[3]=[.1,.1,.98994949366117];n[3]/=np.linalg.norm(n[3])
    n[4]=[-.1,0,.99498743710662];n[5]=[0,-.1,.99498743710662]
    return points,n,np.array([[0,1,2,3,4,5]],int)


def test_curved_affine_ambient_potential_recovered_exactly():
    x,n,ids=cloud();a=np.array([.2,-.4,.7]);mu=123.456 + x@a
    out=fit(x,n,mu,ids,strict=True)
    assert out['count_3d']==1 and out['all_valid']
    assert np.allclose(out['gradient'][0],a-n[0]*(n[0]@a),atol=1e-12)


def test_planar_affine_ambient_recovered_by_identifiable_tangent():
    x,n,ids=cloud();x[:,2]=0;n[:]=[0,0,1]
    a=np.array([.2,-.4,.7]);mu=x@a + 1000
    out=fit(x,n,mu,ids,strict=True)
    assert out['count_planar']==1 and out['count_3d']==0
    assert np.allclose(out['gradient'][0],[.2,-.4,0],atol=1e-12)


def test_curved_rank_poor_is_not_silently_2d():
    x,n,ids=cloud();x[:,2]*=1e-8
    mu=x@np.array([.2,-.4,.7])
    out=fit(x,n,mu,ids,min_ratio_3d=.01,planar_tol=1e-10)
    assert not out['all_valid'] and out['count_unsupported']==1
    assert np.isnan(out['gradient'][0]).all()
    with pytest.raises(ValueError,match='Unsupported'):
        fit(x,n,mu,ids,min_ratio_3d=.01,planar_tol=1e-10,strict=True)


def test_gauge_scale_and_mirror():
    x,n,ids=cloud();a=np.array([.2,-.4,.7]);mu=x@a
    orig=fit(x,n,mu,ids,strict=True)['gradient'][0]
    scaled=fit(7*x,n,7*mu+1e4,ids,strict=True)['gradient'][0]
    assert np.allclose(orig,scaled,atol=1e-12)
    reflect=np.diag([1,-1,1]);x2=x@reflect;n2=n@reflect
    mirror=fit(x2,n2,x2@(a@reflect),ids,strict=True)['gradient'][0]
    assert np.allclose(mirror,orig@reflect,atol=1e-12)


def test_invalid_inputs_rejected():
    x,n,ids=cloud();mu=x[:,0]
    for changed in (dict(neighbor_ids=np.array([[9]])),
                    dict(mu=np.full(len(x),np.nan)),
                    dict(normals=n*2),
                    dict(min_ratio_3d=1.1)):
        kwargs=dict(centers=x,normals=n,mu=mu,neighbor_ids=ids)
        kwargs.update(changed)
        with pytest.raises(ValueError):fit(**kwargs)
