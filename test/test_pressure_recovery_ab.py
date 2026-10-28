import numpy as np
import pytest

from panel.pressure_recovery_ab import (
    mesh_geometry, affine_face_gradient, cut_vertex_gradient, surface_cp,
    mirror_half_wall,
)


def grid(n=7, jitter=True):
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    x = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n*n)))
    if jitter:
        inner = (np.abs(x[:, 0]) < 1) & (np.abs(x[:, 1]) < 1)
        x[inner, :2] += np.random.default_rng(19).uniform(-.03, .03, (inner.sum(), 2))
    f = []
    for i in range(n-1):
        for j in range(n-1):
            a = i*n+j
            f.extend(([a, a+n, a+n+1], [a, a+n+1, a+1]))
    return x, np.array(f)


def sphere(level):
    x = np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    f = np.array([[a,b,c] for a in (0,1) for b in (2,3) for c in (4,5)])
    for i, ids in enumerate(f):
        t = x[ids]
        if np.dot(t.mean(0), np.cross(t[1]-t[0], t[2]-t[0])) < 0:
            f[i] = ids[[0,2,1]]
    for _ in range(level):
        points, edges, new = list(x), {}, []
        def midpoint(a,b):
            key = tuple(sorted((a,b)))
            if key not in edges:
                v = (x[a]+x[b])/2
                edges[key] = len(points)
                points.append(v/np.linalg.norm(v))
            return edges[key]
        for a,b,c in f:
            ab, bc, ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
            new.extend(([a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]))
        x, f = np.array(points), np.array(new)
    return x, f


def test_planar_affine_on_irregular_mesh_and_gauge():
    x, f = grid()
    c = x[f].mean(1)
    mu = .21*c[:,0]-.37*c[:,1]+1.4
    g, corners, audits = cut_vertex_gradient(x,f,mu,[])
    np.testing.assert_allclose(g, np.tile([.21,-.37,0.], (len(f),1)), atol=2e-14)
    exact = .21*x[f,0]-.37*x[f,1]+1.4
    np.testing.assert_allclose(corners, exact, atol=2e-14)
    g2, _, _ = cut_vertex_gradient(x,f,mu+1e4,[])
    np.testing.assert_allclose(g2,g,atol=3e-10)
    assert all(a['singular_ratio'] >= 1e-6 for a in audits)


def test_piecewise_constant_and_linear_jump_never_mixes_cut_banks():
    x, f = grid(jitter=False)
    c = x[f].mean(1)
    seam = np.flatnonzero(x[:,0] == 0)
    seam = seam[np.argsort(x[seam,1])]
    cuts = np.column_stack((seam[:-1], seam[1:]))
    sign = np.where(c[:,0] < 0, -1., 1.)
    mu = .2*c[:,0]-.4*c[:,1]+3*sign
    g, corners, audits = cut_vertex_gradient(x,f,mu,cuts)
    np.testing.assert_allclose(g, np.tile([.2,-.4,0.],(len(f),1)),atol=3e-14)
    np.testing.assert_allclose(corners, .2*x[f,0]-.4*x[f,1]+3*sign[:,None],atol=2e-14)
    for a in audits:
        assert len(set(sign[a['support_faces']])) == 1
    # Deliberately omitting the cut must expose the jump contamination.
    wrong, _, _ = cut_vertex_gradient(x,f,mu,[])
    assert np.max(np.abs(wrong-g)) > 1


def test_constant_potential_and_planar_tangent_freestream():
    x,f = grid()
    g,_,_ = cut_vertex_gradient(x,f,np.full(len(f),2.),[])
    normal = mesh_geometry(x,f)[-1]
    np.testing.assert_allclose(g,0.,atol=1e-14)
    np.testing.assert_allclose(surface_cp(g,normal,[1.,0.,0.]),0.,atol=1e-14)


def test_rotation_scale_and_translation_equivariance():
    x,f = grid()
    c = x[f].mean(1)
    mu = c[:,0]**2 + .3*c[:,1] + .1*c[:,0]*c[:,1]
    q,_ = np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))
    q[:,0] *= np.linalg.det(q)
    g,_,_ = cut_vertex_gradient(x,f,mu,[])
    for scale in (.001,1000.):
        xx = scale*(x@q.T)+np.array([2.,-3.,1.])*scale
        gg,_,_ = cut_vertex_gradient(xx,f,scale*mu,[])
        np.testing.assert_allclose(gg,g@q.T,atol=3e-13)
        cp = surface_cp(g,mesh_geometry(x,f)[-1],[1.,.2,.3])
        cc = surface_cp(gg,mesh_geometry(xx,f)[-1],q@np.array([1.,.2,.3]))
        np.testing.assert_allclose(cc,cp,atol=3e-12)


def test_rank_failure_is_not_silently_regularized():
    x = np.array([[0.,0,0],[1,0,0],[0,1,0]])
    with pytest.raises(ValueError,match='rank deficient'):
        cut_vertex_gradient(x, np.array([[0,1,2]]), np.array([1.]), [])


def test_invalid_topology_and_geometry_rejected():
    x,f = grid()
    with pytest.raises(ValueError,match='real wall edge'):
        cut_vertex_gradient(x,f,np.zeros(len(f)),[[0,len(x)-1]])
    with pytest.raises(ValueError,match='Degenerate'):
        cut_vertex_gradient(x,np.array([[0,0,1]]),[0.],[])
    with pytest.raises(ValueError,match='integer'):
        cut_vertex_gradient(x,f.astype(float),np.zeros(len(f)),[])
    with pytest.raises(ValueError,match='finite scalar'):
        cut_vertex_gradient(x,f,np.full(len(f),np.nan),[])


def test_qr_baseline_matches_unweighted_normal_equations():
    x,f = grid()
    c = x[f].mean(1)
    mu = np.sin(c[:,0])+.2*c[:,1]**2
    ids = np.argsort(np.linalg.norm(c[:,None]-c[None,:],axis=2),axis=1)[:,:8]
    g = affine_face_gradient(x,f,mu,ids)
    tri = x[f]
    t = tri[:,1]-tri[:,0]
    t /= np.linalg.norm(t,axis=1)[:,None]
    s = np.cross(mesh_geometry(x,f)[-1], t)
    for i,js in enumerate(ids):
        d = c[js]-c[i]
        a = np.column_stack((d@t[i],d@s[i]))
        coeff = np.linalg.solve(a.T@a,a.T@(mu[js]-mu[i]))
        np.testing.assert_allclose(g[i],coeff[0]*t[i]+coeff[1]*s[i],atol=5e-14)


def test_mirror_recovery_and_exact_seam():
    x,f = sphere(2)
    mask = np.all(x[f,1] >= 0,axis=1)
    hf = f[mask]
    used = np.unique(hf)
    index = np.full(len(x),-1)
    index[used] = np.arange(len(used))
    hx,hf = x[used],index[hf]
    xx,ff,ee = mirror_half_wall(hx,hf,[])
    mu = xx[ff].mean(1)[:,0]
    g,_,_ = cut_vertex_gradient(xx,ff,mu,ee)
    nf = len(hf)
    np.testing.assert_allclose(g[nf:], g[:nf]*[1.,-1.,1.],atol=2e-14)
    bad = hx.copy(); bad[0,1] = -1e-13
    with pytest.raises(ValueError,match='nonnegative'):
        mirror_half_wall(bad,hf,[])


def test_curved_manufactured_sphere_gradient_and_pressure_refine():
    errors = []
    for level in (1,2,3):
        x,f = sphere(level)
        _,_,_,c,_,a,n = mesh_geometry(x,f)
        radial = c/np.linalg.norm(c,axis=1)[:,None]
        # Exact smooth sphere boundary perturbation potential: mu=U*x/2.
        mu = .5*radial[:,0]
        g,_,_ = cut_vertex_gradient(x,f,mu,[])
        exact = .5*(np.array([1.,0,0])-radial*radial[:,0,None])
        cp = surface_cp(g,n,[1.,0,0])
        cp_exact = 1-2.25*(1-radial[:,0]**2)
        errors.append((np.sqrt(np.sum(a*np.sum((g-exact)**2,axis=1))/a.sum()),
                       np.sqrt(np.sum(a*(cp-cp_exact)**2)/a.sum())))
    errors = np.array(errors)
    assert np.all(errors[1:] < errors[:-1]), errors
    assert errors[-1,0] < .5*errors[0,0], errors
    assert errors[-1,1] < .5*errors[0,1], errors


def test_face_permutation_and_cyclic_vertex_order():
    x,f = grid()
    mu = np.cos(x[f].mean(1)[:,0])
    g,c,_ = cut_vertex_gradient(x,f,mu,[])
    order = np.random.default_rng(5).permutation(len(f))
    gg,cc,_ = cut_vertex_gradient(x,f[order][:,[1,2,0]],mu[order],[])
    np.testing.assert_allclose(gg,g[order],atol=4e-14)
    np.testing.assert_allclose(cc,c[order][:,[1,2,0]],atol=4e-14)
