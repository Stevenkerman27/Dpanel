"""Oriented contact cuts: topology, branch compatibility, and no extra Kutta."""
import json
from pathlib import Path
import numpy as np
import pytest
from panel.p1_wake_contact import split_wall_trace_fans, with_declared_contact_cuts

ROOT = Path(__file__).resolve().parents[1]


def octahedron_contact_ring():
    # Closed wall; an annular +z wake meets the equator. No mesh deformation.
    x = np.array([[0.,0.,1.], [0.,0.,-1.],
                  [1.,0.,0.], [0.,1.,0.], [-1.,0.,0.], [0.,-1.,0.],
                  [2.,0.,0.], [0.,2.,0.], [-2.,0.,0.], [0.,-2.,0.]])
    f, w, e = [], [], []
    for i in range(4):
        a, b = 2+i, 2+(i+1)%4
        aa, bb = 6+i, 6+(i+1)%4
        f.extend([[0,a,b], [1,b,a]])
        w.extend([[a,aa,bb], [a,bb,b]])
        e.append([a,b])
    return x, np.array(f), np.array(w), np.array(e), np.array(['smooth_wall_contact']*4), np.ones((len(x),1))


def test_ring_has_two_traces_and_oriented_edge_vortex_cancellation():
    x,f,w,e,k,h = octahedron_contact_ring()
    out = split_wall_trace_fans(x,f,w,e,k,h)
    assert out['ndof_mu'] == 10  # six wall nodes + four extra trace DOFs
    assert out['contact_pairs'].shape == (4,2)
    assert out['te_pairs'].shape == (0,2)
    mu = np.zeros(out['ndof_mu'])
    mu[out['contact_pairs'][:,0]] = .7
    mu[out['contact_pairs'][:,1]] = -.3
    for edge in out['trace_audit']['oriented_edge_records']:
        for p,m in edge['endpoint_pairs']:
            assert mu[p]-mu[m] == pytest.approx(1.)
    assert out['trace_audit']['contact_constraints_are_not_kutta']


def test_piecewise_trace_offset_does_not_create_a_surface_gradient():
    from panel.p1_cp_diagnostics import p1_surface_gradient
    x,f,w,e,k,h = octahedron_contact_ring()
    out = split_wall_trace_fans(x,f,w,e,k,h)
    df = out['dof_face']; mu = np.full(out['ndof_mu'], np.nan)
    a = np.array([.6,.2,.3])
    for i, face in enumerate(f):
        shift = .5 if i%2 == 0 else -.5
        for vertex,dof in zip(face,df[i]):
            value = x[vertex]@a + shift
            if np.isfinite(mu[dof]):
                assert mu[dof] == pytest.approx(value)
            mu[dof] = value
    grad,geo = p1_surface_gradient(x,f,df,mu)
    expected = a - (geo['normal']@a)[:,None]*geo['normal']
    np.testing.assert_allclose(grad,expected,atol=4e-16)


def test_contact_pairing_is_rotation_invariant_not_normal_z_based():
    x,f,w,e,k,h = octahedron_contact_ring()
    a = split_wall_trace_fans(x,f,w,e,k,h)
    Q,_ = np.linalg.qr(np.random.default_rng(77).normal(size=(3,3)))
    if np.linalg.det(Q)<0: Q[:,0]*=-1
    b = split_wall_trace_fans(x@Q.T+[4.,7.,-2.],f,w,e,k,h)
    np.testing.assert_array_equal(a['dof_face'],b['dof_face'])
    np.testing.assert_array_equal(a['contact_pairs'],b['contact_pairs'])


def test_missing_contact_edge_is_rejected_not_silently_crossed():
    x,f,w,e,k,h = octahedron_contact_ring()
    with pytest.raises(ValueError,match='actual conforming wall/wake edges'):
        split_wall_trace_fans(x,f,w,e[:-1],k[:-1],h)


def test_inconsistent_wake_winding_is_rejected():
    x,f,w,e,k,h = octahedron_contact_ring()
    w[1] = w[1][::-1]
    with pytest.raises(ValueError,match='Inconsistent oriented bank'):
        split_wall_trace_fans(x,f,w,e,k,h)


def test_conv_splits_contact_but_keeps_te_kutta_and_gamma_dimensions():
    from panel.p1_lifting_g1 import from_dpanel_half_surface, _from_dpanel_half_surface_te_only
    from panel.p1_lifting_g2 import combine_wings
    path = ROOT/'cases/geom_robust/surfaces/conv_s075/surface.json'
    old = _from_dpanel_half_surface_te_only(path)
    wing = from_dpanel_half_surface(path)
    assert len(wing.te_pairs) == len(old.te_pairs) == 46
    assert len(wing.contact_pairs) == 67
    assert wing.ndof_mu == old.ndof_mu + len(wing.contact_pairs)
    np.testing.assert_array_equal(wing.xyz,old.xyz)
    np.testing.assert_array_equal(wing.faces,old.faces)
    np.testing.assert_array_equal(wing.gamma_hat,old.gamma_hat)
    assert wing.trace_audit['contact_edges'] == 68
    assert wing.trace_audit['trailing_edges'] == 46
    half_tags = np.asarray(json.loads(path.read_text())['component_ids'])
    tags = np.r_[half_tags,half_tags]
    for top,bottom in wing.te_face_pairs:
        assert np.all(tags[top]==30) and np.all(tags[bottom]==30)
    for record in wing.trace_audit['oriented_edge_records']:
        for v,(p,m) in zip(record['vertices'],record['endpoint_pairs']):
            if v in wing.trace_audit['unsplit_zero_gamma_tips']:
                assert p==m
            else:
                assert p!=m
    mesh = combine_wings([wing,wing],origins=[(0,0,0),(9,0,0)])
    np.testing.assert_array_equal(mesh.groups[1].contact_pairs,
                                  wing.contact_pairs+wing.ndof_mu)
    np.testing.assert_array_equal(mesh.groups[1].contact_gamma_hat,wing.contact_gamma_hat)


@pytest.mark.parametrize('name', ['elliptic_s075','swept30_s075','bwb_s055'])
def test_no_contact_fixtures_are_an_exact_noop(name):
    from panel.p1_lifting_g1 import _from_dpanel_half_surface_te_only
    path = ROOT/f'cases/geom_robust/surfaces/{name}/surface.json'
    data = json.loads(path.read_text())
    old = _from_dpanel_half_surface_te_only(path)
    assert with_declared_contact_cuts(old,data) is old


def test_g1_g2_jump_rows_include_contact_without_adding_kutta_rows():
    # An ordinary generated TE has known rows. Add a redundant identity zero
    # jump at one DOF to test assembler row plumbing without large CAD solves.
    from dataclasses import replace
    from panel.p1_lifting_g1 import make_wing, assemble
    from panel.p1_lifting_g2 import combine_wings, assemble_multi
    w = make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    w = replace(w,contact_pairs=np.array([[0,0]]),
                contact_gamma_hat=np.zeros((1,w.gamma_hat.shape[1])))
    a = assemble(w); mesh = combine_wings([w]); b = assemble_multi(mesh)
    assert a['C'].shape[0] == len(w.te_pairs)+1
    assert a['K'].shape[0] == len(w.te_pairs)
    assert b['J'].shape[0] == len(w.te_pairs)+1
    assert b['K'].shape[0] == len(w.te_pairs)
    np.testing.assert_allclose(b['J'], np.column_stack((a['C'],-a['GH'])))
