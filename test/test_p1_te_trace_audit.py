"""Manufactured TE-topology checks. No full repo or real-CAD assertion."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from panel.p1_te_trace_audit import audit_te_trace, audit_declared_cad_te


def fixture():
    xyz=np.array([[0,0,.1],[1,0,0],[0,1,.1],[1,1,0],
                  [0,0,-.1],[0,1,-.1]],float)
    faces=np.array([[0,1,3],[0,3,2],[4,3,1],[4,5,3]])
    bank=np.array([1,1,-1,-1])
    dface=faces.copy()
    dface[2]=[4,7,6]
    dface[3]=[4,5,7]
    wake_xyz=np.array([[1,0,0],[1,1,0],[2,0,0],[2,1,0]],float)
    wake_faces=np.array([[0,2,3],[0,3,1]])
    gamma_y=np.array([-1.,0.,1.,2.])
    gamma_hat=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    return NS(xyz=xyz,faces=faces,dof_face=dface,face_bank=bank,
              te_geo=np.array([1,3]),te_pairs=np.array([[1,6],[3,7]]),
              te_pair_y=np.array([0.,1.]),ndof_mu=8,
              wake_xyz=wake_xyz,wake_faces=wake_faces,
              gamma_hat=gamma_hat,gamma_y=gamma_y,chord=1.)


def codes(w):
    return {e["code"] for e in audit_te_trace(w)["errors"]}


def test_correct_two_bank_TE_cut_attaches_to_wake_root():
    r=audit_te_trace(fixture())
    assert r["ok"],r["errors"]
    assert r["split_te_vertices"]==2 and r["seam_edges"]==1
    assert r["wake_attached_seam_edges"]==1
    assert not r["pressure_kutta_verified"]


def test_te_pair_must_refer_to_same_geometric_vertex():
    w=fixture()
    w.te_pairs[0]=[1,7]
    assert "te_pair_not_same_geometry" in codes(w)


def test_unintended_cut_on_regular_same_bank_edge_is_rejected():
    w=fixture()
    w.dof_face[1,1]=7
    assert "cut_across_non_te_bank_edge" in codes(w)


def test_missing_lower_dof_is_detected():
    w=fixture()
    w.dof_face[2,2]=1
    assert "te_pair_not_same_geometry" in codes(w)


def test_wake_root_must_match_actual_TE_edge():
    w=fixture()
    w.wake_xyz[1,2]=.01
    assert "te_cut_edge_missing_from_wake_root" in codes(w)


def test_gamma_hat_at_root_must_match_TE_jump_interpolation():
    w=fixture()
    w.gamma_hat[0,0]=.2
    assert "wake_root_gamma_hat_incompatible_with_te_jump" in codes(w)


def test_bad_face_bank_tag_breaks_physical_TE_trace_pairing():
    w=fixture()
    w.face_bank[2]=0
    assert "missing_upper_or_lower_bank_support" in codes(w)


def test_different_upper_lower_thick_wing_points_are_not_te_split():
    w=fixture()
    assert not np.array_equal(w.xyz[0],w.xyz[4])
    assert audit_te_trace(w)["ok"]
    assert audit_te_trace(w)["split_te_vertices"]==2


def test_invalid_DOFS_fail_closed():
    w=fixture()
    w.dof_face[0,0]=90
    with pytest.raises(ValueError):
        audit_te_trace(w)


def test_CAD_pair_length_mismatch_not_silently_zipped():
    src=dict(faces=[[0,1,2],[2,1,3]],
             upper_te_faces=[0,0],lower_te_faces=[1],
             trailing_edges=[[1,2]])
    result=audit_declared_cad_te(src)
    assert "cad_te_face_pairs_length_mismatch" in {e["code"] for e in result["errors"]}


def test_CAD_pair_sharing_only_one_vertex_is_invalid():
    src=dict(faces=[[0,1,2],[0,3,4]],
             upper_te_faces=[0],lower_te_faces=[1],
             trailing_edges=[[0,1]])
    assert "cad_te_pair_not_shared_edge" in {
        e["code"] for e in audit_declared_cad_te(src)["errors"]}


def test_CAD_pair_requires_declared_TE_edge_and_opposite_winding():
    src=dict(faces=[[0,1,2],[2,1,3]],
             upper_te_faces=[0],lower_te_faces=[1],
             trailing_edges=[[1,2]])
    assert audit_declared_cad_te(src)["ok"]
    bad=dict(src,trailing_edges=[[0,1]])
    assert "cad_te_pair_not_declared_edge" in {
        e["code"] for e in audit_declared_cad_te(bad)["errors"]}
    reverse=dict(src,faces=[[0,1,2],[3,1,2]])
    assert "cad_te_edge_winding_mismatch" in {
        e["code"] for e in audit_declared_cad_te(reverse)["errors"]}


def test_nodal_TE_jump_cannot_bridge_unmatched_gamma_hat_knot():
    # On one affine TE edge y=0..1, a hat knot at y=.5 generates a kink.
    w=fixture()
    w.gamma_y=np.array([-.5,.5,1.5])
    w.gamma_hat=np.ones((4,1))*.5
    assert "te_edge_crosses_gamma_knot" in codes(w)


def test_P1_self_jump_is_full_linear_barycentric_trace():
    area=.5
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    mass=sum(area/3*np.outer(N,N) for N in bary)
    expected=area/12*np.array([[2,1,1],[1,2,1],[1,1,2]])
    np.testing.assert_allclose(mass, expected, atol=2e-17)
    values=np.array([.3,-.1,.4])
    N=np.array([.2,.3,.5])
    assert (-.5*N)@values==pytest.approx(-.5*(N@values))
    assert not np.isclose(-.5*(N@values),-.5*np.mean(values))


def test_existing_generated_G1_has_only_TE_cut_and_full_wake_attachment():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=4,span_panels=4,wake_panels=2)
    r=audit_te_trace(wing)
    assert r["ok"],r["errors"]
    assert r["split_te_vertices"]==3
    assert r["te_pair_rows"]==3
    assert r["seam_edges"]==4
    assert r["wake_attached_seam_edges"]==4
