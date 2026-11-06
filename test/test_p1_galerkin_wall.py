import numpy as np
import pytest

from panel.p1_galerkin_wall import p1_mass_matrix, assemble_galerkin_wall


def test_exact_assembled_P1_mass_matrix_is_symmetric_positive_definite():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    M=p1_mass_matrix(wing)
    np.testing.assert_allclose(M,M.T,atol=1e-14)
    assert np.linalg.eigvalsh(M).min()>0
    area=sum(wing.chord*0 for _ in ()) + sum(
        np.linalg.norm(np.cross(wing.xyz[f[1]]-wing.xyz[f[0]],
                                wing.xyz[f[2]]-wing.xyz[f[0]]))/2
        for f in wing.faces)
    assert np.sum(M)==pytest.approx(area,rel=1e-12)


def test_galerkin_rhs_integrates_linear_freestream_exactly():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    U=np.array([.8,0.,.6])
    out=assemble_galerkin_wall(wing,freestream=U,order=2,include_wake=False)
    exact=-sum(
        np.linalg.norm(np.cross(wing.xyz[f[1]]-wing.xyz[f[0]],
                                wing.xyz[f[2]]-wing.xyz[f[0]]))/2
        * (wing.xyz[f].mean(axis=0)@U)
        for f in wing.faces)
    assert np.sum(out["b"])==pytest.approx(exact,abs=1e-11)
    assert out["A"].shape==(wing.ndof_mu,wing.ndof_mu)
    assert not out["includes_wake"]


def test_galerkin_matrix_has_the_exact_P1_self_jump_mass_term():
    from panel.p1_lifting_g1 import make_wing, _doublet_matrix, _mesh_as_kernel
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    mesh=_mesh_as_kernel(wing)
    f=0; tri=mesh.triangles[f]
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    pts=bary@tri; weights=np.full(3,mesh.areas[f]/3)
    # Isolate this source face so shared-node contributions from other faces
    # cannot be mistaken for the local jump mass.
    D=_doublet_matrix(tri[None],pts,self_faces=np.zeros(3,dtype=int),
                      face_dofs=np.array([[0,1,2]]),ndof=3)
    local=np.zeros((3,3))
    for k in range(3):
        local[k]=np.sum((weights*bary[:,k])[:,None]*D,axis=0)
    expected=-.5*(mesh.areas[f]/12)*np.array([[2,1,1],[1,2,1],[1,1,2]])
    np.testing.assert_allclose(local,expected,atol=1e-14)

def test_invalid_quadrature_order_is_rejected():
    from panel.p1_galerkin_wall import _duffy_rule
    with pytest.raises(ValueError):
        _duffy_rule(1)


def test_closed_body_constant_doublet_weak_identity():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    out=assemble_galerkin_wall(wing,order=3,include_wake=False)
    residual=out["A"]@np.ones(wing.ndof_mu)+out["M"]@np.ones(wing.ndof_mu)
    relative=np.linalg.norm(residual)/np.linalg.norm(out["M"]@np.ones(wing.ndof_mu))
    assert relative<1e-7


def test_galerkin_operator_is_finite_at_two_quadrature_orders():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    a2=assemble_galerkin_wall(wing,order=2,include_wake=False)
    a3=assemble_galerkin_wall(wing,order=3,include_wake=False)
    assert np.isfinite(a2["A"]).all() and np.isfinite(a3["A"]).all()
    assert np.linalg.norm(a3["A"]-a2["A"])<np.linalg.norm(a2["A"])*2.


def test_batched_wall_assembly_matches_original_per_face_quadrature():
    from panel.p1_lifting_g1 import make_wing, _doublet_matrix, _mesh_as_kernel
    from panel.p1_galerkin_wall import _duffy_rule
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1)
    mesh=_mesh_as_kernel(wing);bary,qw=_duffy_rule(2)
    n=wing.ndof_mu;A=np.zeros((n,n));W=np.zeros((n,wing.gamma_hat.shape[1]));b=np.zeros(n)
    wake_tris=wing.wake_xyz[wing.wake_faces]
    for f in range(len(wing.faces)):
        pts=bary@mesh.triangles[f];weights=2*mesh.areas[f]*qw
        D=_doublet_matrix(mesh.triangles,pts,self_faces=np.full(len(pts),f),
                          face_dofs=wing.dof_face,ndof=n,block=64)
        Dw=_doublet_matrix(wake_tris,pts,face_dofs=wing.wake_faces,
                           ndof=len(wing.wake_xyz),block=64)@wing.gamma_hat
        for k,row in enumerate(wing.dof_face[f]):
            wk=weights*bary[:,k]
            A[row]+=wk@D
            W[row]+=wk@Dw
            b[row]-=np.dot(wk,pts@np.array([.8,0.,.6]))
    fast=assemble_galerkin_wall(wing,freestream=(.8,0.,.6),order=2,
                                target_face_batch=5,block=128)
    np.testing.assert_allclose(fast["A"],A,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(fast["W"],W,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(fast["b"],b,rtol=2e-12,atol=2e-12)
