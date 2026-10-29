import numpy as np
import pytest

from panel.p1_galerkin_wall import p1_mass_matrix, assemble_galerkin_wall


def test_exact_assembled_P1_mass_matrix_is_symmetric_positive_definite():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=2,span_panels=2,wake_panels=1)
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
    wing=make_wing(chord_panels=2,span_panels=2,wake_panels=1)
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
    wing=make_wing(chord_panels=2,span_panels=2,wake_panels=1)
    mesh=_mesh_as_kernel(wing)
    f=0; tri=mesh.triangles[f]
    # Degree-two exact 3-point rule on target triangle.
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    pts=bary@tri; weights=np.full(3,mesh.areas[f]/3)
    D=_doublet_matrix(mesh.triangles,pts,self_faces=np.full(3,f),
                      face_dofs=wing.dof_face,ndof=wing.ndof_mu)
    local=np.zeros((3,wing.ndof_mu))
    for k,dof in enumerate(wing.dof_face[f]):
        local[k]=np.sum((weights*bary[:,k])[:,None]*D,axis=0)
    expected=np.zeros_like(local)
    expected[:,wing.dof_face[f]]=-.5*(mesh.areas[f]/12)*np.array(
        [[2,1,1],[1,2,1],[1,1,2]])
    np.testing.assert_allclose(local[:,wing.dof_face[f]],expected[:,wing.dof_face[f]],atol=1e-14)


def test_invalid_quadrature_order_is_rejected():
    from panel.p1_galerkin_wall import _duffy_rule
    with pytest.raises(ValueError):
        _duffy_rule(1)
