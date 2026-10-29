import numpy as np
import pytest

from panel.p1_galerkin_solver import (
    assemble_galerkin_multi, solve_galerkin_multi,
    _pressure_residual, _pressure_jacobian,
)


def fixture():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1,camber=.02)
    return combine_wings([wing],names=["main"],origins=[(0.,0.,0.)],
                         reference_chord=wing.chord,reference_point=wing.reference_point)


def test_galerkin_assembly_shapes_and_mass_dual_data():
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    from panel.p1_lifting_g2 import assemble_multi
    collocation=assemble_multi(mesh)
    n=mesh.ndof_mu
    assert ops["W"].shape[1]==collocation["W"].shape[1]
    assert ops["A"].shape==(n,n)
    assert ops["W"].shape[0]==n
    assert ops["M"].shape==(n,n)
    assert ops["b_xyz"].shape==(n,3)
    assert np.linalg.eigvalsh(ops["M"]).min()>0
    assert not ops["physical_accuracy_accepted"]


def test_signed_speed_galerkin_solve_enforces_existing_J_and_K_exactly():
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    state=solve_galerkin_multi(mesh,5.,kutta="signed_speed",operators=ops)
    assert np.isfinite(state["CL"])
    assert np.isfinite(state["CL_gamma"])
    assert state["jump_rms"]<1e-9
    assert state["signed_kutta_rms"]<1e-9
    assert np.isfinite(state["wall_dual_rms"])
    assert state["physical_accuracy_accepted"] is False


def test_pressure_kutta_jacobian_matches_finite_difference():
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    n=mesh.ndof_mu+ops["W"].shape[1]
    rng=np.random.default_rng(123)
    x=rng.normal(size=n)*.01
    analytic=_pressure_jacobian(x,ops,mesh.ndof_mu,ops["W"].shape[1])
    h=1e-6
    eye=np.eye(n)
    fd=np.column_stack([
        (_pressure_residual(x+h*eye[k],ops,mesh.ndof_mu)-
         _pressure_residual(x-h*eye[k],ops,mesh.ndof_mu))/(2*h)
        for k in range(n)])
    np.testing.assert_allclose(analytic,fd,rtol=2e-6,atol=2e-8)


def test_pressure_kutta_galerkin_path_reports_termination_not_just_residual():
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    state=solve_galerkin_multi(mesh,5.,kutta="pressure",operators=ops,max_nfev=3)
    assert state["nonlinear"] is not None
    assert isinstance(state["nonlinear"]["success"],bool)
    assert state["nonlinear"]["nfev"]<=3
    assert np.isfinite(state["pressure_kutta_rms"])


def test_galerkin_solver_couples_two_independent_lifting_groups():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    main=make_wing(chord_panels=3,span_panels=3,wake_panels=1,camber=.02)
    tail=make_wing(chord_panels=3,span_panels=3,span=1.8,chord=.55,
                   wake_panels=1,camber=0.)
    mesh=combine_wings([main,tail],names=["main","tail"],
                       origins=[(0.,0.,0.),(1.6,0.,.35)])
    ops=assemble_galerkin_multi(mesh,order=2)
    state=solve_galerkin_multi(mesh,5.,kutta="signed_speed",operators=ops)
    assert len(state["group_results"])==2
    assert {g["name"] for g in state["group_results"]}=={"main","tail"}
    assert state["jump_rms"]<1e-9
    assert state["signed_kutta_rms"]<1e-9
    assert np.isfinite(state["CL"]) and np.isfinite(state["CM_y"])


def test_single_group_wall_and_multi_group_batched_assemblies_match():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_galerkin_wall import assemble_galerkin_wall
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1,camber=.02)
    mesh=combine_wings([wing],names=["main"],origins=[(0.,0.,0.)],
                       reference_area=wing.reference_area,reference_chord=wing.chord,
                       reference_point=wing.reference_point)
    one=assemble_galerkin_wall(wing,order=2,target_face_batch=5,block=128)
    multi=assemble_galerkin_multi(mesh,order=2,target_face_batch=5,block=128)
    np.testing.assert_allclose(multi["A"],one["A"],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(multi["W"],one["W"],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(multi["b_xyz"]@np.array([1.,0.,0.]),one["b"],rtol=2e-12,atol=2e-12)


def test_batched_collocation_wake_matrix_matches_per_triangle_reference():
    from panel.p1_lifting_g1 import make_wing, _doublet_matrix
    from panel.p1_lifting_g2 import combine_wings, assemble_multi, _mesh
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=2,camber=.02)
    mesh=combine_wings([wing],names=["main"],origins=[(0.,0.,0.)])
    ops=assemble_multi(mesh);ms=_mesh(mesh)
    pieces=[]
    for g in mesh.groups:
        Wref=np.zeros((len(mesh.faces),g.gamma_hat.shape[1]))
        for f in g.wake_faces:
            D=_doublet_matrix(g.wake_xyz[f][None],ms.centers,
                              face_dofs=np.arange(3)[None],ndof=3,block=64)
            Wref+=D@g.gamma_hat[f]
        pieces.append(Wref)
    np.testing.assert_allclose(ops["W"],np.column_stack(pieces),rtol=2e-12,atol=2e-12)
