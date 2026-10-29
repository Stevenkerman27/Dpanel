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
    n=mesh.ndof_mu
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
