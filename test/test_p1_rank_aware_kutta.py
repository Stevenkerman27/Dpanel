"""Issue #69: mathematical oracles for rank-aware experimental pressure Kutta."""
import numpy as np
import pytest

from panel.p1_rank_aware_kutta import (
    RankControls, pressure_squares, pressure_jacobian, solve_wall_nullspace_kutta,
)


def synthetic():
    # Four unknowns [mu0,mu1,mu2,Gamma]. One exact TE jump mu0=Gamma.
    # Wall rows constrain only mu0/mu1: overdetermined but nonzero minimum.
    B=np.array([[1.,0,0,0],[0,1,0,0],[1,1,0,0],
                [0,0,0,0],[1,-1,0,0]])
    b=np.array([1.,1.,2.,.05,0.])
    J=np.array([[1.,0,0,-1.]])
    g=np.zeros((5,3,3))
    g[0,0,0]=g[0,0,2]=1.
    g[1,0,1]=1.
    pairs=[(np.array([0]),np.array([1]))]
    seed=np.array([1.,1.,-.05,1.])
    return B,b,J,g,pairs,seed


def test_analytic_pressure_jacobian_matches_independent_central_difference():
    B,b,J,g,pairs,seed=synthetic()
    x=np.array([1.2,.7,.3,1.2])
    h=1e-6
    fd=np.column_stack([(pressure_squares(x+h*np.eye(4)[k],g,pairs)-
                         pressure_squares(x-h*np.eye(4)[k],g,pairs))/(2*h)
                        for k in range(4)])
    np.testing.assert_allclose(pressure_jacobian(x,g,pairs),fd,atol=5e-10)


def test_wall_best_fit_and_exact_pressure_kutta_can_coexist():
    B,b,J,g,pairs,seed=synthetic()
    result=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed,
                                      expected_kutta_count=1)
    assert result["status"]=="pressure_kutta_in_weak_wall_modes_not_physical_acceptance"
    assert result["rank"]["free_after_jump"]==3
    assert result["rank"]["numerical_wall_rank"]==2
    assert result["rank"]["weak_mode_count"]==1
    assert result["wall_rms_minimizer"]>0.   # Strong residual is not zero.
    assert result["wall_relative_increase"]<1e-10
    assert result["jump_rms"]<1e-10
    assert result["pressure_kutta_rms"]<1e-10
    assert not result["physical_accuracy_accepted"]
    np.testing.assert_allclose(result["candidate"][:2],[1.,1.],atol=1e-10)


def test_full_rank_wall_does_not_invent_kutta_modes():
    B,b,J,g,pairs,seed=synthetic()
    B[3,2]=1.
    r=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed)
    assert r["status"]=="blocked_no_wall_nullspace"
    assert "candidate" not in r


def test_ambiguous_gap_is_not_treated_as_a_physical_nullspace():
    B=np.diag([1.,1.1e-9,.9e-9])
    b=np.ones(3);J=np.zeros((0,3));g=np.zeros((2,3,3))
    g[0,0,0]=g[0,0,2]=1.;g[1,0,1]=1.
    r=solve_wall_nullspace_kutta(
        B,b,J,g,[(np.array([0]),np.array([1]))],
        controls=RankControls(wall_rcond=1e-9))
    assert r["status"]=="blocked_no_spectral_gap"


def test_nullity_must_match_number_of_pressure_constraints():
    B,b,J,g,pairs,seed=synthetic()
    r=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed,
                                  expected_kutta_count=2)
    assert r["status"]=="blocked_nullity_kutta_count_mismatch"


def test_wall_scaling_does_not_change_recovered_physical_state():
    B,b,J,g,pairs,seed=synthetic()
    one=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed)
    seven=solve_wall_nullspace_kutta(7*B,7*b,J,g,pairs,seed=seed)
    np.testing.assert_allclose(one["candidate"],seven["candidate"],
                               rtol=1e-10,atol=1e-10)


def test_degenerate_pressure_jacobian_cannot_be_called_unique_kutta():
    B,b,J,g,pairs,seed=synthetic()
    g[:]=0.
    r=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed)
    assert r["status"]=="blocked_degenerate_pressure_jacobian"
    assert r["pressure_kutta_rms"]==0


def test_bad_controls_and_TE_face_references_are_rejected():
    B,b,J,g,pairs,seed=synthetic()
    with pytest.raises(ValueError):
        solve_wall_nullspace_kutta(B,b,J,g,pairs,
                                  controls=RankControls(wall_rcond=2.))
    with pytest.raises(ValueError):
        pressure_squares(seed,g,[(np.array([22]),np.array([1]))])


def test_zero_minimum_wall_residual_does_not_trigger_false_growth():
    B,b,J,g,pairs,seed=synthetic()
    b[3]=0.
    result=solve_wall_nullspace_kutta(B,b,J,g,pairs,seed=seed)
    assert result["status"]=="pressure_kutta_in_weak_wall_modes_not_physical_acceptance"
    assert result["wall_rms"]<1e-12
    assert result["pressure_kutta_rms"]<1e-10
