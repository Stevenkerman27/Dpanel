"""Mathematical gates for pressure cancellation and idealized Trefftz drag.

These tests require NumPy/Pytest, not Gmsh or saved project runs. The
independent Dpanel small-mesh JAX integration is a separate opt-in test.
"""
import math

import numpy as np
import pytest

from panel.drag_consistency_audit import (
    ideal_trefftz_piecewise_linear, symmetric_half_gamma,
    pressure_integral_audit, wake_alignment_audit,
)


def test_ellipse_induced_drag_matches_cl_squared_over_pi_ar():
    # 4 m span, 4 m² area, c=1 m, AR=4, CL=.6, U=rho=1.
    y=np.linspace(-2.,2.,401)
    gamma0=2*.6/np.pi
    g=gamma0*np.sqrt(np.maximum(0.,1-(y/2)**2))
    result=ideal_trefftz_piecewise_linear(
        y,g,U=1.,density=1.,reference_area=4.)
    expected=.6**2/(np.pi*4)
    assert abs(result["CDi_ideal"]-expected)/expected < .012
    np.testing.assert_allclose(result["CL_circulation"], .6,
                               rtol=.005, atol=0.)


def test_mirror_half_circulation_with_shared_contact_root():
    y=np.array([0.,.2,.4,1.])
    positions,strength=symmetric_half_gamma(
        y,[0,0,1,-1],[.3,.2])
    np.testing.assert_array_equal(
        positions,[-1.,-.4,-.2,0.,.2,.4,1.])
    np.testing.assert_allclose(strength,[0.,.2,.3,.3,.3,.2,0.])
    with pytest.raises(ValueError):
        symmetric_half_gamma(y,[-1,0,1,-1],[.3,.2])


def test_surface_pressure_projection_and_component_banks():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [0.,0.,1.],[0.,1.,1.],[1.,0.,1.]],float)
    faces=np.array([[0,1,2],[3,4,5]])
    report=pressure_integral_audit(
        x,faces,[-.5,.3],[1.,0.,.1],density=1.225,
        reference_area=4.,component_ids=[20,30])
    np.testing.assert_allclose(
        report["drag_x_projection"]+report["drag_z_projection"],
        report["CD_pressure"],rtol=1e-13,atol=1e-13)
    assert report["face_sum_abs_drag"]>=abs(report["CD_pressure"])
    assert report["component_summary"]["20"]["wall_panels"]==1
    assert report["component_summary"]["30"]["wall_panels"]==1


def test_horizontal_wake_normal_flux_sin_alpha():
    xyz=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]
    alpha=math.radians(5)
    flow=[math.cos(alpha),0.,math.sin(alpha)]
    result=wake_alignment_audit(xyz,[[0,1,2]],flow)
    np.testing.assert_allclose(
        result["max_abs_Udotn_over_U"],math.sin(alpha),rtol=1e-13)
    assert wake_alignment_audit(
        xyz,[[0,1,2]],[1.,0.,0.])["max_abs_Udotn_over_U"] == 0.


def test_ideal_far_drag_not_equivalent_to_pressure_without_same_flow_solution():
    y=np.linspace(-2.,2.,201)
    gamma0=2*.6/np.pi
    g=gamma0*np.sqrt(np.maximum(0.,1-(y/2)**2))
    result=ideal_trefftz_piecewise_linear(
        y,g,U=1.,density=1.225,reference_area=4.)
    assert .02 < result["CDi_ideal"] < .03
    assert "CD_pressure" not in result


def test_bad_tip_gamma_or_sideslip_fails_closed():
    with pytest.raises(ValueError,match="tips"):
        ideal_trefftz_piecewise_linear(
            [-1,0,1],[.1,.3,0],U=1,density=1,reference_area=2)
    with pytest.raises(ValueError,match="Only zero-sideslip"):
        pressure_integral_audit(
            [[0,0,0],[1,0,0],[0,1,0]],
            [[0,1,2]],[-1],[1,.1,0],density=1.2,
            reference_area=2)
    with pytest.raises(ValueError,match="Zero-area"):
        wake_alignment_audit(
            [[0,0,0],[1,0,0],[2,0,0]],[[0,1,2]],[1,0,0])


def test_396_panel_projection_matches_local_frozen_force_evidence():
    # Algebraic control from existing PR22/JAX LU report. A separate
    # later test must check every face's Cp directly from saved surface.
    Fx,Fz=-.09838129905838668,1.2978769424355368
    alpha=math.radians(5)
    qS=.5*1.225*4
    xx=Fx*math.cos(alpha)/qS
    zz=Fz*math.sin(alpha)/qS
    np.testing.assert_allclose(xx,-.040002827964629895,rtol=1e-13)
    np.testing.assert_allclose(zz,.04617037914817501,rtol=1e-13)
    np.testing.assert_allclose(
        xx+zz,.006167551183545117,rtol=1e-13)
    assert (abs(xx)+abs(zz))/abs(xx+zz)>13.


def test_moment_pressure_integration_and_cancellation():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [0.,0.,1.],[0.,1.,1.],[1.,0.,1.]])
    report=pressure_integral_audit(
        x,[[0,1,2],[3,4,5]],[-.5,.3],[1.,0.,.1],
        density=1.,reference_area=2.,reference_point=[.25,0.,0.],
        reference_chord=1.)
    assert np.isfinite(report["CM_y"])
    assert report["moment_y_cancellation_factor"] >= 1
    assert all("CM_y" in x for x in report["component_summary"].values())
