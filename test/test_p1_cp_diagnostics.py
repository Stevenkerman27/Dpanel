"""Independent comparators for surface P1 velocity/Cp and off-surface potential."""
import numpy as np
import pytest
from panel.p1_cp_diagnostics import (
    p1_surface_gradient, potential_gradient_fd4, off_surface_cp_ladder,
    compare_p1_to_offsurface, kutta_pressure_diagnostic, pressure_load_diagnostic,
)


def grid(nx=32, ny=3):
    xs=np.linspace(0,2*np.pi,nx+1); ys=np.linspace(0,1,ny+1)
    xyz=np.array([[x,y,0.] for y in ys for x in xs])
    faces=[]
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i; b=a+1; c=a+nx+1; d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    faces=np.asarray(faces,int)
    return xyz,faces,faces.copy()


def total_phi(points, U=np.array([1.,0.,0.]), amp=.1):
    p=np.asarray(points)
    # Harmonic perturbation; d(phi)/dn=0 on z=0.
    return p@U+amp*np.sin(p[:,0])*np.cosh(p[:,2])


def test_p1_gradient_exact_for_affine_potential_even_on_sliver():
    xyz=np.array([[0.,0.,0.],[26.,0.,0.],[26.,.001,0.]])
    faces=np.array([[0,1,2]])
    grad,q=p1_surface_gradient(xyz,faces,faces,xyz@np.array([1.,.2,0.]))
    np.testing.assert_allclose(grad[0],[1.,.2,0.],atol=2e-10)
    assert q["aspect_ratio"][0]>2e4


def test_fd4_independently_recovers_total_potential_gradient():
    pts=np.array([[.3,.2,.1],[2.1,.5,.03],[4.5,.1,.2]])
    g=potential_gradient_fd4(total_phi,pts,1e-3)
    exact=np.column_stack((1.+.1*np.cos(pts[:,0])*np.cosh(pts[:,2]),
                           np.zeros(len(pts)),
                           .1*np.sin(pts[:,0])*np.sinh(pts[:,2])))
    np.testing.assert_allclose(g,exact,atol=2e-11,rtol=2e-11)


def test_comparison_separates_mu_gradient_from_Ut_plus_gradient():
    xyz,faces,dof_face=grid(48,3); mu=total_phi(xyz)
    r=compare_p1_to_offsurface(xyz,faces,dof_face,mu,total_phi,[1.,0.,0.],
                               offsets=[.08,.04,.02],step_fraction=.2)
    direct=r["p1_cp_current_grad_mu"]
    fd=r["offsurface_ladder"][-1]["cp_tangent_speed"]
    candidate=r["p1_cp_candidate_Ut_plus_grad_mu"]
    # Here mu is intentionally the TOTAL surface potential. Adding U twice is wrong.
    assert np.sqrt(np.mean((direct-fd)**2))<5e-3
    assert np.sqrt(np.mean((candidate-fd)**2))>.1


def test_epsilon_ladder_keeps_normal_velocity_and_sensitivity():
    xyz,faces,_=grid(24,2); centers=xyz[faces].mean(axis=1)
    normals=np.tile([0.,0.,1.],(len(faces),1))
    r=off_surface_cp_ladder(total_phi,centers,normals,[1.,0.,0.],[.08,.04,.02])
    assert len(r)==3 and r[0]["cp_tangent_delta_previous"] is None
    assert r[1]["cp_tangent_delta_previous"] is not None
    assert np.max(np.abs(r[-1]["velocity_normal"]))>0


def test_local_scale_offsets_are_face_specific():
    xyz,faces,_=grid(12,2); centers=xyz[faces].mean(axis=1)
    normals=np.tile([0.,0.,1.],(len(faces),1)); scale=np.linspace(.01,.1,len(faces))
    r=off_surface_cp_ladder(total_phi,centers,normals,[1.,0.,0.],[.2,.1],local_scale=scale)
    assert r[0]["offset_min"]==pytest.approx(.002)
    assert r[0]["offset_max"]==pytest.approx(.02)
    assert r[1]["offset_median"]==pytest.approx(.0055)


def test_kutta_pressure_diagnostic_reports_jump_without_modifying_cp():
    cp=np.array([-.4,-.2,.1,.3]); before=cp.copy()
    d=kutta_pressure_diagnostic(cp,np.ones(4),[[0,1]],[[2,3]])
    np.testing.assert_allclose(d["cp_upper"],[-.3])
    np.testing.assert_allclose(d["cp_lower"],[.2])
    np.testing.assert_allclose(d["cp_jump_upper_minus_lower"],[-.5])
    np.testing.assert_array_equal(cp,before)


def test_pressure_load_diagnostic_exposes_drag_cancellation():
    cp=np.array([-.2,.18]); areas=np.ones(2)
    normals=np.array([[1.,0.,0.],[1.,0.,0.]])
    centers=np.array([[0.,0.,0.],[1.,0.,0.]])
    d=pressure_load_diagnostic(cp,areas,normals,centers,1.,[1.,0.,0.],[0.,0.,1.])
    assert d["CD_pressure"]==pytest.approx(.02)
    assert d["drag_cancellation_ratio"]==pytest.approx(19.)
    assert d["drag_positive_sum"]==pytest.approx(.2)
    assert d["drag_negative_sum"]==pytest.approx(-.18)


def test_degenerate_triangle_and_bad_step_fail_loudly():
    with pytest.raises(ValueError):
        p1_surface_gradient(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
                            np.array([[0,1,2]]),np.array([[0,1,2]]),[0.,1.,2.])
    with pytest.raises(ValueError):
        potential_gradient_fd4(total_phi,np.zeros((1,3)),0.)


def test_zero_drag_cancellation_ratio_is_null_with_explicit_reason():
    cp=np.array([1.,-1.]);areas=np.ones(2)
    normals=np.array([[0.,0.,1.],[0.,0.,1.]])
    centers=np.zeros((2,3))
    d=pressure_load_diagnostic(cp,areas,normals,centers,1.,[1.,0.,0.],[0.,0.,1.])
    assert d["CD_pressure"]==pytest.approx(0.)
    assert d["drag_abs_sum"]==pytest.approx(0.)
    assert d["drag_cancellation_ratio"] is None
    assert d["drag_cancellation_status"]=="no_drag_contributions"


def test_nonzero_face_drag_with_zero_net_is_reported_as_complete_cancellation():
    cp=np.array([.2,-.2]);areas=np.ones(2)
    normals=np.array([[1.,0.,0.],[1.,0.,0.]])
    centers=np.zeros((2,3))
    d=pressure_load_diagnostic(cp,areas,normals,centers,1.,[1.,0.,0.],[0.,0.,1.])
    assert d["CD_pressure"]==pytest.approx(0.)
    assert d["drag_abs_sum"]==pytest.approx(.4)
    assert d["drag_cancellation_ratio"] is None
    assert d["drag_cancellation_status"]=="complete_cancellation"
