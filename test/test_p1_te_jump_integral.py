import numpy as np
import pytest
from panel.p1_te_jump_integral import audit_te_jump_integral, audit_g1_te_jump, audit_g2_group_te_jump


def test_aligned_trace_gamma_knots_have_zero_jump_residual():
    y=np.array([-1.,-.5,0.,.5,1.]); g=np.array([0.,.3,.5,.3,0.])
    r=audit_te_jump_integral(y,g,y,g)
    assert r["status"]=="pass" and r["partition_aligned"]
    assert r["l2_rms"]==pytest.approx(0.)


def test_interior_gamma_knot_is_detected_even_when_endpoints_match():
    r=audit_te_jump_integral([0.,1.],[0.,0.],[0.,.5,1.],[0.,1.,0.])
    assert r["nodal_residual_max"]==pytest.approx(0.)
    assert r["gamma_knots_inside_te_edges"]==[.5]
    assert r["integral_residual"]==pytest.approx(-.5)
    assert r["l2_rms"]==pytest.approx(np.sqrt(1/3))


def test_signed_integral_cancellation_does_not_hide_l2_error():
    r=audit_te_jump_integral([0.,1.],[0.,0.],np.linspace(0.,1.,5),[0.,1.,0.,-1.,0.])
    assert abs(r["integral_residual"])<1e-14
    assert r["integral_alone_can_hide_cancellation"]
    assert r["max_abs_residual"]==pytest.approx(1.)
    assert r["l2_rms"]>.5


def test_weak_moments_partition_identity():
    y=np.array([0.,.25,.5,.75,1.]); j=np.array([.1,.2,.3,.4,.5]); g=np.array([0.,.6,.2,.7,0.])
    r=audit_te_jump_integral(y,j,y,g)
    rr=j-g; h=np.diff(y); moments=np.zeros(len(y))
    for i,dx in enumerate(h):
        moments[i]+=dx*(2*rr[i]+rr[i+1])/6
        moments[i+1]+=dx*(rr[i]+2*rr[i+1])/6
    assert np.sum(moments)==pytest.approx(r["integral_residual"],abs=1e-15)


def test_g1_and_g2_adapters_keep_pressure_kutta_separate():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings,assemble_multi,solve_multi
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=1,camber=.02)
    mu=np.zeros(wing.ndof_mu); gamma=np.array([.1,.2])
    mu[wing.te_down[1:-1]]=-gamma
    r=audit_g1_te_jump(wing,mu,gamma)
    assert r["status"]=="pass"
    mesh=combine_wings([wing],names=["main"],origins=[(0.,0.,0.)])
    ops=assemble_multi(mesh); state=solve_multi(mesh,5.,kutta="signed_speed",operators=ops)
    q=audit_g2_group_te_jump(mesh,state,ops,"main")
    assert q["status"]=="pass"
    assert "pressure_kutta_rms" in state
    assert q["condition"]=="potential_jump_compatibility_only_not_pressure_kutta"


def test_invalid_knots_rejected():
    with pytest.raises(ValueError): audit_te_jump_integral([0.,1.,.5],[0.,0.,0.],[0.,1.],[0.,0.])
    with pytest.raises(ValueError): audit_te_jump_integral([0.,1.],[0.,0.],[.1,.9],[0.,0.])
