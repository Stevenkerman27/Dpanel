import numpy as np
import pytest

from panel.p1_te_jump_integral import (
    audit_te_jump_integral, audit_g1_te_jump, audit_g2_group_te_jump,
)


def test_matching_linear_trace_and_gamma_mesh_has_zero_full_edge_residual():
    y=np.array([-1.,-.5,0.,.5,1.])
    gamma=np.array([0.,.3,.5,.3,0.])
    r=audit_te_jump_integral(y,gamma,y,gamma)
    assert r["status"]=="pass"
    assert r["partition_aligned"]
    assert r["nodal_residual_max"]==pytest.approx(0.)
    assert r["integral_residual"]==pytest.approx(0.)
    assert r["l2_rms"]==pytest.approx(0.)
    assert r["weak_moment_linf"]==pytest.approx(0.)


def test_endpoint_nodal_jump_can_pass_while_interior_gamma_knot_fails():
    r=audit_te_jump_integral([0.,1.],[0.,0.],[0.,.5,1.],[0.,1.,0.])
    assert r["nodal_residual_max"]==pytest.approx(0.)
    assert r["gamma_knots_inside_te_edges"]==[.5]
    assert r["status"]=="jump_mismatch"
    assert r["integral_residual"]==pytest.approx(-.5)
    assert r["l2_rms"]==pytest.approx(np.sqrt(1/3))
    assert r["edges"][0]["gamma_knot_inside"]


def test_signed_integral_can_cancel_while_l2_and_max_residual_are_nonzero():
    gy=np.linspace(0.,1.,5)
    gamma=np.array([0.,1.,0.,-1.,0.])
    r=audit_te_jump_integral([0.,1.],[0.,0.],gy,gamma)
    assert abs(r["integral_residual"])<1e-14
    assert r["integral_alone_can_hide_cancellation"]
    assert r["max_abs_residual"]==pytest.approx(1.)
    assert r["l2_rms"]>0.5


def test_weak_moments_sum_to_signed_integral():
    y=np.array([0.,.25,.5,.75,1.])
    j=np.array([.1,.2,.3,.4,.5])
    g=np.array([0.,.6,.2,.7,0.])
    r=audit_te_jump_integral(y,j,y,g)
    rr=j-g; h=np.diff(y); moments=np.zeros(len(y))
    for i,dx in enumerate(h):
        moments[i]+=dx*(2*rr[i]+rr[i+1])/6
        moments[i+1]+=dx*(rr[i]+2*rr[i+1])/6
    np.testing.assert_allclose(np.sum(moments),r["integral_residual"],atol=1e-15)


def test_g1_adapter_audits_generated_te_dofs_without_modifying_state():
    from panel.p1_lifting_g1 import make_wing
    wing=make_wing(chord_panels=4,span_panels=4,wake_panels=2)
    mu=np.zeros(wing.ndof_mu)
    gamma=np.array([.1,.2,.1])
    mu[wing.te_up[1:-1]]=0.
    mu[wing.te_down[1:-1]]=-gamma
    before=mu.copy()
    r=audit_g1_te_jump(wing,mu,gamma)
    assert r["status"]=="pass",r
    np.testing.assert_array_equal(mu,before)
    assert r["condition"]=="potential_jump_compatibility_only_not_pressure_kutta"


def test_g2_signed_speed_state_satisfies_jump_but_reports_pressure_kutta_separately():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings, assemble_multi, solve_multi
    wing=make_wing(chord_panels=4,span_panels=4,wake_panels=2,camber=.02)
    mesh=combine_wings([wing],names=["main_wing"],origins=[(0.,0.,0.)],
                       reference_chord=wing.chord,reference_point=wing.reference_point)
    ops=assemble_multi(mesh)
    state=solve_multi(mesh,5.,kutta="signed_speed",operators=ops)
    jump=audit_g2_group_te_jump(mesh,state,ops,"main_wing")
    assert jump["status"]=="pass",jump
    assert state["jump_rms"]<1e-9
    assert "pressure_kutta_rms" in state
    assert jump["condition"]!="pressure_kutta"


def test_invalid_unsorted_or_uncovered_knots_are_rejected():
    with pytest.raises(ValueError):
        audit_te_jump_integral([0.,1.,.5],[0.,0.,0.],[0.,1.],[0.,0.])
    with pytest.raises(ValueError):
        audit_te_jump_integral([0.,1.],[0.,0.],[.1,.9],[0.,0.])


def test_runner_json_serializes_numpy_boolean_and_solver_status_fields():
    import json
    from tools.issue71.te_jump_integral import jsonable
    value={"flag":np.bool_(True),"n":np.int64(3),"x":np.float64(.5),
           "nested":[np.bool_(False)]}
    encoded=json.dumps(jsonable(value),allow_nan=False)
    assert json.loads(encoded)=={"flag":True,"n":3,"x":.5,"nested":[False]}
