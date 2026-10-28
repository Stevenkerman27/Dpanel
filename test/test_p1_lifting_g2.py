"""Only a few complete G1/G2 solver experiments; do not turn into a gate list."""
import numpy as np
from panel.p1_lifting_g1 import make_wing,solve
from panel.p1_lifting_g2 import combine_wings,make_conventional_fixture,assemble_multi,solve_multi


def test_one_group_is_identical_to_g1_reference():
    wing=make_wing(chord_panels=4,span_panels=4,wake_panels=2,camber=.02)
    a=solve(wing,5,kutta='signed_speed')
    b=solve_multi(combine_wings([wing]),5,kutta='signed_speed')
    assert abs(a['CL']-b['CL'])<1e-10
    assert abs(a['CL_gamma']-b['CL_gamma'])<1e-10
    assert b['jump_rms']<1e-10


def test_independent_two_wakes_couple_to_same_wall():
    w=make_conventional_fixture(main=(4,4,3.,1.,.02),tail=(4,4,1.4,.5,0.),
                                tail_origin=(1.4,0.,.45),wake_panels=1)
    a=assemble_multi(w)
    assert len(w.groups)==2 and len(a['slices'])==2
    assert a['W'].shape[1] == sum(g.gamma_hat.shape[1] for g in w.groups)
    # Both wakes induce potential on both wall components, not post-hoc CL sums.
    for gi,g in enumerate(w.groups):
        other=w.groups[1-gi]
        s=a['slices'][g.name]
        assert np.linalg.norm(a['W'][other.face_ids,s])>1e-8
    r=solve_multi(w,5,kutta='signed_speed',operators=a)
    assert r['CL']>0.1 and r['CL_gamma']>0.1
    assert r['jump_rms']<1e-10
    assert all(len(z['gamma'])==3 for z in r['group_results'])
    assert not np.allclose(r['group_results'][0]['gamma'],r['group_results'][1]['gamma'])


def test_both_groups_pressure_kutta_picks_circulation():
    w=make_conventional_fixture(main=(3,4,2.5,1.,.015),tail=(3,4,1.4,.5,0.),
                                tail_origin=(1.2,0.,.4),wake_panels=1)
    a=assemble_multi(w)
    result=solve_multi(w,5,kutta='pressure',operators=a,max_nfev=50)
    assert result['CL']>0.08
    assert result['jump_rms']<1e-10
    assert result['pressure_kutta_rms']<1e-3
    assert len(result['group_results'])==2
    assert any(np.linalg.norm(z['gamma'])>1e-2 for z in result['group_results'])


def test_coupled_alpha_derivative_against_actual_full_solve():
    from panel.p1_lifting_g2 import alpha_sensitivity_multi
    w=make_conventional_fixture(main=(3,4,2.5,1.,.02),tail=(3,4,1.4,.5,0.),
                                tail_origin=(1.2,0.,.4),wake_panels=1)
    ops=assemble_multi(w)
    analytic=alpha_sensitivity_multi(w,5.,operators=ops)
    h=.001
    up=solve_multi(w,5+h,kutta='signed_speed',operators=ops)
    dn=solve_multi(w,5-h,kutta='signed_speed',operators=ops)
    finite=(up['CL']-dn['CL'])/(2*np.deg2rad(h))
    assert abs(analytic['dCL_dalpha_rad']-finite)<1e-7
    assert len(analytic['dCL_gamma_groups_dalpha_rad'])==2
