"""Focused integration gates for actual steady free wake; not a test ladder."""
import numpy as np
import pytest

from panel.p1_lifting_g2 import make_conventional_fixture,assemble_multi,combine_wings
from panel.p1_lifting_g1 import make_wing
from panel.p1_steady_free_wake import (
    _potentials_at,_probe_segment_crosses_triangle,_safe_probe_scales,mean_wake_velocity,prepare_wake_material,
    reassemble_wake_only,solve_steady_free_wake)


def test_probe_stencil_shrinks_when_difference_crosses_another_face():
    centers=np.array([[0.,0.,0.]])
    normals=np.array([[0.,0.,1.]])
    own=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
    other=np.array([[.01,-1.,-.5],[.01,1.,-.5],[.01,0.,.5]])
    h,step,halvings=_safe_probe_scales(centers,normals,np.array([.1]),
                                       np.array([.02]),np.array([own,other]),
                                       np.array([0]),1.)
    assert halvings[0]==2
    assert h[0]==pytest.approx(.025)
    assert step[0]==pytest.approx(.005)


def test_probe_path_shrinks_when_another_sheet_lies_between_traces():
    own=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
    h,step,halvings=_safe_probe_scales(np.array([[0.,0.,0.]]),
        np.array([[0.,0.,1.]]),np.array([.1]),np.array([.02]),
        np.array([own,own+np.array([0.,0.,.05])]),np.array([0]),1.)
    assert halvings[0]==2
    assert h[0]==pytest.approx(.025)
    assert step[0]==pytest.approx(.005)


def test_probe_fails_when_surface_cannot_be_separated():
    centers=np.array([[0.,0.,0.]])
    normals=np.array([[0.,0.,1.]])
    own=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
    other=np.array([[0.,-1.,-.5],[0.,1.,-.5],[0.,0.,.5]])
    with pytest.raises(ValueError,match="probe intersects another surface"):
        _safe_probe_scales(centers,normals,np.array([.1]),np.array([.02]),
                           np.array([own,other]),np.array([0]),1.,max_halvings=2)


def test_probe_intersection_is_scale_independent():
    triangle=np.array([[0.,-1.,-1.],[0.,1.,-1.],[0.,0.,1.]])
    for scale in (1.,1e-5,1e-9):
        assert _probe_segment_crosses_triangle(
            scale*np.array([-1.,0.,0.]),scale*np.array([1.,0.,0.]),scale*triangle)


def test_full_velocity_of_zero_doublet_state_is_exact_uniform_flow():
    model=make_conventional_fixture(wake_panels=2)
    fake=dict(alpha_deg=5.,mu=np.zeros(model.ndof_mu),gamma=np.zeros(8))
    points=np.array([[2.1,.23,.5],[3.,-.7,-.4]])
    actual=_potentials_at(model,fake,points)
    expected=points@np.array([np.cos(np.deg2rad(5)),0,np.sin(np.deg2rad(5))])
    np.testing.assert_allclose(actual,expected,atol=1e-13)
    panel,_=mean_wake_velocity(model,fake)
    for value in panel.values():
        np.testing.assert_allclose(value[:,0],np.cos(np.deg2rad(5)),atol=1e-8)
        np.testing.assert_allclose(value[:,2],np.sin(np.deg2rad(5)),atol=1e-8)


def test_only_wake_reassembly_reproduces_original_g2_operator():
    model=make_conventional_fixture(wake_panels=3)
    static=assemble_multi(model)
    update=reassemble_wake_only(model,static)
    np.testing.assert_allclose(update['W'],static['W'],atol=2e-14,rtol=0)
    for key in ['A','J','K','grad']:
        assert update[key] is static[key]


def test_real_coupled_state_recomputed_after_both_wakes_relax():
    model=make_conventional_fixture(wake_panels=3,wake_length=5.)
    a=solve_steady_free_wake(model,5,kutta='pressure',iterations=3,relaxation=.2,
                             wall_formulation='collocation')
    assert len(a['history'])==4
    assert abs(a['history'][0]['CL']-.3654723)<.001
    assert a['history'][0]['jump_rms']<1e-12
    assert a['history'][-1]['jump_rms']<1e-12
    assert len(a['state']['group_results'])==2
    assert abs(a['history'][-1]['CL']-a['history'][0]['CL'])>1e-5
    for original,new in zip(model.groups,a['mesh'].groups):
        material=prepare_wake_material(original)
        assert len(material.roots)>0
        np.testing.assert_array_equal(original.gamma_hat,new.gamma_hat)
        np.testing.assert_array_equal(original.wake_faces,new.wake_faces)
        np.testing.assert_array_equal(original.wake_xyz[material.roots],
                                      new.wake_xyz[material.roots])
        assert np.max(abs(original.wake_xyz-new.wake_xyz))>1e-4
        assert a['history'][-1]['wakes'][original.name]['normal_slip_rms'] < a['history'][0]['wakes'][original.name]['normal_slip_rms']


def test_sheet_runs_full_budget_even_when_tolerances_already_met():
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)])
    result=solve_steady_free_wake(mesh,4.,iterations=3,geometry_tol=100.,slip_tol=100.)
    assert result["status"]=="completed"
    assert result["converged"]
    assert [row["iteration"] for row in result["history"]]==[0,1,2,3]
    assert "raw_map_residual" in result["history"][-1]


def test_single_group_steady_wake_and_cli_result(tmp_path):
    from panel.run_p1_steady_wake import main
    report=main(['--main-chord-panels','3','--main-span-panels','4',
                 '--tail-chord-panels','3','--tail-span-panels','3',
                 '--wake-panels','3','--alpha','5','--iterations','2',
                 '--out',str(tmp_path/'fw')])
    assert report['schema']=='dpanel.p1.g2.steady_free_wake.v1'
    assert (tmp_path/'fw'/'report.json').exists()
    a=next((tmp_path/'fw').glob('alpha_*/iteration_000.npz'))
    b=next((tmp_path/'fw').glob('alpha_*/iteration_002.npz'))
    with np.load(a) as aa,np.load(b) as bb:
        assert len(aa['mu'])>0 and len(bb['gamma'])>0
        assert np.max(abs(bb['wake_xyz_0']-aa['wake_xyz_0']))>1e-5
    with pytest.raises(FileExistsError):
        main(['--out',str(tmp_path/'fw')])
