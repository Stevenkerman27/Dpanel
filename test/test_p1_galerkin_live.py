import numpy as np
import pytest
import torch

from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_galerkin_solver import assemble_galerkin_multi as reference_assembly, solve_galerkin_multi
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_steady_free_wake import reassemble_wake_only, solve_steady_free_wake, with_wake_xyz


def fixture(two=False):
    wing=make_wing(chord_panels=3,span_panels=3,wake_panels=2,camber=.02)
    if two:
        tail=make_wing(chord_panels=3,span_panels=3,wake_panels=2,span=1.8,chord=.55)
        return combine_wings([wing,tail],names=["main","tail"],origins=[(0,0,0),(1.6,0,.35)])
    return combine_wings([wing],names=["main"],origins=[(0,0,0)])


def test_duffy_rule_does_not_use_numpy_lapack(monkeypatch):
    from panel.p1_galerkin_wall import _duffy_rule
    monkeypatch.setattr(np.linalg,"eigvalsh",lambda *args: pytest.fail("NumPy LAPACK called"))
    bary,weights=_duffy_rule(5)
    np.testing.assert_allclose(weights.sum(),.5,atol=1e-14)
    np.testing.assert_allclose(weights @ bary,np.full(3,1./6.),atol=1e-14)


@pytest.mark.parametrize("kutta",["signed_speed","pressure"])
def test_torch_matches_scipy_reference(kutta):
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    refops=reference_assembly(mesh,order=2)
    actual=solve_multi(mesh,5.,kutta=kutta,operators=ops,wall_formulation="galerkin",max_nfev=150)
    expected=solve_galerkin_multi(mesh,5.,kutta=kutta,operators=refops,max_nfev=150)
    for key in ("mu","gamma","CL","CL_gamma","CD_pressure","CM_y","wall_dual_rms"):
        np.testing.assert_allclose(actual[key],expected[key],rtol=3e-4,atol=3e-6,err_msg=key)
    assert actual["wall_formulation"]=="galerkin"
    assert actual["physical_accuracy_accepted"] is False


@pytest.mark.parametrize("two",[False,True])
def test_changed_wake_cached_assembly_matches_full(two):
    mesh=fixture(two)
    ops=assemble_galerkin_multi(mesh,order=2)
    coordinates={}
    for group in mesh.groups:
        xyz=group.wake_xyz.copy()
        xyz[:,2]+=.03*np.maximum(xyz[:,0]-xyz[:,0].min(),0.)
        coordinates[group.name]=xyz
    changed=with_wake_xyz(mesh,coordinates)
    cached=reassemble_wake_only(changed,ops)
    full=assemble_galerkin_multi(changed,order=2)
    for key in ("A","W","M","b_xyz","J","K","grad"):
        np.testing.assert_allclose(cached[key],full[key],rtol=2e-12,atol=2e-12,err_msg=key)
    assert not np.allclose(ops["W"],cached["W"])


def test_free_wake_initial_state_uses_galerkin():
    mesh=fixture()
    fixed=solve_multi(mesh,5.,kutta="signed_speed",wall_formulation="galerkin",galerkin_order=2)
    result=solve_steady_free_wake(mesh,5.,kutta="signed_speed",iterations=1,
                                  wall_formulation="galerkin",galerkin_order=2)
    np.testing.assert_allclose(result["history"][0]["CL"],fixed["CL"],rtol=1e-10,atol=1e-10)
    assert result["state"]["wall_formulation"]=="galerkin"
    assert result["history"][0]["wall_dual_rms"] is not None


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_galerkin_cuda_matches_cpu():
    mesh=fixture()
    ops=assemble_galerkin_multi(mesh,order=2)
    cpu=solve_multi(mesh,5.,operators=ops,wall_formulation="galerkin",device="cpu")
    gpu=solve_multi(mesh,5.,operators=ops,wall_formulation="galerkin",device="cuda")
    for key in ("mu","gamma","CL","CD_pressure","CM_y","wall_dual_rms"):
        np.testing.assert_allclose(cpu[key],gpu[key],rtol=3e-5,atol=3e-7,err_msg=key)


def test_live_fixed_wake_chord_refinement():
    values=[]
    for chord in (8,12):
        wing=make_wing(chord_panels=chord,span_panels=8,wake_length=5.,wake_panels=2)
        mesh=combine_wings([wing],reference_chord=1.,reference_area=4.,
                           reference_point=np.array([.25,0.,0.]))
        state=solve_multi(mesh,4.,wall_formulation="galerkin",max_nfev=150)
        assert state["nonlinear"]["success"]
        values.append(state["CL"])
    assert all(.2 < value < .3 for value in values)
    assert abs(values[1]-values[0])/abs(values[1]) < .03
