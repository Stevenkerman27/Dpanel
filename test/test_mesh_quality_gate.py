import numpy as np
import pytest
from panel.mesh_quality_gate import audit_mesh_quality


def grid():
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [2,0,0],[2,1,0]],float)
    wall=np.array([[0,1,2],[1,3,2]],int)
    wake=np.array([[1,4,3],[4,5,3]],int)
    return x,wall,wake


def test_area_angles_and_growth_for_regular_grid():
    x,f,w=grid()
    result=audit_mesh_quality(x,f,w)
    assert result['counts']==dict(wall_panels=2,wake_panels=2,total_panels=4)
    assert result['wall']['min_angle_deg']==pytest.approx(45.)
    assert result['wake']['min_angle_deg']==pytest.approx(45.)
    assert result['wall']['p99_neighbor_area_ratio']==pytest.approx(1.)
    assert result['wall']['shape_quality_p01']>0.8
    assert result['status']=="geometry_quality_only_not_physical_accuracy"


def test_permissive_opt_in_limits_pass():
    x,f,w=grid()
    q=audit_mesh_quality(x,f,w,max_wall_panels=2,max_wake_panels=2,
                         max_total_panels=4,min_wall_angle_deg=20,
                         max_wall_growth_p99=2)
    assert q['requested_gates']['max_total_panels']==4
    assert q['requested_gates']['min_wall_angle_deg']==20


@pytest.mark.parametrize('kw,label',[({'max_wall_panels':1},'wall_panels'),
                                     ({'max_wake_panels':1},'wake_panels'),
                                     ({'max_total_panels':3},'total_panels')])
def test_hard_budget_rejects_before_geometry(kw,label):
    x,f,w=grid()
    with pytest.raises(ValueError, match=label):
        audit_mesh_quality(x,f,w,**kw)


def test_angle_refuses_sliver_not_just_high_counts():
    x,f,w=grid()
    x[2]=[1e-2,1e-3,0]
    with pytest.raises(ValueError,match='angle gate'):
        audit_mesh_quality(x,f,w,min_wall_angle_deg=8)


def test_growth_gate_refuses_adjacent_area_jump():
    x,f,w=grid()
    x[3]=[100,1,0]
    with pytest.raises(ValueError,match='growth gate'):
        audit_mesh_quality(x,f,w,max_wall_growth_p99=3)


def test_invalid_nonmanifold_wall():
    x,f,w=grid()
    face=np.array([[1,2,4]])
    with pytest.raises(ValueError,match='Nonmanifold'):
        audit_mesh_quality(x,np.vstack((f,face)),w)


@pytest.mark.parametrize('kw',[{'max_wall_panels':0},
                               {'max_wall_panels':True},
                               {'max_total_panels':1.5},
                               {'max_wall_growth_p99':1},
                               {'max_wall_growth_p99':float('nan')},
                               {'min_wall_angle_deg':60},
                               {'min_wall_angle_deg':-2}])
def test_invalid_gates_rejected(kw):
    x,f,w=grid()
    with pytest.raises(ValueError):
        audit_mesh_quality(x,f,w,**kw)


def test_invalid_face_indices_and_degenerate_rejected():
    x,f,w=grid()
    with pytest.raises(ValueError,match='invalid vertices'):
        audit_mesh_quality(x,np.array([[0,1,22]]),w)
    with pytest.raises(ValueError,match='degenerate'):
        audit_mesh_quality(x,np.array([[0,1,1]]),w)


def test_inputs_are_not_modified():
    x,f,w=grid()
    xs,fs,ws=x.copy(),f.copy(),w.copy()
    audit_mesh_quality(x,f,w)
    assert np.array_equal(x,xs) and np.array_equal(f,fs) and np.array_equal(w,ws)
