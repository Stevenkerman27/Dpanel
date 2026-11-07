import numpy as np
import pytest
import json
import os
import subprocess
import sys
from dataclasses import replace

from panel.p1_closed_body import influence_block
from panel.p1_filament_free_wake import (
    audit_nonadjacent_intersections, filament_velocity, line_target, resample_structured_wake, solve_filament_free_wake, structured_wake, structure_cad_wake,
    wake_ring_segments,
)
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import _potentials_at, _wake_mesh
from panel.regularized_vortex_filament import regularized_segment_velocity


def test_vectorized_filament_matches_existing_regularized_kernel():
    point=np.array([[.2,.4,.7],[-.3,.2,.5]])
    a=np.array([[-1.,0.,0.],[0.,1.,0.]])
    b=np.array([[1.,.2,0.],[.5,1.,1.]])
    gamma=np.array([.8,-.3])
    actual=filament_velocity(point,a,b,gamma,.1)
    expected=np.array([sum((regularized_segment_velocity(p,x,y,g,.1)
                            for x,y,g in zip(a,b,gamma)),np.zeros(3)) for p in point])
    np.testing.assert_allclose(actual,expected,atol=1e-14)


def test_ribbon_induction_converges_to_p1_potential_gradient_off_sheet():
    group=combine_wings([make_wing(chord_panels=3,span_panels=4,
                                    wake_panels=2,wake_length=2.)]).groups[0]
    gamma=np.array([.2,.5,.3])
    wake=_wake_mesh(group)
    face_mu=(group.gamma_hat @ gamma)[group.wake_faces]
    point=np.array([[2.,.3,.8]])
    h=1e-5
    expected=[]
    for axis in np.eye(3):
        _,_,plus=influence_block(point+h*axis,wake)
        _,_,minus=influence_block(point-h*axis,wake)
        expected.append(np.einsum("bfa,fa->b",plus-minus,face_mu)[0]/(2*h))
    expected=np.array(expected)
    errors=[]
    for subdivisions in (1,4,16):
        a,b,g=wake_ring_segments(group,gamma,subdivisions)
        actual=filament_velocity(point,a,b,g,1e-5)[0]
        errors.append(np.linalg.norm(actual-expected))
    assert errors[-1] < errors[0]*.03
    assert errors[-1] < 1e-4


def test_line_target_uniform_flow_keeps_root_and_material_lengths():
    xyz=np.array([[[0.,0.,0.],[0.,1.,0.]],
                  [[1.,0.,0.],[1.,1.,0.]],
                  [[2.,0.,0.],[2.,1.,0.]]])
    velocity=np.broadcast_to([1.,.2,.3],(2,2,3))
    target=line_target(xyz,velocity,np.ones((2,2)))
    np.testing.assert_array_equal(target[0],xyz[0])
    np.testing.assert_allclose(np.linalg.norm(np.diff(target,axis=0),axis=2),1.)
    np.testing.assert_allclose(np.diff(target,axis=0),
                               velocity/np.linalg.norm(velocity,axis=2)[:,:,None])


def test_structured_experiment_solves_and_keeps_material_gamma():
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,
                                    wake_panels=2,wake_length=2.)])
    group=mesh.groups[0]
    result=solve_filament_free_wake(mesh,4.,iterations=2,core_radius=.08,subdivisions=2)
    new=result["mesh"].groups[0]
    grid=structured_wake(group)
    assert result["status"] in ("completed","blocked_geometry")
    assert len(result["history"])>=2
    np.testing.assert_array_equal(new.wake_xyz[grid[0]],group.wake_xyz[grid[0]])
    np.testing.assert_array_equal(new.gamma_hat,group.gamma_hat)
    assert np.max(np.abs(new.wake_xyz-group.wake_xyz))>1e-4
    assert not result["physical_accuracy_accepted"]


def test_resample_swept_structured_wake_keeps_sheet_and_gamma():
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=2)
    xyz = wing.xyz.copy()
    wake = wing.wake_xyz.copy()
    xyz[:, 0] += .3 * np.abs(xyz[:, 1])
    wake[:, 0] += .3 * np.abs(wake[:, 1])
    mesh = combine_wings([replace(wing, xyz=xyz, wake_xyz=wake)])
    old = mesh.groups[0]
    new = resample_structured_wake(mesh, wake_panels=8).groups[0]
    old_grid = structured_wake(old)
    new_grid = structured_wake(new)
    np.testing.assert_allclose(new.wake_xyz[new_grid[0]], old.wake_xyz[old_grid[0]])
    np.testing.assert_allclose(new.wake_xyz[new_grid[-1]], old.wake_xyz[old_grid[-1]])
    np.testing.assert_array_equal(new.gamma_hat[new_grid[0]], old.gamma_hat[old_grid[0]])


def test_unstructured_wake_is_rejected():
    group=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)]).groups[0]
    with pytest.raises(ValueError,match="structured triangles"):
        structured_wake(replace(group,wake_faces=group.wake_faces[::-1]))


def test_cad_regrid_preserves_sheet_potential_and_attachment():
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2,wake_length=3.)])
    rebuilt=structure_cad_wake(mesh,wake_panels=4)
    group=rebuilt.groups[0]
    assert structured_wake(group).shape==(5,5)
    np.testing.assert_allclose(group.wake_xyz[group.wake_attachment_nodes],
                               mesh.groups[0].wake_xyz[mesh.groups[0].wake_attachment_nodes])
    state=dict(alpha_deg=4.,mu=np.zeros(mesh.ndof_mu),gamma=np.array([.1,.3,-.2]))
    points=np.array([[2.,.2,.8],[3.,-.4,.9]])
    np.testing.assert_allclose(_potentials_at(mesh,state,points),_potentials_at(rebuilt,state,points),
                               atol=1e-12,rtol=0)


def test_cad_regrid_rejects_contact_and_nonmaterial_basis():
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)])
    group=mesh.groups[0]
    with pytest.raises(ValueError,match="接触"):
        structure_cad_wake(replace(mesh,groups=(replace(group,contact_pairs=np.array([[0,1]])),)))
    basis=group.gamma_hat.copy()
    basis[-2,0]+=.1
    with pytest.raises(ValueError,match="环量基函数"):
        structure_cad_wake(replace(mesh,groups=(replace(group,gamma_hat=basis),)))


def test_nonadjacent_intersection_audit_detects_crossing_and_overlap():
    a=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    crossing=np.array([[.25,.25,-1.],[.25,.25,1.],[.8,.25,0.]])
    faces=np.array([[0,1,2],[3,4,5]])
    assert audit_nonadjacent_intersections(np.vstack((a,crossing)),faces)
    assert audit_nonadjacent_intersections(np.vstack((a,a)),faces)[0][2]=="coplanar"
    assert not audit_nonadjacent_intersections(np.vstack((a,a+[0.,0.,1.])),faces)


def test_intersection_guard_reports_blocked_without_moving_root(monkeypatch):
    import panel.p1_filament_free_wake as module
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)])
    monkeypatch.setattr(module,"audit_nonadjacent_intersections",lambda *args: [[0,1,"test"]])
    result=module.solve_filament_free_wake(mesh,4.,iterations=1,core_radius=.08,
                                          intersection_guard=True)
    assert result["status"]=="blocked_intersection"
    np.testing.assert_array_equal(result["mesh"].groups[0].wake_xyz,mesh.groups[0].wake_xyz)
    assert not result["converged"]


def test_residual_convergence_does_not_accept_intersecting_wake(monkeypatch):
    import panel.p1_filament_free_wake as module
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)])
    monkeypatch.setattr(module,"audit_nonadjacent_intersections",lambda *args: [[0,1,"test"]])
    result=module.solve_filament_free_wake(mesh,4.,iterations=0,core_radius=.08,
                          geometry_tol=100.,direction_tol=100.,slip_tol=100.)
    assert result["status"]=="invalid_self_intersection"
    assert not result["converged"]


def test_filament_runs_full_budget_even_when_tolerances_already_met():
    mesh=combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=2)])
    result=solve_filament_free_wake(mesh,4.,iterations=3,core_radius=.08,
                                   geometry_tol=100.,direction_tol=100.,slip_tol=100.)
    assert result["status"]=="completed"
    assert result["converged"]
    assert [row["iteration"] for row in result["history"]]==[0,1,2,3]


def test_filament_study_cli_saves_audited_result(tmp_path):
    output=tmp_path/"study.json"
    snapshot=tmp_path/"wake.npz"
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    result=subprocess.run([sys.executable,"-m","tools.wake_filament_study",
        "--out",str(output),"--snapshot",str(snapshot),"--chord-panels","3",
        "--span-panels","4","--wake-panels","2","--length","2","--core",".08",
        "--iterations","1","--order","2","--intersection-guard"],
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text(encoding="utf-8"))
    assert report["wall_formulation"]=="galerkin"
    assert report["intersection_guard"]
    assert report["nonadjacent_self_intersections"]==[]
    with np.load(snapshot) as data:
        assert {"xyz","faces","gamma_hat","gamma","mu"} <= set(data.files)
    benchmark=tmp_path/"benchmark.json"
    result=subprocess.run([sys.executable,"-m","tools.wake_filament_benchmark",
        "--snapshot",str(snapshot),"--out",str(benchmark),"--chord-panels","3",
        "--span-panels","4","--wake-panels","2","--length","2","--core",".08",
        "--repeat","1"],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert json.loads(benchmark.read_text(encoding="utf-8"))["targets"]==16
    crosscheck=tmp_path/"crosscheck.json"
    result=subprocess.run(
        [sys.executable,"-m","tools.wake_velocity_crosscheck",
         "--snapshot",str(snapshot),"--out",str(crosscheck),"--chord-panels","3",
         "--span-panels","4","--wake-panels","2","--length","2",
         "--cores",".08","--subdivisions","2","--audit-stencils"],
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    report=json.loads(crosscheck.read_text(encoding="utf-8"))
    assert report["state_source"]=="snapshot"
    assert report["targets"]==16
    assert np.isfinite(report["ribbons"][0]["slip_difference_rms"])
    assert isinstance(report["sheet"][0]["requested_stencil_crossings"],list)
    with np.load(snapshot) as data:
        np.savez(tmp_path/"old.npz",xyz_0=data["xyz"],faces_0=data["faces"])
    result=subprocess.run(
        [sys.executable,"-m","tools.wake_velocity_crosscheck",
         "--snapshot",str(tmp_path/"old.npz"),"--out",str(tmp_path/"recomputed.json"),
         "--chord-panels","3","--span-panels","4","--wake-panels","2","--length","2",
         "--cores",".08","--subdivisions","2","--order","2"],
        env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/"recomputed.json").read_text(encoding="utf-8"))["state_source"].startswith("recomputed")
