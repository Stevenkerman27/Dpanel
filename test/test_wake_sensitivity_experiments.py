from dataclasses import replace
import json

import numpy as np
import pytest

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings
from panel.p1_steady_free_wake import _one_wake_target, prepare_wake_material, with_wake_xyz
from tools.wake_mesh_interventions import clip_wake, refine_wake
from tools.wake_sensitivity_geometry import attachment_rows, audit_geometry, geometry_context


def test_physical_attachment_is_distinct_from_predecessor_roots_on_swept_wake():
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, wake = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:, 0] += .2 * abs(wall[:, 1])
    wake[:, 0] += .2 * abs(wake[:, 1])
    mesh = combine_wings([replace(wing, xyz=wall, wake_xyz=wake)])
    top = attachment_rows(mesh)
    assert len(top["attached"]) == 5
    assert len(top["graph_roots"]) < len(top["attached"])
    assert np.all(top["rows"][top["attached"]] == 0)


@pytest.mark.parametrize("sweep", [0., .2])
def test_production_material_uses_declared_te_attachment_identity(sweep):
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, wake = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:,0] += sweep*abs(wall[:,1])
    wake[:,0] += sweep*abs(wake[:,1])
    wing = replace(wing, xyz=wall, wake_xyz=wake)
    mesh = combine_wings([wing])
    group = mesh.groups[0]
    material = prepare_wake_material(group)
    assert set(material.attachments.tolist()) == set(group.wake_attachment_nodes.tolist())
    assert len(material.attachments) == len(wing.te_geo)
    assert (len(np.setdiff1d(material.attachments, material.roots)) > 0) == (sweep > 0)


def test_geometry_audit_does_not_treat_aabb_contact_as_penetration():
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=4, wake_panels=3)])
    context = geometry_context(mesh)
    baseline = audit_geometry(mesh, context)
    assert baseline["violation_count"] == 0
    assert baseline["inside_wall_node_ids"] == []
    assert baseline["wall_closed"]
    xyz = mesh.groups[0].wake_xyz.copy()
    free = np.setdiff1d(context["topology"]["used"], context["topology"]["attached"])
    xyz[free, 2] += .01
    changed = with_wake_xyz(mesh, {mesh.groups[0].name: xyz})
    result = audit_geometry(changed, context, baseline)
    assert result["new_or_deeper_violation_count"] == 0
    assert result["min_wake_wall_distance"] == 0.


def test_closed_wall_winding_detects_a_wake_vertex_inside_the_wing():
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=4, wake_panels=3)])
    context = geometry_context(mesh)
    baseline = audit_geometry(mesh, context)
    xyz = mesh.groups[0].wake_xyz.copy()
    free = np.setdiff1d(context["topology"]["used"], context["topology"]["attached"])
    node = int(free[np.argmin(abs(xyz[free, 1]))])
    xyz[node] = [.5, 0., 0.]
    changed = with_wake_xyz(mesh, {mesh.groups[0].name: xyz})
    result = audit_geometry(changed, context, baseline)
    assert node in result["new_inside_wall_node_ids"]
    assert result["new_or_deeper_violation_count"] > 0
    json.dumps(result, allow_nan=False)


def test_clipping_preserves_common_geometry_and_gamma_unknowns():
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=4, wake_panels=3)])
    changed, receipt = clip_wake(mesh, 2.)
    before, after = mesh.groups[0], changed.groups[0]
    assert receipt["gamma_dofs_unchanged"]
    assert changed.xyz is mesh.xyz and changed.faces is mesh.faces
    np.testing.assert_array_equal(after.wake_xyz[:len(before.wake_xyz)], before.wake_xyz)
    np.testing.assert_array_equal(after.gamma_hat[:len(before.gamma_hat)], before.gamma_hat)
    assert after.wake_xyz[np.unique(after.wake_faces), 0].max() <= receipt["cut_plane_x"]
    assert after.gamma_hat.shape[1] == before.gamma_hat.shape[1]


def test_directional_refinement_preserves_p1_potential_operator_and_orientation():
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=4, wake_panels=3)])
    baseline = assemble_multi(mesh)
    before = mesh.groups[0]
    tri = before.wake_xyz[before.wake_faces]
    area = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2].sum()
    for direction in ("streamwise", "spanwise"):
        changed, receipt = refine_wake(mesh, direction)
        after = changed.groups[0]
        assert receipt["gamma_dofs_unchanged"]
        assert len(after.wake_faces) > len(before.wake_faces)
        tri = after.wake_xyz[after.wake_faces]
        signed = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])[:, 2]
        assert (signed > 0).all()
        np.testing.assert_allclose(signed.sum(), area, atol=1e-14)
        operators = assemble_multi(changed)
        np.testing.assert_allclose(operators["W"], baseline["W"], atol=2e-14, rtol=1e-12)
        np.testing.assert_array_equal(operators["J"], baseline["J"])


@pytest.mark.parametrize("sweep", [0., .2])
def test_uniform_axial_flow_preserves_the_flat_reference_wake(sweep):
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, wake = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:, 0] += sweep*abs(wall[:, 1])
    wake[:, 0] += sweep*abs(wake[:, 1])
    mesh = combine_wings([replace(wing, xyz=wall, wake_xyz=wake)])
    group = mesh.groups[0]
    material = prepare_wake_material(group)
    velocity = np.zeros_like(group.wake_xyz)
    velocity[:, 0] = 1.
    target, frozen = _one_wake_target(group, material, velocity, min_axial=.08, freeze=True)
    assert frozen == []
    np.testing.assert_allclose(target[np.unique(group.wake_faces)],
                               group.wake_xyz[np.unique(group.wake_faces)], atol=1e-12, rtol=0.)

def test_oblique_uniform_flow_has_affine_drift_and_fixed_te_attachments():
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, wake = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:, 0] += .2 * abs(wall[:, 1])
    wake[:, 0] += .2 * abs(wake[:, 1])
    mesh = combine_wings([replace(wing, xyz=wall, wake_xyz=wake)])
    group = mesh.groups[0]
    material = prepare_wake_material(group)
    velocity = np.tile([1., .12, -.07], (len(group.wake_xyz), 1))
    target, frozen = _one_wake_target(group, material, velocity, min_axial=.08, freeze=True)
    assert frozen == []
    expected = material.reference_xyz.copy()
    distance = expected[:,0] - (1. + .2*abs(expected[:,1]))
    expected[:,1] += .12*distance
    expected[:,2] -= .07*distance
    np.testing.assert_allclose(target[np.unique(group.wake_faces)],
                               expected[np.unique(group.wake_faces)], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(target[material.attachments],
                               material.reference_xyz[material.attachments], atol=0., rtol=0.)
    free = np.setdiff1d(material.used, material.attachments)
    assert np.max(np.linalg.norm(target[free]-material.reference_xyz[free], axis=1)) > 0.


def test_irregular_material_rays_integrate_an_affine_slope_from_a_curved_te():
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, xyz = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:,0] += .2*abs(wall[:,1])
    xyz[:,0] += .2*abs(xyz[:,1])
    wall[:,2] += .1*wall[:,1] + .03*abs(wall[:,1])
    xyz[:,2] += .1*xyz[:,1] + .03*abs(xyz[:,1])
    # Off-column interior nodes make predecessor paths cross material-y labels.
    interior = (xyz[:,0] > 1. + .2*abs(xyz[:,1]) + 1e-8) & (abs(xyz[:,1]) < 1.99)
    xyz[interior,1] += .07*np.sin(xyz[interior,0])
    group = combine_wings([replace(wing, xyz=wall, wake_xyz=xyz)]).groups[0]
    material = prepare_wake_material(group, streamwise=True)
    velocity = np.ones_like(xyz)
    velocity[:,1] = .04 + .01*xyz[:,0] + .02*xyz[:,1]
    velocity[:,2] = -.03 + .01*xyz[:,1]
    target, frozen = _one_wake_target(group, material, velocity)
    start = 1. + .2*abs(xyz[:,1])
    length = xyz[:,0]-start
    expected = xyz.copy()
    expected[:,1] += (.04+.02*xyz[:,1])*length + .005*(xyz[:,0]**2-start**2)
    expected[:,2] += (-.03+.01*xyz[:,1])*length
    np.testing.assert_allclose(target, expected, atol=1e-11, rtol=0.)
    assert frozen == []
    np.testing.assert_array_equal(target[material.attachments], xyz[material.attachments])


def test_swept_attachments_survive_relaxation_and_step_limiting():
    from panel.p1_steady_free_wake import solve_steady_free_wake
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=3)
    wall, xyz = wing.xyz.copy(), wing.wake_xyz.copy()
    wall[:,0] += .2*abs(wall[:,1])
    xyz[:,0] += .2*abs(xyz[:,1])
    mesh = combine_wings([replace(wing, xyz=wall, wake_xyz=xyz)])
    result = solve_steady_free_wake(mesh, 4., iterations=2, relaxation=.3, max_step_chord=.01)
    group = result["mesh"].groups[0]
    attached = group.wake_attachment_nodes
    np.testing.assert_array_equal(group.wake_xyz[attached], xyz[attached])
    assert np.max(np.linalg.norm(group.wake_xyz-xyz,axis=1)) > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_potential_preserves_two_sided_finite_difference_velocity(device):
    import torch
    from panel.p1_steady_free_wake import mean_wake_velocity
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=4, wake_panels=3)])
    state = dict(alpha_deg=4., mu=.01*np.sin(np.arange(mesh.ndof_mu)),
                 gamma=np.array([.04,.08,.04]))
    expected, _ = mean_wake_velocity(mesh, state)
    actual, _ = mean_wake_velocity(mesh, state, block=96, potential_backend="torch", device=device)
    for name in expected:
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-9,rtol=1e-10)


def test_material_ray_starts_in_the_component_containing_its_node():
    from panel.p1_steady_free_wake import _material_line_integrals
    group = combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=3)]).groups[0]
    xyz = np.array([[x,y,0.] for x in (0.,1.,2.,3.) for y in (0.,1.)])
    faces = np.array([[0,2,1],[1,2,3],[4,6,5],[5,6,7]])
    matrix = _material_line_integrals(replace(group,wake_xyz=xyz,wake_faces=faces))
    np.testing.assert_allclose(matrix@np.ones(8),[0,0,1,1,0,0,1,1],atol=1e-14)


def test_orientation_backtracking_cannot_turn_a_large_map_residual_into_convergence(monkeypatch):
    import panel.p1_steady_free_wake as module
    mesh = combine_wings([make_wing(chord_panels=3,span_panels=4,wake_panels=3)])
    reference = mesh.groups[0].wake_xyz.copy()
    def unsafe(group, material, velocity, **kwargs):
        target = material.reference_xyz.copy()
        free = np.setdiff1d(material.used,material.attachments)
        target[free,1] *= -10.
        return target,[]
    monkeypatch.setattr(module,"_one_wake_target",unsafe)
    result = module.solve_steady_free_wake(mesh,4.,iterations=2,relaxation=.3)
    faces = mesh.groups[0].wake_faces
    before,after = reference[faces],result["mesh"].groups[0].wake_xyz[faces]
    cross0 = np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])
    cross1 = np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
    assert np.all(np.einsum("fi,fi->f",cross0,cross1)>0)
    assert result["history"][0]["orientation_backtracking"]["factor"] < 1.
    assert max(result["history"][0]["raw_map_residual"].values()) > 1.
    assert not result["converged"]

