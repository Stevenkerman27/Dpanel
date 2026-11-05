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


@pytest.mark.parametrize("sweep", [
    0.,
    pytest.param(.2, marks=pytest.mark.xfail(
        strict=True, reason="Known B3 defect: mesh-edge marching loses reference transverse offsets and moves TE attachment")),
])
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
