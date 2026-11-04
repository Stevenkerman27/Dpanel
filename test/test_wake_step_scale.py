"""Per-node step control uses the mesh resolution, not the map's link length.

A step cap only bounds strain when it is expressed in the mesh's own resolution.
The rooted arclength map marches one REFERENCE LINK per node, and a CAD wake may
carry a truncation row whose link is tens of chords long while its panels stay
short; capping such a node at a quarter of that link (6.5 chord on the G2 CAD
wake) accepts a displacement that deforms the sheet far beyond the area/fold
gate, which is what the map-unit experiment measured
(docs/archive/issue/ISSUE84_STREAMLINE_STEP_UNIT.md).  The link is therefore reported for
diagnostics and truncation marking, never used as the cap.
"""
import numpy as np
import pytest


def _material(reference, predecessor, roots, faces):
    from panel.p1_steady_free_wake import WakeMaterial
    reference = np.asarray(reference, float)
    tri = reference[np.asarray(faces, int)]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    used = np.unique(np.asarray(faces, int))
    order = used[np.argsort(reference[used, 0])]
    return WakeMaterial('wake', used, np.asarray(roots, int), order,
                        np.asarray(predecessor, int), reference, area)


def _long_row_sheet(link=15.):
    """Four resolved columns plus one far column (a truncation row)."""
    xs = [0., 1., 2., 3., 3.+link]
    reference = np.array([[x, y, 0.] for x in xs for y in (0., 1.)])
    faces, predecessor = [], []
    for i in range(len(xs)-1):
        a, b = 2*i, 2*i+1
        faces.append([a, b, a+2])
        faces.append([b, b+2, a+2])   # (2i+1, 2i+3, 2i+2); b+1 would repeat a+2
    for i in range(len(xs)):
        for j in (0, 1):
            predecessor.append(2*(i-1)+j if i else -1)
    return reference, np.array(faces, int), np.array(predecessor, int)


def test_the_step_scale_is_the_shortest_incident_edge_while_the_link_is_reported():
    from panel.p1_coupled_material_wake import node_step_scale, wake_step_scale
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    out = wake_step_scale(reference, faces, material)
    assert np.allclose(out['scale'], node_step_scale(reference, faces))
    assert np.allclose(out['scale'], out['geometric'])
    assert np.allclose(out['geometric'], node_step_scale(reference, faces))
    # The link stays available for diagnostics and never leaks into the cap.
    assert np.isclose(out['link'][8], 15.) and np.isclose(out['link'][2], 1.)
    assert np.isclose(out['link_median'], 1.)
    assert np.all(out['scale'][2:] <= out['link'][2:]+1e-12)
    # Only the far row is a truncation row; it is reported, not hidden.
    assert list(np.nonzero(out['truncation'])[0]) == [8, 9]


def test_truncation_is_relative_so_a_uniformly_coarse_sheet_is_not_flagged():
    from panel.p1_coupled_material_wake import wake_step_scale
    reference = np.array([[0., 0., 0.], [0., 1., 0.], [10., 0., 0.], [10., 1., 0.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    predecessor = np.array([-1, -1, 0, 1])
    material = _material(reference, predecessor, [0, 1], faces)
    out = wake_step_scale(reference, faces, material)
    assert not out['truncation'].any()
    assert np.allclose(out['scale'], out['geometric'])


def test_a_root_keeps_a_finite_scale():
    from panel.p1_coupled_material_wake import wake_step_scale
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    out = wake_step_scale(reference, faces, material)
    assert out['link'][0] == 0. and out['link'][1] == 0.
    assert np.isfinite(out['scale'][0]) and np.isfinite(out['scale'][1])


def test_the_cap_is_in_geometric_units():
    from panel.p1_coupled_material_wake import clip_step_update, wake_step_scale
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    out = wake_step_scale(reference, faces, material)
    # Node 8 sits on a 1-chord panel even though its link is 15 chords long.
    assert np.isclose(out['scale'][8], 1.)
    inside = reference.copy()
    inside[8] += np.array([0., 0.2, 0.])
    _new, ledger = clip_step_update(reference, inside, .25*out['scale'],
                                    roots=material.roots, reference=reference)
    assert ledger['unit'] == 'geometric_edge'
    assert ledger['clipped'] == 0, ledger['clipped_ids']
    # 0.2 * 15 = 3 chord of requested motion is clipped to a quarter panel.
    outside = reference.copy()
    outside[8] += np.array([0., 0.2*out['link'][8], 0.])
    _new, ledger = clip_step_update(reference, outside, .25*out['scale'],
                                    roots=material.roots, reference=reference)
    assert ledger['clipped'] == 1 and 8 in ledger['clipped_ids']
    assert ledger['accepted_max'] <= .25*out['scale'][8]*(1.+1e-9)


def test_roots_stay_pinned_under_clipping():
    from panel.p1_coupled_material_wake import clip_step_update, wake_step_scale
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    out = wake_step_scale(reference, faces, material)
    target = reference.copy()
    target[0] += np.array([1., 1., 0.])
    new, _ledger = clip_step_update(reference, target, .25*out['scale'],
                                    roots=material.roots, reference=reference)
    assert np.array_equal(new[material.roots], reference[material.roots])


def _target(policy, scale):
    from panel.p1_coupled_material_wake import streamline_target
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    velocity = np.tile([1., 0.35, 0.], (len(reference), 1))   # 19 deg off the reference
    eligible = np.ones(len(reference), bool)
    return streamline_target(reference, material, velocity, eligible,
                             step_scale=scale, truncation_policy=policy)


def test_freezing_a_truncation_row_holds_it_and_still_advances_the_resolved_sheet():
    from panel.p1_coupled_material_wake import wake_step_scale
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    scale = wake_step_scale(reference, faces, material)
    advance, freeze = _target('advance', scale), _target('freeze', scale)
    # The 15-chord link asks for link*sin(19 deg) ~ 4.9 chord.
    assert np.linalg.norm(advance-reference, axis=1)[8] > 4.
    assert np.array_equal(freeze[scale['truncation']], reference[scale['truncation']])
    resolved = ~scale['truncation']
    assert np.allclose(freeze[resolved], advance[resolved])


def test_an_unknown_truncation_policy_is_rejected():
    from panel.p1_coupled_material_wake import streamline_target
    reference, faces, predecessor = _long_row_sheet()
    material = _material(reference, predecessor, [0, 1], faces)
    velocity = np.tile([1., 0., 0.], (len(reference), 1))
    eligible = np.ones(len(reference), bool)
    with pytest.raises(ValueError, match="truncation_policy must be"):
        streamline_target(reference, material, velocity, eligible, truncation_policy='sometimes')
    with pytest.raises(ValueError, match="requires the per-node step_scale"):
        streamline_target(reference, material, velocity, eligible, truncation_policy='freeze')


def test_the_solver_reports_the_policy_and_is_unchanged_without_truncation_rows():
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    mesh = combine_wings([make_wing(chord_panels=4, span_panels=4, wake_panels=2)])
    advanced = solve_coupled_material_wake(mesh, 5., kutta='signed_speed', iterations=1,
                                           gate_mode='audit')
    frozen = solve_coupled_material_wake(mesh, 5., kutta='signed_speed', iterations=1,
                                         gate_mode='audit', truncation_policy='freeze')
    for row_adv, row_frz in zip(advanced['history'], frozen['history']):
        for name, counts in row_frz['wakes'].items():
            assert counts['eligibility']['truncation_policy'] == 'freeze'
            assert counts['eligibility']['truncation_nodes'] == 0
        assert row_adv['CL'] == row_frz['CL']
    with pytest.raises(ValueError, match="truncation_policy must be"):
        solve_coupled_material_wake(mesh, 5., iterations=1, truncation_policy='maybe')
