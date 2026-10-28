"""Guards that keep the steady free wake honest on a strongly anisotropic sheet."""
import numpy as np
import pytest

from panel.p1_lifting_g2 import make_conventional_fixture
from panel.p1_steady_free_wake import (
    _one_wake_target, _shortest_edges, _wake_mesh, mean_wake_velocity,
    prepare_wake_material, solve_steady_free_wake)


def test_shortest_edge_scales_the_two_sided_probe():
    model = make_conventional_fixture(wake_panels=2)
    g = model.groups[0]
    wm = _wake_mesh(g)
    edge = _shortest_edges(wm)
    assert edge.shape == (len(wm.faces),)
    area_edge = wm.triangles
    manual = np.min(np.stack([np.linalg.norm(area_edge[:, 1] - area_edge[:, 0], axis=1),
                              np.linalg.norm(area_edge[:, 2] - area_edge[:, 1], axis=1),
                              np.linalg.norm(area_edge[:, 0] - area_edge[:, 2], axis=1)]), axis=0)
    np.testing.assert_allclose(edge, manual, atol=1e-15)
    fake = dict(alpha_deg=0., mu=np.zeros(model.ndof_mu),
                gamma=np.zeros(sum(g.gamma_hat.shape[1] for g in model.groups)))
    _, base = mean_wake_velocity(model, fake)
    _, floored = mean_wake_velocity(model, fake, probe_floor_fraction=2.0)
    for name in base:
        assert base[name]['probe_over_shortest_max'] > 0.0
        assert floored[name]['probe_over_shortest_max'] >= base[name]['probe_over_shortest_max']
        assert floored[name]['probe_over_shortest_max'] >= 2.0 - 1e-9
        assert floored[name]['aspect_p90'] >= 1.0


def test_axial_stagnation_freezes_the_node_and_is_reported():
    model = make_conventional_fixture(wake_panels=2)
    g = model.groups[0]
    material = prepare_wake_material(g)
    stalled = np.zeros((len(g.wake_xyz), 3))
    target, frozen = _one_wake_target(g, material, stalled, freeze=True)
    assert len(frozen) == int(np.sum(material.predecessor[np.asarray(material.order)] >= 0))
    np.testing.assert_allclose(target, g.wake_xyz, atol=0)
    with pytest.raises(ValueError, match='wake streamwise marching requires Vx>0'):
        _one_wake_target(g, material, stalled, freeze=False)


def test_step_cap_and_masks_bound_the_geometry_update():
    model = make_conventional_fixture(wake_panels=3)
    names = [g.name for g in model.groups]
    value = solve_steady_free_wake(model, 5, kutta='pressure', iterations=2,
                                   relaxation=.5, max_step_chord=.01, slip_gate=1e-9)
    for record in value['history']:
        assert record['wake_geometry_change'] <= .01 + 1e-12
        assert all(record['relaxed_nodes'][name] == 0 for name in names)
    assert value['history'][0]['CL'] == pytest.approx(value['history'][-1]['CL'], abs=1e-9)


def test_junction_mask_keeps_the_inner_strip_fixed():
    model = make_conventional_fixture(wake_panels=3)
    names = [g.name for g in model.groups]
    value = solve_steady_free_wake(model, 5, kutta='pressure', iterations=2,
                                   relaxation=.3, relax_min_abs_y=1e9)
    for record in value['history']:
        assert all(record['relaxed_nodes'][name] == 0 for name in names)
        assert record['wake_geometry_change'] == 0.0