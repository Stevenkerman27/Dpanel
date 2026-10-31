"""P0 profiling counters: inert by default, exact when enabled, value-neutral labels."""
from pathlib import Path
from types import SimpleNamespace as NS
import json

import numpy as np

from panel import velocity_profile
from panel.p1_coupled_material_wake import TotalVelocity, limits


def small_field():
    xyz = np.array([[0., 0., 0.], [1., .1, .2], [.2, .9, -.1]])
    faces = np.array([[0, 1, 2]])
    group = NS(name='wake', wake_xyz=xyz+[1.4, -.1, .3], wake_faces=faces,
               gamma_hat=np.array([[1., 0.], [.4, .6], [0., 1.]]))
    mesh = NS(xyz=xyz, faces=faces, dof_face=faces, groups=(group,))
    state = dict(mu=np.array([.2, -.1, .4]), gamma=np.array([.3, -.2]), alpha_deg=5.)
    return mesh, state


def fan_field(order=8, strips=2):
    mesh, state = small_field()
    field = TotalVelocity(mesh, state, method='rm_regularized_p1', core_radius=.1,
                          area_quadrature='projection_fan', fan_order=order,
                          fan_strips=strips)
    return field


def test_counters_are_inert_by_default():
    field = fan_field()
    velocity_profile.enable(False)
    velocity_profile.reset()
    field(np.array([.3, -.4, 1.7]))
    assert velocity_profile.snapshot() == {'counters': {}, 'seconds': {}}


def test_enabled_counters_report_targets_triangles_and_stages():
    field = fan_field()
    points = np.array([[.3, -.4, 1.7], [.1, .2, 1.1]])
    velocity_profile.enable(False)
    velocity_profile.reset()
    velocity_profile.enable(True)
    try:
        values = field.evaluate_many(points, order=8)
    finally:
        velocity_profile.enable(False)
    snap = velocity_profile.snapshot()
    counters, seconds = snap['counters'], snap['seconds']
    assert np.isfinite(values).all()
    assert counters['field.evaluate_many.calls'] == 1
    assert counters['field.evaluate_many.targets'] == 2
    assert counters['field.evaluate_many.fan_targets'] == 2
    assert counters['field.edge_velocity.calls'] == 2
    assert counters['field.area_velocity.calls'] == 2
    assert counters['field.area_velocity.fan_calls'] == 2
    assert counters.get('field.area_velocity.legacy_calls', 0) == 0
    assert counters.get('field.call.count', 0) == 0
    assert counters['total.evaluate_many.calls'] == 1
    assert counters['total.evaluate_many.targets'] == 2
    assert counters['total.evaluate_many.method.rm_regularized_p1'] == 1
    assert counters['area_terms.calls'] == 2
    # wall + wake triangle per target, one area_terms call per target
    assert counters['area_terms.triangles'] == 2*counters['area_terms.calls']
    assert counters['area_terms.node_sets'] == counters['area_terms.calls']
    assert counters['area_terms.subtriangles'] == counters['fan.subtriangles']
    assert counters['area_terms.subtriangles'] >= counters['fan.calls']
    assert counters['fan.calls'] >= counters['area_terms.calls']
    assert sum(v for k, v in counters.items() if k.startswith('fan.region.')) == counters['fan.calls']
    assert seconds['area_terms.kernel'] > 0.0
    assert seconds['area_terms.projection'] >= 0.0
    assert seconds['area_terms.nodes'] >= 0.0


def test_limits_tag_is_value_neutral_and_counted():
    field = fan_field()
    points = np.array([[.3, -.4, 1.7]])
    normals = np.array([[0., 0., 1.]])
    scales = np.array([.5])
    kwargs = dict(eps_fraction=.05, ladder=[1., .5], order=8)
    velocity_profile.enable(False)
    plain = limits(field, points, normals, scales, **kwargs)
    velocity_profile.reset()
    velocity_profile.enable(True)
    try:
        tagged = limits(field, points, normals, scales, tag='wake_face', **kwargs)
    finally:
        velocity_profile.enable(False)
    for key in plain:
        np.testing.assert_array_equal(plain[key], tagged[key])
    counters = velocity_profile.snapshot()['counters']
    assert counters['limits.calls'] == 1
    assert counters['limits.points'] == 1
    assert counters['limits.tag.wake_face.calls'] == 1
    assert counters['limits.tag.wake_face.points'] == 1
    assert counters['limits.ladder_steps'] == 2
    assert counters['limits.targets'] == 4
    assert counters['limits.tag.wake_face.targets'] == 4
    assert counters['limits.refined_calls'] == 1
    assert counters['limits.tag.wake_face.refined_targets'] == 2


def test_peak_rss_and_json_report_are_finite():
    peak = velocity_profile.peak_rss_mb()
    assert peak is not None and np.isfinite(peak) and peak > 0.0
    field = fan_field()
    velocity_profile.enable(False)
    velocity_profile.reset()
    velocity_profile.enable(True)
    try:
        field(np.array([.3, -.4, 1.7]))
    finally:
        velocity_profile.enable(False)
    out = Path(__file__).resolve().parents[1]/'runs'/'velocity_profile_tests'/'profile.json'
    velocity_profile.write(out, status='test', wall_seconds=0.0)
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['status'] == 'test'
    assert payload['peak_rss_mb'] > 0.0
    assert payload['counters']['area_terms.calls'] == 1
    assert isinstance(payload['seconds']['area_terms.kernel'], float)


def test_disable_stops_counting_without_resetting():
    field = fan_field()
    velocity_profile.enable(False)
    velocity_profile.reset()
    velocity_profile.enable(True)
    try:
        field(np.array([.3, -.4, 1.7]))
    finally:
        velocity_profile.enable(False)
    before = velocity_profile.snapshot()['counters']['area_terms.calls']
    field(np.array([.3, -.4, 1.7]))
    assert velocity_profile.snapshot()['counters']['area_terms.calls'] == before
    assert velocity_profile.enable(True) is False
    assert velocity_profile.enable(False) is True
