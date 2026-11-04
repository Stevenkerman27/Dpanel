"""Execution-backend tests for the analytic wake evaluator (issue #92, Phase 4).

The analytic closed form used to be evaluated one target per Python call. These tests
pin the three interchangeable backends:

* 'loop'   - the original per-target reference path, still the default;
* 'numpy'  - the same closed form batched over a block of targets;
* 'torch'  - the batched kernel on an explicit device (CPU or CUDA).

What is asserted is that the backend never changes a number: the batched numpy path must
be bitwise identical to the loop, the gate the coupled solver runs must be unchanged,
and an option that cannot be honoured must raise instead of falling back.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from panel.p1_coupled_material_wake import TotalVelocity, limits
from panel.p1_projection_duffy import torch_available


def small_field():
    """One wall triangle plus one wake triangle: small, but the full analytic path."""
    xyz = np.array([[0., 0., 0.], [1., .1, .2], [.2, .9, -.1]])
    faces = np.array([[0, 1, 2]])
    group = NS(name='wake', wake_xyz=xyz+[1.4, -.1, .3], wake_faces=faces,
               gamma_hat=np.array([[1., 0.], [.4, .6], [0., 1.]]))
    mesh = NS(xyz=xyz, faces=faces, dof_face=faces, groups=(group,))
    state = dict(mu=np.array([.2, -.1, .4]), gamma=np.array([.3, -.2]), alpha_deg=5.)
    return mesh, state


def sample_points(count=9):
    rng = np.random.default_rng(92)
    return rng.uniform([-0.5, -0.5, 0.15], [2.0, 1.5, 1.2], size=(count, 3))


def test_batched_numpy_is_bitwise_identical_to_the_loop():
    mesh, state = small_field()
    loop = TotalVelocity(mesh, state)
    batched = TotalVelocity(mesh, state, analytic_backend='numpy', analytic_chunk=4)
    points = sample_points()
    expected = np.stack([loop(p) for p in points])
    got = batched.evaluate_many(points)
    assert np.array_equal(got, expected)


def test_chunk_size_never_changes_the_result():
    mesh, state = small_field()
    field = TotalVelocity(mesh, state, analytic_backend='numpy')
    points = sample_points(11)
    reference = field.evaluate_many(points, )
    for chunk in (1, 2, 5, 64):
        field.analytic_chunk = chunk
        assert np.array_equal(field.evaluate_many(points), reference)


def test_batched_backend_keeps_the_sampling_gate_unchanged():
    mesh, state = small_field()
    points, normals = sample_points(4), np.tile([0., 0., 1.], (4, 1))
    scales = np.full(4, .3)
    kwargs = dict(eps_fraction=.05, ladder=(1., .5, .25), order=48, tag='backend')
    loop = limits(TotalVelocity(mesh, state), points, normals, scales, **kwargs)
    batched = limits(TotalVelocity(mesh, state, analytic_backend='numpy', analytic_chunk=2),
                     points, normals, scales, **kwargs)
    for key in ('plus', 'minus', 'mean', 'epsilon_error', 'quadrature_error'):
        assert np.array_equal(loop[key], batched[key]), key


def test_single_target_call_honours_the_requested_backend():
    mesh, state = small_field()
    point = np.array([.3, -.4, 1.7])
    reference = TotalVelocity(mesh, state)(point)
    for backend in ('numpy',) + (('torch',) if torch_available() else ()):
        field = TotalVelocity(mesh, state, analytic_backend=backend)
        np.testing.assert_allclose(field(point), reference, atol=1e-14, rtol=1e-14)


def test_batched_evaluator_rejects_invalid_points_and_chunks():
    mesh, state = small_field()
    field = TotalVelocity(mesh, state, analytic_backend='numpy')
    with pytest.raises(ValueError):
        field.evaluate_many(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        field.evaluate_many(np.array([[0., 0., np.nan]]))
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            TotalVelocity(mesh, state, analytic_backend='numpy', analytic_chunk=bad)


def test_analytic_options_fail_closed():
    mesh, state = small_field()
    with pytest.raises(ValueError):
        TotalVelocity(mesh, state, analytic_backend='cuda')
    with pytest.raises(ValueError):
        TotalVelocity(mesh, state, analytic_device='gpu')
    with pytest.raises(ValueError):
        TotalVelocity(mesh, state, analytic_device='cuda')  # only torch can be cuda
    with pytest.raises(ValueError):
        TotalVelocity(mesh, state, method='rm_regularized_p1', core_radius=.1,
                      analytic_backend='numpy')
    with pytest.raises(ValueError):
        TotalVelocity(mesh, state, method='legacy_gauss', analytic_backend='torch')


def test_default_backend_is_still_the_loop():
    mesh, state = small_field()
    field = TotalVelocity(mesh, state)
    assert field.analytic_backend == 'loop'
    assert field.analytic_device == 'cpu'
    assert field._torch_analytic is None


@pytest.mark.skipif(not torch_available(), reason='torch is not importable here')
def test_torch_backend_matches_the_reference():
    mesh, state = small_field()
    points = sample_points(6)
    reference = TotalVelocity(mesh, state).evaluate_many(points)
    field = TotalVelocity(mesh, state, analytic_backend='torch', analytic_device='cpu')
    np.testing.assert_allclose(field.evaluate_many(points), reference, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(field(points[0]), reference[0], atol=1e-13, rtol=1e-13)


@pytest.mark.skipif(not torch_available(), reason='torch is not importable here')
def test_cuda_request_without_a_device_raises():
    import torch
    if torch.cuda.is_available():
        pytest.skip('a CUDA device is present, so the refusal cannot be observed')
    mesh, state = small_field()
    with pytest.raises(RuntimeError):
        TotalVelocity(mesh, state, analytic_backend='torch', analytic_device='cuda')


def _wing_case():
    """A complete wing with a wake: what the coupled solver actually iterates."""
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    return combine_wings([make_wing(chord_panels=4, span_panels=4, wake_panels=2)])


def _solve(backend, chunk=2):
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    captured = {}

    def save(iteration, mesh, state, record, arrays):
        captured['arrays'] = dict(arrays)
        captured['record'] = record

    result = solve_coupled_material_wake(
        _wing_case(), 5., kutta='signed_speed', iterations=1,
        analytic_backend=backend, analytic_chunk=chunk, save_callback=save)
    return result, captured


def test_coupled_solve_with_the_batched_backend_is_bitwise_identical():
    """The backend is an execution choice: every sampled number must be unchanged."""
    reference, ref_capture = _solve('loop')
    batched, bat_capture = _solve('numpy')
    assert reference['history'][0]['analytic_backend'] == 'loop'
    assert batched['history'][0]['analytic_backend'] == 'numpy'
    for name in ('CL', 'CL_gamma', 'CD_pressure', 'CM_y', 'wall_rms', 'jump_rms'):
        assert reference['history'][0][name] == batched['history'][0][name], name
    for name in ('epsilon_error', 'quadrature_error', 'mean', 'plus', 'minus'):
        assert np.array_equal(ref_capture['arrays']['g0_node_'+name],
                              bat_capture['arrays']['g0_node_'+name]), name
    assert str(ref_capture['arrays']['sampling_analytic_backend']) == 'loop'
    assert str(bat_capture['arrays']['sampling_analytic_backend']) == 'numpy'


def test_solve_rejects_a_backend_it_cannot_use():
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    with pytest.raises(ValueError, match="analytic_backend"):
        solve_coupled_material_wake(_wing_case(), 5., iterations=0, analytic_backend='quantum')
    with pytest.raises(ValueError, match="analytic"):
        solve_coupled_material_wake(_wing_case(), 5., iterations=0,
                                    velocity_method='rm_regularized_p1',
                                    core_radius=.01, analytic_backend='numpy')
