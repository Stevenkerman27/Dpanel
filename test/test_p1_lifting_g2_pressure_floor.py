"""Acceptance rule for the floored G2 pressure-Kutta least-squares stop.

Measured on runs/half_cad_wingbody_v1 (872 wall rows, 487 null(J) DOFs, 5 deg):
least_squares stops with status 0 / success False / nfev 80 while the residual is
already stationary (relative projected gradient 5e-7), the Kutta rows are satisfied
to 9.0e-7 and the wall residual sits at its linear floor (1.2888e-3 vs 1.2688e-3).
These tests pin the classification rule, not the iteration budget.
"""
import numpy as np
import pytest

from panel.p1_lifting_g2 import pressure_kutta_acceptance


class _Stop:
    def __init__(self, success, status, nfev, message='stop'):
        self.success = success
        self.status = status
        self.nfev = nfev
        self.message = message


def _system(rows=872, cols=487, seed=7):
    rng = np.random.default_rng(seed)
    op = rng.standard_normal((rows, cols))
    rhs = rng.standard_normal(rows)
    z_floor, _, _, _ = np.linalg.lstsq(op, rhs, rcond=None)
    floor = np.linalg.norm(op @ z_floor - rhs) / np.sqrt(rows)
    return op, rhs, z_floor, floor


def _evaluate(op, rhs, z, pressure_value=1e-7):
    def fun(v):
        return np.r_[op @ v - rhs, np.array([pressure_value])]
    def jac(v):
        return np.vstack((op, np.zeros((1, op.shape[1]))))
    return fun, jac


def test_accepts_floor_stop_without_claiming_plain_success():
    op, rhs, z, floor = _system()
    fun, jac = _evaluate(op, rhs, z)
    out = pressure_kutta_acceptance(result=_Stop(False, 0, 80), fun=fun, jac=jac, z=z,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-7]))
    assert out['success'] is True
    assert out['accepted_at_residual_floor'] is True
    assert out['wall_over_floor'] == pytest.approx(1.0, abs=1e-12)
    assert out['pressure_rms'] == pytest.approx(1e-7)
    assert out['criteria'] == {'stationary': True, 'wall_at_floor': True, 'kutta': True}
    assert out['status'] == 0 and out['nfev'] == 80


def test_rejects_unmet_kutta_rows():
    op, rhs, z, floor = _system()
    fun, jac = _evaluate(op, rhs, z)
    out = pressure_kutta_acceptance(result=_Stop(False, 0, 80), fun=fun, jac=jac, z=z,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-2]))
    assert out['success'] is False
    assert out['criteria']['kutta'] is False


def test_rejects_wall_residual_above_floor():
    op, rhs, z, floor = _system()
    fun, jac = _evaluate(op, rhs, z)
    z_far = z + 3.0 * np.std(z) * np.ones_like(z)
    out = pressure_kutta_acceptance(result=_Stop(False, 0, 80), fun=fun, jac=jac, z=z_far,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-7]))
    assert out['success'] is False
    assert out['criteria']['wall_at_floor'] is False


def test_rejects_non_stationary_stop():
    op, rhs, z, floor = _system()
    def fun(v):
        return np.r_[op @ v - rhs, np.array([1e-2])]
    def jac(v):
        return np.vstack((op, (1e3 * np.ones((1, op.shape[1])))))
    out = pressure_kutta_acceptance(result=_Stop(False, 0, 80), fun=fun, jac=jac, z=z,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-7]))
    assert out['criteria']['stationary'] is False
    assert out['success'] is False


def test_plain_success_is_preserved():
    op, rhs, z, floor = _system()
    fun, jac = _evaluate(op, rhs, z)
    out = pressure_kutta_acceptance(result=_Stop(True, 2, 85), fun=fun, jac=jac, z=z,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-7]))
    assert out['success'] is True
    assert out['accepted_at_residual_floor'] is False


def test_status_one_configuration_stop_is_still_a_failure():
    op, rhs, z, floor = _system()
    fun, jac = _evaluate(op, rhs, z)
    out = pressure_kutta_acceptance(result=_Stop(False, 1, 80), fun=fun, jac=jac, z=z,
                                    wall_operator=op, wall_rhs=rhs,
                                    pressure_residual=np.array([1e-7]))
    assert out['success'] is False


def test_synthetic_fixture_still_converges_normally():
    from panel.p1_lifting_g2 import make_conventional_fixture, assemble_multi, solve_multi
    wing = make_conventional_fixture(main=(3, 4, 2.5, 1., .015), tail=(3, 4, 1.4, .5, 0.),
                                     tail_origin=(1.2, 0., .4), wake_panels=1)
    ops = assemble_multi(wing)
    result = solve_multi(wing, 5., kutta='pressure', operators=ops, max_nfev=50)
    assert result['nonlinear']['success'] is True
    assert result['pressure_kutta_rms'] < 1e-3


CAD = 'runs/half_cad_wingbody_v1/surface.json'


def test_cad_fixture_floor_stop_is_accepted():
    import os
    from pathlib import Path
    from panel.p1_lifting_g1 import make_wing, from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings, solve_multi
    root = Path(os.environ.get('PANEL_ROOT', Path(__file__).resolve().parents[1]))
    fixture = root / CAD
    if not fixture.is_file():
        pytest.skip('CAD fixture absent: ' + str(fixture))
    main = from_dpanel_half_surface(fixture)
    tail = make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0.,
                     wake_panels=3)
    mesh = combine_wings([main, tail], names=['main_wing', 'raised_tail'],
                         origins=[(0., 0., 0.), (1.6, 0., .35)],
                         reference_chord=main.chord, reference_point=main.reference_point)
    state = solve_multi(mesh, 5., kutta='pressure', max_nfev=80)
    nl = state['nonlinear']
    assert nl['success'] is True
    assert nl['accepted_at_residual_floor'] is True
    assert nl['pressure_rms'] < 1e-4
    assert nl['wall_over_floor'] <= 1.05
