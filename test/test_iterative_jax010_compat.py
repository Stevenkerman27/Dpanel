"""JAX 0.10 CPU compatibility path (forced on JAX 0.9 for CI-independent coverage).

Tests actual small lifting-panel operator, not a toy dense replacement. The
fallback must preserve full A/A.T, an arbitrary adjoint RHS, and geometry VJP.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import panel.iterative as iterative
from panel.differentiable import solve_fields
from iterative_fixture import wing_inputs

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("symmetry", [False, True])
def test_cpu_compatibility_gmres_state_and_adjoint(monkeypatch, symmetry):
    # Exercise this branch even if installed JAX still exposes custom_transpose.
    monkeypatch.setattr(iterative, "custom_transpose", None)
    inp = wing_inputs()
    plan = iterative.make_plan(
        inp["vertices"], inp["station_y"], inp["topology"], inp["freestream"],
        block_size=7, preconditioner_block_size=8, symmetry=symmetry,
    )
    op = iterative.PanelOperator(
        inp["vertices"], inp["station_y"], inp["topology"], inp["freestream"], plan
    )
    options = dict(
        tolerance=1e-10, absolute_tolerance=1e-12, restart=25,
        max_cycles=16, preconditioned=True,
    )
    b = op.rhs()
    got = jax.jit(lambda rhs: iterative.solve_state(op, rhs, **options))(b)
    ref = solve_fields(**inp)
    expected = jnp.concatenate((ref["mu"], ref["gamma"]))
    # Explicitly fold the full state under the frozen orbit map.
    expected = jnp.concatenate((
        expected[:len(op.tri)][jnp.asarray(plan.body_rows)],
        expected[len(op.tri):][jnp.asarray(plan.gamma_rows)]
    ))
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(op.matvec(got), b, rtol=1e-8, atol=1e-10)

    weight = jnp.linspace(0.2, 0.8, len(b))
    adj = jax.jit(jax.grad(
        lambda rhs: jnp.dot(weight, iterative.solve_state(op, rhs, **options))
    ))(b)
    np.testing.assert_allclose(op.rmatvec(adj), weight, rtol=1e-8, atol=1e-10)


def test_cpu_compatibility_geometry_implicit_vjp(monkeypatch):
    monkeypatch.setattr(iterative, "custom_transpose", None)
    inp = wing_inputs()
    plan = iterative.make_plan(
        inp["vertices"], inp["station_y"], inp["topology"], inp["freestream"],
        block_size=7, preconditioner_block_size=8, symmetry=False,
    )
    options = dict(
        tolerance=1e-10, absolute_tolerance=1e-12, restart=25,
        max_cycles=16, preconditioned=True,
    )
    base = inp["vertices"]
    y, t, flow = inp["station_y"], inp["topology"], inp["freestream"]
    seed = jnp.linspace(-0.25, 0.5, plan.shape[0])

    def loss(scale):
        vertices = base.at[:, 0].multiply(1 + 0.01 * scale)
        op = iterative.PanelOperator(vertices, y, t, flow, plan)
        return jnp.vdot(seed, iterative.solve_state(op, op.rhs(), **options))

    grad = jax.jit(jax.grad(loss))(jnp.array(0.))
    assert np.isfinite(float(grad))
    h = 1e-4
    finite_difference = float((loss(h) - loss(-h)) / (2*h))
    np.testing.assert_allclose(grad, finite_difference, rtol=2e-5, atol=1e-8)
