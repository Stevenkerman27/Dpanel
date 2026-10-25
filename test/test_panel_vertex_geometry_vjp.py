"""Fixed-topology vertex VJP through Dpanel's own JAX panel kernel and solve.

This deliberately checks a closed NON-LIFTING octahedron and two external
potential samples. It establishes the geometry -> AIC/RHS -> linear solution
-> physical potential derivative path. It does NOT validate Cp, lifting wake,
wing-body loads, CAD Boolean operations, or the current panel/solve.py CLI.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from panel.kernel import triangle_potential

jax.config.update("jax_enable_x64", True)

VERTICES = jnp.array([
    [1., 0., 0.], [-1., 0., 0.],
    [0., 1., 0.], [0., -1., 0.],
    [0., 0., 1.], [0., 0., -1.],
], dtype=jnp.float64)
FACES = jnp.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
], dtype=jnp.int32)
FREESTREAM = jnp.array([1., 0.13, -0.05], dtype=jnp.float64)
FIELD_POINTS = jnp.array([[2., 1.1, 0.8], [-1.9, 0.6, 1.8]], dtype=jnp.float64)
FIELD_SEED = jnp.array([0.25, -0.6], dtype=jnp.float64)
DIRECTION = jnp.array([
    [0.33, 0.2, -0.1], [0., 0., 0.], [0., 0., 0.],
    [-0.1, 0.15, 0.32], [0.13, -0.21, 0.1], [0., 0., 0.],
], dtype=jnp.float64)


def exterior_perturbation_potentials(vertices):
    """Do not freeze vertices in NumPy: they must be JAX traced inputs."""
    tri = vertices[FACES]
    twice_area_normals = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = twice_area_normals / jnp.linalg.norm(twice_area_normals, axis=-1, keepdims=True)
    sigma = -(normals @ FREESTREAM)
    B, D = triangle_potential(tri.mean(axis=1), tri, interior_diagonal=True)
    mu = jnp.linalg.solve(D, B @ sigma)
    evaluation_B, evaluation_D = triangle_potential(FIELD_POINTS, tri)
    return evaluation_D @ mu - evaluation_B @ sigma


def test_nonlifting_vertex_vjp_is_nonzero_and_matches_three_finite_differences():
    values, pullback = jax.vjp(exterior_perturbation_potentials, VERTICES)
    (gradient_vertices,) = pullback(FIELD_SEED)
    assert values.shape == (2,)
    assert gradient_vertices.shape == VERTICES.shape
    assert np.isfinite(np.asarray(gradient_vertices)).all()
    assert float(jnp.linalg.norm(gradient_vertices)) > 1e-6

    directional_ad = float(jnp.sum(gradient_vertices * DIRECTION))
    assert abs(directional_ad) > 1e-6
    objective = lambda x: jnp.dot(FIELD_SEED, exterior_perturbation_potentials(x))
    for h in (1e-3, 1e-4, 1e-5):
        fd = float((objective(VERTICES + h*DIRECTION) - objective(VERTICES - h*DIRECTION)) / (2*h))
        np.testing.assert_allclose(directional_ad, fd, rtol=5e-6, atol=1e-10)


def test_vertex_vjp_composes_with_geometry_parameter_via_chain_rule():
    _, pullback = jax.vjp(exterior_perturbation_potentials, VERTICES)
    (gradient_vertices,) = pullback(FIELD_SEED)
    chain = float(jnp.sum(gradient_vertices * DIRECTION))
    parameter_gradient = jax.grad(
        lambda theta: jnp.dot(
            FIELD_SEED, exterior_perturbation_potentials(VERTICES + theta*DIRECTION)
        )
    )(jnp.asarray(0., dtype=jnp.float64))
    np.testing.assert_allclose(float(parameter_gradient), chain, rtol=1e-10, atol=1e-12)
