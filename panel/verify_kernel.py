"""Independent quadrature and finite-difference checks for the JAX panel kernel."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "8"
import numpy as np
import jax
import jax.numpy as jnp
from kernel import triangle_potential

jax.config.update("jax_enable_x64", True)
triangles = np.array([[[0., 0., 0.], [1.1, .1, 0.], [.2, .9, 0.]]])
targets = np.array([[.31, .24, .36], [.45, .2, -.55], [1.4, -.3, .6]])
nodes, weights = np.polynomial.legendre.leggauss(48)
nodes, weights = (nodes + 1) / 2, weights / 2
u, v = np.meshgrid(nodes, nodes, indexing="ij")
wu, wv = np.meshgrid(weights, weights, indexing="ij")
sample = triangles[0, 0] + u[:, :, None] * (triangles[0, 1] - triangles[0, 0])
sample += ((1 - u) * v)[:, :, None] * (triangles[0, 2] - triangles[0, 0])
normal_area = np.cross(triangles[0, 1] - triangles[0, 0], triangles[0, 2] - triangles[0, 0])
measure = wu * wv * (1 - u) * np.linalg.norm(normal_area)
normal = normal_area / np.linalg.norm(normal_area)
r = targets[:, None, None, :] - sample[None]
radius = np.linalg.norm(r, axis=3)
source_quad = np.sum(measure[None] / radius, axis=(1, 2)) / (4 * np.pi)
doublet_quad = np.sum(measure[None] * np.sum(r * normal, axis=3) / radius**3, axis=(1, 2)) / (4 * np.pi)
moment_quad = np.sum(measure[None] * (sample[None, :, :, 1] - triangles.mean(axis=1)[0, 1])
                     * np.sum(r * normal, axis=3) / radius**3, axis=(1, 2)) / (4 * np.pi)
single, doublet = map(np.asarray, triangle_potential(jnp.asarray(targets), jnp.asarray(triangles)))
wd, moment = map(np.asarray, triangle_potential(jnp.asarray(targets), jnp.asarray(triangles), source=False, linear_y=True))
checks = [
    dict(test="single_vs_48x48_quadrature", error=float(np.max(abs(single[:, 0] - source_quad)))),
    dict(test="doublet_vs_48x48_quadrature", error=float(np.max(abs(doublet[:, 0] - doublet_quad)))),
    dict(test="linear_doublet_moment_vs_48x48_quadrature", error=float(np.max(abs(moment[:, 0] - moment_quad)))),
]
for row in checks:
    row["passed"] = row["error"] < 1e-12
center = triangles.mean(axis=1)
s, d = triangle_potential(jnp.asarray(center), jnp.asarray(triangles), interior_diagonal=True)
checks.append(dict(test="analytic_self_trace", value=float(d[0, 0]), passed=bool(d[0, 0] == -.5)))
for kind in ("single", "doublet", "linear_moment"):
    def evaluate(scale):
        tri = jnp.asarray(triangles).at[:, :, 1].multiply(scale)
        if kind == "linear_moment":
            return triangle_potential(jnp.asarray(targets), tri, source=False, linear_y=True)[1].sum()
        return triangle_potential(jnp.asarray(targets), tri)[0 if kind == "single" else 1].sum()
    ad = float(jax.grad(evaluate)(1.))
    fd = float((evaluate(1. + 1e-5) - evaluate(1. - 1e-5)) / 2e-5)
    checks.append(dict(test=kind + "_geometry_derivative", ad=ad, fd=fd, error=abs(ad - fd),
                       passed=bool(abs(ad - fd) < 1e-9)))
for label, rotation in (
        ("tilted", np.array([[1., 0., 0.], [0., np.cos(.4), -np.sin(.4)], [0., np.sin(.4), np.cos(.4)]])),
        ("general", np.linalg.qr(np.array([[1., .4, -.2], [.3, 1., .5], [.7, -.2, 1.]]))[0])):
    tri = triangles @ rotation.T + [.2, -.4, .7]
    target = targets @ rotation.T + [.2, -.4, .7]
    point = sample @ rotation.T + [.2, -.4, .7]
    n = normal @ rotation.T
    r = target[:, None, None] - point[None]
    rad = np.linalg.norm(r, axis=3)
    for direction in (np.array([[0., 1., 0.]]), np.array([[.3, -.6, .8]])):
        density = np.sum((point - tri.mean(axis=1)[0]) * direction[0], axis=2)
        nr = r @ n
        quad = np.sum(measure[None] * density[None] * nr / rad**3, axis=(1, 2)) / (4 * np.pi)
        gradient_quad = np.sum((measure * density)[None, :, :, None] * (
            n / rad[:, :, :, None]**3 - 3 * r * nr[:, :, :, None] / rad[:, :, :, None]**5), axis=(1, 2)) / (4 * np.pi)
        def potential_at(p):
            return triangle_potential(p, jnp.asarray(tri), source=False, linear_direction=jnp.asarray(direction))[1].sum()
        value = np.asarray(triangle_potential(jnp.asarray(target), jnp.asarray(tri), source=False,
                                             linear_direction=jnp.asarray(direction))[1])[:, 0]
        gradient_ad = np.asarray(jax.grad(potential_at)(jnp.asarray(target)))
        error = max(float(np.max(abs(value - quad))), float(np.max(abs(gradient_ad - gradient_quad))))
        checks.append(dict(test=label + "_material_moment_and_velocity", direction=direction.tolist(),
                           error=error, passed=error < 1e-12))
(args.out / "checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(checks, indent=2), flush=True)
assert all(row["passed"] for row in checks)
