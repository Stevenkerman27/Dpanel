"""Acceptance checks for the small G0 sphere discretisation (issue #56, part D).

`panel.p1_dirichlet` already answers "how close is the surface Cp"; this module asks the two
questions a Dirichlet doublet formulation has to answer before it can be called well posed:

  1. does the discrete layer reproduce the *off-surface* potential at several radial offsets, and
  2. does the reconstructed surface velocity satisfy no penetration and a convergent load?

Everything is measured in float64 against the analytic sphere, and every routine returns its
inputs alongside the errors so a report can be regenerated instead of quoted.
"""

from __future__ import annotations

import numpy as np

from panel.p1_dirichlet import assemble_p1_doublet, face_cp_from_mu, icosphere, solve_p1_dirichlet
from panel.p1_doublet_moment import triangle_p1_doublet_potentials

__all__ = [
    "analytic_potential",
    "layer_potential",
    "offset_potential_check",
    "load_check",
    "penetration_check",
    "jump_check",
]


def analytic_potential(points, *, radius=1.0, freestream=(1.0, 0.0, 0.0)):
    """Uniform flow past a sphere: ``phi = U . x * (1 + a^3 / (2 r^3))``."""
    xyz = np.asarray(points, dtype=float)
    stream = np.asarray(freestream, dtype=float)
    r = np.linalg.norm(xyz, axis=1)
    if np.any(r < radius * 0.5):
        raise ValueError("analytic exterior potential needs points outside the sphere")
    return (xyz @ stream) * (1.0 + radius ** 3 / (2.0 * r ** 3))


def layer_potential(vertices, faces, mu, targets):
    """Doublet layer potential of a nodal density at off-surface targets."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    targets = np.asarray(targets, dtype=float)
    potentials = triangle_p1_doublet_potentials(targets, vertices[faces], max_pairs=10 ** 9)
    weights = np.asarray(mu, dtype=float)[faces]
    return np.einsum("tpk,pk->t", potentials, weights)


def offset_potential_check(level=2, *, radius=1.0, ratios=(1.2, 1.5, 2.0, 3.0), speed=1.0):
    """Compare the discrete layer with the analytic potential at several radial offsets."""
    stream = np.array([speed, 0.0, 0.0])
    vertices, faces = icosphere(level)
    vertices = vertices * radius
    solution = solve_p1_dirichlet(vertices, faces, stream)
    mu = solution["mu"]
    rows = []
    for ratio in ratios:
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0], [1.0, 0.0, -1.0]])
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        points = directions * (ratio * radius)
        exact = analytic_potential(points, radius=radius, freestream=stream)
        discrete = layer_potential(vertices, faces, mu, points) + points @ stream
        rows.append({
            "ratio": float(ratio),
            "max_relative_error": float(np.max(np.abs(discrete - exact)) / max(np.max(np.abs(exact)), 1e-30)),
            "max_absolute_error": float(np.max(np.abs(discrete - exact))),
            "exact_scale": float(np.max(np.abs(exact))),
        })
    return {"level": int(level), "radius": float(radius), "rows": rows}


def load_check(level=2, *, radius=1.0, speed=1.0):
    """Integrate the surface Cp and compare the drag with d Alembert (it must vanish)."""
    stream = np.array([speed, 0.0, 0.0])
    vertices, faces = icosphere(level)
    vertices = vertices * radius
    solution = solve_p1_dirichlet(vertices, faces, stream)
    cp = face_cp_from_mu(vertices, faces, solution["mu"], speed=speed)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    unit = normals / np.linalg.norm(normals, axis=1)[:, None]
    force = -np.sum((cp * areas)[:, None] * unit, axis=0)
    reference = 0.5 * speed ** 2 * 4.0 * np.pi * radius ** 2
    return {
        "level": int(level),
        "drag_coefficient": float(abs(force[0]) / reference),
        "lift_coefficient": float(abs(force[2]) / reference),
        "lateral_coefficient": float(abs(force[1]) / reference),
        "cp_rms": float(np.sqrt(np.mean(cp ** 2))),
        "area_total": float(areas.sum()),
        "area_exact": float(4.0 * np.pi * radius ** 2),
    }


def penetration_check(level=2, *, radius=1.0, speed=1.0, radii=(1.02, 1.05, 1.10), delta=1e-3, max_targets=64):
    """Radial velocity in the fluid, compared with the analytic sphere at the same points.

    No penetration is a statement about the surface only, so the acceptance test is the error of
    the discrete radial derivative against the analytic one, sampled on a ring of offsets.
    """
    stream = np.array([speed, 0.0, 0.0])
    vertices, faces = icosphere(level)
    vertices = vertices * radius
    solution = solve_p1_dirichlet(vertices, faces, stream)
    mu = solution["mu"]
    assembly = assemble_p1_doublet(vertices, faces)
    directions = assembly["centroids"] / np.linalg.norm(assembly["centroids"], axis=1)[:, None]
    stride = max(1, int(np.ceil(len(directions) / int(max_targets))))
    directions = directions[::stride]
    rows = []
    for ratio in radii:
        base = directions * (ratio * radius)
        inner = base - delta * directions
        outer = base + delta * directions
        def discrete(points):
            return layer_potential(vertices, faces, mu, points) + points @ stream
        velocity = (discrete(outer) - discrete(inner)) / (2.0 * delta)
        exact = (analytic_potential(outer, radius=radius, freestream=stream)
                 - analytic_potential(inner, radius=radius, freestream=stream)) / (2.0 * delta)
        rows.append({
            "ratio": float(ratio),
            "max_error_over_speed": float(np.max(np.abs(velocity - exact)) / speed),
            "rms_error_over_speed": float(np.sqrt(np.mean((velocity - exact) ** 2)) / speed),
            "max_exact_over_speed": float(np.max(np.abs(exact)) / speed),
        })
    return {"level": int(level), "radius": float(radius), "delta": float(delta), "rows": rows}

def jump_check(level=2, *, radius=1.0, speed=1.0, offset=1e-4):
    """The potential jump across the surface must equal the local doublet density.

    The measured error is ``O(offset)``: the panel's own contribution jumps exactly by its
    density, while the (continuous) neighbour contributions differ by ``offset * grad``. Dividing
    the reported error by ``offset`` recovers a constant, which is the useful statement.
    """
    stream = np.array([speed, 0.0, 0.0])
    vertices, faces = icosphere(level)
    vertices = vertices * radius
    solution = solve_p1_dirichlet(vertices, faces, stream)
    mu = solution["mu"]
    assembly = assemble_p1_doublet(vertices, faces)
    centroids = assembly["centroids"]
    normals = assembly["normals"]
    plus = layer_potential(vertices, faces, mu, centroids + offset * normals)
    minus = layer_potential(vertices, faces, mu, centroids - offset * normals)
    local = np.mean(mu[faces], axis=1)
    return {"level": int(level),
            "jump_max_error": float(np.max(np.abs((plus - minus) - local))),
            "jump_rms_error": float(np.sqrt(np.mean(((plus - minus) - local) ** 2))),
            "mu_scale": float(np.max(np.abs(local))),
            "offset": float(offset)}