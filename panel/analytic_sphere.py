"""Analytic sphere anchor for the panel potential-field reconstruction.

The half-wing fixtures used by ``panel.audit_reference`` have no exact
solution, so their metrics can only be compared between meshes.  This module
supplies the missing absolute anchor on a geometry where the exact
potential-flow answer is known in closed form: uniform flow past a sphere.

    phi      = -U cos(theta) (r + a^3 / (2 r^2))        (perturbation form)
    V_r      = -U cos(theta) (1 - a^3 / r^3)
    V_theta  =  U sin(theta) (1 + a^3 / (2 r^3))
    Cp       =  1 - (V_r^2 + V_theta^2) / U^2

The discretisation is the same one the production half model assembles: a
constant source sheet with sigma = -n.U_inf plus a constant doublet sheet
whose interior limit cancels the source field, i.e.

    (PV - 1/2 I) mu = S sigma,    sigma = -n.U_inf,

evaluated with the production :func:`panel.kernel.triangle_potential`.  The
sphere has no wake, so no circulation unknown and no Kutta row are involved:
this isolates the kernel, the assembly sign convention, the field
reconstruction and the pressure integral from the lifting-surface model.

Reported per mesh: Cp error statistics against the exact solution at the same
off-surface sample points, the net pressure force (exactly zero for a sphere,
so it exposes pure cancellation error) and the gross |per-face| drag.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import triangle_potential


def icosphere(subdivisions=2, radius=1.0):
    """Subdivided icosahedron: near-uniform triangles with outward winding."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    for _ in range(subdivisions):
        midpoints = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                point = 0.5 * (vertices[a] + vertices[b])
                midpoints[key] = len(vertices_list)
                vertices_list.append(point / np.linalg.norm(point))
            return midpoints[key]

        vertices_list = list(vertices)
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        vertices = np.asarray(vertices_list)
        faces = np.asarray(refined, dtype=np.int32)
    return radius * vertices, faces


def sphere_solution(faces, vertices, freestream, *, radius=1.0):
    """Morino doublet+source solve on a closed non-lifting body."""
    tri = jnp.asarray(vertices[faces])
    area_vector = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
    areas = jnp.linalg.norm(area_vector, axis=1)
    normals = area_vector / areas[:, None]
    centers = tri.mean(axis=1)
    flow = jnp.asarray(freestream)
    sigma = -(normals @ flow).reshape(-1)
    single, doublet = triangle_potential(centers, tri, interior_diagonal=True)
    mu = jnp.linalg.solve(doublet, single @ sigma)
    return {"triangles": tri, "areas": areas, "normals": normals,
            "centers": centers, "sigma": sigma, "mu": mu,
            "source_total": float(jnp.sum(sigma * areas))}


def sphere_field(state, freestream):
    """Total velocity and Cp at arbitrary points of the discrete solution."""
    tri, sigma, mu = state["triangles"], state["sigma"], state["mu"]
    flow = jnp.asarray(freestream)
    speed = float(np.linalg.norm(np.asarray(flow)))

    def potential(point):
        p = point[None]
        single, doublet = triangle_potential(p, tri)
        return (-(single @ sigma) + (doublet @ mu)).reshape(())

    gradient = jax.jit(jax.jacfwd(potential))

    def velocity(points):
        points = jnp.asarray(points, dtype=jnp.float64)
        pieces = [gradient(point) for point in points]
        return flow[None, :] + jnp.stack(pieces)

    def cp(points):
        v = velocity(points)
        return 1 - jnp.sum(v * v, axis=1) / speed ** 2

    return velocity, cp, speed


def exact_cp(points, *, freestream, radius=1.0):
    """Exact sphere Cp at the same off-surface points."""
    flow = np.asarray(freestream, dtype=float)
    speed = float(np.linalg.norm(flow)); axis = flow / speed
    r = np.linalg.norm(points, axis=1)
    cos_theta = (points @ axis) / r
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
    ratio = (radius / r) ** 3
    v_r = -speed * cos_theta * (1.0 - ratio)
    v_t = speed * sin_theta * (1.0 + 0.5 * ratio)
    return 1.0 - (v_r ** 2 + v_t ** 2) / speed ** 2


def field_error(state, *, freestream=(1.0, 0.0, 0.0), radius=1.0,
                 offsets=(0.01, 0.02, 0.05, 0.1, 0.2), density=1.0):
    """Compare the discrete field with the exact sphere along radial rays.

    Sample points are the facet centroids projected onto the exact sphere and
    pushed out by ``offset * radius``.  Projecting first guarantees every probe
    lies outside the exact sphere at every refinement (a raw facet centroid of a
    coarse icosphere is ~6% inside the sphere, where the exact exterior
    solution is not the physical answer and would pollute the statistics).
    """
    velocity_f, cp_f, speed = sphere_field(state, freestream)
    centers = np.asarray(state["centers"])
    normals = np.asarray(state["normals"])
    areas = np.asarray(state["areas"])
    rays = centers / np.linalg.norm(centers, axis=1)[:, None]
    flow = np.asarray(freestream, dtype=float)
    drag_axis = flow / np.linalg.norm(flow)
    out = {}
    for offset in offsets:
        points = rays * (radius * (1.0 + offset))
        computed = np.asarray(cp_f(points), dtype=float)
        exact = exact_cp(points, freestream=freestream, radius=radius)
        error = computed - exact
        velocity = np.asarray(velocity_f(points), dtype=float)
        flux = np.sum(velocity * rays, axis=1) / speed
        out[str(offset)] = {
            "offset_radius": offset,
            "cp_rms_error": float(np.sqrt(np.mean(error ** 2))),
            "cp_max_abs_error": float(np.max(np.abs(error))),
            "cp_mean_error": float(np.mean(error)),
            "cp_min_computed": float(np.min(computed)),
            "cp_min_exact": float(np.min(exact)),
            "cp_max_computed": float(np.max(computed)),
            "cp_max_exact": float(np.max(exact)),
            "normal_flux_rms": float(np.sqrt(np.mean(flux ** 2))),
            "normal_flux_max": float(np.max(np.abs(flux))),
        }
    # Surface pressure integration against the facet areas and normals.
    surface_points = centers + (1e-3 * radius) * normals
    cp_surface = np.asarray(cp_f(surface_points), dtype=float)
    pressure = 0.5 * density * speed ** 2 * cp_surface
    force = -np.sum((pressure * areas)[:, None] * normals, axis=0)
    q_area = 0.5 * density * speed ** 2 * areas.sum()
    out_surface = {
        "cp_min": float(np.min(cp_surface)),
        "cp_max": float(np.max(cp_surface)),
        "force_over_qA": (force / q_area).tolist(),
        "drag_over_qA": float(np.dot(force, drag_axis) / q_area),
        "sum_abs_face_drag_over_qA": float(
            np.sum(np.abs(pressure * areas * (normals @ drag_axis))) / q_area),
    }
    return out, out_surface


def audit_sphere(faces, vertices, *, freestream=(1.0, 0.0, 0.0), radius=1.0,
                 offsets=(0.01, 0.02, 0.05, 0.1, 0.2), density=1.0):
    started = time.perf_counter()
    state = sphere_solution(faces, vertices, freestream, radius=radius)
    samples, surface = field_error(state, freestream=freestream, radius=radius,
                                   offsets=offsets, density=density)
    return {
        "faces": int(len(faces)),
        "radius": radius,
        "freestream": list(freestream),
        "source_strength_total": state["source_total"],
        "samples": samples,
        "surface": surface,
        "seconds": time.perf_counter() - started,
    }


def compare_formulation(faces, vertices, *, freestream=(1.0, 0.0, 0.0),
                        radius=1.0, offsets=(0.02, 0.05)):
    """Production fixed-source assembly vs the Morino interior freestream term.

    The production half assembly closes the interior Dirichlet condition with a
    source sheet of fixed strength ``sigma = -n.U_inf``.  The classical closed-
    body (Morino) form instead requires the interior perturbation to cancel the
    freestream itself, ``D_int mu = -U_inf.x``.  The two right-hand sides are not
    algebraically equal, so this records how far the production choice is from it
    and whether the resulting near-surface error is better or worse.
    """
    state = sphere_solution(faces, vertices, freestream, radius=radius)
    tri = state["triangles"]
    centers = np.asarray(state["centers"])
    normals = np.asarray(state["normals"])
    flow = np.asarray(freestream, dtype=float)
    single, doublet = triangle_potential(centers, tri, interior_diagonal=True)
    single, doublet = np.asarray(single), np.asarray(doublet)
    sigma = np.asarray(state["sigma"])
    # Interior identity: the source sheet must reproduce -phi_inf = -U.x inside.
    inside = centers - (5e-3 * radius) * normals
    probe, _ = triangle_potential(jnp.asarray(inside), tri)
    discrete = np.asarray(probe) @ sigma
    freestream_term = inside @ flow
    mu_morino = np.linalg.solve(doublet, -freestream_term)
    velocity_f, cp_f, _ = sphere_field(state, freestream)
    del velocity_f
    rays = centers / np.linalg.norm(centers, axis=1)[:, None]
    out = {
        "interior_freestream_term_rms": float(np.sqrt(np.mean(freestream_term ** 2))),
        "interior_source_sheet_minus_freestream_rms": float(
            np.sqrt(np.mean((discrete + freestream_term) ** 2))),
        "offsets": {},
    }
    for offset in offsets:
        points = rays * (radius * (1.0 + offset))
        exact = exact_cp(points, freestream=freestream, radius=radius)

        def probe_distribution(query, mu):
            s, d = triangle_potential(jnp.asarray(query), tri)
            return -(s @ state["sigma"]) + (d @ jnp.asarray(mu))

        def morino_distribution(query, mu):
            _, d = triangle_potential(jnp.asarray(query), tri)
            return d @ jnp.asarray(mu)

        for name, function, mu in (("production", probe_distribution, state["mu"]),
                                   ("morino", morino_distribution, mu_morino)):
            def scalar(point):
                return function(point[None], mu).reshape(())

            gradient = jax.jit(jax.jacfwd(scalar))
            velocity = np.stack([np.asarray(gradient(point)) for point in points])
            velocity = velocity + flow[None]
            cp = 1.0 - np.sum(velocity ** 2, axis=1) / np.dot(flow, flow)
            error = cp - exact
            out["offsets"].setdefault(str(offset), {})[name] = {
                "cp_rms_error": float(np.sqrt(np.mean(error ** 2))),
                "cp_max_abs_error": float(np.max(np.abs(error))),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--compare-formulation", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = {"schema": "dpanel.analytic_sphere.v1", "cases": []}
    for subdivisions in args.subdivisions:
        vertices, faces = icosphere(subdivisions)
        entry = audit_sphere(faces, vertices)
        entry["subdivisions"] = subdivisions
        if args.compare_formulation:
            entry["formulation"] = compare_formulation(faces, vertices)
        report["cases"].append(entry)
        print(json.dumps({"subdivisions": subdivisions, "faces": entry["faces"],
                          "seconds": entry["seconds"]}))
    report["physical_accuracy_accepted"] = False
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()