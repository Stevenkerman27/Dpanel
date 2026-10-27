"""P1 G1 assembly: P1 doublet wall + mirrored half + wake sheet + Kutta rows (research only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .p1_dirichlet import (
    assemble_p1_doublet,
    face_cp_from_mu,
    panel_doublet_coefficients,
    self_free_row,
)
from .p1_wake import sheet_potential

FLOAT = np.float64
MIRROR = np.array([1.0, -1.0, 1.0])


def triangle_centroids(vertices, faces):
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    return vertices[faces].mean(axis=1)


def wall_node_index(faces):
    faces = np.asarray(faces, dtype=np.int64)
    nodes, inverse = np.unique(faces, return_inverse=True)
    return nodes, inverse.reshape(faces.shape)


def hat_weights(values, knots):
    """Piecewise-linear hat weights of `values` against the knot vector `knots`."""
    values = np.asarray(values, dtype=FLOAT)
    knots = np.asarray(knots, dtype=FLOAT)
    weights = np.zeros((len(values), len(knots)), dtype=FLOAT)
    for index in range(len(knots) - 1):
        low, high = knots[index], knots[index + 1]
        span = high - low
        inside = (values >= low) & (values <= high)
        if span <= 0.0 or not np.any(inside):
            continue
        fraction = (values[inside] - low) / span
        weights[inside, index] = 1.0 - fraction
        weights[inside, index + 1] = fraction
    return weights


def load_case(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "vertices": np.asarray(data["coordinates"], dtype=FLOAT),
        "wall_faces": np.asarray(data["faces"], dtype=np.int64),
        "wake_faces": np.asarray(data["wake_faces"], dtype=np.int64),
        "stations": np.asarray(data["gamma_stations"], dtype=FLOAT),
        "te_pairs": np.asarray(data["trailing_edges"], dtype=np.int64),
        "freestream": np.asarray(data["freestream"], dtype=FLOAT),
        "reference_area": float(data["reference_area"]),
        "reference_chord": float(data["reference_chord"]),
    }


def mirror_block(vertices, faces, targets, *, tolerance=1e-9):
    """Contributions at ``targets`` of the panels mirrored across y = 0 (even mu)."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    targets = np.asarray(targets, dtype=FLOAT)
    block = np.zeros((len(targets), len(vertices)), dtype=FLOAT)
    on_plane = np.abs(targets[:, 1]) <= tolerance
    for j, face in enumerate(faces):
        triangle = vertices[face] * MIRROR
        coefficients = panel_doublet_coefficients(targets, triangle)
        if np.any(on_plane):
            coefficients[on_plane] = self_free_row()
        for k in range(3):
            block[:, face[k]] += coefficients[:, k]
    return block


def assemble_g1(case, *, wake_sign=1.0, mirrored=True):
    vertices = case["vertices"]
    wall_faces = case["wall_faces"]
    nodes, local_faces = wall_node_index(wall_faces)
    wall_vertices = vertices[nodes]
    centroids = triangle_centroids(wall_vertices, local_faces)
    wall_state = assemble_p1_doublet(wall_vertices, local_faces)
    wall_block = wall_state["matrix"]
    mirror = mirror_block(wall_vertices, local_faces, centroids) if mirrored else 0.0 * wall_block

    knots = np.asarray(case["stations"], dtype=FLOAT)
    wake_nodes = np.unique(np.asarray(case["wake_faces"], dtype=np.int64))
    wake_weights = hat_weights(vertices[wake_nodes, 1], knots)
    image_targets = centroids * MIRROR
    wake_columns = np.zeros((len(wall_faces), len(knots)), dtype=FLOAT)
    for index in range(len(knots)):
        density = np.zeros(len(vertices), dtype=FLOAT)
        density[wake_nodes] = wake_weights[:, index]
        half = sheet_potential(vertices, case["wake_faces"], density, centroids)
        image = sheet_potential(vertices, case["wake_faces"], density, image_targets)
        wake_columns[:, index] = wake_sign * (half - image)

    matrix = np.concatenate([wall_block - mirror, wake_columns], axis=1)
    rhs = -centroids @ np.asarray(case["freestream"], dtype=FLOAT)

    te_pairs = np.asarray(case["te_pairs"], dtype=np.int64)
    lookup = {int(node): index for index, node in enumerate(nodes)}
    kutta = np.zeros((len(te_pairs), matrix.shape[1]), dtype=FLOAT)
    for row, (upper, lower) in enumerate(te_pairs):
        kutta[row, lookup[int(upper)]] = 1.0
        kutta[row, lookup[int(lower)]] = -1.0
        middle = 0.5 * (vertices[upper, 1] + vertices[lower, 1])
        kutta[row, len(nodes):] = -hat_weights(np.array([middle]), knots)[0]

    return {
        "matrix": matrix,
        "rhs": rhs,
        "kutta": kutta,
        "wall_block": wall_block,
        "mirror_block": mirror,
        "wake_columns": wake_columns,
        "nodes": nodes,
        "local_faces": local_faces,
        "wall_vertices": wall_vertices,
        "centroids": centroids,
        "normals": wall_state["normals"],
        "areas": wall_state["areas"],
        "knots": knots,
        "te_pairs": te_pairs,
    }


def solve_g1(case, *, wake_sign=1.0, mirrored=True):
    system = assemble_g1(case, wake_sign=wake_sign, mirrored=mirrored)
    matrix, rhs, kutta = system["matrix"], system["rhs"], system["kutta"]
    stacked = np.concatenate([matrix, kutta], axis=0)
    values = np.concatenate([rhs, np.zeros(len(kutta))])
    solution, _, rank, singular = np.linalg.lstsq(stacked, values, rcond=None)
    n_wall = len(system["nodes"])
    mu, gamma = solution[:n_wall], solution[n_wall:]
    speed = float(np.linalg.norm(case["freestream"]))
    cp = face_cp_from_mu(system["wall_vertices"], system["local_faces"], mu, speed=speed)
    area, normals = system["areas"], system["normals"]
    direction = np.asarray(case["freestream"], dtype=FLOAT) / speed
    outward = np.sign(np.sum(normals * direction, axis=1))
    half_force = -np.sum((cp * area)[:, None] * normals * outward[:, None], axis=0)
    full_force = 2.0 * half_force
    knots = system["knots"]
    span_integral = float(0.5 * np.sum((gamma[1:] + gamma[:-1]) * np.diff(knots)))
    result = dict(system)
    result.update({
        "solution": solution,
        "mu": mu,
        "gamma": gamma,
        "rank": int(rank),
        "singular_values": singular,
        "least_squares_residual": float(np.linalg.norm(stacked @ solution - values)),
        "wall_residual_rms": float(np.sqrt(np.mean((matrix @ solution - rhs) ** 2))),
        "kutta_residual_max": float(np.max(np.abs(kutta @ solution))),
        "row_sum_mean": float(np.mean(system["wall_block"].sum(axis=1) - system["mirror_block"].sum(axis=1))),
        "row_sum_spread": float(np.ptp(system["wall_block"].sum(axis=1) - system["mirror_block"].sum(axis=1))),
        "cp": cp,
        "force_full": full_force.tolist(),
        "cl_pressure": float(full_force[2] / (0.5 * case["reference_area"])),
        "sum_gamma_dy": span_integral,
        "cl_circulation": float(4.0 * span_integral / (speed * case["reference_area"])),
        "gamma_knots": gamma.tolist(),
    })
    return result
