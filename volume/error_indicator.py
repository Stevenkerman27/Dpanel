"""P2 curved-tetra residual indicators for saved total-potential fields."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits


GAMMA = 1.4
TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
TRI_EDGES = ((0, 1), (1, 2), (2, 0))
TET_LAMBDA_GRADIENTS = np.array(
    [[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
)
TRI_LAMBDA_GRADIENTS = np.array([[-1., -1.], [1., 0.], [0., 1.]])


def gauss_rule(order):
    if order not in (3, 4):
        raise ValueError("quadrature order must be 3 or 4")
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (x + 1.0), 0.5 * w


def tetra_rule(order):
    x, w = gauss_rule(order)
    u, v, z = np.meshgrid(x, x, x, indexing="ij")
    wu, wv, wz = np.meshgrid(w, w, w, indexing="ij")
    r = u.ravel()
    s = ((1.0 - u) * v).ravel()
    t = ((1.0 - u) * (1.0 - v) * z).ravel()
    bary = np.column_stack((1.0 - r - s - t, r, s, t))
    weights = (wu * wv * wz * (1.0 - u)**2 * (1.0 - v)).ravel()
    return np.vstack((bary, bary[:, [0, 2, 1, 3]])), np.tile(weights * 0.5, 2)


def triangle_rule(order):
    x, w = gauss_rule(order)
    u, v = np.meshgrid(x, x, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    r = u.ravel()
    s = ((1.0 - u) * v).ravel()
    bary = np.column_stack((1.0 - r - s, r, s))
    weights = (wu * wv * (1.0 - u)).ravel()
    return np.vstack((bary, bary[:, [0, 2, 1]])), np.tile(weights * 0.5, 2)


def tetra_shape(bary):
    bary = np.asarray(bary, dtype=float)
    shape = np.empty(bary.shape[:-1] + (10,))
    gradient = np.empty(bary.shape[:-1] + (10, 3))
    hessian = np.empty((10, 3, 3))
    for i in range(4):
        lam = bary[..., i]
        shape[..., i] = lam * (2.0 * lam - 1.0)
        gradient[..., i, :] = (4.0 * lam - 1.0)[..., None] * TET_LAMBDA_GRADIENTS[i]
        hessian[i] = 4.0 * np.outer(TET_LAMBDA_GRADIENTS[i], TET_LAMBDA_GRADIENTS[i])
    for edge, (i, j) in enumerate(TET_EDGES, start=4):
        li, lj = bary[..., i], bary[..., j]
        shape[..., edge] = 4.0 * li * lj
        gradient[..., edge, :] = 4.0 * (
            li[..., None] * TET_LAMBDA_GRADIENTS[j] + lj[..., None] * TET_LAMBDA_GRADIENTS[i]
        )
        hessian[edge] = 4.0 * (
            np.outer(TET_LAMBDA_GRADIENTS[i], TET_LAMBDA_GRADIENTS[j])
            + np.outer(TET_LAMBDA_GRADIENTS[j], TET_LAMBDA_GRADIENTS[i])
        )
    return shape, gradient, hessian


def triangle_shape(bary):
    bary = np.asarray(bary, dtype=float)
    shape = np.empty(bary.shape[:-1] + (6,))
    gradient = np.empty(bary.shape[:-1] + (6, 2))
    for i in range(3):
        lam = bary[..., i]
        shape[..., i] = lam * (2.0 * lam - 1.0)
        gradient[..., i, :] = (4.0 * lam - 1.0)[..., None] * TRI_LAMBDA_GRADIENTS[i]
    for edge, (i, j) in enumerate(TRI_EDGES, start=3):
        li, lj = bary[..., i], bary[..., j]
        shape[..., edge] = 4.0 * li * lj
        gradient[..., edge, :] = 4.0 * (
            li[..., None] * TRI_LAMBDA_GRADIENTS[j] + lj[..., None] * TRI_LAMBDA_GRADIENTS[i]
        )
    return shape, gradient


def flux_state(points, tet_nodes, phi_nodes, bary, mach):
    _, reference_gradient, reference_hessian = tetra_shape(bary)
    coordinates = points[tet_nodes]
    jacobian = np.einsum("ik,qia->qka", coordinates, reference_gradient, optimize=True)
    determinant = np.linalg.det(jacobian)
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise ValueError("nonpositive tet10 Jacobian at indicator quadrature point")
    inverse_jacobian = np.linalg.inv(jacobian)
    physical_gradient = np.einsum(
        "qia,qak->qik", reference_gradient, inverse_jacobian, optimize=True
    )
    velocity = np.einsum("qik,i->qk", physical_gradient, phi_nodes, optimize=True)

    hessian_phi_reference = np.einsum("i,iab->ab", phi_nodes, reference_hessian, optimize=True)
    hessian_map_reference = np.einsum(
        "ik,iab->kab", coordinates, reference_hessian, optimize=True
    )
    hessian_correction = hessian_phi_reference[None, :, :] - np.einsum(
        "qk,kab->qab", velocity, hessian_map_reference, optimize=True
    )
    hessian_physical = np.einsum(
        "qai,qab,qbj->qij",
        inverse_jacobian,
        hessian_correction,
        inverse_jacobian,
        optimize=True,
    )

    speed2 = np.einsum("qi,qi->q", velocity, velocity)
    mach2 = mach**2
    beta = 1.0 + 0.2 * mach2 * (1.0 - speed2)
    if np.any(~np.isfinite(beta)) or np.any(beta <= 0.0):
        raise ValueError("beta must be strictly positive at indicator quadrature points")
    local_mach2 = mach2 * speed2 / beta
    if np.any(~np.isfinite(local_mach2)) or np.any(local_mach2 >= 1.0):
        raise ValueError("indicator quadrature state is not strictly subsonic")
    density = beta ** (1.0 / (GAMMA - 1.0))
    divergence = density * (
        np.trace(hessian_physical, axis1=1, axis2=2)
        - mach2 / beta * np.einsum(
            "qi,qij,qj->q", velocity, hessian_physical, velocity, optimize=True
        )
    )
    return jacobian, determinant, velocity, hessian_physical, density, divergence


def characteristic_size(coordinates):
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))


def face_nodes_from_tet(tet, ordered_corners):
    node_by_edge = {tuple(sorted((int(tet[i]), int(tet[j])))): int(tet[4 + edge])
                    for edge, (i, j) in enumerate(TET_EDGES)}
    return np.asarray(
        [*ordered_corners,
         node_by_edge[tuple(sorted((int(ordered_corners[0]), int(ordered_corners[1]))))],
         node_by_edge[tuple(sorted((int(ordered_corners[1]), int(ordered_corners[2]))))],
         node_by_edge[tuple(sorted((int(ordered_corners[2]), int(ordered_corners[0]))))]],
        dtype=np.int64,
    )


def face_trace(points, tets, phi, element, face, bary, mach):
    corners = face[:3]
    tet = tets[element]
    expected_face = face_nodes_from_tet(tet, corners)
    face_scale = max(1.0, float(np.max(np.abs(points[expected_face]))))
    if not np.allclose(points[face], points[expected_face], rtol=0.0, atol=1.e-10 * face_scale):
        raise ValueError("tri6 face nodes disagree with adjacent tet10 isoparametric map")
    tet_bary = np.zeros((len(bary), 4))
    for local_face_node, global_corner in enumerate(corners):
        matches = np.flatnonzero(tet[:4] == global_corner)
        if len(matches) != 1:
            raise ValueError("face corner is not in its adjacent tetrahedron")
        tet_bary[:, matches[0]] = bary[:, local_face_node]

    surface_shape, surface_gradient = triangle_shape(bary)
    face_coordinates = points[face]
    surface_points = np.einsum("qi,ik->qk", surface_shape, face_coordinates, optimize=True)
    tangent = np.einsum("qia,ik->qka", surface_gradient, face_coordinates, optimize=True)
    area_vector = np.cross(tangent[:, :, 0], tangent[:, :, 1])
    area = np.linalg.norm(area_vector, axis=1)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("degenerate curved tri6 at indicator quadrature point")

    _, _, velocity, _, density, _ = flux_state(points, tet, phi[tet], tet_bary, mach)
    center_bary = np.full((1, 4), 0.25)
    cell_center = tetra_shape(center_bary)[0] @ points[tet]
    sign = np.sign(np.dot(np.mean(surface_points, axis=0) - cell_center[0], np.mean(area_vector, axis=0)))
    if sign == 0.0:
        raise ValueError("cannot orient a boundary/interface normal from its adjacent cell")
    normal = area_vector * sign / area[:, None]
    mass_flux = density[:, None] * velocity
    return surface_points, normal, area, mass_flux, velocity, area_vector * sign


def face_diameter(points, face):
    return characteristic_size(points[face])


def build_face_map(tets):
    face_map = {}
    for element, tet in enumerate(tets):
        for omitted in range(4):
            corners = np.delete(tet[:4], omitted)
            ordered = np.sort(corners)
            face_map.setdefault(tuple(ordered.tolist()), []).append((element, ordered))
    return face_map


def paired_corner_permutation(points, first, second):
    permutation = []
    for point in points[first[:3]]:
        distance = np.linalg.norm(points[second[:3]] - point, axis=1)
        j = int(np.argmin(distance))
        if distance[j] > 1.0e-10 * max(1.0, np.linalg.norm(point)):
            raise ValueError("wake-cut faces do not coincide geometrically")
        permutation.append(j)
    return np.asarray(permutation, dtype=np.int64)


def mark_dorfler(cell_total, fraction=0.5):
    cell_total = np.asarray(cell_total, dtype=float)
    order = np.lexsort((np.arange(len(cell_total)), -cell_total))
    total = float(np.sum(cell_total))
    marked = np.zeros(len(cell_total), dtype=bool)
    cumulative = np.zeros(len(cell_total))
    if total > 0.0:
        cumulative = np.cumsum(cell_total[order]) / total
        count = int(np.searchsorted(cumulative, fraction, side="left")) + 1
        marked[order[:count]] = True
    return marked, order, cumulative, total


def load_inputs(mesh_path, field_path):
    mesh_bytes = mesh_path.read_bytes()
    field_bytes = field_path.read_bytes()
    with np.load(mesh_path) as archive:
        mesh = {key: archive[key] for key in archive.files}
    with np.load(field_path) as archive:
        field = {key: archive[key] for key in archive.files}
    potential_key = "Phi_total" if "Phi_total" in field else "phi" if "phi" in field else None
    if potential_key is None:
        raise ValueError("field NPZ must contain Phi_total (volume) or phi (lifting3d)")
    for key in ("points", "tets"):
        if key not in mesh or key not in field:
            raise ValueError(f"mesh and field must both contain {key}")
        if not np.array_equal(mesh[key], field[key]):
            raise ValueError(f"mesh and field {key} differ; refusing implicit node reordering")
    points = np.asarray(mesh["points"], dtype=float)
    tets = np.asarray(mesh["tets"], dtype=np.int64)
    phi = np.asarray(field[potential_key], dtype=float)
    faces = np.asarray(mesh["boundary_triangles"], dtype=np.int64)
    tags = np.asarray(mesh["boundary_tags"], dtype=np.int64)
    if phi.shape != (len(points),) or faces.ndim != 2 or faces.shape[1] != 6:
        raise ValueError("expected nodal P2 potential and tri6 boundary connectivity")
    if tags.shape != (len(faces),) or not np.all(np.isfinite(phi)):
        raise ValueError("boundary tags or nodal potential shape/value is invalid")
    if "wake_upper" in mesh or "wake_lower" in mesh:
        if not all(key in mesh for key in ("wake_upper", "wake_lower", "wake_upper_tet", "wake_lower_tet")):
            raise ValueError("wake-cut mesh arrays must be supplied as a complete upper/lower pair")
    return mesh, field, points, tets, phi, faces, tags, potential_key, mesh_bytes, field_bytes


def compute_indicator(mesh, points, tets, phi, faces, tags, mach, order):
    bary, volume_weights = tetra_rule(order)
    tri_bary, tri_weights = triangle_rule(order)
    _, reference_gradient, reference_hessian = tetra_shape(bary)
    cell_bulk = np.zeros(len(tets))
    max_abs_divergence = np.zeros(len(tets))
    cell_diameter = np.zeros(len(tets))
    for first in range(0, len(tets), 128):
        last = min(first + 128, len(tets))
        nodes = tets[first:last]
        coordinates = points[nodes]
        local_phi = phi[nodes]
        differences = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        cell_diameter[first:last] = np.max(np.linalg.norm(differences, axis=3), axis=(1, 2))
        jacobian = np.einsum("eik,qia->eqka", coordinates, reference_gradient, optimize=True)
        determinant = np.linalg.det(jacobian)
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError("nonpositive tet10 Jacobian at indicator quadrature point")
        inverse_jacobian = np.linalg.inv(jacobian)
        physical_gradient = np.einsum(
            "qia,eqak->eqik", reference_gradient, inverse_jacobian, optimize=True
        )
        velocity = np.einsum("eqik,ei->eqk", physical_gradient, local_phi, optimize=True)
        hessian_phi_reference = np.einsum(
            "ei,iab->eab", local_phi, reference_hessian, optimize=True
        )
        hessian_map_reference = np.einsum(
            "eik,iab->ekab", coordinates, reference_hessian, optimize=True
        )
        hessian_correction = hessian_phi_reference[:, None] - np.einsum(
            "eqk,ekab->eqab", velocity, hessian_map_reference, optimize=True
        )
        hessian_physical = np.einsum(
            "eqai,eqab,eqbj->eqij",
            inverse_jacobian,
            hessian_correction,
            inverse_jacobian,
            optimize=True,
        )
        speed2 = np.einsum("eqk,eqk->eq", velocity, velocity)
        beta = 1.0 + 0.2 * mach**2 * (1.0 - speed2)
        if np.any(~np.isfinite(beta)) or np.any(beta <= 0.0):
            raise ValueError("beta must be strictly positive at indicator quadrature points")
        local_mach2 = mach**2 * speed2 / beta
        if np.any(~np.isfinite(local_mach2)) or np.any(local_mach2 >= 1.0):
            raise ValueError("indicator quadrature state is not strictly subsonic")
        density = beta ** (1.0 / (GAMMA - 1.0))
        divergence = density * (
            np.trace(hessian_physical, axis1=2, axis2=3)
            - mach**2 / beta * np.einsum(
                "eqk,eqkl,eql->eq", velocity, hessian_physical, velocity, optimize=True
            )
        )
        cell_bulk[first:last] = cell_diameter[first:last]**2 * np.sum(
            divergence**2 * determinant * volume_weights[None, :], axis=1
        )
        max_abs_divergence[first:last] = np.max(np.abs(divergence), axis=1)

    cell_internal = np.zeros(len(tets))
    face_map = build_face_map(tets)
    internal_face_records = []
    max_internal_relative_area_gap = 0.0
    max_internal_outward_area_gap = 0.0
    for key, adjacent in face_map.items():
        if len(adjacent) != 2:
            if len(adjacent) > 2:
                raise ValueError("non-manifold corner face has more than two adjacent tetrahedra")
            continue
        corners = np.asarray(key, dtype=np.int64)
        faces_on_sides = [face_nodes_from_tet(tets[item[0]], corners) for item in adjacent]
        traces = [
            face_trace(points, tets, phi, item[0], face, tri_bary, mach)
            for item, face in zip(adjacent, faces_on_sides)
        ]
        point_gap = float(np.max(np.linalg.norm(traces[0][0] - traces[1][0], axis=1)))
        if point_gap > 1.0e-10 * max(1.0, float(np.max(np.abs(points[faces_on_sides[0]])))):
            raise ValueError("internal face traces disagree in physical coordinates")
        mean_area = 0.5 * (traces[0][2] + traces[1][2])
        relative_area_gap = np.abs(traces[0][2] - traces[1][2]) / np.maximum(
            mean_area, np.finfo(float).tiny
        )
        outward_area_gap = np.linalg.norm(traces[0][5] + traces[1][5], axis=1) / np.maximum(
            mean_area, np.finfo(float).tiny
        )
        max_internal_relative_area_gap = max(max_internal_relative_area_gap, float(np.max(relative_area_gap)))
        max_internal_outward_area_gap = max(max_internal_outward_area_gap, float(np.max(outward_area_gap)))
        if np.max(relative_area_gap) > 1.0e-8 or np.max(outward_area_gap) > 1.0e-8:
            raise ValueError("internal face traces do not close area Jacobian/orientation")
        jump = (np.einsum("qk,qk->q", traces[0][3], traces[0][1])
                + np.einsum("qk,qk->q", traces[1][3], traces[1][1]))
        integral = float(np.sum(jump**2 * traces[0][2] * tri_weights))
        hface = max(face_diameter(points, face) for face in faces_on_sides)
        contribution = hface * integral
        for element, _ in adjacent:
            cell_internal[element] += 0.5 * contribution
        internal_face_records.append((adjacent[0][0], adjacent[1][0], hface, integral,
                                      point_gap, np.max(relative_area_gap), np.max(outward_area_gap)))

    cell_wake = np.zeros(len(tets))
    wake_face_records = []
    max_wake_relative_area_gap = 0.0
    max_wake_outward_area_gap = 0.0
    wake_up = np.asarray(mesh.get("wake_upper", np.empty((0, 6))), dtype=np.int64)
    wake_low = np.asarray(mesh.get("wake_lower", np.empty((0, 6))), dtype=np.int64)
    wake_up_tet = np.asarray(mesh.get("wake_upper_tet", np.empty(0)), dtype=np.int64)
    wake_low_tet = np.asarray(mesh.get("wake_lower_tet", np.empty(0)), dtype=np.int64)
    if len(wake_up) != len(wake_low):
        raise ValueError("wake upper/lower face counts differ")
    for i, (upper, lower) in enumerate(zip(wake_up, wake_low)):
        upper_element = int(wake_up_tet[i])
        lower_element = int(wake_low_tet[i])
        permutation = paired_corner_permutation(points, upper, lower)
        lower_bary = np.empty_like(tri_bary)
        for upper_corner, lower_corner in enumerate(permutation):
            lower_bary[:, lower_corner] = tri_bary[:, upper_corner]
        trace_upper = face_trace(points, tets, phi, upper_element, upper, tri_bary, mach)
        trace_lower = face_trace(points, tets, phi, lower_element, lower, lower_bary, mach)
        point_gap = float(np.max(np.linalg.norm(trace_upper[0] - trace_lower[0], axis=1)))
        scale = max(1.0, float(np.max(np.abs(points[np.r_[upper, lower]]))))
        if point_gap > 1.0e-10 * scale:
            raise ValueError("paired wake-cut surface maps do not coincide")
        mean_area = 0.5 * (trace_upper[2] + trace_lower[2])
        relative_area_gap = np.abs(trace_upper[2] - trace_lower[2]) / np.maximum(
            mean_area, np.finfo(float).tiny
        )
        outward_area_gap = np.linalg.norm(trace_upper[5] + trace_lower[5], axis=1) / np.maximum(
            mean_area, np.finfo(float).tiny
        )
        max_wake_relative_area_gap = max(max_wake_relative_area_gap, float(np.max(relative_area_gap)))
        max_wake_outward_area_gap = max(max_wake_outward_area_gap, float(np.max(outward_area_gap)))
        if np.max(relative_area_gap) > 1.0e-8 or np.max(outward_area_gap) > 1.0e-8:
            raise ValueError("wake-cut traces do not close area Jacobian/orientation")
        jump = (np.einsum("qk,qk->q", trace_upper[3], trace_upper[1])
                + np.einsum("qk,qk->q", trace_lower[3], trace_lower[1]))
        integral = float(np.sum(jump**2 * trace_upper[2] * tri_weights))
        hface = max(face_diameter(points, upper), face_diameter(points, lower))
        contribution = hface * integral
        cell_wake[upper_element] += 0.5 * contribution
        cell_wake[lower_element] += 0.5 * contribution
        wake_face_records.append((upper_element, lower_element, hface, integral, point_gap,
                                  np.max(relative_area_gap), np.max(outward_area_gap)))

    cell_wall = np.zeros(len(tets))
    wall_velocity_integrals = []
    wall_mass_integrals = []
    wall_face_records = []
    boundary_adjacency = np.asarray(mesh.get("boundary_tet", np.full(len(faces), -1)), dtype=np.int64)
    exterior_faces = np.flatnonzero(tags != 100)
    for face_index in exterior_faces:
        face = faces[face_index]
        if boundary_adjacency[face_index] < 0:
            matches = face_map.get(tuple(sorted(face[:3].tolist())), [])
            if len(matches) != 1:
                raise ValueError("could not uniquely identify boundary face adjacent tetrahedron")
            element = int(matches[0][0])
        else:
            element = int(boundary_adjacency[face_index])
        trace = face_trace(points, tets, phi, element, face, tri_bary, mach)
        normal_velocity = np.einsum("qk,qk->q", trace[4], trace[1])
        normal_mass_flux = np.einsum("qk,qk->q", trace[3], trace[1])
        weights = trace[2] * tri_weights
        integral = float(np.sum(normal_mass_flux**2 * weights))
        hface = face_diameter(points, face)
        cell_wall[element] += hface * integral
        wall_velocity_integrals.append(float(np.sum(normal_velocity * weights)))
        wall_mass_integrals.append(float(np.sum(normal_mass_flux * weights)))
        wall_face_records.append((int(face_index), element, int(tags[face_index]), hface, integral))

    total_cell = cell_bulk + cell_internal + cell_wake + cell_wall
    marked, rank, cumulative, total = mark_dorfler(total_cell, 0.5)
    return {
        "cell_bulk": cell_bulk,
        "cell_internal": cell_internal,
        "cell_wake": cell_wake,
        "cell_wall": cell_wall,
        "cell_total": total_cell,
        "cell_diameter": cell_diameter,
        "cell_max_abs_divergence": max_abs_divergence,
        "marked": marked,
        "dorfler_order": rank,
        "dorfler_cumulative_fraction": cumulative,
        "dorfler_total": total,
        "internal_face_records": np.asarray(internal_face_records, dtype=float).reshape(-1, 7),
        "wake_face_records": np.asarray(wake_face_records, dtype=float).reshape(-1, 7),
        "wall_face_records": np.asarray(wall_face_records, dtype=float).reshape(-1, 5),
        "wall_velocity_integral": float(np.sum(wall_velocity_integrals)),
        "wall_mass_flux_integral": float(np.sum(wall_mass_integrals)),
        "wall_mass_flux_l1": float(np.sum(np.abs(wall_mass_integrals))),
        "marked_count": int(np.count_nonzero(marked)),
        "marked_fraction_of_cells": float(np.mean(marked)),
        "marked_contribution_fraction": float(np.sum(total_cell[marked]) / total) if total > 0.0 else 0.0,
        "bulk_total": float(np.sum(cell_bulk)),
        "internal_total": float(np.sum(cell_internal)),
        "wake_total": float(np.sum(cell_wake)),
        "wall_total": float(np.sum(cell_wall)),
        "max_abs_divergence": float(np.max(max_abs_divergence)),
        "internal_face_count": int(len(internal_face_records)),
        "wake_face_count": int(len(wake_face_records)),
        "wall_face_count": int(len(wall_face_records)),
        "max_internal_relative_area_gap": max_internal_relative_area_gap,
        "max_internal_outward_area_vector_gap": max_internal_outward_area_gap,
        "max_wake_relative_area_gap": max_wake_relative_area_gap,
        "max_wake_outward_area_vector_gap": max_wake_outward_area_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Independent P2 strong-residual and raw-flux indicator")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--mach", required=True, type=float)
    parser.add_argument("--quad-order", type=int, choices=(3, 4), default=3)
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.mesh = args.mesh.resolve()
    args.field = args.field.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {
        "status": "failed",
        "mesh": str(args.mesh),
        "field": str(args.field),
        "mach": args.mach,
        "quadrature_order": args.quad_order,
        "ncpu": args.ncpu,
        "method": "isoparametric tet10 analytic Hessian strong residual plus raw mass-flux face terms",
        "error_bound_claim": False,
        "scope": "research refinement indicator; no certified Cp/load error bound",
        "history": [],
    }
    try:
        if args.ncpu < 1 or args.mach < 0.0 or not np.isfinite(args.mach):
            raise ValueError("ncpu must be positive and Mach must be finite/nonnegative")
        mesh, field, points, tets, phi, faces, tags, potential_key, mesh_bytes, field_bytes = load_inputs(
            args.mesh, args.field
        )
        result.update(
            potential_key=potential_key,
            mesh_sha256=hashlib.sha256(mesh_bytes).hexdigest(),
            field_sha256=hashlib.sha256(field_bytes).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            points=len(points),
            elements=len(tets),
            potential_min=float(np.min(phi)),
            potential_max=float(np.max(phi)),
        )
        with threadpool_limits(limits=args.ncpu):
            data = compute_indicator(mesh, points, tets, phi, faces, tags, args.mach, args.quad_order)
        result.update({key: value for key, value in data.items()
                       if not isinstance(value, np.ndarray)})
        result.update(status="measured", elapsed_seconds=time.perf_counter() - started)
        np.savez_compressed(
            args.out / "indicator.npz",
            cell_bulk=data["cell_bulk"],
            cell_internal=data["cell_internal"],
            cell_wake_cut=data["cell_wake"],
            cell_wall=data["cell_wall"],
            cell_total=data["cell_total"],
            cell_diameter=data["cell_diameter"],
            cell_max_abs_divergence=data["cell_max_abs_divergence"],
            marked=data["marked"],
            dorfler_order=data["dorfler_order"],
            dorfler_cumulative_fraction=data["dorfler_cumulative_fraction"],
            internal_face_records=data["internal_face_records"],
            wake_face_records=data["wake_face_records"],
            wall_face_records=data["wall_face_records"],
        )
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        print(result["error"], file=sys.stderr)
    finally:
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if result["status"] != "measured":
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "elements", "potential_key", "bulk_total", "internal_total", "wake_total",
        "wall_total", "marked_count", "marked_fraction_of_cells", "marked_contribution_fraction"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
