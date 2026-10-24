import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


GAMMA = 1.4
TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
TRI_EDGES = [(0, 1), (1, 2), (2, 0)]
TET_LAMBDA_GRADIENTS = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TRI_LAMBDA_GRADIENTS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


def gauss_points(order):
    if order not in [3, 4]:
        raise ValueError("quadrature order must be 3 or 4")
    points, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (points + 1.0), 0.5 * weights


def tetra_quadrature(order):
    one_d, one_w = gauss_points(order)
    u, v, w = np.meshgrid(one_d, one_d, one_d, indexing="ij")
    wu, wv, ww = np.meshgrid(one_w, one_w, one_w, indexing="ij")
    r = u.ravel()
    s = ((1.0 - u) * v).ravel()
    t = ((1.0 - u) * (1.0 - v) * w).ravel()
    barycentric = np.column_stack((1.0 - r - s - t, r, s, t))
    weights = (wu * wv * ww * (1.0 - u) ** 2 * (1.0 - v)).ravel()
    swapped_barycentric = barycentric[:, [0, 2, 1, 3]]
    return np.vstack((barycentric, swapped_barycentric)), np.tile(0.5 * weights, 2)


def triangle_quadrature(order):
    one_d, one_w = gauss_points(order)
    u, v = np.meshgrid(one_d, one_d, indexing="ij")
    wu, wv = np.meshgrid(one_w, one_w, indexing="ij")
    r = u.ravel()
    s = ((1.0 - u) * v).ravel()
    barycentric = np.column_stack((1.0 - r - s, r, s))
    weights = (wu * wv * (1.0 - u)).ravel()
    swapped_barycentric = barycentric[:, [0, 2, 1]]
    return np.vstack((barycentric, swapped_barycentric)), np.tile(0.5 * weights, 2)


def p2_shape(barycentric):
    barycentric = np.asarray(barycentric, dtype=float)
    shape = np.empty(barycentric.shape[:-1] + (10,))
    reference_gradient = np.empty(barycentric.shape[:-1] + (10, 3))
    for i in range(4):
        lam = barycentric[..., i]
        shape[..., i] = lam * (2.0 * lam - 1.0)
        reference_gradient[..., i, :] = (4.0 * lam - 1.0)[..., None] * TET_LAMBDA_GRADIENTS[i]
    for edge_index, (i, j) in enumerate(TET_EDGES, start=4):
        li = barycentric[..., i]
        lj = barycentric[..., j]
        shape[..., edge_index] = 4.0 * li * lj
        reference_gradient[..., edge_index, :] = 4.0 * (
            li[..., None] * TET_LAMBDA_GRADIENTS[j] + lj[..., None] * TET_LAMBDA_GRADIENTS[i]
        )
    return shape, reference_gradient


def triangle_shape(barycentric):
    barycentric = np.asarray(barycentric, dtype=float)
    shape = np.empty(barycentric.shape[:-1] + (6,))
    reference_gradient = np.empty(barycentric.shape[:-1] + (6, 2))
    for i in range(3):
        lam = barycentric[..., i]
        shape[..., i] = lam * (2.0 * lam - 1.0)
        reference_gradient[..., i, :] = (4.0 * lam - 1.0)[..., None] * TRI_LAMBDA_GRADIENTS[i]
    for edge_index, (i, j) in enumerate(TRI_EDGES, start=3):
        li = barycentric[..., i]
        lj = barycentric[..., j]
        shape[..., edge_index] = 4.0 * li * lj
        reference_gradient[..., edge_index, :] = 4.0 * (
            li[..., None] * TRI_LAMBDA_GRADIENTS[j] + lj[..., None] * TRI_LAMBDA_GRADIENTS[i]
        )
    return shape, reference_gradient


def energy_primitive(q2, mach):
    c = 0.2 * mach**2
    if c == 0.0:
        return 0.5 * q2
    log_ratio = np.log1p(-c * q2 / (1.0 + c))
    return -(1.0 + c) ** 3.5 * np.expm1(3.5 * log_ratio) / (7.0 * c)


def pressure_coefficient(q2, mach):
    if mach == 0.0 or mach**2 == 0.0:
        return 1.0 - q2
    mach2 = mach**2
    log_beta = np.log1p(0.2 * mach2 * (1.0 - q2))
    return 2.0 * np.expm1((GAMMA / (GAMMA - 1.0)) * log_beta) / (GAMMA * mach2)


def evaluate(points, tets, phi, mach, order, with_jacobian, batch_size=256):
    barycentric, quadrature_weights = tetra_quadrature(order)
    shape, reference_gradient = p2_shape(barycentric)
    node_count = len(points)
    residual = np.zeros(node_count)
    jacobian_rows = []
    jacobian_cols = []
    jacobian_data = []
    element_max_mach2 = np.zeros(len(tets))
    element_min_beta = np.full(len(tets), np.inf)
    total_energy = 0.0
    for first in range(0, len(tets), batch_size):
        last = min(first + batch_size, len(tets))
        tet_nodes = tets[first:last]
        coordinates = points[tet_nodes]
        mapping_jacobian = np.einsum("eik,qia->eqka", coordinates, reference_gradient, optimize=True)
        determinant = np.linalg.det(mapping_jacobian)
        if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError("tet10 isoparametric Jacobian must be positive at every volume quadrature point")
        inverse_jacobian = np.linalg.inv(mapping_jacobian)
        physical_gradient = np.einsum("qia,eqak->eqik", reference_gradient, inverse_jacobian, optimize=True)
        element_velocity = np.einsum("eqik,ei->eqk", physical_gradient, phi[tet_nodes], optimize=True)
        q2 = np.einsum("eqk,eqk->eq", element_velocity, element_velocity)
        beta = 1.0 + 0.2 * mach**2 * (1.0 - q2)
        if not np.all(np.isfinite(beta)) or np.any(beta <= 0.0):
            raise ValueError("iterate is not admissible: beta must be positive")
        local_mach2 = mach**2 * q2 / beta
        if not np.all(np.isfinite(local_mach2)) or np.any(local_mach2 >= 1.0):
            raise ValueError("iterate is not strictly subsonic at a volume quadrature point")
        density = beta ** (1.0 / (GAMMA - 1.0))
        flux = density[..., None] * element_velocity
        measure = determinant * quadrature_weights[None, :]
        local_residual = np.einsum("eqik,eqk,eq->ei", physical_gradient, flux, measure, optimize=True)
        np.add.at(residual, tet_nodes.ravel(), local_residual.ravel())
        element_max_mach2[first:last] = np.max(local_mach2, axis=1)
        element_min_beta[first:last] = np.min(beta, axis=1)
        total_energy += float(np.sum(measure * energy_primitive(q2, mach)))
        if with_jacobian:
            constitutive = density[..., None, None] * (
                np.eye(3)[None, None, :, :]
                - (mach**2 / beta)[..., None, None]
                * element_velocity[..., :, None]
                * element_velocity[..., None, :]
            )
            local_jacobian = np.einsum(
                "eqia,eqab,eqjb,eq->eij",
                physical_gradient,
                constitutive,
                physical_gradient,
                measure,
                optimize=True,
            )
            jacobian_rows.append(np.broadcast_to(tet_nodes[:, :, None], local_jacobian.shape).ravel())
            jacobian_cols.append(np.broadcast_to(tet_nodes[:, None, :], local_jacobian.shape).ravel())
            jacobian_data.append(local_jacobian.ravel())
    jacobian = None
    if with_jacobian:
        jacobian = coo_matrix(
            (np.concatenate(jacobian_data), (np.concatenate(jacobian_rows), np.concatenate(jacobian_cols))),
            shape=(node_count, node_count),
        ).tocsr()
    return residual, jacobian, element_max_mach2, element_min_beta, total_energy


def load_mesh(mesh_path):
    raw = Path(mesh_path).read_bytes()
    with np.load(mesh_path) as archive:
        points = np.asarray(archive["points"], dtype=float)
        tets = np.asarray(archive["tets"])
        boundary_triangles = np.asarray(archive["boundary_triangles"])
        boundary_tags = np.asarray(archive["boundary_tags"])
        boundary_tet_input = np.asarray(archive["boundary_tet"]) if "boundary_tet" in archive else None
    if not np.issubdtype(tets.dtype, np.integer) or not np.issubdtype(boundary_triangles.dtype, np.integer):
        raise ValueError("tetrahedron and boundary connectivity arrays must be integer")
    if not np.issubdtype(boundary_tags.dtype, np.integer):
        raise ValueError("boundary_tags must be integer before conversion")
    if boundary_tet_input is not None and not np.issubdtype(boundary_tet_input.dtype, np.integer):
        raise ValueError("boundary_tet adjacency must be integer")
    tets = tets.astype(np.int64, copy=True)
    boundary_triangles = boundary_triangles.astype(np.int64, copy=True)
    boundary_tags = boundary_tags.astype(np.int64, copy=False)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must have shape (n,3) and finite coordinates")
    if tets.ndim != 2 or tets.shape[1] != 10 or len(tets) == 0:
        raise ValueError("P2 tets must have shape (m,10)")
    if boundary_triangles.ndim != 2 or boundary_triangles.shape[1] != 6:
        raise ValueError("P2 boundary_triangles must have shape (b,6)")
    if boundary_tags.shape != (len(boundary_triangles),):
        raise ValueError("boundary_tags must have one tag per tri6")
    if np.any(tets < 0) or np.any(tets >= len(points)) or np.any(boundary_triangles < 0) or np.any(boundary_triangles >= len(points)):
        raise ValueError("mesh contains an out-of-range node index")
    if np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0):
        raise ValueError("every tet10 must contain ten distinct node indices")
    used_nodes = np.unique(tets)
    if len(used_nodes) != len(points):
        raise ValueError(f"mesh has {len(points) - len(used_nodes)} point nodes unused by every tet10")
    corner_xyz = points[tets[:, :4]]
    corner_matrix = np.transpose(corner_xyz[:, 1:]-corner_xyz[:, :1], (0, 2, 1))
    corner_determinant = np.linalg.det(corner_matrix)
    if not np.all(np.isfinite(corner_determinant)) or np.any(corner_determinant <= 0.0):
        raise ValueError("tet10 corner tetrahedra must have positive signed volume")
    edge_pairs = np.concatenate([np.sort(tets[:, [i, j]], axis=1) for i, j in TET_EDGES])
    edge_midnodes = np.concatenate([tets[:, 4+edge] for edge in range(6)])
    unique_edges, edge_inverse = np.unique(edge_pairs, axis=0, return_inverse=True)
    minimum_midnode = np.full(len(unique_edges), len(points), dtype=np.int64)
    maximum_midnode = np.full(len(unique_edges), -1, dtype=np.int64)
    np.minimum.at(minimum_midnode, edge_inverse, edge_midnodes)
    np.maximum.at(maximum_midnode, edge_inverse, edge_midnodes)
    if np.any(minimum_midnode != maximum_midnode):
        raise ValueError("all tet10 cells sharing a corner edge must share the same midside node")
    lattice = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
    _, lattice_gradient = p2_shape(lattice)
    for first in range(0, len(tets), 256):
        jacobian = np.einsum("eia,qib->eqab", points[tets[first:first+256]], lattice_gradient)
        determinant = np.linalg.det(jacobian)
        if not np.all(np.isfinite(determinant)) or np.min(determinant) <= 0.:
            raise ValueError("tet10 Jacobian must be positive on the 165-point reference lattice")
    face_map = {}
    for element, tet in enumerate(tets):
        for omitted in range(4):
            face = tuple(sorted(np.delete(tet[:4], omitted).tolist()))
            face_map.setdefault(face, []).append(element)
    if any(len(adjacent) > 2 for adjacent in face_map.values()):
        raise ValueError("non-manifold corner face has more than two adjacent tetrahedra")
    exterior_faces = {face for face, adjacent in face_map.items() if len(adjacent) == 1}
    boundary_faces = [tuple(sorted(triangle[:3].tolist())) for triangle in boundary_triangles]
    if len(set(boundary_faces)) != len(boundary_faces) or set(boundary_faces) != exterior_faces:
        raise ValueError("tri6 faces must list every exterior corner face exactly once")
    inferred = []
    reoriented_by_tag = {}
    for face_index, triangle in enumerate(boundary_triangles):
        adjacent = face_map[tuple(sorted(triangle[:3].tolist()))]
        element = adjacent[0]
        inferred.append(element)
        tet_corner_ids = tets[element, :4].tolist()
        triangle_local = [tet_corner_ids.index(int(node)) for node in triangle[:3]]
        expected_edges = [
            next(tets[element, 4 + edge_index] for edge_index, pair in enumerate(TET_EDGES) if set(pair) == set((triangle_local[0], triangle_local[1]))),
            next(tets[element, 4 + edge_index] for edge_index, pair in enumerate(TET_EDGES) if set(pair) == set((triangle_local[1], triangle_local[2]))),
            next(tets[element, 4 + edge_index] for edge_index, pair in enumerate(TET_EDGES) if set(pair) == set((triangle_local[2], triangle_local[0]))),
        ]
        if not np.array_equal(triangle[3:], expected_edges):
            raise ValueError(f"tri6 face {face_index} midside nodes do not match adjacent tet10 edges")
        tri_shape, tri_gradient = triangle_shape(np.array([[1.0 / 3.0] * 3]))
        face_coordinates = points[triangle]
        face_center = tri_shape[0] @ face_coordinates
        face_jacobian = face_coordinates.T @ tri_gradient[0]
        normal = np.cross(face_jacobian[:, 0], face_jacobian[:, 1])
        opposite_local = next(i for i in range(4) if i not in triangle_local)
        if np.dot(normal, points[tets[element, opposite_local]] - face_center) > 0.0:
            boundary_triangles[face_index] = boundary_triangles[face_index, [0, 2, 1, 5, 4, 3]]
            tag = str(int(boundary_tags[face_index]))
            reoriented_by_tag[tag] = reoriented_by_tag.get(tag, 0) + 1
    mismatch_count = 0
    if boundary_tet_input is not None:
        if boundary_tet_input.shape != (len(boundary_triangles),):
            mismatch_count = len(boundary_triangles)
        else:
            mismatch_count = int(np.count_nonzero(boundary_tet_input != inferred))
    if 100 not in boundary_tags:
        raise ValueError("boundary tag 100 is required")
    if not np.all(np.isin(boundary_tags, [10, 20, 30, 100])):
        raise ValueError("boundary tags must be one of 10, 20, 30, 100")
    face_rows = [adjacent for adjacent in face_map.values() if len(adjacent) == 2]
    cell_rows = [pair[0] for pair in face_rows] + [pair[1] for pair in face_rows]
    cell_cols = [pair[1] for pair in face_rows] + [pair[0] for pair in face_rows]
    adjacency = coo_matrix((np.ones(len(cell_rows)), (cell_rows, cell_cols)), shape=(len(tets), len(tets))).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("fluid tetrahedral mesh must be connected")
    return points, tets, boundary_triangles, boundary_tags, np.asarray(inferred), hashlib.sha256(raw).hexdigest(), reoriented_by_tag, mismatch_count


def internal_face_flux_jump(points, tets, phi, mach, order, batch_size=256):
    face_map = {}
    for element, tet in enumerate(tets):
        for omitted in range(4):
            face_corners = np.sort(np.delete(tet[:4], omitted))
            key = tuple(face_corners.tolist())
            face_map.setdefault(key, []).append(element)
    internal = [(face, adjacent) for face, adjacent in face_map.items() if len(adjacent) == 2]
    face_corners = np.asarray([item[0] for item in internal], dtype=np.int64)
    adjacent = np.asarray([item[1] for item in internal], dtype=np.int64)
    barycentric, quadrature_weights = triangle_quadrature(order)
    face_shape, _ = triangle_shape(barycentric)
    area_side_sum = np.zeros(2)
    signed_side_flux = np.zeros(2)
    absolute_jump_integral = 0.0
    squared_jump_integral = 0.0
    total_area = 0.0
    max_point_gap = 0.0
    max_relative_area_gap = 0.0
    max_outward_area_vector_gap = 0.0
    max_abs_flux_jump = 0.0
    centers = np.mean(points[tets[:, :4]], axis=1)

    for first in range(0, len(face_corners), batch_size):
        last = min(first + batch_size, len(face_corners))
        triangles = face_corners[first:last]
        cells = adjacent[first:last]
        face_count = len(triangles)
        side_points = []
        side_area_vector = []
        side_flux = []
        for side in range(2):
            cell_ids = cells[:, side]
            cell_nodes = tets[cell_ids]
            cell_corners = cell_nodes[:, :4]
            volume_barycentric = np.zeros((face_count, len(barycentric), 4))
            local_face_corners = np.empty((face_count, 3), dtype=np.int64)
            for corner in range(3):
                local_face_corners[:, corner] = np.argmax(
                    cell_corners == triangles[:, corner, None], axis=1
                )
                volume_barycentric[
                    np.arange(face_count)[:, None],
                    np.arange(len(barycentric))[None, :],
                    local_face_corners[:, corner, None],
                ] = barycentric[None, :, corner]

            shape, reference_gradient = p2_shape(volume_barycentric)
            coordinates = points[cell_nodes]
            mapping_jacobian = np.einsum(
                "bik,bqia->bqka", coordinates, reference_gradient, optimize=True
            )
            determinant = np.linalg.det(mapping_jacobian)
            if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0.0):
                raise ValueError("internal-face tet10 Jacobian is nonpositive")
            physical_gradient = np.einsum(
                "bqia,bqak->bqik", reference_gradient, np.linalg.inv(mapping_jacobian), optimize=True
            )
            velocity = np.einsum("bqik,bi->bqk", physical_gradient, phi[cell_nodes], optimize=True)
            q2 = np.einsum("bqk,bqk->bq", velocity, velocity)
            beta = 1.0 + 0.2 * mach**2 * (1.0 - q2)
            if not np.all(np.isfinite(beta)) or np.any(beta <= 0.0):
                raise ValueError("internal-face quadrature state violates beta > 0")
            local_mach2 = mach**2 * q2 / beta
            if not np.all(np.isfinite(local_mach2)) or np.any(local_mach2 >= 1.0):
                raise ValueError("internal-face quadrature state is not strictly subsonic")
            density = beta ** (1.0 / (GAMMA - 1.0))
            flux = density[:, :, None] * velocity
            mapped_points = np.einsum("bqi,bik->bqk", shape, coordinates, optimize=True)

            dref = np.zeros((face_count, 3, 2))
            for reference_index in range(1, 4):
                dref[:, reference_index - 1, 0] = (
                    (local_face_corners[:, 1] == reference_index).astype(float)
                    - (local_face_corners[:, 0] == reference_index).astype(float)
                )
                dref[:, reference_index - 1, 1] = (
                    (local_face_corners[:, 2] == reference_index).astype(float)
                    - (local_face_corners[:, 0] == reference_index).astype(float)
                )
            face_jacobian = np.einsum("bqka,bas->bqks", mapping_jacobian, dref, optimize=True)
            area_vector = np.cross(face_jacobian[:, :, :, 0], face_jacobian[:, :, :, 1])
            side_points.append(mapped_points)
            side_area_vector.append(area_vector)
            side_flux.append(flux)

        point_gap = np.linalg.norm(side_points[0] - side_points[1], axis=2)
        area0 = np.linalg.norm(side_area_vector[0], axis=2)
        area1 = np.linalg.norm(side_area_vector[1], axis=2)
        mean_area = 0.5 * (area0 + area1)
        relative_area_gap = np.abs(area0 - area1) / np.maximum(mean_area, np.finfo(float).tiny)
        center_direction = centers[cells[:, 1]] - centers[cells[:, 0]]
        normal0 = side_area_vector[0] / area0[:, :, None]
        flip0 = np.einsum("bqk,bk->bq", normal0, center_direction) < 0.0
        normal0[flip0] *= -1.0
        normal1 = side_area_vector[1] / area1[:, :, None]
        flip1 = np.einsum("bqk,bk->bq", normal1, -center_direction) < 0.0
        normal1[flip1] *= -1.0
        flux0 = np.einsum("bqk,bqk->bq", side_flux[0], normal0)
        flux1 = np.einsum("bqk,bqk->bq", side_flux[1], normal1)
        jump = flux0 + flux1
        area_weight = mean_area * quadrature_weights[None, :]
        area_side_sum += np.array(
            [
                np.sum(area0 * quadrature_weights[None, :]),
                np.sum(area1 * quadrature_weights[None, :]),
            ]
        )
        signed_side_flux += np.array(
            [
                np.sum(flux0 * area0 * quadrature_weights[None, :]),
                np.sum(flux1 * area1 * quadrature_weights[None, :]),
            ]
        )
        absolute_jump_integral += float(np.sum(np.abs(jump) * area_weight))
        squared_jump_integral += float(np.sum(jump**2 * area_weight))
        total_area += float(np.sum(area_weight))
        max_abs_flux_jump = max(max_abs_flux_jump, float(np.max(np.abs(jump))))
        max_point_gap = max(max_point_gap, float(np.max(point_gap)))
        max_relative_area_gap = max(max_relative_area_gap, float(np.max(relative_area_gap)))
        outward_area_gap = np.linalg.norm(
            side_area_vector[0] * np.where(flip0, -1.0, 1.0)[:, :, None]
            + side_area_vector[1] * np.where(flip1, -1.0, 1.0)[:, :, None],
            axis=2,
        )
        max_outward_area_vector_gap = max(
            max_outward_area_vector_gap,
            float(np.max(outward_area_gap / np.maximum(mean_area, np.finfo(float).tiny))),
        )

    return {
        "status": "measured",
        "internal_face_count": int(len(face_corners)),
        "quadrature_points_per_face": int(len(barycentric)),
        "area_side_0": float(area_side_sum[0]),
        "area_side_1": float(area_side_sum[1]),
        "area_relative_integral_gap": float(abs(area_side_sum[0] - area_side_sum[1]) / max(np.mean(area_side_sum), np.finfo(float).tiny)),
        "max_point_coordinate_gap": max_point_gap,
        "max_relative_pointwise_area_jacobian_gap": max_relative_area_gap,
        "max_relative_outward_area_vector_sum": max_outward_area_vector_gap,
        "raw_outward_mass_flux_side_0": float(signed_side_flux[0]),
        "raw_outward_mass_flux_side_1": float(signed_side_flux[1]),
        "signed_flux_jump_integral": float(signed_side_flux.sum()),
        "absolute_flux_jump_integral": absolute_jump_integral,
        "area_weighted_flux_jump_rms": float(np.sqrt(squared_jump_integral / total_area)),
        "max_abs_flux_jump": max_abs_flux_jump,
        "area_used_for_jump_norm": total_area,
    }


def evaluate_surface(points, tets, triangles, tags, adjacent_tets, phi, mach, order):
    triangle_bary, triangle_weights = triangle_quadrature(order)
    triangle_shape_values, triangle_reference_gradient = triangle_shape(triangle_bary)
    volume_barycentric = np.zeros((len(triangles), len(triangle_bary), 4))
    for face_index, (triangle, element) in enumerate(zip(triangles, adjacent_tets)):
        for local_face_node in range(3):
            volume_local_node = tets[element, :4].tolist().index(int(triangle[local_face_node]))
            volume_barycentric[face_index, :, volume_local_node] = triangle_bary[:, local_face_node]
    volume_shape_values, volume_reference_gradient = p2_shape(volume_barycentric)
    adjacent_coordinates = points[tets[adjacent_tets]]
    map_jacobian = np.einsum("bik,bqia->bqka", adjacent_coordinates, volume_reference_gradient, optimize=True)
    map_determinant = np.linalg.det(map_jacobian)
    if not np.all(np.isfinite(map_determinant)) or np.any(map_determinant <= 0.0):
        raise ValueError("tet10 Jacobian is nonpositive at a surface quadrature point")
    physical_gradient = np.einsum(
        "bqia,bqak->bqik", volume_reference_gradient, np.linalg.inv(map_jacobian), optimize=True
    )
    surface_velocity = np.einsum("bqik,bi->bqk", physical_gradient, phi[tets[adjacent_tets]], optimize=True)
    surface_q2 = np.einsum("bqk,bqk->bq", surface_velocity, surface_velocity)
    surface_beta = 1.0 + 0.2 * mach**2 * (1.0 - surface_q2)
    if not np.all(np.isfinite(surface_beta)) or np.any(surface_beta <= 0.0):
        raise ValueError("surface quadrature state violates beta > 0")
    surface_mach2 = mach**2 * surface_q2 / surface_beta
    if not np.all(np.isfinite(surface_mach2)) or np.any(surface_mach2 >= 1.0):
        raise ValueError("surface quadrature state is not strictly subsonic")
    surface_density = surface_beta ** (1.0 / (GAMMA - 1.0))
    surface_cp = pressure_coefficient(surface_q2, mach)
    tri_coordinates = points[triangles]
    tri_jacobian = np.einsum("bik,qia->bqka", tri_coordinates, triangle_reference_gradient, optimize=True)
    area_vector = np.cross(tri_jacobian[:, :, :, 0], tri_jacobian[:, :, :, 1])
    area = np.linalg.norm(area_vector, axis=2)
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("tri6 surface Jacobian must be finite and nonzero at every face quadrature point")
    normal = area_vector / area[:, :, None]
    normal_velocity = np.einsum("bqk,bqk->bq", surface_velocity, normal)
    normal_mass_flux = surface_density * normal_velocity
    area_weight = area * triangle_weights[None, :]
    return {
        "velocity": surface_velocity,
        "density": surface_density,
        "cp": surface_cp,
        "area_vector": area_vector,
        "normal": normal,
        "area": area,
        "area_weight": area_weight,
        "normal_velocity": normal_velocity,
        "normal_mass_flux": normal_mass_flux,
        "integrated_mass_flux": np.sum(area_weight * normal_mass_flux, axis=1),
        "max_local_mach": np.sqrt(np.max(surface_mach2, axis=1)),
    }

