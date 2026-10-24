import hashlib
import json
from pathlib import Path

import numpy as np


mesh_root = Path(__file__).resolve().parent / "meshes"
mesh_files = sorted(mesh_root.glob("*/mesh.npz"))
assert len(mesh_files) == 4, f"Expected four final meshes, found {len(mesh_files)}"

for mesh_path in mesh_files:
    mesh_dir = mesh_path.parent
    original_path = mesh_dir / "original_v1" / "mesh.npz"
    metadata_path = mesh_dir / "mesh.json"
    assert original_path.is_file(), f"Missing preserved v1 mesh: {original_path}"
    assert metadata_path.is_file(), f"Missing metadata: {metadata_path}"

    mesh = np.load(mesh_path)
    original = np.load(original_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    points = mesh["points"]
    tets = mesh["tets"]
    boundary_triangles = mesh["boundary_triangles"]
    boundary_tags = mesh["boundary_tags"]
    boundary_tet = mesh["boundary_tet"]

    assert points.ndim == 2 and points.shape[1] == 3
    assert tets.ndim == 2 and tets.shape[1] == 4
    assert boundary_triangles.ndim == 2 and boundary_triangles.shape[1] == 3
    assert len(boundary_triangles) == len(boundary_tags) == len(boundary_tet)
    assert np.isfinite(points).all()
    assert tets.min() >= 0 and tets.max() < len(points)
    assert boundary_triangles.min() >= 0
    assert boundary_triangles.max() < len(points)
    assert boundary_tet.min() >= 0 and boundary_tet.max() < len(tets)
    assert np.array_equal(points, original["points"]), f"{mesh_dir.name}: points changed"
    assert np.array_equal(tets, original["tets"]), f"{mesh_dir.name}: tets changed"

    face_nodes = np.concatenate(
        (
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ),
        axis=0,
    )
    face_tets = np.tile(np.arange(len(tets)), 4)
    sorted_faces = np.sort(face_nodes, axis=1)
    unique_faces, first_indices, face_counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True
    )
    assert np.all((face_counts == 1) | (face_counts == 2)), (
        f"{mesh_dir.name}: non-manifold tetrahedron face incidence"
    )
    expected_boundary = unique_faces[face_counts == 1]
    expected_boundary_tet = face_tets[first_indices[face_counts == 1]]
    actual_order = np.lexsort(
        (
            np.sort(boundary_triangles, axis=1)[:, 2],
            np.sort(boundary_triangles, axis=1)[:, 1],
            np.sort(boundary_triangles, axis=1)[:, 0],
        )
    )
    expected_order = np.lexsort(
        (expected_boundary[:, 2], expected_boundary[:, 1], expected_boundary[:, 0])
    )
    assert np.array_equal(
        np.sort(boundary_triangles[actual_order], axis=1),
        expected_boundary[expected_order],
    ), f"{mesh_dir.name}: boundary triangles do not match unique tet faces"
    expected_tet_by_face = {
        tuple(face): int(tet_id)
        for face, tet_id in zip(expected_boundary, expected_boundary_tet)
    }
    for triangle, tet_id in zip(boundary_triangles, boundary_tet):
        key = tuple(sorted(int(node) for node in triangle))
        assert key in expected_tet_by_face, f"{mesh_dir.name}: non-boundary face exported"
        assert int(tet_id) == expected_tet_by_face[key], (
            f"{mesh_dir.name}: boundary_tet does not own its triangle"
        )

    adjacent_tets = tets[boundary_tet]
    face_membership = np.any(
        adjacent_tets[:, :, None] == boundary_triangles[:, None, :], axis=2
    )
    assert np.count_nonzero(face_membership, axis=1).tolist() == [3] * len(boundary_tet)
    opposite_index = np.argmin(face_membership, axis=1)
    opposite_nodes = adjacent_tets[np.arange(len(boundary_tet)), opposite_index]
    face_points = points[boundary_triangles]
    area_vectors = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    ) / 2.0
    outward_dot = np.einsum(
        "ij,ij->i",
        area_vectors,
        points[opposite_nodes] - face_points[:, 0],
    )
    assert np.all(outward_dot < 0), f"{mesh_dir.name}: boundary normal is not outward"

    tetra_points = points[tets]
    signed_volume = np.linalg.det(
        np.stack(
            (
                tetra_points[:, 1] - tetra_points[:, 0],
                tetra_points[:, 2] - tetra_points[:, 0],
                tetra_points[:, 3] - tetra_points[:, 0],
            ),
            axis=1,
        )
    ) / 6.0
    assert np.all(signed_volume > 0), f"{mesh_dir.name}: non-positive tet volume"

    tag_area_vectors = {}
    for tag in np.unique(boundary_tags):
        tag_area_vectors[int(tag)] = np.sum(
            area_vectors[boundary_tags == tag], axis=0
        )
    area_scale = np.sum(np.linalg.norm(area_vectors, axis=1))
    area_tolerance = 1.0e-10 * area_scale
    assert np.linalg.norm(tag_area_vectors[100]) <= area_tolerance, (
        f"{mesh_dir.name}: farfield area vector does not close"
    )
    if metadata["shape"] in ("sphere", "ellipsoid"):
        assert np.linalg.norm(tag_area_vectors[10]) <= area_tolerance, (
            f"{mesh_dir.name}: body area vector does not close"
        )
    else:
        assert np.linalg.norm(tag_area_vectors[20] + tag_area_vectors[30]) <= area_tolerance, (
            f"{mesh_dir.name}: combined wing/fuselage wall area vector does not close"
        )
        axes = np.asarray(metadata["shape_parameters"]["fuselage_ellipsoid_semiaxes"])
        vertex_level = np.sum((face_points / axes) ** 2, axis=2)
        fuselage_class = np.all(np.abs(vertex_level - 1.0) <= 0.15, axis=1)
        assert np.all(boundary_tags[fuselage_class] == 20)
        assert np.all(boundary_tags[~fuselage_class & (boundary_tags != 100)] == 30)
        assert metadata["coordinate_axes"]["wing_span_axis"] == "z"
        assert "Not aligned" in metadata["coordinate_axes"]["fem_alpha_xz_alignment"]
        assert "not exact NACA 0012" in metadata["shape_parameters"]["airfoil"]

    array_hash = hashlib.sha256()
    for array in (
        points,
        tets,
        boundary_triangles,
        boundary_tags,
        boundary_tet,
    ):
        array_hash.update(np.ascontiguousarray(array).tobytes())
    assert array_hash.hexdigest() == metadata["sha256_arrays"]
    source_hash = hashlib.sha256(
        (Path(__file__).resolve().parent / "build_mesh.py").read_bytes()
    ).hexdigest()
    assert source_hash == metadata["source_sha256"]
    assert metadata["metadata_revision"] == 2

    print(
        mesh_dir.name,
        "points=", len(points),
        "tets=", len(tets),
        "boundary_faces=", len(boundary_triangles),
        "tags=", metadata["quality"]["boundary_tag_counts"],
        "area_vectors=", {
            str(tag): np.round(vector, 12).tolist()
            for tag, vector in tag_area_vectors.items()
        },
    )

print("All four boundary mesh checks passed.")

curved_mesh_files = sorted((mesh_root / "curved").glob("*/mesh.npz"))
for mesh_path in curved_mesh_files:
    mesh_dir = mesh_path.parent
    metadata = json.loads((mesh_dir / "mesh.json").read_text(encoding="utf-8"))
    mesh = np.load(mesh_path)
    points = mesh["points"]
    tets = mesh["tets"]
    boundary_triangles = mesh["boundary_triangles"]
    boundary_tags = mesh["boundary_tags"]
    boundary_tet = mesh["boundary_tet"]

    assert metadata["status"] == "success"
    assert tets.ndim == 2 and tets.shape[1] == 10
    assert boundary_triangles.ndim == 2 and boundary_triangles.shape[1] == 6
    assert len(boundary_triangles) == len(boundary_tags) == len(boundary_tet)
    assert boundary_tet.min() >= 0 and boundary_tet.max() < len(tets)

    lattice_order = metadata["curved_jacobian_check"]["lattice_order"]
    lattice_points = []
    for l0 in range(lattice_order + 1):
        for l1 in range(lattice_order - l0 + 1):
            for l2 in range(lattice_order - l0 - l1 + 1):
                l3 = lattice_order - l0 - l1 - l2
                lattice_points.append(
                    np.array([l0, l1, l2, l3], dtype=np.float64) / lattice_order
                )
    lattice_points = np.asarray(lattice_points)
    d_lambda_d_xi = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    d_shape_d_lambda = np.zeros((len(lattice_points), 10, 4), dtype=np.float64)
    for vertex in range(4):
        d_shape_d_lambda[:, vertex, vertex] = 4.0 * lattice_points[:, vertex] - 1.0
    local_edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    for edge_index, (first, second) in enumerate(local_edges):
        d_shape_d_lambda[:, 4 + edge_index, first] = 4.0 * lattice_points[:, second]
        d_shape_d_lambda[:, 4 + edge_index, second] = 4.0 * lattice_points[:, first]
    d_shape_d_xi = np.einsum("qnk,ki->qni", d_shape_d_lambda, d_lambda_d_xi)
    recomputed_minimum_jacobian = np.inf
    recomputed_nonpositive_samples = 0
    for start in range(0, len(tets), 256):
        element_points = points[tets[start : start + 256]]
        jacobians = np.einsum("enp,qni->eqpi", element_points, d_shape_d_xi)
        determinants = np.linalg.det(jacobians)
        recomputed_minimum_jacobian = min(
            recomputed_minimum_jacobian, float(determinants.min())
        )
        recomputed_nonpositive_samples += int(np.count_nonzero(determinants <= 0.0))
    assert recomputed_nonpositive_samples == 0
    assert recomputed_minimum_jacobian > 0

    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    local_edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    boundary_face_owners = {}
    for tet_index, tet in enumerate(tets):
        for local_face in local_faces:
            face = tuple(sorted(int(tet[index]) for index in local_face))
            boundary_face_owners.setdefault(face, []).append(tet_index)

    linear_face_incidence = {}
    for tet_index, tet in enumerate(tets):
        for local_face in local_faces:
            face = tuple(sorted(int(tet[index]) for index in local_face))
            linear_face_incidence[face] = linear_face_incidence.get(face, 0) + 1
    for triangle, tet_index in zip(boundary_triangles, boundary_tet):
        vertex_face = tuple(sorted(int(node) for node in triangle[:3]))
        assert linear_face_incidence[vertex_face] == 1
        assert boundary_face_owners[vertex_face] == [int(tet_index)]
        parent_tet = tets[int(tet_index)]
        edge_node_by_key = {
            tuple(sorted((int(parent_tet[first]), int(parent_tet[second])))):
            int(parent_tet[4 + edge_index])
            for edge_index, (first, second) in enumerate(local_edges)
        }
        expected_edge_nodes = [
            edge_node_by_key[tuple(sorted((int(triangle[0]), int(triangle[1]))))],
            edge_node_by_key[tuple(sorted((int(triangle[1]), int(triangle[2]))))],
            edge_node_by_key[tuple(sorted((int(triangle[2]), int(triangle[0]))))],
        ]
        assert np.array_equal(triangle[3:], expected_edge_nodes), (
            f"{mesh_dir.name}: triangle6 nodes disagree with adjacent tetra10 edges"
        )

    adjacent_tets = tets[boundary_tet]
    face_membership = np.any(
        adjacent_tets[:, :4, None] == boundary_triangles[:, None, :3], axis=2
    )
    assert np.count_nonzero(face_membership, axis=1).tolist() == [3] * len(boundary_tet)
    opposite_index = np.argmin(face_membership, axis=1)
    opposite_nodes = adjacent_tets[np.arange(len(boundary_tet)), opposite_index]
    face_points = points[boundary_triangles[:, :3]]
    area_vectors = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    ) / 2.0
    outward_dot = np.einsum(
        "ij,ij->i",
        area_vectors,
        points[opposite_nodes] - face_points[:, 0],
    )
    assert np.all(outward_dot < 0), f"{mesh_dir.name}: P2 boundary is not outward"

    tag_area_vectors = {
        int(tag): np.sum(area_vectors[boundary_tags == tag], axis=0)
        for tag in np.unique(boundary_tags)
    }
    area_scale = np.sum(np.linalg.norm(area_vectors, axis=1))
    area_tolerance = 1.0e-8 * area_scale
    assert np.linalg.norm(tag_area_vectors[100]) <= area_tolerance
    if metadata["shape"] == "wingbody":
        assert set(np.unique(boundary_tags)) == {20, 30, 100}
        assert np.linalg.norm(tag_area_vectors[20] + tag_area_vectors[30]) <= area_tolerance
        assert metadata["coordinate_axes"]["wing_span_axis"] == "y"
        assert metadata["coordinate_axes"]["wing_thickness_axis"] == "z"
        assert metadata["boundary_tag_provenance"]["classification"].startswith("OCC")
        assert metadata["cad_residuals"]["wing_to_occ_surface_max_distance"] < 1.0e-7
    else:
        assert set(np.unique(boundary_tags)) == {10, 100}
        assert np.linalg.norm(tag_area_vectors[10]) <= area_tolerance

    jacobian_check = metadata["curved_jacobian_check"]
    assert lattice_order >= 12
    assert jacobian_check["nonpositive_jacobian_samples"] == 0
    assert np.isclose(
        recomputed_minimum_jacobian,
        jacobian_check["minimum_jacobian_determinant"],
        rtol=1.0e-10,
        atol=1.0e-13,
    )

    array_hash = hashlib.sha256()
    for array in (points, tets, boundary_triangles, boundary_tags, boundary_tet):
        array_hash.update(np.ascontiguousarray(array).tobytes())
    assert array_hash.hexdigest() == metadata["sha256_arrays"]
    source_hash = hashlib.sha256(
        (Path(__file__).resolve().parent / "build_curved_mesh.py").read_bytes()
    ).hexdigest()
    if metadata["source_sha256"] == source_hash:
        source_hash_status = "current"
    else:
        assert mesh_dir.name in {
            "ellipsoid_s05_v2",
            "ellipsoid_s05_v3",
            "wingbody_s08_v1",
        }, (
            f"{mesh_dir.name}: unexpected stale build source hash"
        )
        source_hash_status = "preserved-legacy"

    print(
        mesh_dir.name,
        "points=", len(points),
        "tetra10=", len(tets),
        "triangle6=", len(boundary_triangles),
        "tags=", {tag: int(np.count_nonzero(boundary_tags == tag)) for tag in np.unique(boundary_tags)},
        "minimum_jacobian=", jacobian_check["minimum_jacobian_determinant"],
        "source_hash=", source_hash_status,
    )

print("All curved mesh boundary checks passed.")
