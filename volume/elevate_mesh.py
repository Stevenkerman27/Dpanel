import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Elevate a P1 sphere or ellipsoid tetra mesh to curved P2."
)
parser.add_argument("--input", type=Path, required=True, help="P1 mesh directory")
parser.add_argument("--out", type=Path, required=True, help="New P2 output directory")
parser.add_argument(
    "--lattice-order",
    type=int,
    default=12,
    help="Barycentric lattice degree used for curved Jacobian checks",
)
args = parser.parse_args()

input_dir = args.input.resolve()
output_dir = args.out.resolve()
if not input_dir.is_dir():
    parser.error(f"P1 mesh directory does not exist: {input_dir}")
if input_dir == output_dir:
    parser.error("P2 output directory must differ from the P1 input directory")
if output_dir.exists() and any(output_dir.iterdir()):
    parser.error(f"Refusing to overwrite non-empty output directory: {output_dir}")
if args.lattice_order < 2:
    parser.error("--lattice-order must be at least 2")

source_npz_path = input_dir / "mesh.npz"
source_json_path = input_dir / "mesh.json"
if not source_npz_path.is_file() or not source_json_path.is_file():
    parser.error(f"P1 input must contain mesh.npz and mesh.json: {input_dir}")

source_metadata = json.loads(source_json_path.read_text(encoding="utf-8"))
shape = source_metadata["shape"]
if shape not in ("sphere", "ellipsoid"):
    parser.error(f"P2 analytic curving supports sphere/ellipsoid, not {shape}")

source_mesh = np.load(source_npz_path)
points_p1 = np.asarray(source_mesh["points"], dtype=np.float64)
tets_p1 = np.asarray(source_mesh["tets"], dtype=np.int64)
triangles_p1 = np.asarray(source_mesh["boundary_triangles"], dtype=np.int64)
boundary_tags = np.asarray(source_mesh["boundary_tags"], dtype=np.int32)
boundary_tet = np.asarray(source_mesh["boundary_tet"], dtype=np.int64)
radius = float(source_metadata["reference_radius"])
farfield_radius = float(source_metadata["farfield_radius"])
if shape == "sphere":
    body_axes = np.array([radius, radius, radius], dtype=np.float64)
else:
    body_axes = np.asarray(
        source_metadata["shape_parameters"]["semiaxes"], dtype=np.float64
    )

edge_pairs = np.concatenate(
    (
        tets_p1[:, [0, 1]],
        tets_p1[:, [1, 2]],
        tets_p1[:, [2, 0]],
        tets_p1[:, [0, 3]],
        tets_p1[:, [1, 3]],
        tets_p1[:, [2, 3]],
    ),
    axis=0,
)
unique_edges, edge_inverse = np.unique(
    np.sort(edge_pairs, axis=1), axis=0, return_inverse=True
)
edge_inverse = edge_inverse.reshape(6, len(tets_p1))
edge_midpoint_nodes = len(points_p1) + edge_inverse.T
midpoints = 0.5 * (
    points_p1[unique_edges[:, 0]] + points_p1[unique_edges[:, 1]]
)
points_p2 = np.concatenate((points_p1, midpoints), axis=0)

edge_to_midpoint = {
    tuple(edge): len(points_p1) + edge_index
    for edge_index, edge in enumerate(unique_edges)
}
boundary_edge_pairs = np.concatenate(
    (
        triangles_p1[:, [0, 1]],
        triangles_p1[:, [1, 2]],
        triangles_p1[:, [2, 0]],
    ),
    axis=0,
)
boundary_edge_tags = np.tile(boundary_tags, 3)
edge_projection = {}
for edge, tag in zip(boundary_edge_pairs, boundary_edge_tags):
    if tag not in (10, 100):
        continue
    key = tuple(sorted((int(edge[0]), int(edge[1]))))
    if key in edge_projection and edge_projection[key] != int(tag):
        raise RuntimeError(f"Boundary edge has conflicting tags: {key}")
    edge_projection[key] = int(tag)

projected_body_count = 0
projected_farfield_count = 0
for edge, tag in edge_projection.items():
    midpoint_index = edge_to_midpoint[edge]
    midpoint = points_p2[midpoint_index]
    if tag == 10:
        if shape == "sphere":
            direction_length = np.linalg.norm(midpoint)
            if direction_length == 0:
                raise RuntimeError(f"Sphere body edge midpoint is at its center: {edge}")
            points_p2[midpoint_index] = radius * midpoint / direction_length
        else:
            scaled_midpoint = midpoint / body_axes
            scaled_length = np.linalg.norm(scaled_midpoint)
            if scaled_length == 0:
                raise RuntimeError(f"Ellipsoid body edge midpoint is at its center: {edge}")
            points_p2[midpoint_index] = body_axes * scaled_midpoint / scaled_length
        projected_body_count += 1
    else:
        direction_length = np.linalg.norm(midpoint)
        if direction_length == 0:
            raise RuntimeError(f"Farfield edge midpoint is at the origin: {edge}")
        points_p2[midpoint_index] = farfield_radius * midpoint / direction_length
        projected_farfield_count += 1

tets_p2 = np.column_stack(
    (
        tets_p1,
        edge_midpoint_nodes[:, 0],
        edge_midpoint_nodes[:, 1],
        edge_midpoint_nodes[:, 2],
        edge_midpoint_nodes[:, 3],
        edge_midpoint_nodes[:, 4],
        edge_midpoint_nodes[:, 5],
    )
).astype(np.int64, copy=False)
triangles_p2 = np.column_stack(
    (
        triangles_p1,
        np.asarray(
            [
                edge_to_midpoint[tuple(sorted((int(row[0]), int(row[1]))))]
                for row in triangles_p1
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                edge_to_midpoint[tuple(sorted((int(row[1]), int(row[2]))))]
                for row in triangles_p1
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                edge_to_midpoint[tuple(sorted((int(row[2]), int(row[0]))))]
                for row in triangles_p1
            ],
            dtype=np.int64,
        ),
    )
).astype(np.int64, copy=False)

lattice_points = []
for l0 in range(args.lattice_order + 1):
    for l1 in range(args.lattice_order - l0 + 1):
        for l2 in range(args.lattice_order - l0 - l1 + 1):
            l3 = args.lattice_order - l0 - l1 - l2
            lattice_points.append(
                np.array([l0, l1, l2, l3], dtype=np.float64) / args.lattice_order
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
edge_local_vertices = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
for edge_index, (first, second) in enumerate(edge_local_vertices):
    shape_index = 4 + edge_index
    d_shape_d_lambda[:, shape_index, first] = 4.0 * lattice_points[:, second]
    d_shape_d_lambda[:, shape_index, second] = 4.0 * lattice_points[:, first]
d_shape_d_xi = np.einsum(
    "qnk,ki->qni", d_shape_d_lambda, d_lambda_d_xi
)

minimum_jacobian = np.inf
maximum_jacobian = -np.inf
negative_jacobian_samples = 0
chunk_size = 256
for start in range(0, len(tets_p2), chunk_size):
    stop = min(start + chunk_size, len(tets_p2))
    element_points = points_p2[tets_p2[start:stop]]
    jacobians = np.einsum("enp,qni->eqpi", element_points, d_shape_d_xi)
    determinants = np.linalg.det(jacobians)
    minimum_jacobian = min(minimum_jacobian, float(determinants.min()))
    maximum_jacobian = max(maximum_jacobian, float(determinants.max()))
    negative_jacobian_samples += int(np.count_nonzero(determinants <= 0.0))

array_hash = hashlib.sha256()
for array in (
    points_p2,
    tets_p2,
    triangles_p2,
    boundary_tags,
    boundary_tet,
):
    array_hash.update(np.ascontiguousarray(array).tobytes())
source_npz_hash = hashlib.sha256(source_npz_path.read_bytes()).hexdigest()
elevate_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

output_dir.mkdir(parents=True, exist_ok=True)
mesh_metadata = dict(source_metadata)
mesh_metadata["element_order"] = 2
mesh_metadata["element_types"] = {
    "volume": "curved tetra10",
    "boundary": "curved triangle6",
}
mesh_metadata["counts"] = {
    "points": len(points_p2),
    "tets": len(tets_p2),
    "boundary_triangles": len(triangles_p2),
    "boundary_tets": len(boundary_tet),
    "unique_edges": len(unique_edges),
}
mesh_metadata["p2_projection"] = {
    "body_tag": 10,
    "body_projected_edges": projected_body_count,
    "body_method": (
        "Sphere radial projection from origin."
        if shape == "sphere"
        else "Ellipsoid radial normalization: axes * ((midpoint / axes) / "
        "norm(midpoint / axes)); this is not the Euclidean nearest-point projection."
    ),
    "farfield_tag": 100,
    "farfield_projected_edges": projected_farfield_count,
    "farfield_method": "Sphere radial projection from origin.",
    "all_other_edge_midpoints": "Unchanged arithmetic midpoint.",
}
mesh_metadata["curved_jacobian_check"] = {
    "method": "Dense barycentric lattice sampling of the quadratic tetra10 map.",
    "lattice_order": args.lattice_order,
    "samples_per_tet": len(lattice_points),
    "minimum_jacobian_determinant": minimum_jacobian,
    "maximum_jacobian_determinant": maximum_jacobian,
    "nonpositive_jacobian_samples": negative_jacobian_samples,
    "acceptance": "All sampled Jacobian determinants must be strictly positive.",
}
mesh_metadata["p1_source"] = {
    "directory": str(input_dir),
    "sha256_mesh_npz": source_npz_hash,
    "sha256_arrays": source_metadata["sha256_arrays"],
}
mesh_metadata["sha256_arrays"] = array_hash.hexdigest()
mesh_metadata["elevate_mesh_source_sha256"] = elevate_source_hash
mesh_metadata["revision_reason"] = (
    "P1 mesh elevated to shared-edge curved P2 tetra10/triangle6; only tagged "
    "analytic sphere/ellipsoid boundary edges and farfield sphere edges were projected."
)

if negative_jacobian_samples:
    mesh_metadata["status"] = "failed_negative_jacobian"
    mesh_metadata["failure"] = (
        "Curved P2 mesh rejected: at least one sampled Jacobian determinant "
        "was nonpositive. Projected nodes were not clipped or moved."
    )
    (output_dir / "failedmeta.json").write_text(
        json.dumps(mesh_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    raise SystemExit(
        f"Rejected P2 mesh: min sampled Jacobian determinant={minimum_jacobian:.9g}"
    )

np.savez_compressed(
    output_dir / "mesh.npz",
    points=points_p2,
    tets=tets_p2,
    boundary_triangles=triangles_p2,
    boundary_tags=boundary_tags,
    boundary_tet=boundary_tet,
)
mesh_metadata["status"] = "success"
(output_dir / "mesh.json").write_text(
    json.dumps(mesh_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
)
print(
    json.dumps(
        {
            "output": str(output_dir),
            "counts": mesh_metadata["counts"],
            "minimum_jacobian_determinant": minimum_jacobian,
            "nonpositive_jacobian_samples": negative_jacobian_samples,
        },
        ensure_ascii=True,
    )
)
