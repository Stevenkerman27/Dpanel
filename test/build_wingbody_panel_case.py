"""Extract a reproducible flat-panel fixture from an existing CAD union mesh.

The P2 fluid mesh contains duplicated wake-bank nodes. Collapse only its explicit
original_node map, keep wall tags, and orient faces toward the adjacent fluid.
No distance-based welding, overlapping closed bodies, or TE angle detection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--subdivide", action="store_true", help="Split each curved Tri6 into four flat triangles")
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous fixtures"
data = np.load(args.mesh, allow_pickle=False)
metadata = json.loads(args.mesh.with_suffix(".json").read_text(encoding="utf-8"))
wall = data["boundary_tags"] != 100
boundary = data["boundary_triangles"][wall]
tags = data["boundary_tags"][wall]
points = data["points"]
tet_centers = points[data["tets"][data["boundary_tet"][wall], :4]].mean(axis=1)

# Orient the parent using its adjacent fluid cell, then inherit that winding.
# A curved child plane need not place the parent tet center on its outer side.
corners = points[boundary[:, :3]]
parent_area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
reverse = np.einsum("ij,ij->i", parent_area, tet_centers - corners.mean(axis=1)) < 0
boundary[reverse] = boundary[reverse][:, [0, 2, 1, 5, 4, 3]]
if args.subdivide:
    local = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    faces = boundary[:, local].reshape(-1, 3)
    tags = np.repeat(tags, 4)
    tet_centers = np.repeat(tet_centers, 4, axis=0)
else:
    faces = boundary[:, :3].copy()

original = data["original_node"]
node_ids, inverse = np.unique(original[faces], return_inverse=True)
coordinates = np.empty((len(node_ids), 3))
for i, original_id in enumerate(node_ids):
    duplicates = points[original == original_id]
    assert np.max(abs(duplicates - duplicates[0])) < 1e-12 * metadata["chord"]
    coordinates[i] = duplicates[0]
faces = inverse.reshape(-1, 3)
assert len(np.unique(coordinates, axis=0)) == len(coordinates)
assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
triangles = coordinates[faces]
area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
areas = np.linalg.norm(area_vectors, axis=1)
assert np.isfinite(coordinates).all() and np.all(areas > 0)
normals = area_vectors / areas[:, None]
edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
face_ids = np.tile(np.arange(len(faces)), 3)
unique, edge_inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts == 2), "Public exterior must be closed"
order = np.argsort(edge_inverse).reshape(-1, 2)
assert np.all(edges[order[:, 0]] == edges[order[:, 1], ::-1]), "Inconsistent winding"
pairs = face_ids[order]
graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(faces), len(faces)))
components, _ = connected_components(graph, directed=False)
assert components == 1, "This first fixture must be one joined exterior"
volume = np.sum(triangles.mean(axis=1) * area_vectors) / 3
assert volume > 0 and np.linalg.norm(area_vectors.sum(axis=0)) / areas.sum() < 1e-12
assert set(tags) == {20, 30}, "Expect CAD fuselage=20 and wing=30"
seam = tags[pairs[:, 0]] != tags[pairs[:, 1]]
assert seam.any(), "Wing and fuselage must share actual mesh edges"

declared_te = data["trailing_edge_points"]
assert np.max(abs(declared_te[:, 0] - declared_te[0, 0])) < 1e-12
assert np.max(abs(declared_te[:, 2])) < 1e-12
chord = float(metadata["chord"])
tolerance = 1e-7 * chord
on_te = (abs(coordinates[:, 0] - declared_te[0, 0]) < tolerance) & (abs(coordinates[:, 2]) < tolerance)
te_mask = on_te[unique].all(axis=1)
te_edges = unique[te_mask].copy()
te_pairs = pairs[te_mask]
assert len(te_edges) >= 4 and np.all(tags[te_pairs] == 30)
reverse = coordinates[te_edges[:, 0], 1] > coordinates[te_edges[:, 1], 1]
te_edges[reverse] = te_edges[reverse, ::-1]
permutation = np.argsort(coordinates[te_edges[:, 0], 1])
te_edges, te_pairs = te_edges[permutation], te_pairs[permutation]
assert np.all(te_edges[:-1, 1] == te_edges[1:, 0]), "Require continuous exposed TE for this first loop"
assert abs(coordinates[te_edges[0, 0], 1] - declared_te[0, 1]) < tolerance
assert abs(coordinates[te_edges[-1, 1], 1] - declared_te[-1, 1]) < tolerance
upper = te_pairs[np.arange(len(te_pairs)), np.argmax(normals[te_pairs, 2], axis=1)]
lower = te_pairs[np.arange(len(te_pairs)), np.argmin(normals[te_pairs, 2], axis=1)]
assert np.all(normals[upper, 2] > 0) and np.all(normals[lower, 2] < 0)
assert len(np.unique(upper)) == len(upper) and len(np.unique(lower)) == len(lower)
assert coordinates[faces[tags == 20], 0].max() < declared_te[0, 0] - tolerance, (
    "Body/wake contact needs a separate cut/jump contract; use a body ending ahead of TE here"
)

neighbors = [[] for _ in faces]
points_to_faces = [[] for _ in coordinates]
for first, second in pairs:
    neighbors[first].append(int(second))
    neighbors[second].append(int(first))
for i, face in enumerate(faces):
    for node in face:
        points_to_faces[node].append(i)
quality = 4 * np.sqrt(3) * areas / np.sum(
    (triangles - triangles[:, [1, 2, 0]]) ** 2, axis=(1, 2)
)
record = dict(
    description="Closed CAD Boolean union: NACA0012 wing and forward ellipsoid fuselage",
    scope="Full model, straight exposed TE, fuselage ends upstream of TE; fixed topology deformations",
    coordinates=coordinates.tolist(), faces=faces.tolist(), component_ids=tags.tolist(),
    component_names={"20": "body", "30": "wing"},
    trailing_edges=te_edges.tolist(), upper_te_faces=upper.tolist(), lower_te_faces=lower.tolist(),
    neighbors=neighbors, points_to_faces=points_to_faces,
    seam_edges=unique[seam].tolist(), seam_faces=np.unique(pairs[seam]).tolist(),
    reference_area=float(metadata["reference_area"]), reference_chord=chord,
    moment_reference=[0.25 * chord, 0., 0.],
    audit=dict(vertices=len(coordinates), panels=len(faces), closed_components=int(components),
               wing_panels=int(sum(tags == 30)), body_panels=int(sum(tags == 20)),
               seam_edges=int(sum(seam)), te_edges=len(te_edges), volume=float(volume),
               minimum_triangle_quality=float(quality.min()), subdivision=args.subdivide,
               normalized_vector_area_closure=float(np.linalg.norm(area_vectors.sum(axis=0)) / areas.sum()),
               exposed_boundary_source="CAD union fluid boundary; farfield excluded; original_node welding only"),
    provenance=dict(input_mesh=str(args.mesh.resolve()),
                    input_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                    extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    original_cad_hashes=metadata.get("input_sha256", {})),
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(record["audit"], indent=2), flush=True)
