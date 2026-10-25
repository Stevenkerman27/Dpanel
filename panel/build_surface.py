"""Extract a public exterior and its actual fluid-side planar wake cut.

The input is an existing conforming CAD/P2 mesh. No buried wing panels, fake
central trailing edge, distance welding, or wake sheet inside a body is added.
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
parser.add_argument("--subdivide", action="store_true")
parser.add_argument("--wake-length", type=float, default=30.)
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists(), "Preserve previous fixtures"
data = np.load(args.mesh, allow_pickle=False)
metadata = json.loads(args.mesh.with_suffix(".json").read_text(encoding="utf-8"))
points, original = data["points"], data["original_node"]
chord = float(metadata["chord"])
origin = np.asarray(metadata["normalization_origin"])
if metadata.get("surface_only", False):
    assert not args.subdivide, "Surface-only input contains linear triangles"
    wall, tags, wake = data["wall_faces"], data["wall_tags"], data["wake_faces"]
else:
    wall_mask = data["boundary_tags"] != 100
    wall = data["boundary_triangles"][wall_mask].copy()
    tags = data["boundary_tags"][wall_mask]
    fluid_centers = points[data["tets"][data["boundary_tet"][wall_mask], :4]].mean(axis=1)
    corners = points[wall[:, :3]]
    normal_area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    reverse = np.einsum("ij,ij->i", normal_area, fluid_centers - corners.mean(axis=1)) < 0
    wall[reverse] = wall[reverse][:, [0, 2, 1, 5, 4, 3]]
    wake = data["wake_upper"].copy()
    corners = points[wake[:, :3]]
    reverse = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2] < 0
    wake[reverse] = wake[reverse][:, [0, 2, 1, 5, 4, 3]]
if args.subdivide:
    local = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    wall = wall[:, local].reshape(-1, 3)
    tags = np.repeat(tags, 4)
    wake = wake[:, local].reshape(-1, 3)
else:
    wall, wake = wall[:, :3], wake[:, :3]
joined = np.concatenate((wall, wake))
node_ids, inverse = np.unique(original[joined], return_inverse=True)
coordinates = np.empty((len(node_ids), 3))
for i, node in enumerate(node_ids):
    copies = points[original == node]
    assert np.max(abs(copies - copies[0])) < 1e-12 * chord
    coordinates[i] = copies[0]
joined = inverse.reshape(-1, 3)
faces, wake_faces = joined[:len(wall)], joined[len(wall):]
assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
triangles = coordinates[faces]
area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
areas = np.linalg.norm(area_vectors, axis=1)
assert np.all(areas > 0) and np.isfinite(coordinates).all()
wall_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
face_ids = np.tile(np.arange(len(faces)), 3)
unique, inverse, counts = np.unique(np.sort(wall_edges, axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts == 2), "Public wall must be closed"
order = np.argsort(inverse).reshape(-1, 2)
assert np.all(wall_edges[order[:, 0]] == wall_edges[order[:, 1], ::-1])
pairs = face_ids[order]
wall_lookup = {tuple(edge): i for i, edge in enumerate(unique)}
graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(faces), len(faces)))
components, labels = connected_components(graph, directed=False)
volumes = np.bincount(labels, weights=np.sum(triangles.mean(axis=1) * area_vectors, axis=1) / 3)
assert np.all(volumes > 0)
seam = tags[pairs[:, 0]] != tags[pairs[:, 1]]
neighbors = [[] for _ in faces]
for a, b in pairs:
    neighbors[a].append(int(b))
    neighbors[b].append(int(a))

wake_edges = np.concatenate((wake_faces[:, [0, 1]], wake_faces[:, [1, 2]], wake_faces[:, [2, 0]]))
unique_w, inverse_w, counts_w = np.unique(np.sort(wake_edges, axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts_w <= 2)
order_w = np.argsort(inverse_w, kind="stable")
te_polyline = data["trailing_edge_points"]
tolerance = 1e-7 * chord
boundary = unique_w[counts_w == 1]
cut_edges, cut_pairs, cut_kind, te_edges, te_pairs = [], [], [], [], []
for edge in boundary:
    if tuple(edge) not in wall_lookup:
        continue
    pair = pairs[wall_lookup[tuple(edge)]]
    xyz = coordinates[edge]
    distance = abs(xyz[:, 0] - np.interp(xyz[:, 1], te_polyline[:, 1], te_polyline[:, 0]))
    on_te = bool(np.all(distance < tolerance) and np.all(abs(xyz[:, 2] - origin[2]) < tolerance))
    bank_z = triangles[pair].mean(axis=1)[:, 2] - origin[2]
    assert bank_z.max() > 0 and bank_z.min() < 0, "Cut edge must have two distinct planar wall banks"
    pair = pair[np.argsort(-bank_z)]
    edge = edge[np.argsort(coordinates[edge, 1])]
    cut_edges.append(edge.tolist())
    cut_pairs.append(pair.tolist())
    cut_kind.append("trailing_edge" if on_te else "smooth_wall_contact")
    if on_te:
        te_edges.append(edge.tolist())
        te_pairs.append(pair.tolist())
assert te_edges, "A lifting cut needs actual exposed trailing-edge support"

# Extend only the mesh's downstream fluid boundary. The body hole stays intact.
far_x = float(coordinates[wake_faces, 0].max())
end_x = float(te_polyline[:, 0].max() + args.wake_length * chord)
assert end_x > far_x, "Requested wake end must extend beyond the imported cut"
outflow = boundary[np.all(abs(coordinates[boundary, 0] - far_x) < tolerance, axis=1)]
assert len(outflow)
outflow_nodes = np.unique(outflow)
far_nodes = coordinates[outflow_nodes].copy()
far_nodes[:, 0] = end_x
far_map = dict(zip(outflow_nodes.tolist(), np.arange(len(coordinates), len(coordinates) + len(far_nodes)).tolist()))
coordinates = np.concatenate((coordinates, far_nodes))
extension = []
for a, b in outflow:
    if coordinates[a, 1] > coordinates[b, 1]:
        a, b = b, a
    extension.extend([[int(a), far_map[int(a)], far_map[int(b)]], [int(a), far_map[int(b)], int(b)]])
wake_faces = np.concatenate((wake_faces, np.asarray(extension)))
wake_triangles = coordinates[wake_faces]
wake_area_vectors = np.cross(wake_triangles[:, 1] - wake_triangles[:, 0], wake_triangles[:, 2] - wake_triangles[:, 0]) / 2
assert np.all(wake_area_vectors[:, 2] > 0) and np.max(abs(wake_area_vectors[:, :2])) < 1e-10 * chord**2
stations = data["gamma_stations"]
station_dof = data["gamma_station_dof"]
assert stations[0] < stations[-1] and station_dof[0] == station_dof[-1] == -1
wake_centers = wake_triangles.mean(axis=1)
interval = np.minimum(np.searchsorted(stations, wake_centers[:, 1], side="right") - 1, len(stations) - 2)
assert np.all(interval >= 0)
assert np.all(wake_triangles[:, :, 1].min(axis=1) >= stations[interval] - tolerance)
assert np.all(wake_triangles[:, :, 1].max(axis=1) <= stations[interval + 1] + tolerance)
quality = 4 * np.sqrt(3) * areas / np.sum((triangles - triangles[:, [1, 2, 0]])**2, axis=(1, 2))
te_pairs = np.asarray(te_pairs)
record = dict(
    schema="dpanel.planar_cut.v1",
    coordinates=coordinates.tolist(), faces=faces.tolist(), component_ids=tags.tolist(),
    component_names={"10": "wing", "20": "body", "30": "wing"},
    neighbors=neighbors, seam_edges=unique[seam].tolist(), seam_faces=np.unique(pairs[seam]).tolist(),
    trailing_edges=te_edges, upper_te_faces=te_pairs[:, 0].tolist(), lower_te_faces=te_pairs[:, 1].tolist(),
    cut_edges=cut_edges, cut_face_pairs=cut_pairs, cut_kind=cut_kind,
    wake_faces=wake_faces.tolist(), gamma_stations=stations.tolist(), gamma_station_dof=station_dof.tolist(),
    origin=origin.tolist(), reference_area=float(metadata["reference_area"]), reference_chord=chord,
    moment_reference=(origin + [0.25 * chord, 0., 0.]).tolist(),
    audit=dict(vertices=len(coordinates), wall_panels=len(faces), wake_panels=len(wake_faces),
               closed_components=int(components), component_volumes=volumes.tolist(),
               wing_panels=int(sum(tags != 20)), body_panels=int(sum(tags == 20)),
               seam_edges=int(sum(seam)), trailing_edges=len(te_edges),
               smooth_contact_edges=cut_kind.count("smooth_wall_contact"),
               gamma_unknowns=int(station_dof.max() + 1), minimum_triangle_quality=float(quality.min()),
               normalized_vector_area_closure=float(np.linalg.norm(area_vectors.sum(axis=0)) / areas.sum()),
               wall_contact_intervals=metadata.get("wall_contact_intervals", []),
               wake_end_x=end_x, subdivision=args.subdivide),
    provenance=dict(mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(), mesh=str(args.mesh.resolve()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(record["audit"], indent=2), flush=True)
