"""Reproduce one imported wing/body CAD pass and audit wall-wake curve topology."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "audit" / "runtime" / "python"))
sys.path.insert(0, str(root / "volume"))
import gmsh
import cad_components

case = root / "results" / "cad_inputs_v1" / "naca4412_wing_body"
previous = root / "results" / "cad_integration_v1" / "cambered_body_mesh"
out = root / "results" / "cad_integration_v1" / "body_wake_topology_audit_v1"
assert not out.exists() or not any(out.iterdir()), f"Preserve previous audit output: {out}"
out.mkdir(parents=True, exist_ok=True)

builder_snapshot = previous / "build_cad_mesh.py.snapshot"
builder_hash = hashlib.sha256(builder_snapshot.read_bytes()).hexdigest()
current_builder_hash = hashlib.sha256((root / "build_cad_mesh.py").read_bytes()).hexdigest()

wing_path = case / "wing.step"
body_path = case / "body.step"
te_path = case / "trailing_edge.json"
input_hashes = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [wing_path, body_path, te_path]
}
te_data = json.loads(te_path.read_text(encoding="utf-8"))
te = np.asarray(te_data["points"], dtype=float)
origin = np.asarray(te_data.get("origin", [0., 0., 0.]), dtype=float)
chord = float(te_data["chord"])
te = (te - origin) / chord
stations = np.unique(np.r_[np.linspace(te[0, 1], te[-1, 1], 9), te[:, 1]])
radius = 5.0
geometric_tolerance = 1e-7

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
gmsh.model.add("body_wake_topology_audit")
occ = gmsh.model.occ
roles = {}
for role, path in [("wing", wing_path), ("body", body_path)]:
    solids = [entity for entity in occ.importShapes(
        str(path.resolve()), highestDimOnly=True) if entity[0] == 3]
    occ.translate(solids, *(-origin))
    occ.dilate(solids, 0., 0., 0., 1. / chord, 1. / chord, 1. / chord)
    roles[role] = solids
occ.synchronize()
source_copies = cad_components.copy_source_solids(gmsh, roles)
solid_list = [entity for solids in roles.values() for entity in solids]
solid_list, _ = occ.fuse(solid_list[:1], solid_list[1:])
solid_list = [entity for entity in solid_list if entity[0] == 3]
outer = occ.addBox(-radius, -radius, -radius, 2. * radius, 2. * radius, 2. * radius)
fluid, _ = occ.cut([(3, outer)], solid_list)
fluid = [entity for entity in fluid if entity[0] == 3]
assert len(fluid) == 1

wake_inputs = []
for left, right in zip(stations[:-1], stations[1:]):
    vertices = [occ.addPoint(float(x), float(y), 0.) for x, y in [
        (np.interp(left, te[:, 1], te[:, 0]), left),
        (radius, left), (radius, right),
        (np.interp(right, te[:, 1], te[:, 0]), right)]]
    edges = [occ.addLine(vertices[index], vertices[(index + 1) % 4])
             for index in range(4)]
    wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(edges)])))
fragments, ancestry = occ.fragment(fluid, wake_inputs)
fluid = [entity for entity in fragments if entity[0] == 3]
occ.synchronize()

fluid_ids = {tag for _, tag in fluid}
wake_candidates = sorted({
    tag for row in ancestry[1:] for dim, tag in row if dim == 2
})
volume_embedded = sorted({
    tag for _, volume in fluid
    for dim, tag in gmsh.model.mesh.getEmbedded(3, volume) if dim == 2
})
wake_surfaces = [
    tag for tag in wake_candidates
    if tag in volume_embedded
    or fluid_ids.intersection(map(int, gmsh.model.getAdjacencies(2, tag)[0]))
]
wall_tags, wall_diagnostics = cad_components.classify_wall_surfaces(
    gmsh, fluid, source_copies, geometric_tolerance,
    ambiguity_rule="body_precedence")
body_walls = sorted(tag for tag, physical in wall_tags.items() if physical == 20)
body_source_faces = sorted({
    int(tag) for dim, tag in gmsh.model.getBoundary(
        source_copies["body"], combined=False, oriented=False) if dim == 2
})

wall_embedded = []
for surface_tag in body_walls:
    embedded = gmsh.model.mesh.getEmbedded(2, int(surface_tag))
    wall_embedded.append({
        "body_wall_surface_tag": int(surface_tag),
        "embedded_entities": [[int(dim), int(tag)] for dim, tag in embedded],
        "embedded_curve_tags": sorted(int(tag) for dim, tag in embedded if dim == 1),
    })

wall_curve_surfaces = {}
wake_curve_surfaces = {}
for surface_tag in body_walls:
    for dim, curve_tag in gmsh.model.getBoundary(
            [(2, int(surface_tag))], combined=False, oriented=False):
        if dim == 1:
            wall_curve_surfaces.setdefault(int(curve_tag), []).append(int(surface_tag))
for surface_tag in wake_surfaces:
    for dim, curve_tag in gmsh.model.getBoundary(
            [(2, int(surface_tag))], combined=False, oriented=False):
        if dim == 1:
            wake_curve_surfaces.setdefault(int(curve_tag), []).append(int(surface_tag))
body_shared_curves = sorted(set(wall_curve_surfaces) & set(wake_curve_surfaces))


def body_distance(point):
    distances = []
    closest_tags = []
    for source_face in body_source_faces:
        closest, uv = gmsh.model.getClosestPoint(
            2, source_face, np.asarray(point, dtype=float).tolist())
        if len(uv) == 2 and gmsh.model.isInside(
                2, source_face, uv, parametric=True):
            distances.append(float(np.linalg.norm(
                np.asarray(closest, dtype=float) - point)))
            closest_tags.append(int(source_face))
    if not distances:
        return float("inf"), None
    index = int(np.argmin(distances))
    return distances[index], closest_tags[index]


def curve_points(curve_tag, sample_count=17):
    lower, upper = gmsh.model.getParametrizationBounds(1, int(curve_tag))
    parameters = np.linspace(float(lower[0]), float(upper[0]), sample_count)
    return np.asarray([
        gmsh.model.getValue(1, int(curve_tag), [float(parameter)])
        for parameter in parameters
    ], dtype=float)


curve_records = []
all_wake_boundary_curves = sorted(wake_curve_surfaces)
for curve_tag in all_wake_boundary_curves:
    points = curve_points(curve_tag)
    distances_and_faces = [body_distance(point) for point in points]
    distance_values = [row[0] for row in distances_and_faces]
    endpoints = []
    for point in [points[0], points[-1]]:
        distance, face_tag = body_distance(point)
        endpoints.append({
            "xyz": point.tolist(),
            "closest_body_source_face": face_tag,
            "distance": distance,
        })
    curve_records.append({
        "curve_tag": int(curve_tag),
        "wake_surface_tags": sorted(set(wake_curve_surfaces[curve_tag])),
        "body_wall_surface_tags": sorted(set(
            wall_curve_surfaces.get(curve_tag, []))),
        "is_shared_body_wake_curve": curve_tag in body_shared_curves,
        "sample_xyz": points.tolist(),
        "sampled_body_distance_min": float(np.min(distance_values)),
        "sampled_body_distance_max": float(np.max(distance_values)),
        "closest_body_source_face_by_sample": [
            row[1] for row in distances_and_faces],
        "endpoints": endpoints,
    })

wall_curve_records = []
for curve_tag in sorted(wall_curve_surfaces):
    points = curve_points(curve_tag, sample_count=3)
    wall_curve_records.append({
        "curve_tag": int(curve_tag),
        "body_wall_surface_tags": sorted(set(wall_curve_surfaces[curve_tag])),
        "wake_surface_tags": sorted(set(wake_curve_surfaces.get(curve_tag, []))),
        "embedded_in_body_wall": any(
            curve_tag in row["embedded_curve_tags"] for row in wall_embedded),
        "sample_xyz": points.tolist(),
        "sampled_body_distance": [body_distance(point)[0] for point in points],
    })

body_aft_intersection_y = 0.35 * np.sqrt(1. - (1.5 / 2.) ** 2)
report = {
    "status": "completed",
    "scope": "Exact CAD pass reproduction through OCC fragment and wall classification; no 3D meshing",
    "threads": 8,
    "builder_snapshot_sha256": builder_hash,
    "current_builder_sha256": current_builder_hash,
    "builder_hashes_match": builder_hash == current_builder_hash,
    "builder_source_difference_note": (
        "The current builder differs only by Gmsh logger capture and "
        "surface_embeddings_pass*.json diagnostics; geometry operations through "
        "wall classification match the mesh snapshot."
    ),
    "input_sha256": input_hashes,
    "fluid_volumes": fluid,
    "wake_candidate_count": len(wake_candidates),
    "volume_embedded_wake_count": len(volume_embedded),
    "selected_wake_count": len(wake_surfaces),
    "body_wall_surface_tags": body_walls,
    "body_source_face_tags": body_source_faces,
    "body_wall_embedding_query": wall_embedded,
    "body_wall_boundary_curve_count": len(wall_curve_surfaces),
    "wake_boundary_curve_count": len(wake_curve_surfaces),
    "shared_body_wake_curve_tags": body_shared_curves,
    "shared_body_wake_curve_count": len(body_shared_curves),
    "wake_boundary_curve_sample_count": 17,
    "wake_boundary_curve_to_body": curve_records,
    "body_wall_curves": wall_curve_records,
    "geometry_check": {
        "body_model": "ellipsoid centered x=-0.5, axes (2.0, 0.35, 0.35)",
        "wake_plane": "z=0, x from wing TE x=1 to farfield x=5",
        "body_aft_cap_x_at_y0": -0.5 + 2.0,
        "body_wake_intersection_y_extent_at_x_te": [
            -float(body_aft_intersection_y), float(body_aft_intersection_y)],
        "interpretation": (
            "The wake plane intersects the body aft cap for |y| below this extent; "
            "a CAD wall-wake curve should be discoverable if the OCC topology is imprinted."
        ),
    },
    "wall_classification": wall_diagnostics,
}
(out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
gmsh.finalize()
print(json.dumps({
    "status": report["status"],
    "body_wall_surface_tags": body_walls,
    "wall_embedded_curves": [
        row["embedded_curve_tags"] for row in wall_embedded],
    "shared_body_wake_curve_count": len(body_shared_curves),
    "wake_boundary_curves_with_body_distance_below_1e-7": sum(
        row["sampled_body_distance_min"] < geometric_tolerance
        for row in curve_records),
    "geometry_expected_y_extent": report["geometry_check"][
        "body_wake_intersection_y_extent_at_x_te"],
    "output": str(out / "audit.json"),
}, indent=2))
