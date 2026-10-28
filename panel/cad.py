"""Import closed CAD solids and export an oriented surface-only panel mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

root = Path(__file__).resolve().parent
project = root.parent
lifting_root = project / "lifting3d"
sys.path.insert(0, str(lifting_root))
sys.path.insert(0, str(project / "audit" / "runtime" / "python"))
import cad_basis
import cad_components
import cad_contact
import cad_te
import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("--wing-cad", type=Path, required=True)
parser.add_argument("--body-cad", type=Path)
parser.add_argument("--trailing-edge", type=Path, required=True)
parser.add_argument("--size", type=float, default=.3)
parser.add_argument("--far-size", type=float, default=1.5)
parser.add_argument("--radius", type=float, default=5.)
parser.add_argument("--wake-strips", type=int, default=8)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists() or not any(args.out.iterdir()), "Preserve previous mesh attempts"
args.out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
stage = "input_validation"
gmsh_active = False


def record_failure(exception_type, exception, exception_traceback):
    global gmsh_active
    messages = []
    if gmsh_active:
        try:
            messages = gmsh.logger.get()
            gmsh.logger.stop()
        except Exception:
            pass
    failure = {
        "status": "failed",
        "stage": stage,
        "exception_type": exception_type.__name__,
        "message": str(exception),
        "traceback": "".join(traceback.format_exception(
            exception_type, exception, exception_traceback)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if hasattr(exception, "diagnostics"):
        failure["diagnostics"] = exception.diagnostics
    (args.out / "failure.json").write_text(
        json.dumps(failure, indent=2), encoding="utf-8")
    if messages:
        (args.out / "gmsh.log").write_text(
            "\n".join(messages), encoding="utf-8")
    if gmsh_active:
        try:
            gmsh.finalize()
        except Exception:
            pass
        gmsh_active = False
    sys.__excepthook__(exception_type, exception, exception_traceback)


sys.excepthook = record_failure
cad_tolerance = 1e-7
source_names = [
    "build_cad_mesh.py", "cad_basis.py", "cad_components.py",
    "cad_contact.py", "cad_te.py",
]
source_bytes = {name: (lifting_root / name).read_bytes() for name in source_names}
(args.out / "source").mkdir()
for name, content in source_bytes.items():
    (args.out / "source" / name).write_bytes(content)
script_bytes = Path(__file__).read_bytes()
(args.out / "source" / "cad.py").write_bytes(script_bytes)
source_hashes = {
    name: hashlib.sha256(content).hexdigest()
    for name, content in source_bytes.items()
}
source_hashes["cad.py"] = hashlib.sha256(script_bytes).hexdigest()
declaration_bytes = args.trailing_edge.read_bytes()
declaration = json.loads(declaration_bytes.decode("utf-8"))
chord = float(declaration["chord"])
reference_area = float(declaration["reference_area"])
origin = np.asarray(declaration.get("origin", [0., 0., 0.]), dtype=float)
native_te = np.asarray(declaration["points"], dtype=float)
assert chord > 0. and reference_area > 0.
assert origin.shape == (3,) and np.isfinite(origin).all()
assert native_te.ndim == 2 and native_te.shape[1] == 3
te = cad_contact.validate_te_polyline((native_te - origin) / chord)
assert args.size > 0. and args.far_size > 0. and args.radius > 0.
assert np.isfinite([args.size, args.far_size, args.radius]).all()
assert args.wake_strips >= 4 and args.ncpu > 0
assert np.max(abs(te)) < args.radius - args.size, "TE must be strictly inside farfield"
input_paths = [args.wing_cad, args.trailing_edge]
if args.body_cad is not None:
    input_paths.append(args.body_cad)
input_hashes = {
    str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in input_paths
}
(args.out / "run.json").write_text(json.dumps({
    "command": [sys.executable, *sys.argv],
    "inputs": input_hashes,
    "source_sha256": source_hashes,
    "parameters": {
        "size": args.size, "far_size": args.far_size, "radius": args.radius,
        "wake_strips": args.wake_strips, "ncpu": args.ncpu,
    },
}, indent=2), encoding="utf-8")

stations = np.unique(np.r_[
    np.linspace(te[0, 1], te[-1, 1], args.wake_strips + 1), te[:, 1]
])
intervals = []
stage = "gmsh_initialization"
gmsh.initialize()
gmsh_active = True
gmsh.logger.start()
gmsh.option.setNumber("General.NumThreads", args.ncpu)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", args.ncpu)
gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.size / 8.)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.far_size)

for pass_index in range(2):
    stage = f"occ_import_boolean_fragment_pass_{pass_index}"
    gmsh.clear()
    gmsh.model.add("panel_cad_surface")
    occ = gmsh.model.occ
    roles = {}
    for role, path in [("wing", args.wing_cad), ("body", args.body_cad)]:
        if path is None:
            continue
        imported = occ.importShapes(str(path.resolve()), highestDimOnly=True)
        solids = [entity for entity in imported if entity[0] == 3]
        assert solids, f"{role} CAD must contain closed solids"
        occ.translate(solids, *(-origin))
        occ.dilate(solids, 0., 0., 0., 1. / chord, 1. / chord, 1. / chord)
        roles[role] = solids
    occ.synchronize()
    stage = f"declared_te_validation_pass_{pass_index}"
    te_diagnostics = cad_te.validate_declared_te(
        gmsh, roles["wing"], te, cad_tolerance)
    (args.out / f"declared_te_pass{pass_index}.json").write_text(
        json.dumps(te_diagnostics, indent=2), encoding="utf-8")
    stage = f"cad_bounds_pass_{pass_index}"
    for solids in roles.values():
        for dimension, tag in solids:
            bounds = np.asarray(occ.getBoundingBox(dimension, tag))
            assert np.max(abs(bounds)) < args.radius - cad_tolerance, (
                f"CAD solid {tag} bounding box {bounds.tolist()} is not strictly "
                f"inside radius {args.radius}"
            )
            assert occ.getMass(dimension, tag) > 0.
    source_copies = cad_components.copy_source_solids(gmsh, roles)
    solid_list = [entity for solids in roles.values() for entity in solids]
    if len(solid_list) > 1:
        solid_list, _ = occ.fuse(solid_list[:1], solid_list[1:])
        solid_list = [entity for entity in solid_list if entity[0] == 3]
    outer = occ.addBox(
        -args.radius, -args.radius, -args.radius,
        2. * args.radius, 2. * args.radius, 2. * args.radius)
    fluid, _ = occ.cut([(3, outer)], solid_list)
    fluid = [entity for entity in fluid if entity[0] == 3]
    assert len(fluid) == 1, "Require one connected external fluid volume"
    wake_inputs = []
    for left, right in zip(stations[:-1], stations[1:]):
        vertices = [
            occ.addPoint(float(x), float(y), 0.)
            for x, y in [
                (np.interp(left, te[:, 1], te[:, 0]), left),
                (args.radius, left), (args.radius, right),
                (np.interp(right, te[:, 1], te[:, 0]), right),
            ]
        ]
        edges = [occ.addLine(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
        wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(edges)])))
    fragments, ancestry = occ.fragment(fluid, wake_inputs)
    fluid = [entity for entity in fragments if entity[0] == 3]
    occ.synchronize()
    fluid_ids = {tag for _, tag in fluid}
    wake_candidates = sorted({
        tag for row in ancestry[1:] for dimension, tag in row if dimension == 2
    })
    embedded = {
        tag for _, volume in fluid
        for dimension, tag in gmsh.model.mesh.getEmbedded(3, volume)
        if dimension == 2
    }
    wake_surfaces = []
    incidence = {}
    for tag in wake_candidates:
        adjacent_volumes = sorted(map(
            int, gmsh.model.getAdjacencies(2, tag)[0]))
        touches_fluid = bool(fluid_ids.intersection(adjacent_volumes))
        is_embedded = tag in embedded
        incidence[tag] = {
            "adjacent_volume_tags": adjacent_volumes,
            "fluid_boundary_incidence": touches_fluid,
            "embedded_in_fluid": is_embedded,
        }
        if touches_fluid or is_embedded:
            wake_surfaces.append(tag)
    (args.out / f"embedding_pass{pass_index}.json").write_text(
        json.dumps({
            "fluid": fluid, "wake_candidates": wake_candidates,
            "embedded": sorted(embedded), "selected": wake_surfaces,
            "incidence": incidence,
        }, indent=2), encoding="utf-8")
    assert wake_surfaces, "OCC did not retain a wake surface incident to fluid"
    stage = f"wall_classification_pass_{pass_index}"
    component_tags, component_diagnostics = cad_components.classify_wall_surfaces(
        gmsh, fluid, source_copies, cad_tolerance,
        ambiguity_rule="body_precedence")
    wall_surfaces = sorted(component_tags)
    assert wall_surfaces, "CAD source-surface matching found no physical walls"
    stage = f"contact_discovery_pass_{pass_index}"
    new_intervals, contact_diagnostics = cad_contact.discover_contact_intervals(
        wall_surfaces, wake_surfaces, te, te_tolerance=cad_tolerance)
    (args.out / f"contacts_pass{pass_index}.json").write_text(
        json.dumps(contact_diagnostics, indent=2), encoding="utf-8")
    if pass_index == 0:
        stage = "contact_station_refinement"
        intervals = new_intervals
        for endpoint in np.asarray(intervals).ravel():
            nearest = int(np.argmin(abs(stations - endpoint)))
            if abs(stations[nearest] - endpoint) < cad_tolerance:
                stations[nearest] = endpoint
            else:
                stations = np.sort(np.r_[stations, endpoint])
        continue
    stage = "contact_station_validation"
    assert np.asarray(new_intervals).shape == np.asarray(intervals).shape
    assert not intervals or np.max(abs(np.asarray(new_intervals) - intervals)) < cad_tolerance
    station_dof = cad_basis.contact_station_map(stations, intervals)
    removal = cad_components.remove_source_copies_before_mesh(
        gmsh, fluid, source_copies)
    (args.out / "components.json").write_text(
        json.dumps(component_diagnostics, indent=2), encoding="utf-8")
    (args.out / "reference_removal.json").write_text(
        json.dumps(removal, indent=2), encoding="utf-8")
    orphan_surfaces = [
        (2, tag) for tag in wake_candidates if tag not in wake_surfaces
    ]
    if orphan_surfaces:
        occ.remove(orphan_surfaces, recursive=True)
        occ.synchronize()

stage = "surface_mesh_generation"
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", wall_surfaces)
gmsh.model.mesh.field.setNumber(distance, "Sampling", 150)
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
gmsh.model.mesh.field.setNumber(threshold, "SizeMin", args.size)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", args.far_size)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", .3)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.)
gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
gmsh.model.mesh.generate(2)
node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(coordinates).reshape(-1, 3)
node_index = {int(tag): index for index, tag in enumerate(node_tags)}
wall_faces = []
wall_tags = []
for surface_tag in wall_surfaces:
    _, element_nodes = gmsh.model.mesh.getElementsByType(2, surface_tag)
    triangles = np.asarray(element_nodes, dtype=np.int64).reshape(-1, 3)
    wall_faces.extend([
        [node_index[int(node)] for node in row] for row in triangles
    ])
    wall_tags.extend([component_tags[surface_tag]] * len(triangles))
wall_faces = np.asarray(wall_faces, dtype=int).reshape(-1, 3)
wall_tags = np.asarray(wall_tags, dtype=int)
wake_faces = []
for surface_tag in wake_surfaces:
    _, element_nodes = gmsh.model.mesh.getElementsByType(2, surface_tag)
    triangles = np.asarray(element_nodes, dtype=np.int64).reshape(-1, 3)
    wake_faces.extend([
        [node_index[int(node)] for node in row] for row in triangles
    ])
wake_faces = np.asarray(wake_faces, dtype=int).reshape(-1, 3)
gmsh_messages = gmsh.logger.get()
gmsh.logger.stop()
gmsh.finalize()
gmsh_active = False

stage = "surface_orientation_and_mesh_write"
used_nodes = np.unique(np.concatenate((wall_faces.ravel(), wake_faces.ravel())))
compact = np.full(len(points), -1, dtype=int)
compact[used_nodes] = np.arange(len(used_nodes))
points = points[used_nodes] * chord + origin
wall_faces = compact[wall_faces]
wake_faces = compact[wake_faces]
assert len(wall_faces) and len(wake_faces)

edge_faces = {}
for face_index, face in enumerate(wall_faces):
    for edge_index in range(3):
        start_node = int(face[edge_index])
        end_node = int(face[(edge_index + 1) % 3])
        key = (min(start_node, end_node), max(start_node, end_node))
        edge_faces.setdefault(key, []).append(
            (face_index, start_node < end_node))
assert all(len(incident) == 2 for incident in edge_faces.values()), \
    "Selected CAD wall triangles are not a closed two-manifold"
neighbors = [[] for _ in wall_faces]
for incident in edge_faces.values():
    (first, first_direction), (second, second_direction) = incident
    same_direction = first_direction == second_direction
    neighbors[first].append((second, same_direction))
    neighbors[second].append((first, same_direction))
orientation = np.zeros(len(wall_faces), dtype=np.int8)
component_faces = []
for seed in range(len(wall_faces)):
    if orientation[seed] != 0:
        continue
    orientation[seed] = 1
    queue = [seed]
    faces_in_component = []
    while queue:
        current = queue.pop()
        faces_in_component.append(current)
        for neighbor, same_direction in neighbors[current]:
            required = -orientation[current] if same_direction else orientation[current]
            if orientation[neighbor] == 0:
                orientation[neighbor] = required
                queue.append(neighbor)
            else:
                assert orientation[neighbor] == required, \
                    "CAD wall component has inconsistent edge orientation"
    component_faces.append(np.asarray(faces_in_component, dtype=int))
reverse = orientation < 0
wall_faces[reverse] = wall_faces[reverse][:, [0, 2, 1]]
component_volumes = []
for component in component_faces:
    triangles = points[wall_faces[component]]
    signed_volume = np.sum(
        np.einsum(
            "ij,ij->i", triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2])
        )
    ) / 6.
    assert abs(signed_volume) > 1e-14 * chord**3, \
        "Closed CAD wall component has zero signed volume"
    if signed_volume < 0.:
        wall_faces[component] = wall_faces[component][:, [0, 2, 1]]
        signed_volume = -signed_volume
    component_volumes.append(float(signed_volume))
wake_triangles = points[wake_faces]
wake_area_vectors = np.cross(
    wake_triangles[:, 1] - wake_triangles[:, 0],
    wake_triangles[:, 2] - wake_triangles[:, 0])
reverse_wake = wake_area_vectors[:, 2] < 0.
wake_faces[reverse_wake] = wake_faces[reverse_wake][:, [0, 2, 1]]
wake_area_vectors[reverse_wake] *= -1.
assert np.all(wake_area_vectors[:, 2] > 0.)
assert np.max(abs(wake_area_vectors[:, :2])) < 1e-10 * chord**2

gamma_stations = stations * chord + origin[1]
trailing_edge_points = native_te.copy()
mesh_path = args.out / "mesh.npz"
np.savez_compressed(
    mesh_path, points=points, wall_faces=wall_faces, wall_tags=wall_tags,
    wake_faces=wake_faces, original_node=np.arange(len(points), dtype=int),
    gamma_stations=gamma_stations, gamma_station_dof=station_dof,
    trailing_edge_points=trailing_edge_points)
mesh_metadata = {
    "schema": "dpanel.surface_mesh.v1",
    "surface_only": True,
    "chord": chord,
    "reference_area": reference_area,
    "normalization_origin": origin.tolist(),
    "wall_contact_intervals": (
        np.asarray(intervals).reshape(-1, 2) * chord + origin[1]
    ).tolist(),
    "input_sha256": input_hashes,
    "source_sha256": source_hashes,
    "points": int(len(points)),
    "wall_panels": int(len(wall_faces)),
    "wake_panels": int(len(wake_faces)),
    "wall_tag_counts": {
        str(tag): int(np.sum(wall_tags == tag)) for tag in np.unique(wall_tags)
    },
    "wall_component_count": int(len(component_faces)),
    "wall_component_positive_volumes": component_volumes,
    "wake_orientation": "all triangle area vectors have positive z",
    "wake_fluid_incidence_surface_count": int(len(wake_surfaces)),
    "gmsh_threads": args.ncpu,
    "gmsh_warnings": [
        message for message in gmsh_messages
        if message.startswith(("Warning", "Error"))
    ],
    "elapsed_seconds": time.perf_counter() - started,
}
metadata_path = args.out / "mesh.json"
metadata_path.write_text(json.dumps(mesh_metadata, indent=2), encoding="utf-8")
(args.out / "gmsh.log").write_text("\n".join(gmsh_messages), encoding="utf-8")
mesh_metadata["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
metadata_path.write_text(json.dumps(mesh_metadata, indent=2), encoding="utf-8")
print(json.dumps(mesh_metadata, indent=2), flush=True)

stage = "surface_export"
surface_path = args.out / "surface.json"
surface_command = [
    sys.executable, str(root / "build_surface.py"),
    "--mesh", str(mesh_path), "--out", str(surface_path),
]
surface_result = subprocess.run(
    surface_command, capture_output=True, text=True, check=False)
(args.out / "build_surface.stdout.log").write_text(
    surface_result.stdout, encoding="utf-8")
(args.out / "build_surface.stderr.log").write_text(
    surface_result.stderr, encoding="utf-8")
(args.out / "build_surface.command.json").write_text(
    json.dumps({
        "command": surface_command,
        "returncode": surface_result.returncode,
        "surface_json_created": surface_path.exists(),
    }, indent=2), encoding="utf-8")
if surface_result.returncode:
    raise subprocess.CalledProcessError(
        surface_result.returncode, surface_command,
        surface_result.stdout, surface_result.stderr)
print(surface_result.stdout, flush=True)
