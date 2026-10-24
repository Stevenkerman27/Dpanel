"""Import closed lifting CAD and mesh a conforming full-volume planar wake."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import cad_basis
import cad_components
import cad_contact
import cad_cut
import cad_te

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"audit"/"runtime"/"python"))
import gmsh
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--wing-cad", type=Path, required=True)
parser.add_argument("--body-cad", type=Path)
parser.add_argument("--trailing-edge", type=Path, required=True)
parser.add_argument("--size", type=float, default=.25)
parser.add_argument("--radius", type=float, default=5.)
parser.add_argument("--far-size", type=float, default=1.5)
parser.add_argument("--wake-strips", type=int, default=12)
parser.add_argument("--curvature", type=int, default=0)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists() or not any(args.out.iterdir()), "Preserve previous mesh attempts"
args.out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
sources = {name: (root/name).read_bytes() for name in [
    "build_cad_mesh.py", "cad_basis.py", "cad_components.py", "cad_contact.py",
    "cad_cut.py", "cad_te.py", "CAD_PROTOCOL.md"]}
for name, content in sources.items():
    (args.out/(name+".snapshot")).write_bytes(content)


def record_failure(exception_type, exception, traceback):
    failure = dict(status="failed_mesh", exception_type=exception_type.__name__,
                   message=str(exception), elapsed_seconds=time.perf_counter()-started)
    if hasattr(exception, "diagnostics"):
        (args.out/"failure_diagnostics.json").write_text(json.dumps(exception.diagnostics, indent=2), encoding="utf-8")
    (args.out/"failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    sys.__excepthook__(exception_type, exception, traceback)


sys.excepthook = record_failure
declaration = json.loads(args.trailing_edge.read_text(encoding="utf-8"))
length = float(declaration["chord"])
area = float(declaration["reference_area"])
origin = np.asarray(declaration.get("origin", [0., 0., 0.]), dtype=float)
native_te = np.asarray(declaration["points"], dtype=float)
assert length > 0. and area > 0. and np.isfinite([length, area]).all()
assert origin.shape == (3,) and np.isfinite(origin).all()
te = cad_contact.validate_te_polyline((native_te-origin)/length)
assert args.size > 0. and args.radius > 0. and args.far_size > 0.
assert np.isfinite([args.size, args.radius, args.far_size]).all()
assert args.wake_strips >= 4 and args.ncpu > 0 and args.curvature >= 0
assert np.max(abs(te)) < args.radius-args.size, "TE must be strictly inside the farfield"
cad_tolerance = 1e-7
stations = np.unique(np.r_[np.linspace(te[0, 1], te[-1, 1], args.wake_strips+1), te[:, 1]])
intervals = []
gmsh.initialize()
gmsh.logger.start()
gmsh.option.setNumber("General.NumThreads", args.ncpu)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", args.ncpu)
gmsh.option.setNumber("Mesh.MaxNumThreads3D", args.ncpu)
gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", args.curvature)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.size/8.)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.far_size)

# Rebuild once after discovering the span projection of smooth wall/wake contacts.
for pass_index in range(2):
    gmsh.clear()
    gmsh.model.add("lifting_cad_full_volume")
    occ = gmsh.model.occ
    roles = {}
    for role, path in [("wing", args.wing_cad), ("body", args.body_cad)]:
        if path is None:
            continue
        entities = occ.importShapes(str(path.resolve()), highestDimOnly=True)
        solids = [entity for entity in entities if entity[0] == 3]
        assert solids, f"{role} CAD must contain closed solids"
        occ.translate(solids, *(-origin))
        occ.dilate(solids, 0., 0., 0., 1./length, 1./length, 1./length)
        roles[role] = solids
    occ.synchronize()
    te_diagnostics = cad_te.validate_declared_te(gmsh, roles["wing"], te, cad_tolerance)
    (args.out/f"declared_te_pass{pass_index}.json").write_text(
        json.dumps(te_diagnostics, indent=2), encoding="utf-8")
    for solids in roles.values():
        for dim, tag in solids:
            bounds = np.asarray(occ.getBoundingBox(dim, tag))
            assert np.max(abs(bounds)) < args.radius-cad_tolerance, "Solid must fit strictly inside farfield"
            assert occ.getMass(dim, tag) > 0., "Imported solid must have positive volume"
    copies = cad_components.copy_source_solids(gmsh, roles)
    solid_list = [entity for solids in roles.values() for entity in solids]
    if len(solid_list) > 1:
        solid_list, _ = occ.fuse(solid_list[:1], solid_list[1:])
        solid_list = [entity for entity in solid_list if entity[0] == 3]
    outer = occ.addBox(-args.radius, -args.radius, -args.radius,
                       2.*args.radius, 2.*args.radius, 2.*args.radius)
    fluid, _ = occ.cut([(3, outer)], solid_list)
    fluid = [entity for entity in fluid if entity[0] == 3]
    assert len(fluid) == 1, "Require one connected external fluid volume"
    wake_inputs = []
    for left, right in zip(stations[:-1], stations[1:]):
        vertices = [occ.addPoint(float(x), float(y), 0.) for x, y in [
            (np.interp(left, te[:, 1], te[:, 0]), left), (args.radius, left),
            (args.radius, right), (np.interp(right, te[:, 1], te[:, 0]), right)]]
        edges = [occ.addLine(vertices[i], vertices[(i+1) % 4]) for i in range(4)]
        wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(edges)])))
    fragments, ancestry = occ.fragment(fluid, wake_inputs)
    fluid = [entity for entity in fragments if entity[0] == 3]
    occ.synchronize()
    fluid_ids = {tag for _, tag in fluid}
    wake_candidates = sorted({tag for row in ancestry[1:] for dim, tag in row if dim == 2})
    embedded = {tag for _, volume in fluid for dim, tag in gmsh.model.mesh.getEmbedded(3, volume)
                if dim == 2}
    wake_surfaces = [tag for tag in wake_candidates
                     if tag in embedded or fluid_ids.intersection(map(int, gmsh.model.getAdjacencies(2, tag)[0]))]
    (args.out/f"embedding_pass{pass_index}.json").write_text(json.dumps(dict(
        fluid=fluid, wake_candidates=wake_candidates, embedded=sorted(embedded),
        selected=wake_surfaces), indent=2), encoding="utf-8")
    assert wake_surfaces, "OCC did not embed any conforming wake surface in the fluid"
    component_tags, component_diagnostics = cad_components.classify_wall_surfaces(
        gmsh, fluid, copies, cad_tolerance, ambiguity_rule="body_precedence")
    wall_surfaces = sorted(component_tags)
    assert wall_surfaces, "CAD source-surface matching found no physical walls"
    (args.out/f"surface_embeddings_pass{pass_index}.json").write_text(json.dumps({
        str(tag): dict(role=component_tags.get(tag, "wake"),
                       boundary=gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False),
                       embedded=gmsh.model.mesh.getEmbedded(2, tag))
        for tag in sorted(set(wall_surfaces+wake_surfaces))}, indent=2), encoding="utf-8")
    new_intervals, contact_diagnostics = cad_contact.discover_contact_intervals(
        wall_surfaces, wake_surfaces, te, te_tolerance=cad_tolerance)
    (args.out/f"contacts_pass{pass_index}.json").write_text(
        json.dumps(contact_diagnostics, indent=2), encoding="utf-8")
    if pass_index == 0:
        intervals = new_intervals
        for endpoint in np.asarray(intervals).ravel():
            nearest = int(np.argmin(abs(stations-endpoint)))
            if abs(stations[nearest]-endpoint) < cad_tolerance:
                stations[nearest] = endpoint
            else:
                stations = np.sort(np.r_[stations, endpoint])
        continue
    assert np.asarray(new_intervals).shape == np.asarray(intervals).shape
    assert not intervals or np.max(abs(np.asarray(new_intervals)-intervals)) < cad_tolerance
    station_dof = cad_basis.contact_station_map(stations, intervals)
    removal = cad_components.remove_source_copies_before_mesh(gmsh, fluid, copies)
    (args.out/"components.json").write_text(json.dumps(component_diagnostics, indent=2), encoding="utf-8")
    (args.out/"reference_removal.json").write_text(json.dumps(removal, indent=2), encoding="utf-8")
    # Fragments of the wake lying inside solids have no fluid incidence.
    orphan_surfaces = [(2, tag) for tag in wake_candidates if tag not in wake_surfaces]
    if orphan_surfaces:
        occ.remove(orphan_surfaces, recursive=True)
        occ.synchronize()

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
gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder")
node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(coordinates).reshape(-1, 3)
node_index = {int(tag): index for index, tag in enumerate(node_tags)}
_, tet_nodes = gmsh.model.mesh.getElementsByType(11)
tets = np.asarray([[node_index[int(tag)] for tag in row] for row in tet_nodes.reshape(-1, 10)])
assert len(tets), "No Tet10 elements produced"
gmsh_reference = np.asarray(gmsh.model.mesh.getElementProperties(11)[4]).reshape(10, 3)
desired_reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                             [.5, 0., 0.], [.5, .5, 0.], [0., .5, 0.], [0., 0., .5],
                             [.5, 0., .5], [0., .5, .5]])
permutation = np.argmin(np.linalg.norm(desired_reference[:, None]-gmsh_reference[None], axis=2), axis=1)
assert np.max(abs(gmsh_reference[permutation]-desired_reference)) < 1e-12
tets = tets[:, permutation]
surface_triangles = {}
for dim, tag in gmsh.model.getEntities(2):
    _, nodes = gmsh.model.mesh.getElementsByType(9, tag)
    surface_triangles[tag] = np.asarray([[node_index[int(node)] for node in row]
                                        for row in nodes.reshape(-1, 6)], dtype=int).reshape(-1, 6)
wake_triangles = np.vstack([surface_triangles[tag] for tag in wake_surfaces])
gmsh_messages = gmsh.logger.get()
gmsh.logger.stop()
gmsh.finalize()
used = np.unique(tets)
compact = np.full(len(points), -1, dtype=int)
compact[used] = np.arange(len(used))
points, tets = points[used], compact[tets]
wake_triangles = compact[wake_triangles]
assert np.min(wake_triangles) >= 0
surface_triangles = {tag: compact[faces] for tag, faces in surface_triangles.items()}
corner_jacobian = np.stack([points[tets[:, i]]-points[tets[:, 0]] for i in [1, 2, 3]], axis=2)
negative = np.linalg.det(corner_jacobian) < 0.
tets[negative] = tets[negative][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]

# Only reconcile the declared planar cut, keeping every original coordinate.
original_coordinates = points.copy()
wake_nodes = np.unique(wake_triangles)
assert np.max(abs(points[wake_nodes, 2])) < cad_tolerance
points[wake_nodes, 2] = 0.
for station in stations:
    on_strip = wake_nodes[abs(points[wake_nodes, 1]-station) < cad_tolerance]
    points[on_strip, 1] = station
split = cad_cut.split_planar_cut(points, tets, wake_triangles, te[[0, -1], 1])
(args.out/"cut_topology.json").write_text(json.dumps(split["diagnostics"], indent=2), encoding="utf-8")
points, tets = split["points"], split["tets"]
local_faces = [[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
               [0, 2, 3, 6, 9, 7], [1, 2, 3, 5, 9, 8]]
face_nodes = np.concatenate([tets[:, face] for face in local_faces])
face_tet = np.tile(np.arange(len(tets)), 4)
_, inverse, counts = np.unique(np.sort(face_nodes[:, :3], axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts <= 2)
exterior = counts[inverse] == 1
faces, adjacent = face_nodes[exterior], face_tet[exterior]
wake_keys = {tuple(sorted(face[:3])) for face in np.vstack([split["wake_upper"], split["wake_lower"]])}
physical = np.array([tuple(sorted(face[:3])) not in wake_keys for face in faces])
boundary, boundary_tet = faces[physical], adjacent[physical]
source_surface = {}
for tag, triangles in surface_triangles.items():
    for triangle in triangles:
        if np.all(triangle >= 0):
            source_surface[tuple(sorted(triangle[:3]))] = tag
boundary_tags = []
for face in boundary:
    original_key = tuple(sorted(split["original_node"][face[:3]]))
    tag = source_surface[original_key]
    if tag in component_tags:
        boundary_tags.append(component_tags[tag])
    else:
        xyz = points[face]
        assert any(np.max(abs(xyz[:, axis]-side*args.radius)) < cad_tolerance
                   for axis in range(3) for side in [-1., 1.]), f"Unclassified physical CAD wall {tag}"
        boundary_tags.append(100)
boundary_tags = np.asarray(boundary_tags, dtype=int)
assert np.any(boundary_tags != 100)
upper = split["cut_upper_nodes"]
lower = split["cut_lower_nodes"]
te_distance = np.linalg.norm(points[upper]-np.column_stack([
    np.interp(points[upper, 1], te[:, 1], te[:, 0]), points[upper, 1], np.zeros(len(upper))]), axis=1)
te_candidates = upper[te_distance < cad_tolerance]
te_upper, te_lower, te_station_indices = [], [], []
for index, station in enumerate(stations[1:-1], start=1):
    candidates = te_candidates[abs(points[te_candidates, 1]-station) < cad_tolerance]
    assert len(candidates) <= 1, "Ambiguous TE station: multiple coincident mesh nodes"
    if len(candidates):
        node = int(candidates[0])
        te_upper.append(node)
        te_lower.append(int(lower[np.flatnonzero(upper == node)[0]]))
        te_station_indices.append(index)
assert te_upper, "Declared trailing edge has no exposed station knots"
barycentric = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
_, derivative = p2.p2_shape(barycentric)
minimum_jacobian = np.inf
for first in range(0, len(tets), 256):
    jacobian = np.einsum("eia,qib->eqab", points[tets[first:first+256]], derivative)
    minimum_jacobian = min(minimum_jacobian, float(np.linalg.det(jacobian).min()))
assert minimum_jacobian > 0., "Dense-sampled curved Tet10 Jacobian is nonpositive"
np.savez_compressed(
    args.out/"mesh.npz", points=points*length+origin, tets=tets,
    boundary_triangles=boundary, boundary_tet=boundary_tet, boundary_tags=boundary_tags,
    wake_upper=split["wake_upper"], wake_lower=split["wake_lower"],
    wake_upper_tet=split["wake_upper_tet"], wake_lower_tet=split["wake_lower_tet"],
    cut_upper_nodes=upper, cut_lower_nodes=lower,
    te_upper_nodes=np.asarray(te_upper, dtype=int), te_lower_nodes=np.asarray(te_lower, dtype=int),
    te_station_indices=np.asarray(te_station_indices, dtype=int),
    gamma_stations=stations*length+origin[1], gamma_station_dof=station_dof,
    trailing_edge_points=native_te,
    original_node=split["original_node"],
    cad_original_coordinates=original_coordinates*length+origin)
metadata = dict(
    geometry_family="imported_cad_planar_te", chord=length, reference_area=area,
    normalization_origin=origin.tolist(), span=float(native_te[-1, 1]-native_te[0, 1]),
    wall_contact_intervals=(np.asarray(intervals).reshape(-1, 2)*length+origin[1]).tolist(),
    cad_coordinate_tolerance=cad_tolerance, points=len(points), tets=len(tets),
    gamma_unknowns=int(station_dof.max()+1), wake_strips=args.wake_strips,
    actual_wake_strips=len(stations)-1, gmsh_threads=args.ncpu,
    gmsh_warnings=[message for message in gmsh_messages if message.startswith(("Warning", "Error"))],
    min_dense_jacobian=minimum_jacobian*length**3,
    dense_jacobian_samples_per_element=len(barycentric),
    size=args.size, far_radius=args.radius, far_size=args.far_size,
    coordinate_reconciliation=dict(
        operation="Set wake z=0 and CAD strip knot y within declared normalized CAD tolerance",
        maximum_normalized_shift=float(np.max(abs(points[:len(original_coordinates)]-original_coordinates))),
        original_coordinates_saved=True),
    component_tagging="CAD source face ownership, trimmed interior samples; body precedence on coincident surfaces",
    geometry="Full imported closed solids; no symmetry reflection",
    wake="Declared planar piecewise-linear TE, conforming span strips, local node-star bank cut",
    jump_representation="Exactly linear in y in each conforming CAD wake strip; zero at tips",
    source_sha256={name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    input_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [args.wing_cad, args.body_cad, args.trailing_edge] if path is not None},
    mesh_sha256=hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest(),
    elapsed_seconds=time.perf_counter()-started)
(args.out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
