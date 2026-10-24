"""Import closed CAD solids for a nonlifting P2 volume solve, with explicit native-coordinate mapping."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"audit"/"runtime"/"python"))
import gmsh
import p2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cad", type=Path, required=True)
parser.add_argument("--size", type=float, default=.15, help="Wall mesh size / native CAD bounding-box length")
parser.add_argument("--radius", type=float, default=5., help="Fluid-box half width / native CAD bounding-box length")
parser.add_argument("--far-size", type=float, default=1.)
parser.add_argument("--curvature", type=int, default=32)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve existing CAD mesh outputs"
args.out.mkdir(parents=True)
started = time.perf_counter()
source = Path(__file__).read_bytes()
result = dict(status="failed", scope="Closed CAD solids, single-valued nonlifting potential; no Kutta or wake",
              source_sha256=hashlib.sha256(source).hexdigest(), ncpu_requested=args.ncpu,
              cad=str(args.cad.resolve()))
try:
    assert args.cad.is_file(), "CAD input does not exist"
    assert np.isfinite([args.size, args.radius, args.far_size]).all()
    assert args.size > 0. and args.radius > 1. and args.far_size > 0. and args.curvature >= 0 and args.ncpu > 0
    result["cad_sha256"] = hashlib.sha256(args.cad.read_bytes()).hexdigest()
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", args.ncpu)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", args.ncpu)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", args.ncpu)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", args.curvature)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.size/12.)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.far_size)
    gmsh.model.add("imported_closed_cad")
    occ = gmsh.model.occ
    imported = occ.importShapes(str(args.cad.resolve()), highestDimOnly=True)
    solids = [entity for entity in imported if entity[0] == 3]
    assert solids, "CAD input contains no closed solid volumes; open surfaces are unsupported"
    result["imported_solids"] = len(solids)
    occ.synchronize()
    boxes = np.array([gmsh.model.getBoundingBox(*entity) for entity in solids])
    lower, upper = boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)
    native_origin, native_length = .5*(lower+upper), float(np.max(upper-lower))
    assert np.isfinite(native_length) and native_length > 0., "CAD bounding box must have positive finite extent"
    result.update(native_origin=native_origin.tolist(), native_length=native_length,
                  native_bbox=[lower.tolist(), upper.tolist()],
                  native_bounds_method="OCC conservative bounding boxes; B-spline control bounds may exceed geometric extents",
                  native_origin_definition="Center of the recorded conservative bounding box; not necessarily the geometric or mass center",
                  imported_native_volume=float(sum(occ.getMass(*entity) for entity in solids)))
    occ.translate(solids, *(-native_origin))
    occ.dilate(solids, 0., 0., 0., *np.full(3, 1./native_length))
    if len(solids) > 1:
        fused, _ = occ.fuse(solids[:1], solids[1:])
        solids = [entity for entity in fused if entity[0] == 3]
    radius = args.radius
    outer = occ.addBox(-radius, -radius, -radius, 2.*radius, 2.*radius, 2.*radius)
    fluid, _ = occ.cut([(3, outer)], solids)
    fluid = [entity for entity in fluid if entity[0] == 3]
    assert len(fluid) == 1, "CAD subtraction must produce one connected fluid volume; enclosed fluid cavities are unsupported"
    occ.synchronize()
    body_surfaces, far_surfaces = [], []
    for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
        assert dim == 2
        box = np.array(gmsh.model.getBoundingBox(dim, tag))
        is_far = any(abs(box[axis]-coordinate) < 1e-6 and abs(box[axis+3]-coordinate) < 1e-6
                     for axis in range(3) for coordinate in [-radius, radius])
        (far_surfaces if is_far else body_surfaces).append(tag)
    assert body_surfaces and len(far_surfaces) == 6, "CAD body/farfield boundary classification failed"
    gmsh.model.addPhysicalGroup(2, body_surfaces, 10)
    gmsh.model.setPhysicalName(2, 10, "closed_body")
    gmsh.model.addPhysicalGroup(2, far_surfaces, 100)
    gmsh.model.setPhysicalName(2, 100, "farfield")
    gmsh.model.addPhysicalGroup(3, [fluid[0][1]], 1)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", body_surfaces)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 150)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", args.size)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", args.far_size)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", .3)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.model.mesh.generate(3)
    result["body_surface_triangle_counts"] = {
        str(tag): len(gmsh.model.mesh.getElementsByType(2, tag)[0]) for tag in body_surfaces}
    empty_body_surfaces = [tag for tag in body_surfaces if result["body_surface_triangle_counts"][str(tag)] == 0]
    assert not empty_body_surfaces, (
        f"CAD body surfaces {empty_body_surfaces} generated no triangles; inspect imported periodic seams and face wires")
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("HighOrder")
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coordinates).reshape(-1, 3)
    node_index = {int(tag): index for index, tag in enumerate(node_tags)}
    _, native_tets = gmsh.model.mesh.getElementsByType(11)
    tets = np.array([[node_index[int(tag)] for tag in row] for row in native_tets.reshape(-1, 10)])
    assert len(tets), "No quadratic tetrahedra were generated"
    properties = gmsh.model.mesh.getElementProperties(11)
    gmsh_reference = np.array(properties[4]).reshape(-1, 3)
    target_reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                  [.5, 0., 0.], [.5, .5, 0.], [0., .5, 0.], [0., 0., .5],
                                  [.5, 0., .5], [0., .5, .5]])
    permutation = np.argmin(np.linalg.norm(target_reference[:, None]-gmsh_reference[None], axis=2), axis=1)
    assert np.max(abs(gmsh_reference[permutation]-target_reference)) < 1e-12
    tets = tets[:, permutation]
    triangles, tags = [], []
    for tag, surfaces in [(10, body_surfaces), (100, far_surfaces)]:
        for surface in surfaces:
            _, surface_nodes = gmsh.model.mesh.getElementsByType(9, surface)
            face_rows = [[node_index[int(node)] for node in row] for row in surface_nodes.reshape(-1, 6)]
            triangles.extend(face_rows)
            tags.extend([tag]*len(face_rows))
    triangles, tags = np.array(triangles), np.array(tags)
    gmsh.finalize()
    used = np.unique(tets)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    points, tets, triangles = points[used], remap[tets], remap[triangles]
    assert np.min(triangles) >= 0, "CAD exported a surface node unused by any fluid cell"
    corner_matrix = np.transpose(points[tets[:, 1:4]]-points[tets[:, :1]], (0, 2, 1))
    negative = np.linalg.det(corner_matrix) < 0.
    tets[negative] = tets[negative][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]
    face_map = {}
    for cell, tet in enumerate(tets):
        for omitted in range(4):
            face_map.setdefault(tuple(sorted(np.delete(tet[:4], omitted))), []).append(cell)
    adjacent = np.array([face_map[tuple(sorted(face[:3]))][0] for face in triangles])
    np.savez_compressed(args.out/"topology_input.npz", points=points, tets=tets,
                        boundary_triangles=triangles, boundary_tags=tags, boundary_tet=adjacent)
    points, tets, triangles, tags, adjacent, _, corrections, mismatch = p2.load_mesh(args.out/"topology_input.npz")
    assert mismatch == 0
    np.savez_compressed(args.out/"mesh.npz", points=points, tets=tets,
                        boundary_triangles=triangles, boundary_tags=tags, boundary_tet=adjacent)
    result.update(status="mesh_geometry_checks_passed", shape="imported_closed_cad", points=len(points), tets=len(tets),
                  native_origin=native_origin.tolist(), native_length=native_length,
                  coordinate_system="Mesh and solve coordinates = (native CAD coordinates - native_origin) / native_length",
                  native_area_mapping="A_native = A_mesh * native_length**2; use Sref in mesh units for solve_p2",
                  far_radius=radius, size=args.size, far_size=args.far_size,
                  normalization_origin=[0., 0., 0.], chord=1.,
                  body_surfaces=body_surfaces, far_surfaces=far_surfaces,
                  dense_jacobian_samples_per_element=165, boundary_orientation_corrections=corrections,
                  gmsh_tet10_permutation=permutation.tolist(),
                  mesh_sha256=hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest())
except Exception as exception:
    result.update(exception_type=type(exception).__name__, message=str(exception))
finally:
    if gmsh.isInitialized():
        gmsh.finalize()
result["seconds"] = time.perf_counter()-started
(args.out/("mesh.json" if result["status"] == "mesh_geometry_checks_passed" else "failure.json")).write_text(
    json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
(args.out/"build_cad_mesh.py.snapshot").write_bytes(source)
print(json.dumps(result, indent=2))
if result["status"] != "mesh_geometry_checks_passed":
    raise SystemExit(2)
