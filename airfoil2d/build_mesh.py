"""Gmsh P2 airfoil meshes with explicit trailing-edge and edge-to-element topology."""
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
from p2 import shape_functions

parser = argparse.ArgumentParser()
parser.add_argument("--shape", choices=["joukowski", "trefftz", "naca", "file"], default="joukowski")
parser.add_argument("--airfoil", type=Path)
parser.add_argument("--skiprows", type=int, default=0)
parser.add_argument("--size", type=float, default=.04, help="Near-wall size / chord")
parser.add_argument("--radius", type=float, default=8., help="Farfield circle radius / chord")
parser.add_argument("--far-size", type=float, default=1., help="Farfield mesh size / chord")
parser.add_argument("--epsilon", type=float, default=.08)
parser.add_argument("--eta", type=float, default=0.)
parser.add_argument("--thickness", type=float, default=.12)
parser.add_argument("--te-exponent", type=float, default=1.9)
parser.add_argument("--mirror", action="store_true", help="Mesh the upper half and reflect a symmetric contour")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--translate", type=float, nargs=2, default=[0., 0.])
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)


def record_failure(exception_type, exception, traceback):
    failure = dict(status="failed_exception", exception_type=exception_type.__name__, message=str(exception),
                   elapsed_seconds=time.perf_counter()-started,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/"failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    sys.__excepthook__(exception_type, exception, traceback)


sys.excepthook = record_failure
assert np.isfinite([args.size, args.radius, args.far_size, args.scale]).all()
assert args.size > 0. and args.radius > 2. and args.far_size > 0. and args.scale > 0.
input_hash = None
mapping_exponent = args.te_exponent if args.shape == "trefftz" else 2.
if args.shape in ["joukowski", "trefftz"]:
    c = complex(-args.epsilon, args.eta)
    assert args.epsilon > 0.
    a = abs(1.-c)
    theta0 = np.angle(1.-c)
    t = np.linspace(0., 2.*np.pi, 241)[:-1]
    zeta = c+a*np.exp(1j*(theta0+t))
    if args.shape == "joukowski":
        z = zeta+1./zeta
    else:
        assert 1. < mapping_exponent < 2.
        assert args.eta == 0., "Noninteger mapping branch is currently validated only for symmetric contours"
        ratio_power = ((zeta-1.)/(zeta+1.))**mapping_exponent
        z = mapping_exponent*(1.+ratio_power)/(1.-ratio_power)
    contour = np.column_stack([z.real, z.imag])
elif args.shape == "naca":
    x = .5*(1.-np.cos(np.linspace(0., np.pi, 121)))
    y = 5.*args.thickness*(.2969*np.sqrt(x)-.1260*x-.3516*x**2+.2843*x**3-.1036*x**4)
    contour = np.column_stack([np.concatenate([x[::-1], x[1:-1]]),
                               np.concatenate([y[::-1], -y[1:-1]])])
else:
    assert args.airfoil is not None
    input_hash = hashlib.sha256(args.airfoil.read_bytes()).hexdigest()
    contour = np.loadtxt(args.airfoil, skiprows=args.skiprows)[:, :2]
    if np.linalg.norm(contour[0]-contour[-1]) < 1e-12*np.ptp(contour[:, 0]):
        contour = contour[:-1]
assert np.isfinite(contour).all() and len(contour) >= 8
contour = contour*args.scale+np.asarray(args.translate)
chord_physical = float(np.ptp(contour[:, 0]))
assert chord_physical > 0.
normalization_origin = (contour.min(axis=0)+contour.max(axis=0))/2.
contour = (contour-normalization_origin)/chord_physical
segment_start = contour
segment_end = np.roll(contour, -1, axis=0)
ab = segment_end-segment_start
ac = segment_start[None]-segment_start[:, None]
ad = segment_end[None]-segment_start[:, None]
cd = ab[None]
ca = -ac
cb = segment_end[:, None]-segment_start[None]
side_c = ab[:, None, 0]*ac[:, :, 1]-ab[:, None, 1]*ac[:, :, 0]
side_d = ab[:, None, 0]*ad[:, :, 1]-ab[:, None, 1]*ad[:, :, 0]
side_a = cd[:, :, 0]*ca[:, :, 1]-cd[:, :, 1]*ca[:, :, 0]
side_b = cd[:, :, 0]*cb[:, :, 1]-cd[:, :, 1]*cb[:, :, 0]
assert not np.any((side_c*side_d < -1e-24) & (side_a*side_b < -1e-24)), "Self-intersecting contour"
area = .5*np.sum(contour[:, 0]*np.roll(contour[:, 1], -1)-contour[:, 1]*np.roll(contour[:, 0], -1))
if area < 0.:
    contour = contour[np.concatenate([[0], np.arange(len(contour)-1, 0, -1)])]
assert abs(area) > 0. and np.all(np.linalg.norm(np.roll(contour, -1, axis=0)-contour, axis=1) > 1e-12)
chord = float(np.ptp(contour[:, 0]))
assert np.argmax(contour[:, 0]) == 0, "Contour must start at a sharp trailing edge"
leading = int(np.argmin(contour[:, 0]))
assert 1 < leading < len(contour)-2
origin = np.zeros(2)
far_radius = args.radius*chord
gmsh.initialize()
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", 8)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.size*chord/8.)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.far_size*chord)
gmsh.model.add("airfoil_p2")
occ = gmsh.model.occ
vertices = [occ.addPoint(float(x), float(y), 0.) for x, y in contour]
upper = occ.addSpline(vertices[:leading+1])
lower = occ.addSpline(vertices[leading:]+[vertices[0]])
wire = occ.addWire([upper, lower])
body = occ.addPlaneSurface([wire])
outer = occ.addDisk(float(origin[0]), float(origin[1]), 0., far_radius, far_radius)
fluid, _ = occ.cut([(2, outer)], [(2, body)])
assert len(fluid) == 1
if args.mirror:
    reflected_contour = contour*np.array([1., -1.])
    symmetry_error = np.min(np.linalg.norm(contour[:, None]-reflected_contour[None], axis=2), axis=1).max()
    assert symmetry_error < 1e-8*chord, "Mirror construction requires symmetric input geometry"
    upper_box = occ.addRectangle(float(origin[0]-1.1*far_radius), 0., 0., 2.2*far_radius, 1.1*far_radius)
    fluid, _ = occ.intersect(fluid, [(2, upper_box)])
    assert len(fluid) == 1
occ.synchronize()
curves = [tag for dim, tag in gmsh.model.getBoundary(fluid, oriented=False) if dim == 1]
wall_curves = []
for tag in curves:
    box = gmsh.model.getBoundingBox(1, tag)
    if box[3]-box[0] < 2.*chord:
        wall_curves.append(tag)
assert len(wall_curves) == (1 if args.mirror else 2)
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "CurvesList", wall_curves)
gmsh.model.mesh.field.setNumber(distance, "Sampling", 300)
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
gmsh.model.mesh.field.setNumber(threshold, "SizeMin", args.size*chord)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", args.far_size*chord)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", .05*chord)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.*chord)
gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder")
node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(coordinates).reshape(-1, 3)[:, :2]
node_index = {int(tag): index for index, tag in enumerate(node_tags)}
_, triangle_nodes = gmsh.model.mesh.getElementsByType(9)
triangles = np.array([[node_index[int(tag)] for tag in row] for row in triangle_nodes.reshape(-1, 6)],
                     dtype=np.int64)
_, edge_nodes = gmsh.model.mesh.getElementsByType(8)
edges = np.array([[node_index[int(tag)] for tag in row] for row in edge_nodes.reshape(-1, 3)],
                dtype=np.int64)
gmsh.finalize()
original_point_count = len(points)
used = np.unique(triangles)
compact_index = np.full(len(points), -1, dtype=np.int64)
compact_index[used] = np.arange(len(used))
points = points[used]
triangles = compact_index[triangles]
edges = compact_index[edges]
assert np.all(edges >= 0.), "Boundary edge has a node outside the fluid mesh"
unused_control_nodes = original_point_count-len(points)
if args.mirror:
    assert points[:, 1].min() > -1e-10*chord
    on_axis = abs(points[:, 1]) < 1e-10*chord
    points[on_axis, 1] = 0.
    axis_edges = np.all(on_axis[edges], axis=1)
    edges = edges[~axis_edges]
    reflected_index = np.arange(len(points))
    reflected_index[~on_axis] = np.arange(len(points), len(points)+np.count_nonzero(~on_axis))
    mirrored_triangles = reflected_index[triangles][:, [0, 2, 1, 5, 4, 3]]
    mirrored_edges = reflected_index[edges]
    points = np.vstack([points, points[~on_axis]*np.array([1., -1.])])
    triangles = np.vstack([triangles, mirrored_triangles])
    edges = np.vstack([edges, mirrored_edges])
face_map = {}
for element, triangle in enumerate(triangles):
    for pair in [(0, 1), (1, 2), (2, 0)]:
        face_map.setdefault(tuple(sorted(triangle[list(pair)])), []).append(element)
edge_triangle = np.array([face_map[tuple(sorted(edge[:2]))][0] for edge in edges], dtype=np.int64)
assert all(len(face_map[tuple(sorted(edge[:2]))]) == 1 for edge in edges)
assert len(edges) == sum(len(adjacent) == 1 for adjacent in face_map.values())
radii = np.linalg.norm(points[edges]-origin, axis=2)
edge_tags = np.where(np.max(abs(radii-far_radius), axis=1) < 1e-7*far_radius, 100, 10)
trailing_node = int(np.argmin(np.linalg.norm(points-contour[0], axis=1)))
assert np.linalg.norm(points[trailing_node]-contour[0]) < 1e-10*chord
trailing_edges = np.flatnonzero((edge_tags == 10) & np.any(edges[:, :2] == trailing_node, axis=1))
assert len(trailing_edges) == 2
dense_reference = np.array([[i/20., j/20.] for i in range(21) for j in range(21-i)])
_, dense_derivative = shape_functions(dense_reference)
dense_jacobian = np.einsum("eia,qib->eqab", points[triangles], dense_derivative)
dense_determinant = np.linalg.det(dense_jacobian)
assert np.all(dense_determinant > 0.), "Curved mesh has a nonpositive dense-sampled Jacobian"
points = points*chord_physical+normalization_origin
contour = contour*chord_physical+normalization_origin
np.savez_compressed(args.out/"mesh.npz", points=points, triangles=triangles, edges=edges,
                    edge_tags=edge_tags, edge_triangle=edge_triangle, trailing_node=trailing_node,
                    trailing_edges=trailing_edges, contour=contour)
metadata = dict(shape=args.shape, chord=chord_physical, far_radius=far_radius*chord_physical,
                farfield_center=normalization_origin.tolist(),
                normalization_origin=normalization_origin.tolist(), normalization_length=chord_physical,
                coordinate_scale=args.scale, coordinate_translation=args.translate,
                near_wall_size=args.size*chord_physical, far_size_chord=args.far_size,
                far_mesh_size=args.far_size*chord_physical, mesh_size_min=args.size*chord_physical/8.,
                threshold_dist_min_chord=.05, threshold_dist_max_chord=2.,
                leading_edge=contour[leading].tolist(),
                epsilon=args.epsilon, eta=args.eta, thickness=args.thickness,
                mapping_exponent=mapping_exponent,
                points=len(points), elements=len(triangles), boundary_edges=len(edges),
                unused_cad_control_nodes_removed=unused_control_nodes,
                dense_jacobian_samples_per_element=len(dense_reference),
                min_normalized_dense_jacobian=float(dense_determinant.min()),
                mirror_mesh=args.mirror,
                gmsh_version=gmsh.__version__, ncpu_requested=8, input_sha256=input_hash,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                mesh_sha256=hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest(),
                geometry="Interpolating upper/lower splines; curved P2 mesh is an approximation to supplied contour",
                elapsed_seconds=time.perf_counter()-started)
(args.out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
np.savetxt(args.out/"contour.dat", contour)
(args.out/"build_mesh_source.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(metadata, indent=2), flush=True)
