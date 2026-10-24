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

parser = argparse.ArgumentParser()
parser.add_argument("--shape", choices=["joukowski", "naca", "file"], default="joukowski")
parser.add_argument("--airfoil", type=Path)
parser.add_argument("--skiprows", type=int, default=0)
parser.add_argument("--size", type=float, default=.04, help="Near-wall size / chord")
parser.add_argument("--radius", type=float, default=8., help="Farfield circle radius / chord")
parser.add_argument("--epsilon", type=float, default=.08)
parser.add_argument("--eta", type=float, default=0.)
parser.add_argument("--thickness", type=float, default=.12)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert args.size > 0. and args.radius > 2.
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
input_hash = None
if args.shape == "joukowski":
    c = complex(-args.epsilon, args.eta)
    assert args.epsilon > 0.
    a = abs(1.-c)
    theta0 = np.angle(1.-c)
    t = np.linspace(0., 2.*np.pi, 241)[:-1]
    zeta = c+a*np.exp(1j*(theta0+t))
    z = zeta+1./zeta
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
area = .5*np.sum(contour[:, 0]*np.roll(contour[:, 1], -1)-contour[:, 1]*np.roll(contour[:, 0], -1))
if area < 0.:
    contour = contour[np.concatenate([[0], np.arange(len(contour)-1, 0, -1)])]
assert abs(area) > 0. and np.all(np.linalg.norm(np.roll(contour, -1, axis=0)-contour, axis=1) > 1e-12)
chord = float(np.ptp(contour[:, 0]))
assert np.argmax(contour[:, 0]) == 0, "Contour must start at a sharp trailing edge"
leading = int(np.argmin(contour[:, 0]))
assert 1 < leading < len(contour)-2
origin = np.array([.5*(contour[:, 0].min()+contour[:, 0].max()), 0.])
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
gmsh.option.setNumber("Mesh.MeshSizeMax", far_radius/3.)
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
occ.synchronize()
curves = [tag for dim, tag in gmsh.model.getBoundary(fluid, oriented=False) if dim == 1]
wall_curves = []
for tag in curves:
    box = gmsh.model.getBoundingBox(1, tag)
    if box[3]-box[0] < 2.*chord:
        wall_curves.append(tag)
assert len(wall_curves) == 2
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "CurvesList", wall_curves)
gmsh.model.mesh.field.setNumber(distance, "Sampling", 300)
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
gmsh.model.mesh.field.setNumber(threshold, "SizeMin", args.size*chord)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", far_radius/3.)
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
np.savez_compressed(args.out/"mesh.npz", points=points, triangles=triangles, edges=edges,
                    edge_tags=edge_tags, edge_triangle=edge_triangle, trailing_node=trailing_node,
                    trailing_edges=trailing_edges, contour=contour)
metadata = dict(shape=args.shape, chord=chord, far_radius=far_radius, farfield_center=origin.tolist(),
                near_wall_size=args.size*chord, epsilon=args.epsilon, eta=args.eta, thickness=args.thickness,
                points=len(points), elements=len(triangles), boundary_edges=len(edges),
                unused_cad_control_nodes_removed=original_point_count-len(points),
                gmsh_version=gmsh.__version__, ncpu_requested=8, input_sha256=input_hash,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                mesh_sha256=hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest(),
                geometry="Interpolating upper/lower splines; curved P2 mesh is an approximation to supplied contour",
                elapsed_seconds=time.perf_counter()-started)
(args.out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2), flush=True)
