"""Conforming upper-half P2 wing mesh, reflected with an explicit wake potential cut."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import root_wake
import planform_root

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"audit"/"runtime"/"python"))
import gmsh
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=float, default=.2)
parser.add_argument("--span", type=float, default=4.)
parser.add_argument("--thickness", type=float, default=.12)
parser.add_argument("--taper", type=float, default=1., help="Tip chord / root chord; root chord is one")
parser.add_argument("--sweep", type=float, default=0., help="Quarter-chord sweep angle in degrees")
parser.add_argument("--radius", type=float, default=5.)
parser.add_argument("--far-size", type=float, default=0., help="Farfield element size in chords; 0 retains legacy radius/3 sizing")
parser.add_argument("--curvature", type=int, default=0, help="Elements per curvature circle; 0 disables curvature sizing")
parser.add_argument("--body-length", type=float, default=0., help="Optional ellipsoid fuselage length")
parser.add_argument("--body-radius", type=float, default=.3)
parser.add_argument("--body-aft", type=float, default=.9, help="Fuselage tail x coordinate")
parser.add_argument("--root-wake", choices=["auto", "merged"], default="auto",
                    help="Merge the wall-contact circulation basis for an aft ellipsoid; auto selects it when needed")
parser.add_argument("--tip-size", type=float, default=0., help="Optional local size around sharp wing-tip curves")
parser.add_argument("--wake-tilt", type=float, default=0., help="Fixed downstream wake angle in the x-z plane, degrees")
parser.add_argument("--wake-strips", type=int, default=0, help="Conforming span strips; 0 retains the original unstructured jump interpolation")
parser.add_argument("--background-view", type=Path, help="NPZ scalar tetra PostView: points, tets, vertex_size")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
source_bytes = Path(__file__).read_bytes()


def record_failure(exception_type, exception, traceback):
    failure = dict(status="failed_mesh", exception_type=exception_type.__name__, message=str(exception),
                   elapsed_seconds=time.perf_counter()-started,
                   source_sha256=hashlib.sha256(source_bytes).hexdigest())
    (args.out/"failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    sys.__excepthook__(exception_type, exception, traceback)


sys.excepthook = record_failure
assert np.isfinite([args.size, args.span, args.thickness, args.radius, args.far_size,
                    args.body_length, args.body_radius, args.body_aft, args.tip_size,
                    args.taper, args.sweep]).all(), "Geometry parameters must be finite"
assert args.size > 0. and args.span > 0. and args.thickness > 0. and args.curvature >= 0 and args.tip_size >= 0.
assert args.taper > 0. and abs(args.sweep) < 89., "Require positive taper and finite nonvertical quarter-chord sweep"
nonrectangular = args.taper != 1. or args.sweep != 0.
assert not nonrectangular or args.wake_strips >= 4 and args.wake_strips % 2 == 0, "Nonrectangular planforms require positive even conforming wake strips"
sweep_tangent = np.tan(np.deg2rad(args.sweep))
tip_le = args.span/2.*sweep_tangent+.25*(1.-args.taper)
tip_te = tip_le+args.taper
minimum_te = min(1., tip_te)
assert min(0., tip_le) > .5-args.radius and max(1., tip_te) < .5+args.radius, "Swept wing must fit strictly inside the fluid box"
assert args.far_size >= 0.
far_size = args.far_size or args.radius/3.
assert args.radius > max(2., args.span/2.+1.)
assert args.body_length >= 0. and args.body_radius > 0.
assert not args.body_length or args.body_radius < args.radius, "Fuselage must fit within the fluid box"
assert args.wake_strips == 0 or args.wake_strips >= 4
assert np.isfinite(args.wake_tilt) and abs(args.wake_tilt) < 30., "Wake tilt currently limited to |angle|<30 degrees"
assert args.body_length == 0. or (0. < args.body_aft < .5+args.radius
                                 and args.body_aft-args.body_length > .5-args.radius), "Fuselage must fit strictly inside the fluid box"
merged_root = bool(args.body_length and args.body_aft >= minimum_te)
assert args.root_wake != "merged" or merged_root, "Explicit merged root requires a fuselage crossing the wing TE plane"
root_interval = None
cad_tolerance = 1e-8
root_coordinate_budget = 0.
root_normal_y = None
root_directional_normal = None
root_te_slope = sweep_tangent+.75*(args.taper-1.)/(args.span/2.)
root_te_x = 1.
if merged_root:
    assert args.body_aft > 1., "Tail exactly on the wing TE plane is a tangent topology; choose a finite crossing or a tail ahead of the TE"
    assert args.wake_strips >= 4 and args.wake_tilt == 0., "Aft-body root requires conforming wake strips and a planar cut"
    assert args.body_aft-args.body_length/2. <= minimum_te, "Merged root requires body maximum girth at or ahead of every TE station, so wall contact narrows downstream"
    if nonrectangular:
        root_y = planform_root.positive_root_contact(1., args.body_aft-args.body_length/2.,
            args.body_length/2., args.body_radius, args.span/2., root_te_slope, args.taper)
        root_interval = np.array([-root_y, root_y])
    else:
        root_interval = root_wake.ellipsoid_root_interval(
            1., args.body_aft-args.body_length/2., args.body_length/2., args.body_radius, args.span/2.)
    root_te_x = 1.+root_te_slope*root_interval[1]
    root_gradient = np.array([(root_te_x-args.body_aft+args.body_length/2.)/(args.body_length/2.)**2,
                              root_interval[1]/args.body_radius**2])
    root_normal_y = float(abs(root_gradient[1])/np.linalg.norm(root_gradient))
    root_directional_normal = float(abs(root_gradient@[root_te_slope, 1.])/np.linalg.norm(root_gradient))
    root_coordinate_budget = cad_tolerance/root_directional_normal
    assert root_coordinate_budget < .01*min(root_interval[1], args.span/2.-root_interval[1]), "CAD resolution insufficient for root contact or remaining exposed TE: projected tolerance exceeds 1% of local feature"
background_data = None
if args.background_view is not None:
    with np.load(args.background_view, allow_pickle=False) as archive:
        background_points = archive["points"]
        background_tets = archive["tets"]
        background_size = archive["vertex_size"]
    assert background_points.ndim == 2 and background_points.shape[1] == 3
    assert background_tets.ndim == 2 and background_tets.shape[1] in (4, 10)
    assert np.issubdtype(background_tets.dtype, np.integer), "Background connectivity must be integer"
    assert background_size.shape == (len(background_points),)
    assert np.isfinite(background_points).all() and np.isfinite(background_size).all()
    assert len(background_tets) > 0 and background_tets.min() >= 0 and background_tets.max() < len(background_points)
    assert background_size.min() > 0., "Background sizes must be positive"
    background_data = True
gmsh_threads = 1 if background_data else 8
gmsh.initialize()
if merged_root:
    gmsh.option.setNumber("Geometry.Tolerance", cad_tolerance)
gmsh.option.setNumber("General.NumThreads", gmsh_threads)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", gmsh_threads)
gmsh.option.setNumber("Mesh.MaxNumThreads3D", gmsh_threads)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
minimum_size = args.size/(12. if args.curvature else 4.)
if args.tip_size:
    minimum_size = min(minimum_size, args.tip_size/2.)
gmsh.option.setNumber("Mesh.MeshSizeMin", minimum_size)
gmsh.option.setNumber("Mesh.MeshSizeMax", far_size)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", args.curvature)
gmsh.model.add("lifting_wing_cut")
occ = gmsh.model.occ
x = .5*(1.-np.cos(np.linspace(0., np.pi, 81)))
z = 5.*args.thickness*(.2969*np.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
z[[0, -1]] = 0.
if not nonrectangular:
    wing_points = [occ.addPoint(float(xi), -args.span/2., float(zi)) for xi, zi in zip(x, z)]
    upper = occ.addSpline(wing_points)
    base = occ.addLine(wing_points[-1], wing_points[0])
    profile = occ.addPlaneSurface([occ.addWire([upper, base])])
    extruded = occ.extrude([(2, profile)], 0., args.span, 0.)
    wing = [entity for entity in extruded if entity[0] == 3]
else:
    section_wires = []
    for station_y, chord in [(-args.span/2., args.taper), (0., 1.), (args.span/2., args.taper)]:
        leading_x = abs(station_y)*sweep_tangent+.25*(1.-chord)
        wing_points = [occ.addPoint(float(leading_x+chord*xi), station_y, float(chord*zi)) for xi, zi in zip(x, z)]
        upper = occ.addSpline(wing_points)
        base = occ.addLine(wing_points[-1], wing_points[0])
        section_wires.append(occ.addWire([upper, base]))
    loft = occ.addThruSections(section_wires, makeSolid=True, makeRuled=True)
    wing = [entity for entity in loft if entity[0] == 3]
    assert len(wing) == 1, "Ruled planform loft must generate one wing solid"
body_center = np.array([args.body_aft-args.body_length/2., 0., 0.])
body_axes = np.array([args.body_length/2., args.body_radius, args.body_radius])
if args.body_length:
    fuselage = occ.addSphere(0., 0., 0., 1.)
    occ.dilate([(3, fuselage)], 0., 0., 0., *body_axes)
    occ.translate([(3, fuselage)], *body_center)
    half_box = occ.addBox(.5-args.radius, -args.radius, 0., 2.*args.radius, 2.*args.radius, args.radius)
    upper_body, _ = occ.intersect([(3, fuselage)], [(3, half_box)])
    wing, _ = occ.fuse(wing, [entity for entity in upper_body if entity[0] == 3])
outer = occ.addBox(.5-args.radius, -args.radius, 0., 2.*args.radius, 2.*args.radius, args.radius)
fluid, _ = occ.cut([(3, outer)], wing)
wake_partitions = np.linspace(-args.span/2., args.span/2., (args.wake_strips or 1)+1)
if merged_root:
    for root_y in root_interval:
        nearest = int(np.argmin(abs(wake_partitions-root_y)))
        if 0 < nearest < len(wake_partitions)-1 and abs(wake_partitions[nearest]-root_y) < max(1e-8, 100.*root_coordinate_budget):
            wake_partitions[nearest] = root_y
        else:
            wake_partitions = np.sort(np.r_[wake_partitions, root_y])
wake_surfaces = []
for left_y, right_y in zip(wake_partitions[:-1], wake_partitions[1:]):
    left_chord = 1.+(args.taper-1.)*abs(2.*left_y/args.span)
    right_chord = 1.+(args.taper-1.)*abs(2.*right_y/args.span)
    left_te = .25+abs(left_y)*sweep_tangent+.75*left_chord
    right_te = .25+abs(right_y)*sweep_tangent+.75*right_chord
    wake_points = [occ.addPoint(xi, yi, 0.) for xi, yi in
                   [(left_te, left_y), (.5+args.radius, left_y),
                    (.5+args.radius, right_y), (right_te, right_y)]]
    wake_lines = [occ.addLine(wake_points[i], wake_points[(i+1) % 4]) for i in range(4)]
    wake_surfaces.append((2, occ.addPlaneSurface([occ.addWire(wake_lines)])))
fragmented, _ = occ.fragment(fluid, wake_surfaces)
fluid = [entity for entity in fragmented if entity[0] == 3]
assert len(fluid) == 1
occ.synchronize()
surfaces = [tag for dim, tag in gmsh.model.getBoundary(fluid, oriented=False) if dim == 2]
wall_surfaces = []
for tag in surfaces:
    box = np.array(gmsh.model.getBoundingBox(2, tag))
    plane_face = abs(box[2]) < 1e-6 and abs(box[5]) < 1e-6
    outer_face = (abs(box[0]-(.5-args.radius)) < 1e-6 and abs(box[3]-(.5-args.radius)) < 1e-6
                  or abs(box[0]-(.5+args.radius)) < 1e-6 and abs(box[3]-(.5+args.radius)) < 1e-6
                  or abs(box[1]+args.radius) < 1e-6 and abs(box[4]+args.radius) < 1e-6
                  or abs(box[1]-args.radius) < 1e-6 and abs(box[4]-args.radius) < 1e-6
                  or abs(box[2]-args.radius) < 1e-6 and abs(box[5]-args.radius) < 1e-6)
    if not plane_face and not outer_face:
        wall_surfaces.append(tag)
assert wall_surfaces
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", wall_surfaces)
gmsh.model.mesh.field.setNumber(distance, "Sampling", 150)
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
gmsh.model.mesh.field.setNumber(threshold, "SizeMin", args.size)
gmsh.model.mesh.field.setNumber(threshold, "SizeMax", far_size)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", .3)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.)
gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
size_fields = [threshold]
tip_curves = []
if args.tip_size:
    for dim, tag in gmsh.model.getBoundary([(2, tag) for tag in wall_surfaces], combined=False, oriented=False):
        box = np.array(gmsh.model.getBoundingBox(1, tag))
        on_tip = (abs(abs(box[1])-args.span/2.) < 1e-6 and abs(box[4]-box[1]) < 1e-6)
        if on_tip and tag not in tip_curves:
            tip_curves.append(tag)
    assert tip_curves
    tip_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(tip_distance, "CurvesList", tip_curves)
    gmsh.model.mesh.field.setNumber(tip_distance, "Sampling", 300)
    tip_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(tip_threshold, "InField", tip_distance)
    gmsh.model.mesh.field.setNumber(tip_threshold, "SizeMin", args.tip_size)
    gmsh.model.mesh.field.setNumber(tip_threshold, "SizeMax", args.size)
    gmsh.model.mesh.field.setNumber(tip_threshold, "DistMin", .025)
    gmsh.model.mesh.field.setNumber(tip_threshold, "DistMax", .15)
    gmsh.model.mesh.field.setNumber(tip_threshold, "StopAtDistMax", 1)
    size_fields.append(tip_threshold)
if background_data:
    print("stage: registering single-thread PostView size field", flush=True)
    background_view_tag = gmsh.view.add("Local target size")
    xyz = background_points[background_tets[:, :4]]
    view_data = np.concatenate([xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2],
                                background_size[background_tets[:, :4]]], axis=1)
    gmsh.view.addListData(background_view_tag, "SS", len(background_tets), view_data.ravel().tolist())
    background_field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(background_field, "ViewTag", background_view_tag)
    size_fields.append(background_field)
if len(size_fields) > 1:
    minimum_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", size_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder")
node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(coordinates).reshape(-1, 3)
node_index = {int(tag): i for i, tag in enumerate(node_tags)}
cad_component = {}
cad_surface_record = {}
for tag in wall_surfaces:
    component = 10
    if args.body_length:
        uv_min, uv_max = gmsh.model.getParametrizationBounds(2, tag)
        uv = np.array(uv_min)[None]+np.array([.25, .5, .75])[:, None]*(np.array(uv_max)-uv_min)[None]
        samples = np.array(gmsh.model.getValue(2, tag, uv.ravel())).reshape(-1, 3)
        level_error = float(np.max(abs(np.sum(((samples-body_center)/body_axes)**2, axis=1)-1.)))
        component = 20 if level_error < 1e-7 else 30
        cad_surface_record[str(tag)] = dict(component=component, ellipsoid_sample_level_error=level_error)
    _, surface_nodes = gmsh.model.mesh.getElementsByType(9, tag)
    for triangle in surface_nodes.reshape(-1, 6):
        cad_component[tuple(sorted(node_index[int(node)] for node in triangle[:3]))] = component
_, tet_nodes = gmsh.model.mesh.getElementsByType(11)
tets = np.array([[node_index[int(tag)] for tag in row] for row in tet_nodes.reshape(-1, 10)])
properties = gmsh.model.mesh.getElementProperties(11)
gmsh_reference = np.array(properties[4]).reshape(10, 3)
desired_reference = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [.5, 0., 0.], [.5, .5, 0.], [0., .5, 0.], [0., 0., .5],
                               [.5, 0., .5], [0., .5, .5]])
permutation = np.argmin(np.linalg.norm(desired_reference[:, None]-gmsh_reference[None], axis=2), axis=1)
assert np.max(abs(gmsh_reference[permutation]-desired_reference)) < 1e-12
tets = tets[:, permutation]
gmsh.finalize()
used = np.unique(tets)
compact = np.full(len(points), -1, dtype=int)
compact[used] = np.arange(len(used))
points = points[used]
tets = compact[tets]
cad_component = {tuple(sorted(compact[list(corners)])): component for corners, component in cad_component.items()}
cad_original_points = points.copy()
if merged_root:
    # OCC intersection vertices have finite tolerance; restore the declared analytic strip planes.
    for root_y in root_interval:
        on_root_plane = ((abs(points[:, 2]) < 1e-9) & (points[:, 0] >= root_te_x-cad_tolerance)
                         & (abs(points[:, 1]-root_y) <= root_coordinate_budget))
        root_vertex = on_root_plane & (abs(points[:, 0]-root_te_x) < cad_tolerance)
        points[on_root_plane, 1] = root_y
        points[root_vertex, 0] = root_te_x
reconciled_nodes = np.flatnonzero(np.any(points != cad_original_points, axis=1))
reconciliation_shift = float(np.max(abs(points-cad_original_points)))
corner_jacobian = np.stack([points[tets[:, i]]-points[tets[:, 0]] for i in [1, 2, 3]], axis=2)
negative = np.linalg.det(corner_jacobian) < 0.
tets[negative] = tets[negative][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]
local_faces = [[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
               [0, 2, 3, 6, 9, 7], [1, 2, 3, 5, 9, 8]]
face_nodes = np.concatenate([tets[:, face] for face in local_faces])
face_tet = np.tile(np.arange(len(tets)), 4)
_, inverse, counts = np.unique(np.sort(face_nodes[:, :3], axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts <= 2)
exterior = counts[inverse] == 1
faces = face_nodes[exterior]
adjacent = face_tet[exterior]
vertices = points[faces]
plane = np.max(abs(vertices[:, :, 2]), axis=1) < 1e-9
centers = vertices.mean(axis=1)
far = ((np.max(abs(vertices[:, :, 0]-(.5-args.radius)), axis=1) < 1e-8)
       | (np.max(abs(vertices[:, :, 0]-(.5+args.radius)), axis=1) < 1e-8)
       | (np.max(abs(vertices[:, :, 1]+args.radius), axis=1) < 1e-8)
       | (np.max(abs(vertices[:, :, 1]-args.radius), axis=1) < 1e-8)
       | (np.max(abs(vertices[:, :, 2]-args.radius), axis=1) < 1e-8))
center_chord = 1.+(args.taper-1.)*np.minimum(abs(2.*centers[:, 1]/args.span), 1.)
center_te = .25+abs(centers[:, 1])*sweep_tangent+.75*center_chord
wake_mask = plane & (centers[:, 0] > center_te) & (abs(centers[:, 1]) < args.span/2.)
wake_nodes = np.unique(faces[wake_mask])
cut_nodes = wake_nodes[(abs(points[wake_nodes, 1]) < args.span/2.-1e-9)]
cut_chord = 1.+(args.taper-1.)*abs(2.*points[cut_nodes, 1]/args.span)
cut_te = .25+abs(points[cut_nodes, 1])*sweep_tangent+.75*cut_chord
assert np.all(points[cut_nodes, 0] > cut_te-1e-9)
on_plane = abs(points[:, 2]) < 1e-9
points[on_plane, 2] = 0.
duplicated = ~on_plane
duplicated[cut_nodes] = True
reflected = np.arange(len(points))
reflected[duplicated] = np.arange(len(points), len(points)+duplicated.sum())
lower_tets = reflected[tets][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]
physical_faces = ~plane
boundary = np.vstack([faces[physical_faces], reflected[faces[physical_faces]]])
boundary_tet = np.concatenate([adjacent[physical_faces], adjacent[physical_faces]+len(tets)])
physical_tags = np.array([100 if is_far else cad_component[tuple(sorted(face[:3]))]
                          for face, is_far in zip(faces[physical_faces], far[physical_faces])])
boundary_tags = np.tile(physical_tags, 2)
assert not args.body_length or all(tag in boundary_tags for tag in [20, 30])
upper_count = len(points)
points = np.vstack([points, points[duplicated]*[1., 1., -1.]])
tets = np.vstack([tets, lower_tets])
te_nodes = cut_nodes[abs(points[cut_nodes, 0]-cut_te) < 1e-8]
te_nodes = te_nodes[np.argsort(points[te_nodes, 1])]
assert len(te_nodes) > 2
tip_y = np.array([-args.span/2., args.span/2.])
stations = np.concatenate([[tip_y[0]], points[te_nodes, 1], [tip_y[1]]])
if args.wake_strips:
    stations = wake_partitions
    exposed_stations = stations[1:-1]
    if merged_root:
        exposed_stations = exposed_stations[(exposed_stations <= root_interval[0]+1e-12)
                                            | (exposed_stations >= root_interval[1]-1e-12)]
    matching = np.argmin(abs(points[te_nodes, 1, None]-exposed_stations[None]), axis=0)
    te_nodes = te_nodes[matching]
    assert len(np.unique(te_nodes)) == len(te_nodes)
    assert np.max(abs(points[te_nodes, 1]-exposed_stations)) < 1e-9, "Exposed CAD strip knots must coincide with TE nodes"
assert np.all(np.diff(stations) > 1e-10)
station_dof = np.r_[-1, np.arange(len(stations)-2), -1]
root_support = None
if merged_root:
    station_dof = root_wake.merge_root_basis(stations, root_interval, args.span/2.)
    exposed_intervals = np.array([[-args.span/2., root_interval[0]], [root_interval[1], args.span/2.]])
    support = root_wake.mapped_weak_test_support(stations, station_dof, exposed_intervals)
    assert np.all(support.sum(axis=0) > 0.), "Every merged circulation DOF needs exposed TE support"
    root_support = dict(status="constructed", root_interval_y=root_interval.tolist(),
        root_te_x=root_te_x, te_semispan_slope=root_te_slope,
        exposed_te_intervals_y=exposed_intervals.tolist(), station_to_dof=station_dof.tolist(),
        weak_test_support_integrals=support.tolist(),
        condition="Constant potential jump on connected smooth-body contact; central circulation remains unknown",
        limitations="Symmetric straight-TE planform, x-aligned ellipsoid narrowing aft of every TE station, fixed planar cut, weak Kutta only")
reference_points = points.copy()
if args.wake_tilt:
    local_chord = 1.+(args.taper-1.)*np.minimum(abs(2.*points[:, 1]/args.span), 1.)
    local_te = .25+np.minimum(abs(points[:, 1]), args.span/2.)*sweep_tangent+.75*local_chord
    downstream = np.maximum(points[:, 0]-local_te, 0.)
    span_decay = np.maximum((abs(points[:, 1])-args.span/2.)/(args.radius-args.span/2.), 0.)
    displacement = (downstream*np.tan(np.deg2rad(args.wake_tilt))
                    *(1.-abs(points[:, 2])/args.radius)**2*(1.-span_decay)**2)
    physical_body_nodes = np.unique(boundary[boundary_tags != 100])
    assert np.max(abs(displacement[physical_body_nodes])) < 1e-12, "Wake deformation must leave the solid geometry fixed"
    points[:, 2] += displacement
barycentric = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
_, derivative = p2.p2_shape(barycentric)
minimum_jacobian = np.inf
for first in range(0, len(tets), 256):
    jacobian = np.einsum("eia,qib->eqab", points[tets[first:first+256]], derivative)
    minimum_jacobian = min(minimum_jacobian, float(np.linalg.det(jacobian).min()))
assert minimum_jacobian > 0., "Dense-sampled curved mesh Jacobian is nonpositive"
np.savez_compressed(args.out/"mesh.npz", points=points, tets=tets, boundary_triangles=boundary,
                    reference_points=reference_points,
                    boundary_tags=boundary_tags, boundary_tet=boundary_tet,
                    wake_upper=faces[wake_mask], wake_lower=reflected[faces[wake_mask]],
                    wake_upper_tet=adjacent[wake_mask], wake_lower_tet=adjacent[wake_mask]+len(tets)//2,
                    cut_upper_nodes=cut_nodes, cut_lower_nodes=reflected[cut_nodes],
                    te_upper_nodes=te_nodes, te_lower_nodes=reflected[te_nodes], gamma_stations=stations,
                    gamma_station_dof=station_dof,
                    cad_reconciled_upper_nodes=reconciled_nodes,
                    cad_original_upper_coordinates=cad_original_points[reconciled_nodes])
metadata = dict(shape="naca_symmetric_rectangular_wing" if not nonrectangular else "naca_symmetric_trapezoidal_wing",
                span=args.span, chord=1., thickness=args.thickness,
                taper_ratio=args.taper, quarter_chord_sweep_deg=args.sweep,
                tip_le_x=tip_le, tip_te_x=tip_te,
                far_radius=args.radius, size=args.size, reference_area=args.span*(1.+args.taper)/2.,
                far_size=far_size,
                background_size_field=(dict(source=str(args.background_view.resolve()),
                    source_sha256=hashlib.sha256(args.background_view.read_bytes()).hexdigest(),
                    gmsh_field_type="PostView", gmsh_data_type="SS",
                    combination="Min with original distance and optional tip fields") if background_data else None),
                gmsh_threads=gmsh_threads,
                curvature_elements_per_circle=args.curvature,
                tip_size=args.tip_size, tip_curves=tip_curves,
                wake_tilt_deg=args.wake_tilt,
                wake_strips=args.wake_strips,
                actual_wake_strips=len(wake_partitions)-1,
                root_wake_support=root_support,
                cad_root_reconciliation=dict(upper_nodes=len(reconciled_nodes),
                    max_coordinate_shift=reconciliation_shift, cad_absolute_tolerance=cad_tolerance,
                    absolute_tolerance=root_coordinate_budget, root_normal_y=root_normal_y,
                    root_directional_normal=root_directional_normal,
                    budget_definition="CAD tolerance / abs(n_y + TE_slope*n_x); first-order intersection sensitivity in span coordinate",
                    feature_fraction_limit=.01,
                    operation="Restore analytic root-strip y on z=0 downstream nodes and exact root-TE vertex x; originals saved in NPZ",
                    applied=merged_root),
                jump_representation=("Exactly linear in y in each conforming CAD wake strip"
                                     if args.wake_strips else "Nodal interpolation on unstructured wake; streamwise constancy is approximate"),
                dense_jacobian_samples_per_element=len(barycentric),
                min_dense_jacobian=minimum_jacobian,
                points=len(points), tets=len(tets), upper_points=upper_count,
                gamma_unknowns=int(station_dof.max()+1), gmsh_tet10_permutation=permutation.tolist(),
                body_length=args.body_length, body_radius=args.body_radius, body_aft=args.body_aft,
                body_center=body_center.tolist(), body_axes=body_axes.tolist(),
                cad_component_surfaces=cad_surface_record,
                component_tagging="Classify CAD underlying surfaces with 3 analytic ellipsoid level samples; propagate CAD element ownership",
                geometry="OCC NACA spline extrusion or three-section ruled loft; local chord-scaled thickness; planar tips",
                wake="Fixed tilted downstream cut; piecewise-linear potential jump in y; zero at wing tips",
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                mesh_sha256=hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest(),
                ncpu_requested=8, elapsed_seconds=time.perf_counter()-started)
if args.body_length:
    metadata["shape"] = "ellipsoid_fuselage_naca_wing_union"
(args.out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(args.out/"build_mesh.py.snapshot").write_bytes(source_bytes)
(args.out/"root_wake.py.snapshot").write_bytes(Path(root_wake.__file__).read_bytes())
(args.out/"planform_root.py.snapshot").write_bytes(Path(planform_root.__file__).read_bytes())
print(json.dumps(metadata, indent=2))
