"""Differentiable source/doublet panels with an explicit body-contact wake cut.

Run from the repository root. Topology is fixed; geometry, the physical kernel,
weak trailing-edge jump equations, pressure, and loads are differentiated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--bulge", type=float, default=0.)
parser.add_argument("--span", type=float, default=1.)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--derivatives", action="store_true")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--wake-mode", choices=("fixed", "freestream", "far-freestream"), default="fixed")
parser.add_argument("--wake-offset", type=Path, help="Research wake offsets in normalized body coordinates; frozen during AD")
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists(), "Preserve earlier experiments"
args.out.mkdir(parents=True)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = str(args.ncpu)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["TMP"] = os.environ["TEMP"] = str(args.out / "tmp")
(args.out / "tmp").mkdir()
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import lu_factor
from scipy.linalg.lapack import dgecon
from threadpoolctl import threadpool_limits, threadpool_info
import jax
import jax.numpy as jnp
from kernel import triangle_potential
from wake_geometry import check_wake_geometry

jax.config.update("jax_enable_x64", True)
threadpool_limits(args.ncpu)
started = time.perf_counter()
stage = "input_validation"


def record_failure(exception_type, exception, traceback):
    failure = dict(status="input_rejected" if stage == "input_validation" else "failed",
                   stage=stage, exception=exception_type.__name__, message=str(exception),
                   elapsed_seconds=time.perf_counter() - started)
    (args.out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    sys.__excepthook__(exception_type, exception, traceback)


sys.excepthook = record_failure
fixture = json.loads(args.surface.read_text(encoding="utf-8"))
(args.out / "surface_input.json").write_bytes(args.surface.read_bytes())
assert fixture["schema"] == "dpanel.planar_cut.v1", "Require an explicit planar fluid-cut fixture"
xyz_native = np.asarray(fixture["coordinates"], dtype=float)
faces = np.asarray(fixture["faces"], dtype=np.int64)
tags = np.asarray(fixture["component_ids"], dtype=np.int64)
wake_faces = np.asarray(fixture["wake_faces"], dtype=np.int64)
cut_pairs = np.asarray(fixture["cut_face_pairs"], dtype=np.int64)
cut_edges = np.asarray(fixture["cut_edges"], dtype=np.int64)
te_edges = np.asarray(fixture["trailing_edges"], dtype=np.int64)
te_upper, te_lower = np.asarray(fixture["upper_te_faces"]), np.asarray(fixture["lower_te_faces"])
origin, chord = np.asarray(fixture["origin"]), float(fixture["reference_chord"])
sref = float(fixture["reference_area"]) / chord**2
reference = (np.asarray(fixture["moment_reference"]) - origin) / chord
assert np.isfinite([args.alpha, args.bulge, args.span]).all(), "Finite flow and deformation parameters required"
assert args.ncpu > 0, "NCPU must be positive"
assert args.span > 0 and args.bulge > -1, "Deformation must retain positive span and thickness"
assert chord > 0 and sref > 0 and np.isfinite(xyz_native).all()
assert np.isfinite(origin).all() and np.isfinite(reference).all() and np.isfinite([chord, sref]).all(), "Finite reference scales and points required"
xyz = (xyz_native - origin) / chord
assert faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < len(xyz)
assert tags.shape == (len(faces),) and set(tags).issubset({10, 20, 30})
assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces), "Duplicate wall panels"
triangles = xyz[faces]
area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
area = np.linalg.norm(area_vectors, axis=1)
assert np.all(area > 0), "Degenerate wall panels"
normals = area_vectors / area[:, None]
centers = triangles.mean(axis=1)
edges_all = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
ids = np.tile(np.arange(len(faces)), 3)
unique, inverse, counts = np.unique(np.sort(edges_all, axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(counts == 2), "Wall must be a closed two-manifold"
order = np.argsort(inverse).reshape(-1, 2)
assert np.all(edges_all[order[:, 0]] == edges_all[order[:, 1], ::-1]), "Inconsistent wall winding"
pairs = ids[order]
graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(faces), len(faces)))
components, labels = connected_components(graph, directed=False)
volumes = np.bincount(labels, weights=np.sum(centers * area_vectors, axis=1) / 3)
assert np.all(volumes > 0), "Inward wall component"
edge_lookup = {tuple(edge): i for i, edge in enumerate(unique)}
assert len(cut_edges) == len(cut_pairs) == len(fixture["cut_kind"])
assert set(fixture["cut_kind"]).issubset({"trailing_edge", "smooth_wall_contact"})
assert len(np.unique(np.sort(cut_edges, axis=1), axis=0)) == len(cut_edges), "Duplicate cut edges"
for edge, pair in zip(cut_edges, cut_pairs):
    assert tuple(sorted(edge)) in edge_lookup, "Cut edge must be an actual wall edge"
    assert set(pair) == set(pairs[edge_lookup[tuple(sorted(edge))]]), "Cut banks disagree with wall adjacency"
    assert centers[pair[0], 2] > 0 > centers[pair[1], 2], "Cut banks must be ordered upper/lower"
assert wake_faces.ndim == 2 and wake_faces.shape[1] == 3
assert wake_faces.min() >= 0 and wake_faces.max() < len(xyz)
assert len(np.unique(np.sort(wake_faces, axis=1), axis=0)) == len(wake_faces)
wake_tri = xyz[wake_faces]
wake_cross = np.cross(wake_tri[:, 1] - wake_tri[:, 0], wake_tri[:, 2] - wake_tri[:, 0])
assert np.all(wake_cross[:, 2] > 0) and np.max(abs(wake_tri[:, :, 2])) < 1e-12, "Wake must have positive-z planar orientation"
wake_edges_all = np.concatenate((wake_faces[:, [0, 1]], wake_faces[:, [1, 2]], wake_faces[:, [2, 0]]))
wake_unique, wake_inverse, wake_counts = np.unique(np.sort(wake_edges_all, axis=1), axis=0, return_inverse=True, return_counts=True)
assert np.all(wake_counts <= 2), "Nonmanifold wake"
wake_boundary = wake_unique[wake_counts == 1]
wall_intersection = {tuple(edge) for edge in wake_boundary if tuple(edge) in edge_lookup}
assert wall_intersection == {tuple(sorted(edge)) for edge in cut_edges}, "Declared cut must equal the actual wall/wake boundary intersection"
boundary_degree = np.bincount(wake_boundary.ravel(), minlength=len(xyz))
assert np.all((boundary_degree == 0) | (boundary_degree == 2)), "Wake boundary must consist of closed curves"
te_cut = {tuple(sorted(edge)): pair for edge, pair, kind in zip(cut_edges, cut_pairs, fixture["cut_kind"]) if kind == "trailing_edge"}
assert len(te_edges) == len(te_upper) == len(te_lower) == len(te_cut), "TE list must cover all actual TE cut edges"
assert {tuple(sorted(edge)) for edge in te_edges} == set(te_cut), "TE list disagrees with the cut"
for edge, upper, lower in zip(te_edges, te_upper, te_lower):
    assert np.array_equal([upper, lower], te_cut[tuple(sorted(edge))]), "TE bank pairing disagrees with the actual cut"

# Independent solid angle membership test; CAD Boolean provenance alone is not
# evidence that the supplied wake is outside the solid. This samples face centers.
winding_max = 0.
for first in range(0, len(wake_tri), 128):
    r = triangles[None] - wake_tri.mean(axis=1)[first:first + 128, None, None]
    rad = np.linalg.norm(r, axis=3)
    r0, r1, r2 = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    det = np.sum(r0 * np.cross(r1, r2), axis=2)
    den = rad.prod(axis=2) + np.sum(r0 * r1, axis=2) * rad[:, :, 2]
    den += np.sum(r1 * r2, axis=2) * rad[:, :, 0] + np.sum(r2 * r0, axis=2) * rad[:, :, 1]
    winding_max = max(winding_max, float(np.max(abs(np.sum(2 * np.arctan2(det, den), axis=1) / (4 * np.pi)))))
assert winding_max < 1e-8, "Wake contains face centers inside a solid"
stations = (np.asarray(fixture["gamma_stations"]) - origin[1]) / chord
station_dof = np.asarray(fixture["gamma_station_dof"], dtype=int)
assert stations.shape == station_dof.shape and np.isfinite(stations).all()
assert len(stations) >= 3 and np.all(station_dof >= -1), "Circulation map uses -1 only for fixed zero values"
assert np.all(np.diff(stations) > 0) and station_dof[0] == station_dof[-1] == -1
free_boundary = np.array([edge for edge in wake_boundary if tuple(edge) not in wall_intersection], dtype=np.int64)
on_outflow = np.all(abs(xyz[free_boundary, 0] - wake_tri[:, :, 0].max()) < 1e-10, axis=1)
on_tip = np.any(np.all(abs(xyz[free_boundary, 1, None] - stations[[0, -1]]) < 1e-10, axis=1), axis=1)
assert np.all(on_outflow | on_tip), "Unclassified free wake boundary; require TE/contact, wingtip or downstream truncation"
direction = np.where(wake_edges_all[:, 0] < wake_edges_all[:, 1], 1, -1)
edge_balance = np.bincount(wake_inverse, weights=direction)
assert np.all(edge_balance[wake_counts == 2] == 0), "Wake interior orientations do not cancel"
boundary_graph = coo_matrix((np.ones(len(wake_boundary)), (wake_boundary[:, 0], wake_boundary[:, 1])),
                            shape=(len(xyz), len(xyz)))
_, boundary_labels = connected_components(boundary_graph, directed=False)
boundary_loops = []
for label in np.unique(boundary_labels[wake_boundary.ravel()]):
    loop_edges = wake_boundary[boundary_labels[wake_boundary[:, 0]] == label]
    boundary_loops.append(dict(edges=len(loop_edges),
                               wall_cut_edges=sum(tuple(edge) in wall_intersection for edge in loop_edges),
                               x_range=[float(xyz[loop_edges, 0].min()), float(xyz[loop_edges, 0].max())],
                               y_range=[float(xyz[loop_edges, 1].min()), float(xyz[loop_edges, 1].max())]))

# A face center outside a body does not exclude an intersection. Intersect every
# planar wall triangle with z=0, then clip each resulting segment against the
# open interior of every wake triangle. Boundary-coincident contact is allowed.
plane_tolerance, barycentric_tolerance = 1e-12, 1e-10
slice_segments = []
for tri in triangles:
    z = tri[:, 2]
    on_plane = abs(z) <= plane_tolerance
    assert on_plane.sum() < 3, "Wall patch lies in the wake plane; ambiguous fluid cut"
    if z.min() > plane_tolerance or z.max() < -plane_tolerance:
        continue
    intersection = [point[:2] for point in tri[on_plane]]
    for k, q in ((0, 1), (1, 2), (2, 0)):
        if not on_plane[k] and not on_plane[q] and z[k] * z[q] < 0:
            intersection.append((tri[k] + z[k] / (z[k] - z[q]) * (tri[q] - tri[k]))[:2])
    if len(intersection) == 2:
        slice_segments.append(intersection)
slice_segments = np.asarray(slice_segments).reshape(-1, 2, 2)
wa = wake_tri[:, 0, :2]
we1, we2 = wake_tri[:, 1, :2] - wa, wake_tri[:, 2, :2] - wa
den = we1[:, 0] * we2[:, 1] - we1[:, 1] * we2[:, 0]
for first in range(0, len(slice_segments), 128):
    delta = slice_segments[first:first + 128, :, None] - wa[None, None]
    u = (delta[:, :, :, 0] * we2[None, None, :, 1] - delta[:, :, :, 1] * we2[None, None, :, 0]) / den
    v = (we1[None, None, :, 0] * delta[:, :, :, 1] - we1[None, None, :, 1] * delta[:, :, :, 0]) / den
    barycentric = np.stack((1 - u - v, u, v), axis=-1)
    start, slope = barycentric[:, 0], barycentric[:, 1] - barycentric[:, 0]
    bounds = np.divide(-start, slope, out=np.zeros_like(slope), where=slope != 0)
    enter = np.maximum(0., np.max(np.where(slope > 0, bounds, -np.inf), axis=-1))
    leave = np.minimum(1., np.min(np.where(slope < 0, bounds, np.inf), axis=-1))
    middle = start + ((enter + leave) / 2)[:, :, None] * slope
    crossing = (leave > enter) & np.all(middle > barycentric_tolerance, axis=-1)
    assert not crossing.any(), "Wake triangle interior intersects the solid wall slice"
gamma_count = int(station_dof.max() + 1)
assert np.array_equal(np.unique(station_dof[station_dof >= 0]), np.arange(gamma_count))
hat_map = np.zeros((len(stations), gamma_count))
hat_map[np.flatnonzero(station_dof >= 0), station_dof[station_dof >= 0]] = 1
interval = np.minimum(np.searchsorted(stations, wake_tri.mean(axis=1)[:, 1], side="right") - 1, len(stations) - 2)
assert np.all(interval >= 0)
assert np.all(wake_tri[:, :, 1].min(axis=1) >= stations[interval] - 1e-10)
assert np.all(wake_tri[:, :, 1].max(axis=1) <= stations[interval + 1] + 1e-10), "Wake triangles must conform to circulation strips"
fraction = (wake_tri.mean(axis=1)[:, 1] - stations[interval]) / np.diff(stations)[interval]
wake_basis = (1 - fraction[:, None]) * hat_map[interval] + fraction[:, None] * hat_map[interval + 1]
wake_slope = (hat_map[interval + 1] - hat_map[interval]) / np.diff(stations)[interval, None]
gauss = np.array([.5 - np.sqrt(3) / 6, .5 + np.sqrt(3) / 6])
te_y = (1 - gauss) * xyz[te_edges[:, 0], 1, None] + gauss * xyz[te_edges[:, 1], 1, None]
te_interval = np.minimum(np.searchsorted(stations, te_y, side="right") - 1, len(stations) - 2)
assert np.all(te_interval >= 0)
assert np.all(te_interval[:, 0] == te_interval[:, 1]), "TE integration edge crosses circulation strips"
assert np.all(xyz[te_edges, 1].min(axis=1) >= stations[te_interval[:, 0]] - 1e-10)
assert np.all(xyz[te_edges, 1].max(axis=1) <= stations[te_interval[:, 0] + 1] + 1e-10), "TE integration edge crosses circulation strips"
te_fraction = (te_y - stations[te_interval]) / np.diff(stations)[te_interval]
te_basis = (1 - te_fraction[:, :, None]) * hat_map[te_interval] + te_fraction[:, :, None] * hat_map[te_interval + 1]
incidence = np.zeros((len(te_edges), len(faces)))
incidence[np.arange(len(te_edges)), te_upper] += 1
incidence[np.arange(len(te_edges)), te_lower] -= 1
assert np.all(te_basis.sum(axis=(0, 1)) > 0), "Every circulation unknown needs actual exposed TE support"
contact_edges = cut_edges[np.array(fixture["cut_kind"]) == "smooth_wall_contact"]
contact_graph = coo_matrix((np.ones(len(contact_edges)), (contact_edges[:, 0], contact_edges[:, 1])),
                          shape=(len(xyz), len(xyz))) if len(contact_edges) else coo_matrix((len(xyz), len(xyz)))
_, contact_labels = connected_components(contact_graph, directed=False)
contact_intervals = []
for label in np.unique(contact_labels[contact_edges.ravel()]):
    nodes = np.unique(contact_edges[contact_labels[contact_edges[:, 0]] == label])
    left, right = xyz[nodes, 1].min(), xyz[nodes, 1].max()
    selected = (stations >= left - 1e-10) & (stations <= right + 1e-10)
    assert selected.sum() >= 2 and np.min(abs(stations - left)) < 1e-10 and np.min(abs(stations - right)) < 1e-10, "Contact endpoints must be explicit circulation stations"
    assert np.all(station_dof[selected] == station_dof[selected][0]) and station_dof[selected][0] >= 0, "Smooth contact must have constant unknown jump"
    contact_intervals.append([float(left), float(right)])

wake_node_ids = np.unique(wake_faces)
wall_node_ids = np.unique(faces)
wake_motion_x = np.zeros(len(xyz))
wake_hinge_x = None
if args.wake_mode != "fixed":
    assert abs(args.alpha) < 80, "Prescribed x-monotone wake requires |alpha| < 80 degrees"
    if args.wake_mode == "freestream":
        assert not len(contact_edges), "Full freestream wake with body contact requires a new CAD fluid cut; use fixed or far-freestream"
        root_nodes = np.unique(te_edges)
        root_nodes = root_nodes[np.argsort(xyz[root_nodes, 1])]
        root_x = np.interp(xyz[:, 1], xyz[root_nodes, 1], xyz[root_nodes, 0])
        wake_motion_x[wake_node_ids] = xyz[wake_node_ids, 0] - root_x[wake_node_ids]
        assert np.min(wake_motion_x) > -1e-10
    else:
        end_x = xyz[wake_node_ids, 0].max()
        wake_hinge_x = float(xyz[wake_node_ids, 0][xyz[wake_node_ids, 0] < end_x - 1e-10].max())
        assert wake_hinge_x > xyz[wall_node_ids, 0].max(), "Far-wake hinge must be beyond the entire wall"
        wake_motion_x[wake_node_ids] = np.maximum(xyz[wake_node_ids, 0] - wake_hinge_x, 0.)
    assert np.max(abs(wake_motion_x[wall_node_ids])) < 1e-10, "Wake motion would move wall/contact nodes"
    wake_motion_x[wall_node_ids] = 0.
if args.wake_offset is not None:
    offsets = np.load(args.wake_offset, allow_pickle=False)
    wake_offset = np.asarray(offsets["offset"])
    assert wake_offset.shape == xyz.shape and np.isfinite(wake_offset).all()
    assert np.max(abs(wake_offset[wall_node_ids])) == 0., "Preserve wall and contact nodes exactly"
    assert np.max(abs(wake_offset[np.setdiff1d(np.arange(len(xyz)), wake_node_ids)]), initial=0.) == 0.
    (args.out / "wake_offset.npz").write_bytes(args.wake_offset.read_bytes())
else:
    wake_offset = np.zeros_like(xyz)
if args.wake_mode != "fixed" or args.wake_offset is not None:
    moved = xyz.copy()
    moved[:, 1] *= args.span
    moved[:, 2] *= 1 + args.bulge * np.exp(-(xyz[:, 1] / .5)**2)
    moved[:, 2] += np.tan(np.deg2rad(args.alpha)) * wake_motion_x
    moved += wake_offset
    moved_wake_geometry = check_wake_geometry(moved, faces, wake_faces, require_positive_z=args.wake_offset is None)
else:
    moved_wake_geometry = None

# Adapt only the discrete stencil, using a dimensionless geometric rank budget.
# No potential, pressure, or load enters this decision.
adjacency = [set() for _ in faces]
for a, b in pairs:
    adjacency[a].add(int(b))
    adjacency[b].add(int(a))
for a, b in cut_pairs:
    adjacency[a].remove(int(b))
    adjacency[b].remove(int(a))
axis_l = triangles[:, 1] - triangles[:, 0]
axis_l /= np.linalg.norm(axis_l, axis=1)[:, None]
axis_m = np.cross(normals, axis_l)
stencils, ring_counts, ratios = [], [], []
for i in range(len(faces)):
    reached, frontier = {i}, {i}
    for ring in range(1, 7):
        frontier = set().union(*(adjacency[j] for j in frontier)) - reached
        reached.update(frontier)
        neighbors = sorted(reached - {i})
        delta = centers[neighbors] - centers[i]
        local = np.column_stack((delta @ axis_l[i], delta @ axis_m[i]))
        singular = np.linalg.svd(local, compute_uv=False)
        ratio = singular[-1] / singular[0] if len(singular) == 2 and singular[0] > 0 else 0
        if ring >= 2 and ratio >= 1e-6:
            break
    assert ratio >= 1e-6, f"Unresolved tangent stencil at face {i}; no regularization applied"
    stencils.append(neighbors)
    ring_counts.append(ring)
    ratios.append(float(ratio))
width = max(map(len, stencils))
neighbor_ids = np.array([row + [i] * (width - len(row)) for i, row in enumerate(stencils)])
validation = dict(status="input_accepted", wall_components=int(components), wall_panels=len(faces),
                  wake_panels=len(wake_faces), actual_cut_edges=len(cut_edges),
                  actual_te_edges=len(te_edges), contact_intervals_normalized=contact_intervals,
                  wake_centroid_max_abs_winding=winding_max,
                  wake_boundary_loops=boundary_loops,
                  wake_outflow_edges=int(on_outflow.sum()), wake_tip_edges=int(on_tip.sum()),
                  wall_plane_slice_segments=len(slice_segments),
                  wall_slice_wake_interior_intersections=0,
                  plane_tolerance_normalized=plane_tolerance,
                  intersection_barycentric_tolerance=barycentric_tolerance,
                  base_tangent_minimum_singular_ratio=min(ratios),
                  prescribed_wake_mode=args.wake_mode, moved_wake_geometry=moved_wake_geometry,
                  limitations="Planar input cut; actual moved triangle wake checked separately. No wall self-intersection or complete wake overlap/touch certificate")
(args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
if args.validate_only:
    print(json.dumps(validation, indent=2), flush=True)
    sys.exit(0)
seam_faces = np.asarray(fixture["seam_faces"], dtype=int)
body_mask, wing_mask = tags == 20, tags != 20
source_hashes = {}
for path in (Path(__file__), Path(__file__).with_name("kernel.py"), Path(__file__).with_name("build_surface.py"),
             Path(__file__).with_name("wake_geometry.py")):
    content = path.read_bytes()
    source_hashes[path.name] = hashlib.sha256(content).hexdigest()
    (args.out / (path.name + ".snapshot")).write_bytes(content)
(args.out / "surface_input.json").write_bytes(args.surface.read_bytes())
stage = "assembly_and_solve"


def solve_model(parameters):
    alpha, bulge, span = parameters
    coordinates = jnp.asarray(xyz)
    coordinates = coordinates.at[:, 1].set(coordinates[:, 1] * span)
    coordinates = coordinates.at[:, 2].set(coordinates[:, 2] * (1 + bulge * jnp.exp(-(xyz[:, 1] / .5)**2)))
    tri = coordinates[faces]
    av = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
    areas = jnp.linalg.norm(av, axis=1)
    normal = av / areas[:, None]
    center = tri.mean(axis=1)
    angle = alpha * jnp.pi / 180
    flow = jnp.array([jnp.cos(angle), 0., jnp.sin(angle)])
    if args.wake_mode != "fixed":
        coordinates = coordinates.at[:, 2].add(jnp.tan(angle) * wake_motion_x)
    coordinates = coordinates + jnp.asarray(wake_offset)
    sigma = -normal @ flow
    single, doublet = triangle_potential(center, tri, interior_diagonal=True)
    wake = coordinates[wake_faces]
    wake_av = jnp.cross(wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 0])
    wake_area2 = jnp.linalg.norm(wake_av, axis=1)
    wake_normal = wake_av / wake_area2[:, None]
    material_y = jnp.asarray(xyz[wake_faces, 1]) * span
    wake_direction = ((material_y[:, 1] - material_y[:, 0])[:, None] * jnp.cross(wake[:, 2] - wake[:, 0], wake_normal)
                      + (material_y[:, 2] - material_y[:, 0])[:, None] * jnp.cross(wake_normal, wake[:, 1] - wake[:, 0])) / wake_area2[:, None]
    wake_doublet, wake_linear_moment = triangle_potential(center, wake, source=False, linear_direction=wake_direction)
    wake_operator = wake_doublet @ wake_basis + wake_linear_moment @ (wake_slope / span)
    te_length = jnp.linalg.norm(coordinates[te_edges[:, 1]] - coordinates[te_edges[:, 0]], axis=1)
    weights = jnp.broadcast_to(te_length[:, None] / 2, te_basis.shape[:2])
    jump_mass = jnp.einsum("eq,eqi,eqj->ij", weights, te_basis, te_basis)
    tested_jump = jnp.einsum("eq,eqi->ei", weights, te_basis).T @ incidence
    measure = jnp.einsum("eq,eqi->i", weights, te_basis)
    jump_mass, tested_jump = jump_mass / measure[:, None], tested_jump / measure[:, None]
    matrix = jnp.block([[doublet, wake_operator], [-tested_jump, jump_mass]])
    rhs = jnp.concatenate((single @ sigma, jnp.zeros(gamma_count)))
    solution = jnp.linalg.solve(matrix, rhs)
    mu, gamma = solution[:len(faces)], solution[len(faces):]
    l = tri[:, 1] - tri[:, 0]
    l /= jnp.linalg.norm(l, axis=1)[:, None]
    m = jnp.cross(normal, l)
    delta = center[neighbor_ids] - center[:, None]
    dl, dm = jnp.sum(delta * l[:, None], axis=2), jnp.sum(delta * m[:, None], axis=2)
    dmu = mu[neighbor_ids] - mu[:, None]
    ll, mm, lm = jnp.sum(dl * dl, axis=1), jnp.sum(dm * dm, axis=1), jnp.sum(dl * dm, axis=1)
    bl, bm = jnp.sum(dl * dmu, axis=1), jnp.sum(dm * dmu, axis=1)
    determinant = ll * mm - lm * lm
    gl, gm = (bl * mm - bm * lm) / determinant, (bm * ll - bl * lm) / determinant
    surface_velocity = (l @ flow + gl)[:, None] * l + (m @ flow + gm)[:, None] * m
    cp = 1 - jnp.sum(surface_velocity**2, axis=1)
    forces = -cp[:, None] * av
    moments = jnp.cross(center - reference, forces)
    force, moment = forces.sum(axis=0), moments.sum(axis=0)
    lift_axis = jnp.array([-jnp.sin(angle), 0., jnp.cos(angle)])
    objectives = jnp.array([force @ lift_axis / sref, forces[wing_mask, 2].sum() * chord**2,
                            forces[body_mask, 2].sum() * chord**2,
                            forces[seam_faces, 2].sum() * chord**2, moment[1] / sref])
    ratio_squared = (ll + mm - jnp.sqrt((ll - mm)**2 + 4 * lm**2)) / (ll + mm + jnp.sqrt((ll - mm)**2 + 4 * lm**2))
    return dict(objectives=objectives, coordinates=coordinates, panel_center=center, panel_normal=normal,
                panel_area=areas, Cp_raw=cp, panel_forces=forces, panel_moments=moments,
                mu=mu, sigma=sigma, gamma=gamma, mu_wake=wake_basis @ gamma,
                matrix=matrix, rhs=rhs, panel_x_dir=l, panel_y_dir=m, ql=gl, qm=gm,
                tangent_determinant=determinant, tangent_singular_ratio=jnp.sqrt(ratio_squared), surface_velocity=surface_velocity,
                single=single, doublet=doublet, wake_doublet=wake_doublet,
                wake_linear_direction=wake_direction, wake_material_y=material_y,
                wake_linear_moment=wake_linear_moment, wake_operator=wake_operator,
                jump_mass=jump_mass, tested_jump=tested_jump)


parameters = np.array([args.alpha, args.bulge, args.span])
simulate = jax.jit(solve_model)
t0 = time.perf_counter()
output = jax.tree.map(np.asarray, simulate(parameters))
cold_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
output = jax.tree.map(np.asarray, simulate(parameters))
warm_seconds = time.perf_counter() - t0
output.update(faces=faces, component_ids=tags, wake_faces=wake_faces, wake_basis=wake_basis,
              wake_slope=wake_slope / args.span, neighbor_indices=neighbor_ids, te_basis=te_basis)
wake_cross_actual = np.cross(output["coordinates"][wake_faces[:, 1]] - output["coordinates"][wake_faces[:, 0]],
                             output["coordinates"][wake_faces[:, 2]] - output["coordinates"][wake_faces[:, 0]])
wake_area_actual = np.linalg.norm(wake_cross_actual, axis=1) / 2
wake_normal_actual = wake_cross_actual / (2 * wake_area_actual[:, None])
wake_flux = wake_normal_actual @ np.array([np.cos(np.deg2rad(args.alpha)), 0., np.sin(np.deg2rad(args.alpha))])
output.update(wake_normal=wake_normal_actual, wake_area=wake_area_actual,
              wake_freestream_normal_velocity=wake_flux)
np.savez_compressed(args.out / "fields.npz", **output)
assert all(np.isfinite(value).all() for value in output.values()), "Nonfinite raw fields retained"
assert np.all(output["tangent_determinant"] > 0)
assert np.min(output["tangent_singular_ratio"]) >= 1e-6, "Deformed tangent stencil is unresolved; remesh required"
matrix, rhs = output["matrix"], output["rhs"]
solution = np.r_[output["mu"], output["gamma"]]
linear_residual = float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs))
assert linear_residual < 1e-10
lu, piv = lu_factor(matrix)
rcond, info = dgecon(lu, np.linalg.norm(matrix, 1))
assert info == 0 and rcond > 0
force = output["panel_forces"].sum(axis=0)
moment = output["panel_moments"].sum(axis=0)
components_load = {}
for name, mask in (("wing", wing_mask), ("body", body_mask)):
    components_load[name] = dict(force=(output["panel_forces"][mask].sum(axis=0) * chord**2).tolist(),
                                 moment=(output["panel_moments"][mask].sum(axis=0) * chord**3).tolist())
closure = float(np.max(abs(np.sum([row["force"] for row in components_load.values()], axis=0) - force * chord**2)))
source_flux = float(np.sum(output["sigma"] * output["panel_area"]) / output["panel_area"].sum())
assert closure < 1e-10 * max(chord**2, 1) and abs(source_flux) < 1e-12
objective_names = ["CL", "wing_Fz", "body_Fz", "seam_Fz", "CM_y"]
result = dict(
    status="forward_complete", method="Interior Dirichlet source/doublet + contact-compatible spanwise wake hats + weak TE jump",
    parameters=dict(zip(["alpha_deg", "central_thickness", "span_factor"], parameters.tolist())),
    objectives=dict(zip(objective_names, output["objectives"].tolist())),
    CL=float(output["objectives"][0]), CM=(moment / sref).tolist(),
    CD_pressure=float(force @ [np.cos(np.deg2rad(args.alpha)), 0., np.sin(np.deg2rad(args.alpha))] / sref),
    components=components_load, force=(force * chord**2).tolist(), moment=(moment * chord**3).tolist(),
    cp_min=float(output["Cp_raw"].min()), cp_max=float(output["Cp_raw"].max()),
    linear_true_relative_residual=linear_residual, condition_1_estimate=float(1 / rcond),
    normalized_source_flux=source_flux, component_force_closure=closure,
    mesh=fixture["audit"], ncpu_requested=args.ncpu,
    pressure_reconstruction=dict(method="Unweighted affine tangent least squares; all actual cut edges excluded",
                                 rings_min=min(ring_counts), rings_max=max(ring_counts), max_neighbors=width,
                                 base_minimum_singular_ratio=min(ratios), regularization=None, cutoff=None),
    input_validation=validation,
    wake_model=dict(mode=args.wake_mode, hinge_x_normalized=wake_hinge_x,
                    offset_sha256=hashlib.sha256(args.wake_offset.read_bytes()).hexdigest() if args.wake_offset else None,
                    circulation_coordinate="Transported vertex material label; affine gradient on each actual triangle",
                    angle_tracks_alpha=args.wake_mode != "fixed",
                    fully_freestream_aligned=args.wake_mode == "freestream" and args.wake_offset is None,
                    freestream_normal_rms=float(np.sqrt(np.average(wake_flux**2, weights=wake_area_actual))),
                    freestream_normal_max=float(np.max(abs(wake_flux))),
                    local_flow_tangency_solved=False,
                    derivative="Prescribed motion included; supplied offsets frozen; no converged free-wake or CAD/contact derivative"),
    geometry_robustness_accepted=True, physical_accuracy_accepted=False,
    references=dict(area=sref * chord**2, chord=chord, moment=fixture["moment_reference"],
                    origin=fixture["origin"], density=2., speed=1., dynamic_pressure=1.),
    timings=dict(compile_and_first_seconds=cold_seconds, warm_forward_seconds=warm_seconds),
    source_sha256=source_hashes, surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
                     threadpools=threadpool_info()),
    scope="Conforming planar input cut, transported material wake jump, fixed topology deformations; no CAD or converged free-wake derivative",
)
(args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, allow_nan=False), flush=True)
if args.derivatives:
    stage = "derivatives"
    t0 = time.perf_counter()
    derivative = jax.jit(jax.jacfwd(lambda p: solve_model(p)["objectives"]))
    ad = np.asarray(derivative(parameters))
    result["timings"]["derivative_compile_and_first_seconds"] = time.perf_counter() - t0
    assert np.isfinite(ad).all()
    comparisons = []
    for column, name in enumerate(result["parameters"]):
        for step in (1e-3, 1e-4, 1e-5):
            plus, minus = parameters.copy(), parameters.copy()
            plus[column] += step
            minus[column] -= step
            fd = (np.asarray(simulate(plus)["objectives"]) - np.asarray(simulate(minus)["objectives"])) / (2 * step)
            error = abs(ad[:, column] - fd)
            comparisons.append(dict(parameter=name, step=step, ad=ad[:, column].tolist(), fd=fd.tolist(),
                                    max_abs_error=float(error.max()),
                                    max_relative_error=float(np.max(error / np.maximum(abs(ad[:, column]), 1e-10))),
                                    passed=bool(np.all(error <= 2e-7 + 2e-4 * abs(ad[:, column])))))
    result["derivatives"] = comparisons
    result["derivatives_passed"] = all(row["passed"] for row in comparisons)
    result["status"] = "loop_complete" if result["derivatives_passed"] else "derivative_mismatch"
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert result["derivatives_passed"]
    print(json.dumps(dict(derivatives_passed=True, scalar_comparisons=45, ad=ad.tolist()), indent=2), flush=True)
print("Completed:", args.out, flush=True)
