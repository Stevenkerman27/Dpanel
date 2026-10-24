import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


sys.path.insert(0, "D:/3D/Projects/panel/audit/runtime/python")
import gmsh


def tetra10_jacobian_lattice(points, tets, lattice_order):
    lattice_points = []
    for l0 in range(lattice_order + 1):
        for l1 in range(lattice_order - l0 + 1):
            for l2 in range(lattice_order - l0 - l1 + 1):
                l3 = lattice_order - l0 - l1 - l2
                lattice_points.append(
                    np.array([l0, l1, l2, l3], dtype=np.float64) / lattice_order
                )
    lattice_points = np.asarray(lattice_points)
    d_lambda_d_xi = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    d_shape_d_lambda = np.zeros((len(lattice_points), 10, 4), dtype=np.float64)
    for vertex in range(4):
        d_shape_d_lambda[:, vertex, vertex] = (
            4.0 * lattice_points[:, vertex] - 1.0
        )
    local_edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    for edge_index, (first, second) in enumerate(local_edges):
        shape_index = 4 + edge_index
        d_shape_d_lambda[:, shape_index, first] = (
            4.0 * lattice_points[:, second]
        )
        d_shape_d_lambda[:, shape_index, second] = (
            4.0 * lattice_points[:, first]
        )
    d_shape_d_xi = np.einsum(
        "qnk,ki->qni", d_shape_d_lambda, d_lambda_d_xi
    )

    minimum_jacobian = np.inf
    maximum_jacobian = -np.inf
    nonpositive_samples = 0
    for start in range(0, len(tets), 256):
        stop = min(start + 256, len(tets))
        element_points = points[tets[start:stop]]
        jacobians = np.einsum("enp,qni->eqpi", element_points, d_shape_d_xi)
        determinants = np.linalg.det(jacobians)
        minimum_jacobian = min(minimum_jacobian, float(determinants.min()))
        maximum_jacobian = max(maximum_jacobian, float(determinants.max()))
        nonpositive_samples += int(np.count_nonzero(determinants <= 0.0))
    return len(lattice_points), minimum_jacobian, maximum_jacobian, nonpositive_samples


def orient_tetra10(points, tets):
    vertex_points = points[tets[:, :4]]
    determinants = np.linalg.det(
        np.stack(
            (
                vertex_points[:, 1] - vertex_points[:, 0],
                vertex_points[:, 2] - vertex_points[:, 0],
                vertex_points[:, 3] - vertex_points[:, 0],
            ),
            axis=1,
        )
    )
    negative = determinants < 0
    permutation = np.array([0, 2, 1, 3, 6, 5, 4, 7, 9, 8])
    tets[negative] = tets[negative][:, permutation]
    return np.abs(determinants) / 6.0


parser = argparse.ArgumentParser(
    description="Generate a CAD-conforming curved P2 tetrahedral fluid mesh."
)
parser.add_argument(
    "--shape", choices=("sphere", "ellipsoid", "wingbody"), required=True
)
parser.add_argument("--size", type=float, required=True, help="P1 target mesh size")
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--farfield-factor", type=float, default=5.0)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--lattice-order", type=int, default=12)
parser.add_argument(
    "--background-view",
    type=Path,
    help="NPZ with points, corner tets, and vertex_size for a Gmsh PostView field",
)
args = parser.parse_args()

if args.size <= 0 or args.radius <= 0 or args.farfield_factor <= 1:
    parser.error("size/radius must be positive and farfield-factor must exceed 1")
if args.lattice_order < 12:
    parser.error("Acceptance requires --lattice-order >= 12")
background_data = None
if args.background_view is not None:
    with np.load(args.background_view) as archive:
        background_data = {key: archive[key] for key in archive.files}
    if not all(key in background_data for key in ("points", "tets", "vertex_size")):
        parser.error("background view NPZ requires points, tets, and vertex_size arrays")
    background_points = np.asarray(background_data["points"], dtype=np.float64)
    background_tets = np.asarray(background_data["tets"], dtype=np.int64)
    background_size = np.asarray(background_data["vertex_size"], dtype=np.float64)
    if (
        background_points.ndim != 2
        or background_points.shape[1] != 3
        or background_tets.ndim != 2
        or background_tets.shape[1] not in (4, 10)
        or background_size.shape != (len(background_points),)
        or not np.all(np.isfinite(background_points))
        or not np.all(np.isfinite(background_size))
        or np.any(background_tets < 0)
        or np.any(background_tets[:, :4] >= len(background_points))
        or np.any(background_size <= 0.0)
    ):
        parser.error("background view dimensions, indices, or sizes are invalid")
output_dir = args.out.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    parser.error(f"Refusing to overwrite non-empty output directory: {output_dir}")

started = time.time()
print("stage: initialize gmsh", flush=True)
gmsh_threads = 1 if background_data is not None else 8
radius = args.radius
farfield_radius = args.farfield_factor * radius
body_axes = (
    np.array([radius, radius, radius], dtype=np.float64)
    if args.shape == "sphere"
    else (
        np.array([3.0 * radius, radius, radius], dtype=np.float64)
        if args.shape == "ellipsoid"
        else np.array([2.0 * radius, 0.35 * radius, 0.35 * radius], dtype=np.float64)
    )
)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("General.NumThreads", gmsh_threads)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", gmsh_threads)
gmsh.option.setNumber("Mesh.MaxNumThreads3D", gmsh_threads)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.size)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.size)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.HighOrderFixBoundaryNodes", 1)
gmsh.model.add("cad_curved_fluid_domain")
occ = gmsh.model.occ
farfield = occ.addSphere(0, 0, 0, farfield_radius)
if args.shape in ("sphere", "ellipsoid"):
    body = occ.addSphere(0, 0, 0, radius)
    if args.shape == "ellipsoid":
        occ.dilate([(3, body)], 0, 0, 0, *body_axes)
        split_points = [
            occ.addPoint(0, -4 * radius, -4 * radius),
            occ.addPoint(0, 4 * radius, -4 * radius),
            occ.addPoint(0, 4 * radius, 4 * radius),
            occ.addPoint(0, -4 * radius, 4 * radius),
        ]
        split_lines = [
            occ.addLine(split_points[i], split_points[(i + 1) % 4])
            for i in range(4)
        ]
        split_wire = occ.addWire(split_lines)
        split_face = occ.addPlaneSurface([split_wire])
        split_entities, _ = occ.fragment(
            [(3, body)], [(2, split_face)], removeObject=False, removeTool=True
        )
        body_volumes = [entity for entity in split_entities if entity[0] == 3]
    else:
        body_volumes = [(3, body)]
else:
    fuselage = occ.addSphere(0, 0, 0, radius)
    occ.dilate([(3, fuselage)], 0, 0, 0, *body_axes)
    chord = 1.2 * radius
    span = 4.0 * radius
    airfoil_coordinates = (
        (1.0, 0.0),
        (0.75, 0.0316),
        (0.5, 0.0529),
        (0.25, 0.0594),
        (0.0, 0.0),
        (0.25, -0.0594),
        (0.5, -0.0529),
        (0.75, -0.0316),
    )
    airfoil_points = [
        occ.addPoint(x * chord - 0.5 * chord, -0.5 * span, z * chord)
        for x, z in airfoil_coordinates
    ]
    airfoil_lines = [
        occ.addLine(airfoil_points[index], airfoil_points[index + 1])
        for index in range(len(airfoil_points) - 1)
    ]
    airfoil_lines.append(occ.addLine(airfoil_points[-1], airfoil_points[0]))
    airfoil_wire = occ.addWire(airfoil_lines)
    airfoil_face = occ.addPlaneSurface([airfoil_wire])
    wing_entities = occ.extrude([(2, airfoil_face)], 0, span, 0)
    wing_volume = [entity for entity in wing_entities if entity[0] == 3][0]
    fused_entities, _ = occ.fuse(
        [(3, fuselage)], [wing_volume], removeObject=True, removeTool=True
    )
    body_volumes = [entity for entity in fused_entities if entity[0] == 3]
    if len(body_volumes) != 1:
        gmsh.finalize()
        raise RuntimeError(f"Expected one fused wingbody solid, found {len(body_volumes)}")

fluid_entities, _ = occ.cut(
    [(3, farfield)], body_volumes, removeObject=True, removeTool=True
)
fluid_volumes = [entity for entity in fluid_entities if entity[0] == 3]
if len(fluid_volumes) != 1:
    gmsh.finalize()
    raise RuntimeError(f"Expected one fluid volume, found {len(fluid_volumes)}")
occ.synchronize()
print("stage: OCC geometry synchronized", flush=True)

surface_entities = gmsh.model.getBoundary(
    fluid_volumes, combined=True, oriented=False, recursive=False
)
body_surfaces = []
wing_surfaces = []
farfield_surfaces = []
surface_types = {}
for dimension, surface_tag in surface_entities:
    if dimension != 2:
        continue
    surface_type = gmsh.model.getType(2, surface_tag)
    surface_types[str(surface_tag)] = surface_type
    if surface_type == "Sphere":
        bounds = np.asarray(gmsh.model.getBoundingBox(2, surface_tag), dtype=float)
        bounding_radius = 0.5 * float(np.max(bounds[3:] - bounds[:3]))
        if np.isclose(bounding_radius, farfield_radius, rtol=1.0e-7, atol=1.0e-9):
            farfield_surfaces.append(surface_tag)
        elif args.shape == "sphere" and np.isclose(
            bounding_radius, radius, rtol=1.0e-7, atol=1.0e-9
        ):
            body_surfaces.append(surface_tag)
        else:
            gmsh.finalize()
            raise RuntimeError(
                f"Unclassified spherical CAD boundary surface {surface_tag}: "
                f"bounding radius {bounding_radius}"
            )
    elif args.shape == "wingbody" and surface_type == "BSpline surface":
        body_surfaces.append(surface_tag)
    elif args.shape == "wingbody" and surface_type == "Plane":
        wing_surfaces.append(surface_tag)
    elif args.shape != "wingbody" and surface_type in (
        "BSpline surface",
        "Surface of Revolution",
    ):
        body_surfaces.append(surface_tag)
    else:
        gmsh.finalize()
        raise RuntimeError(
            f"Unclassified CAD boundary surface {surface_tag}: {surface_type}"
        )
if not body_surfaces or not farfield_surfaces or (
    args.shape == "wingbody" and not wing_surfaces
):
    gmsh.finalize()
    raise RuntimeError(
        f"Missing CAD boundary surfaces: body={body_surfaces}, "
        f"wing={wing_surfaces}, farfield={farfield_surfaces}"
    )
body_tag = 20 if args.shape == "wingbody" else 10
gmsh.model.addPhysicalGroup(2, body_surfaces, body_tag)
gmsh.model.setPhysicalName(
    2, body_tag, "fuselage" if args.shape == "wingbody" else "body"
)
if args.shape == "wingbody":
    gmsh.model.addPhysicalGroup(2, wing_surfaces, 30)
    gmsh.model.setPhysicalName(2, 30, "wing")
gmsh.model.addPhysicalGroup(2, farfield_surfaces, 100)
gmsh.model.setPhysicalName(2, 100, "farfield")
gmsh.model.addPhysicalGroup(3, [fluid_volumes[0][1]], 1)
gmsh.model.setPhysicalName(3, 1, "fluid")

background_view_tag = None
if background_data is not None:
    print("stage: registering PostView tetra data", flush=True)
    background_view_tag = gmsh.view.add("Dorfler nodal target size")
    view_data = []
    for tet in background_tets:
        xyz = background_points[tet[:4]]
        view_data.extend(xyz[:, 0].tolist())
        view_data.extend(xyz[:, 1].tolist())
        view_data.extend(xyz[:, 2].tolist())
        view_data.extend(background_size[tet[:4]].tolist())
    gmsh.view.addListData(
        background_view_tag, "SS", len(background_tets), view_data
    )
    background_field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(background_field, "ViewTag", background_view_tag)
    gmsh.model.mesh.field.setAsBackgroundMesh(background_field)
    print("stage: PostView field registered", flush=True)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45 * args.size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.size)

print("stage: generate 3D mesh", flush=True)
gmsh.model.mesh.generate(3)
print("stage: 3D mesh generated", flush=True)
gmsh.model.mesh.optimize("Netgen", niter=2)
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder", niter=4)
gmsh.model.mesh.optimize("HighOrderElastic", niter=2)

node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(node_coordinates, dtype=np.float64).reshape(-1, 3)
node_index = {int(tag): index for index, tag in enumerate(node_tags)}
tetra_type = gmsh.model.mesh.getElementType("Tetrahedron", 2)
tetra_tags, tetra_node_tags = gmsh.model.mesh.getElementsByType(tetra_type)
tets = np.asarray(
    [
        [node_index[int(tag)] for tag in row]
        for row in tetra_node_tags.reshape(-1, 10)
    ],
    dtype=np.int64,
)
# Gmsh orders the final two tetra10 edge nodes as 23,13; the contract is 13,23.
tets[:, [8, 9]] = tets[:, [9, 8]]
if len(tets) == 0:
    gmsh.finalize()
    raise RuntimeError("Gmsh did not create quadratic tetrahedra")
tetra_volumes = orient_tetra10(points, tets)

boundary_triangle_rows = []
boundary_tag_rows = []
surface_groups = [(body_tag, body_surfaces)]
if args.shape == "wingbody":
    surface_groups.append((30, wing_surfaces))
surface_groups.append((100, farfield_surfaces))
for tag, surfaces in surface_groups:
    for surface_tag in surfaces:
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(
            2, surface_tag
        )
        for element_type, surface_element_tags, surface_nodes in zip(
            element_types, element_tags, element_node_tags
        ):
            if element_type != gmsh.model.mesh.getElementType("Triangle", 2):
                continue
            triangles = np.asarray(surface_nodes, dtype=np.int64).reshape(-1, 6)
            for triangle in triangles:
                boundary_triangle_rows.append(
                    [node_index[int(node_tag)] for node_tag in triangle]
                )
                boundary_tag_rows.append(tag)
boundary_triangles = np.asarray(boundary_triangle_rows, dtype=np.int64)
boundary_tags = np.asarray(boundary_tag_rows, dtype=np.int32)
if len(boundary_triangles) == 0:
    gmsh.finalize()
    raise RuntimeError("Gmsh did not create quadratic boundary triangles")

face_map = {}
for tet_index, tet in enumerate(tets):
    for face in (
        tet[[0, 1, 2]],
        tet[[0, 1, 3]],
        tet[[0, 2, 3]],
        tet[[1, 2, 3]],
    ):
        face_map[tuple(sorted(int(node) for node in face))] = tet_index
boundary_tet = np.asarray(
    [
        face_map[tuple(sorted(int(node) for node in triangle[:3]))]
        for triangle in boundary_triangles
    ],
    dtype=np.int64,
)
point_count_before_compaction = len(points)
used_points = np.unique(tets)
if not np.all(np.isin(np.unique(boundary_triangles), used_points)):
    gmsh.finalize()
    raise RuntimeError("A boundary triangle references a node unused by every tetra10")
old_to_new = np.full(point_count_before_compaction, -1, dtype=np.int64)
old_to_new[used_points] = np.arange(len(used_points))
points = points[used_points]
tets = old_to_new[tets]
boundary_triangles = old_to_new[boundary_triangles]
removed_point_count = point_count_before_compaction - len(points)
adjacent_vertices = tets[boundary_tet, :4]
face_membership = np.any(
    adjacent_vertices[:, :, None] == boundary_triangles[:, None, :3], axis=2
)
opposite_index = np.argmin(face_membership, axis=1)
opposite_nodes = adjacent_vertices[np.arange(len(boundary_tet)), opposite_index]
triangle_normals = np.cross(
    points[boundary_triangles[:, 1]] - points[boundary_triangles[:, 0]],
    points[boundary_triangles[:, 2]] - points[boundary_triangles[:, 0]],
)
inward = np.einsum(
    "ij,ij->i",
    triangle_normals,
    points[opposite_nodes] - points[boundary_triangles[:, 0]],
) > 0
flip_order = np.array([0, 2, 1, 5, 4, 3])
boundary_triangles[inward] = boundary_triangles[inward][:, flip_order]

lattice_samples, minimum_jacobian, maximum_jacobian, nonpositive_samples = (
    tetra10_jacobian_lattice(points, tets, args.lattice_order)
)

body_nodes = np.unique(boundary_triangles[boundary_tags == body_tag])
farfield_nodes = np.unique(boundary_triangles[boundary_tags == 100])
if args.shape == "wingbody":
    body_cad_residual = 0.0
    wing_cad_residual = 0.0
    for tag, surfaces in ((20, body_surfaces), (30, wing_surfaces)):
        group_nodes = np.unique(boundary_triangles[boundary_tags == tag])
        maximum_distance = 0.0
        for node in group_nodes:
            distances = [
                np.linalg.norm(
                    np.asarray(
                        gmsh.model.getClosestPoint(
                            2, surface_tag, points[int(node)].tolist()
                        )[0]
                    ).reshape(-1, 3)[0]
                    - points[int(node)]
                )
                for surface_tag in surfaces
            ]
            maximum_distance = max(maximum_distance, min(distances))
        if tag == 20:
            body_cad_residual = maximum_distance
        else:
            wing_cad_residual = maximum_distance
else:
    body_level = np.sum((points[body_nodes] / body_axes) ** 2, axis=1)
    body_cad_residual = float(np.max(np.abs(body_level - 1.0)))
farfield_level = np.linalg.norm(points[farfield_nodes], axis=1) / farfield_radius
farfield_cad_residual = float(np.max(np.abs(farfield_level - 1.0)))
area_vectors = {}
for tag, _ in surface_groups:
    selected = boundary_triangles[boundary_tags == tag]
    area_vectors[str(tag)] = np.sum(
        np.cross(
            points[selected[:, 1]] - points[selected[:, 0]],
            points[selected[:, 2]] - points[selected[:, 0]],
        )
        / 2.0,
        axis=0,
    ).tolist()
tetra_quality = np.asarray(
    gmsh.model.mesh.getElementQualities(tetra_tags, "minSICN"), dtype=np.float64
)
gmsh.finalize()

mesh_arrays = (
    points,
    tets,
    boundary_triangles,
    boundary_tags,
    boundary_tet,
)
array_hash = hashlib.sha256()
for array in mesh_arrays:
    array_hash.update(np.ascontiguousarray(array).tobytes())
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
mesh_metadata = {
    "status": "failed",
    "shape": args.shape,
    "reference_radius": radius,
    "shape_parameters": (
        {"body_radius": radius}
        if args.shape == "sphere"
        else (
            {"semiaxes": body_axes.tolist()}
            if args.shape == "ellipsoid"
            else {
                "fuselage_ellipsoid_semiaxes": body_axes.tolist(),
                "wing_chord": 1.2 * radius,
                "wing_span": 4.0 * radius,
                "airfoil": "Eight-point symmetric airfoil-like polygon; not exact NACA 0012.",
            }
        )
    ),
    "coordinate_axes": {
        "body_axis": "x",
        "model_coordinates": "x,y,z",
        "wing_chord_axis": "x" if args.shape == "wingbody" else None,
        "wing_span_axis": "y" if args.shape == "wingbody" else None,
        "wing_thickness_axis": "z" if args.shape == "wingbody" else None,
    },
    "mesh_size": args.size,
    "background_size_field": (
        {
            "source": str(args.background_view.resolve()),
            "source_sha256": hashlib.sha256(args.background_view.read_bytes()).hexdigest(),
            "gmsh_view_tag": background_view_tag,
            "gmsh_field_type": "PostView",
            "gmsh_data_type": "SS",
        }
        if args.background_view is not None
        else None
    ),
    "farfield_radius": farfield_radius,
    "geometry_source": (
        "OpenCASCADE farfield sphere minus fused ellipsoid/wing solid"
        if args.shape == "wingbody"
        else "OpenCASCADE analytic sphere/ellipsoid Boolean subtraction"
    ),
    "gmsh_version": gmsh.__version__,
    "threads": gmsh_threads,
    "element_order": 2,
    "element_types": {
        "volume": "tetra10",
        "boundary": "triangle6",
    },
    "tetra10_node_order": "vertices 0,1,2,3 then edges 01,12,20,03,13,23",
    "gmsh_tetra10_conversion": "Swapped Gmsh edge-node slots 8 and 9 (Gmsh 23,13 to contract 13,23).",
    "triangle6_node_order": "vertices 0,1,2 then edges 01,12,20",
    "physical_tags": (
        {"fuselage": 20, "wing": 30, "farfield": 100}
        if args.shape == "wingbody"
        else {"body": 10, "farfield": 100}
    ),
    "cad_surface_entities": {
        "body": body_surfaces,
        "wing": wing_surfaces,
        "farfield": farfield_surfaces,
        "types": surface_types,
    },
    "boundary_tag_provenance": (
        {
            "20": "Boolean-result OCC BSpline fuselage boundary faces",
            "30": "Boolean-result OCC planar wing boundary faces",
            "100": "farfield OCC sphere",
            "classification": "OCC boundary surface entity membership; no level-set threshold.",
            "airfoil_note": "Eight-point polygonal approximation, not exact NACA 0012.",
        }
        if args.shape == "wingbody"
        else None
    ),
    "counts": {
        "points": len(points),
        "unused_points_removed": removed_point_count,
        "tets": len(tets),
        "boundary_triangles": len(boundary_triangles),
        "boundary_tets": len(boundary_tet),
    },
    "quality": {
        "tet_volume_min": float(tetra_volumes.min()),
        "tet_volume_median": float(np.median(tetra_volumes)),
        "tet_volume_max": float(tetra_volumes.max()),
        "tet_min_sicn": float(tetra_quality.min()),
        "tet_median_sicn": float(np.median(tetra_quality)),
        "tet_sicn_below_0_1": int(np.count_nonzero(tetra_quality < 0.1)),
    },
    "cad_residuals": {
        "body_levelset_max_abs_error": body_cad_residual,
        "wing_to_occ_surface_max_distance": (
            wing_cad_residual if args.shape == "wingbody" else None
        ),
        "farfield_relative_radius_max_abs_error": farfield_cad_residual,
        "acceptance_tolerance": 1.0e-7,
    },
    "closed_boundary_area_vectors": area_vectors,
    "curved_jacobian_check": {
        "method": "Dense barycentric lattice sampling of the quadratic tetra10 map.",
        "lattice_order": args.lattice_order,
        "samples_per_tet": lattice_samples,
        "minimum_jacobian_determinant": minimum_jacobian,
        "maximum_jacobian_determinant": maximum_jacobian,
        "nonpositive_jacobian_samples": nonpositive_samples,
    },
    "sha256_arrays": array_hash.hexdigest(),
    "source_sha256": source_hash,
    "elapsed_seconds": time.time() - started,
}

body_area_norm = np.linalg.norm(area_vectors[str(body_tag)])
farfield_area_norm = np.linalg.norm(area_vectors["100"])
area_scale = max(1.0, float(len(boundary_triangles)))
cad_tolerance = 1.0e-7
area_tolerance = 1.0e-8 * area_scale
acceptance_failures = []
if np.any(tetra_volumes <= 0):
    acceptance_failures.append("nonpositive linear vertex tetrahedron")
if nonpositive_samples:
    acceptance_failures.append("nonpositive curved tetra10 Jacobian sample")
if body_cad_residual > cad_tolerance:
    acceptance_failures.append("body boundary nodes do not lie on analytic CAD")
if args.shape == "wingbody" and wing_cad_residual > cad_tolerance:
    acceptance_failures.append("wing boundary nodes do not lie on OCC CAD faces")
if farfield_cad_residual > cad_tolerance:
    acceptance_failures.append("farfield boundary nodes do not lie on analytic CAD")
if args.shape == "wingbody":
    joined_wall_area = np.asarray(area_vectors["20"]) + np.asarray(area_vectors["30"])
    if np.linalg.norm(joined_wall_area) > area_tolerance:
        acceptance_failures.append("combined wingbody wall area-vector sum exceeds tolerance")
elif body_area_norm > area_tolerance or farfield_area_norm > area_tolerance:
    acceptance_failures.append("closed boundary area-vector sum exceeds tolerance")

output_dir.mkdir(parents=True, exist_ok=True)
if acceptance_failures:
    mesh_metadata["status"] = "failed_acceptance"
    mesh_metadata["acceptance_failures"] = acceptance_failures
    (output_dir / "failedmeta.json").write_text(
        json.dumps(mesh_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    raise SystemExit("Curved mesh rejected: " + "; ".join(acceptance_failures))

mesh_metadata["status"] = "success"
np.savez_compressed(
    output_dir / "mesh.npz",
    points=points,
    tets=tets,
    boundary_triangles=boundary_triangles,
    boundary_tags=boundary_tags,
    boundary_tet=boundary_tet,
)
(output_dir / "mesh.json").write_text(
    json.dumps(mesh_metadata, indent=2, ensure_ascii=True), encoding="utf-8"
)
print(
    json.dumps(
        {
            "output": str(output_dir),
            "counts": mesh_metadata["counts"],
            "min_sicn": mesh_metadata["quality"]["tet_min_sicn"],
            "minimum_jacobian_determinant": minimum_jacobian,
            "cad_residuals": mesh_metadata["cad_residuals"],
        },
        ensure_ascii=True,
    )
)
