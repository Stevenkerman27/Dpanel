import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/3D/Projects/panel/audit/runtime/python")
import gmsh


def oriented_tets(points, tets):
    tetra_points = points[tets]
    determinants = np.linalg.det(
        np.stack(
            (
                tetra_points[:, 1] - tetra_points[:, 0],
                tetra_points[:, 2] - tetra_points[:, 0],
                tetra_points[:, 3] - tetra_points[:, 0],
            ),
            axis=1,
        )
    )
    negative = determinants < 0
    tets[negative, 1], tets[negative, 2] = (
        tets[negative, 2].copy(),
        tets[negative, 1].copy(),
    )
    return np.abs(determinants) / 6.0


def mesh_boundary(points, tets, shape, radius, far_radius, wingbody_axes):
    face_nodes = np.concatenate(
        (
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ),
        axis=0,
    )
    face_tets = np.tile(np.arange(len(tets)), 4)
    sorted_faces = np.sort(face_nodes, axis=1)
    _, inverse, counts = np.unique(
        sorted_faces, axis=0, return_inverse=True, return_counts=True
    )
    boundary_mask = counts[inverse] == 1
    boundary_triangles = face_nodes[boundary_mask].copy()
    boundary_tet = face_tets[boundary_mask]
    face_vertices = points[boundary_triangles]
    opposite_mask = np.any(
        tets[boundary_tet, :, None] == boundary_triangles[:, None, :], axis=2
    )
    opposite_index = np.argmin(opposite_mask, axis=1)
    opposite_nodes = tets[boundary_tet, opposite_index]
    normals = np.cross(
        points[boundary_triangles[:, 1]] - points[boundary_triangles[:, 0]],
        points[boundary_triangles[:, 2]] - points[boundary_triangles[:, 0]],
    )
    inward = np.einsum(
        "ij,ij->i", normals, points[opposite_nodes] - face_vertices[:, 0]
    ) > 0
    boundary_triangles[inward, 1], boundary_triangles[inward, 2] = (
        boundary_triangles[inward, 2].copy(),
        boundary_triangles[inward, 1].copy(),
    )
    if shape == "wingbody":
        vertex_level = np.sum(
            (face_vertices / np.asarray(wingbody_axes)) ** 2, axis=2
        )
        fuselage_face = np.all(np.abs(vertex_level - 1.0) <= 0.15, axis=1)
        body_tags = np.where(fuselage_face, 20, 30).astype(np.int32)
    else:
        body_tags = np.full(len(boundary_triangles), 10, dtype=np.int32)
    face_centers = face_vertices.mean(axis=1)
    farfield = np.linalg.norm(face_centers, axis=1) > far_radius * 0.90
    boundary_tags = np.where(farfield, 100, body_tags).astype(np.int32)
    return boundary_triangles, boundary_tags, boundary_tet


parser = argparse.ArgumentParser(
    description="Generate a tetrahedral fluid-domain mesh with Gmsh OCC."
)
parser.add_argument("--shape", choices=("sphere", "ellipsoid", "wingbody"))
parser.add_argument("--size", type=float, help="Target mesh size")
parser.add_argument("--radius", type=float, default=1.0, help="Reference radius")
parser.add_argument("--out", type=Path, required=True, help="Output directory")
parser.add_argument(
    "--rebuild-boundary",
    action="store_true",
    help="Rebuild boundary arrays and metadata from the existing mesh.npz without Gmsh.",
)
args = parser.parse_args()

if args.rebuild_boundary:
    old_npz = np.load(args.out / "mesh.npz")
    mesh_info = json.loads((args.out / "mesh.json").read_text(encoding="utf-8"))
    points = old_npz["points"]
    tets = old_npz["tets"]
    shape = mesh_info["shape"]
    radius = float(mesh_info["reference_radius"])
    far_radius = float(mesh_info["farfield_radius"])
    wingbody_axes = np.array([2.0 * radius, 0.35 * radius, 0.35 * radius])
    boundary_triangles, boundary_tags, boundary_tet = mesh_boundary(
        points, tets, shape, radius, far_radius, wingbody_axes
    )
    tetra_points = points[tets]
    determinants = np.linalg.det(
        np.stack(
            (
                tetra_points[:, 1] - tetra_points[:, 0],
                tetra_points[:, 2] - tetra_points[:, 0],
                tetra_points[:, 3] - tetra_points[:, 0],
            ),
            axis=1,
        )
    )
    volumes = np.abs(determinants) / 6.0
    edge_pairs = np.concatenate(
        (
            tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
            tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
        ),
        axis=0,
    )
    edge_lengths = np.linalg.norm(
        points[edge_pairs[:, 0]] - points[edge_pairs[:, 1]], axis=1
    )
    np.savez_compressed(
        args.out / "mesh.npz",
        points=points,
        tets=tets,
        boundary_triangles=boundary_triangles,
        boundary_tags=boundary_tags,
        boundary_tet=boundary_tet,
    )
    array_hash = hashlib.sha256()
    for array in (points, tets, boundary_triangles, boundary_tags, boundary_tet):
        array_hash.update(np.ascontiguousarray(array).tobytes())
    mesh_info["counts"] = {
        "points": len(points),
        "tets": len(tets),
        "boundary_triangles": len(boundary_triangles),
        "boundary_tets": len(boundary_tet),
    }
    mesh_info["quality"].update(
        {
            "tet_volume_min": float(volumes.min()),
            "tet_volume_median": float(np.median(volumes)),
            "tet_volume_max": float(volumes.max()),
            "tet_volume_nonpositive": int(np.count_nonzero(determinants <= 0)),
            "edge_length_min": float(edge_lengths.min()),
            "edge_length_median": float(np.median(edge_lengths)),
            "edge_length_max": float(edge_lengths.max()),
            "boundary_tag_counts": {
                str(tag): int(np.count_nonzero(boundary_tags == tag))
                for tag in np.unique(boundary_tags)
            },
        }
    )
    mesh_info["wingbody_tagging"] = (
        "All three triangle vertices must satisfy abs(ellipsoid_levelset - 1) <= 0.15; "
        "otherwise tag as wing. The two patches meet at their classification boundary."
        if shape == "wingbody"
        else None
    )
    if shape == "wingbody":
        mesh_info["shape_parameters"]["airfoil"] = (
            "eight-point symmetric airfoil-like polygon; not exact NACA 0012"
        )
    mesh_info["coordinate_axes"] = {
        "fuselage_axis": "x",
        "wing_chord_axis": "x",
        "wing_thickness_axis": "y",
        "wing_span_axis": "z",
        "fem_alpha_xz_alignment": (
            "Not aligned: wing currently lies in x-y with z span; no geometry transform applied."
            if shape == "wingbody"
            else "Not applicable."
        ),
    }
    mesh_info["sha256_arrays"] = array_hash.hexdigest()
    mesh_info["source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    mesh_info["metadata_revision"] = 2
    mesh_info["revision_reason"] = (
        "Fixed face-to-tet ordering: face groups are concatenated by local face, "
        "so boundary_tet is tiled, not repeated. Rebuilt boundary arrays from "
        "the preserved points/tets without rerunning Gmsh."
    )
    (args.out / "mesh.json").write_text(
        json.dumps(mesh_info, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(mesh_info["counts"], ensure_ascii=True))
    raise SystemExit(0)

if args.shape is None or args.size is None:
    parser.error("--shape and --size are required unless --rebuild-boundary is used")
if args.size <= 0 or args.radius <= 0:
    parser.error("--size and --radius must be positive")

started = time.time()
args.out.mkdir(parents=True, exist_ok=True)
far_radius = 5.0 * args.radius
wingbody_axes = np.array([2.0 * args.radius, 0.35 * args.radius, 0.35 * args.radius])
shape_parameters = {
    "sphere": {"body_radius": args.radius},
    "ellipsoid": {"semiaxes": [3.0 * args.radius, args.radius, args.radius]},
    "wingbody": {
        "fuselage_ellipsoid_semiaxes": wingbody_axes.tolist(),
        "wing_chord": 1.2 * args.radius,
        "wing_span": 4.0 * args.radius,
        "airfoil": "eight-point symmetric airfoil-like polygon, linear OCC extrusion; not exact NACA 0012",
    },
}

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", 8)
gmsh.option.setNumber("Mesh.MaxNumThreads3D", 8)
gmsh.option.setNumber("Mesh.Algorithm", 5)
gmsh.option.setNumber("Mesh.MeshSizeMin", args.size)
gmsh.option.setNumber("Mesh.MeshSizeMax", args.size)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.model.add("panel_fluid_domain")
occ = gmsh.model.occ
farfield = occ.addSphere(0, 0, 0, far_radius)

if args.shape == "sphere":
    body = occ.addSphere(0, 0, 0, args.radius)
    body_volumes = [(3, body)]
elif args.shape == "ellipsoid":
    body = occ.addSphere(0, 0, 0, args.radius)
    occ.dilate([(3, body)], 0, 0, 0, 3, 1, 1)
    split_points = [
        occ.addPoint(0, -4 * args.radius, -4 * args.radius),
        occ.addPoint(0, 4 * args.radius, -4 * args.radius),
        occ.addPoint(0, 4 * args.radius, 4 * args.radius),
        occ.addPoint(0, -4 * args.radius, 4 * args.radius),
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
    occ.synchronize()
else:
    fuselage = occ.addSphere(0, 0, 0, args.radius)
    occ.dilate([(3, fuselage)], 0, 0, 0, *wingbody_axes)
    chord = 1.2 * args.radius
    span = 4.0 * args.radius
    points_2d = [
        (1.0, 0.0),
        (0.75, 0.0316),
        (0.5, 0.0529),
        (0.25, 0.0594),
        (0.0, 0.0),
        (0.25, -0.0594),
        (0.5, -0.0529),
        (0.75, -0.0316),
    ]
    airfoil_points = [
        occ.addPoint(x * chord - 0.5 * chord, y * chord, -0.5 * span)
        for x, y in points_2d
    ]
    airfoil_lines = [
        occ.addLine(airfoil_points[i], airfoil_points[i + 1])
        for i in range(len(airfoil_points) - 1)
    ]
    airfoil_lines.append(occ.addLine(airfoil_points[-1], airfoil_points[0]))
    airfoil_wire = occ.addWire(airfoil_lines)
    airfoil_face = occ.addPlaneSurface([airfoil_wire])
    wing_entities = occ.extrude([(2, airfoil_face)], 0, 0, span)
    wing = [entity for entity in wing_entities if entity[0] == 3][0][1]
    fused, _ = occ.fuse([(3, fuselage)], [(3, wing)], removeObject=True, removeTool=True)
    body_volumes = [entity for entity in fused if entity[0] == 3]

fluid, _ = occ.cut([(3, farfield)], body_volumes, removeObject=True, removeTool=True)
fluid_volumes = [entity for entity in fluid if entity[0] == 3]
if len(fluid_volumes) != 1:
    gmsh.finalize()
    raise RuntimeError(f"Expected one connected fluid volume, found {len(fluid_volumes)}")
occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen", niter=2)

node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
points = np.asarray(node_coordinates, dtype=np.float64).reshape(-1, 3)
node_index = {int(tag): index for index, tag in enumerate(node_tags)}
tetra_type = gmsh.model.mesh.getElementType("Tetrahedron", 1)
tetra_element_tags, tetra_node_tags = gmsh.model.mesh.getElementsByType(tetra_type)
tetra_nodes = tetra_node_tags.reshape(-1, 4)
tetra_sicn = np.asarray(
    gmsh.model.mesh.getElementQualities(tetra_element_tags, "minSICN"),
    dtype=np.float64,
)
tets = np.asarray(
    [[node_index[int(tag)] for tag in row] for row in tetra_nodes], dtype=np.int64
)
if len(points) == 0 or len(tets) == 0:
    gmsh.finalize()
    raise RuntimeError("Gmsh returned an empty point or tetrahedron array")
volumes = oriented_tets(points, tets)
if np.any(volumes <= 0):
    gmsh.finalize()
    raise RuntimeError("Mesh contains a non-positive tetrahedron volume")

boundary_triangles, boundary_tags, boundary_tet = mesh_boundary(
    points, tets, args.shape, args.radius, far_radius, wingbody_axes
)
if args.shape == "wingbody" and (not np.any(boundary_tags == 20) or not np.any(boundary_tags == 30)):
    gmsh.finalize()
    raise RuntimeError("Wingbody classification produced an empty fuselage or wing tag")
npz_path = args.out / "mesh.npz"
np.savez_compressed(
    npz_path,
    points=points,
    tets=tets,
    boundary_triangles=boundary_triangles,
    boundary_tags=boundary_tags,
    boundary_tet=boundary_tet,
)
gmsh.finalize()

array_hash = hashlib.sha256()
for array in (points, tets, boundary_triangles, boundary_tags, boundary_tet):
    array_hash.update(np.ascontiguousarray(array).tobytes())
edge_pairs = np.concatenate(
    (tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]], tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]]),
    axis=0,
)
edge_lengths = np.linalg.norm(
    points[edge_pairs[:, 0]] - points[edge_pairs[:, 1]], axis=1
)
mesh_info = {
    "shape": args.shape,
    "reference_radius": args.radius,
    "shape_parameters": shape_parameters[args.shape],
    "mesh_size": args.size,
    "farfield_radius": far_radius,
    "physical_tags": {"farfield": 100, "sphere_or_ellipsoid": 10, "fuselage": 20, "wing": 30},
    "wingbody_tagging": (
        "All three triangle vertices must satisfy abs(ellipsoid_levelset - 1) <= 0.15; "
        "otherwise tag as wing. The two patches meet at their classification boundary."
        if args.shape == "wingbody"
        else None
    ),
    "coordinate_axes": {
        "fuselage_axis": "x",
        "wing_chord_axis": "x",
        "wing_thickness_axis": "y",
        "wing_span_axis": "z",
        "fem_alpha_xz_alignment": (
            "Not aligned: wing currently lies in x-y with z span; no geometry transform applied."
            if args.shape == "wingbody"
            else "Not applicable."
        ),
    },
    "counts": {
        "points": len(points),
        "tets": len(tets),
        "boundary_triangles": len(boundary_triangles),
        "boundary_tets": len(boundary_tet),
    },
    "quality": {
        "tet_volume_min": float(volumes.min()),
        "tet_volume_median": float(np.median(volumes)),
        "tet_volume_max": float(volumes.max()),
        "tet_volume_nonpositive": int(np.count_nonzero(volumes <= 0)),
        "tet_min_sicn": float(tetra_sicn.min()),
        "tet_median_sicn": float(np.median(tetra_sicn)),
        "tet_sicn_below_0_1": int(np.count_nonzero(tetra_sicn < 0.1)),
        "edge_length_min": float(edge_lengths.min()),
        "edge_length_median": float(np.median(edge_lengths)),
        "edge_length_max": float(edge_lengths.max()),
        "boundary_tag_counts": {
            str(tag): int(np.count_nonzero(boundary_tags == tag))
            for tag in np.unique(boundary_tags)
        },
    },
    "sha256_arrays": array_hash.hexdigest(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "metadata_revision": 2,
    "revision_reason": (
        "Boundary faces are matched to tetrahedra using face-group ordering and "
        "oriented against the opposite tetrahedron vertex."
    ),
    "elapsed_seconds": time.time() - started,
}
(args.out / "mesh.json").write_text(
    json.dumps(mesh_info, indent=2, ensure_ascii=True), encoding="utf-8"
)
print(json.dumps(mesh_info["counts"], ensure_ascii=True))
