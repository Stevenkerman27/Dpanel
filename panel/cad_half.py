"""Mesh the RIGHT half of a real OCC wing/body fluid-domain CAD model.

Unlike panel/cad.py, this script first cuts the fluid domain with y>=0 and
only THEN meshes physical wetted wall and fluid-side wake surfaces. The
temporary y=0 fluid symmetry boundary is not an aerodynamic panel.
No already-meshed full-aircraft triangles are clipped or mirror-matched.

Uses the existing Dpanel Gmsh runtime and lifting3d CAD classification logic.
Topology changes in OCC are discrete; no CAD Boolean AD is claimed.

Example (run from repository root, with original unmodified CAD inputs):
 python -m panel.cad_half --wing-cad WING.step --trailing-edge te.json
     --out runs/NEW_HALF_MESH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "lifting3d"))
sys.path.insert(0, str(PROJECT / "audit" / "runtime" / "python"))

import cad_components
import cad_contact
import cad_te
import gmsh

from .half_surface_mesh import build_half_surface
from .mesh_feature_size import read_policy


def _positive_trailing_edge(te, *, tolerance):
    """Restrict original full-span TE to y>=0 in normalized coordinates.

    If y=0 lies inside a TE CAD segment, linearly interpolate the exact
    declared root point; never replace a physical trailing edge by a
    projected or invented centreline cut.
    """
    te = cad_contact.validate_te_polyline(te)
    if te[0, 1] > tolerance or te[-1, 1] <= tolerance:
        raise ValueError("Full CAD trailing edge must span positive and negative y")
    positive = te[te[:, 1] > tolerance]
    if len(positive) < 1:
        raise ValueError("Missing positive-y trailing-edge stations")
    # If exact root is already declared it must match the interpolated x/z.
    x0 = float(np.interp(0., te[:, 1], te[:, 0]))
    root = np.asarray([x0, 0., 0.])
    at0 = te[np.abs(te[:, 1]) <= tolerance]
    if len(at0) and np.max(np.linalg.norm(at0-root, axis=1)) > tolerance:
        raise ValueError("Declared TE does not pass through a unique x at y=0")
    result = np.vstack((root, positive))
    if np.any(np.diff(result[:, 1]) <= 0):
        raise ValueError("Nonmonotone positive-y trailing edge")
    return result


def _mesh_triangles(gmsh_module, surface_tags, node_index):
    faces = []
    for surface in surface_tags:
        element_tags, element_nodes = gmsh_module.model.mesh.getElementsByType(2, int(surface))
        values = np.asarray(element_nodes, dtype=np.int64).reshape(-1, 3)
        if len(element_tags) != len(values):
            raise RuntimeError("A selected CAD face contains nontriangular cells")
        faces.extend([[node_index[int(tag)] for tag in triangle] for triangle in values])
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _build(args, stage):
    declaration = json.loads(args.trailing_edge.read_text(encoding="utf-8"))
    chord = float(declaration["chord"])
    area = float(declaration["reference_area"])
    origin = np.asarray(declaration.get("origin", [0., 0., 0.]), dtype=float)
    full_te = np.asarray(declaration["points"], dtype=float)
    if (chord <= 0 or area <= 0 or origin.shape != (3,)
            or abs(origin[1]) > 1e-12 or
            not np.isfinite(origin).all() or
            not np.isfinite(full_te).all()):
        raise ValueError("Require symmetric physical origin, positive chord and full Sref")
    normalized_te = (full_te-origin)/chord
    te = _positive_trailing_edge(normalized_te, tolerance=1e-9)
    if (not np.isfinite([args.size, args.far_size, args.radius, args.speed,
                         args.density, args.wake_length]).all()
            or min(args.size, args.far_size, args.radius, args.speed,
                   args.density, args.wake_length) <= 0
            or args.wake_strips < 4 or args.ncpu < 1
            or max(abs(te).ravel()) >= args.radius-args.size):
        raise ValueError("Invalid CAD meshing/flight parameters")
    sizing_path = getattr(args, "mesh_features", None)
    sizing = None
    if sizing_path is not None:
        sizing_path = Path(sizing_path).resolve()
        sizing = read_policy(sizing_path, origin=origin, chord=chord,
                             wall_size=args.size)
    stations = np.unique(np.r_[
        np.linspace(0., te[-1, 1], args.wake_strips+1),
        te[:, 1],
    ])
    if abs(stations[0]) > 1e-12:
        raise ValueError("Circulation half stations must start at y=0")

    gmsh.initialize()
    gmsh.logger.start()
    gmsh.option.setNumber("General.NumThreads", args.ncpu)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", args.ncpu)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.size/8.)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.far_size)

    history = []
    contacts = np.empty((0, 2))
    for attempt in range(2):
        stage["name"] = f"half_OCC_boolean_and_contact_pass_{attempt}"
        gmsh.clear()
        gmsh.model.add("dpanel_positive_half_fluid")
        occ = gmsh.model.occ
        roles = {}
        for role, path in (("wing", args.wing_cad), ("body", args.body_cad)):
            if path is None:
                continue
            imported = occ.importShapes(str(path.resolve()), highestDimOnly=True)
            solids = [(dim, tag) for dim, tag in imported if dim == 3]
            if not solids:
                raise ValueError(f"{role} CAD does not contain closed 3D solids")
            occ.translate(solids, *(-origin))
            occ.dilate(solids, 0., 0., 0., 1./chord, 1./chord, 1./chord)
            roles[role] = solids
        occ.synchronize()
        # Validate against the ORIGINAL imported wing CAD, not an invented
        # straightened TE in a half model.
        te_check = cad_te.validate_declared_te(
            gmsh, roles["wing"], te, 1e-7)
        (args.out/f"te_check_{attempt}.json").write_text(
            json.dumps(te_check, indent=2), encoding="utf-8")
        for role, solids in roles.items():
            bounds = []
            for dim, tag in solids:
                bbox = np.asarray(occ.getBoundingBox(dim, tag), dtype=float)
                if np.max(abs(bbox)) >= args.radius-1e-7:
                    raise ValueError("Source CAD solid must lie strictly inside full farfield")
                if occ.getMass(dim, tag) <= 0:
                    raise ValueError("Invalid source CAD volume")
                bounds.append(bbox)
            # This is a necessary (not sufficient) symmetry guard. Exact
            # CAD symmetry of shape, not just bounding box, is still
            # a separate geometric/physical validation responsibility.
            bottom = min(item[1] for item in bounds)
            top = max(item[4] for item in bounds)
            if bottom >= 0 or top <= 0 or abs(bottom+top) > 1e-6:
                raise ValueError(f"{role} source solid does not span a symmetric y extent")

        reference_copies = cad_components.copy_source_solids(gmsh, roles)
        union = [solid for collection in roles.values() for solid in collection]
        if len(union) > 1:
            union, _ = occ.fuse(union[:1], union[1:])
            union = [entry for entry in union if entry[0] == 3]
        # This is the decisive difference from cad.py: the POSITIVE-Y
        # fluid half-box, not a full cube later cut into two mesh halves.
        half_domain = occ.addBox(
            -args.radius, 0., -args.radius,
            2*args.radius, args.radius, 2*args.radius)
        fluid, _ = occ.cut([(3, half_domain)], union)
        fluid = [item for item in fluid if item[0] == 3]
        if len(fluid) != 1:
            raise ValueError("Expected one connected positive half-fluid volume")

        wake_inputs = []
        for left, right in zip(stations[:-1], stations[1:]):
            corners = [
                (float(np.interp(left, te[:, 1], te[:, 0])), left),
                (args.radius, left),
                (args.radius, right),
                (float(np.interp(right, te[:, 1], te[:, 0])), right),
            ]
            v = [occ.addPoint(float(x), float(y), 0.) for x, y in corners]
            curves = [occ.addLine(v[k], v[(k+1)%4]) for k in range(4)]
            wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(curves)])))
        fragments, ancestry = occ.fragment(fluid, wake_inputs)
        fluid = [item for item in fragments if item[0] == 3]
        occ.synchronize()
        fluid_ids = {tag for dim, tag in fluid}
        embedded = {
            tag for dim, volume in fluid
            for k, tag in gmsh.model.mesh.getEmbedded(3, volume) if k == 2
        }
        candidates = sorted({
            tag for entries in ancestry[1:]
            for dim, tag in entries if dim == 2
        })
        wake_surfaces = []
        for tag in candidates:
            adjacent = set(int(v) for v in gmsh.model.getAdjacencies(2, tag)[0])
            if tag in embedded or bool(fluid_ids & adjacent):
                wake_surfaces.append(tag)
        if not wake_surfaces:
            raise ValueError("No actual half wake incident to positive fluid volume")
        component_tags, component_diagnostics = cad_components.classify_wall_surfaces(
            gmsh, fluid, reference_copies, 1e-7,
            ambiguity_rule="body_precedence")
        wall_surfaces = sorted(component_tags)
        if not wall_surfaces:
            raise ValueError("No physical wetted wall faces matched source CAD")
        # A symmetry-plane face cannot match a curved/full original source
        # wall; require explicit evidence before excluding it from the mesh.
        # OCC inflates the bounding box of a trimmed planar face by the
        # shape tolerance: on the real NACA4412 wing STEP the exact y=0
        # cap reports |y| <= 5.0e-06 even though every generated node lies
        # exactly on y=0. Identify candidate caps by surface kind plus a
        # bbox window far below any physical gap, and then require the
        # generated nodes themselves to be exactly on the plane below.
        plane_candidates = []
        for dim, tag in gmsh.model.getBoundary(
                fluid, combined=False, oriented=False):
            if dim != 2:
                continue
            tag = int(tag)
            bbox = np.asarray(occ.getBoundingBox(2, tag), dtype=float)
            if (max(abs(bbox[1]), abs(bbox[4])) < 1e-5
                    and gmsh.model.getType(2, tag) == "Plane"):
                plane_candidates.append(tag)
        if not plane_candidates:
            raise ValueError("Half fluid has no explicit y=0 symmetry boundary")
        plane_faces = plane_candidates
        if set(plane_faces) & set(wall_surfaces):
            raise ValueError("Artificial symmetry cap classified as real wetted wall")
        contact_intervals, contact_diag = cad_contact.discover_contact_intervals(
            wall_surfaces, wake_surfaces, te, te_tolerance=1e-7)
        contact_intervals = np.asarray(contact_intervals, dtype=float).reshape(-1, 2)
        history.append(dict(
            pass_index=attempt,
            fluid_volumes=len(fluid),
            wake_surfaces=len(wake_surfaces),
            wall_surfaces=len(wall_surfaces),
            symmetry_plane_faces=len(plane_faces),
            contact_intervals=contact_intervals.tolist(),
            component_classification=component_diagnostics,
            contact_diagnostics=contact_diag,
        ))
        if attempt == 0:
            contacts = contact_intervals
            for endpoint in contacts.ravel():
                # The root station must remain EXACTLY y=0 for the
                # half-native circulation basis; OCC contact curves
                # within its tolerance are attached to this fixed root.
                endpoint = 0. if abs(endpoint) < 1e-7 else float(endpoint)
                nearest = int(np.argmin(abs(stations-endpoint)))
                if abs(stations[nearest]-endpoint) < 1e-7:
                    stations[nearest] = endpoint
                else:
                    stations = np.sort(np.r_[stations, endpoint])
            continue
        if contacts.shape != contact_intervals.shape or (
                len(contacts) and np.max(abs(contacts-contact_intervals)) > 1e-7):
            raise ValueError("Contact span intervals changed after refinement")
        contacts = contact_intervals.copy()
        contacts[np.abs(contacts) < 1e-7] = 0.
        removal = cad_components.remove_source_copies_before_mesh(
            gmsh, fluid, reference_copies)
        stale_wake = [(2, tag) for tag in candidates if tag not in wake_surfaces]
        if stale_wake:
            occ.remove(stale_wake, recursive=True)
            occ.synchronize()
        (args.out/"source_copy_removal.json").write_text(
            json.dumps(removal, indent=2), encoding="utf-8")

    stage["name"] = "gmsh_positive_half_surface_meshing"
    field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field, "SurfacesList", wall_surfaces)
    gmsh.model.mesh.field.setNumber(field, "Sampling", 150)
    control = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(control, "InField", field)
    gmsh.model.mesh.field.setNumber(control, "SizeMin", args.size)
    gmsh.model.mesh.field.setNumber(control, "SizeMax", args.far_size)
    gmsh.model.mesh.field.setNumber(control, "DistMin", .3)
    gmsh.model.mesh.field.setNumber(control, "DistMax", 2.)
    gmsh.model.mesh.field.setAsBackgroundMesh(control)
    if sizing is not None:
        gmsh.model.mesh.setSizeCallback(sizing)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(tag): i for i, tag in enumerate(tags)}
    wall = _mesh_triangles(gmsh, wall_surfaces, index)
    wall_tags = np.concatenate([
        np.full(len(gmsh.model.mesh.getElementsByType(2, tag)[0]),
                component_tags[tag], dtype=np.int32)
        for tag in wall_surfaces
    ])
    wake = _mesh_triangles(gmsh, wake_surfaces, index)
    if not len(wall) or not len(wake):
        raise ValueError("Positive-half OCC mesh has no physical wall/wake triangles")
    # Strict plane and sign checks on the generated nodes, not on the
    # inflated OCC bounding boxes: the artificial cap must be exactly on
    # y=0 and no physical wall/wake node may cross to the negative side.
    wall_nodes = np.unique(wall)
    wake_nodes = np.unique(wake)
    if points[wall_nodes, 1].min() < -1e-12 or points[wake_nodes, 1].min() < -1e-12:
        raise ValueError("Positive-half mesh has a physical node below y=0")
    cap_nodes = []
    for tag in plane_faces:
        _, cap_node_tags = gmsh.model.mesh.getElementsByType(2, tag)
        cap_nodes.extend(index[int(node)] for node in cap_node_tags)
    if cap_nodes:
        cap_y = points[np.unique(np.asarray(cap_nodes, dtype=int)), 1]
        if float(np.max(np.abs(cap_y))) > 1e-12:
            raise ValueError(
                "Artificial symmetry cap nodes are not exactly on y=0")
    # The Gmsh arrays are normalized; convert once to physical SI.
    physical_points = points*chord + origin
    physical_te = te*chord + origin
    physical_stations = stations*chord + origin[1]
    physical_contacts = contacts*chord + origin[1]
    half_wake_end = (float(te[:, 0].max())+args.wake_length)*chord+origin[0]
    alpha = math.radians(args.alpha_deg)
    flow = np.asarray([args.speed*math.cos(alpha), 0.,
                       args.speed*math.sin(alpha)])
    stage["name"] = "half_mesh_topology_and_kutta_export"
    fixture = build_half_surface(
        physical_points, wall, wall_tags, wake,
        te_polyline=physical_te, stations=physical_stations,
        contact_intervals=physical_contacts,
        reference_area=area, reference_chord=chord,
        reference_point=np.array([origin[0]+.25*chord, 0., origin[2]]),
        freestream=flow, density=args.density,
        wake_end_x=half_wake_end,
    )
    fixture["provenance"] = dict(
        type="OCC positive-y fluid domain, not a clipped full mesh",
        wing_cad=str(args.wing_cad.resolve()),
        body_cad=None if args.body_cad is None else str(args.body_cad.resolve()),
        te_declaration=str(args.trailing_edge.resolve()),
        CAD_contact_intervals=physical_contacts.tolist(),
        CAD_passes=history,
        half_fluid_symmetry_caps_excluded=plane_faces,
        physical_wall_surface_tags=wall_surfaces,
        physical_wake_surface_tags=wake_surfaces,
        feature_sizing=None if sizing_path is None else dict(
            source=str(sizing_path),
            source_sha256=hashlib.sha256(sizing_path.read_bytes()).hexdigest(),
            feature_names=[name for name, _, _, _ in sizing.features],
            callback="Gmsh model.mesh.setSizeCallback; requests only",
        ),
    )
    surface = args.out/"surface.json"
    surface.write_text(json.dumps(fixture, indent=2, allow_nan=False)+"\n",
                       encoding="utf-8")
    return fixture, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wing-cad", type=Path, required=True)
    p.add_argument("--body-cad", type=Path)
    p.add_argument("--trailing-edge", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=float, default=.3)
    p.add_argument("--mesh-features", type=Path,
                   help="Optional SI feature polylines for local LE/TE/tip refinement")
    p.add_argument("--far-size", type=float, default=1.5)
    p.add_argument("--radius", type=float, default=5.)
    p.add_argument("--wake-strips", type=int, default=8)
    p.add_argument("--wake-length", type=float, default=30.)
    p.add_argument("--alpha-deg", type=float, default=5.)
    p.add_argument("--speed", type=float, default=1.)
    p.add_argument("--density", type=float, default=1.225)
    p.add_argument("--ncpu", type=int, default=8)
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Preserve previously meshed/validated CAD cases")
    args.out.mkdir(parents=True)
    stage = {"name": "input"}
    start = time.perf_counter()
    try:
        fixture, passes = _build(args, stage)
        status = dict(status="half_surface_created_not_physical_accuracy",
                      full_aircraft_mesh_generated=False,
                      right_wall_panels=len(fixture["faces"]),
                      right_wake_panels=len(fixture["wake_faces"]),
                      right_independent_mu=len(fixture["faces"]),
                      right_independent_gamma=max(fixture["gamma_station_dof"])+1,
                      geometry_is_from_OCC_half_domain=True,
                      physical_accuracy_accepted=False,
                      elapsed_seconds=time.perf_counter()-start)
        (args.out/"manifest.json").write_text(
            json.dumps(status, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
    except BaseException as exc:
        (args.out/"failure.json").write_text(
            json.dumps(dict(stage=stage["name"], status="failed",
                            exception=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc(),
                            elapsed_seconds=time.perf_counter()-start),
                       indent=2)+"\n", encoding="utf-8")
        raise
    finally:
        if gmsh.isInitialized():
            messages = gmsh.logger.get()
            (args.out/"gmsh.log").write_text("\n".join(messages), encoding="utf-8")
            gmsh.logger.stop()
            gmsh.finalize()


if __name__ == "__main__":
    main()
