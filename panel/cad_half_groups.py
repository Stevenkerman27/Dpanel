"""Mesh the RIGHT half of a multi-lifting-surface OCC CAD model (v2 groups).

This is the group-aware sibling of panel/cad_half.py. It keeps the same
positive-y OCC fluid-domain construction and the same wall/plane-cap
classification, but it builds SEVERAL horizontal wake sheets, each at its own
declared z, and exports a dpanel.half_shed_groups.v2 declaration next to the
surface. The v1 single-wake path in panel/cad_half.py is untouched.

Deliberate v2 restrictions (refused, not silently degraded):
  * no body CAD, therefore no wake/wall contact intervals;
  * every wake sheet must be planar at its declared z with +z orientation;
  * every sheet must be attached to a real declared TE curve of the wing CAD.

Example (run from the repository root):
 python -m panel.cad_half_groups --wing-cad TWO_LAYER.step
     --wake-groups groups.json --out runs/NEW_HALF_GROUPS
"""
from __future__ import annotations

import argparse
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
import cad_te
import gmsh

from .half_group_surface import build_half_groups_surface
from .wake_group_declaration import merged_stations, read_wake_groups


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
    declaration, groups = read_wake_groups(
        args.wake_groups, chord=float(args.chord),
        origin=np.asarray(args.origin, dtype=float))
    chord = float(args.chord)
    area = float(args.reference_area)
    origin = np.asarray(args.origin, dtype=float)
    if (chord <= 0 or area <= 0 or origin.shape != (3,)
            or abs(origin[1]) > 1e-12 or not np.isfinite(origin).all()):
        raise ValueError("Require symmetric physical origin, positive chord and full Sref")
    if not groups:
        raise ValueError("Wake-group declaration contains no groups")
    identifiers = [g["group_id"] for g in groups]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate wake group_id")
    planes = [round(g["normalized_z"], 9) for g in groups]
    if len(set(planes)) != len(planes):
        raise ValueError("Two wake groups share the same z plane")
    if (not np.isfinite([args.size, args.far_size, args.radius, args.speed,
                         args.density, args.wake_length]).all()
            or min(args.size, args.far_size, args.radius, args.speed,
                   args.density, args.wake_length) <= 0
            or args.wake_strips < 4 or args.ncpu < 1):
        raise ValueError("Invalid CAD meshing/flight parameters")
    for group in groups:
        if np.max(abs(group["normalized_te"])) >= args.radius-args.size:
            raise ValueError("Declared wake group TE does not fit inside the farfield")

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
    for attempt in range(2):
        stage["name"] = f"half_OCC_boolean_pass_{attempt}"
        gmsh.clear()
        gmsh.model.add("dpanel_positive_half_groups")
        occ = gmsh.model.occ
        imported = occ.importShapes(str(Path(args.wing_cad).resolve()),
                                    highestDimOnly=True)
        solids = [(dim, tag) for dim, tag in imported if dim == 3]
        if not solids:
            raise ValueError("Wing CAD does not contain closed 3D solids")
        occ.translate(solids, *(-origin))
        occ.dilate(solids, 0., 0., 0., 1./chord, 1./chord, 1./chord)
        occ.synchronize()
        te_checks = {}
        for group in groups:
            te_checks[group["group_id"]] = cad_te.validate_declared_te(
                gmsh, solids, group["normalized_te"], 1e-7)
        (args.out/f"te_check_{attempt}.json").write_text(
            json.dumps(te_checks, indent=2), encoding="utf-8")
        bounds = []
        for dim, tag in solids:
            bbox = np.asarray(occ.getBoundingBox(dim, tag), dtype=float)
            if np.max(abs(bbox)) >= args.radius-1e-7:
                raise ValueError("Source CAD solid must lie strictly inside full farfield")
            if occ.getMass(dim, tag) <= 0:
                raise ValueError("Invalid source CAD volume")
            bounds.append(bbox)
        bottom = min(item[1] for item in bounds)
        top = max(item[4] for item in bounds)
        if bottom >= 0 or top <= 0 or abs(bottom+top) > 1e-6:
            raise ValueError("Wing source solids do not span a symmetric y extent")
        reference_copies = cad_components.copy_source_solids(gmsh, {"wing": solids})
        union = [solid for solid in solids]
        if len(union) > 1:
            union, _ = occ.fuse(union[:1], union[1:])
            union = [entry for entry in union if entry[0] == 3]
        half_domain = occ.addBox(
            -args.radius, 0., -args.radius,
            2*args.radius, args.radius, 2*args.radius)
        fluid, _ = occ.cut([(3, half_domain)], union)
        fluid = [item for item in fluid if item[0] == 3]
        if len(fluid) != 1:
            raise ValueError("Expected one connected positive half-fluid volume")
        wake_inputs, owners = [], []
        for index, group in enumerate(groups):
            te = group["normalized_te"]
            z = float(group["normalized_z"])
            stations = merged_stations(te[:, 1], args.wake_strips)
            for left, right in zip(stations[:-1], stations[1:]):
                corners = [
                    (float(np.interp(left, te[:, 1], te[:, 0])), left),
                    (args.radius, left),
                    (args.radius, right),
                    (float(np.interp(right, te[:, 1], te[:, 0])), right),
                ]
                v = [occ.addPoint(float(x), float(y), z) for x, y in corners]
                curves = [occ.addLine(v[k], v[(k+1) % 4]) for k in range(4)]
                wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(curves)])))
                owners.append(index)
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
        live = []
        for tag in candidates:
            adjacent = set(int(v) for v in gmsh.model.getAdjacencies(2, tag)[0])
            if tag in embedded or bool(fluid_ids & adjacent):
                live.append(tag)
        if not live:
            raise ValueError("No actual half wake incident to positive fluid volume")
        # Map each surviving wake face back to its owning group through the
        # ancestry blocks: ancestry[i+1] are the fragments of wake_inputs[i].
        owner_of = {}
        for owner, entries in zip(owners, ancestry[1:]):
            for dim, tag in entries:
                if dim == 2 and int(tag) in live:
                    owner_of[int(tag)] = owner
        wake_surfaces = {
            index: sorted(tag for tag, owner in owner_of.items() if owner == index)
            for index in range(len(groups))
        }
        for index, group in enumerate(groups):
            if not wake_surfaces[index]:
                raise ValueError(
                    f"Wake group {group['group_id']!r} has no meshed wake face after the boolean")
        component_tags, component_diagnostics = cad_components.classify_wall_surfaces(
            gmsh, fluid, reference_copies, 1e-7,
            ambiguity_rule="body_precedence")
        wall_surfaces = sorted(component_tags)
        if not wall_surfaces:
            raise ValueError("No physical wetted wall faces matched source CAD")
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
        declared = [tag for tags in wake_surfaces.values() for tag in tags]
        history.append(dict(
            pass_index=attempt,
            fluid_volumes=len(fluid),
            wake_inputs=len(wake_inputs),
            wake_surfaces={groups[k]["group_id"]: len(v)
                           for k, v in wake_surfaces.items()},
            wall_surfaces=len(wall_surfaces),
            symmetry_plane_faces=len(plane_faces),
            component_classification=component_diagnostics,
        ))
        if attempt == 0:
            continue
        removal = cad_components.remove_source_copies_before_mesh(
            gmsh, fluid, reference_copies)
        stale = [(2, tag) for tag in candidates if tag not in live]
        if stale:
            occ.remove(stale, recursive=True)
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
    wake_by_group = {k: _mesh_triangles(gmsh, v, index)
                     for k, v in wake_surfaces.items()}
    wake = np.vstack([wake_by_group[k] for k in sorted(wake_by_group)])
    if not len(wall) or not len(wake):
        raise ValueError("Positive-half OCC mesh has no physical wall/wake triangles")
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
            raise ValueError("Artificial symmetry cap nodes are not exactly on y=0")
    physical_points = points*chord + origin
    alpha = math.radians(args.alpha_deg)
    flow = np.asarray([args.speed*math.cos(alpha), 0.,
                       args.speed*math.sin(alpha)])
    stage["name"] = "half_group_topology_and_kutta_export"
    group_records = []
    for k, group in enumerate(groups):
        group_records.append(dict(
            group_id=group["group_id"],
            role=group["role"],
            kind=group["kind"],
            root_condition=group["root_condition"],
            wake_faces=wake_by_group[k],
            te=group["physical_te"],
            z=float(group["physical_z"]),
        ))
    fixture, spec = build_half_groups_surface(
        physical_points, wall, wall_tags, group_records,
        reference_area=area, reference_chord=chord,
        reference_point=np.array([origin[0]+.25*chord, 0., origin[2]]),
        freestream=flow, density=args.density,
        wake_length=args.wake_length, wake_strips=args.wake_strips,
    )
    fixture["provenance"] = dict(
        type="OCC positive-y multi-wake fluid domain, not a clipped full mesh",
        wing_cad=str(Path(args.wing_cad).resolve()),
        wake_groups_declaration=str(Path(args.wake_groups).resolve()),
        CAD_passes=history,
        half_fluid_symmetry_caps_excluded=plane_faces,
        physical_wall_surface_tags=wall_surfaces,
        physical_wake_surface_tags={groups[k]["group_id"]: v
                                    for k, v in wake_surfaces.items()},
        body_contact_intervals="unsupported in v2 (no body CAD accepted)",
    )
    (args.out/"surface.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.out/"groups.json").write_text(
        json.dumps(spec, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return fixture, spec, history

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wing-cad", type=Path, required=True)
    p.add_argument("--wake-groups", type=Path, required=True,
                   help="JSON with chord/reference_area/origin and per-group TE points+z")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chord", type=float, required=True)
    p.add_argument("--reference-area", type=float, required=True)
    p.add_argument("--origin", type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument("--size", type=float, default=.3)
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
        fixture, spec, passes = _build(args, stage)
        status = dict(
            status="half_group_surface_created_not_physical_accuracy",
            full_aircraft_mesh_generated=False,
            right_wall_panels=len(fixture["faces"]),
            right_wake_panels=len(fixture["wake_faces"]),
            groups=[dict(group_id=item["group_id"],
                         wake_panels=len(item["wake_faces"]),
                         shed_edges=len(item["shed_edges"]),
                         stations=len(item["station_dof"]),
                         circulation_unknowns=int(np.max(item["station_dof"])) + 1,
                         wake_z=item["wake_z"])
                    for item in spec["groups"]],
            right_independent_mu=len(fixture["faces"]),
            right_independent_gamma=sum(int(np.max(item["station_dof"])) + 1
                                        for item in spec["groups"]),
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