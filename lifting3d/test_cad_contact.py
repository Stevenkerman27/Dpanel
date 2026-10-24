"""Regression checks for wall/wake curve discovery in imported CAD."""
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "audit" / "runtime" / "python"))
import gmsh
import cad_components
import cad_contact

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Geometry.Tolerance", 1e-8)

# Replay the OCC import/cut/fragment topology used by the full-volume builder.
inputs = root / "results" / "cad_inputs_v1" / "naca4412_wing_body"
declaration = json.loads((inputs / "trailing_edge.json").read_text(encoding="utf-8"))
te = np.asarray(declaration["points"], dtype=float)
gmsh.model.add("cambered_body_embedded_contact")
occ = gmsh.model.occ
roles = {}
for role in ["wing", "body"]:
    roles[role] = occ.importShapes(str((inputs / f"{role}.step").resolve()), highestDimOnly=True)
occ.synchronize()
source_copies = cad_components.copy_source_solids(gmsh, roles)
solid_list = [entity for solids in roles.values() for entity in solids]
if len(solid_list) > 1:
    solid_list, _ = occ.fuse(solid_list[:1], solid_list[1:])
    solid_list = [entity for entity in solid_list if entity[0] == 3]
outer = (3, occ.addBox(-5., -5., -5., 10., 10., 10.))
fluid, _ = occ.cut([outer], solid_list)
fluid = [entity for entity in fluid if entity[0] == 3]
stations = np.unique(np.r_[np.linspace(te[0, 1], te[-1, 1], 9), te[:, 1]])
wake_inputs = []
for left, right in zip(stations[:-1], stations[1:]):
    points = [
        occ.addPoint(float(x), float(y), 0.)
        for x, y in [
            (np.interp(left, te[:, 1], te[:, 0]), left),
            (5., left),
            (5., right),
            (np.interp(right, te[:, 1], te[:, 0]), right),
        ]
    ]
    edges = [occ.addLine(points[index], points[(index + 1) % 4]) for index in range(4)]
    wake_inputs.append((2, occ.addPlaneSurface([occ.addWire(edges)])))
fragments, ancestry = occ.fragment(fluid, wake_inputs)
fluid = [entity for entity in fragments if entity[0] == 3]
occ.synchronize()
fluid_ids = {tag for _, tag in fluid}
wake_candidates = sorted({
    tag for row in ancestry[1:] for dimension, tag in row if dimension == 2
})
embedded_wakes = {
    tag for _, volume in fluid
    for dimension, tag in gmsh.model.mesh.getEmbedded(3, volume)
    if dimension == 2
}
wake_surfaces = [
    tag for tag in wake_candidates
    if tag in embedded_wakes
    or fluid_ids.intersection(map(int, gmsh.model.getAdjacencies(2, tag)[0]))
]
wall_map, _ = cad_components.classify_wall_surfaces(
    gmsh, fluid, source_copies, 1e-7, ambiguity_rule="body_precedence"
)
wall_surfaces = sorted(wall_map)

intervals, diagnostics = cad_contact.discover_contact_intervals(
    wall_surfaces, wake_surfaces, te, te_tolerance=1e-7
)
assert len(wall_surfaces) == 9
assert len(wake_surfaces) == 40
assert diagnostics["shared_curve_source_counts"] == {
    "wall_boundary_wake_boundary": 36,
    "wall_embedded_wake_boundary": 6,
    "wall_boundary_wake_embedded": 0,
    "wall_embedded_wake_embedded": 0,
}
assert intervals == [[-0.2315032397181942, 0.2315032397181942]]
embedded_contacts = [
    curve for curve in diagnostics["curves"]
    if "embedded" in curve["wall_curve_sources"] and not curve["is_user_te_polyline"]
]
assert len(embedded_contacts) == 6
assert all(curve["wake_curve_sources"] == ["boundary"] for curve in embedded_contacts)
print(json.dumps({
    "case": "naca4412_wing_body_builder_topology_replay",
    "wall_surface_count": len(wall_surfaces),
    "wake_surface_count": len(wake_surfaces),
    "shared_curve_source_counts": diagnostics["shared_curve_source_counts"],
    "contact_intervals": intervals,
    "embedded_contact_curves": [
        {
            "curve_tag": curve["curve_tag"],
            "y_interval": [curve["projection_y_min"], curve["projection_y_max"]],
            "wall_curve_sources": curve["wall_curve_sources"],
            "wake_curve_sources": curve["wake_curve_sources"],
        }
        for curve in embedded_contacts
    ],
}, indent=2))
gmsh.clear()

# Check a minimal wall-embedded / wake-boundary contact without a closed solid.
gmsh.model.add("embedded_curve_contact")
occ = gmsh.model.occ
left = occ.addPoint(0., -1., 0.)
right = occ.addPoint(0., 1., 0.)
contact_curve = occ.addLine(left, right)
far_left = occ.addPoint(2., -1., 0.)
far_right = occ.addPoint(2., 1., 0.)
wake = occ.addPlaneSurface([occ.addWire([
    occ.addLine(left, far_left),
    occ.addLine(far_left, far_right),
    occ.addLine(far_right, right),
    contact_curve,
])])
wall_points = [
    occ.addPoint(0., -1., -1.),
    occ.addPoint(0., 1., -1.),
    occ.addPoint(0., 1., 1.),
    occ.addPoint(0., -1., 1.),
]
wall_edges = [
    occ.addLine(wall_points[index], wall_points[(index + 1) % 4])
    for index in range(4)
]
wall = occ.addPlaneSurface([occ.addWire(wall_edges)])
occ.synchronize()
gmsh.model.mesh.embed(1, [contact_curve], 2, wall)
assert (1, contact_curve) in gmsh.model.mesh.getEmbedded(2, wall)
assert (1, contact_curve) not in gmsh.model.getBoundary(
    [(2, wall)], combined=False, oriented=False
)
embedded_intervals, embedded_diagnostics = cad_contact.discover_contact_intervals(
    [wall], [wake], [[-1., -2., 0.], [-1., 2., 0.]]
)
assert embedded_intervals == [[-1., 1.]]
assert embedded_diagnostics["shared_curve_source_counts"]["wall_embedded_wake_boundary"] == 1
assert embedded_diagnostics["curves"][0]["wall_curve_sources"] == ["embedded"]
assert embedded_diagnostics["curves"][0]["wake_curve_sources"] == ["boundary"]
print(json.dumps({
    "case": "minimal_embedded_curve_contact",
    "contact_intervals": embedded_intervals,
    "shared_curve_source_counts": embedded_diagnostics["shared_curve_source_counts"],
}, indent=2))
gmsh.clear()

# Check the reverse source combination: wall boundary and wake embedding.
gmsh.model.add("wake_embedded_curve_contact")
occ = gmsh.model.occ
left_bottom = occ.addPoint(-1., -1., 0.)
contact_left = occ.addPoint(0., -1., 0.)
contact_right = occ.addPoint(0., 1., 0.)
left_top = occ.addPoint(-1., 1., 0.)
contact_curve = occ.addLine(contact_left, contact_right)
wall = occ.addPlaneSurface([occ.addWire([
    occ.addLine(left_bottom, contact_left),
    contact_curve,
    occ.addLine(contact_right, left_top),
    occ.addLine(left_top, left_bottom),
])])
wake_points = [
    occ.addPoint(0., -1., -1.),
    occ.addPoint(0., 1., -1.),
    occ.addPoint(0., 1., 1.),
    occ.addPoint(0., -1., 1.),
]
wake_edges = [
    occ.addLine(wake_points[index], wake_points[(index + 1) % 4])
    for index in range(4)
]
wake = occ.addPlaneSurface([occ.addWire(wake_edges)])
occ.synchronize()
gmsh.model.mesh.embed(1, [contact_curve], 2, wake)
assert (1, contact_curve) in gmsh.model.mesh.getEmbedded(2, wake)
assert (1, contact_curve) not in gmsh.model.getBoundary(
    [(2, wake)], combined=False, oriented=False
)
reverse_intervals, reverse_diagnostics = cad_contact.discover_contact_intervals(
    [wall], [wake], [[-1., -2., 0.], [-1., 2., 0.]]
)
assert reverse_intervals == [[-1., 1.]]
assert reverse_diagnostics["shared_curve_source_counts"]["wall_boundary_wake_embedded"] == 1
assert reverse_diagnostics["curves"][0]["wall_curve_sources"] == ["boundary"]
assert reverse_diagnostics["curves"][0]["wake_curve_sources"] == ["embedded"]
print(json.dumps({
    "case": "wake_embedded_curve_contact",
    "contact_intervals": reverse_intervals,
    "shared_curve_source_counts": reverse_diagnostics["shared_curve_source_counts"],
}, indent=2))

gmsh.finalize()
print("PASS: CAD boundary and embedded contact discovery")
