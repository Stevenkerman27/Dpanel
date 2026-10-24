"""OCC ownership checks on fused, cut and fragmented CAD solids."""
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "audit" / "runtime" / "python"))
import gmsh
import cad_components

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.18)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.55)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.model.add("cad_component_ownership_test")
occ = gmsh.model.occ

outer = (3, occ.addBox(-2.0, -2.0, -2.0, 4.0, 4.0, 4.0))
body = (3, occ.addSphere(-0.25, 0.0, 0.0, 0.72))
wing = (3, occ.addBox(0.20, -0.28, -0.08, 1.10, 0.56, 0.16))
source_copies = cad_components.copy_source_solids(
    gmsh, {"body": [body], "wing": [wing]})
fused, _ = occ.fuse([body], [wing], removeObject=True, removeTool=True)
fused_solids = [(dim, tag) for dim, tag in fused if dim == 3]
assert len(fused_solids) == 1
fluid, _ = occ.cut([outer], fused_solids, removeObject=True, removeTool=True)
fluid_volumes = [(dim, tag) for dim, tag in fluid if dim == 3]
assert len(fluid_volumes) == 1

partition_points = [
    occ.addPoint(-1.5, -2.0, -2.0),
    occ.addPoint(-1.5, 2.0, -2.0),
    occ.addPoint(-1.5, 2.0, 2.0),
    occ.addPoint(-1.5, -2.0, 2.0),
]
partition_curves = [
    occ.addLine(partition_points[index], partition_points[(index + 1) % 4])
    for index in range(4)
]
partition = (2, occ.addPlaneSurface([occ.addWire(partition_curves)]))
fragmented, _ = occ.fragment(
    fluid_volumes, [partition], removeObject=True, removeTool=True)
fluid_volumes = [(dim, tag) for dim, tag in fragmented if dim == 3]
assert len(fluid_volumes) == 2
occ.synchronize()

wall_tags, ownership = cad_components.classify_wall_surfaces(
    gmsh, fluid_volumes, source_copies, geometric_tolerance=1e-7,
    sample_count=5, ambiguity_rule="body_precedence")
assert wall_tags
assert 20 in wall_tags.values()
assert 30 in wall_tags.values()
assert all(row["sample_count"] >= 3 for row in ownership if row["physical_tag"] is not None)
assert all(row["ambiguous"] is False for row in ownership)

wing_only_tags, wing_only_diagnostics = cad_components.classify_wall_surfaces(
    gmsh, fluid_volumes, {"wing": source_copies["wing"]},
    geometric_tolerance=1e-7, sample_count=5)
assert wing_only_tags
assert set(wing_only_tags.values()) == {10}

body_owned_surface = next(tag for tag, physical in wall_tags.items() if physical == 20)
ambiguous_tags, ambiguous_diagnostics = cad_components.classify_wall_surfaces(
    gmsh, fluid_volumes, {
        "body": source_copies["body"],
        "wing": source_copies["body"],
    }, geometric_tolerance=1e-7, sample_count=5,
    ambiguity_rule="body_precedence")
ambiguous_row = next(
    row for row in ambiguous_diagnostics
    if row["surface_tag"] == body_owned_surface)
assert ambiguous_tags[body_owned_surface] == 20
assert ambiguous_row["ambiguous"] is True
assert ambiguous_row["matched_roles"] == ["body", "wing"]
assert ambiguous_row["ambiguity_rule"] == "body_precedence"

cleanup = cad_components.remove_source_copies_before_mesh(
    gmsh, fluid_volumes, source_copies)
assert cleanup["fluid_mass_abs_change"] == 0.0
assert cleanup["fluid_boundary_unchanged"]
gmsh.model.mesh.generate(3)
tetrahedra = len(gmsh.model.mesh.getElementsByType(4)[0])
assert tetrahedra > 0

for physical_tag in [10, 20, 30]:
    surfaces = [tag for tag, value in wall_tags.items() if value == physical_tag]
    if surfaces:
        gmsh.model.addPhysicalGroup(2, surfaces, physical_tag)

print({
    "status": "passed",
    "operation_sequence": ["copy_source_solids", "fuse", "cut", "fragment",
                           "classify_trimmed_faces", "remove_source_copies", "generate3"],
    "fluid_volumes_after_fragment": len(fluid_volumes),
    "classified_surface_count": len(wall_tags),
    "physical_tag_surface_counts": {
        str(tag): sum(value == tag for value in wall_tags.values())
        for tag in [10, 20, 30]
    },
    "maximum_classified_role_residual": max(
        row["role_max_sample_residual"][row["selected_role"]]
        for row in ownership if row["physical_tag"] is not None),
    "ambiguity_policy": ambiguous_row["ambiguity_rule"],
    "source_copy_cleanup": cleanup,
    "tetrahedron_count": tetrahedra,
    "thread_setting": 8,
})
gmsh.finalize()
