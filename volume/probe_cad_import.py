"""Bounded OCC import experiments; preserve geometry, mass and meshing evidence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"audit"/"runtime"/"python"))
import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("--cad", type=Path, required=True)
parser.add_argument("--variant", choices=["bounds", "heal", "sew", "split"], required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists()
args.out.mkdir(parents=True)
gmsh.initialize()
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
gmsh.model.add("import_probe")
occ = gmsh.model.occ
solids = occ.importShapes(str(args.cad.resolve()), highestDimOnly=True)
solids = [entity for entity in solids if entity[0] == 3]
occ.synchronize()
result = dict(variant=args.variant, imported_mass=sum(occ.getMass(*entity) for entity in solids),
              imported_boxes=[gmsh.model.getBoundingBox(*entity) for entity in solids])
if args.variant in ["heal", "sew"]:
    healed = occ.healShapes(solids, tolerance=1e-8, fixDegenerated=True,
                            fixSmallEdges=False, fixSmallFaces=False,
                            sewFaces=args.variant == "sew", makeSolids=args.variant == "sew")
    solids = [entity for entity in healed if entity[0] == 3]
    occ.synchronize()
    result["healed_solids"] = len(solids)
boxes = np.array([gmsh.model.getBoundingBox(*entity) for entity in solids])
lo, hi = boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)
origin, length = (lo+hi)/2., float(max(hi-lo))
result.update(native_origin=origin.tolist(), native_length=length)
occ.translate(solids, *(-origin))
occ.dilate(solids, 0., 0., 0., *np.full(3, 1./length))
occ.synchronize()
result["normalized_mass"] = sum(occ.getMass(*entity) for entity in solids)
if args.variant == "split":
    plane_points = [occ.addPoint(0., y, z) for y, z in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]]
    curves = [occ.addLine(plane_points[i], plane_points[(i+1) % 4]) for i in range(4)]
    plane = occ.addPlaneSurface([occ.addWire(curves)])
    pieces, _ = occ.fragment(solids, [(2, plane)])
    pieces = [entity for entity in pieces if entity[0] == 3]
    solids, _ = occ.fuse(pieces[:1], pieces[1:], removeObject=True, removeTool=True)
    solids = [entity for entity in solids if entity[0] == 3]
outer = occ.addBox(-5., -5., -5., 10., 10., 10.)
fluid, _ = occ.cut([(3, outer)], solids)
occ.synchronize()
result["fluid_mass"] = sum(occ.getMass(*entity) for entity in fluid if entity[0] == 3)
gmsh.option.setNumber("Mesh.MeshSizeMin", .1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.5)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
try:
    gmsh.model.mesh.generate(3)
except Exception as exception:
    result.update(status="failed_mesh", message=str(exception))
    (args.out/"failure.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    gmsh.finalize()
    raise
result["tetrahedra"] = len(gmsh.model.mesh.getElementsByType(4)[0])
result["surface_elements"] = {str(tag): len(gmsh.model.mesh.getElementsByType(2, tag)[0])
                              for dim, tag in gmsh.model.getEntities(2)}
gmsh.write(str(args.out/"geometry.brep"))
gmsh.finalize()
(args.out/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
