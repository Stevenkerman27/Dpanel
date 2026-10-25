"""Export actual triangles, unmodified cell Cp, component labels and wake jump."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
args.run = args.run.resolve()
result = json.loads((args.run / "result.json").read_text())
fixture = json.loads((args.run / "surface_input.json").read_text())
data = np.load(args.run / "fields.npz", allow_pickle=False)
chord, origin = result["references"]["chord"], np.asarray(result["references"]["origin"])
points = data["coordinates"] * chord + origin
wall_data = dict(Cp_raw=data["Cp_raw"], component_id=data["component_ids"],
                 pressure_force=data["panel_forces"] * chord**2,
                 pressure_moment=data["panel_moments"] * chord**3,
                 surface_velocity=data["surface_velocity"], mu=data["mu"] * chord,
                 panel_area=data["panel_area"] * chord**2)
if (args.run / "independent_field.npz").exists():
    independent = np.load(args.run / "independent_field.npz")
    wall_data.update(independent_normal_error=independent["normal_error"],
                     independent_Cp=independent["cp"])
for name, faces, fields in [
    ("surface", data["faces"], wall_data),
    ("wake", data["wake_faces"], dict(gamma_at_centroid=data["mu_wake"] * chord)),
]:
    target = args.run / (name + ".vtu")
    assert not target.exists(), "Preserve previous export"
    tree = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(tree, "UnstructuredGrid")
    piece = ET.SubElement(grid, "Piece", NumberOfPoints=str(len(points)), NumberOfCells=str(len(faces)))
    arrays = [(ET.SubElement(piece, "Points"), "Points", points)]
    cells = ET.SubElement(piece, "Cells")
    arrays += [(cells, "connectivity", faces.ravel()), (cells, "offsets", np.arange(1, len(faces) + 1) * 3),
               (cells, "types", np.full(len(faces), 5, dtype=int))]
    cell_data = ET.SubElement(piece, "CellData", Scalars="Cp_raw" if name == "surface" else "gamma_at_centroid")
    arrays += [(cell_data, key, value) for key, value in fields.items()]
    for parent, key, values in arrays:
        values = np.asarray(values)
        components = values.shape[1] if values.ndim == 2 else 1
        array = ET.SubElement(parent, "DataArray", type="Int64" if values.dtype.kind in "iu" else "Float64",
                              Name=key, NumberOfComponents=str(components), format="ascii")
        array.text = " ".join(map(str, values.ravel()))
    ET.ElementTree(tree).write(target, encoding="utf-8", xml_declaration=True)
    print(target)
