"""Compare actual wall triangle sizes near a declared leading edge."""
import argparse
import json
from pathlib import Path

import numpy as np

from panel.mesh_feature_size import _distance, _polyline

parser = argparse.ArgumentParser()
parser.add_argument("--features", type=Path, required=True)
parser.add_argument("surfaces", nargs="+", type=Path)
args = parser.parse_args()
data = json.loads(args.features.read_text(encoding="utf-8"))
feature = next(item for item in data["features"] if item["name"] == "leading_edge")
line = _polyline(feature["points"], "leading_edge")
for path in args.surfaces:
    surface = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(surface["coordinates"])
    triangles = points[np.asarray(surface["faces"], dtype=int)]
    distance = np.array([_distance(p, line) for p in triangles.mean(axis=1)])
    near = distance < feature["radius_chords"] * surface["reference_chord"]
    edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).mean(axis=1)
    print(path, "wall_faces", len(triangles), "near_le", int(near.sum()),
          "near_mean_edge_m", float(edges[near].mean()))
