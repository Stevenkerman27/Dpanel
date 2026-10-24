"""Preserve VSP's exported triangles, component tags and trailing-edge node lines."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("geometry", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
lines = args.geometry.read_text().splitlines()
assert lines[0].strip() == "# vspgeom v3"
nodes, polygons, wakes = map(int, lines[2].split())
xyz = np.array([line.split() for line in lines[3:3+nodes]], dtype=float)
polygon_tags = np.array([int(line.split()[0]) for line in lines[4+nodes+polygons:4+nodes+2*polygons]])
tokens = iter(" ".join(lines[4+nodes+3*polygons:]).split())
wake_count = int(next(tokens))
assert wake_count == wakes
wake_edges, wake_part = [], []
for wake in range(wake_count):
    count, part = abs(int(next(tokens))), int(next(tokens))
    node_ids = np.array([int(next(tokens))-1 for node in range(count)])
    wake_edges.extend(zip(node_ids[:-1], node_ids[1:]))
    wake_part.extend([part]*(count-1))
tri, tags, parents = [], [], []
for polygon in range(polygons):
    polygon_id, triangle_count = int(next(tokens)), int(next(tokens))
    assert polygon_id == polygon+1
    for triangle in range(triangle_count):
        tri.append([int(next(tokens))-1 for corner in range(3)])
        tags.append(polygon_tags[polygon])
        parents.append(polygon_id)
settings = {}
for line in args.geometry.with_suffix(".vspaero").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=",1)
        if key.strip() in ["Sref", "Cref", "Bref", "AoA", "Beta"]:
            settings[key.strip()] = float(value.strip())
metadata = dict(source=str(args.geometry.resolve()),
                sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
                nodes=nodes, polygons=polygons, triangles=len(tri), wake_lines=wakes,
                note="Native exported triangulation; differs from VSPAERO polygon discretization",
                settings=settings)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, xyz=xyz, tri=np.asarray(tri,dtype=np.int64), tags=tags,
                    wake_edges=np.asarray(wake_edges,dtype=np.int64).reshape(-1,2), wake_part=wake_part,
                    parent_polygon=parents, sref=settings["Sref"], cref=settings["Cref"],
                    alpha=settings["AoA"], metadata_json=json.dumps(metadata))
print(json.dumps(metadata, indent=2))
