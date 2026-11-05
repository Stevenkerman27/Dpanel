"""Capture the exact half-mesh bank failure from an existing Web STEP input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from panel import cad_half
from panel.half_surface_mesh import _orient_open_half_wall

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", type=Path, required=True, help="Existing Web run containing wing.step and trailing_edge.json")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--size", type=float, default=.1)
parser.add_argument("--ncpu", type=int, default=1)
args = parser.parse_args()
original = cad_half.build_half_surface
command = [sys.executable, *sys.argv]
root = Path(__file__).resolve().parents[1]


def capture(points, wall_faces, wall_tags, wake_faces, **kwargs):
    np.savez_compressed(args.out / "half_mesh_input.npz", points=points, wall_faces=wall_faces,
                        wall_tags=wall_tags, wake_faces=wake_faces, **kwargs)
    x = np.asarray(points)
    te = np.asarray(kwargs["te_polyline"])
    tolerance = 1e-8*max(float(np.ptp(x, axis=0).max()), kwargs["reference_chord"], 1.)
    wall, _, _ = _orient_open_half_wall(x, wall_faces, tolerance)
    edges = {}
    for i, face in enumerate(wall):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    wake_edges = {}
    for i, face in enumerate(wake_faces):
        for a, b in zip(face, np.roll(face, -1)):
            wake_edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    records = []
    for edge, wake_ids in wake_edges.items():
        adjacent = edges.get(edge, [])
        v = x[list(edge)]
        if len(wake_ids) != 1 or len(adjacent) != 2:
            continue
        if (np.max(abs(v[:, 0]-np.interp(v[:, 1], te[:, 1], te[:, 0]))) > tolerance
                or np.max(abs(v[:, 2]-np.interp(v[:, 1], te[:, 1], te[:, 2]))) > tolerance):
            continue
        bank = x[wall[adjacent]]
        center = bank.mean(axis=1)
        z = center[:, 2]-np.interp(center[:, 1], te[:, 1], te[:, 2])
        wall_normal = np.cross(bank[:, 1]-bank[:, 0], bank[:, 2]-bank[:, 0])
        wall_normal /= np.linalg.norm(wall_normal, axis=1)[:, None]
        triangle = x[np.asarray(wake_faces)[wake_ids[0]]]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        normal *= np.sign(normal[2])/np.linalg.norm(normal)
        records.append(dict(edge=list(edge), xyz=v.tolist(), adjacent=adjacent,
                            centroid_height=z.tolist(), normal_dot_wake=(wall_normal @ normal).tolist(),
                            legacy_height_test=bool(z.min() < 0 and z.max() > 0)))
    receipt = dict(command=command, te_bank_rows=records,
                   legacy_rejected_edges=sum(not r["legacy_height_test"] for r in records),
                   input_sha256={name: hashlib.sha256((args.case/name).read_bytes()).hexdigest()
                                 for name in ("wing.step", "trailing_edge.json", "params.json")},
                   half_mesh_input_sha256=hashlib.sha256((args.out/"half_mesh_input.npz").read_bytes()).hexdigest(),
                   source_sha256=hashlib.sha256((root/"panel/half_surface_mesh.py").read_bytes()).hexdigest())
    (args.out/"bank_audit.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("TE bank audit: edges=%d rejected_by_height=%d" % (
        len(records), receipt["legacy_rejected_edges"]), flush=True)
    return original(points, wall_faces, wall_tags, wake_faces, **kwargs)


cad_half.build_half_surface = capture
declaration = json.loads((args.case/"trailing_edge.json").read_text(encoding="utf-8"))
chord = declaration["chord"]
reach = max(max(abs(p[0]), abs(p[1])) for p in declaration["points"])
size = args.size/chord
radius = 1.5*reach+4*size
sys.argv = ["panel.cad_half", "--wing-cad", str(args.case/"wing.step"),
            "--trailing-edge", str(args.case/"trailing_edge.json"), "--out", str(args.out),
            "--size", str(size), "--radius", str(radius), "--far-size", str(.3*radius),
            "--wake-length", "30", "--alpha-deg", "4", "--ncpu", str(args.ncpu)]
cad_half.main()
