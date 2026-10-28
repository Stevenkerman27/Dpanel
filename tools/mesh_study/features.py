"""Generate dpanel.mesh_features.v1 declarations for the elliptic test case.

The points follow the same section formulas that generated the CAD
(runs/geom_robust/make_cases.py): chord(y) = tip + (root - tip) * sqrt(1 - (y/L)^2),
leading edge x = le_offset * (root - chord). Only the *features* are taken from
that formula; nothing here guesses the mesh.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(r"D:\3D\Projects\panel")


def sections(span=2.0, root_chord=1.0, tip_chord=0.05, le_offset=0.25, count=21):
    rows = []
    for i in range(count):
        y = span * i / (count - 1)
        f = min(abs(y) / span, 1.0)
        c = tip_chord + (root_chord - tip_chord) * math.sqrt(max(0.0, 1.0 - f * f))
        rows.append((y, le_offset * (root_chord - c), c))
    return rows


def build(bands, ratio, radius):
    rows = sections()
    leading = [[le, y, 0.0] for y, le, _ in rows]
    trailing = [[le + c, y, 0.0] for y, le, c in rows]
    features = []
    if "le" in bands:
        features.append(dict(name="leading_edge", target_ratio=ratio,
                             radius_chords=radius, points=leading))
    if "te" in bands:
        features.append(dict(name="trailing_edge", target_ratio=ratio,
                             radius_chords=radius, points=trailing))
    return dict(
        schema="dpanel.mesh_features.v1",
        units="m",
        features=features,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=("le", "te", "lete"), required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = build(args.bands, args.ratio, args.radius)
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("WROTE", args.out, "features", [f["name"] for f in payload["features"]],
          "ratio", args.ratio, "radius", args.radius)


if __name__ == "__main__":
    main()
