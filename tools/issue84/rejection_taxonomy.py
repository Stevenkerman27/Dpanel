"""Issue #84 item B: map the 102 rejected G2 wake nodes onto the six requested classes.

The contract (PR #85 comment 6517332531) asks for the rejections to be split into
verifiable classes - sampling/epsilon unreliable, geometric degeneracy,
collision/self-intersection, root/attachment constraint, step/line-search failure,
other - with a count, representative node IDs and a minimal reproduction per class.

wake_node_rejection.py emits exactly one machine reason token per rejected node, so
this module is a projection, not a new opinion: it reads the per-node CSV, groups the
tokens, and additionally proves that the five non-sampling classes are empty by
checking the geometry they would have to violate (positive min area, three incident
faces, no root node, no near-coincident other-sheet face).

Run:  python tools/issue84/rejection_taxonomy.py --csv runs/issue84/wake_rejection_g2_analytic.csv
          --json runs/issue84/wake_rejection_g2_analytic.json --out docs/issue84/rejection_taxonomy.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = {
    "sampling_epsilon": "sampling/epsilon unreliable",
    "sampling_quadrature": "sampling/quadrature unreliable",
    "degenerate_geometry": "geometric degeneracy",
    "collision_or_selfintersection": "collision/self-intersection",
    "root_or_attachment": "root/attachment constraint",
    "step_or_linesearch": "step/line-search failure",
    "other": "other",
}

# tokens emitted by wake_node_rejection.py -> contract class
TOKEN_TO_CLASS = {
    "sampling_epsilon": "sampling_epsilon",
    "sampling_quadrature": "sampling_quadrature",
    "zero_speed": "other",
    "opposed_tangent": "other",
}


def _float(row, key):
    value = row.get(key)
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool(row, key):
    return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")


def load(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def taxonomy(rows):
    rejected = [r for r in rows if not _bool(r, "eligible")]
    classes = {name: [] for name in CLASSES}
    for row in rejected:
        text = str(row.get("reasons", "")).replace("[", "").replace("]", "").replace("'", "")
        tokens = [t.strip() for t in text.split(",") if t.strip()]
        cls = {TOKEN_TO_CLASS.get(t, "other") for t in tokens} or {"other"}
        for name in cls:
            classes[name].append(row)

    # evidence that the empty classes really are empty, not merely unlabelled
    area = [_float(r, "local_area_min") for r in rejected]
    faces = [_float(r, "incident_faces") for r in rejected]
    other = [_float(r, "dist_wake_other") for r in rejected]
    same = [_float(r, "dist_wake_same") for r in rejected]
    scale = [_float(r, "scale") for r in rejected]
    h = [_float(r, "h_over_scale") for r in rejected]
    checks = {
        "min_local_area_min": float(np.min(area)) if area else None,
        "min_incident_faces": int(np.min(faces)) if faces else None,
        "min_dist_wake_other": float(np.min(other)) if other else None,
        "min_dist_wake_same_over_scale": float(np.min(np.asarray(same) / np.asarray(scale))) if same and scale else None,
        "rejected_is_root_count": int(sum(_bool(r, "is_root") for r in rejected)),
        "rejected_on_sheet_boundary": int(sum(_bool(r, "on_sheet_boundary") for r in rejected)),
        "rejected_near_tip": int(sum(_bool(r, "near_tip") for r in rejected)),
        "rejected_h_over_scale_min": float(np.min(h)) if h else None,
    }

    summary = {}
    for name, members in classes.items():
        if not members:
            summary[name] = {"label": CLASSES[name], "count": 0, "examples": [],
                             "min_eps": None, "median_eps": None, "max_eps": None}
            continue
        eps = [(_float(r, "epsilon_error"), r) for r in members]
        eps = [(e, r) for e, r in eps if e is not None]
        eps.sort(key=lambda item: -item[0])
        examples = [
            {
                "group": r["group"],
                "node": int(r["node"]),
                "epsilon_error": e,
                "h_over_scale": _float(r, "h_over_scale"),
                "on_sheet_boundary": _bool(r, "on_sheet_boundary"),
                "near_tip": _bool(r, "near_tip"),
                "dist_wall": _float(r, "dist_wall"),
                "dist_sheet_edge": _float(r, "dist_sheet_edge"),
                "local_aspect_max": _float(r, "local_aspect_max"),
                "speed": _float(r, "speed"),
                "xyz": [_float(r, "xyz_x"), _float(r, "xyz_y"), _float(r, "xyz_z")],
            }
            for e, r in eps[:3]
        ]
        values = np.array([e for e, _ in eps])
        summary[name] = {
            "label": CLASSES[name],
            "count": len(members),
            "min_eps": float(values.min()) if len(values) else None,
            "median_eps": float(np.median(values)) if len(values) else None,
            "max_eps": float(values.max()) if len(values) else None,
            "examples": examples,
        }
    return {"rows": len(rows), "rejected": len(rejected), "classes": summary, "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load(args.csv)
    report = taxonomy(rows)
    if args.json and args.json.exists():
        registration = json.load(open(args.json, encoding="utf-8"))
        report["registration"] = {
            key: registration.get(key)
            for key in ("velocity_method", "quadrature_order", "core_radius", "eps_fraction",
                        "epsilon_tol", "quadrature_tol", "seconds", "wall_faces", "case", "alpha_deg")
        }
    report["provenance"] = {"csv": str(args.csv), "json": str(args.json) if args.json else None}

    print(json.dumps(report, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
