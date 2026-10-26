"""Recompute stencil-conditioning and neighbour-growth metrics for existing cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\3D\Projects\panel")
PR43 = ROOT / "runs" / "pr43_git"
sys.path.insert(0, str(PR43))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from panel.run_half_native import load_half_surface  # noqa: E402

CASES = ROOT / "runs" / "mesh_study" / "cases"


def growth(surface_path):
    data = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    x = np.asarray(data["coordinates"], float)
    f = np.asarray(data["faces"], np.int64)
    tri = x[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    owner = np.tile(np.arange(len(f)), 3)
    key = np.sort(edges, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    left, right = owner[:-1][same], owner[1:][same]
    if left.size == 0:
        return dict(neighbour_area_ratio_p99=None, neighbour_area_ratio_max=None)
    lo = np.minimum(area[left], area[right])
    hi = np.maximum(area[left], area[right])
    ratio = hi / np.maximum(lo, 1e-300)
    return dict(
        neighbour_area_ratio_median=float(np.median(ratio)),
        neighbour_area_ratio_p99=float(np.percentile(ratio, 99)),
        neighbour_area_ratio_max=float(ratio.max()),
        neighbour_pairs=int(left.size),
    )


def stencil(surface_path):
    _, audit = load_half_surface(
        surface_path, stencil_rings=2, max_stencil_rings=6, return_stencil_audit=True
    )
    initial = np.asarray(audit["initial_singular_ratios"], float)
    final = np.asarray(audit["final_singular_ratios"], float)
    depth = np.asarray(audit["selected_ring_depth"], int)
    return dict(
        initial_ratio_min=float(initial.min()),
        initial_ratio_p01=float(np.percentile(initial, 1)),
        initial_ratio_median=float(np.median(initial)),
        final_ratio_min=float(final.min()),
        faces_below_1e_3=int((initial < 1e-3).sum()),
        faces_below_1e_6=int((initial < 1e-6).sum()),
        max_ring_depth=int(depth.max()),
        ring_depth_histogram={str(int(k)): int(v) for k, v in zip(*np.unique(depth, return_counts=True))},
        expanded_faces=len(audit["expanded_faces"]),
        initial_rank_deficient_faces=[int(i) for i in audit["initial_rank_deficient_faces"]],
        pressure_neighbor_width=int(audit.get("pressure_neighbor_width", 0)),
    )


def main():
    tags = sys.argv[1:] or sorted(p.name for p in CASES.iterdir() if (p / "case.json").is_file())
    for tag in tags:
        case_json = CASES / tag / "case.json"
        if not case_json.is_file():
            print("SKIP", tag)
            continue
        row = json.loads(case_json.read_text(encoding="utf-8"))
        surface = CASES / tag / "surface.json"
        row["mesh"].update(growth(surface))
        row["solve"]["stencil_condition"] = stencil(surface)
        case_json.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
        sc = row["solve"]["stencil_condition"]
        print("%-12s walls=%5d ratio_min=%.2e p01=%.2e below1e-3=%d depth=%d growth_p99=%s" % (
            tag, row["mesh"]["wall_faces"], sc["initial_ratio_min"], sc["initial_ratio_p01"],
            sc["faces_below_1e_3"], sc["max_ring_depth"],
            ("%.2f" % row["mesh"]["neighbour_area_ratio_p99"]) if row["mesh"].get("neighbour_area_ratio_p99") else "n/a"))


if __name__ == "__main__":
    main()