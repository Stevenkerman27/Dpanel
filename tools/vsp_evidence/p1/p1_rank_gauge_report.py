"""Collect the rank and gauge evidence for issue #56 part C (research only).

Writes docs/issue49_vspaero/p1/rank_gauge.json. Nothing here is imported by the package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from panel.p1_dirichlet import assemble_p1_doublet
from panel.p1_g1 import load_case, mirror_block
from panel.p1_rank_gauge import constraint_effect, operator_stats, topology_report

ROOT = Path(__file__).resolve().parents[3]
SURFACE = Path(r"D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json")


def wing_case():
    case = load_case(SURFACE)
    vertices = case["vertices"]
    faces = case["wall_faces"]
    nodes = np.unique(faces)
    local = -np.ones(len(vertices), dtype=int)
    local[nodes] = np.arange(len(nodes))
    assembly = assemble_p1_doublet(vertices[nodes], local[faces])
    wall = assembly["matrix"]
    mirror = mirror_block(vertices, faces, assembly["centroids"])[:, nodes]
    rhs = -(assembly["centroids"] @ case["freestream"])
    return {
        "n_wall_panels": int(len(faces)),
        "n_wall_nodes": int(len(nodes)),
        "n_coordinates": int(len(vertices)),
        "wall_only": operator_stats(wall),
        "wall_minus_mirror": operator_stats(wall - mirror),
        "wall_plus_mirror": operator_stats(wall + mirror),
        "pin_probe": constraint_effect(wall - mirror, rhs, 0),
    }


def main():
    payload = {"topologies": topology_report(), "wing": wing_case()}
    out = ROOT / "docs" / "issue49_vspaero" / "p1" / "rank_gauge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", out)
    for name, stats in payload["topologies"].items():
        print("%-14s rows %d cols %d rank %d nullity %d sigma_min %.4g cst_rel %.3f" % (
            name, stats["n_rows"], stats["n_cols"], stats["rank"], stats["nullity"],
            stats["sigma_min"], stats["constant_response_relative"]))
    wing = payload["wing"]
    print("wing panels %d nodes %d" % (wing["n_wall_panels"], wing["n_wall_nodes"]))
    for name in ("wall_only", "wall_minus_mirror", "wall_plus_mirror"):
        stats = wing[name]
        print("%-18s rank %d nullity %d sigma_min %.4g rowsum %+.6f spread %.1e" % (
            name, stats["rank"], stats["nullity"], stats["sigma_min"],
            stats["row_sum_mean"], stats["row_sum_spread"]))
    print("pin: free %.6f pinned %.6f is_gauge %s" % (
        wing["pin_probe"]["residual_free"], wing["pin_probe"]["residual_pinned"],
        wing["pin_probe"]["is_gauge"]))


if __name__ == "__main__":
    main()
