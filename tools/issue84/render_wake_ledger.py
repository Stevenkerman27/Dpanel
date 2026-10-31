"""Compress a G2 wake-ledger run into a small review artefact.

The full ledger has one row per (iteration, group, node) and is tens of
megabytes; the repository keeps only the counters, the per-iteration ID sets,
the set intersections, the update records and the worst representative nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _stats(values):
    if not values:
        return None
    a = np.asarray(values, float)
    return dict(count=int(len(a)), min=float(a.min()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), max=float(a.max()))


def summarize(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {key: payload[key] for key in ("schema", "python", "numpy", "case",
                                        "alpha_deg", "kutta", "iterations",
                                        "tolerances", "status", "converged",
                                        "physical_accuracy_accepted", "wall_faces",
                                        "seconds", "counter_definitions", "increments_at")}
    out["iterations"] = []
    for entry in payload["iterations_out"]:
        rows = entry["rows"]
        summary = dict(iteration=entry["iteration"], counters=entry["counters"],
                       map_residual_chord=entry["map_residual_chord"],
                       kinematic_candidate=entry["kinematic_candidate"],
                       outgoing_update=entry["outgoing_update"],
                       trial_reasons=entry["trial_reasons"],
                       intersections=entry["intersections"], groups={})
        digest = hashlib.sha256()
        for group, sets in entry["sets"].items():
            group_rows = [r for r in rows if r["group"] == group]
            free = set(sets["free"])
            convergence = set(sets["convergence"])
            free_rows = [r for r in group_rows if r["node"] in free]
            worst = sorted(free_rows, key=lambda r: -r["epsilon_error"])[:8]
            boundary = sum(1 for r in free_rows
                           if r["node"] in convergence and r["incident_faces"] < 6)
            summary["groups"][group] = dict(
                total_nodes=len(group_rows), root=len(sets["roots"]), free=len(free),
                eligible=len(sets["eligible"]), convergence=len(convergence),
                speed=len(sets["speed"]), tangent=len(sets["tangent"]),
                rejected=len(sets["rejected"]),
                roots_and_convergence=len(set(sets["roots"]) & convergence),
                convergence_boundary=boundary,
                convergence_interior=len(convergence) - boundary,
                epsilon_error=_stats([r["epsilon_error"] for r in free_rows]),
                displacement=_stats([r["displacement"] for r in free_rows]),
                wall_distance=_stats([r["wall_distance"] for r in group_rows]),
                wall_contact_nodes=int(sum(1 for r in group_rows
                                          if r["wall_distance"] < 1e-9)),
                worst_epsilon=[dict(node=r["node"], epsilon_error=r["epsilon_error"],
                                    incident_faces=r["incident_faces"],
                                    incident_aspect_max=r["incident_aspect_max"],
                                    wall_distance=r["wall_distance"],
                                    eligible=r["eligible"],
                                    xyz=r["xyz"],
                                    displacement=r["displacement"]) for r in worst])
            digest.update(json.dumps(sets, sort_keys=True).encode())
        summary["sets_sha256"] = digest.hexdigest()
        out["iterations"].append(summary)
    first = payload["iterations_out"][0]
    out["convergence_ids_first_iteration"] = {name: sets["convergence"]
                                              for name, sets in first["sets"].items()}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="label:path pairs; one entry per solved case")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = dict(schema="dpanel.issue84.wake_ledger_summary.v1", runs={})
    for item in args.run:
        label, _, path = item.partition(":")
        payload["runs"][label] = summarize(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
