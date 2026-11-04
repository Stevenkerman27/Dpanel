"""What still has to move before panel/ can run without a NumPy solver (Route B, S9).

Scans every module under panel/ for LAPACK-grade calls (np.linalg.svd/qr/solve/lstsq/
eig/eigh/pinv/matrix_rank, scipy.linalg, scipy.optimize.least_squares) and pairs each
hit with the torch counterpart that already exists, if any. np.linalg.norm and plain
array algebra are not LAPACK and are not listed: they cannot crash a torch process.

Usage:
    python -m tools.issue92.retirement_inventory [--out docs/issue92/s9_inventory.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "panel"

PATTERN = re.compile(
    r"np\.linalg\.(svd|svdvals|qr|solve|lstsq|eig|eigh|pinv|matrix_rank|cholesky)\b"
    r"|scipy\.linalg|scipy\.optimize\.least_squares|from scipy\.linalg import")

# Curated disposition. Everything not listed here is either plumbing with no LAPACK or a
# module the scan found nothing in; the scan is authoritative for the site list.
DISPOSITION = {
    "p1_rank_aware": ("solver", "p1_rank_aware_torch", "delete; null_space/lstsq/constrained_solve are duplicated by the torch module"),
    "p1_pressure_kutta_reduced": ("solver", "p1_pressure_kutta_torch + p1_pressure_kutta_torch_driver", "delete; structured QR/solve_triangular/OptimizeResult are duplicated"),
    "p1_lifting_g1": ("entry", "p1_solve_torch.solve_single_torch", "delete the NumPy solve branch and the trf call; keep assemble/mesh plumbing"),
    "p1_lifting_g2": ("entry", "p1_solve_torch.solve_multi_torch", "same as p1_lifting_g1"),
    "p1_closed_body": ("variant", "none", "closed-body Dirichlet demo (panel/run_p1_closed.py); not on the coupled path, keep until its own slice"),
    "p1_closed_jax": ("variant", "jax counterpart of p1_closed_body", "not on the coupled path"),
    "p1_geometry_contact": ("gate", "torch not needed", "SVD only for point sets larger than a triangle; the gate passes triangles and takes the cross product"),
    "p1_coupled_material_wake": ("loop", "torch branch at :1085", "constraint spectrum already taken in torch when solve_backend='torch'"),
}


def scan():
    rows = []
    for path in sorted(PANEL.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        sites = []
        for number, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                sites.append(dict(line=number, text=line.strip()[:140]))
        if not sites:
            continue
        role, counterpart, action = DISPOSITION.get(
            path.stem, ("unclassified", "none", "review"))
        rows.append(dict(module="panel/%s.py" % path.stem, role=role,
                         sites=sites, count=len(sites), torch_counterpart=counterpart,
                         action=action))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = scan()
    report = dict(schema="dpanel.issue92.s9.inventory.v1", root=str(ROOT),
                  modules=len(rows), sites=sum(row["count"] for row in rows), rows=rows)
    for row in rows:
        print("%-38s %-6s %2d  %-42s %s" % (row["module"], row["role"], row["count"],
                                            row["torch_counterpart"], row["action"]))
    print("modules=%d sites=%d" % (report["modules"], report["sites"]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
