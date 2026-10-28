"""Regenerate the G0 acceptance evidence for issue #56, part D."""

from __future__ import annotations

import json
from pathlib import Path

from panel.p1_dirichlet import sphere_case
from panel.p1_g0_acceptance import jump_check, load_check, offset_potential_check, penetration_check

ROOT = Path(__file__).resolve().parents[3]


def main():
    payload = {"levels": [], "jump": jump_check(1), "jump_fine": jump_check(1, offset=1e-5)}
    for level in (1, 2, 3):
        sphere = sphere_case(level)
        payload["levels"].append({
            "level": level,
            "nodes": int(sphere["vertices"]),
            "panels": int(sphere["faces"]),
            "cp_rms": sphere["cp_rms"],
            "cp_max": sphere["cp_max"],
            "area_total": sphere["area_total"],
            "least_squares_residual": sphere["least_squares_residual"],
            "row_sum_mean": sphere["row_sum_mean"],
            "offset": offset_potential_check(level),
            "load": load_check(level),
            "penetration": penetration_check(level),
        })
        print("L%d done" % level, flush=True)
    out = ROOT / "docs" / "issue49_vspaero" / "p1" / "g0_acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
