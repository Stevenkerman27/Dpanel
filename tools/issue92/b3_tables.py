"""Render the markdown tables for docs/ISSUE92_B3_VSPAERO_PROTOCOL.md."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
sys.path.insert(0, str(ROOT))

MATRIX = ROOT / "docs/issue92/b3_matrix.json"
CASES = ROOT / "docs/issue92/b3_cases.json"
OUT = ROOT / "runs/benchmarks/b3_vspaero/b3_tables.md"

CASE_ORDER = ("rect", "swept30", "elliptic", "bwb")


def _f(v, fmt="%+.5f"):
    return "-" if v is None else fmt % v


def main():
    report = json.loads(MATRIX.read_text(encoding="utf-8"))
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    lines = []

    lines.append("### Lift and drag fits, per case")
    lines.append("")
    lines.append("| case | stations | VSPAERO panels (coarse/fine, panel; vlm) | DPanel CL | VSPAERO panel CLwtot | VLM CLi |")
    lines.append("|---|---|---|---|---|---|")
    for name in CASE_ORDER:
        entry = report["cases"][name]
        spec = cases[name]
        vsp = entry["vspaero"]
        panels = "%d / %d" % (vsp["panel/coarse"]["panels"], vsp["panel/fine"]["panels"])
        vlm = "%d / %d" % (vsp["vlm/coarse"]["panels"], vsp["vlm/fine"]["panels"])
        dp_cl = entry["dpanel"]["CL"]["fit"]
        pc = vsp["panel/coarse"]["CLwtot"]["fit"]
        pfi = vsp["panel/fine"]["CLwtot"]["fit"]
        vc = vsp["vlm/coarse"]["CLi"]["fit"]
        vf = vsp["vlm/fine"]["CLi"]["fit"]
        lines.append("| %s | %d | %s; %s | %+.5f (%+.5f) | %+.5f / %+.5f | %+.5f / %+.5f |" % (
            name, len(spec["stations"]), panels, vlm,
            dp_cl["slope_per_deg"], dp_cl["intercept"],
            pc["slope_per_deg"], pfi["slope_per_deg"],
            vc["slope_per_deg"], vf["slope_per_deg"]))
    lines.append("")
    lines.append("Slope in CL per degree; the bracketed number is the zero-incidence")
    lines.append("intercept (large only where the section carries camber).")
    lines.append("")

    lines.append("### Gate, all 32 comparisons")
    lines.append("")
    lines.append("| case | pair | quantity | value | threshold | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for name in CASE_ORDER:
        entry = report["cases"][name]
        for pair, comp in entry["comparisons"].items():
            if "slope_relative" in comp:
                lines.append("| %s | %s | CL slope | %.4f | 0.15 | %s |" % (
                    name, pair, comp["slope_relative"], "pass" if comp["passed"] else "FAIL"))
            elif "rms_relative" in comp:
                lines.append("| %s | %s | circulation shape | %.4f | 0.15 | %s |" % (
                    name, pair, comp["rms_relative"], "pass" if comp["passed"] else "FAIL"))
    lines.append("")
    lines.append("| case | cambered | camber lift offset (VSPAERO - DPanel) | gated |")
    lines.append("|---|---|---|---|")
    for name in CASE_ORDER:
        entry = report["cases"][name]
        offsets = [c["camber_lift_offset"] for p, c in entry["comparisons"].items()
                   if p.startswith("panel/fine") and "camber_lift_offset" in c]
        lines.append("| %s | %s | %+.4f | no (camber omitted in the rebuild) |" % (
            name, "yes" if entry["cambered"] else "no", offsets[0] if offsets else float("nan")))
    lines.append("")

    lines.append("### Induced drag, reported and not gated")
    lines.append("")
    lines.append("| case | DPanel CD_trefftz(4 deg) | VLM CDi(4 deg) | CD ratio | k = CDi/CL^2 (DPanel) | k (VLM) |")
    lines.append("|---|---|---|---|---|---|")
    for name in CASE_ORDER:
        entry = report["cases"][name]
        for tag, cd in entry["cd_report"].items():
            lines.append("| %s | %+.5f | %+.5f | %.2f | %.4f | %.4f |" % (
                name + " " + tag, cd["cd_dpanel"], cd["cd_vlm"], cd["cd_relative"],
                cd["k_dpanel"], cd["k_vlm"]))
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, len(lines), "lines")


if __name__ == "__main__":
    main()
