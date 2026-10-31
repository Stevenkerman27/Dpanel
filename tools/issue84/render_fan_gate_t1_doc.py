"""Render docs/ISSUE84_FAN_QUADRATURE_GATE.md from the T1 acceptance JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FENCE = "%%FENCE%%"


def _code(text):
    return text


def _fmt(value, digits=3):
    if isinstance(value, float):
        return ("%." + str(digits) + "e") % value
    return str(value)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render(payload, title="Issue 84 task 1: the projection-fan quadrature gate"):
    production = payload["production"]
    refinement = payload["refinement"]
    summary = payload.get("summary", {})
    regressions = payload.get("regressions", {})
    lines = []
    add = lines.append
    add("# " + title)
    add("")
    add("Status: opt-in. The default velocity method remains analytic_edges and no")
    add("physical accuracy claim is made here (physical_accuracy_accepted=false).")
    add("")
    add("## 1. Defect")
    add("")
    add("With area_quadrature='projection_fan', RegularizedP1SheetField._area_velocity")
    add("ignored the quadrature order that limits() passed in and always used its own")
    add("fan_order/fan_strips. The gate therefore compared one configuration with")
    add("itself: quadrature_error was identically zero and the epsilon ladder never")
    add("refined the area integral. The epsilon channel still worked, so the fan path was")
    add("not obviously broken; only the quadrature channel was dead.")
    add("")
    add("## 2. Fix")
    add("")
    add("- RegularizedP1SheetField now owns fan_refinement_order and")
    add("  fan_refinement_strips (default max(48, 2*fan_order) and the base strips).")
    add("- quadrature_arguments(order) returns the explicit (base, refined) fan")
    add("  configurations; quadrature_plan(order) returns a receipt-ready description.")
    add("- limits() compares those two configurations, raises when they are equal, and")
    add("  records the plan. A self-comparison is now an error, never a silent pass.")
    add("- The receipt stores sampling_quadrature_base, sampling_quadrature_refined,")
    add("  sampling_fan_refinement_order, sampling_fan_refinement_strips and")
    add("  quadrature_plan; replay_snapshot restores the recorded refinement and its")
    add("  reproduce_recorded_samples mode requires the refinement to match.")
    add("- analytic_edges keeps reporting quadrature_error = 0 with an explicit")
    add("  not-applicable note, and legacy_duffy still compares order N with 2N.")
    add("")
    add("## 3. Tests")
    add("")
    add("test/test_p1_fan_quadrature_gate.py (8 tests) locks the contract:")
    add("")
    add("1. the plan records base and refined fan configurations;")
    add("2. the default refinement is at least 48 and differs from the base;")
    add("3. an identical pair is rejected at construction;")
    add("4. a coarse base (4:2 against 48:2) is flagged above 1e-3 while a converged base")
    add("   (32:2 against 48:2) stays below 1e-6;")
    add("5. a field whose two configurations are equal makes limits() raise;")
    add("6. analytic_edges reports zero and finite epsilon errors;")
    add("7. the legacy path still equals the directly computed N-versus-2N difference;")
    add("8. scalar and batched fan refinement agree bitwise.")
    add("")
    add("## 4. Frozen 109-point acceptance")
    add("")
    add("Geometry: %s case, alpha %.1f deg, core radius %.3g, %d wall faces, wake faces %s."
        % (payload["case"], payload["alpha_deg"], payload["core_radius"],
           payload["wall_faces"], payload["wake_faces"]))
    add("Each frozen point uses the normal and scale of its nearest wake face; the")
    add("production candidate is %d:%d against %d:%d."
        % (production["order"], production["strips"], refinement["order"],
           refinement["strips"]))
    add("")
    rows = []
    for key, label in (("prod", "production %d:%d" % (production["order"], production["strips"])),
                       ("coarse", "coarse %d:%d" % (payload["coarse"]["order"],
                                                    payload["coarse"]["strips"])),
                       ("legacy", "legacy Duffy 48 vs 96")):
        stats = summary.get(key, {})
        rows.append([label, stats.get("count", 0), _fmt(stats.get("median")),
                     _fmt(stats.get("p90")), _fmt(stats.get("max"))])
    if summary:
        add(_table(["configuration", "points", "median", "p90", "max"], rows))
        add("")
        add("Points above the %.3g gate: production %d, coarse %d, of %d."
            % (payload["tolerance"], summary.get("prod_flagged", 0),
               summary.get("coarse_flagged", 0), summary.get("point_count", 0)))
        add("")
    truth_rows = [[row["index"], _fmt(row["true_error_vs_reference"]),
                   _fmt(row["prod_quadrature_error"]), _fmt(row["coarse_quadrature_error"])]
                  for row in payload.get("truth", [])]
    if truth_rows:
        add("### 4.1 Gate estimate against a saturated reference")
        add("")
        add("The reference is fan %d:%d with %d:%d as refinement, evaluated on the same"
            % (payload["reference"]["order"], payload["reference"]["strips"],
               refinement["order"], refinement["strips"]))
        add("targets as the base configuration.")
        add("")
        add(_table(["point", "true error", "gate estimate", "coarse estimate"], truth_rows))
        add("")
        add("True error statistics: %s." % json.dumps(summary.get("truth", {}), sort_keys=True))
        add("")
    target_rows = [[row["index"], row["kind"], _fmt(row["scale"]),
                    _fmt(row["prod_quadrature_error"]), _fmt(row["seconds"], 2)]
                   for row in payload.get("targets", [])]
    if target_rows:
        add("## 5. Real B2 target sample")
        add("")
        add("Face centroids and free nodes drawn with a fixed seed from the same frozen")
        add("state, measured with the production gate:")
        add("")
        add(_table(["index", "kind", "scale", "gate estimate", "seconds"], target_rows))
        add("")
        add("Target statistics: %s." % json.dumps(summary.get("targets", {}), sort_keys=True))
        add("")
    add("## 6. Regressions")
    add("")
    add("- analytic_edges: maximum quadrature error %s (must be exactly 0)."
        % _fmt(regressions.get("analytic_quadrature_error_max")))
    add("- legacy_duffy: the gate value equals the directly computed N-versus-2N")
    add("  difference to %s (bitwise equality expected)."
        % _fmt(regressions.get("legacy_gate_matches_direct_difference")))
    add("")
    add("## 7. Reproduce")
    add("")
    add(FENCE)
    add("python -m tools.issue84.fan_quadrature_gate \\")
    add("  --main-half-surface runs/half_cad_wingbody_v1/surface.json \\")
    add("  --production %d:%d --refinement %d:%d --coarse %d:%d --reference %d:%d"
        % (production["order"], production["strips"], refinement["order"],
           refinement["strips"], payload["coarse"]["order"], payload["coarse"]["strips"],
           payload["reference"]["order"], payload["reference"]["strips"]))
    add(FENCE)
    add("")
    add("Artifact: docs/issue84/fan_quadrature_gate.json.")
    add("")
    add("Wall clock: %.1f s total, %.1f s of it the frozen pressure solve."
        % (payload.get("total_seconds", 0.), payload.get("solve_seconds", 0.)))
    add("")
    return "\n".join(line.replace(FENCE, "~~~") for line in lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path,
                        default=Path("docs/issue84/fan_quadrature_gate.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/ISSUE84_FAN_QUADRATURE_GATE.md"))
    args = parser.parse_args(argv)
    payload = json.loads(args.gate.read_text(encoding="utf-8"))
    args.out.write_text(render(payload) + "\n", encoding="utf-8")
    print("wrote %s (%d bytes)" % (args.out, args.out.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
