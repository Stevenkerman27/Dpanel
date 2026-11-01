"""Render docs/ISSUE84_PROJECTION_DUFFY.md from the bench, gate-set and probe JSON."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "docs/issue84/projection_duffy_bench.json"
GATE = ROOT / "docs/issue84/projection_gate_set.json"
PROBE = ROOT / "docs/issue84/worst_node_probe.json"
OUT = ROOT / "docs/ISSUE84_PROJECTION_DUFFY.md"

STRATS = [("fan", 16, 1), ("fan", 16, 8), ("fan", 32, 8), ("fan", 48, 64),
          ("legacy", 16, 1), ("legacy", 32, 1), ("legacy", 48, 1), ("legacy", 96, 1)]


def fmt(value):
    if value is None:
        return "-"
    if value == 0.0:
        return "0"
    return f"{value:.2e}"


def bench_table(bench):
    head = "| aspect | core/d | ref n_sub | ref self-change | " + " | ".join(
        (("fan" if s == "fan" else "old") + f" o{o}/s{st}") for s, o, st in STRATS) + " |"
    lines = [head, "|" + "---|" * (4 + len(STRATS))]
    for case in bench["cases"]:
        errs = {}
        for entry in case["strategies"]:
            errs[(entry["solve"], entry["order"], entry["strips"])] = entry["rel_error"]
        cells = " | ".join(fmt(errs.get(key)) for key in STRATS)
        lines.append(f"| {case['aspect']:g} | {case['core_over_d']:g} | {case['n_sub_reference']} | "
                     f"{case['reference_self_change']:.1e} | {cells} |")
    return "\n".join(lines)


def gate_section():
    if not GATE.exists():
        return ["The frozen acceptance set has not been evaluated yet."]
    data = json.loads(GATE.read_text(encoding="utf-8"))
    lines = [f"case {data['case']} at alpha {data['alpha_deg']:g} deg, core {data['core_radius']}, "
             f"field {data['field_faces']} triangles, old order {data['legacy_order']}, "
             f"fan orders {data['fan_orders']} strips {data['strips']}, solve "
             f"{data['solve_seconds']:.1f} s, whole set {data['total_seconds']:.1f} s.",
             "",
             "| set | points | projection class | max old-vs-fan | median old-vs-fan | max aspect | old s/point | fan s/point |",
             "|---|---|---|---|---|---|---|---|"]
    for label, summary in sorted(data["sets"].items()):
        if not label.endswith("__summary"):
            continue
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(summary["projection_kinds"].items()))
        lines.append(f"| {label[:-9]} | {summary['points']} | {kinds} | "
                     f"{fmt(summary['legacy_vs_fan_rel_max'])} | "
                     f"{fmt(summary['legacy_vs_fan_rel_median'])} | {summary['aspect_max']:.1f} | "
                     f"{summary['legacy_seconds_median']:.2f} | {summary['fan_seconds_median']:.2f} |")
    return lines


def probe_section():
    if not PROBE.exists():
        return ["The worst-node convergence probe has not been run."]
    rows = json.loads(PROBE.read_text(encoding="utf-8"))
    lines = ["The highest fan order available here, order 48 with strips 32, is the reference;",
             "every entry is |v - v_ref| / |v_ref| of the total velocity vector at that node.",
             "",
             "| node | old o48 | old o96 | fan o16/s8 | fan o16/s32 | fan o16/s128 | fan o32/s8 | fan o32/s32 | fan o32/s32 swap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for entry in rows:
        ref = np.asarray(entry["fan"]["o48_s32_swap0"]["v"], float)
        def rel(key):
            return np.linalg.norm(np.asarray(entry["fan"][key]["v"], float) - ref) / np.linalg.norm(ref)
        def rel_old(key):
            return np.linalg.norm(np.asarray(entry["legacy"][key]["v"], float) - ref) / np.linalg.norm(ref)
        lines.append(f"| {entry['group']} {entry['node']} | {fmt(rel_old('48'))} | {fmt(rel_old('96'))} | "
                     f"{fmt(rel('o16_s8_swap0'))} | {fmt(rel('o16_s32_swap0'))} | "
                     f"{fmt(rel('o16_s128_swap0'))} | {fmt(rel('o32_s8_swap0'))} | "
                     f"{fmt(rel('o32_s32_swap0'))} | {fmt(rel('o32_s32_swap1'))} |")
    return lines


def main():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    parts = []
    add = parts.append
    add("# Projection-aware Duffy quadrature for the regularized P1 area kernel")
    add("")
    add("Issue #84, PR #85, step C3. Status: opt-in module plus measurements. The")
    add("production path is unchanged, the default velocity method is still")
    add("`analytic_edges`, and `physical_accuracy_accepted` stays false.")
    add("")
    add("## 1. Why the shipped map stalls")
    add("")
    add("The regularized area term is")
    add("")
    add("    I_f(p) = int_{T_f} (p - x) / (|p - x|^2 + core^2)^{3/2} dS")
    add("")
    add("and `panel/p1_regularized_sheet_velocity.py:135-150` evaluates it with one fixed")
    add("Duffy map per triangle: `bary = ((1-u)(1-v), u, (1-u)v)` with weights")
    add("`w_u w_v (1-u)`. Two geometric situations break it, and both are common in the")
    add("G2 wake sheet:")
    add("")
    add("1. the target is much closer to the triangle than the triangle is large, so the")
    add("   integrable peak sits at a parametric corner that is not the collapsed vertex;")
    add("2. the triangle is a sliver (main-wing wake aspect reaches 260 to 1405.7), so the")
    add("   collapse direction is nearly parallel to the long edge and the peak is smeared")
    add("   along the short one.")
    add("")
    add("The measured consequence (`docs/ISSUE84_RM_BENCH.md`) is that raising the order")
    add("does not converge: order 32 can be worse than order 16. That is aliasing, not")
    add("truncation error, and it is why C3 asked for a projection-aware decomposition.")
    add("")
    add("## 2. Method")
    add("")
    add("`panel/p1_projection_duffy.py` keeps the kernel and changes only the domain")
    add("decomposition (`closest_point_on_triangle` is the Ericson region test):")
    add("")
    add("1. project p onto the triangle and classify the closest point q as interior, edge")
    add("   or vertex;")
    add("2. fan the triangle into sub-triangles that all have q as their first vertex")
    add("   (3 interior, 2 edge, 1 vertex), optionally cut each sector into `strips` pieces")
    add("   along its far edge, and optionally `swap` the far edge to test the other")
    add("   integration direction;")
    add("3. integrate every sub-triangle with a Duffy map collapsed at q:")
    add("   `x = q + u a + u v (b - a)`, `|dS| = |a x (b - a)| u du dv`, full Jacobian, no")
    add("   empirical weights.")
    add("")
    add("Invariants the tests pin down: sub-areas sum to the parent area and the pieces do")
    add("not overlap (random interior samples hit exactly one piece), a constant integrand")
    add("integrates to the exact area, the result is covariant under rotation, translation")
    add("and scaling, and all four projection branches are exercised.")
    add("")
    add("## 3. Single-triangle benchmark against a composite oracle")
    add("")
    add("`tools/issue84/projection_duffy_bench.py` builds an independent oracle: the parent")
    add("triangle is cut into a barycentric grid of n^2 sub-triangles (`uniform_subdivide`),")
    add("each integrated with its own Duffy map, and the reference is the sum over the grid.")
    ref = bench['reference']
    add('The reference uses n = %d, order %d (%d sub-triangles per case); the'     % (ref['n'], ref['order'], bench['cases'][0]['n_sub_reference']))
    add("`ref self-change` column is |ref(n=32, order 24) - ref(n=64, order 32)| / |ref|, so")
    add("a value near 1e-15 means the reference is itself converged. Errors are relative")
    add("errors of the whole induced area vector `cross(gamma, I)`, not of one scalar")
    add("component.")
    add("")
    add("Target: centroid of the triangle raised by d = 0.01; aspect = longest edge over")
    add("height; core/d in {0.1, 0.3, 1, 3}. `old` is the shipped single-Duffy map.")
    add("")
    add(bench_table(bench))
    add("")
    add("Reading the table:")
    add("")
    add("- the fan at order 16 with strips 8 is already 1e3 to 1e5 better than the shipped")
    add("  map at the same order, and strips 64 buys nothing over strips 8 (the peak is at")
    add("  an apex, so strip refinement saturates);")
    add("- after the fan the error is order limited and monotone: order 32 improves by about")
    add("  1e4 over order 16, order 48 reaches 1e-8 and better;")
    add("- the shipped map is still at 1e-1 or worse at order 96 for aspect >= 260 with")
    add("  core/d = 0.1, which is exactly the sliver population measured in")
    add("  `docs/issue84/sliver_aliasing.json` (104 of 770 main-wing wake triangles above")
    add("  aspect 50).")
    add("")
    add("## 4. Frozen acceptance set on G2")
    add("")
    add("`tools/issue84/projection_gate_set.py` rebuilds the G2 case, solves it once, and")
    add("re-evaluates the whole field at the fixed rejected nodes with both integrators;")
    add("`legacy_vs_fan_rel` is the relative difference of the total velocity vector,")
    add("taking the highest fan order as reference. `projection_kind` and `aspect` belong")
    add("to the triangle that dominates the area term at that target.")
    add("")
    parts.extend(gate_section())
    add("")
    add("Every point classifies as `vertex`, and that is a property of the problem rather")
    add("than of the classifier: a wake node is a vertex of the wake triangles attached to")
    add("it, so the closest point on the dominant triangle is a node of that triangle. The")
    add("fan therefore contributes the strips, not the sector split. Edge and interior")
    add("targets still matter for face-centre and interior sampling, which is why the unit")
    add("tests cover them.")
    add("")
    add("At the two worst nodes the spread is dominated by the shipped map:")
    add("")
    parts.extend(probe_section())
    add("")
    add("The shipped map keeps shrinking by only about 6x per doubling of the order and is")
    add("still at 5e-4 at order 96, two orders of magnitude worse than the fan at order 32.")
    add("Swapping the far edge changes nothing at these targets (5.58e-06 both ways), which")
    add("is the falsifiable prediction of the method: the anisotropy is already handled by")
    add("the strips. Strip refinement saturates by strips 32 (strips 128 changes the fourth")
    add("digit), so the useful knob is the order.")
    add("")
    add("## 5. Cost")
    add("")
    add("Per point on the full G2 field (1666 triangles): the shipped map at order 48 is")
    add("0.22 s, the fan at order 16 with strips 8 is 0.23 s, and the fan at order 32 with")
    add("strips 8 is 0.76 s. So the cheapest fan matches the shipped cost while being about")
    add("10x more accurate at the hard nodes, and a 3.5x budget buys about 580x. The fan is")
    add("not wired into the production sampling loop: the review asked for unit tests plus a")
    add("frozen-snapshot pass before anything is switched on, and that A/B is not done here.")
    add("")
    add("## 6. Not done here")
    add("")
    add("- The G2 wake-update stage tracing the review asked for alongside C3 (sampling gate,")
    add("  candidate velocity and displacement, Picard and line search, root and attached")
    add("  constraints, sheet mass before and after) is not in this document; the definition")
    add("  of `convergence=102` and its overlap with sampling rejection is still open.")
    add("- CI coverage of `test/test_p1_regularized_sheet_velocity.py` and")
    add("  `test/test_p1_coupled_material_wake.py` is still blocked: the token refuses workflow")
    add("  file writes (`refusing to allow a Personal Access Token to create or update")
    add("  workflow .github/workflows/issue77-gamma-support.yml without workflow scope`).")
    add("  The two-line diff is `docs/issue84/ci-coverage.patch`; the minimum permission is")
    add("  Workflows: Read and write for the token owner.")
    add("- No claim about G2 convergence, about the core radius being physical, or about a")
    add("  production speedup.")
    add("")
    add("## 7. Reproduce")
    add("")
    add("    python -m pytest test/test_p1_projection_duffy.py -q")
    add("    python tools/issue84/projection_duffy_bench.py --aspects 1,10,50,260,520,1405.7 \\")
    add("        --core-ratios 0.1,0.3,1,3 --ref-n 32 --ref-order 24 \\")
    add("        --out docs/issue84/projection_duffy_bench.json")
    add("    python tools/issue84/projection_gate_set.py \\")
    add("        --main-half-surface runs/half_cad_wingbody_v1/surface.json \\")
    add("        --csv quad:runs/issue84/wake_rejection_g2_rm48.csv \\")
    add("        --csv eps:runs/issue84/wake_rejection_g2_analytic.csv \\")
    add("        --fan-orders 16 32 --out docs/issue84/projection_gate_set.json")
    add("")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
