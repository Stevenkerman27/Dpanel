"""Render docs/ISSUE84_P0_PROFILE.md from the measured P0 profiling artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ISSUE = DOCS / "issue84"


def load(name, default=None):
    path = ISSUE / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return "{:,.{d}f}".format(value, d=digits)


def thousands(value):
    return "{:,}".format(int(round(value)))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out


def main():
    gate = load("velocity_profile_109.json", {})
    analytic = load("profile_g2_analytic.json", {})
    signed = load("profile_g2_signed_speed.json", {})
    b2_prod = load("b2_target_profile_prod.json", {})
    b2_aligned = load("b2_target_profile_aligned.json", {})
    semantics = load("fan_order_semantics.json", {})

    lines = []
    add = lines.append
    add("# Issue #84 P0: where the coupled material-wake solve actually spends time")
    add("")
    add("Status: measurement only. No solver, default or gate changed. "
        "physical_accuracy_accepted=false still holds for every number below.")
    add("")
    add("## 1. Retracted estimate")
    add("")
    add("An earlier estimate of about 7.7 h per G2 iteration multiplied the pressure-Kutta "
        "least_squares evaluation count (85) by the free-node count (378) by one "
        "regularized velocity evaluation. That model is **withdrawn**:")
    add("")
    add("* solve_coupled_material_wake runs the pressure-Kutta least_squares **before** it "
        "constructs TotalVelocity and before it calls limits() "
        "(panel/p1_coupled_material_wake.py:485-560), so the regularized kernel is not "
        "inside the residual evaluations at all;")
    add("* the driver is the number of target points that limits() pushes through "
        "evaluate_many, which this report measures directly.")
    add("")
    add("No cache keyed on a least_squares iteration is therefore valid, and none is "
        "proposed here.")
    add("")
    add("## 2. Opt-in counters")
    add("")
    add("panel/velocity_profile.py adds enable()/disable()/reset()/snapshot()/write() and a "
        "_profile helper used by four call sites:")
    add("")
    lines.extend(table(
        ["counter group", "where"],
        [["limits.* (calls, points, ladder_steps, targets, refined_targets, tag.<t>.*)",
          "panel/p1_coupled_material_wake.py:195-226"],
         ["total.* (call.count, evaluate_many calls/targets/method)",
          "panel/p1_coupled_material_wake.py:120,166-171"],
         ["area_terms.* (calls, triangles, node_sets, subtriangles, seconds)",
          "panel/p1_projection_duffy.py"],
         ["fan.* (calls, strips, subtriangles, region counts)",
          "panel/p1_projection_duffy.py::projection_fan"]]))
    add("")
    add("test/test_velocity_profile.py (5 tests) proves that the counters are inert when "
        "disabled, that limits(tag=...) returns bitwise identical values with and without a "
        "label, and that peak_rss_mb() is finite and positive.")
    add("")
    add("peak_rss_mb() reads the process peak working set through "
        "kernel32.K32GetProcessMemoryInfo; the psapi.GetProcessMemoryInfo alias returns 0 "
        "on this Windows build and is only a fallback.")
    add("")
    add("## 3. Target-set composition of one B2 iteration")
    add("")
    counters = analytic.get("counters", {}) if analytic else {}
    if analytic:
        add("Measured on the frozen G2 CAD case (runs/half_cad_wingbody_v1/surface.json, "
            "alpha 5 deg, --kutta pressure, analytic edges, two iterations requested):")
        add("")
        lines.extend(table(
            ["quantity", "value"],
            [["limits.calls in the recorded round",
              "{} (two groups x two target sets; the run blocked at iteration 0)".format(
                  counters.get("limits.calls", "n/a"))],
             ["wake-face sample points", str(counters.get("limits.tag.wake_face.points", "n/a"))],
             ["free-node sample points", str(counters.get("limits.tag.free_node.points", "n/a"))],
             ["ladder steps", "{} (4 factors x 2 signs)".format(counters.get("limits.ladder_steps", "n/a"))],
             ["velocity evaluations per iteration", thousands(counters.get("limits.targets", 0))],
             ["evaluate_many calls", str(counters.get("total.evaluate_many.calls", "n/a"))],
             ["wall clock, whole process", "{} s".format(fmt(analytic.get("wall_seconds")))],
             ["peak RSS", "{} MB".format(fmt(analytic.get("peak_rss_mb")))]]))
        add("")
        add("The floor is 2 x 1187 x 4 = {} sign-resolved targets: 794 wake-face centroids "
            "plus 393 free nodes, each sampled at four ladder radii on both sides of the "
            "sheet. A non-analytic method adds one more refinement pass at twice the order, "
            "i.e. another 2 x 1187 targets.".format(thousands(counters.get("limits.targets", 0))))
    else:
        add("docs/issue84/profile_g2_analytic.json is missing.")
    add("")
    add("## 4. Cost of a single target, per fan configuration")
    add("")
    if gate:
        add("109 fixed points (7 legacy-quadrature rejects + 102 epsilon rejects, rebuilt "
            "from the G2 rejection tables), one evaluation each, on the same 1666-triangle "
            "sheet (872 wall + 794 wake). Every point projects onto a triangle vertex.")
        add("")
        legacy = None
        for name, block in gate.get("configs", {}).items():
            row = block.get("eps")
            if row is not None and name.startswith("legacy"):
                legacy = row.get("milliseconds_per_point")
        rows = []
        for name, block in gate.get("configs", {}).items():
            row = block.get("eps")
            if row is None:
                continue
            sec = row.get("seconds", {})
            ms = row.get("milliseconds_per_point")
            ratio = "" if legacy in (None, 0) or ms is None else "{:.2f}".format(ms/legacy)
            rows.append([name, fmt(ms, 1), fmt(sec.get("area_terms.kernel")),
                         fmt(sec.get("area_terms.projection")),
                         fmt(row.get("fan_calls_per_target"), 0),
                         fmt(row.get("subtriangles_per_target"), 0), ratio])
        lines.extend(table(["configuration", "ms/point", "kernel s", "projection s",
                            "fan calls/point", "sub-triangles/point", "x legacy"], rows))
        add("")
        add("The fan partition (closest-point projection plus sector split) is at most 9 % of "
            "the point cost; 91-97 % is the Duffy kernel, and the kernel scales with "
            "sub_triangles * order**2. Sub-triangles per point follow "
            "8*vertex_regions + 16*edge_regions + 24*interior_regions.")
    add("")
    add("## 5. Extrapolated iteration cost")
    add("")
    add("Model: iteration_seconds = (ladder targets + refinement targets) x measured "
        "per-target cost. It is a linear scaling of a measurement, not a measurement of a "
        "whole iteration; section 6 measures the same quantity a second, independent way.")
    add("")
    targets = counters.get("limits.targets", 0)
    configs = gate.get("configs", {}) if gate else {}
    if targets and "28/8" in configs:
        refined = 2*1187
        evals = targets + refined

        def cost(name):
            return configs[name]["eps"]["milliseconds_per_point"]/1000.0
        rows = [["analytic_edges (measured)", "0.0015", "{}".format(thousands(targets)),
                 "{:.0f} s".format(targets*0.0015)],
                ["24/8 (ladder + refinement)", "{:.3f}".format(cost("24/8")),
                 thousands(evals), "{:.1f} h".format(evals*cost("24/8")/3600.0)],
                ["28/8 (ladder only)", "{:.3f}".format(cost("28/8")),
                 thousands(targets), "{:.1f} h".format(targets*cost("28/8")/3600.0)],
                ["28/8 (ladder + refinement)", "{:.3f}".format(cost("28/8")),
                 thousands(evals), "{:.1f} h".format(evals*cost("28/8")/3600.0)],
                ["48/8 (ladder + refinement)", "{:.3f}".format(cost("48/8")),
                 thousands(evals), "{:.1f} h".format(evals*cost("48/8")/3600.0)]]
        lines.extend(table(["method", "per-target s", "evaluations", "iteration"], rows))
    add("")
    add("The fan-order flag, not the limits order, sets this cost; see section 7. Every extra "
        "ladder factor and the refinement pass cost a full re-evaluation.")
    add("")
    add("")
    add("## 6. B2 target set, measured")
    add("")
    for label, payload in (("production --quadrature-order 48", b2_prod),
                           ("aligned --quadrature-order 28", b2_aligned)):
        if not payload:
            add("* {}: missing artifact.".format(label))
            add("")
            continue
        add("**{}** (--max-points {}, method {}, fan {}/{})".format(
            label, payload.get("max_points", 0), payload.get("method"),
            payload.get("fan_order", 0), payload.get("fan_strips", 0)))
        add("")
        rows = []
        for group_name, block in payload.get("groups", {}).items():
            for tag, row in block.get("samples", {}).items():
                rows.append([group_name, tag, row.get("sampled_points", 0),
                             row.get("points_total", 0), fmt(row.get("measured_seconds")),
                             fmt(row.get("seconds_per_point"), 3),
                             fmt(row.get("extrapolated_seconds"), 1)])
        lines.extend(table(["group", "set", "sampled", "total", "measured s", "s/point",
                            "extrapolated s"], rows))
        add("")
        add("Extrapolated whole iteration: **{} s** ({} h).".format(
            thousands(payload.get("extrapolated_iteration_seconds", 0)),
            "{:.2f}".format(payload.get("extrapolated_iteration_seconds", 0)/3600.0)))
        add("")
        first_row = None
        for block in payload.get("groups", {}).values():
            for row in block.get("samples", {}).values():
                first_row = row
                break
            if first_row is not None:
                break
        if first_row is not None:
            counters_row = first_row.get("counters", {})
            sampled = first_row.get("sampled_points", 0)
            ladder_targets = counters_row.get("limits.targets", 0)
            refined_targets = counters_row.get("limits.refined_targets", 0)
            add("The counters verify the linear model on the sampled points: {} points cost "
                "{} ladder targets plus {} refinement targets, i.e. {} evaluations per point, "
                "and targets_full_set is exactly 10 x the set size."
                .format(sampled, ladder_targets, refined_targets,
                        "{:.0f}".format((ladder_targets+refined_targets)/max(sampled, 1))))
            add("")
    if b2_prod and b2_aligned:
        prod = b2_prod.get("extrapolated_iteration_seconds", 0)
        aligned = b2_aligned.get("extrapolated_iteration_seconds", 0)
        add("The two runs differ by {:.1f} s on the same points, i.e. {:.2f} %. Changing "
            "--quadrature-order from 48 to 28 changes nothing in the fan path, which "
            "section 7 shows directly. The cost is set by fan_order and fan_strips."
            .format(abs(aligned-prod), 100.0*abs(aligned-prod)/max(prod, 1e-9)))
        add("")
    add("## 7. Which flag actually controls the fan area integral")
    add("")
    if semantics:
        add("Four fixed wake-face centroids, G2, core 0.01, fan_strips 8, ladder 1 and 0.5 "
            "(so each point is sampled four times plus two refinement samples):")
        add("")
        rows = []
        for run in semantics.get("runs", []):
            rows.append(["fan {}/{}".format(run.get("fan_order"), run.get("fan_strips")),
                         str(run.get("limits_order")),
                         "{:.3e}".format(max(run.get("quadrature_error") or [0.0])),
                         "{:.3e}".format(max(run.get("epsilon_error") or [0.0])),
                         fmt(run.get("seconds"))])
        lines.extend(table(["field", "limits order", "max quadrature error",
                            "max epsilon error", "seconds"], rows))
        add("")
        rows = []
        for row in semantics.get("comparisons", []):
            rows.append([row.get("pair"), "{:.3e}".format(row.get("max_abs_difference", 0.0)),
                         "{:.3e}".format(row.get("max_relative_difference", 0.0))])
        lines.extend(table(["comparison", "max abs difference", "max relative difference"], rows))
        add("")
        add("Conclusions, all measured rather than argued:")
        add("")
        add("* limits(order=...) is inert in the regularized path: fan 28 with limits order 48 "
            "and fan 28 with limits order 28 return bitwise identical means. "
            "panel/p1_regularized_sheet_velocity.py:153-160 forwards self.fan_order to "
            "area_velocity instead of the order argument it receives;")
        add("* fan_order is the binding flag: fan 28 and fan 16 differ, here by "
            "{:.1e} on four smooth centroids and by up to 7.8e-4 on the 109-point set "
            "(16/8 vs 48/8);".format(
                max((row.get("max_relative_difference", 0.0) for row in
                     semantics.get("comparisons", [])), default=0.0)))
        add("* the quadrature channel of the sampling gate is dead in the fan path: the "
            "refinement pass re-evaluates area_terms with the same fan order, so "
            "quadrature_error is exactly 0.0 on every one of these runs. Only epsilon_error "
            "still measures anything;")
        add("* this is reported, not repaired. Making limits' order reach the fan, or gating "
            "on a fan-order refinement instead, changes what the gate certifies and belongs "
            "in the P1/P2 change set.")
    else:
        add("docs/issue84/fan_order_semantics.json is missing.")
    add("")
    add("## 8. Geometry update")
    add("")
    if signed:
        sc = signed.get("counters", {})
        lines.extend(table(
            ["quantity", "value"],
            [["status", str(signed.get("status"))],
             ["limits.calls", str(sc.get("limits.calls", "n/a"))],
             ["velocity evaluations", thousands(sc.get("limits.targets", 0))],
             ["limits.refined_targets", thousands(sc.get("limits.refined_targets", 0))],
             ["wall clock", "{} s".format(fmt(signed.get("wall_seconds")))],
             ["peak RSS", "{} MB".format(fmt(signed.get("peak_rss_mb")))]]))
        add("")
        add("Three sampling rounds ran for --iterations 2 (12 limits calls, 28,488 = 3 x "
            "9,496), so an accepted geometry update adds one round of the same size; it does "
            "not change the per-point cost. The analytic run above blocked after the first "
            "round, which is why it records 4 calls.")
    else:
        add("docs/issue84/profile_g2_signed_speed.json is missing.")
    add("")
    add("## 9. What this implies for P1/P2")
    add("")
    add("* A whole-iteration cache is only legitimate if the same target coordinate recurs in "
        "the same iteration with the same field state; the ladder samples 2*N*4 distinct "
        "coordinates per set, so reuse is limited to repeated sets.")
    add("* Reuse across the ladder exists at the level of quadrature nodes and triangle "
        "geometry, which is what P1 caches.")
    add("* The 91-97 % kernel share is work, not overhead: batching it (P2) removes Python "
        "loops and temporaries but not the sub_triangles * order**2 volume. Cutting the "
        "iteration from hours to minutes needs a cheaper per-face integration for the far "
        "field, decided only after the P2 batching measurement.")
    add("")
    add("## 10. P1 read-only caches, measured")
    add("")
    before_109 = load("velocity_profile_109.json")
    after_109 = load("velocity_profile_109_p1.json")
    if before_109 and after_109:
        rows = []
        for key in ("legacy_duffy/48", "16/8", "24/8", "28/8", "48/8"):
            try:
                before = before_109["configs"][key]["eps"]
                after = after_109["configs"][key]["eps"]
            except KeyError:
                continue
            rows.append([key,
                         fmt(before["milliseconds_per_point"], 1),
                         fmt(after["milliseconds_per_point"], 1),
                         "{:+.2f} %".format(100.0*(after["milliseconds_per_point"]
                                                    /before["milliseconds_per_point"]-1.0)),
                         fmt(before["seconds"].get("area_terms.nodes", 0.0), 5),
                         fmt(after["seconds"].get("area_terms.nodes", 0.0), 5)])
        lines.extend(table(["configuration", "ms/point before", "ms/point after", "change",
                            "node-set s before", "node-set s after"], rows))
        add("")
        add("* p1_projection_duffy._duffy_nodes is now an lru_cache keyed on the order only and "
            "hands out read-only arrays; the node-set stage drops from 0.0235 s to 0.0002 s over "
            "102 targets at order 16 (cache.duffy_hits = 102), but that stage was 0.1 % of a "
            "target;")
        add("* RegularizedP1SheetField now builds read-only per-triangle edge lengths, centroid, "
            "longest/shortest edge, aspect and circumradius once per field. Nothing in the "
            "production path consumes them yet - they exist for a near/far split - so they "
            "cannot show a speedup, and the measured changes above are noise plus the node cache;")
        add("* verdict: P1 is in place and worth at most about 1.5 %, so node sets, triangle "
            "geometry and fan partitions are not where an iteration goes. Any hour-to-minute "
            "change has to come from the per-target kernel volume itself (14,382 sub-triangles x "
            "order**2 nodes at fan 28/8), i.e. from P2/P3, not from caching.")
        add("")
        add("test/test_p1_readonly_caches.py (5 tests) covers cache identity, read-only flags, "
            "clear_caches, bad orders, bitwise equality of cold and warm area_terms and of the "
            "field velocity.")
    else:
        add("velocity_profile_109.json or velocity_profile_109_p1.json is missing.")
    add("")
    add("## 11. P2 kernel variants, measured")
    add("")
    bench = load("fan_kernel_bench.json")
    if bench:
        rows = []
        for key, entry in sorted(bench.get("configs", {}).items()):
            variants = entry.get("variants", {})
            rows.append([key,
                         fmt(variants.get("v1", {}).get("milliseconds_per_point", 0.0), 1),
                         fmt(variants.get("v2", {}).get("milliseconds_per_point", 0.0), 1),
                         "{:.2f}x".format(variants.get("v2", {}).get("speedup_vs_v1", 0.0)),
                         fmt(variants.get("v3", {}).get("milliseconds_per_point", 0.0), 1),
                         "{:.2f}x".format(variants.get("v3", {}).get("speedup_vs_v1", 0.0)),
                         "{:.1e}".format(variants.get("v3", {}).get(
                             "max_relative_difference_vs_v1", 0.0))])
        lines.extend(table(["configuration", "v1 production ms/pt", "v2 inv-den ms/pt",
                            "v2 speedup", "v3 no inner chunk ms/pt", "v3 speedup",
                            "max rel. diff vs v1"], rows))
        add("")
        add("* three kernel shapes were timed on the same 109 targets, same geometry, same order "
            "and strips: production (chunked einsum with an (s,q,3) division), inverse-denominator "
            "einsum, and no inner chunk loop at all. Results agree to 3.6e-16 relative, so the "
            "differences are rounding, not a different integral;")
        add("* both rewrites save about 10 % (28/8 689.5 to 626.5 ms, 48/8 1,759.1 to 1,584.4 ms). "
            "Chunking is therefore not the cost: the same 10 % cap applies to any Python-level "
            "batching of this kernel, because the work is 14,382 sub-triangles x order**2 nodes "
            "per target and every node is touched once;")
        add("* a batch over target points cannot remove that volume either - each target has its "
            "own projection point and therefore its own fan - so P2 is a 1.1x item, not an "
            "hours-to-minutes item.")
    else:
        add("docs/issue84/fan_kernel_bench.json is missing.")
    add("")
    add("## 12. P3 near/far split prototype, measured")
    add("")
    split = load("far_field_split.json")
    if split:
        rows = []
        for key, cell in sorted(split.get("cells", {}).items(),
                                key=lambda item: (item[1].get("ratio", 0.0),
                                                  item[1].get("far_order", 0))):
            rows.append(["{} (far order {})".format(cell.get("ratio"), cell.get("far_order")),
                         "{:.1%}".format(cell.get("near_fraction_median", 0.0)),
                         "{:.2e}".format(cell.get("rel_error_median", 0.0)),
                         "{:.2e}".format(cell.get("rel_error_max", 0.0)),
                         fmt(cell.get("seconds_median", 0.0), 3),
                         "{:.2f}x".format(cell.get("speedup_vs_production_median", 0.0))])
        lines.extend(table(["near/far ratio", "near faces", "median rel. error",
                            "max rel. error", "s/target", "vs production 28/8"], rows))
        add("")
        add("Reference: the full projection fan at 48/32 ({} s per target); production fan 28/8 "
            "differs from it by at most {:.2e} relative.".format(
                fmt(split.get("reference_seconds_median", 0.0), 1),
                split.get("production_relative_error_max", 0.0)))
        add("")
        add("* the near/far split is the first measured lever that moves the seconds rather than "
            "the percent: with the fan only on the faces within a few longest-edge lengths of the "
            "target and a plain order-6 Duffy map elsewhere, a target costs fractions of a second;")
        add("* two facts in that table matter more than the speedups: the error is identical "
            "(1.08e-6 median, 9.81e-5 max) in every cell, and it equals the error of production "
            "fan 28/8 against the same reference. On this subset the far faces therefore do not "
            "move the answer at all - what is left is the near fan's own error - so neither a "
            "lower far order nor a larger near radius costs anything measurable;")
        add("* it is not yet admissible as a production path: the subset is the near-sheet "
            "rejection set, where the 1/d^2 far kernel is smallest. Points far from the sheet "
            "make exactly those faces dominant, and there the single-Duffy far map has no stated "
            "error bound (the 109-point sweep measured legacy max 3.19e-3 on near-sheet targets). "
            "A far kernel with a bound - multipole expansion, or a graded near set tied to "
            "distance and core - plus a far-point validation set is the missing piece;")
        add("* therefore the P3 recommendation is a near/far architecture with an error budget, "
            "not a lower global order: lowering the order of the fan everywhere is what the 109 "
            "points already show cannot reach the declared 2e-4 threshold.")
    else:
        add("docs/issue84/far_field_split.json is missing.")
    add("")
    add("## 13. Reproduce")
    add("")
    add("```")
    add("python -m pytest test/test_velocity_profile.py -q")
    add("python -m tools.issue84.velocity_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --csv quad:runs/issue84/projection_gate_quad.csv --csv eps:runs/issue84/projection_gate_eps.csv --case g2 --alpha 5 --core-radius 0.01 --configs legacy_duffy 16:8 24:8 28:8 48:8 --out docs/issue84/velocity_profile_109.json")
    add("python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 5 --kutta pressure --iterations 2 --profile-json docs/issue84/profile_g2_analytic.json --out runs/issue84/g2_analytic_prof")
    add("python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 5 --kutta signed_speed --iterations 2 --profile-json docs/issue84/profile_g2_signed_speed.json --out runs/issue84/g2_signed_speed_prof")
    add("python -m tools.issue84.b2_target_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --method rm_regularized_p1 --core-radius 0.01 --area-quadrature projection_fan --fan-order 28 --fan-strips 8 --quadrature-order 48 --max-points 6 --out docs/issue84/b2_target_profile_prod.json")
    add("python -m tools.issue84.b2_target_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --method rm_regularized_p1 --core-radius 0.01 --area-quadrature projection_fan --fan-order 28 --fan-strips 8 --quadrature-order 28 --max-points 6 --out docs/issue84/b2_target_profile_aligned.json")
    add("python -m tools.issue84.fan_order_semantics --main-half-surface runs/half_cad_wingbody_v1/surface.json --case g2 --core-radius 0.01 --fan-strips 8 --points 4 --out docs/issue84/fan_order_semantics.json")
    add("python -m tools.issue84.velocity_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --csv quad:runs/issue84/projection_gate_quad.csv --csv eps:runs/issue84/projection_gate_eps.csv --case g2 --alpha 5 --core-radius 0.01 --configs legacy_duffy 16:8 24:8 28:8 48:8 --out docs/issue84/velocity_profile_109_p1.json")
    add("python -m pytest test/test_p1_readonly_caches.py -q")
    add("python -m tools.issue84.fan_kernel_bench --main-half-surface <surface.json> --csv quad:<quad.csv> --csv eps:<eps.csv> --case g2 --alpha 5 --core-radius 0.01 --config 28:8 --config 48:8 --limit 0 --out docs/issue84/fan_kernel_bench.json")
    add("python -m tools.issue84.far_field_split --main-half-surface <surface.json> --csv quad:<quad.csv> --csv eps:<eps.csv> --case g2 --alpha 5 --core-radius 0.01 --fan-order 28 --fan-strips 8 --reference-order 48 --reference-strips 32 --ratios 2 3 4 --far-orders 6 8 10 --limit 12 --out docs/issue84/far_field_split.json")
    add("python -m tools.issue84.render_p0_profile_doc")
    add("```")
    add("")
    (DOCS / "ISSUE84_P0_PROFILE.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", DOCS / "ISSUE84_P0_PROFILE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
