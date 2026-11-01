# Phase 1 step 3: where the wake-velocity evaluator actually spends its time

Status: measured 2026-11-01 on the frozen CAD wing-body case (G2, alpha = 5 deg,
872 wall faces + 794 wake triangles, 393 free wake sample points). Phase 0
(docs/ISSUE92_PHASE0_BASELINE.md) reported 57.7 s for one coupled wake iteration,
which is 32 batched calls over 18992 target evaluations, i.e. 3.0 ms per point.
This document splits that number into the stages the evaluator really runs and
tests the assumption that the cost is Python bookkeeping rather than arithmetic.

The short answer is that the assumption is wrong, and the wrongness matters:
the production default (method 1, analytic_edges) costs 1.13 ms per point while
the projection-fan backend of method 2 costs 292 ms to 3567 ms per point on the
same points, and 72 to 96 percent of the fan cost is quadrature-node arithmetic,
not Python overhead. Vectorising the per-target loop therefore cannot close the
gap; only reducing the node count or restricting the fan to the near field can.

## 1. What was measured

tools/issue92/wake_evaluator_profile.py builds the same case the issue #84
batteries use, solves it once, and then evaluates a fixed target set once per
configuration while panel/velocity_profile.py records stage seconds and counters.

Targets are the free wake nodes moved off the sheet, p + eps * s_node * n_hat with
eps = 1e-3, s_node = shortest incident wake edge, n_hat = area-weighted sheet
normal. This mirrors what the sampler does (incl. limits(), which evaluates at
p + h n and p - h n) because the free nodes themselves lie on the sheet, where the
unregularized edge integral is singular by construction.

Five configurations, all on the same 1666 triangles:

| configuration | targets | total s | ms/target |
| --- | --- | --- | --- |
| method 1 analytic_edges (production default) | 393 | 0.443 | 1.126 |
| method 2 rm_regularized_p1, legacy_duffy, order 48 | 393 | 109.333 | 278.200 |
| method 2 rm_regularized_p1, projection_fan 16x8 | 16 | 5.242 | 327.620 |
| method 2 rm_regularized_p1, projection_fan 48x8 | 16 | 35.066 | 2191.634 |
| method 2 rm_regularized_p1, projection_fan_adaptive 16x8 | 16 | 57.077 | 3567.299 |

Raw data: docs/issue92/wake_evaluator_profile.json. Absolute seconds are not
portable between machines; only ratios inside one run are used here.

## 2. Stage attribution for the fan

velocity_profile reports four stages inside one fan area term: projection
(fan construction incl. closest-point classification), nodes (the cached Duffy
node set), kernel (the quadrature sum) and the evaluator's edge term.

| configuration | edge ms/pt | projection ms/pt | kernel ms/pt | nodes ms/pt | kernel share |
| --- | --- | --- | --- | --- | --- |
| legacy_duffy order 48 | 0.14 | 0 (no fan) | 278.0 (area stage) | 0 | 100% |
| fan 16x8 | 0.09 | 91.33 | 235.39 | 0.026 | 72% |
| fan 48x8 | 0.10 | 93.17 | 2096.67 | 0.049 | 96% |
| fan_adaptive 16x8 | 0.10 | 42.50 | 3188.93 | 0.422 | 89% |

Two conclusions follow directly.

1. The edge term is free: 0.1 ms per target against 91 to 2097 ms.
2. The kernel is 72 percent of the 16x8 fan cost and 96 percent of the 48x8 one.
   Projection is 4 percent at order 48 and 28 percent at order 16, and it is
   identical for both orders (93 ms), because the fan shape depends on the target
   and the triangle, not on order. Vectorising only the Python loop over faces can
   therefore win at most 28 percent, and only at the cheap end of the order range.

## 3. The cost law

Counters from the same runs give the shape of the work: 14016 sub-triangles per
target at strips = 8, i.e. 8.5 per triangle, because 93 percent of the wake
triangles project to one of their own vertices (12252 vertex, 862 edge, 2 interior
for 1666 triangles), and a vertex sector is one sub-triangle that the strip
refinement then cuts into 8.

Nodes per target = (sub-triangles per target) * order^2, and the measured rate is
9 to 15 million nodes per second across the whole table. Fitting all eight strip
and order points from section 4 gives a two-parameter law:

    seconds per target ~ 1.0e-8 * sub_triangles_per_target * order^2

which is what the numbers say: 1752 sub * 256 nodes = 0.45 M nodes -> 51.5 ms,
1752 * 2304 = 4.04 M -> 292 ms, 14016 * 2304 = 32.3 M -> 2209 ms.

An isolated micro-benchmark with the identical array shape (2048 sub-triangles of
order 2304, the same weighted accumulation) runs at 48.7 M nodes per second,
3.3 times faster than the production kernel, so about 70 percent of the kernel
time is spent in per-64-face assembly inside area_terms (concatenate, the
1024-sub-triangle chunk loop, and np.add.at), not in the arithmetic itself.
Replacing the 1.5 power by sqrt(x)*x in the same micro-benchmark is worth only
1.41x (0.0967 s -> 0.0686 s, parity 1.25e-13).

## 4. Fan work versus accuracy

The same tool sweeps (fan_order, strips) against a fixed 48x8 reference on 16
targets and reports the maximum relative velocity difference over the target set
(docs/issue92/wake_fan_strips_sweep.json):

| fan_order x strips | ms/target | sub/target | max rel err |
| --- | --- | --- | --- |
| 16x1 | 51.5 | 1752 | 3.14e-02 |
| 16x2 | 102.8 | 3504 | 1.39e-02 |
| 16x8 | 329.3 | 14016 | 5.47e-04 |
| 32x1 | 142.4 | 1752 | 3.14e-03 |
| 32x2 | 286.0 | 3504 | 7.90e-04 |
| 48x1 | 292.4 | 1752 | 3.58e-04 |
| 48x2 | 584.8 | 3504 | 4.80e-05 |
| 48x8 | 2208.6 | 14016 | 0 (reference) |

Two readings. Increasing strips and increasing order buy similar accuracy per
node, so the strip refinement is not a better use of the budget than order: 48x1
(292 ms, 3.58e-04) beats 16x8 (329 ms, 5.47e-04) at the same cost. And the
cheapest configuration that is inside the 1e-3 scale of the sampling gate is
32x1 at 142 ms per target, still 126 times the analytic edge path.

## 4b. Kernel rewrite and its measured ceiling

The kernel was rewritten in panel/p1_projection_duffy.py: the strip loop now builds
the source points in place, computes |r|^3 as sqrt(|r|^2) * |r|^2 instead of
pow(x, 1.5), divides in place, and accumulates with one np.add.reduceat over the
contiguous per-face runs instead of a scattered np.add.at. Both changes are
rearrangements, so the result is bit-for-bit the same up to rounding:

| check | order x strips | max relative difference |
| --- | --- | --- |
| area_terms vs per-face reference, 300 random triangles | 16x8 | 4.33e-16 |
| same | 48x1 | 4.16e-16 |
| same | 48x8 | 3.71e-16 |

Cost, measured inside one process on the same arrays (best of three):

| scope | configuration | before | after | speedup |
| --- | --- | --- | --- | --- |
| area_terms, 300 triangles | 16x8 | 0.0433 s | 0.0351 s | 1.23x |
| area_terms, 300 triangles | 48x1 | 0.0403 s | 0.0314 s | 1.28x |
| area_terms, 300 triangles | 48x8 | 0.3111 s | 0.2428 s | 1.28x |
| full TotalVelocity, CAD case, 4 targets | 16x8 | 0.943 s | 0.781 s | 1.21x |
| full TotalVelocity, CAD case, 4 targets | 48x1 | 0.891 s | 0.697 s | 1.28x |

That is the ceiling of this kind of work, and the reason is structural rather than
a missed trick: about half of the kernel time is the (sub-triangle, node, 3)
source-point construction that any Duffy evaluation must perform, and section 3's
3.3x gap to the isolated micro-benchmark was measured on an array that was handed
in ready-made. A batched target-by-face kernel can still remove the per-face
projection and per-chunk allocation, i.e. at most the 4 to 28 percent projection
share plus what is left of the assembly overhead, so it cannot change the order of
magnitude: the fan stays 100 to 260 times the analytic edge path.

## 5. What this means for the plan

The Phase 1 plan expected the induced-velocity stage to be dominated by per-target
bookkeeping (3.0 ms per point against an arithmetic floor of 0.0045 ms). On the
measured path the floor calculation was wrong, not the code: the 0.0045 ms figure
was a single-pass pairwise proxy over 794 triangles with one interaction each,
whereas the fan evaluates order^2 nodes per sub-triangle, i.e. 1.4e4 * 2304 nodes
per target at 48x8. The production configuration is not the fan at all, it is
analytic_edges, and its 1.13 ms per target is consistent with Phase 0's 3.0 ms
once the ladder (two evaluations per point) and the gate bookkeeping are counted.

Consequences, in order of value:

1. Do not vectorise the fan as a performance project. Measured above: the kernel
   rewrite is worth 1.21 to 1.28x with parity at 4e-16, and a batched kernel can
   add at most the projection share on top of that, on a backend 100 to 260 times
   slower than the default. Record the law and spend the effort on what the
   default runs.
2. The fan needs a near-field restriction to be viable at all. A hybrid that uses
   the fan only for the triangles within near_ratio * core of the target and the
   closed-form edge integral or a far-field moment expansion elsewhere is the only
   version of this backend that can approach the analytic path. The existing
   projection_fan_adaptive backend is the opposite (it escalates the whole set) and
   measured 3567 ms per target, 1.6 times the fixed fan.
3. If the fan is kept for accuracy studies, 48x1 is the defensible setting: same
   error scale as 16x8 at the same cost with a simpler fan.

## 6. Non-claims

* These timings are one machine, one run, no repetition; only same-run ratios are
  quoted.
* No accuracy claim is made for any configuration: section 4 compares a
  configuration against 48x8, which is itself a numerical choice, not a reference
  solution.
* The 3.3x gap between the production kernel and the isolated micro-benchmark is a
  measurement of the current code, not a statement that the code will be rewritten
  in this unit.
* No production default changed: analytic_edges, legacy_duffy and fan 16x8 are
  untouched.

## 7. Reproduction

    python -m tools.issue92.wake_evaluator_profile \
        --fan-targets 16 --sweep-targets 16 \
        --out docs/issue92/wake_evaluator_profile.json \
        --sweep-out docs/issue92/wake_fan_strips_sweep.json

Contract tests: test/test_issue92_wake_profile.py (5 tests, 0.4 s, no fixtures).

## 8. Next

* Batched target x face kernel, kept as an opt-in backend and gated on parity with
  area_terms at 1e-13; its expected ceiling is the projection share, not an order
  of magnitude (section 4b).
* Near-field fan plus far-field reduction, since that is the only route to a fan
  that can be used in production.
* CUDA for the influence kernel and the near-field fan only after CPU parity.
