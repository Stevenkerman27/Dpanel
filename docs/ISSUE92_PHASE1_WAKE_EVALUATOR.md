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

## 6c. The hybrid evaluator: fan near, plain Duffy far

The far-field probe left one route open: only 0.4 to 1.3 percent of the wake triangles lie
within 10 to 20 core radii of a real sampler target, so the fan is resolving a peak
that is not there for 99 percent of the sheet. This section is that route,
implemented and measured end to end.

### What the closed form actually is

The closed-form path is two different pieces, and only one of them is a boundary
integral. For a planar triangle with linear doublet density mu
(panel/p1_analytic_sheet_velocity.py:1-16),

    u = (B + sum_faces gamma x (S - Omega n)) / (4 pi),
    B = sum_edges integral_edge mu (t x r)/|r|**3 dl,
    S = sum_face_edges (t x n) integral_edge 1/|r| dl,
    gamma = grad_s(mu) x n.

Both S and B are boundary integrals of a surface integral, by the surface form of
Stokes/Green,

    integral_T grad_s phi dS = oint_dT phi (n x dl),   phi = 1/|r|,

and the straight-edge integrals, including the linear-strength first moment
(I1, I3, Il3 in panel/p1_analytic_sheet_velocity.py:23-65), have elementary
antiderivatives. The normal part is not a boundary integral at all: for a flat
triangle with no core it is the solid angle, an arctan2 of the three edge vectors
(panel/p1_analytic_sheet_velocity.py:68-77). So 'closed form' means 'edge integrals
plus solid angle': exact on a flat triangle, linear in the face count, with a tiny
constant (1.13 ms per target for all 770 triangles, section 2).

Under a core radius the in-plane part survives in closed form: the regularised field
evaluates it in _edge_velocity (panel/p1_regularized_sheet_velocity.py:207-228) with
effective2 = |rho|**2 + core**2, which is the same antiderivative with r replaced by
sqrt(r**2 + core**2). The normal part loses its elementary form, which is why the
regularised area term is integrated numerically at all
(panel/p1_regularized_sheet_velocity.py:391-404: induced = _edge_velocity +
_area_velocity). Two constraints follow for any hybrid, and they are the reason this
section looks the way it does:

* only the area term can be split -- the edge term is already exact and cheap;
* the far set must be integrated with the same regularised kernel. Substituting
  analytic_edges would replace the kernel by one without a core, and the difference
  is a model change of order (core/distance)**2, not a quadrature error.

### The split

panel/p1_hybrid_area.py adds three things and changes nothing else:

* point_triangle_distance: exact point-triangle distance (Ericson 5.1.5, vectorised)
  used instead of a centroid distance, because a centroid test misclassifies every
  large face whose centroid is far while its surface is close, and the wake sheet has
  such faces;
* near_face_mask: distance < near_cores * core -- the core radius is the only length
  that sets the width of the regularisation peak, so the band is measured in core
  radii, not in edge lengths (the earlier prototype in tools/issue84/far_field_split.py
  used a multiple of the longest edge and could not be given an error interpretation);
* hybrid_area_terms / hybrid_area_velocity / hybrid_velocity: fan (order, strips) on
  the near set, plain tensor Duffy map (order) on the far set, edge term untouched.
  hybrid_velocity(field, p, near_cores=inf) reproduces field(p) bit for bit, which is
  how the parity test pins the module to the production evaluator.

### The measurement

Acceptance is tools/issue92/hybrid_acceptance.py. Its thresholds are declared in the
module docstring, before any measurement:

* max |dv|/U <= 1e-3 and p90 |dv|/U <= 1e-4, with U the freestream magnitude -- the
  same normalisation as the production sampling gate's epsilon_error, so 1e-3 is a
  twentieth of the gate's epsilon_tol and not a number chosen after the fact;
* cost <= 0.05x the reference evaluation, the reference being the refined fan 48x8
  that section 4 uses as the converged pass.

All 109 targets are the points the production gate currently REJECTS (they come from
the sampling_epsilon and sampling_quadrature diagnostics of
tools/issue84/wake_node_rejection.py), i.e. the hard tail of the G2 wake, not a
typical sample. Costs are medians of best-of-three per point, all in one process.

Two facts fix the design before any tuning:

1. The near rule's strip count is not the limiter. Near 48x1 and near 48x8 give the
   same maximum error to within a factor 1.5 at fixed far order (2.2e-2 vs 2.6e-2 at
   10 core, far 16), so the residual is not the near fan.
2. The far quadrature is the limiter, and it converges slowly because the far set is
   dominated by high-aspect sliver faces whose contributions cancel strongly. On the
   worst target (node 408, near set exactly equal to the reference), the far set alone
   gives 6.7e-3 at order 16, 4.4e-5 at 32 and 4.4e-8 at 48 against the same far set
   evaluated with the fan at 48x8. A plain tensor map needs order 32 for the 1e-4
   level; order 16 is nowhere near it even at 20 core radii.

Near fan 48x1 (docs/issue92/hybrid_acceptance_g2.json), 109 targets, 9 cells,
cost ratios against the 48x8 reference and against 48x1:

| near core / far order | max | p90 | median | cost | vs 48x1 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 5 / 8  | 2.00e-01 | 9.66e-02 | 8.46e-05 | 0.015 | 0.113 | no |
| 5 / 12 | 1.13e-01 | 5.75e-03 | 1.95e-06 | 0.018 | 0.136 | no |
| 5 / 16 | 2.72e-02 | 1.39e-02 | 5.41e-08 | 0.023 | 0.170 | no |
| 10 / 8  | 1.87e-01 | 4.34e-02 | 1.56e-05 | 0.016 | 0.117 | no |
| 10 / 12 | 9.06e-03 | 4.56e-03 | 2.11e-07 | 0.018 | 0.139 | no |
| 10 / 16 | 2.21e-02 | 4.03e-03 | 2.01e-09 | 0.023 | 0.173 | no |
| 20 / 8  | 2.20e-02 | 5.38e-03 | 1.24e-06 | 0.016 | 0.124 | no |
| 20 / 12 | 1.86e-02 | 4.30e-03 | 1.59e-08 | 0.019 | 0.145 | no |
| 20 / 16 | 5.01e-03 | 3.36e-03 | 1.98e-10 | 0.024 | 0.179 | no |

Near fan 48x8 (docs/issue92/hybrid_acceptance_g2_near8.json), same grid; the cost
column is also against 48x1 because the baseline and the near rule are then the same
configuration:

| near core / far order | max | p90 | median | cost | verdict |
| --- | --- | --- | --- | --- | --- |
| 5 / 8  | 2.03e-01 | 9.23e-02 | 8.46e-05 | 0.019 | no |
| 5 / 12 | 1.16e-01 | 9.18e-03 | 1.95e-06 | 0.022 | no |
| 5 / 16 | 3.16e-02 | 1.25e-02 | 5.09e-08 | 0.026 | no |
| 10 / 8  | 1.83e-01 | 4.32e-02 | 1.56e-05 | 0.023 | no |
| 10 / 12 | 8.68e-03 | 4.56e-03 | 2.11e-07 | 0.025 | no |
| 10 / 16 | 2.64e-02 | 5.03e-03 | 1.33e-09 | 0.030 | no |
| 20 / 8  | 2.14e-02 | 8.01e-03 | 1.24e-06 | 0.029 | no |
| 20 / 12 | 1.92e-02 | 5.60e-03 | 1.04e-08 | 0.032 | no |
| 20 / 16 | 3.72e-03 | 6.76e-04 | 1.37e-10 | 0.037 | no |

The far order is then raised (docs/issue92/hybrid_acceptance_g2_far32.json, near fan
48x8):

| near core / far order | max | p90 | median | cost | verdict |
| --- | --- | --- | --- | --- | --- |
| 10 / 24 | 1.11e-03 | 2.28e-04 | 4.13e-13 | 0.043 | no (both error bounds miss) |
| 10 / 32 | 5.28e-04 | 1.16e-04 | 2.67e-15 | 0.070 | no (p90 misses) |
| 20 / 24 | 3.70e-04 | 7.50e-05 | 1.24e-14 | 0.050 | no (cost 0.0504 vs 0.05) |
| 20 / 32 | 1.90e-05 | 5.16e-06 | 1.78e-15 | 0.077 | no (cost only) |

### What this unit claims

* The substitution error of the hybrid is dominated by the far quadrature and is
  controlled: at 20 core / far 32 the worst of the 109 hard targets is 1.9e-5 in
  |dv|/U and the 90th percentile 5.2e-6, i.e. 50x and 19x inside the declared bounds,
  with a median of 1.8e-15.
* The declared cost bound is NOT met. The cheapest cell that meets both error bounds
  is 20 core / far 24 at 0.0504x the 48x8 reference, which misses 0.05 by 0.8 percent;
  the cell with real margin on accuracy costs 0.077x.
* In absolute terms that is 60 to 110 ms per target against 1.38 to 1.97 s for the
  48x8 reference and 328 ms for the 16x8 fan the gate uses as its base (section 1).
  The hybrid is therefore about 3x cheaper than the base fan and 13 to 23x cheaper
  than the refined one, with an error that neither fan achieves on this target set --
  which is why the comparison that matters is accuracy at equal cost, not cost alone.
* The near rule is cheap and does not need to be clever: 48x1 is within a factor 1.5
  of 48x8 in the worst case, so the near fan can stay at the cheapest setting.
* The 4 to 10 ms per target projected in section 8 is refuted: the flat tensor map has
  to run at order 24 to 32 over 99 percent of the sheet to beat 1e-4, and that is where
  the 60 to 110 ms goes. A cheaper far field would need a better quadrature (projection
  or moment splitting on the far set), not a cheaper rule.

### Non-claims

* The acceptance set is the rejected set. A pass here would say the substitution error
  is below the gate tolerance even on the difficult targets; it says nothing about
  whether the fan configuration itself is converged (section 6).
* No production default changed: analytic_edges and fan 16x8 are untouched, and the
  hybrid is reachable only by calling panel/p1_hybrid_area.py explicitly.
* Timings are one machine, best of three, same process; only same-run ratios are
  meaningful.
* The closed-form (core-free) far evaluation is not measured here: the suite measures
  quadrature substitutions of the same kernel only, and a core-free far field would
  need its own (core/distance)**2 model-error study.

## 6d. Three backends, one answer: parity first, then the GPU

The gate order for this phase is explicit: no CUDA before CPU parity, and no silent
fallback. This section reports both halves on the same frozen geometry, then measures
what a GPU is actually worth on each of the two evaluators this project has.

### Parity

Geometry: the G2 CAD half model, 1666 triangles (872 wall + 794 wake), core radius 0.01,
109 gate targets, frozen in runs/issue92/parity_g2.npz. Reference is the NumPy
evaluator; error is max |dv|/U with U = 1, the same normalisation the production gate
uses for epsilon_error.

| configuration | targets | NumPy ms/target | torch CPU ms/target | CUDA ms/target | CPU error | CUDA error |
| --- | --- | --- | --- | --- | --- | --- |
| fan 16x4 | 109 | 107.2 | 71.1 (1.5x) | 55.8 (1.9x) | 3.55e-15 | 4.44e-15 |
| fan 48x8 | 8 | 1376.0 | 378.6 (3.6x) | 208.9 (6.6x) | 7.99e-15 | 7.11e-15 |
| fan 8x1 | 40 | 20.2 | 26.5 (0.76x) | 29.6 (0.68x) | 1.78e-15 | 2.66e-15 |

Declared bounds were 1e-12 (CPU) and 1e-10 (CUDA); every cell is three orders inside
them. The two paths are not bitwise equal and cannot be: the NumPy kernel accumulates
per-face totals with a reduceat over contiguous runs, the torch kernel with index_add_
over the same runs, and float64 addition is not associative. 1e-15 relative is the
floor for that difference, which is why the contract is a bound and not equality.

The fan 8x1 row is the important negative result. At that size the kernel does no work
worth scheduling: per-target Python dispatch and launch cost dominate, and both
accelerated backends are *slower* than NumPy. A per-target loop over torch is therefore
not a strategy at the order the gate uses in its cheap configuration; the only form
that pays is a batch of targets per call, which is what section 4b argued from the
kernel side.

### The analytic evaluator is where the GPU pays

The regularised P1 fan is not the production evaluator. panel/p1_analytic_sheet_velocity.py
is: closed-form edge integrals plus solid angles, 0.69 ms per target on the G2 sheet,
and the coupled loop spends its sampling time there. The torch port
(panel/p1_torch_analytic_sheet.py) keeps the same edge merge, the same moment branches
and the same fail-closed behaviour, hoists the target-independent geometry (unit
tangents, edge lengths) out of the point loop, and batches targets.

18,992 targets is one real G2 coupled update (section 1):

| backend | wall (s) | ms/target | speed-up |
| --- | --- | --- | --- |
| NumPy | 13.07 | 0.688 | 1.0x |
| torch CPU float64 | 3.83 | 0.202 | 3.41x |
| torch CUDA float64 | 0.570 | 0.030 | 22.9x |

Parity on the gate probes: 3.78e-15 (CPU) and 3.96e-15 (CUDA) against the NumPy
reference, inside the same declared bounds. Peak CUDA allocation 232 MB at chunk 256,
which is the target batch, not the sheet: the sheet's 4998 face edges and 262 boundary
edges are a few hundred kilobytes.

One behavioural detail that a port must not lose: the raw gate nodes lie on the sheet
boundary, where the analytic kernel is genuinely singular, and the NumPy evaluator
raises FloatingPointError ("field point lies on a singular source edge") rather than
returning inf. The port raises the same error, and the parity probes are the gate nodes
offset by +-0.05 along z (218 of 218 evaluable). A backend that returned inf or NaN here
would silently poison an iteration instead of stopping it.

### What this unit claims

* float64 CPU and CUDA ports of both evaluators reproduce NumPy to 4e-15 relative on a
  frozen G2 snapshot, which buys the permission Phase 4 needs.
* The analytic evaluator is the GPU lever in this codebase: 23x end to end on a full
  update's target set, against 1.5x to 6.6x for the fan kernel.
* Small fan configurations get slower on both torch backends, so any accelerated
  backend must be batched over targets and must not be enabled for the cheap rule.

### Non-claims

* No production default changed: analytic_edges and the NumPy fan are untouched, and
  both ports are opt-in (`device=`, `backend=`) with an explicit device; an unavailable
  CUDA device raises instead of falling back.
* Timings are one machine (RTX 5060, torch 2.13.0+cu132), same process, warm after one
  untimed call. Only the ratios matter.
* Parities are on one geometry. Step 5 repeats them across G1 and G2 before any of this
  is considered portable.
* The 232 MB is torch's allocator high-water mark at chunk 256, not a tuned figure;
  chunking is exposed and larger chunks trade memory for fewer launches.

## 7. Reproduction

    python -m tools.issue92.wake_evaluator_profile \
        --fan-targets 16 --sweep-targets 16 \
        --out docs/issue92/wake_evaluator_profile.json \
        --sweep-out docs/issue92/wake_fan_strips_sweep.json

Contract tests: test/test_issue92_wake_profile.py (5 tests, 0.4 s, no fixtures).

Far-field probe: python -m tools.issue92.far_field_probe --bands 3 10 30 100 --near-share (docs/issue92/far_field_bands.json).

Hybrid acceptance (section 6c):

    python -m tools.issue92.hybrid_acceptance \
        --main-half-surface <repo>/runs/half_cad_wingbody_v1/surface.json \
        --csv eps:runs/issue84/projection_gate_eps.csv \
        --csv quad:runs/issue84/projection_gate_quad.csv \
        --fan-strips 8 --near-cores 10 20 --far-orders 24 32 \
        --out docs/issue92/hybrid_acceptance_g2_far32.json

Hybrid contract tests: test/test_p1_hybrid_area.py (16 tests, 0.14 s, no fixtures).

Three-backend parity (section 6d). The snapshot is dumped once with .conda_env (JAX,
no torch) and then replayed by the torch interpreter, because scipy plus an active CUDA
context dies with a Windows access violation in one process:

    python -m tools.issue92.backend_parity --dump runs/issue92/parity_g2.npz \
        --main-half-surface <repo>/runs/half_cad_wingbody_v1/surface.json \
        --csv eps:runs/issue84/projection_gate_eps.csv \
        --csv quad:runs/issue84/projection_gate_quad.csv
    python -m tools.issue92.backend_parity \
        --geometry runs/issue92/parity_g2.npz --out docs/issue92/backend_parity.json
    python -m tools.issue92.analytic_bench \
        --geometry runs/issue92/parity_g2.npz --out docs/issue92/analytic_torch_bench.json

Backend tests need torch, which lives in the other interpreter; test/conftest.py imports
jax, so a stub is used to run them there:

    python -m pytest test/test_p1_torch_area.py test/test_p1_torch_analytic_sheet.py

with PYTHONPATH containing the repo plus a one-function jax stub (jax.config.update).
16 and 9 tests respectively; 1 CUDA-absence test skips when a device is present.

## 8. Next

* Batched target x face kernel, kept as an opt-in backend and gated on parity with
  area_terms at 1e-13; its expected ceiling is the projection share, not an order
  of magnitude (section 4b).
* Near-field fan plus a far field: implemented and measured in section 6c. It is
  60 to 110 ms per target (0.077x the 48x8 reference, about a third of the 16x8 base
  fan) with a worst-case substitution error of 1.9e-5 on the 109 rejected targets,
  which meets both error bounds but misses the declared 0.05 cost bound. The remaining
  lever is the far quadrature, not the near rule: the flat tensor map needs order 24
  to 32 over 99 percent of the sheet, so a cheaper far field needs projection or a
  moment split on the far set.
* CUDA is now unblocked and its target is chosen by measurement, not by guess: the
  analytic evaluator is 22.9x on a full update's targets (0.030 ms/target, 232 MB),
  while the fan kernel is 1.5x to 6.6x and is slower than NumPy in its cheap 8x1
  configuration (section 6d). The next unit therefore ports the analytic evaluator into
  the coupled loop as an explicit backend option and keeps the fan kernel batched over
  targets, never per target.