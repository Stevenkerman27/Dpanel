# Projection-aware Duffy quadrature for the regularized P1 area kernel

Issue #84, PR #85, step C3. Status: opt-in module plus measurements. The
production path is unchanged, the default velocity method is still
`analytic_edges`, and `physical_accuracy_accepted` stays false.

## 1. Why the shipped map stalls

The regularized area term is

    I_f(p) = int_{T_f} (p - x) / (|p - x|^2 + core^2)^{3/2} dS

and `panel/p1_regularized_sheet_velocity.py:135-150` evaluates it with one fixed
Duffy map per triangle: `bary = ((1-u)(1-v), u, (1-u)v)` with weights
`w_u w_v (1-u)`. Two geometric situations break it, and both are common in the
G2 wake sheet:

1. the target is much closer to the triangle than the triangle is large, so the
   integrable peak sits at a parametric corner that is not the collapsed vertex;
2. the triangle is a sliver (main-wing wake aspect reaches 260 to 1405.7), so the
   collapse direction is nearly parallel to the long edge and the peak is smeared
   along the short one.

The measured consequence (`docs/ISSUE84_RM_BENCH.md`) is that raising the order
does not converge: order 32 can be worse than order 16. That is aliasing, not
truncation error, and it is why C3 asked for a projection-aware decomposition.

## 2. Method

`panel/p1_projection_duffy.py` keeps the kernel and changes only the domain
decomposition (`closest_point_on_triangle` is the Ericson region test):

1. project p onto the triangle and classify the closest point q as interior, edge
   or vertex;
2. fan the triangle into sub-triangles that all have q as their first vertex
   (3 interior, 2 edge, 1 vertex), optionally cut each sector into `strips` pieces
   along its far edge, and optionally `swap` the far edge to test the other
   integration direction;
3. integrate every sub-triangle with a Duffy map collapsed at q:
   `x = q + u a + u v (b - a)`, `|dS| = |a x (b - a)| u du dv`, full Jacobian, no
   empirical weights.

Invariants the tests pin down: sub-areas sum to the parent area and the pieces do
not overlap (random interior samples hit exactly one piece), a constant integrand
integrates to the exact area, the result is covariant under rotation, translation
and scaling, and all four projection branches are exercised.

## 3. Single-triangle benchmark against a composite oracle

`tools/issue84/projection_duffy_bench.py` builds an independent oracle: the parent
triangle is cut into a barycentric grid of n^2 sub-triangles (`uniform_subdivide`),
each integrated with its own Duffy map, and the reference is the sum over the grid.
The reference uses n = 32, order 24 (1024 sub-triangles per case); the
`ref self-change` column is |ref(n=32, order 24) - ref(n=64, order 32)| / |ref|, so
a value near 1e-15 means the reference is itself converged. Errors are relative
errors of the whole induced area vector `cross(gamma, I)`, not of one scalar
component.

Target: centroid of the triangle raised by d = 0.01; aspect = longest edge over
height; core/d in {0.1, 0.3, 1, 3}. `old` is the shipped single-Duffy map.

| aspect | core/d | ref n_sub | ref self-change | fan o16/s1 | fan o16/s8 | fan o32/s8 | fan o48/s64 | old o16/s1 | old o32/s1 | old o48/s1 | old o96/s1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 1024 | 1.1e-15 | 6.80e-03 | 6.80e-03 | 1.57e-05 | 4.06e-08 | 3.13e-01 | 1.81e-01 | 1.07e-01 | 2.22e-02 |
| 1 | 0.3 | 1024 | 2.2e-15 | 6.22e-03 | 6.22e-03 | 1.43e-05 | 2.49e-08 | 3.00e-01 | 1.69e-01 | 9.90e-02 | 1.95e-02 |
| 1 | 1 | 1024 | 1.0e-15 | 1.93e-03 | 1.93e-03 | 1.76e-06 | 1.70e-09 | 2.05e-01 | 9.64e-02 | 4.96e-02 | 6.26e-03 |
| 1 | 3 | 1024 | 1.7e-16 | 5.21e-05 | 5.21e-05 | 1.71e-09 | 8.39e-14 | 5.86e-02 | 2.02e-02 | 4.70e-03 | 4.25e-05 |
| 10 | 0.1 | 1024 | 3.3e-15 | 8.78e-02 | 7.26e-04 | 1.57e-06 | 3.00e-09 | 6.64e-01 | 3.74e-01 | 2.07e-01 | 3.15e-02 |
| 10 | 0.3 | 1024 | 9.6e-16 | 8.71e-02 | 6.32e-04 | 1.16e-06 | 1.82e-09 | 6.54e-01 | 3.58e-01 | 1.93e-01 | 2.70e-02 |
| 10 | 1 | 1024 | 3.2e-16 | 7.93e-02 | 9.41e-05 | 1.12e-07 | 6.79e-11 | 5.61e-01 | 2.39e-01 | 9.80e-02 | 6.33e-03 |
| 10 | 3 | 1024 | 1.6e-15 | 4.11e-02 | 3.68e-06 | 6.24e-11 | 2.87e-15 | 2.55e-01 | 3.21e-02 | 3.88e-03 | 7.73e-06 |
| 50 | 0.1 | 1024 | 2.2e-15 | 5.05e-01 | 3.61e-04 | 7.75e-07 | 1.48e-09 | 8.87e-01 | 6.48e-01 | 4.14e-01 | 7.42e-02 |
| 50 | 0.3 | 1024 | 1.7e-15 | 4.96e-01 | 3.24e-04 | 5.90e-07 | 9.08e-10 | 8.80e-01 | 6.32e-01 | 3.94e-01 | 6.54e-02 |
| 50 | 1 | 1024 | 8.2e-16 | 4.14e-01 | 4.88e-05 | 7.14e-08 | 4.31e-11 | 8.10e-01 | 4.80e-01 | 2.36e-01 | 1.89e-02 |
| 50 | 3 | 1024 | 2.5e-15 | 1.46e-01 | 2.01e-06 | 7.57e-11 | 2.15e-15 | 4.32e-01 | 8.24e-02 | 1.28e-02 | 3.71e-05 |
| 260 | 0.1 | 1024 | 3.2e-16 | 5.68e-01 | 3.15e-04 | 6.72e-07 | 1.28e-09 | 9.11e-01 | 7.07e-01 | 4.86e-01 | 1.06e-01 |
| 260 | 0.3 | 1024 | 5.2e-16 | 5.55e-01 | 2.87e-04 | 5.19e-07 | 7.98e-10 | 9.04e-01 | 6.90e-01 | 4.62e-01 | 9.31e-02 |
| 260 | 1 | 1024 | 1.5e-15 | 4.44e-01 | 4.64e-05 | 6.87e-08 | 4.14e-11 | 8.35e-01 | 5.24e-01 | 2.75e-01 | 2.60e-02 |
| 260 | 3 | 1024 | 1.3e-15 | 1.42e-01 | 1.94e-06 | 8.17e-11 | 2.61e-15 | 4.47e-01 | 8.85e-02 | 1.42e-02 | 4.51e-05 |
| 520 | 0.1 | 1024 | 4.9e-16 | 5.71e-01 | 3.13e-04 | 6.69e-07 | 1.27e-09 | 9.12e-01 | 7.10e-01 | 4.89e-01 | 1.08e-01 |
| 520 | 0.3 | 1024 | 6.1e-16 | 5.57e-01 | 2.85e-04 | 5.17e-07 | 7.94e-10 | 9.05e-01 | 6.92e-01 | 4.65e-01 | 9.45e-02 |
| 520 | 1 | 1024 | 2.0e-16 | 4.45e-01 | 4.63e-05 | 6.86e-08 | 4.14e-11 | 8.35e-01 | 5.25e-01 | 2.76e-01 | 2.63e-02 |
| 520 | 3 | 1024 | 1.8e-15 | 1.41e-01 | 1.94e-06 | 8.19e-11 | 3.70e-16 | 4.47e-01 | 8.87e-02 | 1.42e-02 | 4.54e-05 |
| 1405.7 | 0.1 | 1024 | 5.9e-16 | 5.71e-01 | 3.13e-04 | 6.68e-07 | 1.27e-09 | 9.12e-01 | 7.10e-01 | 4.89e-01 | 1.08e-01 |
| 1405.7 | 0.3 | 1024 | 4.3e-16 | 5.58e-01 | 2.85e-04 | 5.16e-07 | 7.93e-10 | 9.05e-01 | 6.92e-01 | 4.66e-01 | 9.49e-02 |
| 1405.7 | 1 | 1024 | 1.4e-15 | 4.45e-01 | 4.63e-05 | 6.86e-08 | 4.14e-11 | 8.36e-01 | 5.26e-01 | 2.76e-01 | 2.64e-02 |
| 1405.7 | 3 | 1024 | 3.7e-16 | 1.41e-01 | 1.94e-06 | 8.20e-11 | 7.22e-16 | 4.47e-01 | 8.87e-02 | 1.43e-02 | 4.55e-05 |

Reading the table:

- the fan at order 16 with strips 8 is already 1e3 to 1e5 better than the shipped
  map at the same order, and strips 64 buys nothing over strips 8 (the peak is at
  an apex, so strip refinement saturates);
- after the fan the error is order limited and monotone: order 32 improves by about
  1e4 over order 16, order 48 reaches 1e-8 and better;
- the shipped map is still at 1e-1 or worse at order 96 for aspect >= 260 with
  core/d = 0.1, which is exactly the sliver population measured in
  `docs/issue84/sliver_aliasing.json` (104 of 770 main-wing wake triangles above
  aspect 50).

## 4. Frozen acceptance set on G2

`tools/issue84/projection_gate_set.py` rebuilds the G2 case, solves it once, and
re-evaluates the whole field at the fixed rejected nodes with both integrators;
`legacy_vs_fan_rel` is the relative difference of the total velocity vector,
taking the highest fan order as reference. `projection_kind` and `aspect` belong
to the triangle that dominates the area term at that target.

case g2 at alpha 5 deg, core 0.01, field 1666 triangles, old order 48, fan orders [16, 32] strips 8, solve 7.0 s, whole set 228.2 s.

| set | points | projection class | max old-vs-fan | median old-vs-fan | max aspect | old s/point | fan s/point |
|---|---|---|---|---|---|---|---|
| eps | 102 | vertex 102 | 3.19e-03 | 1.57e-08 | 520.0 | 0.22 | 0.79 |
| quad | 7 | vertex 7 | 3.19e-03 | 2.52e-03 | 260.0 | 0.22 | 0.77 |

Every point classifies as `vertex`, and that is a property of the problem rather
than of the classifier: a wake node is a vertex of the wake triangles attached to
it, so the closest point on the dominant triangle is a node of that triangle. The
fan therefore contributes the strips, not the sector split. Edge and interior
targets still matter for face-centre and interior sampling, which is why the unit
tests cover them.

At the two worst nodes the spread is dominated by the shipped map:

The highest fan order available here, order 48 with strips 32, is the reference;
every entry is |v - v_ref| / |v_ref| of the total velocity vector at that node.

| node | old o48 | old o96 | fan o16/s8 | fan o16/s32 | fan o16/s128 | fan o32/s8 | fan o32/s32 | fan o32/s32 swap |
|---|---|---|---|---|---|---|---|---|
| main_wing 407 | 3.19e-03 | 5.30e-04 | 3.24e-04 | 7.25e-05 | 7.24e-05 | 5.54e-06 | 5.58e-06 | 5.58e-06 |
| main_wing 11 | 3.15e-03 | 5.23e-04 | 3.19e-04 | 7.25e-05 | 7.25e-05 | 5.49e-06 | 5.53e-06 | 5.53e-06 |

The shipped map keeps shrinking by only about 6x per doubling of the order and is
still at 5e-4 at order 96, two orders of magnitude worse than the fan at order 32.
Swapping the far edge changes nothing at these targets (5.58e-06 both ways), which
is the falsifiable prediction of the method: the anisotropy is already handled by
the strips. Strip refinement saturates by strips 32 (strips 128 changes the fourth
digit), so the useful knob is the order.

## 5. Cost

Per point on the full G2 field (1666 triangles): the shipped map at order 48 is
0.22 s, the fan at order 16 with strips 8 is 0.23 s, and the fan at order 32 with
strips 8 is 0.76 s. So the cheapest fan matches the shipped cost while being about
10x more accurate at the hard nodes, and a 3.5x budget buys about 580x. The fan is
not wired into the production sampling loop: the review asked for unit tests plus a
frozen-snapshot pass before anything is switched on, and that A/B is not done here.

## 6. Not done here

- The G2 wake-update stage tracing the review asked for alongside C3 (sampling gate,
  candidate velocity and displacement, Picard and line search, root and attached
  constraints, sheet mass before and after) is not in this document; the definition
  of `convergence=102` and its overlap with sampling rejection is still open.
- CI coverage of `test/test_p1_regularized_sheet_velocity.py` and
  `test/test_p1_coupled_material_wake.py` is still blocked: the token refuses workflow
  file writes (`refusing to allow a Personal Access Token to create or update
  workflow .github/workflows/issue77-gamma-support.yml without workflow scope`).
  The two-line diff is `docs/issue84/ci-coverage.patch`; the minimum permission is
  Workflows: Read and write for the token owner.
- No claim about G2 convergence, about the core radius being physical, or about a
  production speedup.

## 7. Reproduce

    python -m pytest test/test_p1_projection_duffy.py -q
    python tools/issue84/projection_duffy_bench.py --aspects 1,10,50,260,520,1405.7 \
        --core-ratios 0.1,0.3,1,3 --ref-n 32 --ref-order 24 \
        --out docs/issue84/projection_duffy_bench.json
    python tools/issue84/projection_gate_set.py \
        --main-half-surface runs/half_cad_wingbody_v1/surface.json \
        --csv quad:runs/issue84/wake_rejection_g2_rm48.csv \
        --csv eps:runs/issue84/wake_rejection_g2_analytic.csv \
        --fan-orders 16 32 --out docs/issue84/projection_gate_set.json

