# Issue #84 / PR #85 item B — per-node rejection diagnostic for the CAD G2 free wake

Date: 2026-10-30.  Scope: item B of the cloud review on PR #85
(comment 6514806741): *"for root=90 / eligible=276 / convergence=102 /
rejected=102 and node_epsilon_error_max=0.1934 emit a per-node diagnostic
table with stable node and material ids, wake group, rejection-reason enum,
current/candidate position, displacement, local face area/aspect, minimum
distance to the wall and to other wake nodes, local velocity, epsilon error and
sampling offset h, h/core_radius, quadrature order, TE/contact/root/tip/fold/
collision/degenerate-face flags, grouped into verifiable classes; do not relax
any threshold."*

Everything below is measured on `HEAD 3308433f` + the A-item fix
(`e1c92b0`) of `research/issue84-rm-regularized-p1-wake`.  No production
threshold, default velocity method, or `physical_accuracy_accepted` flag was
changed.

## 0. Reproduction

    python -m tools.issue84.wake_node_rejection \
        --main-half-surface runs/half_cad_wingbody_v1/surface.json \
        --out runs/issue84/wake_rejection_g2.json \
        --csv runs/issue84/wake_rejection_g2.csv

    python -m tools.issue84.ladder_probe \
        --main-half-surface runs/half_cad_wingbody_v1/surface.json \
        --csv runs/issue84/wake_rejection_g2.csv --nodes 56 452 626 407 546 422

Case: `panel/run_p1_coupled_material_wake.py:100-124` reproduced exactly
(`half_cad_wingbody_v1` + `make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0., wake_panels=3)`
at `(1.6, 0, 0.35)`), alpha = 5 deg, `kutta='pressure'`,
`velocity_method='analytic_edges'`, `order=48`,
`ladder=(1, .5, .25, .125)`, `eps_fraction=0.05`,
`epsilon_tol=0.02`, `quadrature_tol=0.002`.

The tool re-derives the gate instead of reading it: with 872 wall faces it
reproduces the production counters bit for bit
(`root=90, eligible=276, sampling_epsilon=102, sampling_quadrature=0, zero_speed=0, opposed_tangent=0`;
`raised_tail`: 15/15 eligible, `epsilon_error_max=5.763e-3`).

## 1. Classification of the 102 rejections

| class | count | note |
| --- | --- | --- |
| `sampling_epsilon` (epsilon ladder does not settle) | **102** | the only class that occurs |
| `sampling_quadrature` | 0 | `analytic_edges` has no edge quadrature, so this metric is not applicable (it is reported as 0 by `limits()`, see `panel/p1_coupled_material_wake.py:205-206`) |
| `zero_speed` | 0 | |
| `opposed_tangent` | 0 | |
| geometric degeneracy / collision / self-intersection | 0 | see section 3 |
| root / attachment constraint | 0 | 90 roots are excluded from the free set before sampling |
| line-search / update failure | 0 | not applicable at iteration 0 |

`epsilon_error`: rejected min 2.057e-2 / median 3.93e-2 / max 1.934e-1;
eligible median 6.268e-3 / max 1.978e-2.  Local speed: rejected median 1.026,
eligible median 0.993.

Sub-buckets of the 102 (measured, not assumed):

* 29 sit on an actual sheet-boundary node (29 of the 74 boundary nodes are
  rejected) — the side edges and the downstream truncation;
* 14 are wing-tip nodes (14 of 14 are rejected);
* 73 are interior nodes (73 of 304).

## 2. The estimator has no limit at these nodes

`ladder_probe.py` re-evaluates the one-sided mean at nine offsets spanning
256x in h (factors 8 -> 1/32 of the production `h = 0.05*scale*0.125`):

| node | class | increments per halving of h (8 steps) | speed at h_max -> h_min |
| --- | --- | --- | --- |
| 56 | boundary, worst (eps 0.1934) | 1.82e-1 ... 1.93e-1 | 0.6416 -> 1.3425 |
| 452 | mirror of 56 (eps 0.1931) | 1.81e-1 ... 1.93e-1 | 0.6423 -> 1.3422 |
| 626 | tip (eps 0.1349) | 1.33e-1 ... 1.35e-1 | 1.0986 -> 1.8469 |
| 407 | tip, aspect 260 (eps 0.1349) | 1.28e-1 ... 1.35e-1 | 1.0617 -> 1.7508 |
| 546 | interior, y=-0.7 (eps 0.1019) | 1.01e-1 constant | 1.0065 -> 1.4782 |
| 422 | interior, aspect 520 (eps 0.1019) | 9.8e-2 ... 1.02e-1 | 1.0272 -> 1.4532 |
| 59 | eligible control | 7.09e-5 -> 4.86e-9 (divide by 4 per halving) | 0.976326 -> 0.976360 |
| 3 | eligible control | 9.7e-5 -> 6.19e-9 (divide by 4 per halving) | 0.997645 -> 0.997624 |

Two different behaviours, not two amplitudes:

* eligible nodes converge as `O(h^2)` (each halving divides the increment by ~4),
  so the ladder is measuring discretisation error and the gate can accept it;
* rejected nodes have a **constant increment per halving**, i.e. the mean grows
  like `log(1/h)` and has **no limit**.  The reported `epsilon_error` is
  therefore not a residual that a finer ladder would shrink: it is the slope of
  a divergence.  Any relaxation of `epsilon_tol` would only accept an
  unconverged quantity, which is exactly what the gate exists to prevent.

## 3. Where the divergence comes from (measured, not modelled)

At node 546 (interior, `x=1.2918, y=-0.70, z=0`) the field is split into its
wall part and its wake-sheet part by rebuilding `AnalyticSheetField` with
either part zeroed (`runs/issue84/probe_decomp.py`).  Sheet-normal component:

| h | full v_z (+/- mean) | wall contribution | sheet contribution |
| --- | --- | --- | --- |
| 1e-2 | 3.1311e-1 | 1.9851e-1 / 2.0113e-1 | 1.1329e-1 (both sides equal) |
| 3e-3 | 4.8887e-1 | 1.9961e-1 / 2.0039e-1 | 2.8887e-1 |
| 1e-3 | 6.5027e-1 | 1.9988e-1 / 2.0015e-1 | 4.5026e-1 |
| 3e-4 | 8.2727e-1 | 1.9998e-1 / 2.0006e-1 | 6.2725e-1 |
| 1e-4 | 9.8878e-1 | 2.0000e-1 / 2.0003e-1 | 7.8876e-1 |

* the wall contribution is **constant to five digits** (1.9998e-1 ... 2.0003e-1);
* the whole divergence is produced by the wake sheet's own `analytic_edges`
  evaluation at its own node, and it is **symmetric** (the +h and -h values
  agree to five digits), i.e. it is not a jump that averaging would cancel but a
  non-integrable near-node term of the edge integral
  (`panel/p1_analytic_sheet_velocity.py` `segment_moments`, the `I1 = int dl/r`
  trace, whose `log(1/h)` behaviour at an edge endpoint is exactly the slope
  measured above).

This is a property of the *unregularized* P1 layer, not of the geometry: the
slope is the sheet's local surface-vorticity strength (the y=-0.7 column shares
one slope, 1.019e-1, along seven stations).

Geometry exclusions, all measured on the same 770-face sheet
(`runs/issue84/fan_check.py`, `runs/issue84/tjunction.py`):

* non-manifold edges (shared by >2 faces): **0**; boundary edges: 164;
* every free node's incident-face fan is connected: **0 split fans**;
* no hanging/T-junction vertices: the smallest distance from a node to a
  non-incident edge is **1.487e-2** (nodes 74/465), i.e. three orders of
  magnitude above the `h` used (4.1e-4 ... 6.3e-4);
* sliver faces are not the discriminator: rejected `local_aspect_max` median
  11.7 (max 520) versus eligible median 13.4 (max 1405.7), and the distance to
  the sheet edge is *larger* for eligible nodes (median 0.5 = 6.35x scale)
  than for rejected ones (0.292 = 4.0x scale).

## 4. The cure is regularization, and it is measurable

Same six nodes, same ladder: replace the evaluator by `rm_regularized_p1`
with an explicit core (no change to the solve, the state, or any gate), and the
rejected increment collapses while the eligible controls stay put:

| core radius | node 56 | node 546 | node 626 | node 407 | node 59 (eligible) | node 3 (eligible) |
| --- | --- | --- | --- | --- | --- | --- |
| none (`analytic_edges`) | 1.934e-1 | 1.019e-1 | 1.349e-1 | 1.349e-1 | 7.775e-8 | 9.910e-8 |
| 0.005 | 3.913e-3 | 2.138e-4 | 1.128e-3 | 2.695e-3 | 7.760e-8 | 9.901e-8 |
| 0.010 | 9.848e-4 | 5.282e-5 | 2.827e-4 | 9.157e-4 | 7.684e-8 | 9.873e-8 |
| 0.020 | 2.400e-4 | 1.255e-5 | 6.952e-5 | 2.841e-4 | 7.418e-8 | 9.763e-8 |
| 0.050 | 3.356e-5 | 1.564e-6 | 9.949e-6 | 4.013e-5 | 5.894e-8 | 9.032e-8 |

At `core=0.01` the worst node improves by 196x (1.934e-1 -> 9.848e-4) and the
eligible controls change by 1.2 %, i.e. the regularization acts precisely on
the nodes where the unregularized estimator is ill-posed and is nearly inert
elsewhere.  So the 102 rejections are the correct behaviour of the gate, and
the admissible fix is a regularized sampling kernel with a documented core
bias — not a larger `epsilon_tol`, not a different ladder, and not a mesh
relaxation.

## 5. Consequences (kept conservative)

* G2's `status=blocked_initial_nonlinear_solve` / `iteration_limit` chain is
  *not* fixed by this diagnostic; item B only shows that the 102 rejected
  wake nodes are rejected for a verifiable, non-threshold reason.
* Free-wake convergence for the CAD G2 case must **not** be claimed
  (issue #65 Phase B2 stays the blocking item).  The precondition is a
  regularized kernel whose core bias is quantified against the 48/96-order
  reference (item C).
* The `sampling_quadrature=0` number must be read as *not applicable* under
  `analytic_edges`; it is not evidence that quadrature is converged.

## 6. Files

* `tools/issue84/wake_node_rejection.py` — per-node table (CSV + JSON), stable
  ids, exact point-to-triangle wall distance, sheet-edge distance, duplicate
  neighbour distance, ladder trace for the two worst nodes.
* `tools/issue84/ladder_probe.py` — wide-h ladder (8x ... 1/32) for named
  nodes; the table in section 2.
* `runs/issue84/probe_decomp.py`, `runs/issue84/fan_check.py`,
  `runs/issue84/tjunction.py`, `runs/issue84/core_test.py` — the four
  exclusion/cure measurements quoted above.
