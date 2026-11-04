# Issue 102: long material links as explicit truncation boundaries

Status: implementation experiment committed on the curved-TE branch; local validation required.

## Evidence and failure mode

The Issue #84 G2 CAD ledger records material predecessor links up to 26 chord lengths against a median near 0.666 chord. The existing rooted arclength Picard map advances one reference predecessor link per node. A link this long is not a physical interior panel interval; it is a truncation/attachment boundary in the material graph. Applying the normalized streamline tangent across it requests a displacement proportional to that anomalous length, after which the geometric step cap clips many nodes and hides the bad graph.

## Change in this branch

`wake_step_scale` already identifies links exceeding four times the median positive predecessor-link length. The coupled wake update now defaults to `truncation_policy="boundary"`:

- A flagged node is held at its current coordinate during the kinematic map.
- The following node starts a new streamline path from that held coordinate.
- Velocity averaging does not cross the flagged predecessor edge.
- Flagged nodes remain in the geometry audit and in eligibility accounting; they are not silently removed from collision, Jacobian, area or contact checks.
- The fixed-point residual is evaluated on interior material nodes only. Boundary IDs, count, link lengths, treatment and per-group interior residual are recorded in the iteration ledger and snapshot arrays.
- `advance` and `freeze` remain explicit compatibility/reproduction modes.

This is a graph-boundary treatment, not a claim that the underlying material graph has been remeshed or that the boundary condition is physically validated. The four-times-median threshold is a diagnostic heuristic and must be checked against the G2 CAD graph before being accepted as a physical attachment classification.

## What this does not solve

This change does not repair finite-thickness trailing-edge wake/wall transverse penetration. The #84 T5 ledger reports persistent transverse contact pairs and a maximum depth of 0.0333 chord. That requires a geometry-resolved wake attachment treatment or a validated attachment corridor; contact tolerances must not be widened to suppress it.

It also does not change the P1 induction model. `analytic_edges` remains the reference; `rm_regularized_p1` still requires same-point off-sheet comparison, core and area-quadrature refinement, and cannot replace the P1 boundary plus in-panel area-vorticity contributions with constant-strength filaments.

## Local-agent validation required

1. Run the new long-link unit test in `test/test_p1_coupled_material_wake.py`.
2. Re-run the exact G2 CAD T5 reproduction with `truncation_policy="boundary"`; report raw predecessor-link distribution, flagged IDs, interior residual, accepted displacement, clipped IDs/fraction and full geometry gates.
3. Compare `advance`, `freeze` and `boundary` on the same frozen input. Do not compare different mesh or state snapshots.
4. Confirm no change in wall solve, circulation solve, P1 velocity evaluator, physical loads at the fixed initial geometry.
5. Retain the transverse penetration IDs/depths; this change must not make them disappear by relaxing contact tolerance.
6. Run focused tests and full suite; record environment, commit SHA, exact commands, exit codes and known failures.

No production default outside the coupled material-wake experiment changes. `physical_accuracy_accepted=false`.
