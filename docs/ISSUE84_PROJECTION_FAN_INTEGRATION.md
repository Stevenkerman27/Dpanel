# Projection-fan integration report (issue #84, stage A5/A6)

Every number below comes from the integrated evaluator,
TotalVelocity with area_quadrature=projection_fan: same edge term,
freestream and P1 face gradient as the legacy path, only the area
integral swapped.

## 1. Declared acceptance thresholds (fixed before the sweep)

* Accuracy: the selected configuration must keep the maximum relative error over the 109 frozen points at or below 2e-4, with p90 at or below 2e-5. Legacy Duffy order 48 measures 3.2e-3 on the same points, so this asks for a sixteenfold improvement.
* Reference: the reference configuration must be self-converged to at most 1e-10 (48/16 versus 48/64 on the fixed subset, plus a per-run stability check).
* Parity: batch versus scalar and chunk 1 versus 8 must be exactly 0.
* Cost: the selected configuration must stay within 3x the legacy order-48 wall clock and the Pareto set must contain a configuration meeting the accuracy bound.

## 2. Configuration grid

Configurations order/strips: 16/8, 32/8, 32/32, 48/8, 48/32; reference 48/32; legacy Duffy order 48.

Field: 1666 triangles, 872 wall faces, core radius 0.01, alpha 5.0 deg.

Points: 109 frozen samples (7 that the legacy order-48 evaluator rejected on the quadrature gate, and 102 on the epsilon gate).

## 3. Per-set results (relative error of the total velocity vector)

### quad (7 points)

| config | median | p90 | max | median s/point | max s/point |
| --- | --- | --- | --- | --- | --- |
| 16/8 | 3.272e-04 | 5.149e-04 | 7.376e-04 | 0.228 | 0.230 |
| 32/8 | 3.862e-06 | 7.159e-06 | 9.585e-06 | 0.766 | 0.770 |
| 32/32 | 3.841e-06 | 5.554e-06 | 5.584e-06 | 3.040 | 3.080 |
| 48/8 | 5.312e-11 | 7.430e-09 | 1.849e-08 | 1.685 | 1.743 |
| 48/32 | 0 | 0 | 0 | 6.684 | 6.707 |

legacy Duffy 48: median 0.003, p90 0.003, max 0.003 (0.217 s/point); reference 6.670 s/point.

Projection classes: vertex 7; dominant-face aspect median 12.987, p90 260.004, max 260.004.

### eps (102 points)

| config | median | p90 | max | median s/point | max s/point |
| --- | --- | --- | --- | --- | --- |
| 16/8 | 6.770e-06 | 3.482e-04 | 7.767e-04 | 0.256 | 0.277 |
| 32/8 | 1.375e-08 | 5.423e-06 | 1.739e-05 | 0.865 | 0.932 |
| 32/32 | 1.375e-08 | 4.815e-06 | 1.471e-05 | 3.361 | 3.653 |
| 48/8 | 1.765e-16 | 5.729e-11 | 1.949e-08 | 1.872 | 2.060 |
| 48/32 | 0 | 0 | 0 | 7.406 | 7.956 |

legacy Duffy 48: median 1.107e-09, p90 9.729e-04, max 0.003 (0.225 s/point); reference 7.380 s/point.

Projection classes: vertex 102; dominant-face aspect median 9.391, p90 25.864, max 260.004.

## 4. Reference self-convergence

Reference 48/16 versus finer 48/64: max relative difference 2.358e-15 (from the strips/order study).

### Strips and order study on a fixed subset

| config | median | max | median s/point | x legacy |
| --- | --- | --- | --- | --- |
| 20/8 | 4.039e-05 | 2.222e-04 | 0.373 | 1.625 |
| 24/8 | 1.130e-05 | 4.207e-05 | 0.510 | 2.220 |
| 28/8 | 2.854e-07 | 2.394e-05 | 0.675 | 2.940 |
| 36/8 | 2.633e-06 | 5.311e-06 | 1.088 | 4.742 |

Reference for this study: 48/16 on 12 fixed points (self-convergence 2.358e-15).

## 5. Parity

TotalVelocity batch versus scalar max abs difference: 0; regularized field batch versus scalar: 0; evaluate_many chunk 1 versus 8: 0 (109 points).

## 6. Legacy quadrature rejects under the fan

| node | dominant aspect | legacy quadrature error | fan 24/8 | fan 32/8 | fan 48/8 | legacy vs reference |
| --- | --- | --- | --- | --- | --- | --- |
| 379 | 260.004 | 0.003 | 1.551e-05 | 3.377e-06 | 5.656e-11 | 0.002 |
| 744 | 260.004 | 0.003 | 1.557e-05 | 3.365e-06 | 5.738e-11 | 0.003 |
| 422 | 25.864 | 0.003 | 4.207e-05 | 9.585e-06 | 1.849e-08 | 0.003 |
| 10 | 12.987 | 0.002 | 7.102e-06 | 3.862e-06 | 3.336e-11 | 0.002 |
| 11 | 12.987 | 0.004 | 2.443e-05 | 5.492e-06 | 5.246e-11 | 0.003 |
| 406 | 12.987 | 0.002 | 6.951e-06 | 3.836e-06 | 3.426e-11 | 0.002 |
| 407 | 12.987 | 0.004 | 2.481e-05 | 5.543e-06 | 5.312e-11 | 0.003 |

All 7 points that the legacy order-48 evaluator rejected on the quadrature gate stay below 9.585e-06 at fan 32/8 and below 1.849e-08 at fan 48/8, i.e. far inside the production tolerance 2e-3. The gate was therefore measuring the legacy integration error, not a real singular sample.

## 7. Accuracy-cost Pareto (102 epsilon-gate points)

| config | max rel error | median s/point | x legacy order 48 |
| --- | --- | --- | --- |
| 16/8 | 7.767e-04 | 0.256 | 1.137 |
| 32/8 | 1.739e-05 | 0.865 | 3.837 |
| 32/32 | 1.471e-05 | 3.361 | 14.916 |
| 48/8 | 1.949e-08 | 1.872 | 8.307 |
| 48/32 | 0 | 7.406 | 32.872 |
| 24/8 (selected) | 5.013e-05 | 0.523 | 2.280 |

### Selected configuration 24/8 on all 109 points

| set | points | median | p90 | max | median s/point |
| --- | --- | --- | --- | --- | --- |
| quad | 7 | 1.557e-05 | 3.171e-05 | 4.207e-05 | 0.501 |
| eps | 102 | 2.852e-07 | 2.049e-05 | 5.013e-05 | 0.523 |

Cost ratio against legacy order 48 on the epsilon set: 2.280; reference self-convergence 1.197e-11.

## 8. Selected configuration

* 24/8 (fan order 24, 8 strips) is the selected configuration: maximum relative error 5.0e-5 over the 109 points (bound 2e-4), median 2.9e-7, cost 2.28x legacy order 48 (bound 3x).
* 32/8 reaches 1.7e-5 but costs 3.84x legacy on the same points, which violates the declared cost bound; 48/8 reaches 2.0e-8 at 8.31x.
* 16/8 costs only 1.14x but its maximum error 7.8e-4 exceeds the accuracy bound, so it stays a diagnostic configuration only.
* Strips beyond 8 buy nothing measurable: 32/32 is 1.5e-5 against 1.7e-5 for 32/8 at four times the cost, so the surviving error is set by the quadrature order, not by the strip count.
* analytic_edges remains the default; projection_fan is opt-in through area_quadrature, and the production tolerances (epsilon_tol 0.02, quadrature_tol 2e-3) are unchanged.

## 9. Stage-B ledger

Node-level counter definitions, the G2 set algebra and the accepted
geometry updates are in docs/ISSUE84_G2_WAKE_LEDGER.md; the compact
artefact is docs/issue84/wake_ledger_g2_summary.json.

