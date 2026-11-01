# Phase 1 step 2: a structured pressure-Kutta solver without the dense Jacobian SVD

Status: research implementation, opt-in (`pressure_solver='structured'`). The default
remains the reference `scipy.optimize.least_squares` path, the analytic pressure
residual and Jacobian are unchanged, and `physical_accuracy_accepted` stays false.

## 1. The measured need

Phase 0 (`docs/ISSUE92_PHASE0_BASELINE.md`) attributed the elliptic fixture's 22.89 s
pressure-Kutta solve as follows:

| stage | seconds | share |
| --- | --- | --- |
| `scipy.optimize.least_squares` total | 9.84 | 43% |
| dense Jacobian SVD inside it (13 calls, 0.71 s each) | 9.26 | 40% |
| influence-kernel assembly (`_doublet_matrix`, 1529 calls) | 12.21 | 45% |
| `lstsq(gelsd)` for the linear start | 0.51 | 2% |
| `null_space` | 0.18 | <1% |

The nonlinear stage is not slow because the linear systems are large: it is slow
because every iteration takes a dense SVD of a 3303 x 1645 Jacobian whose only
changing part is 83 pressure rows. The forward dense solve is 2% of the stage, which
is why Phase 1 does not move the LU to the GPU.

## 2. The structure exploited

Write the reduced problem as `min_z 0.5 * || r(z) ||^2` with
`r = [M z - b_w ; w * p(N z)]`, where `M = B N` is the weighted wall block in the
feasible subspace, `p` is the pressure-Kutta residual (83 rows on elliptic) and `w`
is `kutta_weight`. Then

- `M` and `G = M^T M` do not change: they are formed and factored once per solve;
- only `F = w * P(v) N` changes, and it has as many rows as the pressure block;
- the Gauss-Newton step solves `(G + F^T F) dz = -g` with
  `g = M^T r_w + F^T (w p)`, and the Woodbury identity reduces it to an
  `n_pressure x n_pressure` system
  `(I + F G^-1 F^T) y = F G^-1 g`, `dz = -G^-1 g + G^-1 F^T y`.

This is algebraically the same step the SVD computes, so the fixed point and the
objective are unchanged; only the linear algebra is reorganised. The reference
termination tests (ftol/xtol/gtol = 1e-11 on the same residual) and its Armijo
backtracking line search control convergence.

## 3. What is deliberately not changed

- the pressure residual itself and its analytic Jacobian (`pressure_jacobian`);
  no finite differences, no reduced-load model;
- `ftol = xtol = gtol = 1e-11` and the caller's `max_nfev`;
- the acceptance path: `solve` still returns a `scipy.optimize.OptimizeResult`, so
  `pressure_kutta_acceptance` and the `status`/floor semantics are untouched;
- default solver choice, default meshes, and every wake/velocity default.

## 4. Two numerical decisions that decide the result

### 4.1 Factor `M` by QR, not by a Cholesky of `M^T M`

`cond(M) = 534` on elliptic; `cond(M^T M) = 2.9e5`. A Cholesky of `M^T M` gives a step
noise of order `cond(M^T M) * eps`, and near the residual floor that noise is the same
size as the real step: the measured consequence was the cost stalling at
`1.30078201475e-4` while the gradient grew from `1.2e-8` back to `8e-3`, the Kutta
residual grew from `7.64e-7` to `2.3e-6`, and the run ended on `max_nfev` with
`CL = 0.3474268` against the reference `0.3472736` (4.4e-4 relative).

Factorising `M = Q R` and eliminating through the triangular factor keeps the same
Woodbury step but only needs triangular solves, so the error stays at `cond(M) * eps`.
Note the elimination order: `(R^T R)^-1 = R^-1 R^-T`, so `R^T` is solved first. Doing
it the other way round computes `(R R^T)^-1`, which is not a descent direction and
shows up as a line search that backtracks six times per iteration with the cost stuck
at `3.3e-4`.

### 4.2 Terminate on stationarity relative to the operator scale

`least_squares` stops the elliptic fixture at `nfev = 12` on its `ftol` test, with
`cost = 1.3007820147594e-4` and `optimality = 7.7e-9`. Reaching the same point with the
structured iteration required a stop test that does not depend on the absolute cost,
because once the wall block has a residual floor the cost keeps decreasing by ~1e-11
relative per iteration while the iterate slides along a shallow valley:

| iteration | nfev | cost | gradient inf | relative gradient |
| --- | --- | --- | --- | --- |
| 3 | 7 | 1.300782050660234e-04 | 1.527e-04 | 4.083e-06 |
| 6 | 13 | 1.300782015735058e-04 | 4.455e-07 | 1.191e-08 |
| 8 | 17 | 1.300782014870768e-04 | 6.036e-08 | 1.614e-09 |
| 9 | 19 | 1.300782014807288e-04 | 5.949e-08 | 1.591e-09 |
| 10 | 22 | 1.300782014767886e-04 | 4.042e-08 | 1.081e-09 |
| 11 | 24 | 1.300782014685656e-04 | 7.349e-08 | 1.965e-09 |
| 12 | 26 | 1.300782014234003e-04 | 5.203e-08 | 1.391e-09 |
| (drift, old rule) 36 | 100 | 1.300689731128874e-04 | 8.204e-03 | n/a |

The relative gradient is `||g||_inf / (||J||_F * ||r||)` with
`||J||_F^2 = ||M||_F^2 + ||F||_F^2`, i.e. the same quantity the acceptance helper
already uses (`gradient_relative`), only far tighter. The solver stops when it reaches
`relative_gtol = 1e-9`, or when two consecutive iterations fail to improve the best
relative gradient by more than 5% (`stagnation_gate = 1e-6`, `stagnation_factor = 1.05`).
The second rule is what stops the drift: on elliptic the best relative gradient is
`1.081e-9`, just above the absolute threshold, and the next two iterations are both
worse, so the run stops at `nfev = 26` with a cost `2e-12` relative below the reference
and `CL` within `6.6e-8` relative.

## 5. Termination status codes

| status | meaning |
| --- | --- |
| 1 | absolute gradient tolerance `gtol` reached |
| 2 | `ftol` cost-reduction test satisfied |
| 3 | `xtol` step-size test satisfied (or the step was not a descent direction) |
| 4 | relative gradient tolerance reached |
| 5 | relative gradient stagnated (two consecutive non-improvements) |
| 0 | evaluation budget exhausted, or the line search could not reduce the residual |

`success` is `status > 0`, matching `least_squares`; `status = 0` means the budget or
the line search ended the run, which is the same situation the reference reports.

## 6. A/B over the eight frozen fixtures

`docs/issue92/pressure_solver_bench.json`, `repeat = 1`, `max_nfev = 100` (80 for the
multi-body cases), same fixtures and digests as Phase 0 (`fixture_verified` in the
JSON). `nonlin` is the nonlinear stage only, so the assembly work is excluded and the
comparison is about the stage this change replaces.

| fixture | reference nfev / nonlin (s) | structured nfev / nonlin (s) | stop | dCL rel | speedup nonlin / total |
| --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 4 / 0.006 | 7 / 0.002 | 2 | -3.28e-14 | 2.46x / 1.15x |
| g1_wing_camber | 10 / 0.008 | 7 / 0.002 | 4 | -3.13e-14 | 3.46x / 1.18x |
| g2_conventional | 5 / 0.057 | 7 / 0.005 | 4 | +2.43e-09 | 10.60x / 1.70x |
| g2_tail_low | 5 / 0.058 | 7 / 0.005 | 4 | +2.80e-09 | 11.29x / 1.77x |
| g1_wingbody_cad | 15 / 1.144 | 29 / 0.162 | 2 | -4.92e-10 | 7.07x / 1.94x |
| g1_swept30 | 7 / 3.949 | 13 / 0.457 | 2 | +2.79e-14 | 8.64x / 1.56x |
| g1_wingbody_cad_3x | 17 / 7.038 | 26 / 0.468 | 2 | -2.62e-07 | 15.05x / 2.20x |
| g1_elliptic | 12 / 9.628 | 26 / 0.933 | 5 | +6.56e-08 | 10.31x / 1.94x |

Pressure-Kutta residuals agree to the last printed digit everywhere (worst case
`7.642e-7` against `7.641e-7` on elliptic). The worst load deviation is `2.6e-7`
relative on `g1_wingbody_cad_3x`, where the reference stops on `ftol` at a different
point of the same flat valley; costs agree to `2e-12` relative or better and are never
higher than the reference.

Two honest caveats. First, the small fixtures barely gain (the assembly dominates and
the structured solve takes three extra evaluations); the gain is on the meshes that
matter, 7-15x from the cad fixture upwards, or 2.5-15x across all eight. Second,
absolute seconds are not portable between runs on this machine; only ratios measured
inside one run are, which is why this table was regenerated on the branch that ships
with it.

## 7. Where the step loses digits

The Woodbury correction subtracts two large terms, so its accuracy degrades as the
wall block stiffens. Measured against `numpy.linalg.lstsq` on the stacked `[M ; F]`
system, the step error is `1.6e-9` at `cond(M) = 117` and `2.9e-6` at
`cond(M) = 1.17e4`. The condition number is reported as `condition_number_M` so a
caller can see when the step is no longer trustworthy. At the production conditioning
the step formed by this solver agrees with the reference dense-SVD step to `3.16e-7`
relative (measured on elliptic by `tools/issue92/pressure_structure_probe.py`), which
is the level the eight-fixture A/B confirms. This is a limitation, not a property to
rely on: the tests in `test/test_pressure_kutta_reduced.py` pin both ends of it.

## 8. Tests

`test/test_pressure_kutta_reduced.py`, 10 tests:

- the step equals `numpy.linalg.lstsq` on the stacked `[M ; F]` system, and the
  gradient equals `[M ; F]^T r`;
- a moderately ill-conditioned wall block still matches, and a stiffer one is
  measurably worse (so the degradation is pinned rather than hidden);
- rank-deficient and malformed inputs are refused with the documented messages;
- a linear problem converges to the least-squares solution;
- a residual floor terminates on the relative tolerance when `ftol`/`xtol` are off;
- the evaluation budget is respected;
- `solve(..., pressure_solver='structured')` and `solve_multi(...)` reproduce the
  reference loads on a small G1 and a small G2 case, and reject unknown solvers.

## 9. Non-claims

- Not bit-identical to the reference: the iteration path differs, so loads agree to
  `2.6e-7` relative at worst, not to machine precision.
- Not a smaller problem: the objective, the residual, the Jacobian and the
  tolerances are the same; only the linear algebra of the step changed.
- Not a new default: `pressure_solver='reference'` is still what the production
  callers use unless they ask otherwise.
- Not a physical-accuracy statement: `physical_accuracy_accepted` remains false.

## 10. Reproduction

```
set PYTHONPATH=<worktree>;D:/3D/Projects/panel/runs
python -m tools.issue92.pressure_structure_probe --fixture g1_elliptic
python -m tools.issue92.pressure_solver_bench --fixtures g1_elliptic --repeat 1
python -m tools.issue92.pressure_iteration_diag --fixture g1_elliptic --solver structured
python -m tools.issue92.pressure_summary
```

## 11. Next

Phase 1 step 3 profiles and vectorises the induced-velocity evaluator on the real
18992-target coupling update (57.73 s per iteration, 3.0 ms per point against a 0.0045
ms per point arithmetic floor), then the already-verified batched influence kernel and
that evaluator are the only candidates for a CUDA port.
