# G2 pressure-Kutta initial solve: why it was reported blocked

Context: PR #85 (issue #84) could not run its G2 single-variable comparison because
python -m panel.run_p1_coupled_material_wake --case g2 --iterations 0 returned
status=blocked_initial_nonlinear_solve, exit_code=2 after 13.94 s. Cloud review
6514806741 asked to separate constraint/null-space, pressure residual scale, initial
guess and CAD mesh topology before touching max_nfev or any default.

Verdict: **none of the four**. The stop is a budget/tolerance mismatch on a residual
that has a genuine least-squares floor. The state it stops at is the same state a
longer budget converges to, bit for bit.

## Reproduction is exact

runs/issue84/g2_rootcause.py rebuilds the CLI mesh (main wing from
runs/half_cad_wingbody_v1/surface.json plus make_wing(chord_panels=4, span_panels=4,
span=1.8, chord=.55, wake_panels=3) at (1.6, 0, .35)), re-implements pressure/jac
exactly as panel/p1_lifting_g2.solve_multi builds them, and reruns the production
least_squares call:

| quantity | value |
|---|---|
| wall rows / null(J) DOFs | 872 / 487 |
| mu, gamma vs runs/issue84/g2_ref/iteration_000.npz | max abs diff **0.0 / 0.0** |
| residual at x0 (signed-speed linear solution) | 4.0073 |
| residual at nfev 80 | 0.038057 |
| wall_rms / pressure_rms at nfev 80 | 1.2888e-3 / 9.0265e-7 |
| least-squares floor of the wall block in null(J) | 1.2688e-3 |
| wall floor in null(J,K) | 1.2896e-3 |
| relative projected gradient at nfev 80 | 2.19e-7 |

The recorded pressure_kutta_rms in iteration_000.json is 9.026496367145126e-07,
identical to the probe pressure_rms to the last digit.

## Four hypotheses, measured

**(a) constraint / null-space.** J (2 TE pairs plus contact jumps) leaves
null_space(J) = 487 columns; null_space(vstack(J,K)) = 396. The nonlinear system
Jacobian at the stop is 963 x 487, **full rank 487**, sv_max=1.80e3, sv_min=7.64e-3,
cond=2.35e5. The linear constrained solve has rank=396, sv_min=7.56e-3. No
deficiency, no near-null mode, no gauge problem.

**(b) pressure residual scale.** The Kutta rows are satisfied: pressure_rms reaches
9.0e-7 (dimensionless speed-squared difference) at nfev 80. The cost is dominated by
the wall block, whose floor is structural: 872 collocation rows against 487 unknowns.
Weight sweeps from the same start: kut_weight=0.5 -> success in 36 nfev
(pressure_rms=6.1e-5), 1.0 -> success in 41 nfev (1.8e-5), 5 (production) -> reported
fail at 80, 50 -> reported fail at 80 with pressure_rms=4.2e-8. Weights only matter
when gtol becomes reachable; the wall residual stays 1.2888e-3 in every case.

**(c) initial guess.** Production starts from the signed-speed linear solution
projected into null(J); x0=0 ends at cost=0.163, wall_rms=1.93e-2, far worse.
The projected linear solution is already the good start.

**(d) CAD mesh topology.** TE sampling faces are ordinary triangles (area_min=1.77e-4,
aspect median 0.56, p90 1.26, max 8.6 in the equilateral-normalised metric), and the
worst pressure pairs sit at the wing/tail TE with area-balanced faces. No sliver
drives the stop.

## What actually stops it

With the same tolerances but max_nfev=200 the very same call **succeeds at nfev=85**
with ftol termination and identical numbers (cost=7.24171408e-4, wall_rms=1.2888e-3,
pressure_rms=9.026e-7). The cost trace is 4.0073, 1.0256, 0.2444, 0.1026, 0.0493,
then a plateau at 0.038057: from nfev ~10 onwards each evaluation buys a reduction of
only ~1.8e-13 (relative 5e-12), i.e. roundoff-sized polishing towards a non-zero
least-squares minimum. gtol=1e-11 compares an absolute gradient (3.37e-5, relative
2.2e-7) with a value that cannot be reached on a floored residual, and
ftol=1e-11 * cost = 7.2e-15 sits below that per-evaluation noise. The production
budget of 80 evaluations is five short of the point where ftol finally fires.

So the physics is fine and the budget is not. Raising max_nfev alone would still be
fragile (85 vs 80 is a five-evaluation margin that depends on the mesh), so the fix
classifies the stop instead.

## Fix (this PR)

panel/p1_lifting_g2.py gains pressure_kutta_acceptance(...), called with the
least_squares result plus the residual, Jacobian, wall operator and Kutta residual.
A stop is accepted as converged when **all** of:

1. least_squares status is 0 (a termination condition, not a numerical failure);
2. the relative projected gradient ||J^T r||_inf / (||J||_F ||r||) is below 1e-6
   (scale-free stationarity);
3. the Kutta rows satisfy pressure_rms <= 1e-4;
4. the wall residual is within 5% of the floor of the linear subproblem
   min ||B null(J) z - bw||, computed with one gelsd on the same matrix.

The result dictionary keeps success for all existing consumers and adds status,
accepted_at_residual_floor, gradient_relative, wall_rms, wall_floor_rms,
wall_over_floor, pressure_rms, criteria, tolerances. Nothing else changes: no default
weight, no tolerance, no velocity_method, no physical_accuracy_accepted.

Measured effect on the reported case, --case g2 --iterations 0:

| quantity | before | after |
|---|---|---|
| status | blocked_initial_nonlinear_solve | iteration_limit |
| exit_code | 2 | 0 |
| nonlinear.success | false | true (accepted_at_residual_floor=true) |
| CL / CL_gamma / CD_pressure / CM_y | 0.43406115014632407 / 0.43843173700189036 / 0.006357145692946399 / -0.11965433057432534 | identical |
| mu, gamma, cp snapshots | - | max abs diff **0.0 / 0.0 / 0.0** |

--case g1 is unchanged (iteration_limit, exit 0).

## Remaining, explicitly not fixed here

iteration_limit is now the true status: the free-wake loop still has
eligibility root=90, eligible=276, convergence=102, rejected=102 and
node_epsilon_error_max=0.1934 on this CAD case, i.e. the wake update does not reach
the geometric gates. That is issue #65 Phase B2 work, separate from this nonlinear
stop, and it is why G2 still cannot serve as a converged single-variable comparison.
physical_accuracy_accepted=false everywhere.
