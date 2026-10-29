# Issue #71 — exact TE potential-jump integral audit

This branch adds a diagnostic only. It does not replace pressure Kutta or
change production solver defaults.

## Important distinction

The existing algebraic relation is

    mu_upper(y_i) - mu_lower(y_i) = Gamma(y_i).

It is a potential-jump/wake-strength compatibility relation. Integrating it,

    integral_TE (mu_upper - mu_lower) ds = integral_TE Gamma ds,

does not create an independent physical Kutta condition; it is a consequence
of the same jump relation when the traces and Gamma use the same piecewise
linear partition. The pressure Kutta condition instead compares appropriate
upper/lower limiting speeds at a sharp trailing edge (or uses a separately
derived edge-singularity treatment). These residuals must always be reported
as separate quantities.

## Why the scalar integral is insufficient

A signed integral can be zero even if the integrand is not zero: positive and
negative TE jump errors can cancel. For that reason
panel/p1_te_jump_integral.py constructs the exact common refinement of the
wall TE trace stations and Gamma material knots, then reports:

- nodal residuals at every wall TE station;
- every Gamma interpolation knot lying inside a geometric TE edge;
- exact signed integral of the jump residual;
- exact L2 residual and maximum pointwise residual;
- P1 weak moments and per-wall-edge residual summaries;
- a deliberate state-perturbation diagnostic in the G1/G2 runner.

On each common-refinement interval the wall jump and Gamma are linear, so for
endpoint residuals r0/r1 and interval length h,

    integral r ds = h (r0+r1)/2
    integral r^2 ds = h (r0^2 + r0*r1 + r1^2)/3
    integral (1-t) r ds = h (2*r0+r1)/6
    integral t r ds = h (r0+2*r1)/6.

No numerical quadrature is needed for these diagnostics.

## Running the actual G1/G2 states

    python -m pytest test/test_p1_te_jump_integral.py -q
    python tools/issue71/te_jump_integral.py --chord-panels 4 8 12 16 24 --methods pressure signed_speed --out runs/issue71_te_jump_integral.json

The report includes pressure_kutta_rms, signed_kutta_rms, jump_rms,
the TE jump integral/L2/max residual, and a deliberately perturbed Gamma
case. This shows whether the integrated jump is merely reproducing the
already-enforced J equation while pressure Kutta remains nonzero or
mesh-sensitive.

## Interpretation

- If wall TE knots and Gamma knots align and J is exact, the integral and L2
  jump residuals should both be roundoff-small. This is a consistency check,
  not evidence of Kutta satisfaction.
- If Gamma knots fall inside a TE wall edge, nodal J at the endpoints does
  not prove pointwise equality in the interior. Use a compatible TE mesh or a
  separately derived mortar/weak trace condition; do not simply assert the
  integral is enough.
- If signed integral is small but L2/max residual is not, cancellation hides
  a local mismatch.
- If jump compatibility passes but pressure Kutta remains unstable, the next
  issue is how the circulation is selected, not the definition of the jump.
- Keep all original CL, CL_gamma, velocity and wall residuals. A small jump
  integral alone is not an acceptance criterion.

## Sandbox scope

The closed-form segment formula was independently exercised against four
manufactured cases: aligned traces, unmatched interior Gamma knot, cancellation
of signed residual, and partition-of-unity identity for the weak moments.
An isolated copy of the diagnostic module and its five pure-function tests was actually run in the sandbox: **5 passed / exit 0 in 0.06 s**. The G1/G2 adapter integration tests, full repository tests and actual runner have not been executed in this sandbox because it cannot clone the repository; local execution of the commands above remains required. Receipt: /mnt/data/dpanel_te_jump_integral_sandbox_receipt.json.


## Local PR #73 acceptance findings and fixes (2026-10-29)

The local review confirms the closed-form integral and L2 formulas against
independent refined quadrature (signed integral relative difference 0; L2^2
relative difference 5.9e-8, consistent with the reference midpoint quadrature
error). It also found two integration bugs, now fixed in this branch:

- G1 adapter now uses actual `Wing.te_up` and `Wing.te_down` fields; its
  test was corrected accordingly.
- JSON conversion now handles `np.bool_`; the CLI test checks JSON
  round-trip. Per-row nonlinear success and nfev are explicit, and a failed
  solve at the 200-evaluation cap is labelled `iteration_limit`.

The local sweep is the critical scientific result: with the jump residual
at roundoff (about 1e-16), pressure-Kutta RMS was about 2.8e-8 to 1.3e-7
at chord counts 4/8/12/16, while CL changed from +0.457 to -0.450.
Thus **small pressure-Kutta residual is not a circulation-validity
criterion**. It only says the discrete nonlinear condition has been fitted.
It does not certify convergence of the underlying surface velocity field,
identifiability of Gamma, or physical accuracy. The c=24 pressure solve hit
nfev=200 without success and must not be treated as a converged sample.
