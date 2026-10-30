# Issue #84 — RM-regularized P1 wake velocity reconstruction

Status: opt-in research mode. No production default changes. `physical_accuracy_accepted=false`.

## Why this is not a constant-strength filament replacement

The continuous P1 doublet-sheet velocity decomposition contains two distinct pieces:

1. Boundary and declared-cut vortex-filament terms. Their edge strength is the linearly varying P1 trace, so the exact line integral needs both its zeroth and first edge moments.
2. An in-panel distributed-vorticity term, with face-wise vector strength
   \[
   \boldsymbol\gamma_f=\nabla_s\mu_f\times\mathbf n_f.
   \]
   Its induced velocity is an area integral over the triangle.

Dropping the second term turns the P1 sheet into a different model. This is why PR #83's constant-strength filament kernel is not a drop-in replacement for the P1 wake.

## Regularized P1 decomposition

For explicit core radius \(a>0\), use the same Rosenhead–Moore denominator in both pieces:

\[
\mathbf u_a(\mathbf P)=\mathbf U_\infty-\frac{1}{4\pi}
\left[
\sum_{e\in\partial S}
\int_e\frac{\mu(s)\,\mathbf t_e\times(\mathbf P-\mathbf X)}
{\left(|\mathbf P-\mathbf X|^2+a^2\right)^{3/2}}\,ds
+
\sum_f\boldsymbol\gamma_f\times
\int_{T_f}\frac{\mathbf P-\mathbf X}
{\left(|\mathbf P-\mathbf X|^2+a^2\right)^{3/2}}\,dS
\right].
\]

Opposite-oriented traces on a shared continuous P1 edge cancel before evaluating the edge integral. A true potential cut retains its jump. The edge integral is analytic, including the linear-strength first moment. The area integral uses a Duffy-mapped tensor-product Gauss–Legendre rule. The public default is 48 points per Duffy coordinate. In the B2 batch path, far faces use an 8-point rule and faces within `2*circumradius + 2*a` of a target are recomputed at the requested high order. This is a performance heuristic, not an error estimator; the selected order must be checked against a higher-order run.

This is a regularized P1 **velocity reconstruction** for the wake-geometry iteration. The existing \(\mu/\Gamma\) solve and its \(A/J/K/W\) potential operators remain unregularized. Consequently this experiment is not a fully regularized BEM formulation and does not claim a self-consistent viscous-core solution. It is an A/B model study that asks how an explicit near-sheet core changes the kinematic wake update while holding the strength solve fixed.

## Code

- `panel/p1_regularized_sheet_velocity.py`: RM-regularized P1 edge plus in-panel gradient velocity field.
- `panel/p1_coupled_material_wake.py`: new opt-in `velocity_method="rm_regularized_p1"`; requires explicit `core_radius`.
- `panel/run_p1_coupled_material_wake.py`: CLI exposes `--velocity-method rm_regularized_p1 --core-radius <length>`.
- `tools/issue84/compare_iteration0.py`: verifies same frozen geometry, strengths, alpha and Kutta before comparing iteration-0 velocity samples and slip.
- `test/test_p1_regularized_sheet_velocity.py`: independent edge/area quadrature, P1 gradient, edge cancellation, zero-core limit and on-sheet finiteness.
- `test/test_p1_coupled_material_wake.py`: integration tests for explicit-core selection and unchanged strength solve.

The default remains `analytic_edges`. The old `legacy_gauss` mode remains available for historical A/B checks. No core is inferred or silently selected. RM-P1 uses an explicit positive core and a default high-order Duffy rule of 48; the B2 batched evaluator uses far-order 8 plus high-order recomputation for near faces. Do not use order 12 for near-wake core sweeps.

## Required tests

~~~bash
python -m pytest test/test_p1_regularized_sheet_velocity.py test/test_p1_coupled_material_wake.py -q
python -m py_compile panel/p1_regularized_sheet_velocity.py panel/p1_coupled_material_wake.py panel/run_p1_coupled_material_wake.py tools/issue84/compare_iteration0.py
~~~

No current-branch local execution receipt is available yet. The consolidated branch now also includes batch evaluation, replay overrides, order reporting, a tolerant floating-point gradient assertion and an explicit O(a^2) zero-core regression. Do not report these as passed until the local agent checks out the final PR head and returns the actual environment, command, exit codes and output.

## Generated G1/G2 comparison plan

Run the same generated geometry, alpha, Kutta and reference quantities with `iterations=0`. The strength solve is intentionally unchanged; compare the saved iteration-0 states and velocity samples.

Example G1:

~~~bash
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 0 --velocity-method analytic_edges --out runs/issue84/g1_analytic
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 0 --velocity-method rm_regularized_p1 --core-radius 0.01 --quadrature-order 48 --out runs/issue84/g1_rm_a001
python -m tools.issue84.compare_iteration0 --reference runs/issue84/g1_analytic --candidate runs/issue84/g1_rm_a001 --out runs/issue84/g1_a001_compare.json
~~~

Repeat for G2 and for a declared sensitivity set, e.g. \(a/c=0.005,0.01,0.02\). These are sensitivity values, not a claim that one is physically calibrated. Do not select a core because it moves \(C_L\) toward a reference number.

Only after iteration-0 comparisons pass should short 3–4 iteration geometry updates be run for G1 and G2 at the same core values. Preserve every run directory; never overwrite earlier receipts. Report slip, epsilon/area-quadrature sensitivity, rejected nodes, geometry updates, \(\mathrm{CL}_p\), \(\mathrm{CL}_\Gamma\), Kutta residuals and the full iteration history. A finite or bounded run is not automatically converged.

Real CAD is optional and may only be run if `runs/half_cad_wingbody_v1/surface.json` exists in the local checkout. Record its SHA-256; do not substitute an invented fixture.

## Acceptance boundary

- The regularized field must preserve the in-panel P1 gradient and true trace cuts.
- As \(a\to0\), off-sheet values must approach the existing analytic P1 field.
- Positive-core on-sheet samples must remain finite.
- Iteration-0 solved strengths and loads must be identical across velocity methods to solver tolerance because the BEM solve is unchanged.
- The velocity and geometry-update differences are the subject of the experiment.
- No claim of physical free-wake convergence or absolute aerodynamic accuracy is permitted from this PR alone.


## Independent near-sheet accuracy envelope (single-triangle audit)

The local independent audit measured the following **total velocity** relative errors for core (a=0.05), using a refined composite-Duffy reference. Here (d) is target-to-sheet distance in the same geometry length unit:

| (d) | Duffy order 16 | Duffy order 48 | Duffy order 96 |
|---:|---:|---:|---:|
| 0.10 | 0.40% | 2.3e-8 | 2.8e-15 |
| 0.05 | 3.40% | 1.6e-5 | 1.8e-10 |
| 0.02 | 8.60% | 2.6e-4 | 4.8e-8 |
| 0.01 | 10.28% | 4.3e-4 | 1.3e-7 |

This table is specific to the tested triangle, core and offsets. It is not a universal bound for CAD meshes. The area term alone can have much larger relative error because it partially cancels the edge term. The acceptance metric is the **total vector velocity**, not the isolated area term. For any new geometry/core combination, compare at least order 48 against 96 at representative closest-offset samples.

## Frozen-state replay for G2

A failed pressure-Kutta nonlinear solve must not be used as a valid A/B baseline. The B2 runner now accepts:

~~~bash
python -m panel.run_p1_coupled_material_wake --replay-snapshot <iteration_000.npz> --expected-state-sha <sha256> --replay-velocity-method rm_regularized_p1 --replay-core-radius 0.01 --replay-quadrature-order 48 --out runs/issue84/g2_replay_rm
~~~

This re-evaluates the alternate velocity on the exact saved (mu,Gamma), geometry, alpha and material labels; it does not re-solve the pressure-Kutta system. A successful finite replay is a velocity-kernel comparison only. It does not make the failed nonlinear state a converged G2 solution. The receipt identifies the recorded and evaluated methods, core radius, order and frozen unregularized strength-solve provenance.
