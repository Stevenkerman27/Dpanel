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

Opposite-oriented traces on a shared continuous P1 edge cancel before evaluating the edge integral. A true potential cut retains its jump. The edge integral is analytic, including the linear-strength first moment. The area integral uses a Duffy-mapped tensor-product Gauss–Legendre rule.

This is a regularized P1 **velocity reconstruction** for the wake-geometry iteration. The existing \(\mu/\Gamma\) solve and its \(A/J/K/W\) potential operators remain unregularized. Consequently this experiment is not a fully regularized BEM formulation and does not claim a self-consistent viscous-core solution. It is an A/B model study that asks how an explicit near-sheet core changes the kinematic wake update while holding the strength solve fixed.

## Code

- `panel/p1_regularized_sheet_velocity.py`: RM-regularized P1 edge plus in-panel gradient velocity field.
- `panel/p1_coupled_material_wake.py`: new opt-in `velocity_method="rm_regularized_p1"`; requires explicit `core_radius`.
- `panel/run_p1_coupled_material_wake.py`: CLI exposes `--velocity-method rm_regularized_p1 --core-radius <length>`.
- `tools/issue84/compare_iteration0.py`: verifies same frozen geometry, strengths, alpha and Kutta before comparing iteration-0 velocity samples and slip.
- `test/test_p1_regularized_sheet_velocity.py`: independent edge/area quadrature, P1 gradient, edge cancellation, zero-core limit and on-sheet finiteness.
- `test/test_p1_coupled_material_wake.py`: integration tests for explicit-core selection and unchanged strength solve.

The default remains `analytic_edges`. The old `legacy_gauss` mode remains available for historical A/B checks. No core is inferred or silently selected.

## Required tests

~~~bash
python -m pytest test/test_p1_regularized_sheet_velocity.py test/test_p1_coupled_material_wake.py -q
python -m py_compile panel/p1_regularized_sheet_velocity.py panel/p1_coupled_material_wake.py panel/run_p1_coupled_material_wake.py tools/issue84/compare_iteration0.py
~~~

No current-branch local execution receipt is available yet. Do not report these as passed until the local agent checks out the exact PR head and returns the actual environment, command, exit codes and output.

## Generated G1/G2 comparison plan

Run the same generated geometry, alpha, Kutta and reference quantities with `iterations=0`. The strength solve is intentionally unchanged; compare the saved iteration-0 states and velocity samples.

Example G1:

~~~bash
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 0 --velocity-method analytic_edges --out runs/issue84/g1_analytic
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 0 --velocity-method rm_regularized_p1 --core-radius 0.01 --quadrature-order 12 --out runs/issue84/g1_rm_a001
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
