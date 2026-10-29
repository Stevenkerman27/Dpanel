# Issue #82 — Core-regularized filament induction pilot

Status: independent research module; not coupled to the P1 free-wake solver. `physical_accuracy_accepted=false`.

## Model

This phase uses the Rosenhead–Moore regularization of Biot–Savart for a constant-circulation finite segment:

\[
\mathbf u(\mathbf P)=\frac{\Gamma}{4\pi}
\int_A^B \frac{d\boldsymbol\ell\times(\mathbf P-\mathbf X)}
{(|\mathbf P-\mathbf X|^2+a^2)^{3/2}},\qquad a>0.
\]

The core radius (a) is an explicit length-scale parameter. It is not inferred from the mesh and has no hidden default. It regularizes the kernel, but does not by itself calibrate the model to a physical vortex core or viscous flow.

Let the oriented segment have length (L), unit tangent (mathbf t), and (mathbf P-mathbf A=q\mathbf t+\boldsymbol\rho), where (oldsymbol\rho\cdot\mathbf t=0). Then

\[
\mathbf u=\frac{\Gamma}{4\pi}(\mathbf t\times\boldsymbol\rho)
\frac{1}{\rho^2+a^2}
\left[
\frac{q}{\sqrt{\rho^2+a^2+q^2}}
-\frac{q-L}{\sqrt{\rho^2+a^2+(q-L)^2}}
\right].
\]

This is an analytic evaluation of the regularized line integral, not a numerical softening of the unregularized formula. For a field point on the segment axis, the cross product is zero and the result is finite; endpoints are finite for (a>0).

## Implementation

- `panel/regularized_vortex_filament.py`: analytic segment formula and oriented polyline summation.
- `test/test_regularized_vortex_filament.py`: independent quadrature oracle, sign/orientation, rigid transforms, scale covariance, core sensitivity, finite on-filament behavior, curved-loop refinement and invalid-input checks.

Required commands from the repository root:

```bash
python -m pytest test/test_regularized_vortex_filament.py -q -p dsh_tmp_plugin
python -m py_compile panel/regularized_vortex_filament.py test/test_regularized_vortex_filament.py
```

These commands have not yet been run in the full local checkout for this branch. Do not mark them as passed until the local agent returns the actual HEAD, environment, exit codes and output.

## Important interpretation boundary

At fixed positive core radius, the regularized polygonal-loop test should remain finite as the number of segments increases. Different core radii are expected to produce different velocities. This is a convergence/sensitivity property of the selected regularized model, not proof that any chosen core is physically correct.

A constant-strength filament sum is not generally equivalent to a continuous P1 doublet sheet. The P1 field has in-panel strength-gradient contributions; those must be retained or derived in any future mapping. This pilot deliberately does not couple to `solve_coupled_material_wake`, does not change solver defaults, and makes no claim of free-wake equilibrium.

## Next decision gate

After local test execution, inspect the core-radius sensitivity and decide how a physically defensible core length could be selected (or whether the P1 sheet field should remain the near-wake model). Only after that decision should a separate change couple this induction model into G1/G2 and compare the same frozen geometry and iteration-0 loads against the existing P1 sheet velocity.
