# Issue #82 — Core-regularized filament induction pilot

Status: independent research module; not coupled to the P1 free-wake solver. `physical_accuracy_accepted=false`.

## Model

This phase uses the Rosenhead–Moore (RM) regularization of Biot–Savart for a constant-circulation finite segment:

\[
\mathbf u(\mathbf P)=\frac{\Gamma}{4\pi}
\int_A^B \frac{d\boldsymbol\ell\times(\mathbf P-\mathbf X)}
{(|\mathbf P-\mathbf X|^2+a^2)^{3/2}},\qquad a>0.
\]

The core radius \(a\) is an explicit length-scale parameter. It is not inferred from the mesh and has no hidden default. It regularizes the kernel, but does not by itself calibrate the model to a physical vortex core or viscous flow.

Let the oriented segment have length \(L\), unit tangent \(\mathbf t\), and \(\mathbf P-\mathbf A=q\mathbf t+\boldsymbol\rho\), where \(\boldsymbol\rho\cdot\mathbf t=0\). Then

\[
\mathbf u=\frac{\Gamma}{4\pi}(\mathbf t\times\boldsymbol\rho)
\frac{1}{\rho^2+a^2}
\left[
\frac{q}{\sqrt{\rho^2+a^2+q^2}}
-\frac{q-L}{\sqrt{\rho^2+a^2+(q-L)^2}}
\right].
\]

This is an analytic evaluation of the regularized line integral, not a numerical softening of the unregularized formula. For a field point on the segment axis, the cross product is zero and the result is finite; endpoints are finite for \(a>0\).

## What analytic self-induction means here

Self-induction is the velocity a curved vortex filament induces at its own location through the Biot–Savart field of the entire filament. It is not just the contribution of one straight segment to a point on that segment. For an ideal zero-core filament, the near-filament contributions of neighboring segments accumulate logarithmically as the polygonal representation is refined. Declaring the symmetric principal value of one straight segment to be zero does not remove that curvature-related accumulation.

For a circular vortex ring of radius \(R\), the continuous RM kernel gives the on-filament binormal (ring-axis) speed

\[
U_{\mathrm{RM}}(R,a)=
\frac{\Gamma}{4\pi}\int_0^{2\pi}
\frac{R^2(1-\cos\theta)}
{\left[a^2+2R^2(1-\cos\theta)\right]^{3/2}}\,d\theta.
\]

For a thin core, \(a/R\ll1\), its asymptotic self-induction law is

\[
U_{\mathrm{RM}}\sim
\frac{\Gamma}{4\pi R}
\left[\ln\left(\frac{8R}{a}\right)-1\right].
\]

The additive constant is core-model dependent. The \(-1\) constant belongs to this Rosenhead–Moore parameterization. The commonly quoted \(-1/4\) expression in a Gaussian/Lamb–Oseen or other core convention uses a different core definition; its numerical radius must not be copied directly into the RM formula. For \(R=1\), \(\Gamma=1\), \(a=0.05\), the two constants alone change the predicted speed by \(0.75\Gamma/(4\pi R)=0.05968\), about 18.4% of the RM speed. This is large enough to invalidate a core radius calibrated under the wrong convention.

The thin-core law is asymptotic, not an exact finite-\(a\) identity. Local independent integration of the continuous RM ring kernel and the polygonal segment sum produced:

| \(a/R\) | Continuous RM integral | Polygon, \(N=4096\) | Thin-core law | Relative polygon-to-law residual |
|---:|---:|---:|---:|---:|
| 0.20 | 0.21258123 | 0.21258 | 0.21397423 | -6.51e-3 |
| 0.10 | 0.26867929 | 0.26868 | 0.26913313 | -1.69e-3 |
| 0.05 | 0.32415253 | 0.32416 | 0.32429203 | -4.30e-4 |
| 0.025 | 0.37940958 | 0.37943 | 0.37945093 | -1.09e-4 |
| 0.0125 | 0.43459787 | — | 0.43460983 | -2.75e-5 |
| 0.00625 | 0.48976534 | — | 0.48976873 | -6.93e-6 |

With \(R=\Gamma=1\), the asymptotic-law residual decreases approximately as \(O((a/R)^2)\). At \(a/R=0.05\), the 4096-segment polygon is within about \(4.1\times10^{-4}\) relative error of the thin-core law. The exact continuous integral is the stronger finite-core oracle; the asymptotic law is a separate physics-level check.

## Implementation

- `panel/regularized_vortex_filament.py`: analytic segment formula and oriented polyline summation.
- `test/test_regularized_vortex_filament.py`: independent composite-quadrature oracle, near-core checks, sign/orientation, rigid transforms, scale covariance, core sensitivity, continuous-ring oracle, thin-core self-induction law and invalid-input checks.

Required commands from the repository root:

~~~bash
python -m pytest test/test_regularized_vortex_filament.py -q
python -m py_compile panel/regularized_vortex_filament.py test/test_regularized_vortex_filament.py
~~~

## Local execution receipt (pre-revision commit)

The local agent executed the original test set at commit `e425236f209a8ed816a1166864f0a108933977dc` on Python 3.11.17 / NumPy 2.4.6:

~~~text
python -m py_compile panel/regularized_vortex_filament.py test/test_regularized_vortex_filament.py
exit 0
python -m pytest test/test_regularized_vortex_filament.py -q
12 passed, 1 warning in 0.18s
exit 0
~~~

The same local run reported `python -m pytest test/ -q` as 603 passed, 11 failed, 8 skipped in 141.16s. The 11 failures were reproduced on a clean checkout of master `dfb723c` with the same test set: 2 `test_matrix_free_symmetry_gate`, 6 `test_p1_coupled_material_wake`, and 3 `test_vortexad_*`. The `test/test_p1_*` subset passed 47 tests. The local environment also could not import the legacy `test_torch_*` dependencies from its `.conda_env` because Torch resides under `runs/torch_site`; this is an environment/path limitation, not a result for the new regularized kernel.

### Current-head revalidation

At current PR head `186ee66979876e3f3d787f41f95ef6fb1f0775b4`, the local agent reran the revised test set on Python 3.11.17 / NumPy 2.4.6:

```text
python -m py_compile panel/regularized_vortex_filament.py test/test_regularized_vortex_filament.py
exit 0
python -m pytest test/test_regularized_vortex_filament.py -q
16 passed, 1 warning in 0.53s
exit 0
```

The warning is the previously observed Windows pytest-cache permission warning. The local agent independently compared the continuous circular-ring quadrature against SciPy `quad` (relative differences from 0 to 1.07e-14 for the reported core radii), and checked the near-core composite quadrature at offsets 0.1a, 0.05a and 0.01a with a separate 24-point × 800-subinterval oracle (relative differences 0 to 6.55e-13, including offsets deliberately placed away from quadrature subinterval boundaries).

The full-repository baseline remains 603 passed, 11 failed, 8 skipped on both the PR source revision and clean master `dfb723c`; the same 11 failures were identified in the matrix-free symmetry, coupled P1 wake and VortexAD groups. The P1 subset passed 47 tests. These baseline failures are not claimed as passing and were not changed by this research PR.

One non-blocking test-margin note: the thin-core asymptotic-law assertion at a/R=0.05 measured a relative residual of 4.11e-4 against a 5e-4 tolerance. This is dominated by the finite-core O((a/R)^2) correction, not polygon discretization. Do not tighten this tolerance without either moving to a smaller a/R or including a separately justified finite-core correction.

## Important interpretation boundary

At fixed positive core radius, the regularized polygonal-loop sum should converge to the continuous RM ring integral as the segment count increases. Different core radii produce different velocities. This is a convergence and sensitivity property of the selected regularized model, not proof that any chosen core is physically calibrated.

A constant-strength filament sum is not generally equivalent to a continuous P1 doublet sheet. The P1 field has in-panel strength-gradient contributions; those must be retained or derived in any future mapping. This pilot deliberately does not couple to `solve_coupled_material_wake`, does not change solver defaults, and makes no claim of free-wake equilibrium.

## Next decision gate

Before coupling this induction model into G1/G2, choose and justify the core convention and core length using the same convention as the kernel. Do not transfer a Gaussian/Lamb–Oseen radius numerically into the RM model without a derived conversion. Then compare the regularized filament field against the existing P1 sheet field on the same frozen geometry, preserving the P1 in-panel gradient contribution. Only after iteration-0 velocity and load differences are understood should a separate change couple the model into G1/G2.
