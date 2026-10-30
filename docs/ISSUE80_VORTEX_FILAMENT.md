# Issue #80 — Lagrangian vortex-filament kernel pilot

Status: research prototype only. No solver coupling; `physical_accuracy_accepted=false`.

## Why start with a standalone segment oracle?

The free-wake report recommends a Lagrangian wake with x-deformable nodes and a Biot–Savart principal-value (PV) kernel with an analytic self term. Current master already has an analytic straight-edge evaluator for a continuous P1 doublet sheet and a 3-D material-coordinate wake iteration. The next defensible step is to isolate the filament kernel and its singularity contract before changing that coupled solver.

A constant-strength vortex filament and a continuous P1 doublet sheet are not interchangeable in general. The P1 sheet has an in-panel strength gradient; its induced field includes a surface-gradient contribution as well as edge terms. This pilot therefore does not replace `AnalyticSheetField` and does not yet claim a P1-consistent wake field.

## Segment formula and conventions

For a straight segment oriented from A to B, signed circulation Gamma, and off-filament point P, define r1=P-A and r2=P-B. The induced velocity is

\[
\mathbf{u}(P)=\frac{\Gamma}{4\pi}
\frac{\mathbf{r}_1\times\mathbf{r}_2}
{|\mathbf{r}_1\times\mathbf{r}_2|^2}
\left[(\mathbf{B}-\mathbf{A})\cdot
\left(\frac{\mathbf{r}_1}{|\mathbf{r}_1|}
-\frac{\mathbf{r}_2}{|\mathbf{r}_2|}\right)\right].
\]

Positive Gamma follows the right-hand rule for A → B. With length in metres and circulation in m²/s, velocity is m/s. Reversing endpoints or circulation reverses velocity. Uniform geometric scaling by s with fixed Gamma scales velocity by 1/s.

The implementation in `panel/vortex_filament.py` has no hidden core radius, clipping, or epsilon softening. A point exactly on the segment interior raises by default. The opt-in `on_segment="straight_pv_zero"` returns zero for the symmetric PV contribution of that *straight segment alone*. A point at an endpoint still raises. This is not an analytic curvature/self-induction correction for a curved polygonal filament, and excluding one segment does not regularize adjacent segments at a shared endpoint.

## What is tested

`test/test_vortex_filament.py` checks:
- agreement with independent Gauss–Legendre integration of the Biot–Savart line integral at an off-axis point;
- orientation and circulation sign;
- translation and proper-rotation covariance;
- inverse-length geometric scaling and finite-segment far-field decay;
- near-axis inverse-distance signature;
- explicit on-segment and endpoint singularity policy;
- polyline summation and explicit segment exclusion;
- a closed square vortex loop against independent segment line integrals;
- the logarithmic growth of the remaining polygonal-loop velocity when one straight self-segment is excluded;
- invalid-input rejection.

Run from the repository root:

```bash
python -m pytest test/test_vortex_filament.py -q
```

## Local execution receipt

The local agent executed the original 9-test version at HEAD `b6aa91cb202d05c2072d6c46c51863642fbcac85` on Windows, Python 3.11.17 and NumPy 2.4.6:

```
python -m pytest test/test_vortex_filament.py -q
9 passed, 1 warning in 0.11s
exit code: 0
```

The warning reported a Windows `.pytest_cache` access-denied condition; the agent notes that its normal sandbox workflow uses `-p dsh_tmp_plugin` and `DSH_TMP_ROOT` to avoid cache writes. The agent also ran independent numerical checks: 200 off-filament points versus a separate composite Gauss–Legendre oracle had maximum relative error `1.18e-14`; a closed square loop differed from independently integrated segments by `9.999e-16`; a triangular ring matched the existing `wake_velocity.triangle_ring_velocity` at `1.245e-15`; and near-axis evaluation down to `h=1e-9` matched the finite-segment analytic limit to `3.6e-16`.

The agent found an important limitation: excluding the straight segment's own PV contribution does not regularize a curved polygonal filament. At the midpoint of one segment of a unit-radius polygonal loop, the remaining induced speed grew with refinement; the increment under each doubling approached `(Gamma/(4*pi*R))*ln(2)`, the expected logarithmic self-induction divergence. This confirms that the zero straight-segment PV convention is **not** a complete curved-filament self-induction model.

Two repository regressions were added after the above 9-test run: a closed-loop oracle and the logarithmic-divergence test. Therefore the current PR head still needs a fresh full test run; the 9-pass receipt does not cover the two new tests. Do not mark the current head as passing until the local agent reports the new HEAD, exact command, exit code and output.

## Interpretation and next gate

This is a verified-formula candidate, not a completed free-wake method. A curved filament's velocity depends on the discretization and self/near-self treatment; the straight-segment PV convention does not supply a curvature term. Before coupling it into the wake iteration, the next experiment must compare:
1. refine the committed closed-loop and logarithmic-divergence regression checks in the full repository;
2. decide whether to add a physically justified finite-core or curvature-aware self-induction model (with parameter sensitivity), or retain the P1 sheet field for near-wake induction;
3. compare the resulting representation against the existing P1 analytic sheet field only where their mathematical assumptions make equivalence valid.

If the intended free wake remains a continuously varying P1 sheet, the filament method must retain or derive the contribution associated with the in-panel strength gradient. A constant-strength filament-only sum is not an acceptable drop-in replacement.

No generated G1/G2/CAD iteration was run in this phase. Fixed-wake defaults and all existing solver paths remain unchanged.
