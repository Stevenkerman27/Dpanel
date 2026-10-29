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
- invalid-input rejection.

Run from the repository root:

```bash
python -m pytest test/test_vortex_filament.py -q
```

The repository test command has **not yet been executed in the full DPanel checkout by this authoring environment**. Do not mark these tests as passing until the local agent reports the exact HEAD, Python/NumPy versions, command, exit code and output.

## Interpretation and next gate

This is a verified-formula candidate, not a completed free-wake method. A curved filament's velocity depends on the discretization and self/near-self treatment; the straight-segment PV convention does not supply a curvature term. Before coupling it into the wake iteration, the next experiment must compare:
1. a closed polygonal vortex loop against an independent direct line-integral oracle at off-filament targets;
2. mesh refinement and near-filament offset sensitivity;
3. at least one analytically controlled curved-filament case, with the chosen self-induction convention stated explicitly;
4. the resulting representation against the existing P1 analytic sheet field only where their mathematical assumptions make equivalence valid.

If the intended free wake remains a continuously varying P1 sheet, the filament method must retain or derive the contribution associated with the in-panel strength gradient. A constant-strength filament-only sum is not an acceptable drop-in replacement.

No generated G1/G2/CAD iteration was run in this phase. Fixed-wake defaults and all existing solver paths remain unchanged.
