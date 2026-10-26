# Issue #28: half-model wake audit with the two-sided total-flow residual

This is the local deliverable for GitHub issue #28 (H2). It answers one read-only question:
on a frozen half-model native solution, how much normal flux, how much material circulation
transport and how much pressure jump does the *actual total flow* carry across the wake
sheet, and how do those numbers behave with mesh refinement and with the two-sided
distance?

- Delivered: `panel/audit_half_wake.py` and `test/test_half_wake_audit.py` (13 cases).
- No production file is touched: pressure recovery, kernels, wake modes, default solver path,
  Gmsh geometry, CAD, Kutta conditions are all unchanged. No wake relaxation, no merging,
  no training.
- `physical_accuracy_accepted=false` is written into every report.

## 1. Inputs and reproduction

Each case takes a *saved* half surface (`dpanel.half_planar_cut.v1`) plus one existing
half-native LU solve. The auditor solves once itself (same production path), expands the
positive half by the mirror involution into a complete discrete model, and evaluates the
potential field on that *complete* discretisation.

    $env:TMP=$env:TEMP=$env:TMPDIR='<repo>\runs\pipwork'
    $env:PYTHONPATH='<repo>\runs\pr28_git;<repo>\runs\torch_site'
    python -m peak_wrap panel.audit_half_wake <peak_dir>
        --surface <abs>\surface.json --out <abs new>\audit.json
        --max-half-panels <N> --offsets 1e-2,1e-3,1e-4 --chunk 512

`--out` must not exist (`FileExistsError: Do not overwrite a previous wake audit`), and the
peak directory must differ from `--out`. Running needs numba
(`core/triangle_kernel.py`); on this host numba only exists in `runs/torch_site`, not in
`.conda_env`, so the test module starts with `pytest.importorskip("numba", ...)` and
skips instead of pretending to pass. Reports record `surface_sha256`, `state_sha256`
(mu/gamma/sigma), `code_sha256`, Python/NumPy/JAX versions and the true residual.

## 2. Method

1. **Mirror expansion** reuses the already-accepted
   `panel.verify_half_mirror.make_full_mirror_topology` / `mirrored_full_input`; wall,
   wake and gamma degrees of freedom are duplicated by the involution and the winding is
   flipped so exterior normals stay consistent.
2. **Field evaluation (total flow, not freestream)** uses the independent analytic kernel
   `core/triangle_kernel.influence` at orders 0/1 on the complete discretisation:

       V = U_inf - grad(sigma) + grad(mu) + grad(constant wake) + grad(linear wake moment)

   The linear wake moment is rewritten line by line from `panel/audit.py:54-72`
   (`relative_y*doublet + height*d_y(single)`, the three-edge Hessian loop,
   `moment_gradient[:,:,1] += doublet`, `[:,:,2] += grad_single[:,:,1]`).
   The wake normal is the *discrete* wake normal (+z); `U_inf . n` is never substituted.
3. **Two-sided sampling**: 4 points per wake triangle (centroid plus three interior
   barycentric points) offset along +-eps, with `eps = factor*sqrt(area)` and factor in
   {1e-2, 1e-3, 1e-4}; area weighted.
4. **near / far rule**: `x_gap` is the streamwise distance from a wake sample to the nearest
   trailing-edge midpoint, `fraction = x_gap / max|x_gap|`, `near = fraction <= 0.5` (first
   half of the truncated wake streamwise extent). The absolute `0.5*c` rule is kept as
   `near_absolute` but is not the primary rule: on the small fixture it would be empty
   because the wake is 3c long. The rule text is written verbatim into `near_far_rule`.

## 3. The three quantities and their units

- `flux = Vmean . n / U` (dimensionless), where `Vmean = (V+ + V-)/2`.
- `transport = Vmean . grad(Gamma) / U^2` (dimensionless; Gamma is m^2/s).
  The wake slope is mapped to samples with `np.repeat(wake_slope, per_panel)`.
- `dcp = (|V-|^2 - |V+|^2)/U^2`, with the algebraic identity
  `identity = -2 Vmean . (V+ - V-) / U^2` reported as a self-check.

## 4. Self-checks (all reported)

| check | meaning | result |
| --- | --- | --- |
| `velocity_equals_potential_gradient` | central difference of the total potential vs the analytic velocity | step 1e-3*c -> 1.9e-06 relative; 1e-4*c -> 1.9e-08 (second order) |
| `potential_numba_vs_jax` | numba kernel vs JAX `triangle_potential` | 9.8e-16 / 8.0e-16 / 1.2e-15 relative |
| `wall_tangency` | wall exterior-trace normal residual and interior Dirichlet potential | see below |
| `mirror_equivariance` | flux / dcp / transport symmetric, Vy antisymmetric, slope sign flips, mu symmetric | 1e-14 to 1e-13 |
| `image_contribution` | complete mirrored field minus positive-half-only field | 0.26 to 0.30 U |
| `units` | dimension strings and the "no U_inf . n" constraint | written as text |

On the wall normal residual: the production quantity at `panel/audit.py:112-117` is
0.0147 to 0.0297 in the existing docs and `strict_centroid_gate_passed` is **false
everywhere**. This solver is Dirichlet-style (`matrix = [[doublet, wake], [-tested, mass]]`,
`rhs = single@sigma`): it enforces interior potential zero, not the discrete normal
velocity, so the normal residual is O(h) rather than round-off. This audit reproduces the
same quantity and additionally reports `interior_potential_rms ~ 5e-17`, i.e. the interior
Dirichlet condition is exact.

## 5. Results

Each row is one saved surface and one frozen solve. The module hash is the same for every
row: `panel/audit_half_wake.py` SHA-256
`DBD5146D763D4AABE42CB06B0CCC987DB6C4A363F0E813BB7D8A52B9C705D44E` (33685 B).

<!-- WAKE_TABLE -->
| wall | wake (half/full) | surface SHA | all flux | all dcp | all transport | near flux | near dcp | near area | far flux | far dcp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 396 | 385/770 | d04e439cddb2 | 0.084271 | 0.003655 | 0.001828 | 0.086214 | 0.010053 | 0.1322 | 0.083971 | 1.881e-05 |
| 1205 | 733/1466 | 034c216906e6 | 0.081694 | 0.003557 | 0.001779 | 0.085407 | 0.009784 | 0.1322 | 0.081113 | 1.977e-05 |
| 1966 | 838/1676 | 6b772c9a39d3 | 0.086655 | 0.003659 | 0.001830 | 0.091200 | 0.010063 | 0.1322 | 0.085941 | 2.166e-05 |
| 2921 | 999/1998 | 2ff3646240d7 | 0.085282 | 0.003986 | 0.001993 | 0.090140 | 0.010962 | 0.1322 | 0.084518 | 2.093e-05 |
| 4371 | 1190/2380 | 98b87c2fa85a | 0.084081 | 0.003440 | 0.001721 | 0.089181 | 0.009462 | 0.1322 | 0.083276 | 2.128e-05 |
<!-- /WAKE_TABLE -->

The five rows are five *different* Gmsh meshes of the same wing-body (4371 has 11x the
surface of 396), all with the same frozen-wake model and the same audit module
(`dbd5146d763d`). The readings neither grow nor shrink monotonically: all-flux spans
0.0817 to 0.0867 U, near dcp 0.0095 to 0.0110, far dcp 1.88e-05 to 2.17e-05, transport
0.00172 to 0.00199. **The wake defect is therefore not a discretisation artifact that
refines away: it is a fixed property of the frozen-wake model at this angle of attack.**

Distance sensitivity (offset 1e-2 / 1e-3 / 1e-4), all-flux RMS:

<!-- SENS_TABLE -->
| wall | 1e-2 | 1e-3 | 1e-4 | successive differences |
| --- | --- | --- | --- | --- |
| 396 | 0.079123 | 0.084209 | 0.084271 | -5.086e-03 then -6.204e-05 |
| 1205 | 0.076897 | 0.081636 | 0.081694 | -4.738e-03 then -5.785e-05 |
| 1966 | 0.081136 | 0.086589 | 0.086655 | -5.453e-03 then -6.522e-05 |
| 2921 | 0.080190 | 0.085220 | 0.085282 | -5.030e-03 then -6.174e-05 |
| 4371 | 0.079235 | 0.084024 | 0.084081 | -4.790e-03 then -5.661e-05 |
<!-- /SENS_TABLE -->

Shrinking the offset by one decade shrinks the successive difference by about 1.4
decades, so the two-sided average converges as O(eps) and the 1e-4 column is the
trustworthy reading. `transport` changes by less than 1e-6 relative across the three
offsets, i.e. it does not depend on the probe distance at all.

## 6. Reading

1. **The total-flow normal flux is 0.082 to 0.087 U over an 11x mesh range, i.e. essentially
   sin(5 deg) = 0.087156** (ratio 0.937 to 0.994) and it does not trend with refinement. The
   discrete wake is not a streamwise-aligned surface: with the wake frozen as an x-aligned
   strip, the freestream itself crosses it at 5 deg, and the induced field almost never
   bends it back to tangency.
2. **The pressure jump is "sick near the trailing edge, clean far downstream"**: near-segment
   RMS is 0.0095 to 0.0110, far-segment RMS is only 1.9e-05 to 2.2e-05, a factor of about
   500. The violated Kutta / pressure-continuity condition is concentrated in the first wake
   row; far downstream the jump is numerically almost zero.
3. **Material circulation transport is about 45x smaller than the normal flux** (0.00172 to
   0.00199 versus 0.084). On the frozen steady state `Vmean . grad(Gamma)` is small, so the
   streamwise change of circulation is not the main issue; the normal flux is.
4. `identity_abs_max ~ 5e-16`: the algebraic identity
   `dcp = -2 Vmean . (V+ - V-)/U^2` holds to round-off, so the reported dcp is not
   evaluation noise.
5. **The image matters**: evaluating only the positive half loses 0.26 to 0.30 U of normal
   flux (`flux_shift_rms` 0.2627 / 0.2944 / 0.2970). Any wake audit done on a
   positive-half-only field is systematically wrong.

## 7. What this does *not* claim

- It does not claim the wake is fixed, nor that the wake mode should be changed on this basis.
- A smaller flux is **not** treated as a physical improvement. This module only supplies
  readings and an explicit sampling rule; model selection stays with the #26/#27 audit line.
- It does not require the direct analytic field to match a surface-fitted Cp face by face
  (that is the #27 / PR #29 scope).
- No objective-function or training-side inference: near-field CD/CM still stay out of GRASS
  training and `physical_accuracy_accepted=false` is unchanged.

## 8. Relation to PR #26 audit A

Audit A reported `rms|U_inf . n|/U` identically equal to sin(5 deg) across five meshes and
self-labelled it `freestream_flux_only_not_full_local_wake_tangency`. Using the same
surfaces, this audit replaces that quantity with the true total-flow `Vmean . n / U` on the
complete mirrored discretisation and gets 0.082 to 0.087. The difference from the
freestream-only value is 3 to 6 percent, which confirms audit A's qualitative reading, but
the two are not the same physical quantity and must not be cross-calibrated.

## 9. Known limits

- Steady state with frozen wake geometry only; the wake shape is not iterated.
- near/far uses a relative streamwise split; meshes with very short or very long wakes
  would get a different near area fraction (it is 0.1322 for every case here because the
  wake topology is identical).
- Only 1e-2/1e-3/1e-4 offsets are verified; larger offsets enter the near-field singularity
  of neighbouring panels.
- Resources (one process per case, `runs/peak_wrap.py` sampling every 20 ms):

  | wall | wall seconds | peak RSS |
  | --- | --- | --- |
  | 396 | 31.8 | 755.2 MB |
  | 1205 | 86.9 | 975.3 MB |
  | 1966 | 135.2 | 1115.6 MB |
  | 2921 | 243.4 | 1624.6 MB |
  | 4371 | 275.9 | 3000.0 MB |

  The 4371 case needs `--max-half-panels 4600`; it already uses 3.0 GB, so it is
  the practical upper bound on this 8 GB machine and is not the default.
