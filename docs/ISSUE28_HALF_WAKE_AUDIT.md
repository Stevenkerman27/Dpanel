# Issue #28: two-sided total-flow wake audit on the half model

This is the local deliverable for GitHub issue #28 (H2), revised after the PR #30 code review
(comment 6411659093). It answers one read-only question: on a frozen half-model native
solution, how much normal flux, how much material circulation transport and how much static
pressure jump does the *actual total flow* carry across the wake sheet, how does that split
over fixed streamwise bands behind the trailing edge, and how do the physical jump closures
behave with the two-sided distance?

- Delivered: `panel/audit_half_wake.py` (1012 lines) and `test/test_half_wake_audit.py`
  (21 cases, 21 passed / 5.49 s on this host).
- No production file is touched: pressure recovery, kernels, wake modes, default solver path,
  Gmsh geometry, CAD and Kutta conditions are unchanged. No wake relaxation, no merging, no
  training, no Cp clipping, no face deletion.
- `physical_accuracy_accepted=false` is written into every report.

## 1. What the review changed

| review item | change in this revision |
| --- | --- |
| 1. The old default solved the full mirrored model and built a dense expansion matrix E | The default now solves the half model exactly once (`panel.half_native.solve_half_fields`) and mirrors that single state by O(N) index maps: `mu` and `sigma` are copied, the wake doublet is copied, the wake circulation slope is negated (the same convention as `panel/half_native.py:342`, where the image term enters with the opposite spanwise derivative). The dense full solve survives only as an explicit `--full-oracle` cross-check with its own `--full-oracle-panels` cap (default 700 half panels). A test monkeypatches `panel.verify_half_mirror.make_full_mirror_topology` and `panel.differentiable.solve_fields` to raise and then runs the default audit, so any regression that reintroduces the full model fails the suite. `--frozen-state npz` replays a saved half state and verifies the surface SHA first, so a 396 or 1205 case can be re-read without another solve. |
| 2. `near` was the first half of the truncated wake extent, not the first row | The primary split is now six fixed bands of `x_gap / Cref` behind the trailing edge: [0, 0.05), [0.05, 0.1), [0.1, 0.25), [0.25, 0.5), [0.5, 1), [1, inf). A band with no samples is reported as `null` and is never redefined to make it non-empty. Samples are wake-triangle centroids plus three interior barycentric points, area weighted, and each sample carries its own distance to the trailing edge: the wake strips are long triangles, so banding by triangle centroid would file a sample right behind the trailing edge under a far band. The band rule is written into the report. The literal first wake layer is reported separately as `first_row`, selected by real trailing-edge adjacency (a wake triangle that shares a node with a declared TE edge), not by an x threshold. The old relative half-extent split is kept only as `historical_near` / `historical_far`. The absolute 0.5 Cref rule is reported as `near_absolute` and is allowed to be empty. |
| 3. The algebraic two-sided identity alone cannot validate the wake basis or the jump sign | Three physical closures are now reported next to the identity, per sample and area weighted: the potential jump against the wake doublet strength, the normal velocity jump, and the tangential velocity jump against +/- grad_s(Gamma). The sign is *measured*: both `tangential_plus_slope` and `tangential_minus_slope` residuals are computed and `smallest_residual` names whichever is smaller. Nothing is preset and the transport term is not dismissed because it is numerically small. |
| 4. The conclusions were stronger than the evidence | See section 8. Successive differences and their empirical orders are now reported instead of an O(eps) claim; the historical five-size series changes wall and wake discretisation together with Gamma fixed, so it can only say the defect did not disappear over that sequence; the Dirichlet form of the solve does not prove an O(h) wall residual. |
| 5. The PR was based on `master` | The PR base is now `research/near-vs-trefftz-drag-consistency`, the branch that already contains the frozen half solver and the audit A results, so the diff is this task only (8 files, +2757). |

## 2. Inputs and reproduction

Each case takes a saved half surface (`dpanel.half_planar_cut.v1`) plus exactly one half
solve. The auditor never writes into the surface directory, and `--out` must not exist
(`FileExistsError: Do not overwrite a previous wake audit`). The peak directory must differ
from the report path.

    $env:TMP=$env:TEMP=$env:TMPDIR='<repo>\runs\pipwork'
    $env:PYTHONPATH='<repo>\runs\pr28_git;<repo>\runs\torch_site'
    python -m peak_wrap panel.audit_half_wake <peak_dir>
        --surface <abs>\surface.json --out <abs new>\audit.json
        --max-half-panels 2000 --offsets 1e-2,1e-3,1e-4 --chunk 512

    # replay a frozen state without solving the flow again
    python -m peak_wrap panel.audit_half_wake <peak_dir>
        --surface <abs>\surface.json --out <abs new>\replay.json
        --frozen-state <abs>\frozen_state.npz

    $env:PYTHONPATH='<repo>\runs\pr28_git;<repo>\runs\torch_site'
    python -m pytest -q test/test_half_wake_audit.py -p no:cacheprovider

numba is required by `core/triangle_kernel.py`; on this host it exists only in
`runs/torch_site`, so the test module starts with `pytest.importorskip("numba", ...)` and
skips rather than pretending to pass. Reports record `surface_sha256`, `state_sha256`, the
frozen state path, `code_sha256`, Python/NumPy/JAX versions and the true residual.

## 3. Method

1. **One half solve, mirrored by index.** `build_mirror_state` solves the half model once and
   builds the mirrored full discretisation from index maps: full face n+i is the mirror
   partner of half face i, with the winding reversed so exterior normals stay consistent.
   `mu` and `sigma` are scalars and are duplicated, the wake doublet sheet value is duplicated,
   and the wake slope is negated. The wake basis and slope are assembled from geometry only
   (`_half_wake_te_geometry`), so a fresh solve and a frozen replay produce bit-identical wake
   operators. No dense expansion matrix exists anywhere on the default path.
2. **Field evaluation (total flow, not freestream)** uses the independent analytic kernel
   `core/triangle_kernel.influence` at orders 0/1 on the mirrored discretisation:

       V = U_inf - grad(sigma) + grad(mu) + grad(constant wake) + grad(linear wake moment)

   The linear wake moment is rewritten line by line from `panel/audit.py:54-72`
   (`relative_y*doublet + height*d_y(single)`, the three-edge Hessian loop,
   `moment_gradient[:,:,1] += doublet`, `[:,:,2] += grad_single[:,:,1]`). The normal used for
   the flux is the *discrete* wake normal; the freestream direction is never substituted for
   it.
3. **Two-sided sampling**: 4 points per wake triangle (centroid plus three interior
   barycentric points), offset along +/-eps with `eps = factor*sqrt(area)` and factor in
   {1e-2, 1e-3, 1e-4}; area weighted, and every sample is banded by its own x_gap / Cref.
   Edge and corner samples are included; the report states that the estimated orders are
   empirical for this mixed sampling and are not a universal order guarantee. The literal first
   wake layer stays a per-triangle property (wake triangles sharing a node with a declared
   trailing-edge edge), because that describes the mesh, not a sample.

The state hash is `sha256(mu, wake_mu, wake_slope, sigma)` over the mirrored arrays, so it
identifies the state actually evaluated and is independent of how the state was obtained.

## 4. Implementation checks

These are properties of the auditor itself, reported separately from the physical residuals so
a passing implementation check can never be read as physical accuracy.

<!-- CHECK_TABLE -->
| check | 396 walls | 1205 walls |
| --- | --- | --- |
| velocity = grad(potential), central diff at 1e-3 c | 1.895e-06 | 1.644e-06 |
| same at 1e-4 c | 1.895e-08 | 1.644e-08 |
| numba kernel vs JAX kernel (relative) | 9.460e-16 | 8.641e-16 |
| mirror map: panel normal partner max abs | 2.220e-16 | 3.331e-16 |
| mirror map: wake operator vs solver max abs | 1.804e-16 | 1.735e-16 |
| dense expansion matrix | not formed | not formed |
| image term: RMS over speed | 0.2945 | 0.3273 |
| image term: flux shift RMS | 0.2627 | 0.2944 |
| wall interior Dirichlet potential RMS | 5.034e-17 | 7.905e-17 |
| wall normal residual RMS / max | 0.0262 / 0.4910 | 0.0202 / 0.2535 |
| mirror equivariance: flux / dcp / transport max abs | 3.966e-15 / 9.437e-16 / 6.682e-17 | 6.883e-15 / 2.442e-15 / 1.527e-16 |
<!-- /CHECK_TABLE -->

The wall entry belongs to the *physics* side even though it is an identity of the discrete
system: this solver is Dirichlet-style (`matrix = [[doublet, wake], [-tested, mass]]`,
`rhs = single@sigma`), so it enforces zero interior potential, not zero discrete normal
velocity. The interior Dirichlet potential is therefore round-off, while the normal residual
is a discretisation quantity that the solve does not drive to zero. The production gate at
`panel/audit.py:112-117` reports the same quantity and its `strict_centroid_gate_passed` flag
is false throughout the existing docs.

## 5. Physical residuals

### 5.1 Flux, transport and pressure jump over fixed bands

Bands are per sample: the x distance from the sample to the nearest trailing-edge midpoint,
divided by the reference chord, selects the band, and the parent triangle's area is shared by
its four samples. The first band also absorbs any sample with a negative gap, and the count of
such samples is reported as `below_range`, so the band union provably covers every sample.

Per-sample definitions: `flux = Vmean . n / U` (total-flow tangency),
`transport = Vmean . grad(Gamma) / U^2` (material circulation transport),
`dcp = (|V-|^2 - |V+|^2)/U^2` (two-sided static pressure jump), with
`Vmean = (V+ + V-)/2` at +/- eps along the discrete wake normal.

<!-- BAND_TABLE -->
The smallest gap a sample can reach is set by the size of its own wake triangle, because a
sample is an interior point. At 396 half panels the reachable minimum is 0.0544 Cref (median
1.30, maximum 24.8), which is larger than the first band edge: the [0, 0.05) band is therefore
empty by resolution, and it is reported as `null`. At 1205 half panels the reachable minimum is
0.0117 Cref (median 0.58), so that band is resolved. The gap summary is in
`band_rule.gap_summary` and no band edge is ever moved to make a band non-empty.

**396 half panels, two-sided factor 1e-4, 3080 samples, 385 wake triangles per half**

| band (x_gap / Cref) | samples | area % | flux | dcp | transport | potential jump | normal jump |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [0,0.05) | 0 | null | null | null | null | null | null |
| [0.05,0.1) | 266 | 0.32 | 0.131277 | 0.038900 | 0.019456 | 0.276152 | 9.713e-06 |
| [0.1,0.25) | 282 | 0.55 | 0.097124 | 0.029715 | 0.014862 | 0.276670 | 5.692e-06 |
| [0.25,0.5) | 260 | 0.68 | 0.086215 | 0.017304 | 0.008655 | 0.278568 | 8.534e-06 |
| [0.5,1) | 506 | 1.65 | 0.087020 | 0.008071 | 0.004036 | 0.280056 | 3.424e-06 |
| [1,inf) | 1766 | 96.80 | 0.083931 | 7.538e-04 | 3.770e-04 | 0.279728 | 5.984e-08 |
| all samples | 3080 | 100.00 | 0.084271 | 0.003655 | 0.001828 | 0.279698 | 1.083e-06 |

Empty bands: [0,0.05). An empty band is reported as null and the band edges are never moved.

**1205 half panels, two-sided factor 1e-4, 5864 samples, 733 wake triangles per half**

| band (x_gap / Cref) | samples | area % | flux | dcp | transport | potential jump | normal jump |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [0,0.05) | 424 | 0.14 | 0.144224 | 0.047410 | 0.023729 | 0.299153 | 1.699e-05 |
| [0.05,0.1) | 328 | 0.17 | 0.120752 | 0.036600 | 0.018303 | 0.298246 | 7.298e-06 |
| [0.1,0.25) | 776 | 0.41 | 0.105851 | 0.026073 | 0.013038 | 0.300230 | 3.537e-06 |
| [0.25,0.5) | 1162 | 0.83 | 0.103873 | 0.017716 | 0.008860 | 0.300909 | 7.534e-06 |
| [0.5,1) | 972 | 1.64 | 0.084206 | 0.008573 | 0.004287 | 0.304124 | 2.538e-06 |
| [1,inf) | 2202 | 96.80 | 0.081104 | 8.033e-04 | 4.017e-04 | 0.303730 | 5.433e-08 |
| all samples | 5864 | 100.00 | 0.081694 | 0.003557 | 0.001779 | 0.303683 | 1.058e-06 |

Empty bands: none. An empty band is reported as null and the band edges are never moved.
<!-- /BAND_TABLE -->

### 5.2 First trailing-edge layer versus threshold and extent splits

<!-- ROW_TABLE -->
| split | 396 walls: samples / area pct / flux / dcp | 1205 walls: samples / area pct / flux / dcp |
| --- | --- | --- |
| first_row (real TE adjacency) | 536 / 0.86 / 0.111407 / 0.033569 | 688 / 0.27 / 0.137175 / 0.042699 |
| near_absolute (x_gap <= 0.5 Cref) | 808 / 1.55 / 0.100881 / 0.027533 | 2690 / 1.56 / 0.110458 / 0.026367 |
| historical_near (first half of extent) | 2680 / 13.22 / 0.086214 / 0.010053 | 5464 / 13.22 / 0.085407 / 0.009784 |
| historical_far (second half of extent) | 400 / 86.78 / 0.083971 / 1.881e-05 | 400 / 86.78 / 0.081113 / 1.977e-05 |
| all samples | 3080 / 100.00 / 0.084271 / 0.003655 | 5864 / 100.00 / 0.081694 / 0.003557 |
<!-- /ROW_TABLE -->

### 5.3 Physical jump closures

The algebraic identity `dcp = -2 Vmean . (V+ - V-) / U^2` is reported as `identity_abs_max`
and holds to round-off by construction. The closures below are the physical content: do the
two one-sided potentials differ by exactly the doublet strength, does the sheet carry normal
velocity, and does the tangential jump equal the spanwise circulation gradient?

<!-- JUMP_TABLE -->
| quantity | 396 walls | 1205 walls |
| --- | --- | --- |
| potential jump phi+ - phi- RMS | 0.279698 | 0.303683 |
| wake doublet strength RMS | 0.279647 | 0.303642 |
| |potential jump - doublet| RMS | 0.006134 | 0.005992 |
| |potential jump + doublet| RMS | 0.559311 | 0.607296 |
| normal jump (V+ - V-) . n RMS | 1.083e-06 | 1.058e-06 |
| tangential jump RMS | 0.385310 | 0.384932 |
| residual with -grad_s(Gamma) | 7.148e-04 | 7.009e-04 |
| residual with +grad_s(Gamma) | 0.771 | 0.770 |
| measured sign (smallest residual) | tangential_minus_slope | tangential_minus_slope |
| plus / minus residual ratio | 1078.8 | 1099.2 |
<!-- /JUMP_TABLE -->

### 5.4 Two-sided distance: successive differences and empirical order

<!-- SENS_TABLE -->
**396 half panels**

| offset factor | flux RMS | dcp RMS | transport RMS |
| --- | --- | --- | --- |
| 0.01 | 0.079123 | 0.003557 | 0.001828 |
| 0.001 | 0.084209 | 0.003646 | 0.001828 |
| 0.0001 | 0.084271 | 0.003655 | 0.001828 |

Successive differences (coarse minus fine): flux -0.005086, -6.204e-05; dcp -8.923e-05, -9.082e-06.

Empirical order, log(d_i/d_i+1)/log(ratio): flux 1.91; dcp 0.99. Reported, not assumed: these are empirical orders for this mixed edge and corner sampling, they are not an O(eps) theorem, and no order is preset anywhere in the code.

**1205 half panels**

| offset factor | flux RMS | dcp RMS | transport RMS |
| --- | --- | --- | --- |
| 0.01 | 0.076897 | 0.003435 | 0.001779 |
| 0.001 | 0.081636 | 0.003545 | 0.001779 |
| 0.0001 | 0.081694 | 0.003557 | 0.001779 |

Successive differences (coarse minus fine): flux -0.004738, -5.785e-05; dcp -1.101e-04, -1.183e-05.

Empirical order, log(d_i/d_i+1)/log(ratio): flux 1.91; dcp 0.97. Reported, not assumed: these are empirical orders for this mixed edge and corner sampling, they are not an O(eps) theorem, and no order is preset anywhere in the code.
<!-- /SENS_TABLE -->

## 6. Frozen-state replay

The 396 and 1205 reports are also reproduced from the `frozen_state.npz` files written by the
PR #29 acceptance runs, i.e. from a state saved by a different module. The surface SHA is
checked first and a mismatch is a hard error (`Frozen state surface SHA does not match
--surface`), never a silent warning.

<!-- REPLAY_TABLE -->
| item | 396 walls | 1205 walls |
| --- | --- | --- |
| fresh / replay state hash equal | yes | yes |
| largest relative difference, all numeric fields | bit-identical in every numeric field | bit-identical in every numeric field |
| fresh state_sha256 | 874d3c77f5da03d6 | 42772e182aeb35e1 |
| fresh solve seconds / peak RSS MB | 17.95 / 808.6 | 101.72 / 1016.6 |
| replay seconds / peak RSS MB | 23.25 / 738.2 | 68.59 / 933.3 |
| replay true residual | not computed (no solve) | not computed (no solve) |
| fresh true residual | 2.706e-16 | 4.025e-16 |
<!-- /REPLAY_TABLE -->

## 7. Historical control: the previous implementation

These five sizes were produced by the pre-revision module
(`panel/audit_half_wake.py` SHA-256 `DBD5146D...C705D44E`, 33685 B), which solved the full
mirrored model. They are kept verbatim as the control. Per the review they were not re-run;
the new default reproduces the 396 readings exactly (all-sample flux 0.084271 both before and
after, see 5.1), which is the expected result because both paths evaluate the same analytic
field on the same discretisation.

| wall | wake (half/full) | surface SHA | all flux | all dcp | all transport | near flux | near dcp | near area | far flux | far dcp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 396 | 385/770 | d04e439cddb2 | 0.084271 | 0.003655 | 0.001828 | 0.086214 | 0.010053 | 0.1322 | 0.083971 | 1.881e-05 |
| 1205 | 733/1466 | 034c216906e6 | 0.081694 | 0.003557 | 0.001779 | 0.085407 | 0.009784 | 0.1322 | 0.081113 | 1.977e-05 |
| 1966 | 838/1676 | 6b772c9a39d3 | 0.086655 | 0.003659 | 0.001830 | 0.091200 | 0.010063 | 0.1322 | 0.085941 | 2.166e-05 |
| 2921 | 999/1998 | 2ff3646240d7 | 0.085282 | 0.003986 | 0.001993 | 0.090140 | 0.010962 | 0.1322 | 0.084518 | 2.093e-05 |
| 4371 | 1190/2380 | 98b87c2fa85a | 0.084081 | 0.003440 | 0.001721 | 0.089181 | 0.009462 | 0.1322 | 0.083276 | 2.128e-05 |

## 8. What this establishes and what it does not

**Established by this revision.**

1. The default audit is now a genuine one-solve, O(N)-map evaluation of the same frozen half
   state; the full model is gone from the default path and its removal is enforced by a test,
   not by a comment.
2. The wake sheet in this discrete solution is kinematically a vortex sheet: the normal jump is
   at the level of the reported round-off of the sampling procedure (see 5.3), and the two-sided
   potential jump equals the wake doublet strength with the convention `phi+ - phi- = doublet`.
   The tangential jump matches `-grad_s(Gamma)` (the measured sign) three orders of magnitude
   better than `+grad_s(Gamma)`. So the residual normal flux reported in 5.1 is not a broken
   wake basis, it is a through-flow of the sheet.
3. The pressure jump concentrates just behind the trailing edge, and the flux does not.
   Across the fixed bands `dcp` falls monotonically by 51x at 396 (0.0389 at [0.05, 0.1) to
   7.54e-04 at [1, inf)) and by 59x at 1205 (0.0474 at [0, 0.05) to 8.03e-04), while `flux` spans
   only 0.084 to 0.144, a factor 1.7, and is essentially flat over 96.8 percent of the sampled
   sheet area. The literal `first_row` layer (real TE adjacency) shows the same thing:
   `dcp` 0.0336 at 396 and 0.0400 at 1205, against 7.54e-04 and 8.03e-04 in the far band. The one
   caveat is stated in 5.1: at 396 the [0, 0.05) band cannot be sampled at all, so its absence of
   data is a resolution statement, not a zero-flux statement.
4. The two-sided approach is empirically second order in the flux (see 5.4) and first order in
   `dcp` for this exactly planar sheet; both are reported with the numbers that produce them.
   `dcp` is *not* O(eps) and the earlier draft's claim of that was wrong.
5. A saved half state can be re-read (including a state written by another module) and gives
   the same report, which makes the 396 and 1205 rows independently reproducible without a
   solve.

**Not established.**

1. Not a physical accuracy statement. `physical_accuracy_accepted=false`; near-field CD/CM are
   not used for GRASS training and nothing here validates the drag of the configuration.
2. Not a demonstration that the residual flux is an artifact that refinement will remove. The
   historical series changed wall and wake discretisation together while Gamma stayed fixed, so
   the only defensible reading is that the defect did not disappear over 11x refinement. Nor is
   the opposite claim available: the sequence cannot attribute the residual to the wall
   discretisation, the wake model or the Gamma space separately.
3. Not a comparison against the reconstructed surface pressure. This audit evaluates the
   analytic field, so its `dcp` is not required to reproduce `Cp_raw` and is not a check of the
   pressure recovery operator (that is PR #29's subject).
4. The Dirichlet form does not prove an O(h) wall normal residual, only that the residual is
   not driven to zero by the solve.
5. `transport` being about 45x smaller than `flux` is a statement about this frozen geometry
   (the slope used is the one the frozen solution itself carries), not evidence that the
   circulation dynamics are unimportant.

## 9. Limits and next single-variable experiments

- The wall discretisation, the wake model and the Gamma space are still coupled in every row
  produced so far. The three candidate single-variable experiments remain: cut-aware pressure
  recovery (already run in PR #29 and it did not reduce the near-field cancellation), a wake
  kinematics plus dynamics joint residual with an explicit control surface, and an independent
  refinement of wall, gamma and wake.
- The index map removes an O(N^2) object, but that is *not* visible in the measured peak at
  these sizes: at 396 half panels the previous implementation peaked at 752.7 MB and the new
  default at 816.5 MB (1205: 1015.5 MB new, the previous full-solve path was not re-measured).
  At 396 the removed dense full operator is only about 5.6 MB (835 unknowns, 8 bytes each), so
  the peak is dominated by the O(N) numba field evaluation and by JAX compilation, and the
  difference between the two builds is within that noise. The argument for the change is
  structural (a dense full-model factorization is (2N+...) squared, about 0.6 GB of doubles plus
  a factorization copy at 4371 half panels), not a demonstrated peak reduction at 396 or 1205.
  The honest present summary: the default path no longer needs the full model, and its measured
  peak is the same order as before at the sizes re-run here.
- Nothing in this module writes a production artefact, and no default mode changes.
