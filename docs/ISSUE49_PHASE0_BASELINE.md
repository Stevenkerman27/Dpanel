# Issue #49 路 Phase 0 鈥?frozen reference cases and error budget baseline

Status: research baseline. `physical_accuracy_accepted = false`. No production
default changed, nothing merged into `master`, no GRASS training touched.

Phase 0 of issue #49 is a **read-only measurement phase**: it freezes what a real
half-model solution actually is (digests of the case, the mesh and the solution,
the linear residual, and the traces that any later change must beat). Phase 1
(geometry-consistent gradient recovery) and Phase 2 (continuous P1 doublets) are
separate patches; nothing here re-assembles the physics.

## 1. What was added

| module | role |
| --- | --- |
| `panel/baseline_manifest.py` | canonical SHA-256 receipts of the case, the surface fields, the solution state, the loads and the `panel/` sources; `verify_manifest()` re-checks a stored manifest against a re-solve |
| `panel/audit_reference.py` | total-field traces on a frozen solve: `V路n/U` off every wall panel, two-sided `Cp` limits at the true TE, wake normal flow and `螖Cp`, spanwise `d螕/dy`, circulation lift, near-field drag |
| `panel/analytic_sphere.py` | closed-form anchor: an icosphere carrying the exact incompressible sphere solution, used to measure the reconstruction error of the same operators with no wing, no wake and no CAD |

Two design rules were fixed before measuring:

1. **No finite differences.** The velocity is the analytic derivative of the same
   layer operators the assembly used (`jax.jacfwd` through `triangle_potential`),
   so a trace is the discrete field itself, not a difference of two solves.
2. **No free-stream projection.** Every reported normal flow is the total field
   (`V = U鈭?+ 鈭囅哷) dotted with the geometric normal. `U鈭灺穘` is never substituted
   for it, which is what makes the 7 % wall residual visible at all.

## 2. Frozen cases

Three real half cases, all dense LU, all `dpanel.half_planar_cut.v1`:

| tag | case dir | wall | wake | 纬 dofs | unknowns | hot solve |
| --- | --- | --- | --- | --- | --- | --- |
| `wing_only_276` | `runs/half_cad_wing_only_v2` | 276 | 360 | 24 | 300 | 0.070 s |
| `elliptic_1028` | `runs/geom_robust/mesh/elliptic_s100` | 1028 | 577 | 17 | 1045 | 0.428 s |
| `elliptic_1610` | `runs/geom_robust/mesh/elliptic_s075` | 1610 | 764 | 17 | 1627 | 0.885 s |

The elliptic pair is the point of the selection: 1028 鈫?1610 wall panels
(+57 %) is the smallest refinement step in the existing cases, so it shows
whether a metric is converging or merely moving.

Evidence: `runs/ref_audit/<tag>/{manifest.json,metrics.json,report.json}`
(copied to `docs/issue49_phase0/`; the solution state itself is regenerable from the case and is kept out of git, its digest is in the manifest).

## 3. Linear algebra is not the error source

| tag | `||Ax-b||/||b||` | interior `|蠁|/(U c)` |
| --- | --- | --- |
| `wing_only_276` | 1.80e-15 | 7.2e-05 |
| `elliptic_1028` | 1.11e-15 | 1.3e-05 |
| `elliptic_1610` | 1.28e-15 | 1.1e-05 |

The solve is exact to round-off and the assembled interior Dirichlet condition is
satisfied to 1e-5 of the potential scale, so every number below is a
**discretization** property. Any Phase 1/2 patch must keep both columns at the
same order.

## 4. The wall boundary condition on the true total field

`V路n/U` rms at offsets `1e-4, 1e-3, 1e-2` chords off every wall centroid:

| tag | 1e-4 c | 1e-3 c | 1e-2 c |
| --- | --- | --- | --- |
| `wing_only_276` | 0.0701 | 0.0679 | 0.0564 |
| `elliptic_1028` | 0.0141 | 0.0149 | 0.0316 |
| `elliptic_1610` | 0.0122 | 0.0133 | 0.0340 |

Two facts worth keeping:

- The slender NACA wing is **5脳 worse** than the elliptic planform at the same
  offset (0.070 vs 0.014). A single `wall_flux` tolerance would either pass the
  elliptic case by luck or fail the wing for a reason that is not the assembly.
- On the elliptic cases the residual **grows** with offset (0.0122 鈫?0.0340),
  while on the wing it shrinks. The near-wall value is therefore not a monotone
  鈥渉ow well is the BC satisfied鈥?measure, and must always be reported as a
  sweep, never as one number.

## 5. Two-sided Cp at the true trailing edge

| tag | TE edges | `Cp` jump rms | worst edge `|Cp|` | linear extrapolation |
| --- | --- | --- | --- | --- |
| `wing_only_276` | 48 | 28.81 | 192.7 | invalid |
| `elliptic_1028` | 35 | 3.35 | 11.78 | invalid |
| `elliptic_1610` | 42 | 2.22 | 9.12 | invalid |

The jump shrinks with refinement (28.8 鈫?3.35 鈫?2.22), but the sampled values are
still singular at the closest offsets, so the linear intercept is flagged
`linear_extrapolation_valid = false` in all three cases and **no extrapolated
Kutta value is quoted**. The TE limit is the one quantity here that is not yet
measurable at the current mesh scales; Phase 2 must not claim it.

## 6. The local wake-transport test does not apply to this representation

A first version of the wake diagnostic compared the **mean** tangential velocity
across the sheet with `d螕/dy`. It gives a residual of ~1.0 U, i.e. it never
passes. Measuring the **jump** instead explains why:

| tag | mean `Vt/U` | jump `Vt/U` | `(d螕/dy)/U` | jump / gradient |
| --- | --- | --- | --- | --- |
| `wing_only_276` | 0.9821 | 0.0133 | 0.3269 | 0.041 |
| `elliptic_1028` | 0.9625 | 0.0222 | 0.3633 | 0.061 |
| `elliptic_1610` | 0.9564 | 0.0190 | 0.3116 | 0.061 |

The doublet density is piecewise constant along the span, so the tangential
velocity jump **vanishes inside a strip by construction** (4鈥? % of `d螕/dy`). The
local identity is therefore not a property this representation can be graded on,
and `wake_transport` now reports the three measured quantities plus that
interpretation instead of pretending to be a residual. What *is* meaningful is the
normal flow through the sheet:

| tag | `V路nw/U` at 1e-3 c | `螖Cp` rms at 1e-3 c |
| --- | --- | --- |
| `wing_only_276` | 0.0654 | 0.0072 |
| `elliptic_1028` | 0.0704 | 0.0178 |
| `elliptic_1610` | 0.0648 | 0.0159 |

## 7. The pressure channel, and why it is the target of Phase 1/2

| tag | `CL` from `Cp` | `CL` from `螕` | difference | `CD_p` | `危|Cd|` |
| --- | --- | --- | --- | --- | --- |
| `wing_only_276` | 0.51209 | 0.52135 | 鈭?.78 % | 2.31e-04 | 0.0997 |
| `elliptic_1028` | 0.37325 | 0.39443 | 鈭?.37 % | 3.24e-03 | 0.0908 |
| `elliptic_1610` | 0.38371 | 0.39482 | 鈭?.81 % | 2.89e-03 | 0.0983 |

- `CL_螕` is **converged**: 0.39443 鈫?0.39482 between the two elliptic meshes
  (+0.10 %). The circulation unknowns are one per station and barely move.
- `CL_p` is **not**: 0.37325 鈫?0.38371 (+2.8 %), still climbing toward `CL_螕`.
  The pressure integral and the circulation integral do not converge to the same
  number, which is the claim issue #49 opened with, now measured on a case where
  the discretization error is not confounded by the wake model.
- `CD_p` is 30脳 smaller than `危|Cd|` on the elliptic cases (431脳 on the wing):
  the signed near-field drag is a difference of large cancelling terms, so it is
  not a quantity to optimize against directly.

## 8. Anchors

### 8.1 Closed-form sphere (no wing, no wake, no CAD)

An icosphere carrying the exact incompressible sphere solution, probed on radial
rays: the facet centroid is projected onto the exact unit sphere and then pushed
out by `offset 路 a`, so every probe is outside the body at every refinement (a raw
centroid of a coarse icosphere sits up to 6 % inside the sphere, where the exterior
closed form is not the physical answer). `Cp` rms error:

| faces | 0.01 a | 0.02 a | 0.05 a | 0.10 a | 0.20 a |
| --- | --- | --- | --- | --- | --- |
| 80 | 0.438 | 0.395 | 0.289 | 0.178 | 0.0825 |
| 320 | 0.392 | 0.318 | 0.164 | 0.0606 | 0.0185 |
| 1280 | 0.331 | 0.212 | 0.0528 | 0.0099 | 0.0038 |
| 5120 | 0.220 | 0.0832 | 0.0065 | 0.0015 | 0.0009 |

Interpretation, stated conservatively:

- **Away from the body the reconstruction converges cleanly.** At `0.2 a` the
  error drops 0.0825 鈫?0.0185 鈫?0.0038 鈫?0.0009, a factor of about 4.5 per halving
  of the panel size: second order. The operators, the kernel and the gradient are
  consistent.
- **Inside the first panel of the surface it does not.** At `0.01 a` the same four
  meshes give 0.438 鈫?0.392 鈫?0.331 鈫?0.220, an observed order near 0.5. The
  5120-panel mesh still has 22 % `Cp` rms error a hundredth of a radius off a
  sphere that it resolves with 5120 facets.
- So the pressure channel is not a wing problem, a wake problem or a CAD problem:
  it is the accuracy of a constant-per-panel layer representation evaluated near
  its own surface. That is exactly the object of Phase 1 (geometry-consistent
  gradient recovery) and Phase 2 (continuous P1 doublets).

Two identities also come out exactly and are useful as regression anchors: the
source strength of a closed surface sums to ~1e-16, and the pressure integral over
the closed faceted body cancels to ~1e-16 `qA` while the per-face magnitudes sum to
0.21 `qA`. Replacing `Cp` by random numbers gives 0.15 `qA` for the same sum, so
the cancellation is a property of the discrete solution and not of a zero array.

### 8.2 External anchors are **not** established in this phase

Issue #49 asks for VSPAERO / FLOWPanel on identical CAD and for published
experiments (NACA 0012, Weber鈥揃rebner). None of that is claimed here: those
anchors need their own reproducible run records and are Phase 0 follow-ups, not
part of this patch. What this patch freezes is the measurement, so an anchor can
later be compared against it without re-deriving what 鈥渢he baseline鈥?means.

## 9. A diagnostic, not a defect: the interior term of the assembly

The half assembly closes the interior Dirichlet condition with a frozen source
sheet `蟽 = -n路U鈭瀈 (`D_int 渭 = S 蟽`). The classical closed-body (Morino) form uses
the freestream term instead (`D_int 渭 = -U鈭灺穢`). The two right-hand sides are not
algebraically equal; on the sphere the measured interior mismatch is
`|S 蟽 + 蠁鈭瀨_rms = 0.357 / 0.376 / 0.381` against `|U鈭灺穢|_rms = 0.541 / 0.566 /
0.572` (80/320/1280 faces).

Measured near-surface error, however, is *smaller* for the production choice, not
larger:

| faces | offset | production `Cp` rms | Morino `Cp` rms |
| --- | --- | --- | --- |
| 320 | 0.02 a | 0.318 | 0.754 |
| 320 | 0.05 a | 0.164 | 0.394 |
| 1280 | 0.02 a | 0.212 | 0.558 |
| 1280 | 0.05 a | 0.0528 | 0.141 |

Conclusion: these are two discretizations of one singular integral equation, the
shipped one is not worse, and **Phase 0 does not recommend changing the
assembly**. The difference is recorded because it bounds how much of the
near-surface error can be attributed to the Dirichlet closure at all.

## 10. What Phase 0 explicitly does not establish

- No convergence order is claimed for the surface pressure, the near-field drag or
  the TE limit.
- No external anchor is claimed (see 8.2).
- No claim that the mesh-level ranking survives on another geometry: the two
  elliptic meshes are one case at two sizes, not two cases.
- The local wake-transport identity is *not* verified; only the normal flow
  through the sheet is measured (section 6).
- `physical_accuracy_accepted` stays `false`. Nothing here is a validated
  aerodynamic prediction.

## 11. Reproduction

```powershell
cd runs/pr49_git
$env:TMP=$env:TEMP=$env:TMPDIR="D:\3D\Projects\panel\runs\pipwork"
# one frozen reference case (writes manifest/metrics/report/state)
python -m panel.audit_reference `
    --case D:\3D\Projects\panel\runs\geom_robust\mesh\elliptic_s100 `
    --out D:\3D\Projects\panel\runs\ref_audit\elliptic_1028
# closed-form sphere anchor, with the alternative formulation A/B
python -m panel.analytic_sphere --subdivisions 1 2 3 4 --compare-formulation `
    --out D:\3D\Projects\panel\runs\sphere_anchor.json
# re-check a stored manifest against a fresh solve
python -m panel.baseline_manifest --surface <case>\surface.json `
    --report <run>\report.json --out <new>\manifest.json
```

Tests (12 total; `test/conftest.py` fixes the x64 flag and the sandbox-broken
`tmp_path`):

```powershell
[D:\3D\Projects\panel\.conda_env\python.exe] -m pytest test/test_baseline_manifest.py test/test_audit_reference.py -q
```

## 12. Next

Phase 1 (`panel/affine_consistent_recovery.py`) and Phase 2 (continuous P1 wall
doublets, `危_a D_{i,a} = D_{i,P0}`) each start from this baseline and must report
the same tables plus the linear residual and the interior trace of section 3. The
sphere table of 8.1 is the acceptance anchor for both: a geometry-consistent
gradient or a P1 doublet that does not move the `0.01 a`鈥揱0.05 a` columns is not
doing anything.
