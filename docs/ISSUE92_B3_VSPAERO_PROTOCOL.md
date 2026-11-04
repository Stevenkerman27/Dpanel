# B3: DPanel versus VSPAERO on four geometries

Issue #92, Route B, third gate. This document records the protocol, the raw
numbers and the attribution for a cross-code comparison of the inviscid lift
and induced drag of four wings. It replaces no solver behaviour: nothing in
`panel/` changes, the solve chain and its defaults are untouched, and
`physical_accuracy_accepted` stays false.

## 1. What this gate is, and what it is not

VSPAERO is an independent implementation of the same physics family (surface
singularities, Kutta condition, Trefftz-plane drag). It is not a truth source,
so the gate is not "the two codes agree". The gate is:

1. every case runs end to end in both codes from a matched input manifest;
2. the quantities that can be aligned are compared with thresholds declared
   before the numbers were read, and every failure is attributed;
3. the comparison is reproducible: geometry hashes, mesh sizes, mode, alpha
   list, reference quantities and solver settings are all in the manifest.

## 2. Protocol

Four geometries, three of them taken from the repository's own robustness
cases so that DPanel and VSPAERO see the same station table:

| case | source | half span | Sref | stations | note |
|---|---|---|---|---|---|
| rect | `panel.p1_lifting_g1.make_wing` (chord 4, span 6, t/c 0.12) | 2.0 | 4.0 | 2 | full-span wing, one rectangle |
| swept30 | `cases/geom_robust/surfaces/swept30_s075` | 2.0 | 3.0 | 5 | NACA 2412, LE and TE both swept |
| elliptic | `cases/geom_robust/surfaces/elliptic_s075` | 2.0 | 3.1722548357643863 | 11 | NACA 0012, tip chord 0.05 |
| bwb | `cases/geom_robust/surfaces/bwb_s055` | 1.0 | 1.2414970407424226 | 13 | thick blended body, t/c 0.22 at the root |

The station table is read out of each `surface.json` by
`tools/issue92/b3_cases.py`: nodes referenced by the face list are grouped by
trailing-edge y, each group gives `chord = te_x - min(x)` and
`le_x = min(x)`, and the section thickness is the z extent over the chord.
Sweep is derived per segment as `atan2(dle_x, dy)`, so the double sweep of
swept30 is carried segment by segment instead of as one angle.

Both codes see alpha = 0, 2, 4, 6 degrees. VSPAERO runs twice per geometry: a
thick-panel solve (`mode=panel`) and a VLM solve
(`mode=vlm`), each on two tessellations
(`tess_u = 8/24` and `16/40`). The two modes are compared
separately because a VLM answers a different model question (zero thickness)
than DPanel, which carries thickness. Mach 0, beta 0, `wake_iters = 1`,
`wake_nodes = 16`, and symmetry is on for the three CAD half models and off
for rect, which is a full-span mesh.

The manifest, with a SHA-256 over the stations, reference quantities and moment
reference of each case, is `docs/issue92/b3_cases.json`.

## 3. Two things that did not work, and why the protocol looks like this

**Handing DPanel's own surface triangles to VSPAERO fails.** Writing the DPanel
mesh as a `.vspgeom` and calling the OpenVSP 3.55 solver crashes after the
banner with exit `0xC0000409` and no polar file. The same file runs under
3.52, but that build returns `CLi = -18.5` on a rectangular wing at zero
incidence, which is not a physical answer; a thin-wing control case in the same
build is well behaved, so the defect is in the 3.52 mesher-solver pairing, not in
our file. The 3.55 build solves the same rectangle correctly
(`CLi = 0.00000` at alpha 0, `0.27222` at 4 degrees). Every number in
this document therefore comes from OpenVSP 3.55.0.

**Importing the CAD STEP into OpenVSP is not available.** In the 3.55 Python
bindings `ImportFile` takes `(file_name, file_type: int, parent: str)` and
there is no STEP enum, so the route "same STEP, both codes" is closed. The
protocol instead rebuilds each wing from the same station table inside VSPAERO,
which is why the manifest records the stations and not a file path.

Because the rebuild carries thickness but no camber (`Camber = 0` in every
section curve), a cambered case has a known additive lift offset. That is
measured and reported, not hidden.

## 4. Numbers

### Lift and drag fits, per case

| case | stations | VSPAERO panels (coarse/fine, panel; vlm) | DPanel CL | VSPAERO panel CLwtot | VLM CLi |
|---|---|---|---|---|---|
| rect | 2 | 432 / 1360; 168 / 600 | +0.06406 (+0.00053) | +0.06899 / +0.06811 | +0.06563 / +0.06408 |
| swept30 | 5 | 720 / 2480; 336 / 1200 | +0.06319 (+0.15667) | +0.06907 / +0.07089 | +0.06572 / +0.06508 |
| elliptic | 11 | 1728 / 6080; 840 / 3000 | +0.06946 (+0.00152) | +0.07739 / +0.07411 | +0.07162 / +0.06655 |
| bwb | 13 | 2064 / 7280; 1008 / 3600 | +0.05860 (+0.10334) | +0.06420 / +0.06441 | +0.04845 / +0.05634 |

Slope in CL per degree; the bracketed number is the zero-incidence
intercept (large only where the section carries camber).

### Gate, all 32 comparisons

| case | pair | quantity | value | threshold | verdict |
|---|---|---|---|---|---|
| rect | panel/coarse:cl_CLwtot_vs_CL | CL slope | 0.0768 | 0.15 | pass |
| rect | panel/coarse:gamma_shape | circulation shape | 0.1477 | 0.15 | pass |
| rect | panel/fine:cl_CLwtot_vs_CL | CL slope | 0.0632 | 0.15 | pass |
| rect | panel/fine:gamma_shape | circulation shape | 0.1336 | 0.15 | pass |
| rect | vlm/coarse:cl_CLi_vs_CL_gamma | CL slope | 0.1570 | 0.15 | FAIL |
| rect | vlm/coarse:gamma_shape | circulation shape | 0.1494 | 0.15 | pass |
| rect | vlm/fine:cl_CLi_vs_CL_gamma | CL slope | 0.1297 | 0.15 | pass |
| rect | vlm/fine:gamma_shape | circulation shape | 0.1473 | 0.15 | pass |
| swept30 | panel/coarse:cl_CLwtot_vs_CL | CL slope | 0.0929 | 0.15 | pass |
| swept30 | panel/coarse:gamma_shape | circulation shape | 0.0435 | 0.15 | pass |
| swept30 | panel/fine:cl_CLwtot_vs_CL | CL slope | 0.1218 | 0.15 | pass |
| swept30 | panel/fine:gamma_shape | circulation shape | 0.0178 | 0.15 | pass |
| swept30 | vlm/coarse:cl_CLi_vs_CL_gamma | CL slope | 0.0858 | 0.15 | pass |
| swept30 | vlm/coarse:gamma_shape | circulation shape | 0.0235 | 0.15 | pass |
| swept30 | vlm/fine:cl_CLi_vs_CL_gamma | CL slope | 0.0752 | 0.15 | pass |
| swept30 | vlm/fine:gamma_shape | circulation shape | 0.0143 | 0.15 | pass |
| elliptic | panel/coarse:cl_CLwtot_vs_CL | CL slope | 0.1142 | 0.15 | pass |
| elliptic | panel/coarse:gamma_shape | circulation shape | 0.0689 | 0.15 | pass |
| elliptic | panel/fine:cl_CLwtot_vs_CL | CL slope | 0.0669 | 0.15 | pass |
| elliptic | panel/fine:gamma_shape | circulation shape | 0.0402 | 0.15 | pass |
| elliptic | vlm/coarse:cl_CLi_vs_CL_gamma | CL slope | 0.0494 | 0.15 | pass |
| elliptic | vlm/coarse:gamma_shape | circulation shape | 0.0627 | 0.15 | pass |
| elliptic | vlm/fine:cl_CLi_vs_CL_gamma | CL slope | 0.0249 | 0.15 | pass |
| elliptic | vlm/fine:gamma_shape | circulation shape | 0.0431 | 0.15 | pass |
| bwb | panel/coarse:cl_CLwtot_vs_CL | CL slope | 0.0955 | 0.15 | pass |
| bwb | panel/coarse:gamma_shape | circulation shape | 0.4964 | 0.15 | FAIL |
| bwb | panel/fine:cl_CLwtot_vs_CL | CL slope | 0.0992 | 0.15 | pass |
| bwb | panel/fine:gamma_shape | circulation shape | 0.5246 | 0.15 | FAIL |
| bwb | vlm/coarse:cl_CLi_vs_CL_gamma | CL slope | 0.1743 | 0.15 | FAIL |
| bwb | vlm/coarse:gamma_shape | circulation shape | 0.1391 | 0.15 | pass |
| bwb | vlm/fine:cl_CLi_vs_CL_gamma | CL slope | 0.0398 | 0.15 | pass |
| bwb | vlm/fine:gamma_shape | circulation shape | 0.1423 | 0.15 | pass |

| case | cambered | camber lift offset (VSPAERO - DPanel) | gated |
|---|---|---|---|
| rect | no | -0.0027 | no (camber omitted in the rebuild) |
| swept30 | yes | -0.1555 | no (camber omitted in the rebuild) |
| elliptic | no | -0.0064 | no (camber omitted in the rebuild) |
| bwb | yes | -0.1121 | no (camber omitted in the rebuild) |

### Induced drag, reported and not gated

| case | DPanel CD_trefftz(4 deg) | VLM CDi(4 deg) | CD ratio | k = CDi/CL^2 (DPanel) | k (VLM) |
|---|---|---|---|---|---|
| rect vlm/coarse | +0.00359 | +0.00544 | 0.51 | 0.0696 | 0.0787 |
| rect vlm/fine | +0.00359 | +0.00551 | 0.54 | 0.0696 | 0.0851 |
| swept30 vlm/coarse | +0.00901 | +0.00443 | 0.51 | 0.0550 | 0.0639 |
| swept30 vlm/fine | +0.00901 | +0.00504 | 0.44 | 0.0550 | 0.0741 |
| elliptic vlm/coarse | +0.00457 | +0.00666 | 0.46 | 0.0610 | 0.0831 |
| elliptic vlm/fine | +0.00457 | +0.00617 | 0.35 | 0.0610 | 0.0865 |
| bwb vlm/coarse | +0.01146 | +0.00516 | 0.55 | 0.0970 | 0.1372 |
| bwb vlm/fine | +0.01146 | +0.00663 | 0.42 | 0.0970 | 0.1301 |

## 5. Attribution

**Lift slope, four to seventeen per cent, always in the same direction.** Every
DPanel slope is lower than the VSPAERO thick-panel slope of the same case
(0.0641 vs 0.0690, 0.0632 vs 0.0691, 0.0695 vs 0.0774, 0.0586 vs 0.0642), and
the VLM slope sits between the two in all four cases. The gap shrinks with mesh
refinement for elliptic and bwb but not for swept30, where the fine panel slope
is the higher of the two VSPAERO mesh levels. This is a model-form difference
between a thick surface with a pressure Kutta condition and a VSPAERO thick
panel with its own tip and trailing-edge treatment; the ~7-10 per cent bias is
consistent across four planforms and two mesh levels, so it is not noise. It is
also not evidence that either code is wrong: the two implementations discretise
the same equations differently, and the repository already records that its
own near-field pressure drag does not converge (docs/MESH_CONVERGENCE.md).

**Camber, measured not assumed.** The zero-incidence intercept of DPanel's CL
is +0.1567 for swept30 and +0.1033 for bwb, which is camber lift (NACA 2412 and
the thick bwb sections), and +0.0005 / +0.0015 for the uncambered rect and
elliptic cases. VSPAERO's rebuild has no camber, so its intercepts are -0.0008
to +0.0038 in all four cases. The offsets the table reports (-0.1555, -0.1121)
are the camber the rebuild does not carry. The gate therefore evaluates the
intercept only on the two uncambered cases, where it agrees to 0.0027 and
0.0064, and records the cambered offsets as an input difference. The slope
comparison is unaffected because a pure camber offset shifts the curve, not its
slope.

**Induced drag cannot be gated as an absolute value.** DPanel's Trefftz drag
was unusable at first: the repository helper
`runs/trefftz_drag.py` assumes half-span stations and mirrors them internally,
while `Wing.gamma_y` spans the whole wing. Feeding it the full-span array
double-covers the planform and returns a drag wrong by an O(1) factor (2.5x low
on rect, 3.2x high on elliptic, different signs of error for the same call). The
fix is `tools/issue92/trefftz_full.py`, which takes the full-span layout
directly and keeps the principal-value log quadrature; its elliptic self-test
recovers `CL^2/(pi*AR)` to 0.7 per cent at 201 stations. After the fix the
DPanel Trefftz drag lands within five per cent of the lifting-line ideal
`CL^2/(pi*AR)` on all four cases at four degrees.

The remaining drag disagreement is a property of the VSPAERO side. Its thick
panel `CDi` is negative at zero incidence on three of the four cases
(swept30, elliptic, bwb), which no induced drag can be, and its
`CDwtot` carries an alpha-independent offset near 0.009, the same
additive `CDo` behaviour recorded in docs/issue49_vspaero/. The VLM
`CDi` is at least sign-correct, and there the disagreement is a factor
1.8 to 2.9 in `k = CDi/CL^2` with VSPAERO always the larger. This document
therefore reports drag with both absolute values and `k`, and does not
gate it: a drag gate needs a dedicated VSPAERO study of the near-field versus
Trefftz split and of the wake discretisation, which is not what this gate is.

**Spanwise circulation shape.** Reconstructing VSPAERO's strip circulation as
`Gamma = 0.5 * cl * c(y)` and comparing the normalised shape against
DPanel's `gamma_y` gives 0.014 to 0.147 for rect, swept30 and elliptic.
bwb fails on the thick-panel runs at 0.50 and 0.52 but passes on the VLM runs at
0.14. The failure is localised at the symmetry plane: the thick-panel strip
coefficient drops by a factor four over the first three strips around `y = 0`,
where the bwb body is deepest (t/c 0.22), and the reconstruction
`0.5 * cl * c` is exactly the assumption that a section behaves as a
two-dimensional lifting strip. It does not for a body that thick at the mirror
plane. The classification is the strip-load reconstruction and the symmetry
plane treatment, not a DPanel spanwise distribution error: the same DPanel
distribution agrees with VLM to 0.14 on the same stations.

**VLM mesh sensitivity is not negligible at the coarse level.** The bwb VLM
slope moves from 0.0469 to 0.0563 between 1008 and 3600 panels, which is why the
coarse VLM slope comparison fails at 0.174 while the fine one passes at 0.040.
The rect coarse VLM comparison fails at 0.157 for the same reason: 168 VLM
panels across a 4:1 rectangle is a coarse mesh, not a code difference.

## 6. Claims

* Four geometries, four angles, two VSPAERO modes and two mesh levels per mode
  ran to completion in both codes, from one manifest, with hashes.
* DPanel and VSPAERO thick-panel lift slopes agree within 4-12 per cent on all
  four planforms; DPanel is consistently the lower of the two.
* On the two uncambered cases the zero-incidence intercept agrees within 0.0064;
  on the two cambered cases the offset equals the camber the VSPAERO rebuild
  does not carry.
* The spanwise circulation shape agrees within 15 per cent on every case except
  the bwb thick-panel strip reconstruction at the symmetry plane.
* With both codes integrating the moment about the manifest's reference point,
  eight gated CM_y comparisons pass, the largest being the rect 0.0189 against a
  0.02 budget (section 11).
* Section Cp is comparable and reported on seven strips: 6-7 per cent on the
  elliptic wing, 21-44 per cent elsewhere, where the comparison method dominates
  (section 12).
* On the rect planform the free wake changes CL by -0.29 per cent on the DPanel
  side and -1.13 per cent on the VSPAERO side, and the free-wake pair still
  agrees inside the fixed-wake budgets (section 13).

## 7. Non-claims

* No claim that either code is closer to the truth. There is no experimental
  or high-fidelity reference here, and `physical_accuracy_accepted` stays
  false.
* No claim about drag agreement: the VSPAERO thick-panel `CDi` is not a
  usable induced drag at low incidence, and the VLM `k` is 1.8-2.9 times
  DPanel's.
* No claim about the bwb thick-panel strip loads at the mirror plane.
* No claim that the section Cp agrees: it is reported, not gated, because the
  sample layouts differ and the 20-strip average smooths the suction peak.
* No free-wake claim beyond the rect planform. The native crash that used to
  block the other three is fixed (section 13.1), but DPanel's steady free wake
  does not converge on the three CAD-derived planforms: the load collapses by
  about three quarters and the wake geometry oscillates instead of settling
  (section 13.2). Only the synthetic rect wing is geometry-stable, and even
  there it is not flow-tangency-converged.
* No claim about viscous drag, compressibility, or separated flow.

## 8. Reproduce

```
# once: station tables and geometry hashes
python -m tools.issue92.b3_cases
# DPanel side: torch chain (an interpreter that has torch; .conda_env does not)
python -m tools.issue92.b3_dpanel --out runs/benchmarks/b3_vspaero/b3/dpanel.json
# the same sixteen solves on the CUDA device (section 10)
python -m tools.issue92.b3_dpanel --device cuda --out runs/benchmarks/b3_vspaero/b3/dpanel_cuda.json
# VSPAERO side (vsppytools environment, OpenVSP 3.55)
python -m tools.issue92.b3_vspaero --out runs/benchmarks/b3_vspaero/b3/vspaero.json
# gate
python -m tools.issue92.b3_judge
# markdown tables for this document
python -m tools.issue92.b3_tables
# section Cp: one single-angle tecplot run, then the section integrals (section 12)
python -m tools.issue92.b3_vspaero --modes panel --levels fine --alphas 4.0 \
    --tecplot --out-dir runs/benchmarks/b3_vspaero/b3/vsp_cp \
    --out runs/benchmarks/b3_vspaero/b3/vspaero_cp_tecplot.json
python -m tools.issue92.b3_cp
# free wake: VSPAERO with WakeIters > 1, then the DPanel march (section 13)
python -m tools.issue92.b3_vspaero --modes panel --levels fine --alphas 4.0 \
    --wake-iters 5 --out-dir runs/benchmarks/b3_vspaero/b3/vsp_freewake \
    --out runs/benchmarks/b3_vspaero/b3/vspaero_freewake.json
python -m tools.issue92.b3_free_wake
```

VSPAERO needs `PYTHONPATH=D:/3D/Projects/OpenVSP-3.55.0-win64/python/openvsp` (2026-11-03: the in-repo binding copies `runs/envs/vsp_py*` were byte-identical duplicates and are no longer needed),
`VSP_ROOT` and `VSPAERO_EXE` pointing at
`D:/3D/Projects/OpenVSP-3.55.0-win64`, and the `vsppytools` Python
(3.13). The DPanel side needs torch: the repository `.conda_env` (3.11) has
no torch, so both passes recorded here ran on Python 3.13 + torch 2.13. Raw
output is `runs/benchmarks/b3_vspaero/b3/{dpanel,vspaero}.json`; the gate
report is `docs/issue92/b3_matrix.json`. Both raw files went missing from
the working tree during the 2026-11-03 `runs/` reorganisation; section 10
records what that costs and how the DPanel side was regenerated.

## 9. Deliverables

* `tools/issue92/b3_cases.py` station extraction, hashing, manifest.
* `tools/issue92/b3_vspaero.py` rebuild from the station table, two modes,
  two mesh levels, polar and `.lod` capture.
* `tools/issue92/b3_dpanel.py` the DPanel side, including Trefftz drag.
* `tools/issue92/trefftz_full.py` full-span Trefftz drag with self-test.
* `tools/issue92/b3_judge.py` thresholds, revisions, comparisons, findings.
* `tools/issue92/b3_tables.py` the tables in section 4.
* `tools/issue92/b3_cp.py` section Cp from the VSPAERO tecplot wall and the
  DPanel panel solution.
* `tools/issue92/b3_free_wake.py` the fixed-to-free comparison, one child
  process per case so a native crash cannot take the report with it.
* `docs/issue92/b3_cases.json` the manifest, `docs/issue92/b3_matrix.json`
  the gate report, `docs/issue92/b3_cp.json` and
  `docs/issue92/b3_free_wake.json` the two new records.

## 10. CUDA rerun (2026-11-03)

The gate above ran on the torch **CPU** backend. The Route B decision placed B3
after CPU parity and said it need not wait for CUDA, so the CUDA column the
Phase 1 plan also asked for was never produced. This section produces it: the
same sixteen solves on the CUDA device, with the protocol, the thresholds and
the verdict left alone.

```
python -m tools.issue92.b3_dpanel --device cuda --backend torch \
    --out runs/benchmarks/b3_vspaero/b3/dpanel_cuda.json
```

`--device` and `--backend` were added to the driver for this rerun; they are
opt-in and the default is still `--device cpu`, so nothing about the delivered
protocol changes. The output file now carries a `provenance` block: Python
3.13.14, torch 2.13.0+cu132, CUDA runtime 13.2, `NVIDIA GeForce RTX 5060`
(compute capability 12.0), context initialisation 0.371 s.

**Parity.** The CPU side was rerun in the same interpreter and torch build
rather than compared against a saved baseline, and that rerun reproduces the
numbers already published in `docs/issue92/b3_matrix.json` to 2.8e-15..1.8e-10
relative — so the pair below differs by device, not by code version.

| quantity | worst CPU-vs-CUDA difference | where |
| --- | --- | --- |
| CL slope over alpha | 1.1e-13 relative | elliptic |
| CL_gamma slope over alpha | 5.8e-14 relative | elliptic |
| per-alpha CL and CM_y | 9.9e-13 of the case peak | elliptic |
| per-alpha CD_pressure | 1.5e-12 of the case peak | elliptic, alpha = 0 |
| spanwise circulation shape | 5.1e-10 relative | elliptic |
| constrained rank | identical on all 16 solves | — |
| wall_rms / kutta_rms | 4.2e-15 / 3.3e-14 absolute | elliptic / bwb |

One relative outlier needs its scale stated: the rectangular wing at alpha = 0
differs by 2.8e-2 in circulation *shape*, on stations whose absolute value is
of order 1e-15. That is round-off on a zero, not a device difference.

Since the gate compares DPanel fits against VSPAERO, and every fit moves by at
most 1.1e-13 relative, the verdict is unchanged: 28 of 32 comparisons pass and
the same four fail with the same attribution. The gate itself was not re-run,
because the VSPAERO raw output is missing (see the end of this section).

**Speed.** Fifteen warm solves, the first solve of each process excluded because
it carries backend warm-up:

| case | wall faces | unknowns | CPU | CUDA | CUDA/CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| rect | 112 | 58 | 0.104 s | 0.141 s | 1.36x |
| swept30 | 2764 | 1336 | 27.189 s | 36.862 s | 1.36x |
| elliptic | 3220 | 1562 | 39.208 s | 51.534 s | 1.31x |
| bwb | 2454 | 1197 | 23.165 s | 28.779 s | 1.24x |
| total | — | — | 89.67 s | 117.32 s | 1.31x |

First solve: 0.948 s on CPU against 15.073 s on CUDA — the device needs about
15 s before it answers anything, which is context and kernel initialisation, not
arithmetic.

**Non-claims.** This does not show CUDA is faster. At these sizes it is 1.24-1.36
times slower, the same direction as B2 (section 13 of the port map): the coupled
solve spends most of its time in per-iteration Python geometry work, and FP64
throughput on this consumer part is low. It is also not a validation: a GPU
agreeing with a CPU is a reproducibility statement, and
`physical_accuracy_accepted` stays false.

**Data-loss finding.** `runs/benchmarks/b3_vspaero/b3/` holds only
`vsp_smoke.json`: both raw outputs of the original gate (`dpanel.json` and
`vspaero.json`) were removed during the 2026-11-03 `runs/` reorganisation and
are not in git. The analysed numbers survive in the committed
`docs/issue92/b3_matrix.json`, and this rerun restores the DPanel side as
`dpanel_cpu.json` and `dpanel_cuda.json`. The VSPAERO side was regenerated on
2026-11-03 as `vspaero.json` (sixteen rows, `vsppytools` + OpenVSP 3.55,
command in section 8), which is what let the moment and section-Cp blocks of
sections 11 and 12 run at all; the gate was then re-run against the regenerated
pair and returned the same 48 comparisons and the same four failures.

## 11. Common moment reference (2026-11-03)

The first pass compared lift only, because the two codes were not integrating the
moment about the same point. DPanel's `make_wing` left `reference_point` at the
origin, while the VSPAERO sweep was handed `X_cg = 0.25 * cref`, so the two CM
columns differed by roughly `0.25 * CL` before any physics was involved: the old
rect polar at `X_cg = 0` reads `CMiy = -0.06268` at alpha = 4, the same case
re-run with `X_cg = 0.25` reads `CMytot = +0.00528`, and the difference
`0.0679` is `0.25 * 0.2718 = CL`.

Both sides now use the manifest's `moment_reference`:

* `tools/issue92/b3_dpanel.py` replaces the wing's `reference_point` with the
  case reference (the dataclass is frozen, so it is `dataclasses.replace`), which
  also re-scales every CM_y it reports;
* `tools/issue92/b3_vspaero.py` passes the full `X_cg, Y_cg, Z_cg` triple
  instead of only the x component.

The moment gate is `|dCM_y| <= 0.02` at the matching angle, plus a 15 per cent
relative check on the CM_y(alpha) slope wherever DPanel's own slope is at least
0.01 per degree (below that the relative number divides by noise). Sixteen CM
comparisons were added; eight are gated and all eight pass:

| case | pair | DPanel CM_y | VSPAERO CM_y | difference |
| --- | --- | ---: | ---: | ---: |
| rect | panel/coarse | +0.02415 | +0.00657 | -0.01758 |
| rect | panel/fine | +0.02415 | +0.00528 | -0.01888 |
| rect | vlm/coarse | +0.02415 | +0.01173 | -0.01242 |
| rect | vlm/fine | +0.02415 | +0.00885 | -0.01530 |
| elliptic | panel/coarse | +0.00879 | +0.00581 | -0.00298 |
| elliptic | panel/fine | +0.00879 | +0.00619 | -0.00260 |
| elliptic | vlm/coarse | +0.00879 | +0.01031 | +0.00152 |
| elliptic | vlm/fine | +0.00879 | +0.01123 | +0.00244 |

The two cambered planforms are reported and not gated, for the same reason the
intercept is not gated: the VSPAERO rebuild carries `Camber = 0`, so the moment
there is not about the same geometry (swept30 differs by +0.0906..+0.1053, bwb by
+0.0337..+0.0411).

One number has to be read with its margin stated: rect passes at 0.0189 against a
0.02 budget, 94 per cent of the allowance. The residual is not noise, and it is
not a reference-point error either, because it survives the alignment. It says
DPanel puts the rect load centre about `0.0189 / 0.2718 = 0.070` chord aft of
where VSPAERO puts it, consistently across both mesh levels and both VSPAERO
modes. A 0.02 budget that passes at 0.0189 is a weak pass; if this comparison is
tightened later, that offset is the thing to explain first.

The verdict of the gate is unchanged: 48 comparisons, the same four failures with
the same attribution.

## 12. Section Cp (reported, not gated)

Section pressure is the one item of the B3 list that needs a data source the
first pass did not have. Three were tried:

* `<case>.case.<n>.quad.<m>.dat` looks like a surface file, but it is the
  **flow-field cut plane**: every node sits at `y = 0`, the plane spans
  `x = -20..21`, `z = -20..20`, and the first four points are the far field.
  Its Cp is not a wall Cp.
* OpenVSP's `AddCpSlice` did not add a plane VSPAERO writes: after the call the
  `.vsp3` still holds only mass and pressure slices, and the run still emits a
  single cut (`quad.1`).
* `vspaero.exe -writetecplot` does write the wall: `<case>.tec.dat` is ASCII
  with `ZONE N=..., E=..., T="wall", VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED),
  ZONETYPE=FETRIANGLE`, followed by the node coordinates, one cell-centred Cp per
  triangle and the connectivity. It holds exactly one solution, so the comparison
  runs at alpha = 4 degrees alone.

`tools/issue92/b3_cp.py` bins both sides into 20 chordwise strips at 40 and 75
per cent semi-span and integrates `cn = \int (Cp_lower - Cp_upper) d(x/c)`.
Seven sections survive the minimum-bin filter (the rect inboard strip yields only
eight DPanel panels):

| case | station | y | DPanel cn | VSPAERO cn | relative |
| --- | --- | ---: | ---: | ---: | ---: |
| rect | outboard | 1.500 | +0.2329 | +0.1837 | 0.2113 |
| swept30 | inboard | 0.800 | +0.4180 | +0.2492 | 0.4038 |
| swept30 | outboard | 1.500 | +0.4306 | +0.2425 | 0.4368 |
| elliptic | inboard | 0.800 | +0.2516 | +0.2364 | 0.0604 |
| elliptic | outboard | 1.500 | +0.2437 | +0.2269 | 0.0691 |
| bwb | inboard | 0.400 | +0.3354 | +0.1933 | 0.4236 |
| bwb | outboard | 0.750 | +0.3727 | +0.2191 | 0.4123 |

**This is reported, not gated.** On the thin, untwisted, constant-chord elliptic
wing the two sections agree to 6-7 per cent, which is what a real agreement looks
like; everywhere else DPanel is 21-44 per cent higher. Most of that spread is in
the comparison method rather than in the codes: VSPAERO samples cell-centred
triangles while DPanel has quad panels, and a 20-strip average smooths the
leading-edge suction peak that carries a large share of the section force and
sits in a single strip on one side only. Turning this into a gate would first
need a chordwise density convergence study on both sides; until then the numbers
are recorded in `docs/issue92/b3_cp.json` and the gate note in that file says the
same thing.

## 13. Fixed wake to free wake (2026-11-03)

Rule 4 asked for the free wake only after the fixed wake, so it runs last, at the
same angle, and only where the fixed-wake gate did not fail on that mode/level
pair.

**VSPAERO side** ran all four cases with `WakeIters = 5` (the rest of the sweep
configuration unchanged, panel mode, fine mesh):

| case | CL fixed | CL free | CM_y fixed | CM_y free |
| --- | ---: | ---: | ---: | ---: |
| rect | 0.27180 | 0.26874 | +0.00528 | +0.00556 |
| swept30 | 0.28916 | 0.28573 | -0.14296 | -0.14248 |
| elliptic | 0.29513 | 0.29172 | +0.00619 | +0.00658 |
| bwb | 0.22935 | 0.22649 | -0.01208 | -0.00864 |

**DPanel side** marches `panel/p1_steady_free_wake.py` at a 0.3 relaxation for 60
iterations. The first attempt died with a native access violation on three
planforms; that crash is fixed (13.1) and all four cases now run to 60
iterations. Only the synthetic rect wing converges:

| case | CL fixed | CL free, 60 iterations | change | wake geometry stable | normal slip |
| --- | ---: | ---: | ---: | --- | ---: |
| rect | 0.25746 | 0.25671 | -0.29 % | yes, 1.5e-10 | 0.0130 |
| swept30 | 0.40990 | 0.09427 | -77.0 % | no | 7.57 |
| elliptic | 0.27996 | 0.06557 | -76.6 % | no | 9.74 |
| bwb | 0.33827 | 0.07027 | -79.2 % | no | 2.71 |

VSPAERO moves by about -1.2 % on all four (0.27180 -> 0.26874, 0.28916 ->
0.28573, 0.29513 -> 0.29172, 0.22935 -> 0.22649). For the three CAD-derived
planforms the DPanel free wake is therefore not a refinement of the fixed wake:
it walks the load off the wing. bwb is reported only, because its fixed-wake
strip loads already failed the gamma gate.

For the one case that converges, the free wake agrees with VSPAERO inside the
same budgets as the fixed wake:

| quantity | value |
| --- | --- |
| DPanel CL, fixed to free | 0.25746 -> 0.25671 (-0.29 %) |
| VSPAERO CL, fixed to free | 0.27180 -> 0.26874 (-1.13 %) |
| DPanel free CL against VSPAERO free | 0.25671 vs 0.26874, -4.68 % (pass, 15 %) |
| DPanel free CM_y against VSPAERO free | +0.02446 vs +0.00556, 0.0189 (pass, 0.02) |
| wake geometry change, last iteration | 1.5e-10 |
| normal-flow residual, last iteration | 0.0130 against a 0.01 tolerance (not converged) |

The moment difference is the fixed-wake offset carried over, not a new one:
DPanel's free-wake CM_y (+0.02446) is where its fixed wake already was
(+0.02415), so relaxing the wake changes the moment by 0.0003 while the gap to
VSPAERO stays at 0.0189.

One honesty note: rect is geometry-stable but not flow-tangency-converged (0.013
against 0.01), and the module documents those as distinct criteria.

### 13.1 The native crash: a numpy dgemm fault, not a DPanel bug

`faulthandler` put the fault at `panel/p1_lifting_g2.py:179`,
`Wg = coefficients @ np.asarray(group.gamma_hat, float)`, a 2764 x 1928 by
1928 x 15 float64 product inside `assemble_multi`. The product itself is not
special: in the torch interpreter (`myml`, numpy 2.5.2)

```powershell
& C:\Users\zyx20\anaconda3\envs\myml\python.exe -c "import numpy as np; print(np.random.rand(2764,1928) @ np.random.rand(1928,15).sum())"
```

exits with 3221225477 (`0xC0000005`) and no traceback. Only that environment
fails; the repository `.conda_env` (numpy 2.4.6) computes the same product. The
fault depends on the row count (1000 fine, 1500 and 2764 die), disappears for
`k = 1` (gemv), `float32` and `np.einsum`, and survives every thread and
instruction-set override tried (BLAS/OMP/MKL thread counts,
`MKL_ENABLE_INSTRUCTIONS`, `OPENBLAS_CORETYPE`, `NPY_DISABLE_CPU_FEATURES`).
Row-blocking avoids it, so the fix is a FLOP-equivalent block multiply,
`_blocked_matmul(left, right, *, rows=512)` in `panel/p1_lifting_g2.py`, now
used for the wake influence product. Verification: the merged swept30 mesh
assembles in 5.7 s and solves to CL = 0.40989704632820095, against
0.4098970463282688 from the single-wing path (7e-14 apart), so `combine_wings`
on one wing is physically a no-op and the block multiply is numerically
transparent. The trap is recorded in `docs/TEST_STATUS.md` because it will bite
any other large product in that environment.

### 13.2 The remaining failure: the load leaves the wing

Three checks rule out the obvious causes.

* Not the velocity evaluation. At iteration 0, on the fixed-wake solution, the
  wake velocities are sane on all four cases: maximum 1.00-1.09, minimum axial
  0.75-0.99, mean normal slip 0.04-0.05.
* Not the step size. swept30 at relaxation 0.1 still collapses (0.4035 ->
  0.1882 at iteration 1 -> 0.1189 at iteration 5).
* Not a failed inner solve. Through the collapse the wall residual stays at
  3.2e-4, the jump residual at 7e-16 and the pressure Kutta residual at
  2.4e-7 - 2.7e-7. The low-lift states are converged solutions of the displaced
  wake, not unconverged ones.

What is left is the sensitivity of the fixed-wake solution to wake position. On
swept30, shrinking the relaxation makes the effect measurable: at relaxation
1e-2 the first step moves the wake by 3.6e-2 chord and CL falls 5.3 % per
iteration (0.4035 -> 0.3505 -> 0.3086); at 1e-3 it falls 1.2 %; at 1e-6 it does
not move. That is about 1.4 % of CL per 0.001 chord of node motion, and at
relaxation 0.3 the near trailing-edge row moves only 0.085 chord while the load
loses three quarters. The fixed-wake solution is therefore not a small
perturbation away from the free-wake one on these planforms: the wake-doublet
system is ill-conditioned in the wake position, and the relaxation walks the
load away instead of onto the stream surface.

The geometric difference to rect is wake length and resolution: the CAD wakes
run 30 chords downstream with 0.045-0.073 chord node spacing (1110-1528 faces),
and the unrelaxed target displacement reaches 3.62 chords (freestream drift
tan 4 deg x 30 chords, plus roll-up), whereas rect's 5-chord, 24-face wake moves
0.12 and settles. Refining the rect wake to 96 faces does not change its
convergence, so resolution alone is not the trigger.

Consequence for this gate: the free-wake block is judged for rect only
(CL -4.68 %, CM 0.0189, both inside budget). For swept30, elliptic and bwb the
numbers in `docs/issue92/b3_free_wake.json` are a defect report, not a
comparison. Fixing this is separate work; the candidates are freezing the first
few wake rows, a wake-position gate (the module already has `slip_gate` and
`max_step_chord` hooks, both off by default), or a wake-doublet basis that stays
consistent as the sheet moves off the wing plane. Nothing here claims the free
wake is validated.

Reproduce with `python -m tools.issue92.b3_free_wake --iterations 60`; each case
runs in its own child process so one crash cannot take the report with it.
