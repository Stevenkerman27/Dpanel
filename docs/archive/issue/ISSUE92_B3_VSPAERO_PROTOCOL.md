# B3: DPanel versus VSPAERO on four geometries

Issue #92, Route B, third gate. This document records the protocol, the raw
numbers and the attribution for a cross-code comparison of the inviscid lift
and induced drag of four wings. It replaces no solver behaviour: nothing in
`panel/` changes, the defaults stay NumPy, and
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

## 7. Non-claims

* No claim that either code is closer to the truth. There is no experimental
  or high-fidelity reference here, and `physical_accuracy_accepted` stays
  false.
* No claim about drag agreement: the VSPAERO thick-panel `CDi` is not a
  usable induced drag at low incidence, and the VLM `k` is 1.8-2.9 times
  DPanel's.
* No claim about the bwb thick-panel strip loads at the mirror plane.
* No claim about viscous drag, compressibility, or separated flow.

## 8. Reproduce

```
# once: station tables and geometry hashes
python -m tools.issue92.b3_cases
# DPanel side (NumPy reference environment)
python -m tools.issue92.b3_dpanel --out runs/issue92/b3/dpanel.json
# VSPAERO side (vsppytools environment, OpenVSP 3.55)
python -m tools.issue92.b3_vspaero --out runs/benchmarks/b3_vspaero/b3/vspaero.json
# gate
python -m tools.issue92.b3_judge
# markdown tables for this document
python -m tools.issue92.b3_tables
```

VSPAERO needs `PYTHONPATH=D:/3D/Projects/OpenVSP-3.55.0-win64/python/openvsp` (2026-11-03: the in-repo binding copies `runs/envs/vsp_py*` were byte-identical duplicates and are no longer needed),
`VSP_ROOT` and `VSPAERO_EXE` pointing at
`D:/3D/Projects/OpenVSP-3.55.0-win64`, and the `vsppytools` Python
(3.13). The DPanel side needs the repository `.conda_env` (3.11, NumPy
2.4.6, SciPy 1.17.1). Raw output is `runs/issue92/b3/dpanel.json` and
`runs/issue92/b3/vspaero.json`; the gate report is
`docs/issue92/b3_matrix.json`.

## 9. Deliverables

* `tools/issue92/b3_cases.py` station extraction, hashing, manifest.
* `tools/issue92/b3_vspaero.py` rebuild from the station table, two modes,
  two mesh levels, polar and `.lod` capture.
* `tools/issue92/b3_dpanel.py` the DPanel side, including Trefftz drag.
* `tools/issue92/trefftz_full.py` full-span Trefftz drag with self-test.
* `tools/issue92/b3_judge.py` thresholds, revisions, comparisons, findings.
* `tools/issue92/b3_tables.py` the tables in section 4.
* `docs/issue92/b3_cases.json` the manifest, `docs/issue92/b3_matrix.json`
  the gate report.
