# G1 adapter fix: TE pairing from the declared upper/lower TE faces

Issue #58 asked to separate small blunt-TE geometry from Kutta topology. Before any
thickness ladder this fixes the loader defect that made the true-CAD G1 study
meaningless: the adapter paired the two banks through `trailing_edges`, which is the
TE **polyline** (a chain of TE edges), not a bank pairing. No pair shared a node, so
the `if upper == lower` potential-trace split never fired, the model had no
circulation unknown, and least squares returned `Gamma = 0` on a converged system.

## Change

`panel/p1_lifting_g1.py::from_dpanel_half_surface` now

1. mirrors `upper_te_faces`/`lower_te_faces` by face index (`+len(faces)`),
2. takes the nodes each pair shares (the physical TE edge),
3. duplicates only the lower-bank P1 potential unknown of each TE node,
4. anchors `Gamma = 0` at the extremes of the real TE node set,
5. refuses to guess: a pair with no shared node raises
   `ValueError('upper/lower TE faces share no node at row ...')`.

The legacy `trailing_edges` path survives as `_from_dpanel_half_surface_edges`.

## Result on the 396-panel half CAD (792 wall faces, 486 DOFs, 88 TE rows, 49 Gamma)

| mode | alpha | CL | CL_gamma | CD_pressure | wall rms | kutta rms | max abs gamma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| signed_speed | 0 | 0.115317 | 0.135451 | 0.005964 | 1.3e-3 | 3.0e-14 | 0.1215 |
| signed_speed | 5 | 0.185110 | 0.195863 | -0.006339 | 1.6e-3 | 3.1e-14 | 0.1698 |
| pressure | 0 | 0.150269 | 0.163937 | 0.003355 | 1.3e-3 | 4.3e-6 | 0.1439 |
| pressure | 5 | 0.304391 | 0.289408 | -0.002495 | 1.5e-3 | 8.3e-7 | 0.2293 |

Before the fix the same mesh gave `CL = -0.023` (pressure) with `max abs gamma = 0.014`.

`Gamma(y)` is now a smooth lobe of about 0.20-0.23 across the span, but it still
changes sign in `|y| < 0.3`, the wing/body junction strip (the body reaches y = 0.35).
Those TE rows are not a clean Kutta edge and are reported as an open modelling
question, not masked.

## The remaining gap is not obviously a closure defect

Measured on this mesh the full projected planform is 5.556 (wing 3.424, body 2.132)
while the surface declares `reference_area = 4.0`, so both solvers' CL are inflated
by 1.39 relative to the true planform. With `b = 4`, `S = 5.556` (AR 2.88) the
finite-wing slope is 3.71/rad, giving `CL_planform = 0.324` at 5 deg, i.e.
`CL(S_ref = 4) = 0.450`. The fixed P1 value 0.304 is -32% and the production P0 value
0.5312 is +18% of that estimate, so the two are roughly symmetric about lifting line
rather than P1 being catastrophically low.

Two questions remain before a thickness ladder: which area P0 divides by on a
wingbody, and whether TE rows inside the body junction belong in the Kutta functional.

`physical_accuracy_accepted=false`; the production P0 path was not touched.