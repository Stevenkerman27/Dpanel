# Issue #71 — measured edge exponent in the existing P1 solution

Companion to `docs/ISSUE71_EDGE_SINGULARITY_DERIVATION.md` (PR #74, merged `5a06d69`).
That document derives `lambda = pi/Omega` for a straight Neumann wedge and asks (gate 3)
for a measurement of the residual in edge-distance bins. This note reports the
measurement on the *existing* closed-body P1 solution, without any enrichment.

Script: `tools/issue71/edge_exponent.py`. Run from a worktree with `panel/`.

## Setup

`panel.p1_lifting_g1.make_wing(chord_panels=c, span_panels=8, camber=0.02)`, alpha = 5 deg,
closed body (no TE cut, no wake, Gamma = 0), `A mu + W Gamma = -phi_inf.n` solved by least
squares with area weights. At the mid-span station the upper and lower nodal potentials are
compared. `_airfoil` has `t(0) = t(1) = 0`, so both LE and TE are zero-thickness cusps:
`Omega -> 2*pi`, `lambda = 1/2`.

## (a) Leading edge: clean square root, consistent with lambda = 1/2

`mu_upper(x) - mu_lower(x)` on `x in [0.005, 0.05]`:

| c | log-log slope | fitted amplitude (Delta_mu = 2 a sqrt(x)) | points |
| --- | --- | --- | --- |
| 16 | +0.527 | 0.202 | 2 |
| 24 | +0.447 | 0.174 | 2 |

The LE nodes at `x = 0` are shared (`make_wing` drops the bank tag at `i = 0`), so
`Delta_mu(0) = 0` and `Delta_mu ~ sqrt(x)` is exactly the antisymmetric `lambda = 1/2`
mode of the `2*pi` fluid wedge. The amplitude changes by 14% between the two meshes.

## (b) Trailing edge: a one-cell jump, not a square root

With `s = 1 - x` the four stations closest to the TE are:

| c | Delta_mu (s descending) | s of the last real station | Delta_mu/sqrt(s) there |
| --- | --- | --- | --- |
| 8 | -0.2658 -0.3900 -0.5437 -0.6850 | 0.038 | -1.019 |
| 12 | -0.4461 -0.5563 -0.6764 -0.7794 | 0.017 | -2.149 |
| 16 | -0.5055 -0.5906 -0.6737 -0.7371 | 0.0096 | -3.027 |
| 24 | -0.5041 -0.5499 -0.5934 -0.6227 | 0.0024 | -4.213 |

Two readings: the jump one cell from the TE is 0.62-0.78 and does not shrink with h, and
`Delta_mu/sqrt(s)` grows roughly linearly with c, i.e. `Delta_mu` is nearly constant rather
than `sqrt(s)`. The TE node itself is shared for both banks, so `Delta_mu` must be exactly
zero there. The discrete shape is therefore a jump of order 0.7 held over the whole domain
that collapses to zero inside the **last cell**. The jump width is one cell, so its gradient
is `O(1/h)` and mesh refinement cannot remove it.

## (c) A velocity exponent fit is not an acceptance quantity

Near-TE `|V|` log-log slope at mid span:

| c | window `s < 0.2` | window `s in [0.02, 0.15]` |
| --- | --- | --- |
| 8 | -0.489 | -0.547 |
| 12 | -0.449 | -0.056 |
| 16 | -0.423 | -0.247 |
| 24 | -0.414 | +0.050 |

The sign changes with the window, so the velocity field does not provide a stable exponent.
The reliable witnesses are the potential trace (LE square root) and the jump shape (TE).

## Consequence

1. The `lambda = 1/2` antisymmetric mode lives on a boundary pair with opposite signs. In a
   closed body it is visible at the LE; at the TE it is squeezed into one cell by the shared
   node and the missing cut. Enrichment therefore cannot be separated from the cut: the cut
   and Gamma select the amplitude, the enrichment decides whether it can be represented.
2. The edge-band residual is controlled by the width of the last cell, not by h, so
   "residual decreases with order" is not a usable gate here. Use instead: fixed physical
   distance `s` point values converging with h, and the measured jump width shrinking with h.
3. Do not fit lambda to improve CL (as the derivation document already states).

## Reproduction

```
pytest test/test_p1_edge_singularity.py -q -p dsh_tmp_plugin
python tools/issue71/edge_exponent.py
```