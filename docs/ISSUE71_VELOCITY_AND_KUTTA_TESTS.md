# Issue #71: velocity-field convergence and Kutta estimators that avoid the last chordwise cell

Date: 2026-10-29. Status: **negative result**. `physical_accuracy_accepted=false`, no production
default changed. Scripts: `tools/issue71/`. Earlier note: `docs/ISSUE71_CHORDWISE_DIVERGENCE.md`.

## 0. Question

Two hypotheses were tested against the measured chordwise divergence (CL 0.457 -> -0.819 for
c = 4..32):

- **H1** the P1 surface velocity is not a convergent field, so every number built on it (CL, CM,
  the pressure Kutta residual) inherits the divergence;
- **H2** the divergence is a Kutta-selection artefact, i.e. evaluating the pressure jump on the
  single cell that touches the trailing edge; excluding that cell should fix it.

H1 holds for the base solve; H2 is false as stated. Details below.

## 1. Pointwise velocity convergence (production pressure Kutta)

`make_wing(chord_panels=c, span_panels=8, camber=.02)`, alpha = 5 deg, mid-span probes on the
upper surface. |V| is read from the face that contains each probe point, so each refinement
samples the same physical location with a smaller stencil.

| c | x/c=0.02 | 0.10 | 0.30 | 0.60 | 0.90 | 0.98 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1.519 | 1.518 | 1.288 | 1.058 | 1.030 | 1.030 |
| 6 | 1.323 | 1.398 | 1.230 | 1.105 | 1.033 | 0.914 |
| 8 | 1.307 | 1.326 | 1.287 | 1.104 | 0.904 | 0.871 |
| 12 | 1.057 | 1.212 | 1.115 | 1.102 | 0.730 | 0.316 |
| 16 | 1.167 | 1.290 | 1.055 | 0.966 | 0.664 | 0.285 |
| 24 | 0.700 | 0.984 | 1.041 | 0.863 | 0.352 | 0.196 |

Successive relative changes are 6-64% and non-monotone: this is not a Cauchy sequence. (Part of
it is the circulation itself moving, which is why the next test fixes Gamma.)

## 2. Same test with the circulation prescribed

Gamma(y) = 0.60*sqrt(1-(y/half)^2) held fixed for every mesh, the algebraic TE jump imposed
exactly, wall solved by least squares. Then the TE cut, the wake strength and the Kutta rule are
all removed as variables.

| c | 0.02 | 0.10 | 0.30 | 0.60 | 0.90 | 0.98 | wall residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1.721 | 1.721 | 1.528 | 1.486 | 0.906 | 1.326 | 2.14e-03 |
| 6 | 1.875 | 1.572 | 1.509 | 1.417 | 1.539 | 1.401 | 1.53e-03 |
| 8 | 1.752 | 1.586 | 1.466 | 1.403 | 1.259 | 1.104 | 1.30e-03 |
| 12 | 1.588 | 1.440 | 1.338 | 1.136 | 1.003 | 1.168 | 1.05e-03 |
| 16 | 1.119 | 1.197 | 1.309 | 1.048 | 0.937 | 0.950 | 9.32e-04 |
| 24 | 1.255 | 1.223 | 1.199 | 1.017 | 0.861 | 0.649 | 8.22e-04 |

Still no convergence (up to 30% jumps), and the Dirichlet residual only crawls from 2.1e-3 to
8.2e-4.

## 3. Bisect the defect: no cut / cut with Gamma = 0 / cut with Gamma != 0

| stage | residual c=4,6,8,12,16,24 | 0.02 at c=4..24 | 0.98 at c=4..24 |
| --- | --- | --- | --- |
| no cut, no wake, no Gamma | 1.5e-3, 9.0e-4, 7.8e-4, 7.1e-4, 7.1e-4, 7.2e-4 | 1.02, 0.86, 0.62, 1.16, 0.80, 1.14 | 0.32, 0.07, 0.17, 0.47, 0.42, 0.45 |
| cut, Gamma = 0 | 1.6e-3, 9.4e-4, 8.0e-4, 7.2e-4, 7.1e-4, 7.2e-4 | 1.15, 1.00, 0.75, 1.19, 0.82, 1.14 | 0.43, 0.20, 0.54, 0.49, 0.43, 0.44 |
| cut, Gamma elliptical | 2.1e-3, 1.5e-3, 1.3e-3, 1.1e-3, 9.3e-4, 8.2e-4 | 1.72, 1.88, 1.75, 1.59, 1.12, 1.26 | 1.33, 1.40, 1.10, 1.17, 0.95, 0.65 |

The cut and the circulation are nearly irrelevant to the defect: **the closed, uncut, unloaded
wing already fails to converge**, with a residual floor of 7.1e-4 and pointwise speeds that
scatter by 40%. For contrast, the smooth sphere oracle in the same collocation scheme reaches
1.10e-04 at 5120 faces (observed order ~2) with surface speeds good to 0.2%.

## 4. Kutta estimators that avoid the trailing-edge-adjacent cell

Gamma is the only nonlinear unknown; for each estimator we re-solve the wall least squares with
the TE jump exact and compare CL and sum(Gamma) across the chordwise mesh.

| c | baseline CL / sum(Gamma) | skip 1 cell | skip 2 cells | chordwise component only |
| --- | --- | --- | --- | --- |
| 4 | +0.4573 / +1.737 | +0.4860 / +1.835 | -0.0069 / -0.104 | +0.0056 / -0.055 |
| 6 | +0.3886 / +1.519 | +0.2488 / +0.969 | +0.2620 / +1.000 | +0.2676 / +1.040 |
| 8 | -0.0231 / -0.549 | +0.1024 / +0.379 | -0.0214 / -0.516 | +0.2782 / +1.115 |
| 12 | -0.3486 / -1.813 | -0.1633 / -0.849 | +0.0324 / -0.034 | +0.0451 / +0.021 |
| 16 | -0.7227 / -3.363 | -0.5485 / -1.991 | -0.1937 / -0.833 | -0.1882 / -0.801 |

Every estimator diverges. Skipping the last cell shifts the constant (c=4..8 stay positive
longer) but does not produce convergence; skipping two cells and using only the chordwise
component are no better and are not even monotone. **H2 is rejected.**

Note the baseline column here comes from an independent driver (different start point and Kutta
weight 10) and differs from the production value at c=8 (+0.2445): the Kutta least-squares
problem has more than one near-stationary point, so a research driver can land on another one.

## 5. The production solver is not weight- or iteration-limited

| c | kut_weight 1 | 5 | 50 | max_nfev 80 | 400 | nfev |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | +0.244531 | +0.244532 | +0.244532 | +0.244532 | +0.244532 | 5 |
| 12 | -0.158043 | -0.158040 | -0.158040 | -0.158040 | -0.158040 | 25 |

Seven digits of agreement: the bad answer is a converged solution of a well-posed discrete
problem whose solution is wrong, not a solver failure. Tuning the weight or the iteration cap
cannot help, and neither should be used to 'rescue' the model.

## 6. What this pins down, and what to do next

1. The divergence lives in the **base P1 collocation on a surface with sharp edges**, before
   any trailing-edge or wake logic. The evidence is section 3: closed, uncut, unloaded, and the
   field still does not converge.
2. Consequences: the wall residual floor (~7e-4) is a discretisation floor; the pressure-Kutta
   residual built on those gradients is a ratio of two non-convergent numbers, which is why the
   circulation is set by discretisation noise (see `docs/ISSUE71_CHORDWISE_DIVERGENCE.md`).
3. Next test, in order: (a) repeat section 3 on a **rounded** leading/trailing edge and rounded
   tip - if the residual then falls at order ~2 the sharp edge is confirmed as the cause;
   (b) if confirmed, the options are edge grading, an edge-singular enrichment of the P1 space,
   or higher-order panels, and the acceptance test is section 1-3 repeated, not a CL that
   happens to look right; (c) only afterwards revisit the Kutta rule, which is measured here to
   be insensitive to weight, iteration cap and to which cell it is evaluated on.
4. Do not use 'closer to VSPAERO', 'larger CL' or 'smaller drag' as the selection criterion for
   any of the above.