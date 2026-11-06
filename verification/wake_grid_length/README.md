# Fixed/free wake and length study

## Weak-form comparison, 2026-11-06

Windows `myml` Python 3.13 / NumPy 2.4.6 / Torch 2.13, CPU,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Full-span
NACA0012 rectangular wing, span 4c, alpha 4 degrees, pressure Kutta,
quarter-chord moment reference, target quadrature order 3. The 30x32
cosine-chord/uniform-span wall/Gamma grid passed the preceding two-step
3% joint-refinement screen at a fixed 5c wake; this is not an independently
refined wall, circulation, streamwise wake or quadrature study. The reference
for `CD_trefftz` below is a Gamma-only straight-sheet diagnostic, not a
far-field integration of the relaxed wake.

| Fixed length | Wake x panels | CL | CL Gamma | CD pressure | CD Trefftz | CM y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2c | 2 | 0.252226 | 0.255473 | 0.005840 | 0.004972 | 0.004505 |
| 5c | 2 | 0.262728 | 0.265795 | 0.005853 | 0.005381 | 0.003843 |
| 10c | 2 | 0.264818 | 0.267870 | 0.005828 | 0.005465 | 0.003787 |
| 20c | 2 | 0.265393 | 0.268444 | 0.005817 | 0.005489 | 0.003782 |

2c->5c changes CL 3.997%, CL Gamma 3.883%, CD pressure 0.217%,
CD Trefftz 7.598% and CM y 17.230%, relative to the later value.
All five changes are below 3% on both 5c->10c and 10c->20c.
At 5c and 10c, increasing streamwise wake panels from 2 to 8 changes
each displayed fixed coefficient by less than `1e-9` absolute. This is
expected only for this straight wake and strength representation; it is
not evidence that two segments suffice for a curved free wake.

On the same 5c grid, two streamwise wake panels give initial fixed
CL `0.262728` and after four free updates `0.262921`. Extending the run
ends at update 13 with `blocked_geometry`, CL `0.262931`, normal-slip
RMS `0.014740`, and raw map residual `0.172764c`. The orientation
backtracking factor shrinks from 1 to `5.96e-8`, then no admissible step
is found. A nearly stationary CL therefore does not establish wake
equilibrium. With eight streamwise panels, after eight updates CL is
`0.262603`, slip RMS `0.017649`, preceding raw map residual `0.134355c`,
and backtracking factor `6.10e-5`; status is `iteration_limit`, not
converged. Both runs keep `physical_accuracy_accepted=false`.

At 10c with eight streamwise panels and four updates, the fixed/free CL
is `0.264818/0.264814`; the last slip RMS is `0.018979`, and the prior
raw map residual and backtracking factor are `0.473872c` and `0.001953`.
At the same update count, 5c free CL is `0.262600`. Both are
`iteration_limit`; their difference is an iteration diagnostic, not a
converged free-wake length effect.

For the coarse 4x4 symmetric wing at zero incidence, order-3 quadrature
gives CL `0.000859`; order 5 gives `0.000059`. This is small for the
current study and does not block it, but documents quadrature sensitivity.

Data: `weak_fixed_free_length_30x32.json`,
`weak_free_iterations_30x32.json`, `weak_free_streamwise8_30x32.json`,
`weak_free_length10_streamwise8_30x32.json`.
These outputs record iteration diagnostics and coefficients, not per-node
final wake snapshots. Timings are not benchmarks because other tests ran
concurrently.

```powershell
python -m tools.wake_fixed_free_length_study --out <fresh.json> --chord-panels 30 --span-panels 32 --wake-panels 2 --lengths 2,5,10,20 --free-length 5 --iterations 4 --order 3
python -m tools.wake_fixed_free_length_study --out <fresh.json> --chord-panels 30 --span-panels 32 --wake-panels 2 --lengths 5 --free-length 5 --iterations 16 --order 3
python -m tools.wake_fixed_free_length_study --out <fresh.json> --chord-panels 30 --span-panels 32 --wake-panels 8 --lengths 5,10 --free-length 5 --iterations 8 --order 3
python -m tools.wake_fixed_free_length_study --out <fresh.json> --chord-panels 30 --span-panels 32 --wake-panels 8 --lengths 10 --free-length 10 --iterations 4 --order 3
```

## Earlier collocation grid study

Update 2026-11-06: the collocation result below remains a negative result.
The restored optional weak-wall path subsequently passed the specified 3%
joint-refinement screen on the same test family; see
`verification/wake_galerkin_revival/README.md` for the new runs and their
limitations. The subsequent fixed/free and wake-length comparisons are recorded above.

2026-11-06. Windows / Python 3.13 / NumPy 2.4.6 / Torch 2.13, single-thread
OpenBLAS, OMP and MKL. These are P1 `combine_wings` / `solve_multi` results,
not the archived P0 grid study. Physical accuracy is not accepted.

## Protocol

Full-span NACA0012 rectangular wing, span 4c, alpha 4 degrees, moment
reference at x/c=0.25, pressure Kutta, straight wake of length 5c with two
streamwise panels. `make_wing` uses cosine chordwise spacing (clustering at
both leading and trailing edges) and uniform spanwise spacing; this is not a
curvature-adaptive CAD mesh. The wall and wake share span stations, so varying
span panels simultaneously varies wall resolution, Gamma degrees of freedom,
and wake faces. Chord-only scans hold span/Gamma/wake fixed.

The requested preliminary screen is two consecutive refinements for which
each of pressure CL, circulation CL, pressure CD, Trefftz CD and pitching CM
changes by at most 3% relative to the finer value. For coefficients where
both values have magnitude below 0.001, use an absolute difference of 0.00003
instead; report near-zero differences rather than unstable percentages. This is
an operational screen, not proof of discretization accuracy or physical
validation. The screen fails even for CL alone, so no choice of the near-zero
threshold changes the conclusion. A valid comparison would additionally need
an accepted inner solution and a converged, geometrically valid free wake.

## Joint refinement

| chord x span | wall faces | Gamma DOF | wake faces | CL | CL Gamma | CD pressure | CD Trefftz | CM y | wall/floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 x 6 | 112 | 5 | 24 | 0.256144 | 0.225644 | -0.019765 | 0.003548 | 0.024582 | 1.039 |
| 6 x 8 | 216 | 7 | 32 | 0.198864 | 0.186726 | -0.011243 | 0.002731 | 0.035475 | 1.088 |
| 8 x 10 | 352 | 9 | 40 | 0.115687 | 0.113123 | -0.009172 | 0.001503 | 0.059589 | 1.119 |
| 10 x 12 | 520 | 11 | 48 | 0.022751 | 0.021266 | -0.013045 | 0.001025 | 0.090068 | 1.129 |
| 12 x 14 | 720 | 13 | 56 | -0.072507 | -0.068340 | -0.017336 | 0.001723 | 0.123544 | 1.128 |
| 14 x 16 | 952 | 15 | 64 | -0.148308 | -0.146299 | -0.017906 | 0.003280 | 0.148712 | 1.114 |

None of the five adjacent CL changes passes 3%. The nonlinear solver reports
termination at every level, but the independently computed wall/floor
criterion of 1.05 fails at all but the coarsest. Small pressure Kutta residual
and a solver success flag do not establish physical acceptance.

## Independent scans

With span fixed at 8, chord panels 4/6/8/10/12/14 produce CL
0.260093/0.198864/0.092451/-0.043110/-0.202085/-0.325661.
With chord fixed at 6, span panels 4/6/8/10/12/14 produce CL
0.167677/0.184041/0.198864/0.209123/0.217640/0.224433. The former
isolates a chordwise sensitivity while the latter jointly varies span and
Gamma. Neither series establishes convergence.

For the chord scan, the condition number of the area-weighted wall matrix
restricted to the null space of the jump constraint is 367/614/841/1024/
1159/1255. With jump and signed-speed constraints both imposed it is
83/146/222/305/394/488. These are linear constrained-wall diagnostics, not
the nonlinear pressure-Kutta Jacobian condition number; growth alone does
not prove the cause of the sign change.

The sign flip resembles the archived P1 chordwise observation in
`docs/archive/issue/ISSUE71_CHORDWISE_DIVERGENCE.md`, but the older study
uses different camber, alpha, software environment and code version. The
underlying cause has not been independently established here.

## Outcome and Reproduction

In this earlier collocation study, the prerequisite did not pass, so **no grid was selected as a 3%-converged
baseline** and the requested fixed/free and wake-length comparisons were not
run. In particular, 5c is just the initial setting, not an accepted length.
Investigate the fixed-wake chordwise behavior first, then repeat the grid
screen before starting those comparisons.

In PowerShell, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` to `1`; use `C:/Users/zyx20/anaconda3/envs/myml/python.exe`:

```powershell
python -m tools.wake_grid_length_study --wall-method collocation --mode grid --out <fresh-grid.json>
python -m tools.wake_grid_length_study --wall-method collocation --mode grid --levels 4x8,6x8,8x8,10x8,12x8,14x8 --condition --out <fresh-chord.json>
python -m tools.wake_grid_length_study --wall-method collocation --mode grid --levels 6x4,6x6,6x8,6x10,6x12,6x14 --out <fresh-span.json>
```

`grid.json`, `chord.json`, `chord_condition.json` and `span.json` retain the
original measurements; `<fresh-...>` must be nonexistent files. A second
conditioned chord run reproduced all printed coefficients exactly. Input
geometry is generated by the command, without external mesh files.
