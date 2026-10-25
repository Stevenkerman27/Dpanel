# Local acceptance of the half-native panel operator (PR #21)

Head `e4ab137164bfa74e1dceadfcc1157a0ec0cb887a`, branch
`feature/half-native-panel-operator`, worktree `runs/pr21_git`. Nothing in this
report changes production code, relaxes a tolerance, or claims physical
accuracy; `physical_accuracy_accepted` stays `false`.

## 1. Environment and test result

| item | value |
| --- | --- |
| Python | 3.11.17 |
| JAX | 0.10.2, CPU, `jax_enable_x64=True` (set by the test module) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| `pytest -q test/test_half_native_panel.py` | **8 passed, 18.21 s, exit 0** |
| skips / xfails | none |

The eight cases are: half matrix/RHS/state against the true full reference,
raw Cp/force/moment against the full pressure oracle, geometry and station VJP
against three central differences, streamed `Av`/`Av` transpose plus the dot
identity, preconditioned primal and transposed GMRES, rejection of sideslip /
off-plane reference / unclosed mesh, the right-half JSON contract, and the guard
that the streamed action never traces a full square AIC.

## 2. Half-native operator versus the full mirrored oracle

The full aircraft (40 wall, 8 wake, 3 gamma) is built **inside the test only**;
the production input is the right half (20 wall, 4 wake, 2 gamma, 22 unknowns).
Measured on this machine (`runs/pr21_diag.py`, `runs/pr21_diag.json`):

| quantity | max abs difference |
| --- | --- |
| `A_half` vs `A[R] @ E` | 1.11e-16 |
| `rhs_half` vs `rhs[R]` | 1.39e-17 |
| `mu` (20 wall doublets) | 2.15e-16 |
| `gamma` (2 stations) | 5.13e-16 |
| raw `Cp` on the half wall | 1.11e-15 |
| total force | 8.74e-16 (relative 8.47e-15) |
| total moment | 2.78e-16 (relative 5.06e-15) |

Streamed operator versus the dense half matrix: `matvec` 2.22e-16, transpose
`matvec` 2.22e-16, dot identity `w.Av = (A^T w).v` 4.72e-16. The streaming path
therefore reproduces the assembled system to rounding, without materialising it.

## 3. Preconditioned GMRES, primal and transpose

| run | iterations | matvec calls | true relative residual |
| --- | --- | --- | --- |
| primal, half Kutta-aware near-field PC | 13 | 15 | 5.50e-12 |
| transposed, same preconditioner | 13 | 15 | 5.91e-12 |

GMRES state versus the dense half solve: 8.16e-13 absolute (dense state max
0.1406, so about 5.8e-12 relative). Adjoint pairing `seed.res = adj.b` holds to
2.18e-13. Iteration counts match the numbers reported from the isolated JAX
0.9.0.1 environment (13 / 13), and the residuals match to two digits
(5.50e-12 / 5.91e-12 there and here).

## 4. Geometry VJP against central differences

Loss `CL + 0.37 CM_y + 0.01 sum(Fx)` differentiated with respect to a joint
vertex and circulation-station direction:

| step | finite difference | absolute error | relative error |
| --- | --- | --- | --- |
| 1e-3 | 0.002322608574662 | 5.38e-13 | 2.32e-10 |
| 1e-4 | 0.002322608571234 | 3.97e-12 | 1.71e-09 |
| 1e-5 | 0.002322608546462 | 2.87e-11 | 1.24e-08 |

The 1e-5 row is finite-difference cancellation, not a derivative defect. This
differentiates the **dense half LU** path; the host SciPy GMRES still has no
geometry `custom_vjp`, so it is not the differentiable route.

## 5. CLI: LU versus streamed GMRES (separate processes, peak RSS sampled)

`cases/panel_half/thick_wing_two_strip.json`, 22 unknowns, one core:

| mode | first call | hot call | peak RSS | CL | CD_pressure | CM_y | true residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lu` | 0.4802 s | 0.000231 s | 391.1 MB | 0.0832845943872 | 0.0094924275905 | -0.0444593952007 | 2.00e-16 |
| `gmres` (`--block 7 --near 6 --restart 40 --cycles 20 --tol 1e-10`) | 3.7245 s | 0.4612 s | 711.7 MB | 0.0832845943888 | 0.0094924275908 | -0.0444593952005 | 5.50e-12 |

Cross-mode agreement: CL 1.54e-12 absolute (1.85e-11 relative), CD 2.83e-13,
CM_y 2.39e-13, force 1.96e-16.

**Memory and speed must be read separately.** At 22 unknowns the streamed GMRES
needs *more* peak RSS than dense LU (711.7 MB vs 391.1 MB) and is about eight
times slower per hot solve, because the iteration runs on the host through
SciPy callbacks and re-enters the JAX kernel 15 times. This fixture is far too
small to show the O(block*N) working-set advantage; what it does establish is
that the streaming and transpose paths are correct, and that a half-native
assembly is possible without ever storing a left-side state or a full AIC.
Extrapolating a memory win from this fixture would be dishonest.

## 6. What this does not establish

- No true half-domain CAD mesh exists yet; issue #20 covers the OCC/Gmsh
  `y>=0` mesher, the discarded symmetry cap and shared seam ids. The old Gmsh
  fixtures remain `blocked` (see `docs/ISSUE18_INDEPENDENT_SYMMETRY_AUDIT.md`).
- No sideslip, no non-planar or free wake, no wing-body CAD union, no GPU.
- The host GMRES is not an autograd custom VJP; only the half dense LU provides
  exact geometry derivatives today.
- No production pressure accuracy claim: the Cp here is validated against a
  mirrored analytic oracle, not against wind-tunnel or VSPAERO data.

## 7. Reproduction

```
python -m pytest -q test/test_half_native_panel.py
python -m panel.run_half_native --surface cases/panel_half/thick_wing_two_strip.json --out runs/half_native_lu_v1 --mode lu
python -m panel.run_half_native --surface cases/panel_half/thick_wing_two_strip.json --out runs/half_native_gmres_v1 --mode gmres --block 7 --near 6 --tol 1e-10 --restart 40 --cycles 20
python runs/peak_wrap.py panel.run_half_native <peakOutDir> --surface cases/panel_half/thick_wing_two_strip.json --out <cliOutDir> --mode lu
```

The peak-RSS wrapper must use a directory different from the CLI `--out`,
because it creates its own directory and the CLI refuses an existing report
path.
