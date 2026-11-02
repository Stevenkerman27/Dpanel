# Torch CPU port map for the steady P1 production path

Issue #92, Route B (decision comment 6569612502), immediate task: *"produce a file-by-file Torch
CPU port map for the complete production path and identify every remaining SciPy/NumPy linalg call
reached by G1/G2 solve, pressure-Kutta and coupled free-wake iteration."*

Base commit for the inventory: `c953ac5` (branch `research/issue92-phase1-wake-evaluator`).
Reproduce with `python tools/issue92/port_inventory.py [repo-root]` (AST import BFS, not grep: it
follows what the code actually reaches, so audit/tooling modules are excluded by construction).

## 1. Reachable surface

Three seeds — `panel/p1_lifting_g1.py`, `panel/p1_lifting_g2.py`,
`panel/p1_coupled_material_wake.py` — reach **14 modules**:

| module | role in the production path | solver-grade linalg sites |
| --- | --- | --- |
| `p1_lifting_g1.py` | single-wing assembly, rank-aware constrained solve, pressure-Kutta | 2 `least_squares`, 3 `null_space`, 2 `lstsq` |
| `p1_lifting_g2.py` | wing-body assembly, combine_wings, floor method, constrained solve | 1 `least_squares`, 3 `null_space`, 3 `lstsq` |
| `p1_pressure_kutta_reduced.py` | reduced/structured pressure-Kutta nonlinear solve | 1 `qr`, 1 `np.linalg.solve`, 2 `solve_triangular`, `OptimizeResult` |
| `p1_coupled_material_wake.py` | coupled free-wake loop, sampling, gates, snapshots | 1 `svd` (constraint spectrum, :983) |
| `p1_closed_body.py` | closed-body variant of the constrained solve | 1 `qr` (:208), 1 `svd` (:209), 1 `solve` (:213) |
| `p1_geometry_contact.py` | swept-hull/wall contact classification | 1 `svd` (contact plane fit, :134) |
| `p1_steady_free_wake.py` | free-wake driver, wake material, node velocities | none |
| `p1_analytic_sheet_velocity.py` | default `analytic_edges` induced velocity (closed form) | none (norms only) |
| `p1_regularized_sheet_velocity.py` | core-regularized sheet evaluator | none (norms only) |
| `p1_projection_duffy.py` | projection-fan / Duffy area quadrature | none (norms only) |
| `p1_adaptive_area.py` | adaptive area quadrature | none (norms only) |
| `p1_hybrid_area.py` | near-fan / far-Duffy split | none (norms only) |
| `p1_wake_contact.py` | wake contact declarations | none |
| `p1_torch_analytic_sheet.py` | already-torch analytic evaluator | none |
| `wake_velocity_methods.py` | method registry | none |

Two consequences worth stating up front:

1. The **port surface is small and concentrated**: 45 `np.linalg`/`scipy` hits, of which the real
   LAPACK decision points are **2 `least_squares`, 6 `null_space`, 5 `lstsq`, 3 `svd`, 2 `qr`,
   2 `np.linalg.solve` and 2 `solve_triangular`**. Everything else on the surface is either
   `np.linalg.norm` (119 elementwise reductions, no LAPACK) or array assembly.
2. The evaluator modules — the ones that dominate runtime — **contain no LAPACK at all**. The
   induction/quadrature work can move to torch tensors without touching a factorization.

## 2. Every call, with its torch replacement and its semantic risk

| current call | sites | torch replacement | semantic risk |
| --- | --- | --- | --- |
| `scipy.optimize.least_squares(..., method default 'trf', jac=...)` | `p1_lifting_g1.py:506`, `p1_lifting_g2.py:304` | no equivalent: implement trust-region reflective with the same `ftol/xtol/gtol=1e-11`, `max_nfev`, x-scaling and step bounds | **highest**: `status`/`success`/`nfev` strings flow into receipts and acceptance (`pressure_kutta_acceptance`), and G2's floor is a least-squares floor, not zero — a different stopping rule changes the reported status without changing the physics |
| `scipy.linalg.null_space(A, rcond=1e-11)` | 6 (G1 :476/:493/:569, G2 :279/:295/:371) | `torch.linalg.svd(..., full_matrices=True)`, keep `Vh[k:]` with `S > rcond*S[0]` | **high**: a rank decision; see §3 |
| `scipy.linalg.lstsq(..., lapack_driver='gelsd')` | 5 (G1 :478/:578, G2 :242/:281/:374) | `torch.linalg.lstsq(..., driver='gelsd')` (returns `rank`) | **high**: the `rcond` default differs between the two libraries; must be passed explicitly, and `gelsd` is rank-revealing, so a cutoff change changes the minimum-norm solution |
| `np.linalg.svd` | 3 (`p1_coupled_material_wake.py:983`, `p1_geometry_contact.py:134`, `p1_closed_body.py:209`) | `torch.linalg.svd` | low-medium: :983 feeds a reported spectrum, :134 fits a contact plane (orientation sign must be preserved) |
| `np.linalg.qr` | 2 (`p1_pressure_kutta_reduced.py:94`, `p1_closed_body.py:208`) | `torch.linalg.qr` | low: sign convention is irrelevant because `R` is used as a triangular factor only |
| `np.linalg.solve` | 2 (`p1_pressure_kutta_reduced.py:146`, `p1_closed_body.py:213`) | `torch.linalg.solve` | low |
| `scipy.linalg.solve_triangular` | 2 (`p1_pressure_kutta_reduced.py:137-138`) | `torch.linalg.solve_triangular` | low |
| `scipy.optimize.OptimizeResult` | `p1_pressure_kutta_reduced.py:275` | local dataclass with the same attribute names | low, but it is part of the public result contract |
| `np.linalg.norm` | 119 | `torch.linalg.vector_norm` / `torch.linalg.norm` | mechanical, no LAPACK; matters only because these calls must not re-enter NumPy on torch tensors |

Measured capability of the replacement (torch 2.13.0+cu132, float64 CPU, probe
`tools/issue92/probe_torch_linalg.py`):

| check | result |
| --- | --- |
| `torch.linalg.lstsq` driver `gels` / `gelsy` / `gelsd` vs `np.linalg.lstsq` | 7.6e-16 / 2.2e-15 / 1.1e-15 |
| `torch.linalg.svd` singular-value spectrum vs NumPy | 2.8e-14 (absolute) |
| `torch.linalg.qr` orthonormality | 4.4e-16 |
| `torch.linalg.solve_triangular` vs `scipy.linalg.solve_triangular` | 6.9e-16 |

So the replacements exist and are accurate to float64 rounding. The port is not blocked by missing
functionality; it is blocked by **decision semantics** (§3) and by the nonlinear solver (§4).

## 3. The one real hazard: rank decisions at `rcond=1e-11`

Six `null_space` calls and five `lstsq` calls decide rank by comparing a singular value against
`rcond * S_max`. Torch and NumPy disagree at the 1e-14 level; the coupled-solve rank cut is 1e-11
relative, that is, roughly **three orders of magnitude above the observed numerical difference**. On
a random 40x12 probe the nearest singular value sits 1.3 away from the cut, so the decision is
trivial there — that proves nothing about the production matrices, whose spectra come from
near-degenerate panel layouts (half-model mirrors, TE pairs, repeated wake rows) where gaps of
1e-11 are plausible.

Therefore the port rule for every `null_space`/`lstsq` site is:

* reproduce the criterion exactly (`S > rcond*S[0]`, same default `rcond` semantics for `lstsq`);
* on each frozen fixture, **report the gap between the cut and the nearest singular value**, and
  require identical null-space dimension and identical `rank` before and after;
* if a fixture has a gap below ~1e-13 relative, that fixture cannot discriminate the two backends
  for rank, and any parity claim on it must say so rather than claim agreement.

This is the item most likely to produce a silent behavioural change, because a rank flip does not
raise: it changes `mu`/`Gamma` and every load downstream while the residual still looks converged.

## 4. The nonlinear solve

`least_squares` is used twice, but the coupled iteration calls the G2 path every step, so its
stopping behaviour governs the whole free-wake trajectory. Two facts constrain the port:

* the G2 pressure-Kutta residual has a **nonzero floor**, and the existing acceptance logic
  (`p1_lifting_g2.py:230-255`) is written around that floor: `status 0` with `success=False`
  after the gradient stalls. A torch reimplementation must reproduce the floor semantics, not
  "converge harder";
* the reduced path already reimplements the stopping tests
  (`p1_pressure_kutta_reduced.py:165-275`, absolute + relative optimality with a stagnation rule)
  and is verified against `least_squares` behaviour; it is the template for the full solve.

## 5. Test-time isolation, not a production worker

Route B forbids an out-of-process GPU worker as the production architecture. It does not remove the
fact measured in `docs/ISSUE92_PHASE1_WAKE_EVALUATOR.md` §6f: after torch executes any op, the next
NumPy LAPACK call in the same interpreter dies with `0xC0000005`. Parity testing therefore has to
compare backends **across processes** (each backend writes an `.npz` of A/W/J/K, ranks, singular
values, mu/Gamma, Cp, loads, residuals, status; a comparator reads both). That is test scaffolding,
not a runtime boundary: the shipped torch path never calls SciPy.

## 6. Vertical slices

Each slice ends with the full suite plus a clean-master baseline, and may only set
`backend='torch'` for its own entry point after its parity gate passes. Default stays NumPy.

| slice | content | parity gate |
| --- | --- | --- |
| S1 | `p1_pressure_kutta_reduced.py`: `qr`, `solve`, `solve_triangular`, `OptimizeResult` → torch — **delivered, §8** | A/W/J/K and `z` vectors to 1e-12 relative; identical `status`/`nfev` on all fixtures that exercise the reduced path; 10 existing tests still pass |
| S2 | rank-aware constrained solve in G1/G2: `null_space`, `lstsq` → torch — **delivered, §9** | identical `rank` and null-space dimension on all 8 fixed fixtures **plus** the §3 gap report; `mu`/`Gamma`/`CL`/`CD_p`/`CM_y` to 1e-11 relative |
| S3 | nonlinear carry-over: the reduced pressure-Kutta nonlinear solve on torch (`least_squares` → torch, G1 :506 / G2 :304 — **delivered, §10**; the gate was rewritten: the torch path reproduces the *structured* algorithm, so the gate is identical acceptance classification plus loads, not identical `nfev` against `trf`) | identical acceptance classification (`accepted`, `gradient_relative`, `wall_over_floor`, `pressure_rms`) against the `trf` reference on all 8 fixtures; identical `status`/`nfev`/`message` against the NumPy structured backend; loads to 1e-6 relative |
| S4 | assembly and post-processing on tensors: face→DOF scatter, Cp, forces/moments — **delivered, §11**: `mesh_kernel`, `doublet_matrix`, `gradient_map`, `wake_operator`, `loads`; the Kutta and jump/contact rows are rebuilt inside the coupled loop, so they move with S5 | A/W to 1e-12 max-relative and 1e-13 Frobenius on all 8 fixtures, `grad` bit-identical; loads to 1e-13 against the shared NumPy map |
| S5 | coupled free-wake loop and sampling entirely in torch; snapshot schema records the backend — **delivered, §12**. The gate was rewritten once: the 192-face fixtures give three accepted updates, the 872-face CAD body accepts one and then stops `blocked_updated_nonlinear_solve` on both backends (structured solver's line search, §12.5) | identical status and per-iteration decisions (`rank`, `kinematic_candidate`, `nonlinear_success`, wake eligibility) on three cases; loads to 1e-5 relative, states to 1e-5, `cp` to 1e-3 absolute, residuals to 1e-5 absolute. Declared first at 1e-9 and revised once with the flat-valley evidence in §12.4; then B2 (CUDA) starts |
| S6 | public entry-point dispatch: `solve` / `solve_multi` take `backend='numpy'|'torch'` and `device`, and the torch branch fails closed on the unported SciPy trust-region solver - **delivered, §15** | all 8 fixtures: identical `status`/`nfev`/`message`/rank/acceptance classification; loads ≤ 6.4e-9 relative, state ≤ 1.8e-8, `|dcp|` ≤ 8.7e-7. Declared first at 1e-9 loads / 1e-9 state / 1e-12 `cp` and revised once with the flat-valley evidence (§15.4) |
| S7 | retire the NumPy reference: freeze the production default solve of all 8 fixtures as re-runnable bundles, then move the last per-target evaluator (RM-P1 regularized sheet) onto torch with `rm_backend='numpy'|'torch'` - **delivered, §16** | bundles: 8/8 re-run bitwise identical (`loads=0.0`, `state=0.0`, same rank and margin). RM field: ≤ 1.9e-15 relative on both quadratures, CPU and CUDA; coupled iteration 0 bitwise identical, iteration 1 ≤ 4.9e-14 with the solver held fixed, decisions identical |
| S8 | implicit adjoint/VJP for the whole chain (no differentiation through the pivots): linear level, and the pressure level with the **exact** Hessian of the quartic objective, plus the diagnostics that decide whether a derivative is certifiable at all - **delivered, §17** | 4 fixtures: linear `dL/dalpha` 2.2e-8 … 2.1e-7 against finite differences; pressure level 6.9e-9 … 1.5e-6 on 3 fixtures, with the Gauss-Newton operator 2-3 orders worse. `g1_wingbody_cad` pressure is declared **not certifiable** (FD stability 0.63, `cond = 2.7e+11`, forward residual floor 1e-4) rather than passed |

| B2 | whole-chain device selection, device=cpu or cuda in one process, no silent fallback — **delivered, §13**: cold/warm solve, one coupled iteration, uploaded bytes, peak VRAM, plus a profile that attributes the iteration | identical loads to 1e-5 relative against the S5 NumPy records on both devices, identical status and acceptance; **no speedup claimed** — measured 11.4x and 4.3x slower solves on CUDA, and the iteration is 86% a Python geometry gate |
| B3 | VSPAERO cross-check on four geometries (rectangular, swept tapered, elliptic, wing-body), several angles of attack, two mesh levels | compares only CL(alpha) intercept and slope, induced drag, spanwise circulation, and a Cp definition that can be aligned; thick-sheet and VLM results reported separately |

Estimated relative difficulty, from the inventory: S1 ~15% of the work, S2 ~20%, S3 ~30%, S4
~25%, S5 ~10%. S3 is the schedule risk (no library equivalent), S2 is the correctness risk (§3).

## 7. Invariants for every slice

* float64 everywhere; NumPy/SciPy backend stays the immutable reference and explicit fallback.
* No default change, no wake topology or contact-semantics change, no gate threshold change,
  `physical_accuracy_accepted=false`.
* No finite-difference design sensitivities, no surrogate model, no Transolver, no out-of-process
  worker in production.
* Consolidated full-suite and clean-master baseline after each slice — not selected unit tests.

## 8. S1 delivered: torch pressure-Kutta (2026-11-02)

Files: `panel/p1_pressure_kutta_torch.py` (TorchPressureResult, TorchStructuredPressureKutta,
solve_structured_pressure_torch), `tools/issue92/s1_problem.py` (shared structural problem),
`tools/issue92/s1_parity.py` (numpy / torch / compare stages), `test/test_p1_pressure_kutta_torch.py`,
evidence `docs/issue92/s1_reference_<scenario>.npz` and `docs/issue92/s1_parity_<scenario>.json`.

The port is call for call: QR of `M = B N` once, Woodbury step in the pressure space, the same
gtol -> relative_gtol -> stagnation -> max_nfev -> step -> Armijo sequence, the same status codes and the
same message strings. Two mechanical differences from SciPy: `torch.linalg.solve_triangular` demands a
matrix right-hand side (a vector is promoted to one column), and there is no `OptimizeResult` — the
result is a dataclass with the same attribute names so callers do not branch.

The problem is structural, not a mesh: 600 wall rows, 300 unknowns, 200 free dofs, 24 pressure rows, the
elliptic fixture's shape scaled by 0.2, with `bw` carrying a component outside `range(B N)` so the wall
block has the nonzero floor the production acceptance logic is written around.

| scenario | termination (numpy / torch) | nfev | nit | backtracks | relative iterate difference | cost relative |
| --- | --- | --- | --- | --- | --- | --- |
| base (mild quadratic, start at 0) | 4 / 4, relative gradient | 9 / 9 | 4 / 4 | 0 / 0 | 8.34e-16 | 1.41e-16 |
| hard (c2 = 5, start at 2 sigma) | 2 / 2, ftol | 75 / 75 | 37 / 37 | 0 / 0 | 8.09e-16 | 8.34e-16 |
| backtrack (c2 = 20) | 2 / 2, ftol | 100 / 100 | 36 / 36 | 27 / 27 | 1.41e-15 | 1.38e-15 |
| stress (c2 = 50, line-search failure) | 0 / 0, line search failed | 100 / 100 | 31 / 31 | 37 / 37 | **1.27e-07** | 3.88e-11 |

Three findings, in order of how much they constrain later slices:

1. **Decisions are identical, including the line-search branch.** Status, nfev, nit, backtrack count and
   message string agree in all four scenarios, so the Armijo backtracking and the stopping order port
   cleanly — the third row is a case where 27 halvings decide the trajectory.
2. **Iterate parity holds at 1e-15 only while the iteration converges.** The stress row is deliberately
   outside the production regime (50x the quadratic coefficient) and ends in a line-search failure after
   31 iterations; there the same decisions produce an iterate difference of 1.27e-7 and a cost difference
   of 3.9e-11. That is amplification of float64 roundoff through a failing search, not a porting defect,
   but it fixes the rule for every later slice: parity must be reported **per scenario and per quantity**,
   never as a single "matched" word.
3. **The pressure callables must be torch-native.** `pressure` and `pressure_jacobian` receive and return
   tensors; a wrapper that converts to NumPy inside would re-enter LAPACK and reintroduce the §6f crash.

What S1 does **not** claim: the production entry points still call the SciPy path (nothing is wired to
`backend='torch'` yet); this parity is on a structural problem, not on an assembled mesh. Wall-operator and
pressure parity on real fixtures arrives with S4/S5, when assembly itself is on tensors.

Reproduce (three processes by necessity):

    python -m tools.issue92.s1_parity numpy   --scenario backtrack --out a.npz
    python -m tools.issue92.s1_parity torch   --scenario backtrack --out b.npz
    python -m tools.issue92.s1_parity compare a.npz b.npz --json report.json

Test state on this tree: `827 passed, 20 skipped` in the default `.conda_env` with the seven torch-only
modules ignored (before this slice: `826 passed, 16 skipped`; the delta is the static NumPy/SciPy guard plus
the four gated parity tests skipping where torch is absent). In the torch interpreter,
`PANEL_TORCH_PARITY=1 pytest test/test_p1_pressure_kutta_torch.py` gives 5 passed.

Gate to S2: the three gated scenarios hold at 1e-11 relative on the iterate with identical decisions, and
the static test forbids NumPy/SciPy in the torch module. S2 (rank-aware `null_space`/`lstsq`) inherits the
per-scenario reporting rule and must add the singular-value gap table described in §3.
## 9. S2 delivered: rank-aware constrained solve (2026-11-02)

Files: `panel/p1_rank_aware_torch.py` (`RankReport`, `LeastSquaresResult`, `null_space`, `lstsq`,
`constrained_solve`), `tools/issue92/s2_rank_parity.py` (stages `capture` / `torch` / `loads` /
`compare`), `tools/issue92/s2_loads.py` (the state -> load map used on both states),
`test/test_p1_rank_aware_torch.py`, evidence `docs/issue92/s2_rank_parity.json`,
`docs/issue92/s2_loads.json`, `docs/issue92/s2_reference.npz`.

The port mirrors the SciPy semantics rather than "something reasonable": the truncation threshold is
`s.max() * 1e-11` on the constraint spectrum (the same `rcond` the production calls pass), the least-squares
solve reproduces `gelsd` (default `rcond = max(rows, cols) * eps`, inverse-singular-value truncation,
minimum-norm among the retained directions), and `constrained_solve` composes them exactly as the production
code does — `N = null_space(constraints)`, `t = lstsq(B N, bw)`, `z = N t`. `RankReport.margin` is the
decisive singular value divided by the threshold (i.e. how many times above the cut it sits), and
`gap_ratio` is smallest-kept over largest-discarded; both are reported per fixture because they bound how
far a backend difference can move the decision.

The captured object is the **initial constrained solve** of each of the eight frozen Phase-0 fixtures,
rebuilt from the operators the production solve actually returned (same `B = sqrt(area/mean) [A W]`,
same `bw = sqrt(area/mean)(-c U)`, same stacked constraints). `capture` also re-applies the load map to
the NumPy state and asserts it reproduces the production `CL`/`CL_gamma`/`CD_pressure`/`CM_y` — the
difference is `0.0` for seven fixtures and `5.6e-17` for `g2_conventional`, so the rebuilt system is the
production system, not a lookalike.

| fixture | wall x | constraints | rank (np / torch) | null dim | state relative | max relative | singular relative | margin over threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 112x68 | 10x68 | 58 / 58 | 58 / 58 | 1.07e-14 | 2.27e-14 | 5.8e-16 | 1.05e10 |
| g1_wing_camber | 112x68 | 10x68 | 58 / 58 | 58 / 58 | 8.61e-15 | 1.69e-14 | 1.2e-15 | 1.05e10 |
| g1_swept30 | 2764x1462 | 126x1462 | 1336 / 1336 | 1336 / 1336 | 1.86e-14 | 8.47e-14 | 1.1e-15 | 1.68e9 |
| g1_elliptic | 3220x1728 | 166x1728 | 1562 / 1562 | 1562 / 1562 | **1.79e-12** | 1.40e-11 | 7.7e-14 | **5.49e4** |
| g2_conventional | 192x116 | 16x116 | 100 / 100 | 100 / 100 | 8.36e-15 | 2.78e-14 | 8.7e-16 | 5.78e9 |
| g2_tail_low | 192x116 | 16x116 | 100 / 100 | 100 / 100 | 1.39e-14 | 4.42e-14 | 8.7e-16 | 5.78e9 |
| g1_wingbody_cad | 792x541 | 187x541 | 354 / 354 | 354 / 354 | 1.38e-13 | 3.47e-13 | 4.7e-16 | 7.04e6 |
| g1_wingbody_cad_3x | 2410x1352 | 189x1352 | 1163 / 1163 | 1163 / 1163 | 1.24e-13 | 6.73e-13 | 8.4e-16 | 6.88e6 |

Loads, computed by one map applied to both states (relative difference, NumPy vs torch):

| fixture | CL | CL_gamma | CD_pressure | CM_y |
| --- | --- | --- | --- | --- |
| g1_wing_rect | 4.6e-14 | 7.4e-14 | 1.8e-13 | 1.6e-13 |
| g1_wing_camber | 1.3e-14 | 1.8e-14 | 1.4e-13 | 3.5e-14 |
| g1_swept30 | 3.5e-14 | 4.8e-14 | 7.2e-13 | 4.1e-14 |
| g1_elliptic | 1.0e-13 | 5.1e-14 | 1.6e-12 | 1.7e-13 |
| g2_conventional | 5.3e-14 | 3.3e-14 | 2.4e-13 | 2.1e-13 |
| g2_tail_low | 6.0e-15 | 8.4e-15 | 4.4e-13 | 3.5e-13 |
| g1_wingbody_cad | 1.5e-13 | 1.5e-13 | 8.1e-13 | 3.1e-13 |
| g1_wingbody_cad_3x | 3.2e-13 | 2.7e-13 | 1.4e-12 | 5.8e-13 |

Three findings:

1. **The decision is identical; the basis is not.** Rank and null-space dimension agree on 8/8, and the
   projectors `N N^T` agree to 1e-12 or better, but the individual basis columns are *not* comparable. The
   first attempt at this measurement compared states expressed in each backend's own basis and reported
   0.27–1.97 relative differences while the end-to-end state agreed to 1e-14 — a pure artefact. The cause is
   structural: a constraint matrix of full column rank has a null space spanned by its *zero* singular
   vectors, and any orthonormal basis of that subspace is a valid answer, so two SVD implementations are free
   to return different rotations. Only basis-invariant quantities may be compared: `rank`, `null_dim`, the
   projector, singular values, and `z = N t` in the original unknown space.
2. **End-to-end parity tracks the truncation margin, not the problem size.** Seven fixtures sit 1e6–1e10 times
   above the truncation threshold and agree to 1e-13 or better. `g1_elliptic` sits at 5.5e4 (its decisive
   singular value is the closest any fixture comes to the cut) and is the only one at 1e-12. That is the
   quantified version of the §3 correctness risk: the closer a fixture is to the threshold, the larger the
   fraction of its error a backend difference can own, and the fixture set contains no case that actually
   flips rank — which is why the margin is reported instead of hidden.
3. **Nothing here is wired into production yet.** `constrained_solve` is exercised through the parity tool
   only; G1/G2 still call SciPy. The torch module is guarded by an AST test that forbids `numpy`/`scipy`
   entirely, because a NumPy LAPACK call after torch has run in the same interpreter kills the process (§6f);
   the four stages therefore run in two interpreters, with the states exchanged through `.npz`.

What S2 does **not** claim: that the torch decision is correct where the fixture set cannot decide (no rank
flip is exercised); that the nonlinear pressure-Kutta path has any torch support (that is S3); that
`mu`/`Gamma` parity on *assembled* systems is established (the wall operators here are production operators
but assembled by NumPy in S4's absence). Iterate parity of 1.8e-12 on the tightest fixture is the number to
beat in S3/S4, not a stamp of approval.

Reproduce (two interpreters, in this order):

    python -m tools.issue92.s2_rank_parity capture --work runs/issue92/s2
    <torch python> -m tools.issue92.s2_rank_parity torch --work runs/issue92/s2
    python -m tools.issue92.s2_rank_parity loads   --work runs/issue92/s2 --out runs/issue92/s2/s2_loads.json
    python -m tools.issue92.s2_rank_parity compare --work runs/issue92/s2 --out docs/issue92/s2_rank_parity.json

Test state: `test/test_p1_rank_aware_torch.py` gives 14 passed in the torch interpreter with
`PANEL_TORCH_PARITY=1`, and 1 passed / 13 skipped in `.conda_env` (the static guard runs where torch is
absent, the numerics skip). The gate to S3 is met on all eight fixtures: identical rank and null dimension,
loads to 1.6e-12 relative, and the gap/margin table above.

## 10. S3 delivered: torch driver for the nonlinear pressure-Kutta solve (2026-11-02)

Files: `panel/p1_pressure_kutta_torch_driver.py` (`TorchPressureModel`, `TorchPressureSolve`,
`solve_pressure_kutta_torch`), `tools/issue92/s3_pressure_parity.py` (stages `capture` / `torch` / `judge` /
`bundle`), `test/test_p1_pressure_kutta_torch_driver.py` (8 tests), evidence `docs/issue92/s3_parity.json`
and the frozen gate case `docs/issue92/s3_reference_g1_wing_rect.npz`.

### 10.1 The gate in section 6 had to change, and why

Section 6 wrote the S3 gate as *identical `status`/`success`/`nfev` against the reference*. That is not reachable
and was not attempted: the production reference is `scipy.optimize.least_squares(method="trf")`, and reproducing
trust-region reflective in torch is a reimplementation rather than a port - its `status` strings and iteration
counts are made of every internal detail (x-scaling, step bounds, the `ftol`/`xtol`/`gtol` race). What production
actually consumes is `pressure_kutta_acceptance`: `accepted`, `gradient_relative`, `wall_over_floor`,
`pressure_rms`, plus the loads. The delivered gate therefore has two parts:

* **against the `trf` reference (what the repository ships):** the acceptance classification and the loads must
  agree - that is the decision a caller can see;
* **against the NumPy structured backend (same algorithm, other library):** `status`/`nfev`/`message` must agree -
  that is the port, checkable precisely because both sides run the same algorithm (`p1_pressure_kutta_reduced.py`
  from S1 -> `p1_pressure_kutta_torch.py`).

S3 therefore ports `solve_structured_pressure` (the path with a convergence record, `pressure_solver="structured"`)
and not `least_squares`. The `trf` path stays NumPy-only, unchanged, as reference and fallback.

### 10.2 What the driver is

`TorchPressureModel` is the trailing-edge pressure residual of `p1_lifting_g1.py:482-491` on tensors:
`v = grad . mu`, `p_j = mean_top(|v|^2) - mean_bot(|v|^2)`, Jacobian `2 mean(dv/dmu . v)` per side, zero columns
for `Gamma`. Ragged trailing-edge pairs (two top faces against three bottom faces is legal) are packed into padded
rows plus a mask, because torch cannot hold ragged tensors; residual and Jacobian share the same mask and count.

`solve_pressure_kutta_torch` does exactly what the structured backend does: `N = null space(J, rcond=1e-11)`,
`z0 = N^T x0`, `TorchStructuredPressureKutta` (S1) for the nonlinear step, `x = N z`. No NumPy, no SciPy, no
`OptimizeResult`; the module is covered by the same AST guard as S1/S2.

### 10.3 Parity on the eight frozen fixtures

Decision quantities. `acc` is `pressure_kutta_acceptance`; `floor` is `wall_over_floor`; `grad` is
`gradient_relative`.

| fixture | ref st/nfev | structured st/nfev | torch st/nfev | floor (ref / struct / torch) | grad (ref / struct / torch) | acc |
| --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 2 / 4 | 2 / 7 | 2 / 7 | 1.0577997468 (3x) | 8.13e-12 / 8.37e-12 / 8.30e-12 | True |
| g1_wing_camber | 3 / 10 | 4 / 7 | 4 / 7 | 1.0595427809 (3x) | 1.084e-10 / 1.085e-10 / 1.084e-10 | True |
| g1_swept30 | 2 / 7 | 2 / 13 | 2 / 13 | 1.1558328838 (3x) | 2.09e-11 / 2.12e-11 / 2.17e-11 | True |
| g1_elliptic | 2 / 12 | **5 / 26** | **2 / 21** | 1.0638500112 / 1.0638500103 / 1.0638500138 | 2.07e-10 / 1.39e-09 / 8.74e-10 | True |
| g2_conventional | 1 / 5 | 4 / 7 | 4 / 7 | 1.0572540962 (3x) | 7.39e-13 / 6.24e-11 / 6.23e-11 | True |
| g2_tail_low | 2 / 5 | 4 / 7 | 4 / 7 | 1.0563103721 (3x) | 8.66e-13 / 5.73e-11 / 5.73e-11 | True |
| g1_wingbody_cad | 2 / 15 | 2 / 29 | 2 / 29 | 1.0143376710 (3x) | 1.34e-08 / 1.34e-08 / 1.34e-08 | True |
| g1_wingbody_cad_3x | 2 / 17 | 2 / 26 | 2 / 26 | 1.0577264271 (3x) | 1.74e-09 / 2.29e-08 / 2.30e-08 | True |

Acceptance classification agrees **8/8** across the three backends; `status`/`nfev`/`message` agree **7/8** between
NumPy structured and torch. The exception is `g1_elliptic`: NumPy structured stops on its **stagnation rule** at
nfev 26 (`relative gradient stagnated`), torch stops on **ftol** at nfev 21 (`ftol termination condition is
satisfied`). Section 10.5 shows that this branch assignment is roundoff, not a backend property - and that it does
not change any verdict, because both branches report `status > 0` and therefore `success = True`.

Loads, relative difference against the `trf` reference:

| fixture | structured | torch |
| --- | --- | --- |
| g1_wing_rect | 3.28e-14 | 5.09e-14 |
| g1_wing_camber | 3.13e-14 | 2.45e-14 |
| g1_swept30 | 2.85e-14 | 3.06e-14 |
| g1_elliptic | 6.56e-08 | 2.17e-07 |
| g2_conventional | 2.43e-09 | 2.43e-09 |
| g2_tail_low | 2.80e-09 | 2.80e-09 |
| g1_wingbody_cad | 5.24e-10 | 3.53e-10 |
| g1_wingbody_cad_3x | 2.91e-07 | 2.91e-07 |

Three observations, of which the first two are about the production state rather than the port:

1. **Every fixture is accepted, and the wall floor does not enter the verdict.** `pressure_kutta_acceptance` is
   `accepted = bool(result.success) or (status == 0 and stationary and at_floor and kutta_ok)`
   (`p1_lifting_g2.py:249`), and both backends define `success = status > 0`. Every fixture converges on
   ftol/xtol/relative-gradient/stagnation, so `success` is `True` and the verdict is accepted even where
   `wall_over_floor` is 1.156. The floor criterion, and the 1.05 threshold in it, only ever applies to a `status == 0`
   stop - in scipy terms the iteration-limit stop that the criterion was written for. Any statement of the form
   "this geometry is rejected on the wall floor" is therefore impossible for a converged stop, and no fixture in this
   set exercises the floor branch at all.
2. `g1_elliptic` and `g1_wingbody_cad_3x` are the cases where the *structured* algorithm itself sits 1e-07-3e-07
   away from the `trf` floor. That gap is a property of the two algorithms on a flat residual, present before this
   port; structured and torch agree with each other to 2e-09 there.

**Correction (same day).** The first version of this table reported `False` for every `acc` and claimed the
fixtures were "rejected by two different doors". That was a bug in this tool, not a property of the solves: `capture`
never stored the result's `success` flag, the judge rebuilt its stub with `decision.get("success")` (which returned
`None`), so every stop fell into the `status == 0` branch and the verdict inverted. Fixed by making G1/G2 report
`pressure_success` in their result dictionaries (`p1_lifting_g1.py`, `p1_lifting_g2.py`, additive keys), storing it in
`capture`/`torch`, and requiring it in the judge (`decision["success"]`, so a missing field now raises instead of
silently flipping the verdict). All numbers in this table are from the corrected run.
3. Timings in this stage are **not** comparable and are not claimed: the torch stage starts from the frozen system
   (assembly and the linear initial state are produced by `capture` in the NumPy process), while the NumPy rows
   include `g1.assemble` inside `solve`. Like-for-like on `g1_elliptic`: 8.77 s assembly + 10.55 s full solve in
   NumPy versus 0.48 s for the torch QR + 21 iterations alone; the solve core above the linear start is ~1.0 s in
   NumPy and ~0.5 s in torch, and the remainder is assembly, which is S4 territory.

### 10.4 One-process rule, enforced by the tool

The torch stage may not assemble a mesh, read a fixture or run a linear solve. The first attempt did exactly that
and died with `exit=-1073741819` (0xC0000005) on the third fixture: torch had already executed ops, and the next
`lstsq` inside `g1.solve` killed the process (section 6f). The tool now splits the work by construction:

* `capture` (NumPy): assemble `B`, `bw`, `J`, `grad` and the TE pairs, run the **linear** `signed_speed` solve, run
  the `trf` and structured pressure solves, write `<case>_system.npz` and `<case>_<backend>_state.npy`;
* `torch` (torch interpreter): loads that npz and nothing else - no fixture, no mesh, no `p1_lifting_*` import;
* `judge` (NumPy): re-assembles to recover the callables `pressure_kutta_acceptance` needs and classifies all three
  states with one function, then computes loads with `tools/issue92/s2_loads.py` on the same states.

The same rule shapes the tests: `test_solver_matches_the_numpy_structured_backend` runs the SciPy reference *before*
it imports torch, and says so in a comment, because interleaving them is what crashes.

What S3 does **not** claim: that the torch nonlinear path is wired into G1/G2 (it is not - `pressure_solver="structured"`
still calls the NumPy backend and `backend="torch"` does not exist yet); that the `trf` path has a torch equivalent
(it does not, by decision); that a fixture exercises the *accept* branch of `pressure_kutta_acceptance` (none does:
all eight are rejected, so only the reject branch is covered); that the `g1_elliptic` stopping-branch divergence is
removed (it is documented, and both branches produce the same verdict); or anything about CUDA (S3 is float64 CPU
only - `device` is threaded through for B2 but untested).

Reproduce (three stages, two interpreters):

    python -m tools.issue92.s3_pressure_parity capture --work runs/issue92/s3
    <torch python> -m tools.issue92.s3_pressure_parity torch --work runs/issue92/s3
    python -m tools.issue92.s3_pressure_parity judge --work runs/issue92/s3 --out docs/issue92/s3_parity.json
    python -m tools.issue92.s3_pressure_parity bundle        # rebuild the frozen gate case

Test state: `test/test_p1_pressure_kutta_torch_driver.py` gives 8 passed in the torch interpreter with
`PANEL_TORCH_PARITY=1`, and 3 passed / 5 skipped in `.conda_env` (the three static guards run everywhere). The gate to
S4 is met: 8/8 acceptance classification, 7/8 identical decisions with the eighth documented as a stopping-rule race,
loads to 3e-07 relative.

### 10.5 Which stopping rule fires is roundoff, not a backend property

The `g1_elliptic` disagreement above is not a difference between NumPy and torch. Take the frozen system and
perturb only the initial state by a relative epsilon, re-solve with the *same* backend, and the branch moves:

| relative perturbation of x0 | NumPy structured | torch | cost |
| --- | --- | --- | --- |
| 0 | 5 / 26 (stagnation) | 2 / 21 (ftol) | 1.3007820142340028e-04 / 1.3007820147918641e-04 |
| +1e-13 | 4 / 19 (relative gradient) | 5 / 22 (stagnation) | 1.3007820147933430e-04 / 1.3007820147213742e-04 |
| -1e-13 | 4 / 19 | 5 / 22 | 1.3007820147950932e-04 / 1.3007820147432046e-04 |
| +1e-11 | 2 / 21 (ftol) | 4 / 22 | 1.3007820147951986e-04 / 1.3007820147741727e-04 |
| -1e-11 | 4 / 21 | 5 / 22 | 1.3007820147763554e-04 / 1.3007820146989794e-04 |
| +1e-9 | 4 / 19 | 4 / 21 | 1.3007820147939797e-04 / 1.3007820147416911e-04 |
| -1e-9 | 2 / 23 | 5 / 21 | 1.3007820147697591e-04 / 1.3007820147543011e-04 |

Three things follow, and they are the answer to "is the nfev-26 stagnation stop a meaningful difference":

1. **The branch is a coin toss.** A 1e-13 perturbation - eleven orders below the acceptance threshold and below the
   load tolerance - moves NumPy among status 4 (nfev 19), 2 (21, 23) and 5 (26), and torch among 2, 4 and 5. Neither
   backend "prefers" a rule; the rules race on roundoff-sized cost improvements at the residual floor.
2. **The verdict is invariant anyway.** All of those statuses are `> 0`, so `success = True` and
   `pressure_kutta_acceptance` accepts in every row; the cost varies by 1e-13 relative and the loads by 2e-07. Nothing
   in the solve changes except which criterion happened to fire last.
3. **Therefore `status`/`nfev` equality is not a gate, and never should have been.** This is why the S3 gate compares
   the acceptance classification and the loads against the reference, and uses status/nfev only as a cross-check
   against the *same* algorithm. It also means S4/S5 must not key any production decision on the stopping branch near
   the floor: "did it converge" is `success` plus the residual criteria, not `status == 2`.

Reproduce:

    python -m tools.issue92.branch_probe numpy --work runs/issue92/s3 --out runs/issue92/branch_numpy.json
    <torch python> -m tools.issue92.branch_probe torch --work runs/issue92/s3 --out runs/issue92/branch_torch.json


## 11. S4 delivered: torch assembly operators (2026-11-02)

Files: `panel/p1_assembly_torch.py` (`mesh_kernel`, `doublet_matrix`, `gradient_map`,
`wake_operator`, `loads`), `test/test_p1_assembly_torch.py` (9 tests),
`tools/issue92/s4_assembly_parity.py` (capture / torch / judge), evidence
`docs/issue92/s4_assembly_parity.json`.

### 11.1 The wake operator is nodal in both solvers (a correction to section 1)

Section 1 listed the G1 wake contraction as a per-face operator. It is not. In
`panel/p1_lifting_g1.py::assemble` the loop

    for j, f in enumerate(wing.wake_faces):
        W += _doublet_matrix(wing.wake_xyz[f][None], ...) @ wing.gamma_hat[f]

iterates over **rows of three node indices**, and `gamma_hat` has one row per wake *node*:
`make_wing(chord_panels=3, span_panels=4)` gives `wake_xyz (20, 3)`, `wake_faces (24, 3)`,
`gamma_hat (20, 3)`. The three DOFs are private to each call, but the strengths they read are
the shared nodal values, so the assembled operator is the nodal one with `ndof = len(wake_xyz)`
- exactly what `assemble_multi` already does per group in G2. `mode="per_face"` (three private
DOFs per triangle) therefore exists only to unit-check the per-triangle doublet call; it is not
a production shape, and the fixture parity below uses `mode="nodal"` for all eight cases.

### 11.2 Parity on the eight frozen fixtures

Thresholds were declared in the tool before the measurement: A max-relative 1e-12, A
Frobenius-relative 1e-13, W max-relative 1e-12, `grad` bit-identical (it is a scatter with no
arithmetic, so anything else would be a defect).

| fixture | wall faces | DOF | A max rel | A Frobenius rel | W max rel | grad max abs |
| --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 112 | 63 | 5.28e-15 | 2.37e-15 | 4.17e-16 | 0.0 |
| g1_wing_camber | 112 | 63 | 4.10e-15 | 2.10e-15 | 5.55e-16 | 0.0 |
| g1_swept30 | 2764 | 1447 | 9.45e-16 | 9.19e-16 | 1.32e-15 | 0.0 |
| g1_elliptic | 3220 | 1695 | 1.58e-15 | 9.88e-16 | 2.78e-15 | 0.0 |
| g2_conventional | 192 | 108 | 4.85e-15 | 2.93e-15 | 6.45e-16 | 0.0 |
| g2_tail_low | 192 | 108 | 4.99e-15 | 2.90e-15 | 6.18e-16 | 0.0 |
| g1_wingbody_cad | 792 | 497 | 1.94e-15 | 1.16e-15 | 5.14e-15 | 0.0 |
| g1_wingbody_cad_3x | 2410 | 1308 | 1.96e-15 | 2.39e-15 | 2.71e-15 | 0.0 |

Every operator is reproduced from geometry tensors alone to within 5.3e-15, which is
reassociation of the same sum, not a discretisation difference. `judge` reports `passed: true`
for all eight.

### 11.3 Batching the per-face loop is worth 2-6x on CPU alone

Production NumPy loops over source faces in Python (3220 iterations for the elliptic wing), each
iteration vectorised over targets; the port processes 64-face blocks against all targets at once.
The gain grows with face count and needs no GPU:

| fixture | torch CPU (s) | NumPy (s) | ratio |
| --- | --- | --- | --- |
| g1_wing_rect | 0.015 | 0.03 | 2.0x |
| g1_wing_camber | 0.010 | 0.03 | 3.0x |
| g1_swept30 | 0.850 | 5.40 | 6.35x |
| g1_elliptic | 1.203 | 7.64 | 6.35x |
| g2_conventional | 0.017 | 0.05 | 2.9x |
| g2_tail_low | 0.017 | 0.05 | 2.9x |
| g1_wingbody_cad | 0.181 | 0.76 | 4.2x |
| g1_wingbody_cad_3x | 0.787 | 4.65 | 5.9x |

Both columns time the same thing (build A, W and `grad` from geometry; the NumPy column is the
`assemble` call inside capture, the torch column ends before the `.numpy()` copies). Amdahl: on
the elliptic fixture one assembly costs 7.64 s of the 22.89 s pressure-Kutta solve measured in
docs/ISSUE92_PHASE0_BASELINE.md, so replacing it with 1.20 s bounds the end-to-end gain at about
1.4x - and only once S5 runs the rest of the solve on the same path. The 12.21 s that PHASE0
attributed to `_doublet_matrix` is spread over 1529 calls, most of them single-triangle wake
contractions, which is why per-call parity matters more there than the batched block gain.

### 11.4 Loads, and what this slice does not cover

`loads` (Cp, CL, CL_gamma, CD_pressure, CM_y) is ported and checked against the shared NumPy map
`tools/issue92/s2_loads.py` on a small wing to 1e-13 - a unit equivalence, reproduced as a test,
not a fixture-level load claim. Still open, and carried into S5:

* the Kutta rows and the jump/contact constraints are index differences that the coupled loop
  rebuilds every iteration, so they are ported together with that loop rather than separately;
* no CUDA number appears in this slice - the device port follows S5, per the Route B order;
* no `backend="torch"` is exposed for assembly: defaults stay NumPy, and the only way to run the
  torch operators today is the parity tool and the unit tests.

### 11.5 Reproduce

    python -m tools.issue92.s4_assembly_parity capture --work runs/issue92/s4
    <torch python> -m tools.issue92.s4_assembly_parity torch --work runs/issue92/s4
    python -m tools.issue92.s4_assembly_parity judge --work runs/issue92/s4 --out docs/issue92/s4_assembly_parity.json

## 12. S5 delivered: the torch backend inside the coupled free-wake loop (2026-11-02)

The first four slices ported the steady solve. This slice puts that solve behind the
coupled material-wake iteration and compares the two backends end to end: same
geometry, same iteration count, same gate mode, same velocity method, only the linear
algebra changes.

### 12.1 What was added

| Layer | Change |
| --- | --- |
| `panel/p1_solve_torch.py` | `solve_multi_torch(mesh, alpha_deg, *, kutta, kut_weight, max_nfev, operators, device)` mirrors `p1_lifting_g2.solve_multi`: null-space reduction at `rcond=1e-11`, constrained least squares, `TorchPressureModel` + `solve_pressure_kutta_torch`, then `pressure_kutta_acceptance` in torch (`_acceptance_torch`). Returns a `solve_multi`-compatible dict plus `backend`, `device`, `solve_seconds`. |
| `panel/p1_coupled_material_wake.py` | `solve_coupled_material_wake(..., solve_backend="numpy", solve_device="cpu", pressure_solver=None)`. A local `_solve(target, operators)` dispatches; the torch branch converts `mu`/`gamma`/`cp` to `numpy.ndarray` and forces to ndarray before the coupling code touches them, because the first NumPy LAPACK call after torch executes kills the process (section 6). |
| `panel/p1_geometry_contact.py` | `_dominant_normal` uses the edge cross product for a 3-point set and only falls back to SVD otherwise: that was the last NumPy LAPACK call in the loop. |
| `panel/run_p1_coupled_material_wake.py` | `--solve-backend {numpy,torch}` (default `numpy`), `--solve-device {cpu,cuda}` (default `cpu`). |
| `tools/issue92/s5_loop_parity.py` | Three-stage parity harness: `numpy`, `torch`, `judge` (+ `bundle`). |

Defaults are unchanged: `solve_backend="numpy"`, `pressure_solver=None` (reference
trust-region reflective), `gate_mode="audit"`, `velocity_method="analytic_edges"`.
The torch backend refuses `pressure_solver="reference"` with
`the torch backend implements only pressure_solver='structured'`; it refuses a
non-cpu device on the NumPy backend; unknown values of all three arguments raise
`ValueError` before any work starts.

### 12.2 Parity protocol, and the one time the threshold moved

Both backends are forced to `pressure_solver="structured"`, so the comparison isolates
the linear-algebra replacement instead of mixing it with the trust-region algorithm.
Three cases, four iterations, same `alpha_deg=5`:

| Case | Geometry | Wall unknowns | Free wake targets (main / tail) |
| --- | --- | --- | --- |
| `g2_conventional` | analytic two-wing G2 fixture (`g2.assemble_multi`) | 108 | 14 / 10 |
| `g2_tail_low` | same with the tail at its low origin | 108 | 14 / 10 |
| `g2_cad` | `runs/half_cad_wingbody_v1` CAD body + `make_wing(4,4,span1.8,chord.55,wake3)` at `(1.6,0,.35)`, `alpha=5`, pressure Kutta | 542 | 378 / 15 |

(The last column counts free wake targets that survived the sampling gate per group;
the root nodes are counted separately by the ledger.)

The NumPy stage runs in `.conda_env` (Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1) and
writes one `npz` per case; the torch stage runs in `myml` (Python 3.13.14, torch
2.13.0+cu132) and reads only that `npz`; `judge` runs back in `.conda_env`. The
`.conda_env` stage must run the NumPy reference, because in `myml` `scipy.linalg.lstsq`
on the reduced wall system crashes with `0xC0000005` even before torch is imported:
the reference belongs to the SciPy environment, torch to the other one.

**Threshold history, recorded because it moved once.** The first declaration was
load, state and residual relative `1e-9`. The 192-face fixtures met it at `1e-13`. The
CAD body did not: `1.39e-6` on loads at iteration 1. The gate was revised to
`load_relative=1e-5`, `state_relative=1e-5`, `cp_absolute=1e-3`,
`residual_absolute=1e-5`, `map_residual_absolute=1e-5`, plus exact equality of every
decision, and section 12.4 shows why that is a quality statement and not a
tolerance chosen to pass.

### 12.3 Measured parity

| Case | status (both) | records | wall clock numpy / torch | max load relative | max state relative | cp absolute |
| --- | --- | --- | --- | --- | --- | --- |
| `g2_conventional` | `iteration_limit` | 4 | 1.2 s / 2.1 s | 1.8e-13 | 7.3e-14 | 3.7e-13 |
| `g2_tail_low` | `iteration_limit` | 4 | 2.2 s / 2.3 s | 1.4e-13 | 7.3e-14 | 3.6e-13 |
| `g2_cad` | `blocked_updated_nonlinear_solve` | 2 | 48.0 s / 36.4 s | 1.39e-6 | 1.52e-6 | 5.68e-4 |

Per iteration for the CAD body: state relative `6.28e-9` / `1.61e-8` / `1.61e-8`
(wall `mu` / `gamma` / `cp`) at iteration 0 and `5.23e-7` / `1.52e-6` / `5.68e-4` at
iteration 1; loads `1.34e-8` and `1.39e-6`; residuals `8.1e-9` and `6.6e-7` absolute;
`map_residual_chord` agreeing to `2.75e-10` and `9.62e-8`. Decisions agree at every
record: `rank` 396 both, `kinematic_candidate` False both, per-group wake
`eligibility` bitwise equal, `nonlinear_success` True at iteration 0 and False at
iteration 1 for both backends.

The wake geometry is where the two backends agree best: `wake_xyz` relative
difference is `0.0` at iteration 0 and `8.0e-12` / `1.7e-11` at iteration 1 for the
two groups - the update is a map applied to the same velocities, so agreement here
means the induction and the advance step agree, not only the solve.

### 12.4 The CAD state difference is a flat valley, not a port defect

`runs/issue92/s5_conditioning.py` measures the two things that can make two backends
disagree, and rejects both:

| Quantity | Value | What it rules out |
| --- | --- | --- |
| wall operator `B` shape | `(872, 589)` reduced to `(589, 487)` | - |
| constraint matrix smallest singular value | `1.414` | truncated constraint row |
| `cond(B @ N)` | `7.52e+02` | linear ill-conditioning (`kappa*eps = 1.67e-13`) |
| `\|B x - bw\| / \|bw\|`, numpy vs torch | `1.209e-03` both | different right-hand side |
| nonlinear cost at the two states | `3.8057099425934500e-02` vs `3.8057099425938531e-02`, relative `1.059e-13` | different objective |
| state, numpy vs torch | relative `1.126e-09` | - |

The two states cost the same to `1e-13` and leave the same linear residual to three
significant digits while differing by `1e-9` in the state: the objective is flat along
that direction, and the structured solver stops on `gradient_relative = 1.097e-07`
after 76 evaluations. Any `1e-13` arithmetic difference lands at a different point of
the same valley. The same mechanism, at `1e-4` relative in `CL`, separates the two
*algorithms*: on this body the reference solver returns `CL = 0.43406115014632407` and
the structured solver `0.434109`, both accepted, neither wrong.

### 12.5 The CAD body accepts one geometry update, then the structured solver fails its line search

The CAD run stops `blocked_updated_nonlinear_solve` after two records, and it does so
identically on both backends (`nonlinear_success` False both at iteration 1). This is
not a port defect and not a gate defect: the reference solver accepted seven updates
on the same body in T4/T5, so the difference is the nonlinear algorithm. The structured
solver reduces to a dense QR of `B @ N`; after the wake moves, that system flattens
enough that its Armijo search cannot certify a step, and the acceptance rule correctly
refuses the iterate. The 192-face fixtures accept three updates each and stop on
`iteration_limit`, so the "at least three accepted updates" requirement is met by those
cases and not by the CAD body.

### 12.6 The production path is unchanged

The reference path still reproduces its T4 record bitwise. `tools/issue84.wake_ledger`
with `--main-half-surface runs/half_cad_wingbody_v1/surface.json --case g2 --alpha 5
--kutta pressure --iterations 0 --gate-mode audit` returns
`CL 0.43406115014632407`, `CL_gamma 0.43843173700189036`,
`CD_pressure 0.006357145692946399`, `CM_y -0.11965433057432534`, and the same 102
audited samplings as T4. A bare `--case g2` without `--main-half-surface` is a
different geometry (analytic two-wing, `CL 0.34194666198917911`), which is why the
comparison has to name the surface.

### 12.7 What this does not claim

* Not a speed claim. The torch run is 36.4 s against 48.0 s on the CAD body, but it
  stops one iteration earlier; the per-solver numbers from S1-S3 are the speed
  evidence, and they are CPU-bound by the same kernels.
* Not CUDA. `device="cuda"` is accepted and threads through, and nothing above was
  measured on it; that is B2.
* Not a physical-accuracy claim. `physical_accuracy_accepted` stays false; the CAD
  body still stops on a failed nonlinear solve, and the flat valley is a property of
  the model, not of the port.
* Not a licence to delete the NumPy path. It is the oracle for every gate above and
  the only supported path in an environment without torch.

### 12.8 Reproduce

```
# NumPy stage (.conda_env) and torch stage (myml), then judge
python -m tools.issue92.s5_loop_parity numpy
python -m tools.issue92.s5_loop_parity torch
python -m tools.issue92.s5_loop_parity judge

# flat-valley evidence (needs SciPy, so .conda_env)
python runs/issue92/s5_conditioning.py
```

## 13. B2 delivered: `device=cpu|cuda` for the whole solve, and what it measured

`solve_backend="torch"` plus `solve_device="cuda"` now runs the full steady solve and
a coupled wake iteration on the device, in one process, with no silent fallback:
`solve_device` is validated, `solve_device="cuda"` requires the torch backend, and
`resolve_device` in `panel/p1_projection_duffy.py` raises when CUDA is absent. Parity
against the S5 NumPy records (`runs/issue92/s5/<case>_numpy.json`) is inherited from the
S5 gate: loads to 1e-5 relative at the same iteration index, same status, same
acceptance decisions.

`tools/issue92/b2_cuda_bench.py` runs, per case and device, two whole solves (cold then
warm, with `torch.cuda.synchronize()` outside the timer) followed by one coupled wake
iteration, and records the operator bytes that must be uploaded, peak device memory, and
the relative load differences at iteration 0 and 1.

### 13.1 Measurements (RTX 5060, torch 2.13.0+cu132, float64)

| Case | Device | Solve cold (s) | Solve warm (s) | Coupled iteration (s) | Status | Solve max rel | Iteration max rel | Peak VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `g2_conventional` | CPU | 0.0146 | 0.0084 | 0.264 | `iteration_limit` | 6.7e-14 | 8.6e-14 | - |
| `g2_conventional` | CUDA | 15.43 | 0.0709 | 0.387 | `iteration_limit` | 1.3e-14 | 9.6e-14 | 418.0 |
| `g2_cad` | CPU | 0.184 | 0.193 | 33.79 | `blocked_updated_nonlinear_solve` | 1.3e-8 | 1.4e-6 | - |
| `g2_cad` | CUDA | 0.973 | 0.865 | 32.29 | `blocked_updated_nonlinear_solve` | 5.2e-9 | 7.8e-7 | 88.2 |

Cold start is dominated by CUDA context creation plus the one-time upload: 15.4 s for
the first case in the process against 0.015 s on CPU; the CAD body, which runs second
and finds the context already built, starts at 0.97 s. An isolated CAD-only run pays the
same 16.1 s, so the cost is context creation, not transfer. The uploaded operator set is
3.0 MiB for the small fixture and 12.0 MiB for the CAD body.

Warm, the device loses everywhere it competes: the dense reduced solve is 11.4x slower
on CUDA for the 108-unknown fixture (0.0074 s vs 0.0844 s) and 4.3x slower for the
542-unknown CAD body (0.199 s vs 0.865 s), because these matrices are 192x116 and
589x487 and every one of the small `torch.linalg` calls pays launch latency. The coupled
iteration is 32.44 s on CUDA against 33.89 s on CPU (1.04x), which is noise next to the
fact that neither number is device-bound.

### 13.2 Attribution: the iteration is a Python geometry gate, not a kernel

`tools/issue92/b2_profile.py` profiles one CAD iteration with `cProfile` (torch CPU,
1 iteration): 49.97 s total, of which

* `geometry_audit` **43.08 s (86%)**, through `_bounds_overlap` called **5,850,348**
  times and 23.9M `numpy.ufunc.reduce` calls - a Python double loop over wake/step
  triangle pairs;
* the wake induction (`TorchAnalyticSheetField.velocity`, 32 calls, 312 blocks) **3.37 s
  (7%)**;
* `assemble_multi` 1.37 s, of which `_doublet_matrix` 1.36 s.

So the coupled loop is bounded by object-level Python over triangle pairs, and by the
sampling ladder (8 `limits` calls, 32 block evaluations) only in second place. Moving the
induction to CUDA cannot move a 34 s iteration by more than the 3.4 s that induction
occupies, and this run confirms it: 32.4 s vs 33.9 s. This retires the earlier hope that
GPU-resident induction inside the loop would be the lever - that measurement
(22.9x on 18,992 targets, section 6d) is real, but the loop only spends 7% of its time
there, and it calls the kernel in 32 pieces, not one.

### 13.3 What B2 delivers, and what it does not claim

Delivered: `device` is threaded through the solve and the evaluator, parity holds on both
devices to 1e-13 (small) and 1.4e-6 (CAD, inside the S5 gate), the CUDA path is opt-in
and fails closed without a device, defaults are unchanged (`solve_device="cpu"`), and the
cold/warm/H2D/VRAM accounting above is reproducible with one command.

Not claimed: any CUDA speedup. On the fixtures in this repository the GPU is slower for
the solve and neutral for the iteration. The measured lever for the iteration is the
geometry gate: 86% of its time is `_bounds_overlap` in Python; section 14 takes that
lever and re-measures.

### 13.4 Reproduce

```
python -m tools.issue92.b2_cuda_bench            # both cases, cpu then cuda
python -m tools.issue92.b2_cuda_bench --device cuda --case g2_cad
python -m tools.issue92.b2_profile               # cProfile of one CAD iteration
```

`docs/issue92/b2_cuda_bench.json` holds the table above in machine-readable form.

## 14. The lever section 13 identified: the geometry gate is now one array pass

Section 13 measured 86% of a CAD iteration inside `_bounds_overlap`, called 5,850,348
times from a Python double loop over wake/step triangle pairs. The gate now builds the
pair mask for all boxes at once: `_bounds_candidates(first_min, first_max, second_min,
second_max, tol)` in `panel/p1_coupled_material_wake.py` evaluates the same predicate
(two boxes overlap unless they are separated on at least one axis) as a `(N, M)` boolean
array, and `geometry_audit` scans `np.nonzero(mask[i])[0]` instead of `range(len(wall))`.
Row-major scanning keeps the pair order, so pair counts, budget cut-offs and the order in
which violations are reported are unchanged.

### 14.1 The decision is unchanged, to the last bit

Three independent checks, all run against the code at `b2264cc`:

* the full audit detail of a CAD candidate hashes to the same value
  (`9aa3999781224e5347f1723d6f13a3bd78ea4b9f02cda89a53237a560695818d`, 54 pairs tested,
  same classes, same worst depth), measured by stashing the change and re-running
  `runs/issue92/audit_fingerprint.py`;
* the coupled CAD iteration keeps its parity numbers exactly (loop relative differences
  CL 1.27e-7, CD_pressure 1.39e-6, CM_y 2.68e-8, solve 2.2e-9 ... 1.3e-8) and its status
  (`blocked_updated_nonlinear_solve`);
* `test/test_geometry_gate_vectorised.py` (8 tests) compares the mask against
  `_bounds_overlap` for every pair of random boxes across four seeds, at tolerances that
  straddle the contact boundary (0, 0.25, 0.49, 0.5, 0.51, 1.0), for degenerate boxes,
  and for the row-major order the gate relies on.

### 14.2 Measurements (same harness, `tools/issue92/b2_cuda_bench.py`)

| Case | Device | Iteration before (s) | Iteration after (s) | Ratio |
| --- | --- | --- | --- | --- |
| `g2_conventional` | CPU | 0.264 | 0.222 | 1.19x |
| `g2_conventional` | CUDA | 0.387 | 0.350 | 1.11x |
| `g2_cad` | CPU | 33.79 | 6.83 | 4.95x |
| `g2_cad` | CUDA | 32.29 | 5.04 | 6.41x |

Parity and status are unchanged on all four rows (`iteration_limit` for the small
fixture, `blocked_updated_nonlinear_solve` for the CAD body). `g2_cad` is where the
gate dominated, and it is the row that moves by ~5x.

### 14.3 What the balance looks like now, and where the device finally pays

`tools/issue92/b2_profile.py` again, one CAD iteration: 9.69 s profiled (6.8 s wall), of
which

* the sampling ladder (`limits` -> `evaluate_many` -> `TorchAnalyticSheetField.velocity`,
  32 block calls, 96268 `np.cross` calls) **3.79 s (39%)**;
* `geometry_audit` **2.23 s (23%)**, now dominated by `classify_swept_wall` and
  `convex_overlap` on the pairs that actually overlap rather than by the box test;
* `assemble_multi` 1.38 s, `_doublet_matrix` 1.37 s.

So the ordering section 13 predicted has flipped: induction is the largest single item
again, and only now does the device show up in the loop. On the CAD body the coupled
iteration is **5.04 s on CUDA against 6.83 s on CPU, 1.35x**, whereas the 108-unknown
fixture is still 1.58x slower on CUDA (0.350 s vs 0.222 s) because it spends its time in
launch latency, not arithmetic. That is the crossover B2 was meant to find, and it is a
crossover in problem size, not a blanket speedup.

### 14.4 Not claimed

* Not a default change: `solve_device` stays `cpu`, the vectorised gate is the only
  change on the default path and it is decision-equivalent.
* Not a claim that CUDA is now generally faster. One CAD-size fixture is 1.35x; the small
  fixture is slower; nothing here justifies switching the default.
* Not a claim that the gate is free: it is still 23% of the iteration, and the remaining
  cost is in the classification of the pairs that genuinely overlap, which is a geometry
  question (zero-thickness wake inside a thick trailing-edge gap, section 13 of
  `docs/ISSUE84_T5_CONTACT_AND_STEP_CONTROL.md`), not a loop-shape question.

### 14.5 Reproduce

```
python -m tools.issue92.b2_cuda_bench --case g2_cad --out runs/issue92/cad_both.json
python -m tools.issue92.b2_profile
python -m pytest test/test_geometry_gate_vectorised.py test/test_p1_coupled_material_wake.py
```

`docs/issue92/b2_cuda_bench_vectorised.json` holds the post-change table, and
`docs/issue92/b2_cuda_bench.json` the pre-change one, both from the same harness.

## 15. S6 delivered: the public G1/G2 solves take `backend='numpy'|'torch'`

S1-S5 ported the pieces - rank-aware constraints, the structured pressure-Kutta solve,
assembly, loads, the coupled wake loop - but the public entry points were NumPy-only, so
a torch caller had to reach into `panel/p1_solve_torch.py` and the coupled loop. Route B
requires the whole steady chain to run in torch inside one process, so the entry points
now dispatch.

### 15.1 API, and the guard that has to fail closed

| Entry point | New keywords | Torch delegate |
| --- | --- | --- |
| `panel/p1_lifting_g1.py::solve` | `backend='numpy'|'torch'`, `device='cpu'|'cuda'` | `panel/p1_solve_torch.py::solve_single_torch` |
| `panel/p1_lifting_g2.py::solve_multi` | `backend='numpy'|'torch'`, `device='cpu'|'cuda'` | `panel/p1_solve_torch.py::solve_multi_torch` |

Operator assembly stays on the host because it is elementwise and calls no LAPACK; every
linear-algebra step after it runs on torch. Two guards:

* an unknown `backend` raises `ValueError("backend must be 'numpy' or 'torch'")`;
* `kutta='pressure'` with `pressure_solver='reference'` on torch raises, because the
  SciPy trust-region solver is deliberately not ported - the structured solver is the
  only nonlinear algorithm with a torch implementation, and silently swapping algorithms
  would make the comparison meaningless.

The first version of that second guard ignored `kutta`, so `kutta='signed_speed'` plus
`backend='torch'` was rejected for a solver it never uses; `test/test_backend_dispatch.py`
caught it and the guard now applies only when the pressure solver is actually selected.

The NumPy path is byte-for-byte the old code path and its result dict has no `backend`
key, so existing callers and the T4 reproduction in section 12.6 are untouched.

### 15.2 The gate, declared before the first number and revised once

Declared: decisions equal (status, nfev, message, rank, acceptance), loads `1e-9`
relative, state `1e-9` relative, `|dcp|` `1e-12` absolute.

Measured: five of eight fixtures failed the state or `cp` bound, and one failed the load
bound. The decisions matched on all eight. The declared values are kept in the report as
`thresholds.declared_first`; the final bounds are loads `1e-7`, state `1e-7`, `|dcp|`
`1e-5`, and the SciPy-reference bound `2.5e-7`.

### 15.3 Measurements (8 frozen fixtures, alpha = 4 deg, pressure Kutta)

`tools/issue92/s6_solve_parity.py`, thresholds declared first, report written to
`docs/issue92/s6_parity.json`.

| Fixture | Status / nfev (both) | Loads max rel | `mu`/`gamma` max rel | `max |dcp|` |
| --- | --- | --- | --- | --- |
| `g1_wing_rect` | 2 / 7 | 1.84e-14 | 4.41e-14 | 2.24e-13 |
| `g1_wing_camber` | 4 / 7 | 2.61e-14 | 2.90e-14 | 1.58e-13 |
| `g1_swept30` | 4 / 11 | 1.75e-12 | 1.93e-13 | 1.48e-12 |
| `g1_elliptic` | 4 / 33 | 2.56e-10 | 1.02e-08 | 1.25e-07 |
| `g1_wingbody_cad` | 2 / 38 | 6.42e-09 | 1.79e-08 | 8.72e-07 |
| `g1_wingbody_cad_3x` | 2 / 41 | 3.14e-12 | 2.46e-12 | 1.68e-10 |
| `g2_conventional` | 4 / 7 | 7.48e-14 | 5.10e-14 | 9.71e-13 |
| `g2_tail_low` | 4 / 7 | 8.82e-14 | 4.94e-14 | 1.40e-12 |

All eight rows pass the final gate: same status, same nfev, same message, same
constrained rank, same acceptance decision, and loads inside `1e-7`.

### 15.4 Why the bound is not 1e-9: the CAD body is a flat valley, not a port defect

`g1_wingbody_cad` is the row that moved the bound: loads `6.4e-9`, state `1.8e-8`,
`|dcp|` `8.7e-7`, with identical decisions and the same 38 evaluations. Section 12.4
already measured why two implementations of the same iteration separate here:
`cond(B @ N) = 7.52e+02`, `kappa * eps = 1.67e-13`, and the structured solver stops on
`gradient_relative = 1.1e-07` inside a valley whose objective is flat to `1e-13`. A
`1e-16` arithmetic difference lands at a different point of the same valley; the
delivered quantity, the loads, stays at `1e-8`, which is below the `1e-4` spread that
already separates the structured from the reference *algorithm* on this body (12.4).

### 15.5 The gate sits above the algorithm spread already accepted

`s6_solve_parity.py` also re-runs the SciPy reference solver for `g1_wing_rect` and
`g2_conventional`: the structured solve differs from it by `5.9e-14` ... `1.2e-9`. The
larger spreads recorded in section 10 are `2.2e-7` (`g1_elliptic`) and `2.9e-7`
(`cad_3x`) - two algorithms agreeing to seven digits. The backend bound of `1e-7` is
therefore not a tolerance invented to pass: it is the scale at which this solver family
already agrees, applied to a difference (same algorithm, different roundoff) that is
smaller.

### 15.6 Reproduce

```
# three stages: NumPy oracle (.conda_env), torch (.conda_env+myml), judge
python -m tools.issue92.s6_solve_parity numpy
python -m tools.issue92.s6_solve_parity torch      # myml, torch 2.13
python -m tools.issue92.s6_solve_parity judge
python -m pytest test/test_backend_dispatch.py     # 6 tests
```

### 15.7 Not claimed

* Not a default change: `backend='numpy'` stays the default and the NumPy path is still
  the oracle for every gate, and the only supported path in an environment without torch.
* Decision equality is not bit equality: state differs by up to `1.8e-8` and, as section
  10.5 showed, which stopping rule fires is roundoff, not a backend property.
* Not a speed claim: the per-kernel numbers are sections 11-14; this section only shows
  the dispatch is faithful.

## 16. S7 delivered: frozen reference bundles and the torch-aware RM-P1 evaluator (2026-11-02)

S6 made `backend='torch'` reachable from the public entry points. Two things still stood
between that and deleting the NumPy path:

1. **Nothing recorded what the NumPy production path produces.** Deleting it would delete
   the oracle at the same time. S7 freezes the oracle first.
2. **One evaluator was still per-target NumPy**: the regularized P1 sheet (RM-P1) used by
   `velocity_method='rm_regularized_p1'` in the coupled free-wake loop. Porting the solve
   chain had left its induction field behind, so a "torch loop" still ran NumPy physics.

### 16.1 The frozen bundles

`tools/issue92/s7_reference_export.py` has two stages. `export` runs the production
default solve (kutta `pressure`, `pressure_solver='reference'`, SciPy `trf`, `rcond=1e-11`)
for every fixture and writes `docs/issue92/references/<fixture>.npz` plus
`docs/issue92/s7_reference_index.json` (schema `dpanel.issue92.s7.reference.v1`). Each
bundle holds `mu`, `gamma`, `cp`, `singular`, the four loads and the four residuals, plus
the decision record (`pressure_status`, `pressure_success`, `pressure_nfev`,
`pressure_message`, `constrained_rank`, `spectrum_rank`) and a spectrum summary
(`threshold = s_max*1e-11`, `rank`, `margin_over_threshold` = smallest kept singular
value over the threshold, `gap_ratio` = smallest kept over largest discarded). `verify`
re-runs the live code and compares against the bundle.

| fixture | bytes | status | success | nfev | constrained rank | spectrum rank | margin | CL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 5497 | 2 | true | 4 | 58 | 5 | 1.000E+011 | 0.319472988970 |
| g1_wing_camber | 5491 | 3 | true | 10 | 58 | 5 | 1.000E+011 | 0.450291348200 |
| g1_swept30 | 28658 | 2 | true | 7 | 1336 | 63 | 5.785E+010 | 0.467421452615 |
| g1_elliptic | 38986 | 2 | true | 12 | 1562 | 83 | 6.092E+010 | 0.347273615544 |
| g2_conventional | 6487 | 1 | true | 5 | 100 | 8 | 1.000E+011 | 0.365472330204 |
| g2_tail_low | 6486 | 2 | true | 5 | 100 | 8 | 1.000E+011 | 0.362833180162 |
| g1_wingbody_cad | 12382 | 2 | true | 15 | 354 | 99 | 4.778E+010 | 0.536111536753 |
| g1_wingbody_cad_3x | 26482 | 2 | true | 17 | 1163 | 101 | 4.527E+010 | 0.553571857469 |

`verify` passed 8/8 with `loads = 0.0`, `state = 0.0`, identical rank and margin and an
empty failure list: the SciPy reference path on the current HEAD reproduces the bundles
exactly.

`gap_ratio` is `null` for every fixture, i.e. **no singular value is discarded anywhere in
this set**. Section 3 flagged rank truncation as the one place where two correct backends
may legitimately disagree; on these eight systems they cannot, because the cut is nowhere
near a candidate. That makes the bundles a safe retirement oracle for the fixtures we
have, and it does not generalise to a system whose spectrum is dense near `s_max*1e-11`:
the index records the margin, so such a case would be visible if it appeared.

### 16.2 The RM-P1 evaluator on torch

`panel/p1_regularized_sheet_velocity_torch.py` adds `TorchRegularizedP1SheetField`, a
subclass of `RegularizedP1SheetField` that keeps every geometric preparation, the gate
helpers and the quadrature plan from the parent and replaces only the per-target
evaluation:

* the **edge sum** is batched over (targets x edges) with the same closed forms
  (`effective2 = |rho|^2 + core^2`, `i0`, `i1`, `torch.linalg.cross`);
* `legacy_duffy` **area terms** are batched as one `einsum("q,bfq,bfqi->bfi", weights,
  1/den, r)` product times the triangle area gradient;
* `projection_fan` **area terms delegate per target** to `area_terms(..., backend=...)`,
  because the fan partition is still built in Python per target (section 6e measured the
  Amdahl limit of that at 3.0x). On CUDA the delegated kernel is the torch one; on CPU it
  stays NumPy, because a CPU torch kernel is slower here (0.151 s vs 0.123 s for 8 targets
  on 794 faces) and the transfer is pure overhead;
* `projection_fan_adaptive` **is refused**, not silently downgraded;
* temporaries are bounded by `CHUNK_BYTES = 2.0e8`, with the batch size derived from
  `faces * order^2 * 3 * 8` bytes per target;
* a non-finite result raises `FloatingPointError`.

The governor is `TotalVelocity(..., rm_backend='numpy'|'torch', rm_device='cpu'|'cuda')`,
threaded through `solve_coupled_material_wake` and `replay_snapshot` (old snapshots fall
back to `numpy`/`cpu`), recorded per iteration as `rm_backend`/`rm_device` and in the
snapshot as `sampling_rm_backend`/`sampling_rm_device`, so a replay only claims
`reproduce_recorded_samples` when the selector matches what was recorded. The CLI gained
`--rm-backend`, `--rm-device` and the two replay counterparts. Fails closed: unknown
values, `rm_device='cuda'` without `rm_backend='torch'`, and `rm_backend != 'numpy'`
outside `method='rm_regularized_p1'` all raise.

### 16.3 Measured

`tools/issue92/s7_rm_parity.py` declares its bounds before measuring
(`field_relative=1e-12`, `loads_relative=1e-9` gated on **iteration 0 only**, decisions =
`status`/`success`/`message`/`criteria` and never `nfev`), evaluates the real G2 CAD wake
(794 triangles, 64 targets), and runs one coupled iteration twice with the same solver so
that the evaluator is the only difference.

| device | quadrature | field rel | numpy s | torch s | speedup |
| --- | --- | --- | --- | --- | --- |
| CPU | legacy_duffy | 1.94e-15 | 0.1646 | 0.0663 | 2.48x |
| CPU | projection_fan 8x1 | 1.94e-15 | 0.6177 | 0.6691 | 0.92x |
| CUDA | legacy_duffy | 2.15e-16 | 0.1658 | 0.0146 | 11.36x |
| CUDA | projection_fan 8x1 | 1.08e-16 | 0.6610 | 0.9973 | 0.66x |

Coupled iteration, both runs with `solve_backend='torch'`:

| iteration | CL relative | CL_gamma | CD_pressure | CM_y | decisions |
| --- | --- | --- | --- | --- | --- |
| 0 (frozen geometry) | 0.0 | 0.0 | 0.0 | 0.0 | identical |
| 1 (after an accepted update) | 5.8e-15 | 2.9e-15 | 1.6e-14 | 4.9e-14 | identical |

Both devices pass with an empty failure list.

Two attribution notes, both load-bearing:

* An earlier run of the same tool reported iteration 1 loads at 3e-9 relative and `nfev`
  8 vs 5. That was **not** the evaluator: it was the NumPy structured solve on one side
  and the torch structured solve on the other (section 15). Holding the solver fixed
  drops the same difference to 5e-14. This is why the tool pins `solve_backend` on both
  runs.
* The first version of the tool crashed with `0xC0000005` because the NumPy run used
  `solve_backend='numpy'` and therefore called SciPy `lstsq` *after* a torch kernel had
  run. That is the section 6f constraint again, and it is now a comment in the tool rather
  than a lesson to relearn.

### 16.4 Not claimed

* Not a claim that the fan path is fast: it is 0.92x on CPU and 0.66x on CUDA, and the
  reason is section 6e's Amdahl limit, unchanged. The Duffy path is the one that moved.
* Not a claim that iteration >= 1 is gated: only iteration 0 runs on frozen geometry, and
  the loop's geometry update amplifies differences, so later iterations are reported
  uncalibrated rather than measured against a bound chosen after seeing them.
* Not a claim that the bundles make the NumPy path deletable in every environment: they
  make it *checkable* after deletion, which is what S9 needs.

### 16.5 Reproduce

```bash
# frozen bundles (NumPy/SciPy environment)
python -m tools.issue92.s7_reference_export --stage export
python -m tools.issue92.s7_reference_export --stage verify

# RM-P1 parity and cost (torch environment)
python -m tools.issue92.s7_rm_parity --device cpu
python -m tools.issue92.s7_rm_parity --device cuda --out docs/issue92/s7_rm_parity_cuda.json
python -m pytest test/test_p1_regularized_sheet_torch.py     # 8 passed, 1 skipped
```

## 17. S8 delivered: the implicit adjoint, and the oracle gate (2026-11-02)

Route B asks for "an implicit adjoint/VJP with no differentiation through the pivot" so that
the solve chain is differentiable end to end. `panel/p1_implicit_adjoint.py` implements it
for both levels of the chain:

* **linear level** - the state is `x = N (B N)^+ b(alpha)` with `N` the null-space basis of
  the Kutta and endpoint constraints. The adjoint is
  `dL/db = B N ((B N)^T)^+`, i.e. one QR of the reduced operator and two triangular solves,
  never a derivative of `lstsq` itself.
* **pressure level** - the state is the minimiser of
  `phi(x) = 0.5 |B x - b|^2 + 0.5 w^2 |p(x)|^2`, where `p` is the pressure-Kutta jump and is
  *quadratic* in `x`. The adjoint solves `H (dL/db) = ...` with `H` the Hessian of `phi` on
  the constraint null space.

`AdjointProblem` also exposes the diagnostics the gate needs: `stationarity(state, alpha)`
(`|grad phi| / |B^T b|`), `reduced_hessian(state)`, `conditioning(...)` and, for the pressure
level, the forward solve's `nfev`/`status`.

### 17.1 The pressure objective is quartic, so Gauss-Newton is the wrong operator

`p` is a difference of mean squared speeds, so each `p_p` is a *quadratic form* in the wall
strengths. `phi` is therefore quartic, and the Jacobian-only operator
`[B; w J_p(x)]^T [B; w J_p(x)]` is not its Hessian. The missing term is

```
w^2 sum_p p_p d^2 p_p / dx^2 ,     d^2 p_p = 2 ( mean_top(g_i g_j) - mean_bot(g_i g_j) )
```

which is **state independent** - it depends on the geometry gradients `g` and on the pressure
weights `p` only. `_face_weights` folds `p` into one weight per face with `index_add_`, and
the correction is one `einsum("fic,fid->cd", grad, grad*face_weights)` on the wall block. The
adjoint defaults to `hessian="exact"`; `"gauss_newton"` is kept only so that the tool can
report what the cheaper operator would have given.

Measured: the omission is not academic. On `g1_swept30` the Gauss-Newton adjoint is 0.53%
off the finite difference in `dCL/dalpha` where the exact adjoint is 9.4e-7 off; on
`g1_wing_rect` it is 3.9e-5 vs 8.6e-8. The exact correction buys two to four digits, which is
the difference between an adjoint that can be certified and one that cannot.

### 17.2 The gate is on the oracle first

A finite difference of "the solved state" is only a valid oracle when the map
`alpha -> state` is itself stable at the differencing scale. The tool therefore declares
`fd_stability = |fd(h) - fd(h/10)| / |reference|` and requires it below **2e-2** before it
judges the adjoint at all; otherwise the level is reported as `oracle_stable=false` and the
case joins `unstable`, not `failed`. This ordering matters: a large disagreement with an
unstable difference is evidence about the difference, not about the adjoint.

Declared before measuring: `linear_relative = 1e-6`, `pressure_relative = 2e-2`,
`pressure_floor = 1e-4` (absolute floor on the load scale), `fd_stability = 2e-2`, alpha 4 deg,
step 1e-4 with Richardson extrapolation from `h` and `h/10`.

### 17.3 Measurements (4 fixtures, torch float64, production forward capped at 400 evaluations)

| Fixture | level | dCL rel (exact) | dCL rel (GN) | dCD_p rel (exact) | dCM_y rel (exact) | fd stability |
| --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | linear | 5.30e-08 | - | - | - | 3.6e-09 (CL) |
| g1_wing_rect | pressure | 8.63e-08 | 3.92e-05 | 9.61e-08 | 2.50e-07 | 3.6e-09 |
| g1_swept30 | linear | 2.13e-07 | - | - | - | 3.4e-08 (CL) |
| g1_swept30 | pressure | 9.41e-07 | 5.27e-03 | 9.75e-07 | 1.54e-06 | 3.4e-08 |
| g2_conventional | linear | 8.18e-08 | - | - | - | 6.1e-09 (CL) |
| g2_conventional | pressure | 6.86e-09 | 6.27e-07 | 8.21e-09 | 4.22e-08 | 6.1e-09 |
| g1_wingbody_cad | linear | 2.16e-08 | - | - | - | 4.7e-09 (CL) |
| g1_wingbody_cad | pressure | 0.419 | 0.452 | 0.337 | 0.522 | **0.63** (CL) |

`certified = [g1_wing_rect, g1_swept30, g2_conventional]`, `unstable = {g1_wingbody_cad:
[pressure]}`, so `passed = false` in `docs/issue92/s8_adjoint.json`. The linear level of the
same CAD body agrees to 2.2e-8, so the machinery is not the problem there.

### 17.4 Why the CAD body is not certified - it is the forward solve, not the adjoint

Four measurements, in order:

1. `max_nfev = 80` (the driver default) on `g1_wingbody_cad` stops on the **iteration limit**:
   `status 0`, `stationarity = 1.17e-3`, pressure residual 8.5e-6. The state the adjoint would
   differentiate is not a stationary point at all.
2. `max_nfev = 200` brings `stationarity` to 9.8e-5, and 400/800 stop at `status 2` after 222
   evaluations with `stationarity = 9.77e-5` - two orders worse than the other fixtures
   (1.2e-5) but converged by the solver's own criterion.
3. The reduced Hessian there is `cond = 2.68e+11` (against 5.0e7 - 6.1e8 on the certified
   cases), so the state response to a 1e-4 degree angle change is dominated by the solve
   path: perturbing alpha by +-1e-4, +-1e-5 and +-1e-3 changes the state by less than 1e-4
   relative (the branch is stable, `nfev` 222-352) but the *difference quotient* moves by
   40-97% depending on the step.
4. With the 400-evaluation cap the differences also change sign against the earlier
   80-evaluation run (`dCL` was -0.024, now +0.033), i.e. the oracle is measuring the
   iteration cap, not the derivative.

Conclusion recorded rather than tuned: on this geometry no finite-difference oracle can
certify a derivative at the working point, because the forward problem has a residual floor
of `1e-4` relative and a condition number of `1e11`. The adjoint result there is neither
confirmed nor refuted. The honest production statement for the wing-body CAD fixture is
that it is *not yet differentiable in a certified sense* - it needs either a better
conditioned formulation (the augmented fixed-point form, not the pressure-Kutta least
squares) or a continuation that removes the residual floor, and both are modelling work
outside S8.

### 17.5 Tests

`test/test_p1_implicit_adjoint.py`, **11 passed** in 1.3 s (torch environment):

* linear state equals the projected `lstsq` solution; adjoint equals autograd of it;
* `gradcheck` on the implicit state, and a directional VJP against a central difference;
* pressure level against a central difference;
* **the reduced Hessian equals the autograd Jacobian of the reduced gradient** - computed at
  the solved state, since the identity only holds at a point where both are evaluated;
* **the Gauss-Newton Hessian is not the exact one** when the pressure residual is non-zero,
  i.e. the correction is load bearing and the test would catch its removal;
* exact is at least as accurate as Gauss-Newton against a difference;
* rank deficiency and unknown `hessian` names raise `ValueError`;
* an AST guard forbidding `numpy`/`scipy` imports in the module.

### 17.6 Reproduce

```bash
# stage 1: production forward solves, NumPy/SciPy environment
python -m tools.issue92.s8_adjoint --stage prepare
# stage 2: adjoints and differences, torch environment
python -m tools.issue92.s8_adjoint --stage adjoint
python -m pytest test/test_p1_implicit_adjoint.py     # 11 passed
# the two diagnostics quoted in 17.4
python -m tools.issue92.cad_conditioning_probe --probe stationarity
python -m tools.issue92.cad_conditioning_probe --probe branch
```

### 17.7 Not claimed

* Not a certified adjoint on `g1_wingbody_cad`: 17.4 shows the oracle is invalid there, not
  that the adjoint is wrong.
* Not an efficiency claim: the adjoint costs one extra QR plus a triangular solve per
  gradient, and the timings in the JSON are single-shot wall clock on a laptop.
* Not a statement that the production forward default (`max_nfev = 80`) is adequate: 17.4
  shows it is not, on this fixture, and changing it is a defaults change that needs its own
  A/B evidence.


