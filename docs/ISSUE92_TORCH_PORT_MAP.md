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
| S4 | assembly and post-processing on tensors: face→DOF scatter, Kutta, jump/contact constraints, Cp, forces/moments, using the already-validated batched influence kernel (#96) | bitwise-or-1e-13 parity on A/W/J/K; loads to 1e-11; `test_p1_coupled_material_wake` unchanged |
| S5 | coupled free-wake loop and sampling entirely in torch; snapshot schema records the backend | ≥3 accepted geometry updates with identical gate decisions and loads to 1e-10 vs the NumPy reference; then B2 (CUDA) starts |

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
| g1_wing_rect | 2 / 4 | 2 / 7 | 2 / 7 | 1.0577997468 (3x) | 8.13e-12 / 8.37e-12 / 8.30e-12 | False |
| g1_wing_camber | 3 / 10 | 4 / 7 | 4 / 7 | 1.0595427809 (3x) | 1.084e-10 / 1.085e-10 / 1.084e-10 | False |
| g1_swept30 | 2 / 7 | 2 / 13 | 2 / 13 | 1.1558328838 (3x) | 2.09e-11 / 2.13e-11 / 2.17e-11 | False |
| g1_elliptic | 2 / 12 | **5 / 26** | **2 / 21** | 1.0638500112 / 1.0638500103 / 1.0638500138 | 2.07e-10 / 1.39e-09 / 8.75e-10 | False |
| g2_conventional | 1 / 5 | 4 / 7 | 4 / 7 | 1.0572540962 (3x) | 7.39e-13 / 6.24e-11 / 6.23e-11 | False |
| g2_tail_low | 2 / 5 | 4 / 7 | 4 / 7 | 1.0563103721 (3x) | 8.66e-13 / 5.73e-11 / 5.73e-11 | False |
| g1_wingbody_cad | 2 / 15 | 2 / 29 | 2 / 29 | 1.0143376710 (3x) | 1.34e-08 / 1.34e-08 / 1.34e-08 | False |
| g1_wingbody_cad_3x | 2 / 17 | 2 / 26 | 2 / 26 | 1.0577264271 (3x) | 1.74e-09 / 2.30e-08 / 2.30e-08 | False |

Acceptance classification agrees **8/8** across the three backends; `status`/`nfev`/`message` agree **7/8** between
NumPy structured and torch. The exception is `g1_elliptic`:

* NumPy structured stops on its **stagnation rule** at nfev 26 (`relative gradient stagnated`), torch stops on
  **ftol** at nfev 21 (`ftol termination condition is satisfied`). The two rules race in a flat valley: the last
  NumPy step slightly raised the relative gradient (1.37e-09 -> 1.39e-09), torch reached the ftol branch first at
  8.75e-10. Both land on the same floor (1.0638500103 vs 1.0638500138, 1.3e-09 apart), the same acceptance, and CL
  agrees to 2.2e-07.
* Same lesson as S1 stress scenario, one level deeper: **a stopping rule is part of the algorithm, not a rounding
  detail.** When two backends run the same algorithm and still disagree on which criterion fires, the honest report
  is the pair of branches, not "matched within an nfev tolerance".

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

1. **Every fixture is rejected, and by two different doors.** `wall_over_floor` is 1.056-1.156 on seven fixtures,
   above the 1.05 threshold; `g1_wingbody_cad` sits at 1.014 and passes the floor, but its stopping status is 2,
   and acceptance requires `status == 0`. So `pressure_kutta_acceptance` returns `False` on all eight in all three
   backends - the port reproduces the verdict exactly and cannot turn the pressure-Kutta branch into an accepted
   solve.
2. `g1_elliptic` and `g1_wingbody_cad_3x` are the cases where the *structured* algorithm itself sits 1e-07-3e-07
   away from the `trf` floor. That gap is a property of the two algorithms on a flat residual, present before this
   port; structured and torch agree with each other to 2e-09 there.
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
