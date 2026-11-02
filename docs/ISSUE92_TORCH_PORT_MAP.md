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
| S1 | `p1_pressure_kutta_reduced.py`: `qr`, `solve`, `solve_triangular`, `OptimizeResult` → torch | A/W/J/K and `z` vectors to 1e-12 relative; identical `status`/`nfev` on all fixtures that exercise the reduced path; 10 existing tests still pass |
| S2 | rank-aware constrained solve in G1/G2: `null_space`, `lstsq` → torch | identical `rank` and null-space dimension on all 8 fixed fixtures **plus** the §3 gap report; `mu`/`Gamma`/`CL`/`CD_p`/`CM_y` to 1e-11 relative |
| S3 | nonlinear solve: `least_squares` → torch trust-region (G1 :506, G2 :304) | identical `status`, `success`, `nfev` (or documented ±1), identical floor behaviour, loads to 1e-10 |
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
