# Master integration: P1 research solver (branch `research/master-integration-p1`)

Status: **prepared, not merged.** `physical_accuracy_accepted=false`. This document is the
audit trail requested in issue #62; it does not claim physical validation.

## 1. master state

- master = `7b806439e6269c4d419e27fc979318b17f7b6468` ("Initial commit", **1 commit, 222 files**).
- **master contains no `panel/` directory at all.** The whole panel solver — the production P0
  half-model path included — exists only on research branches. Integration is therefore not a
  merge of PRs but the introduction of a new package baseline.
- Local worktree of the repository is on master with 24 modified/untracked paths (research
  artefacts under `runs/`, plus `.gitignore`/docs edits); none of them are required by this
  branch.
- Top-level directories on master: `airfoil2d cases core lifting3d potential volume` (+ docs and the
  legacy `aero.py`). None of them is imported by the P1 modules.

## 2. PR dependency DAG (verified from the GitHub API, 2026-10-28)

The P1 stack is a linear chain of *branches*, not of PR heads:

    master 7b806439
      └─ … research/issue49-* (issue #49 baseline, PR #50/#51/#52)
           └─ PR #52  e0ff576f  panel/p1_doublet_moment.py          (P1 kernel oracle)
                └─ PR #57  c1f2bf4a  panel/p1_closed_body.py, p1_closed_jax.py,
                                      run_p1_closed.py                (P1 G0 continuous solve)
                     └─ PR #59  670bd0d7  panel/p1_lifting_g1.py, run_p1_lifting_g1.py (G1)
                          └─ fix 4051ee8  panel/p1_lifting_g1.py  (real TE pairing adapter)
                               └─ PR #60 45518d8  panel/p1_lifting_g2.py, run_p1_lifting_g2.py (G2)
                                    └─ PR #61 313987a  panel/p1_steady_free_wake.py (experimental)
                                         └─ fix 0bffd14  short-edge PV probe + frozen-node guard

Each PR's base is the *previous PR's head branch*, so `mergeable` in the GitHub UI says nothing
about whether the base branch still exists or is current. PR #59's base
(`research/continuous-p1-g0-solve`) is PR #57's head branch; PR #60's base is the local fix branch
`research/p1-g1-te-pairing-fix` (not a PR), and PR #61's base is PR #60's head branch.

**Minimal commits to transplant** (no historical PR is merged; only these objects' file sets
matter): `e0ff576f` (3 files), `c1f2bf4a` (7 files), `670bd0d7` (5 files), `4051ee8` (2 files,
supersedes `670bd0d7:panel/p1_lifting_g1.py`), `45518d8` (5 files), `313987a` (5 files), `0bffd14` (6 files,
supersedes `313987a:panel/p1_steady_free_wake.py` and `panel/run_p1_steady_wake.py`).

Unrelated research PRs (#26, #30, #32, #34, #36, #40, #43, #44, #45, #47, #50, #51, #53, #54,
#55) are **not** dependencies and are not transplanted.

## 3. What this branch contains

From master, plus:

- `panel/` package (43 modules) taken from the current research tip (`0bffd14`), i.e. the P0
  half-model solver (`kernel.py half_native.py pressure_stencil.py solve.py run_half_native.py ...`) that
  master lacks entirely.
- P1 modules transplanted file-by-file from the chain above:
  `p1_doublet_moment.py` (kernel oracle), `p1_closed_body.py`/`p1_closed_jax.py`/`run_p1_closed.py` (G0),
  `p1_lifting_g1.py`/`run_p1_lifting_g1.py` (G1 + real TE pairing adapter), `p1_lifting_g2.py`/
  `run_p1_lifting_g2.py` (G2), `p1_steady_free_wake.py`/`run_p1_steady_wake.py` (experimental free wake).
- Their tests and receipts: `test/test_p1_*.py` (8 files), `docs/P1_*.md`, `docs/P1_*RECEIPT.json`,
  `docs/ISSUE49_P1_DOUBLET_KERNEL_ORACLE.md`, `docs/P1_FREE_WAKE_PROBE_FIX.md`,
  `docs/issue49_vspaero/p1/free_wake_probe.json`, `tools/vsp_evidence/p1/free_wake_probe.py`.

Not included on purpose: the issue #56 audit modules (`p1_solid_angle.py p1_gate_checks.py p1_rank_gauge.py`
`p1_dirichlet.py p1_wake.py p1_g0_acceptance.py p1_g1.py`). They are evidence for the mathematical review
(3-D solid-angle free terms, DOF/gauge audit, G0 acceptance) and stay on
`research/issue49-vspaero-evidence`; the solver does not import them.

Interfaces: every P1 entry point is a `python -m panel.<cli>` module with `--out DIR` (the directory
must not exist), writing `report.json` + per-alpha `*.npz` fields. Library entry points:
`panel.p1_closed_body.solve/velocity_fd`, `panel.p1_lifting_g1.make_wing/from_dpanel_half_surface/solve`,
`panel.p1_lifting_g2.combine_wings/assemble_multi/solve_multi`,
`panel.p1_steady_free_wake.solve_steady_free_wake`.

## 4. Acceptance commands and real exit codes

Environment: `.conda_env` (Python 3.11, JAX 0.10.2 CPU), `TMP=runs/pipwork`,
`PYTHONPATH=<worktree>`; pytest needs `-p dsh_tmp_plugin` + `DSH_TMP_ROOT` on this sandbox.

| command | exit | result |
| --- | --- | --- |
| `pytest test/test_p1_*.py -q -p dsh_tmp_plugin` | **0** | **62 passed / 9.7 s** |
| `python -c 'import panel.kernel'` | **0** | P0 kernel imports |
| `python -m panel.run_half_native --surface runs/half_cad_wingbody_v1/surface.json --mode lu --max-panels 6000` | **0** | P0 half solve, `true_relative_residual = 9.92e-16` |
| `python -m panel.run_p1_closed --levels 1 2 --samples 24` | **0** | G0 sphere; P1 `cp_rms_error` 0.0287 at offset 0.05 (level 2) |
| `python -m panel.run_p1_lifting_g1 --chord-panels 4 --span-panels 6 --alpha 5 --kutta signed_speed` | **0** | CL +0.321120, CL_gamma +0.283773, `kutta_rms` 1.2e-15, dCL/dα 3.635 /rad |
| `python -m panel.run_p1_lifting_g1 --half-surface .../half_cad_wingbody_v1/surface.json --alpha 5 --kutta pressure` | **0** | CL +0.304434, CL_gamma +0.289437, `kutta_pressure_rms` 1.08e-06, `jump_rms` 5.6e-16 |
| `python -m panel.run_p1_lifting_g2 --main-half-surface .../surface.json --tail-span 1.8 --tail-chord .55 --tail-x 1.6 --tail-z .35 --alpha 5 --kutta pressure --reference-area 4.0` | **0** | total CL +0.339459 (main 0.2700, tail 0.0695) |
| `python -m panel.run_p1_steady_wake ... --iterations 4 --relaxation .05 --max-step-chord .01` | **0** | **not converged**: main slip 0.1774 → 0.1954, tail 0.0985 → 0.0875 |

Failure log retained: `runs/fw_head.log` (PR #61 head crashes at iteration 2 with
`ValueError: main_wing: wake streamwise marching requires Vx>0`); analysis in
`docs/P1_FREE_WAKE_PROBE_FIX.md`.

## 5. Unresolved mathematics and accuracy limits

1. **P1 is a collocation/least-squares discretisation, not Galerkin.** Wall rows are panel
   centroids, unknowns are nodal doublet strengths; the system is overdetermined
   (396 rows vs 213 wall nodes on the real half model) and solved by `numpy.linalg.lstsq`.
   Do not call it Galerkin.
2. **Free terms.** `p1_closed_body` uses the analytic `-N_a(x)/2` interior trace for closed bodies;
   the issue #56 audit showed the single-panel `theta/(4*pi)` node coefficient is *not* the 3-D
   solid-angle free term (orthogonal tetrahedron corner: sector 0.375 vs true 0.875). The
   solver only uses the interior-point value and is therefore restricted to smooth interior
   collocation points; corner/edge free terms are an open item, and no gauge is pinned.
3. **Kutta closure is genuinely independent** in PR #59/#60 (`signed_speed` or `pressure` equality at
   the TE, solved as extra constraints with a null-space / nonlinear least-squares step), not
   the definition `mu_up - mu_lo - Gamma = 0`. On the real CAD the two closures differ by up to
   26 % in G2, which is an open physics question, not a convergence issue.
4. **Real CAD blunt TE.** The `dpanel.half_planar_cut.v1` `trailing_edges` field is a TE *edge
   chain*; the upper/lower TE faces share two nodes (sharp TE). PR #59 as shipped paired on the
   edge chain, so its potential-trace split never triggered; the adapter transplanted here
   (`4051ee8`) pairs on `upper_te_faces`/`lower_te_faces` and duplicates the lower trace.
5. **Reference area.** `runs/half_cad_wingbody_v1/surface.json` declares `reference_area = 4.0` while its
   full-aircraft projected planform is 5.556; every CL from that geometry is scaled by
   5.556/4.0 = 1.39 relative to planform-based values. The G1/G2 CLs above use the declared
   area, so they are **not** directly comparable to planform CLs.
6. **Near-field pressure drag is not a well-defined functional of the discrete potential**
   (documented in `docs/ISSUE49_PHASE0_BASELINE.md` and the Trefftz work): CD_pressure changes sign
   between grids while CL converges. Use CL and Trefftz CD, not CD_pressure, for statements.
7. **Free wake has no fixed point** (`P1_FREE_WAKE_PROBE_FIX.md`): bounded but not converged; it stays
   opt-in and must not be described as converged.

## 6. PR disposition proposal

| PR | proposal |
| --- | --- |
| #52 → #57 → #59 → #60 | **absorbed** by this integration branch (file sets + the TE fix). Keep the PRs open as research history; close as superseded only after this branch lands on master, with a link to the merge commit. |
| #61 | **keep open as an experimental branch.** Absorbed only as an opt-in CLI with the probe fix; its result is explicitly non-converged. |
| #30, #54, #50, #51, #55 | **keep as research records** (wake audit, frozen baseline + VSPAERO evidence, spanwise-lift erratum, Trefftz quadrature). Only `panel/spanwise_circulation.py` is a code dependency and it arrives with the panel package. |
| #32, #34, #36, #43, #44, #45, #47, #53 | **stay independent** (pressure recovery, Gamma refinement, feature-size/growth gates, shed groups, affine recovery). None is imported here. |
| #26 | closed as superseded already; no code dependency. |

## 7. Open follow-ups

- Decide the reference-area convention (declared 4.0 vs planform 5.556) before any CL table is
  published from this branch.
- Resolve the sign flip of Gamma in the wing-body junction strip (`|y| < 0.35`) — either mask
  those Kutta rows explicitly or model the junction vortex.
- Replace the frozen-x free wake with a Lagrangian filament wake (x-deformable nodes, analytic
  self term) before any free-wake result is reported.
- Port the issue #56 corner/gauge audit into the solver only after the free-term derivation is
  agreed; do not pin a gauge by hand.
