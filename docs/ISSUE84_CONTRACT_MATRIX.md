# Issue #84 / PR #85: A–E contract, item by item, with attribution

Audited HEAD **`b9cd4e1`**, branch `research/issue84-rm-regularized-p1-wake`, base `master` at
`91d4d2a1`. Contract: PR #85 comment 6517332531 (2026-10-30T07:25:36Z). The point of this file is
not to claim the unit is finished — it is to make every requested item traceable to a command, an
artifact or a named blocker, so the next revision can be planned from it.

| item | status | evidence | open |
|---|---|---|---|
| A1 `tri_parts` constructor | fixed (cloud `e1c92b0`), now guarded | `tools/issue84/default_path_guard.py` check A1: one-part field builds, `_triangle_area2=[1.0]`, finite velocity | — |
| A2 `constant.grad_face` | fixed, now guarded | guard check A2: 0 occurrences of `grad_face`, 2 of `varying.gamma` in `test/test_p1_regularized_sheet_velocity.py` | — |
| A3 CI covers both files | **blocked by credentials** | contents API PUT 403 `... without 'workflow' scope` | needs Workflows:write or cloud push |
| A4 default path unchanged | verified | guard A4a/A4b/A4c + `git diff 91d4d2a1..HEAD -- panel/p1_analytic_sheet_velocity.py` empty | — |
| B six-class census | done | `tools/issue84/rejection_taxonomy.py`, `docs/issue84/rejection_taxonomy.json` | classes 2–6 are empty by check, see §B |
| C1 scalar vs batched | done | `docs/ISSUE84_RM_BENCH.md` §1: 2.66x, max abs diff 0.0, peaks split per path | — |
| C2 order 48 vs 96 on total vector | done | §3 and `docs/issue84/rm_order_bands.json` | — |
| C3 adaptive / local refinement | studied, negative results, attribution corrected | §3.1, `docs/issue84/sliver_aliasing.json`, `tools/issue84/adaptive_sheet_quad.py` | projection-aware split not implemented |
| C4 872-face replay cost | done | 1483.9 s, peak 2028.7 MB, EXIT 0, order 48, chunk per `rm_bench` | — |
| D five commands on one HEAD | done | §D table below, 2026-10-30 | — |
| E six delivery gates | 5/6 | A (guarded), C, D met; CI gate open; G2 convergence not claimed | CI gate |

## A. Blocking defects

A1 and A2 were fixed by the cloud in `e1c92b0`; the previous receipts only had "the test suite is
green" as evidence, which is not the same claim. `tools/issue84/default_path_guard.py` now pins both
with a reproduction that needs no solve (one-triangle stub, `SimpleNamespace` mesh):

```
PASS  A1 single-part constructor + evaluate      n_triangles: 1   triangle_area2: [1.0]
PASS  A2 regularized test uses a real attribute  grad_face_occurrences: 0   varying_gamma_occurrences: 2
PASS  A4a analytic_edges is the default evaluator  signature_default/instance_method: analytic_edges, has_regularized_attribute: false
PASS  A4b RM still requires an explicit positive core
        TotalVelocity_without_core: ValueError: rm_regularized_p1 requires an explicit positive core_radius
        RegularizedP1SheetField_without_core: TypeError: missing 1 required keyword-only argument: 'core_radius'
PASS  A4c no report claims physical accuracy  files: 47  occurrences: 69  true_occurrences: []
```

A4 "default path unchanged" has three independent legs:

1. the evaluator itself is untouched — `git diff 91d4d2a1..HEAD -- panel/p1_analytic_sheet_velocity.py`
   is empty, so `analytic_edges` values cannot have moved;
2. the default is still `analytic_edges` (signature default and instance state), and the default
   constructor creates no regularized object (`has_regularized_attribute: false`);
3. `rm_regularized_p1` cannot be entered without an explicit positive core, at both levels.

The only modified solver module is `panel/p1_coupled_material_wake.py` (+129/−37 vs `91d4d2a1`):
the 37 removed lines are docstring/comment text, the `method=` signature (now `method=` plus
`core_radius=`), the sampling loop moved into a helper, and the `replay_snapshot` signature. No line of
the analytic evaluation math is deleted.

**A3 remains blocked**: the two list entries live in `.github/workflows/issue77-gamma-support.yml`, and
the contents API PUT returns 403 *"refusing to allow a Personal Access Token to create or update
workflow ... without 'workflow' scope"*. The patch exists only as an uncommitted working-tree change.

## B. The 102 rejected G2 nodes, in the requested six classes

`wake_node_rejection.py` emits exactly one machine token per rejected node, so the census is a
projection of that token, not a second opinion (`tools/issue84/rejection_taxonomy.py`). Reproduction is
cheap and deterministic — 12.9 s for the whole 393-node G2 mesh:

```
python tools/issue84/wake_node_rejection.py --main-half-surface runs/half_cad_wingbody_v1/surface.json \
    --case g2 --alpha 5 --order 48 --velocity-method analytic_edges --eps-fraction 0.05 --epsilon-tol 0.02 \
    --out runs/issue84/wake_rejection_g2_analytic.json --csv runs/issue84/wake_rejection_g2_analytic.csv
```

| class | count | representative nodes | minimal reproduction |
|---|---|---|---|
| sampling/epsilon unreliable | **102** | 56 (eps 0.1934), 452 (0.1931), 626 (0.1349, near tip) | the 12.9 s command above; per-node ladder trace is in the `repro` block of the JSON |
| sampling/quadrature unreliable | 0 | — | n/a (quadrature tolerance never binds on the analytic path) |
| geometric degeneracy | 0 | — | `min local_area_min = 2.212e-03`, `min incident_faces = 2` (a sheet-boundary node, not a degenerate face) |
| collision/self-intersection | 0 | — | `min dist_wake_other = 0.3654`, `min dist_wake_same/scale = 0.742` |
| root/attachment constraint | 0 | — | `rejected_is_root_count = 0` (the 95 root nodes are never the ones rejected) |
| step/line-search failure | 0 | — | not produced by this tool; the Picard step either accepts or aborts, and aborts are reported separately |
| other | 0 | — | `zero_speed = 0`, `opposed_tangent = 0` |

Class 1 splits by where the node sits, which is what a fix has to act on: 29 on the sheet boundary
(39 within one face of it), 14 near the wing tip, 73 strictly interior; `h_over_scale` is pinned at
0.00625 for all of them and the epsilon error is 0.0206 min / 0.0393 median / 0.1934 max. Class 1 is
therefore **one mechanism, not five**: the epsilon estimator averages velocity samples offset along the
sheet normal, and at a sheet boundary or tip the offset falls off the sheet (the wide-h ladder in
`docs/ISSUE84_WAKE_NODE_REJECTION.md` shows the increment does not go to zero with h).

## C. Performance and accuracy

C1–C4 are unchanged from receipt 6519165714 plus the C3 correction in 6520167910, summarised:

- scalar vs batched: 22.59 s vs 8.50 s (2.66x), max abs difference **0.0** (bit identical), same peak;
  order-48 production path peaks at 612.1 MB, the 2105.8 MB peak belongs to the order-96 reference;
- order 48 vs 96 reported on the **total velocity vector**, near/on-sheet/far bands;
- the order sweep is aliasing, and the alias source is now identified: 13.5% of main-wing wake
  triangles exceed 50:1 aspect (max 1405.7) and the wall part is exact to 1.3e-17 at the same point;
- full 872-face replay 1483.9 s (24.7 min), EXIT 0, peak 2028.7 MB, order 48.

## D. One-shot validation on `b9cd4e1`

Environment: Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, pytest 9.1.1, Windows, `.conda_env`.

| command | exit | result | wall |
|---|---|---|---|
| `python -m py_compile` on the four files | 0 | ok | 0.08 s |
| 3 targeted test files | 0 | 70 passed | 2.39 s |
| `test/test_p1_*.py` (20 files) | 0 | 281 passed, 4 skipped | 33.81 s |
| `pytest test/` | 1 | 5 failed, 666 passed, 9 skipped | 139.22 s |
| clean master `91d4d2a1`, `pytest test/` | 1 | 11 failed, 640 passed, 9 skipped | 137.18 s |

Failure attribution: the 6 `test_p1_coupled_material_wake.py` failures on master pass here; the 5 that
remain are 2 `test_matrix_free_symmetry_gate.py` (also failing on master, fixed by PR #86 `2cc95b31`)
and 3 VortexAD tests (also failing on master, sample STL files absent from the environment). **No test
that passes on master fails on this branch.**

## E. Gate status

| gate | status |
|---|---|
| A defects fixed, targeted tests pass | met (guarded by `default_path_guard.py`) |
| CI explicitly covers both new test files | **not met — credential blocker**, patch ready |
| B classification with evidence, no threshold loosening | met |
| C speed/accuracy/memory benchmark | met |
| D on the final HEAD | met |
| no claim of G2 convergence | honoured (`status=iteration_limit`, pressures unchanged) |
