# G2 wake-update node ledger (issue #84, stage B)

This document answers the stage-B question from the reviewer: what do the B2
counters `root`, `eligible`, `convergence`, `speed` and `rejected` actually
count, where are they incremented, and do `convergence` and the sampling
rejection overlap. The answer is derived from the solver source and from a
re-run of the G2 case with a save callback that captures the arrays the
solver stores (`tools/issue84/wake_ledger.py`).

## 1. Counter definitions (source facts)

All five counters are computed once per iteration inside the per-group loop at
`panel/p1_coupled_material_wake.py:561-564`:

```python
counts[group.name] = dict(root=len(mat.roots), eligible=int(eligible.sum()),
                          convergence=int((~finite_limit).sum()),
                          speed=int((finite_limit & ~fast).sum())+tangent_rejections,
                          rejected=int((~eligible[free_nodes]).sum()))
```

with the gate inputs one line above (`panel/p1_coupled_material_wake.py:543-547`):

```python
finite_limit = ((samples['epsilon_error'] <= epsilon_tol)
                & (samples['quadrature_error'] <= quadrature_tol))
fast = np.linalg.norm(samples['mean'], axis=1) > 1e-6
eligible[free_nodes] = finite_limit & fast
```

| counter | definition | incremented at |
| --- | --- | --- |
| `root` | `len(material.roots)`; roots are pinned to `reference_xyz` and never sampled | `:561` |
| `eligible` | free nodes passing epsilon, quadrature, speed and tangent gates | `:543-556`, summed `:561` |
| `convergence` | free nodes with `epsilon_error > epsilon_tol` **or** `quadrature_error > quadrature_tol` | `:562` |
| `speed` | free nodes resolving a sample with `|v| <= 1e-6`, plus opposed-tangent rejections | `:545-555`, `:563` |
| `rejected` | free nodes not eligible, i.e. `convergence + speed + tangent` | `:564` |

Consequences that the reviewer asked to state explicitly:

* `convergence` is **not** an iteration-convergence counter. It is the
  *sampling* rejection set of this iteration (finite-difference epsilon error
  and quadrature-refinement error against their tolerances).
* `rejected ⊇ convergence` by construction, and `speed` is disjoint from
  `convergence` because the `fast` mask is only evaluated on `finite_limit`.
* `roots` are excluded from all four free-node counters; `total_nodes` in the
  ledger is much larger than `root + free` because a wake sheet carries unused
  duplicate nodes.

## 2. Reproduce

```
python tools/issue84/wake_ledger.py \
  --main-half-surface runs/half_cad_wingbody_v1/surface.json \
  --case g2 --alpha 5 --kutta pressure --iterations 1 \
  --out runs/issue84/wake_ledger_g2_pressure.json

python tools/issue84/wake_ledger.py \
  --main-half-surface runs/half_cad_wingbody_v1/surface.json \
  --case g2 --alpha 5 --kutta signed_speed --iterations 2 \
  --out runs/issue84/wake_ledger_g2_signed.json
```

Every row carries `group`, `node`, `predecessor`, `is_root`, `is_free`,
`eligible`, `epsilon_error`, `quadrature_error`, `epsilon_ok`, `speed`,
`tangent_rejected`, `xyz`, `velocity`, `target`, `displacement`,
`incident_faces`, `incident_area_min`, `incident_aspect_max` and
`wall_distance` (exact point-to-wall-triangle distance, validated against
analytic cases). The per-iteration `sets` block lists the actual node IDs of
`roots`, `free`, `convergence`, `speed`, `tangent`, `rejected` and `eligible`,
and `intersections` reports their pairwise overlaps.

## 3. G2 with pressure Kutta, iteration 0 (the frozen geometry)

`status = blocked_initial_nonlinear_solve`, 872 wall triangles, 14.8 s.

| group | total nodes | root | free | eligible | convergence | speed | rejected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main_wing | 765 | 90 | 378 | 276 | 102 | 0 | 102 |
| raised_tail | 20 | 5 | 15 | 15 | 0 | 0 | 0 |

Set algebra from the ledger:

* `convergence ∩ rejected = 102`, `rejected - convergence = 0`,
  `convergence - rejected = 0` -> the two counters are the **same set**.
* `roots ∩ convergence = 0`, `convergence ∩ speed = 0`, `speed = 0`,
  `tangent = 0`.
* Therefore `rejected = 102` is entirely epsilon/quadrature sampling rejection,
  and `eligible = 378 - 102 = 276`.

The 102 nodes are the same node set that the analytic-edge diagnostic
(`tools/issue84/wake_node_rejection.py --velocity-method analytic_edges`)
classifies as `sampling_epsilon` rejects; the count coincidence is exact and is
confirmed by the ID lists, not just by the totals. They are **not** the 7
quadrature rejects of the RM path, which are a different screen produced by a
different velocity evaluator.

Epsilon error over the 378 free nodes: median 0.01036, maximum 0.19338
(node 56). Representative rejected nodes: 56 (0.1941, sheet-boundary node,
`incident_faces = 2`), 452 (0.1938, the mirror node of the same feature), 626
(0.1478, wing tip), 407/622/623/624/625 (0.1478, an unresolved cluster).

## 4. G2 with signed-speed Kutta, two real geometry updates

`status = iteration_limit`, 33.3 s, `converged = False`.

| iteration | root | free | eligible | convergence | rejected | overlap | new | healed | map residual / chord | eps median | eps max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 90 | 378 | 295 | 83 | 83 | 83 | - | - | 4.8648 | 0.00757 | 0.19411 |
| 1 | 90 | 378 | 251 | 127 | 127 | 127 | 44 | 0 | 1.2894 | 0.01109 | 20.26964 |
| 2 | 90 | 378 | 248 | 130 | 130 | 130 | 3 | 0 | 0.7668 | 0.01109 | 20.05682 |

`overlap` is `convergence ∩ rejected`; `new` and `healed` are the set
differences against the previous iteration. Both updates are accepted, so this
is a real geometry update and not a dry run:

| iteration | accepted | relaxation | moved nodes | `displacement_max` | `displacement_rms` | backtrack reasons |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | yes | 0.0005859375 | 310 | 0.002850 | 0.000371 | 8 x `step` |
| 1 | yes | 0.0046875 | 266 | 0.006044 | 0.000743 | 5 x `step` |
| 2 | not attempted (`iterations=2`) | - | - | - | - | - |

`displacement_max` is the accepted, relaxed displacement; the raw Picard
candidate `|target - xyz|` is reported per node and reaches 4.8648 chord at
iteration 0 (median 0.0568 over free nodes), which is what forces the
relaxation down to 5.9e-4 by `step` backtracks. The root constraint is visible
as `reason_counts['root'] = 95` (90 main-wing roots plus 5 tail roots), and
`area`, `fold`, `wall`, `intersection`, `nonfinite` and `attachment` never
fire.

Per-node `wall_distance` at iteration 0 over the 378 free main-wing nodes: 11
nodes are exactly on the wall (the attached root-line nodes, IDs 0, 1, 4, 56,
67, 74, 400, 402, 452, 458, 465), median 0.790, maximum 28.873 chord.

## 5. What the ledger shows about the update machine

* Sampling rejection is monotone under the update: 83 -> 127 -> 130, and
  nothing heals (`healed = 0` at both steps). The new rejects (44 at step 1, 3
  at step 2) are nodes whose epsilon error explodes after the sheet moves
  (node 399: 0.1478 -> 20.2696; node 377: 0.1478 -> 18.4761), not nodes that
  were previously fine for another reason.
* The rejected set is dominated by sheet-boundary nodes: with
  `incident_faces < 6` as the boundary proxy, 37 of 83 rejects are boundary at
  iteration 0, 81 of 127 at iteration 1 and 84 of 130 at iteration 2. Interior
  rejects stay constant at 46, i.e. the update neither fixes nor creates
  interior rejects.
* The Picard map is the cost driver, not the gates: `map_residual_chord` falls
  4.86 -> 1.29 -> 0.77 while the accepted displacement stays below 0.007 chord
  because of `step` backtracks.

## 6. Open items

* `status = iteration_limit` with `converged = False`: G2 is not converged and
  nothing here claims otherwise. `physical_accuracy_accepted` stays `false`.
* The ledger is written for the default `analytic_edges` evaluator. Running it
  with `--velocity-method rm_regularized_p1` requires the RM parity decision of
  stage A, and the 7 quadrature rejects would then appear as an additional
  rejection class in the same ledger.
* Root/attached/free classification is reported per node (`is_root`, `is_free`,
  `predecessor`); the constraint reasons are only available in aggregate
  (`reason_counts`), because the solver's `geometry_failure` returns the first
  failing check as a string rather than a per-check map. Adding that map is a
  solver change and is deliberately left out of this diagnostic stage.

