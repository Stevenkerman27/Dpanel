# ISSUE84 / PR #85 item C — RM-P1 cost, near-sheet accuracy and gate channels

Status: research measurement, 2026-10-30. `physical_accuracy_accepted` stays **false**,
the default `velocity_method` stays **analytic_edges**, no threshold was relaxed to turn a
rejected wake node into an eligible one, and **no G2 free-wake convergence is claimed**.

All numbers below come from the frozen CAD G2 state of `runs/issue84/g2_ref/iteration_000.npz`
(`state_sha256 = 83d301192e341b5abfe8358caf15e0d3b1088d0696a61b1bc3632aadb8279694`), core
`0.01`, ladder `(1, .5, .25, .125)`, `eps_fraction = 0.05`, tolerances `0.02 / 0.002`.
Machine: Windows 10, Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, single process (no BLAS
thread pinning unless stated).

## 1. Scalar vs batched evaluation

`tools/issue84/rm_bench.py`, 104 points (24 near-sheet, 40 on-sheet, 40 far), 1666 sheet
panels, order 48, frozen state above.

| quantity | value |
|---|---|
| frozen snapshot bit-identical | **True** |
| batched (`evaluate_many`, chunk 2) | 8.50 s → **81.7 ms/point** |
| scalar loop (`field(point)`)` | 22.59 s → **217.2 ms/point** |
| speed-up | **2.66x** |
| max abs difference scalar vs batched | **0.0** (bit-identical) |
| order-96 reference batch | 33.05 s |
| peak RSS after solve | 158.1 MB |
| peak RSS after order-48 batch | **612.1 MB** |
| peak RSS after order-48 scalar loop | 612.1 MB (no further growth) |
| peak RSS after order-96 reference | 2105.8 MB |

Two facts worth separating:

* Scalar and batched agree **bit for bit**, and scalar does not need more memory than batched
  (612.1 MB after both). The batched path is therefore a pure cost reduction: 2.7x at identical
  output and identical peak.
* The 2.1 GB peak reported by the script belongs to the **order-96 reference**, not to the
  production order-48 path: order 96 allocates 4x the quadrature points per panel
  (158 MB → 612 MB at order 48 → 2106 MB at order 96). Any statement that RM-P1 costs 2 GB at
  order 48 would be wrong.

A concurrent run of the same script (while the full-mesh gate job was running) gave
11.03 s vs 23.09 s = 2.09x, i.e. the ratio is what transfers, the absolute seconds are not.

## 2. Frozen replay cost (872 wall faces, 1666 panels total)

`python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface
runs/half_cad_wingbody_v1/surface.json --iterations 0 --replay-snapshot
runs/issue84/g2_ref/iteration_000.npz --expected-state-sha 83d30119… --replay-velocity-method
rm_regularized_p1 --replay-core-radius 0.01 --replay-quadrature-order 48`

| item | value |
|---|---|
| wall faces / wake faces | 872 / 770 + 24 |
| evaluated method / core / order | rm_regularized_p1 / 0.01 / 48 |
| recorded method | analytic_edges |
| replay mode | `alternate_velocity_on_frozen_state` |
| state SHA check | matches expected |
| wall clock | **1483.9 s (24.7 min)**, EXIT 0 |
| peak RSS | **2028.7 MB** |
| chunk size | 2 (class default) |

Diagnostic deltas against the recorded analytic_edges samples (max abs per field):

| field | max abs difference |
|---|---|
| g0 node epsilon_error | 0.192392 (analytic 0.1934 → RM ~1.0e-3) |
| g0 node quadrature_error | **4.307e-3** (recorded 0 = NOT APPLICABLE) |
| g0 face quadrature_error | 0.332365 |
| g0 node plus/minus | 445.677 (analytic singular value at the sheet) |
| g0 face mean | 0.128737 |
| g1 node/face quadrature_error | 4.96e-9 / 2.39e-5 |

The RM path collapses the epsilon channel by roughly **200x** at the worst node, and in exchange
switches on the order-doubling channel, whose worst on-sheet value is **4.31e-3**, i.e. above the
production `quadrature_tol = 2e-3`.

## 3. Is any quadrature order certified near the sheet? No.

`panel/p1_regularized_sheet_velocity.py:135-150` builds the area term from a **fixed
tensor-product Gauss–Legendre rule of order n** mapped by a Duffy collapse of **one vertex**
(`weights = w[:,None]*w[None,:]*(1-u)`, `bary = ((1-u)(1-v), u, (1-u)v)`). There is no grading or
subdivision toward the projection of the field point, so for a panel of size L and a peak width
`max(core, d)` the rule cannot resolve the integrand once `L / max(core, d)` is large. The
observed order behaviour is therefore **aliasing, not convergence**:

`runs/issue84/rm_order_bands.py`, 12 points per band, reference order 192, max |Δv|/U:

| r/core | order 8 | 16 | 32 | 48 | 96 |
|---|---|---|---|---|---|
| 0.3 | 6.87e-2 | 5.03e-2 | **3.09e-1** | 3.61e-2 | 7.79e-2 |
| 1 | 6.66e-2 | 4.74e-2 | **2.89e-1** | 3.32e-2 | 7.42e-2 |
| 3 | 5.38e-2 | 3.67e-2 | **1.82e-1** | 2.35e-2 | 4.70e-2 |
| 10 | 2.54e-2 | 1.39e-2 | 3.88e-2 | 6.58e-3 | 7.66e-3 |
| 30 | 2.47e-2 | 3.45e-3 | 3.49e-3 | 1.86e-4 | 1.00e-4 |

`runs/issue84/rm_bench.py` gives the same picture against its own order-96 reference:
near band `max rel` 1.04e-1 / 1.16e-1 / 1.99e-1 / 1.02e-1 for orders 8 / 16 / 32 / 48; on-sheet
nodes 1.91e-1 / 1.22e-1 / 3.96e-2 / 2.93e-3; far band **5.41e-3 / 2.24e-4 / 8.71e-7 / 2.63e-9**
(monotone, i.e. honest there).

Consequence: near the sheet (r/core ≲ 3) **no order between 8 and 192 is monotone**, order 32 is
the worst of the set, and a single order-doubling comparison cannot distinguish convergence from
aliasing — both orders can alias.

Raising the global order is therefore not an acceptable fix, and **lowering it is clearly wrong**:
at r/core = 30, order 16 is 3.45e-3 while order 48 is 1.86e-4 (17x worse).

## 4. Far field is correct (independent check)

`runs/issue84/rm_far_check.py` compares the regularized field with the **unregularized
analytic_edges** evaluation of the *same frozen strengths*, on random points at least 0.5 / 1.0 /
3.0 from every sheet panel centroid (12 points each):

| minimum distance to any sheet panel | order 48 | order 96 |
|---|---|---|
| 0.5 | 8.43e-07 | 8.43e-07 |
| 1.0 | 1.57e-06 | 1.57e-06 |
| 3.0 | 7.01e-07 | 7.01e-07 |

Order-independent agreement at the 1e-6 level: the core and the area quadrature are both
irrelevant far away, and the implementation is right where it is supposed to be right.
(A previous attempt that offset points along the face normal by `r*core` was discarded: those
points fall inside the wing/tail, and the resulting 2e-2 numbers measured geometry, not quadrature.)

## 5. What the production gate actually sees with RM

`limits` (`panel/p1_coupled_material_wake.py:151-214`) reports `epsilon_error` (change of the ±h
mean along the ladder) and `quadrature_error` (order vs 2·order at the smallest h; exactly 0 for
analytic_edges, explicitly labelled NOT APPLICABLE).

Sample of 20 nodes (`runs/issue84/rm_gate_channels.py`), 12 previously rejected by the analytic
path plus 8 previously eligible:

| sample | order | epsilon max (fail/12 resp. /8) | quadrature max (fail) |
|---|---|---|---|
| analytic-rejected | 16 | 9.85e-4 (0) | **1.245e-1 (4)** |
| analytic-rejected | 48 | 9.85e-4 (0) | **4.308e-3 (2)** |
| analytic-rejected | 96 | 1.148e-3 (0) | 6.116e-4 (0) |
| analytic-eligible | 16 | 1.68e-4 (0) | **2.284e-2 (2)** |
| analytic-eligible | 48 | 1.68e-4 (0) | 6.572e-4 (0) |
| analytic-eligible | 96 | 1.68e-4 (0) | 9.016e-5 (0) |

So the regularization does what it was meant to do (epsilon channel ≤ 1.15e-3 everywhere, versus
0.1934 before), and the **order-doubling channel becomes the binding constraint**: at the
production order 48 it still exceeds `2e-3` on 2 of the 12 worst nodes and clears only at 96,
at 4x the cost (16.7 s vs 66.1 s for those 12 nodes, i.e. 1.39 s vs 5.51 s per node).

Full mesh (`--case g2 --order 48 --velocity-method rm_regularized_p1 --core-radius 0.01`,
552.2 s, peak RSS 2107.0 MB, EXIT 0), compared with the analytic_edges run of item B
(`docs/ISSUE84_WAKE_NODE_REJECTION.md`):

| quantity | analytic_edges | rm_regularized_p1 @ 48 |
|---|---|---|
| main_wing free nodes | 378 | 378 |
| eligible | 276 | **371** |
| rejected: sampling epsilon | **102** | **0** |
| rejected: sampling quadrature | 0 (not applicable) | **7** |
| epsilon_error max / median | 1.934e-1 / 3.93e-2 | 1.642e-3 / 2.34e-5 |
| quadrature_error max | n/a | 4.308e-3 |
| raised_tail eligible | 15 / 15 | 15 / 15 |
| total rejected | 102 | 7 |

The 7 remaining rejections are a different animal from the 102: 4 sit on the sheet boundary,
4 are near the wingtip, 6 are within 1.5 face sizes of a sheet edge, the smallest h/scale is
0.00625, and the minimum distance to the wall is 2.87 — i.e. **none of them is a wall-contact,
root or collision problem**. They are the near-sheet quadrature aliasing of §3.

Read together with §1-§3: the regularizer **does** remove the sampling-ladder divergence of the
analytic edge kernel (`ladder_probe` in item B: constant increments, no limit) and replaces it with
a bounded but order-dependent near-sheet error. That is progress, not certification.


## 6. Required actions

1. **Do not lower the quadrature order.** Order 16 fails the gate on both samples and is 17x worse
   than 48 even in the far field.
2. **Do not accept an order-doubling pass as a convergence certificate.** Because the rule aliases,
   a pass at (48, 96) is necessary, not sufficient. If the RM path is to be used for production
   gates, record at least a three-order sequence (e.g. 48/96/192) per node, or implement the
   refinement below.
3. **The real fix is graded/local integration**: split the Duffy map at the projection of the field
   point (or subdivide the nearest panels) so that the peak `max(core, d)` is resolved. That is a
   research item and is **not implemented here**; until it exists, RM near-sheet values carry an
   order-dependent error of 1e-2…1e-1·U inside r/core ≲ 3 and must be reported that way.
4. The epsilon channel is no longer the reason G2 is blocked; the remaining G2 blockers are the
   eligibility bookkeeping of #65 Phase B2 (`root 90 / eligible 276 / convergence 102 / rejected 102`)
   and, with RM, this quadrature channel.

## 7. Artifacts

| file | content |
|---|---|
| `tools/issue84/wake_node_rejection.py` | per-node diagnosis; now `--velocity-method {analytic_edges,rm_regularized_p1}` + `--core-radius` |
| `tools/issue84/ladder_probe.py` | wide-h scan that established the log divergence of the analytic path |
| `tools/issue84/` … | see `docs/ISSUE84_WAKE_NODE_REJECTION.md` for item B |
| `runs/issue84/rm_bench.py`, `rm_bench.json` | scalar vs batched, order sweep, per-band |
| `runs/issue84/rm_order_bands.py`, `rm_order_bands.json` | order validity vs distance (reference 192) |
| `runs/issue84/rm_far_check.py`, `rm_far_check.json` | RM vs analytic_edges far field |
| `runs/issue84/rm_gate_channels.py`, `rm_gate_channels.json` | gate channels at orders 16/48/96 |
| `runs/issue84/g2_replay_c48b/` | frozen replay receipt + replay.json |
| `runs/issue84/wake_rejection_g2_rm48.json`, `.csv` | full-mesh gate counts with RM at order 48 |
