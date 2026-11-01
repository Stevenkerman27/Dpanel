# Streamline step unit, truncation rows, and the residual amplitude (issue #84 / PR #91)

Status: measured, partially resolved. The map-unit step control and the truncation-row
policy are in production code and tested; the **remaining** blocker is the wall-contact
gate on the finite-thickness trailing-edge gap, not the step amplitude. No default,
threshold, or `physical_accuracy_accepted` flag is changed by this document.

## 1. Question

T5 measured that the worst streamline request on the G2 CAD wake was 22.98 chord while the
per-node step cap was 2.31e-3 chord (a 10^4 mismatch), so an accepted update moved the sheet
far less than the map asked for. This unit asks whether that request is an evaluation
artifact (bad velocity/eligibility) or a property of the wake mesh, and what the step
control should be.

## 2. The request is one truncation link, not a velocity artifact

`wake_step_scale` reports, for the G2 CAD main wing (765 nodes, 770 faces):

| quantity | value |
| --- | --- |
| reference link median | 0.6659 chord |
| reference link max | 26.0 chord |
| nodes on a 26-chord link | 102 (51 of them are free nodes on the marcher) |
| shortest incident edge median | 0.0500 chord |
| cos(tangent) median over the sheet | 0.9973 |
| normal-velocity fraction median | 0.0576 |

The map advances exactly one reference link per node, so the request at a linked node is
`link * |u_hat - t_ref|`. A 26-chord link with a 0.6 tangent error asks for ~15-23 chord no
matter how smooth the velocity is; the same sheet's median node asks for
`0.666 * 0.05 = 0.03` chord. The 22.98-chord maximum is therefore a **mesh defect** (one row
of 26-chord panels), not a singularity of the velocity evaluator.

## 3. Three changes

1. **Step unit = map arclength.** `wake_step_scale(xyz, faces, material)` returns
   `scale = max(reference_link, shortest_incident_edge)`; the cap is
   `GeometryGates.max_step_fraction * scale` and the step gate in `geometry_audit` uses the
   same array (`step_scale=` argument). Before this change the cap and the gate both used the
   shortest incident edge, which is one to two orders of magnitude smaller than the map step.
2. **Truncation rows are relative, and reported.** `truncation = known & (link >
   truncation_ratio * median(link > 0))` with `truncation_ratio = 4`. The first version used
   `link > 4 * shortest_incident_edge`; on G2 that flags 332 of 765 nodes, i.e. the normal
   coarse sheet, because a thin triangle's shortest edge (0.05) is 13x smaller than the node
   spacing (0.67). The relative criterion flags 51 nodes (the 26-chord row).
3. **`truncation_policy`.** `streamline_target(..., step_scale, truncation_policy)` with
   `'advance'` (default, legacy) or `'freeze'`; `solve_coupled_material_wake(...)` exposes it
   and every iteration reports `truncation_policy` and `truncation_nodes` in the per-group
   counts. `'freeze'` requires the step scale (a freeze without the scale would silently hold
   the wrong nodes). Freezing is never a silent gate exemption: frozen nodes are reported and
   the map residual is still measured on the moving set.

## 4. A/B evidence (G2 CAD, alpha = 5 deg, kutta = signed_speed, gate_mode = audit)

All runs use `max_backtracks = 12`. `advance` rows come from
`docs/issue84/t5_gate_ab_new_gate_per_node_v5.json`; `freeze` rows from
`docs/issue84/t5_gate_ab_frozen_truncation_v3.json`, `..._frozen_step1.json`, and
`..._frozen_step4.json`.

| run | step unit | cap fraction | policy | iter0 request | last request | accepted updates | status | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | map arclength | 0.25 | advance | 22.982 | 52.313 | 3 | blocked at iter 3 | 59.9 s |
| B | map arclength | 0.25 | freeze | 3.500 | 3.353 | 5 | blocked at iter 5 | 117.8 s |
| C | map arclength | 1.00 | freeze | 3.500 | 3.386 | 2 | blocked at iter 2 | 101.8 s |
| D | map arclength | 4.00 | freeze | 3.500 | 3.231 | 2 | blocked at iter 2 | 108.9 s |

Per-iteration relaxation (the factor the geometry gate finally admitted):

| run | iterations |
| --- | --- |
| A | 2^-6, 2^-8, 2^-11 |
| B | 2^-6, 2^-4, 2^-5, 2^-7, 2^-12 |
| C | 2^-6, 2^-7 |
| D | 2^-6, 2^-9 |

Loads stay inside 0.2 % across B (`CL 0.441355 -> 0.441353`, `CD_p 0.007127 -> 0.007019`): the
updates that the gate accepts are far too small to change the aerodynamics, which is the same
"accepted is not progress" statement as T4/T5, now with the amplitude mismatch removed.

Reading:

* Freezing the truncation rows removes 85 % of the iter-0 request (22.982 -> 3.500) and
  changes the failure mode from `blocked at iteration 0` (T5 with the geometric unit) to five
  accepted updates.
* The cap magnitude is **not** the binding constraint: raising it to 1.0 or 4.0 links makes the
  run block *earlier* (2 accepted updates instead of 5) because a larger candidate trips the
  wall gate sooner and the bounded global fallback then burns the iteration budget.
* The residual decays monotonically but slowly (`3.500 -> 3.353` in five updates, ~0.9 % per
  update), consistent with a damped fixed-point iteration whose admitted relaxation is
  <= 1/16.

## 5. What actually blocks the remaining updates

With the amplitude mismatch gone, every remaining refusal is the wall gate, and the offending
pairs are the finite-thickness trailing-edge faces:

| wake face | wall face | class | depth (chord) | old class |
| --- | --- | --- | --- | --- |
| 300 | 18 | transverse | 9.6e-4 | contact |
| 684 | 18 | transverse | 3.31e-2 | contact |
| 684 | 19 | transverse | 3.21e-2 | contact |
| 685 | 18 | transverse | 9.6e-4 | contact |

The wake sheet leaves the trailing edge at the mid-plane, so the first rows sit *inside* the
gap between the upper and lower TE faces. `geometry_audit` tests the swept hull (old + new
triangle) against every overlapping wall face; any motion of a triangle inside that gap makes
the hull reach the TE face planes, and the protrusion depth is the local TE half-thickness
(here 3.3e-2 chord). The declared-contact corridor only covers points close to the *shared*
simplex, so a triangle that slides off the TE edge inside the gap is a violation even at
2^-12 relaxation. This is a modelling question (the zero-thickness sheet inside a thick TE),
not a threshold that should be relaxed, and it is not fixed here.

## 6. Reproduction

```powershell
$env:PYTHONPATH='D:/3D/Projects/panel/runs/fix_master;D:/3D/Projects/panel/runs;D:/3D/Projects/panel/runs/torch_site'
& D:/3D/Projects/panel/.conda_env/python.exe -m tools.issue84.t5_gate_ab new_gate_per_node_frozen 8 `
    docs/issue84/t5_gate_ab_frozen_truncation_v3.json 0.25
```

The fourth positional argument is `GeometryGates.max_step_fraction` (default 0.25); the driver
passes `truncation_policy='freeze'` whenever the condition name contains `frozen`.

## 7. Open items

1. TE-gap contact rule: allow a wake triangle whose roots are on the TE to occupy the TE
   thickness corridor (the union of the declared simplex and the wall faces incident to that
   edge, with the local thickness as the corridor width), while still refusing motion that
   leaves the gap or overlaps a wall face at positive area.
2. Wake mesh refinement at the trailing edge (replace the 26-chord row with resolved panels),
   which removes the truncation row instead of freezing it.
3. Only then re-run the >= 6-update sequence required by T5-D and report the residual decay
   per update, not per accepted gate step.