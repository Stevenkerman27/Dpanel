# Streamline step unit, trailing-edge contact, and the residual amplitude (issue #84 / PR #91)

Status: revised 2026-11-01. Two blockers were removed and one was re-characterised.

* The per-node step cap is expressed in **mesh resolution** (shortest incident edge). The
  map-unit experiment is refuted by the A/B in section 5.
* The wall gate no longer convicts a zero-thickness wake sheet that slides inside the
  finite-thickness trailing-edge gap (section 4). The false positive was a **global depth
  verdict** applied to a per-point corridor test. An additional "enclosure certificate" was
  implemented, checked against an exact reference and **removed** because its distance helper
  over-estimated in 7 of 200 cases, i.e. it could excuse real penetrations.
* What remains is not a gate artefact: seven consecutive updates are accepted with zero wall
  violations, but the loads still drift (`CL 0.441355 -> 0.422972`) while the residual
  plateaus, i.e. accepted updates are not convergent updates (section 6).

No default threshold is relaxed, no production default is switched, and
`physical_accuracy_accepted` stays `false`.

## 1. Question

T5 measured that the worst streamline request on the G2 CAD wake was 22.98 chord while the
per-node step cap was 2.31e-3 chord (a 10^4 mismatch), so an accepted update moved the sheet
far less than the map asked for. This unit asks whether that request is an evaluation
artifact (bad velocity/eligibility) or a property of the wake mesh, what the step control
should be, and why the wall gate refused updates whose entire motion stayed inside the TE gap.

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

## 3. Step control

### 3.1 The cap is in mesh resolution, not in map arclength (revised)

`wake_step_scale(xyz, faces, material)` returns `scale = shortest incident edge`
(`geometric`); the cap is `GeometryGates.max_step_fraction * scale` and the step gate in
`geometry_audit` uses the same array (`step_scale=` argument). The reference link is returned
separately (`link`, `link_median`, `truncation`) and is never used as the cap.

Why the earlier "map arclength" version was wrong: a cap only bounds **strain**, and strain is
set by the panel size. On the G2 CAD wake `0.25 * 26 chord = 6.5 chord` per node is not a cap
at all. Measured (`docs/issue84/t5_gate_ab_contact_v8.json`, map-unit caps): three updates are
accepted and the fourth is refused by the **area/fold** gate, because the admitted 6.5-chord
node motions fold the sheet (`cap_min = 0` for the nodes whose panels are shortest). With the
geometric cap the same sequence passes the geometry gate eight times in a row (section 5).

### 3.2 Truncation rows are relative, and reported

`truncation = known & (link > truncation_ratio * median(link > 0))` with
`truncation_ratio = 4`. The first version used `link > 4 * shortest_incident_edge`; on G2 that
flags 332 of 765 nodes, i.e. the normal coarse sheet, because a thin triangle's shortest edge
(0.05) is 13x smaller than the node spacing (0.67). The relative criterion flags 51 nodes (the
26-chord row).

### 3.3 `truncation_policy`

`streamline_target(..., step_scale, truncation_policy)` with `'advance'` (default, legacy) or
`'freeze'`; `solve_coupled_material_wake(...)` exposes it and every iteration reports
`truncation_policy` and `truncation_nodes` in the per-group counts. `'freeze'` requires the
step scale (a freeze without the scale would silently hold the wrong nodes) and never hides a
node: frozen rows are reported and the residual is still measured on the moving set.

## 4. Trailing-edge contact: root cause and fix

The T5 blocker was `wake face 299/684/685` against the root-chord TE faces `18/19/415` at depth
3.3e-2 chord, persisting down to `relaxation = 2^-12`. Two real defects produced the earlier
false convictions, and one further "fix" was implemented, tested against a reference and then
**removed** because it was unsafe.

1. **A global depth verdict.** `_classify_set` computed one `depth = -min(signed distance)`
   over the whole swept hull and then applied that number to the **corridor** test
   (`beyond = distance_to_declared_simplex > graze_tol | depth > graze_tol`). One deep vertex
   anywhere on the hull therefore convicted every point that merely grazed the declared simplex.
   Fix: `own_depth = -(offending - wall_tri[0]) @ normal` per point, so depth is judged where
   the point actually is. This is the direct cause of the TE false positive: the TE faces are
   slanted (`normal (0.555, +/-0.271, +/-0.786)`), so a wake node on the mid-plane is 3.3e-2
   chord behind their plane while its crossing point sits exactly on the shared edge.
   Regression test: `test_depth_is_judged_per_point_not_over_the_whole_hull` builds exactly that
   configuration (hull depth 0.6, crossing point 0.018 from the shared edge, `graze_tol = 0.05`)
   and asserts `contact` with no violation, `transverse` with `graze_tol = 0`, and a violation
   when the deep vertex projects inside the wall footprint.
2. **A hull AABB early exit** (already fixed in T5) that returned `disjoint` for a triangle
   fully behind the wall plane.
3. **The enclosure certificate was tried and rejected.** A point inside a closed body is at most
   its own depth away from the boundary, so `_surface_distance(point, wall_triangles) <= depth`
   looked like a sound way to dismiss "behind the plane but not on the surface" sliver hits. The
   helper was checked against an exact point-triangle reference (3 vertex distances, 3 segment
   distances, interior projection): **7 of 200 random cases disagree and the error is always an
   over-estimate** (e.g. `1.945593` vs `1.425165`, `3.052988` vs `1.470044`). An over-estimate
   is the dangerous direction: it certifies a point as "outside the body" and lets a real
   penetration through. The certificate, `_surface_distance` and the `wall_surface=` plumbing
   were therefore deleted. Its measured effect was to remove the two persistent pairs at
   iteration 7 and to admit a four-times larger step at iteration 0; with it the run produced
   `CL 0.441355 -> 0.367831` (a -16.6 % drift that hid behind an over-optimistic clearance
   test), which is not evidence of anything except a bug.
4. **What remains is a modelling question.** In the final runs the only blocking pair at
   iteration 7 is `wake face 299` against `wall 19` (declared edge, depth 5.56e-4) and
   `wall 415` (declared vertex, depth 3.33e-2), both with `old_kind = disjoint`; the sheet
   already sits 3.33e-2 chord behind the slanted TE plane (`old_depth 3.3257e-2`) and the
   update moves a crossing point into that sliver's footprint. The depth saturates: at
   `relaxation = 2^-12` it is still 3.33e-2, so this is not a step-size artefact and shrinking
   the step cannot clear it. A zero-thickness sheet inside a finite-thickness trailing-edge gap
   needs either a resolved wake mesh at the TE (section 8) or an explicit, auditable declaration
   that covers the TE thickness — not a geometric heuristic that silently excuses crossings.

Tests: `test/test_p1_geometry_contact.py` (12 + 2) and `test/test_p1_gate_contact.py` (13 + 2)
cover the declared vertex/edge corridors, the two-sided graze tolerance, "a deeper or new
crossing still blocks", the inherited-crossing attribution, the per-point depth rule and the
sampling semantics of `audit` vs `safe`; `test/test_wake_step_scale.py` (8) covers the geometric
cap, the relative truncation criterion, root pinning, freeze and the solver-level reporting.
Those three files: 35 passed in 0.74 s.

## 5. A/B evidence (G2 CAD, alpha = 5 deg, kutta = signed_speed, gate_mode = audit)

All runs: `max_backtracks = 12`, `max_step_fraction = 0.25`, 8 iterations, geometric cap.

| run | gate | policy | iter0 request | last request | accepted | status | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | boolean SAT wall test, no declaration | advance | 22.9815 | - | 0 | blocked at iter 0 | 27.9 s |
| new | declaration + per-point depth | advance | 22.9815 | 52.7423 | 7 | blocked at iter 7 | 277.3 s |
| new | declaration + per-point depth | freeze | 3.4997 | 3.2523 | 7 | blocked at iter 7 | 262.5 s |

Files: `docs/issue84/t5_gate_ab_final_{advance,frozen,legacy}.json`. The legacy gate refuses
iteration 0 fourteen times in a row on a single coplanar SAT overlap
(`wake face 293` / `wall 13`, a **declared** contact pair, `kinds sat_overlap 1`) — the false
positive T5 reported. The new gate admits seven consecutive updates with **zero** wall
violations and zero pre-existing violations, and blocks at iteration 7 on the two pairs of
section 4.4.

Relaxation admitted per iteration (new gate): `2^-6, 1, 1/4, 1/8, 2^-6, 2^-11, 2^-12`;
`dmax` 1.75e-3 / 1.05e-1 / 7.39e-3 / 3.80e-3 / 6.87e-4 / 2.22e-5 / 1.11e-5 chord.
Per-node clip: 342 of 765 nodes clipped in `advance`, 291 in `freeze`, `accepted_max`
2.92e-2 chord against a raw request of 52.74 / 3.25 chord.

Historical map-unit A/B (superseded, kept for the record; `advance` from
`t5_gate_ab_new_gate_per_node_v5.json`, `freeze` from `_frozen_truncation_v3.json`,
`_frozen_step1.json`, `_frozen_step4.json`):

| run | step unit | cap fraction | policy | iter0 request | last request | accepted | status | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | map arclength | 0.25 | advance | 22.982 | 52.313 | 3 | blocked at iter 3 | 59.9 s |
| B | map arclength | 0.25 | freeze | 3.500 | 3.353 | 5 | blocked at iter 5 | 117.8 s |
| C | map arclength | 1.00 | freeze | 3.500 | 3.386 | 2 | blocked at iter 2 | 101.8 s |
| D | map arclength | 4.00 | freeze | 3.500 | 3.231 | 2 | blocked at iter 2 | 108.9 s |

That table is also where the "cap magnitude is not the binding constraint" claim came from; it
is true for the map-unit caps only, and it is the reason those caps were abandoned.

## 6. Accepted is still not converged

Loads and residual over the seven admitted updates (new gate):

| iter | residual advance | residual freeze | CL | CD_p | relaxation |
| --- | --- | --- | --- | --- | --- |
| 0 | 22.9815 | 3.4997 | 0.4413554 | 0.00712689 | 2^-6 |
| 1 | 52.0151 | 3.4988 | 0.4414617 | 0.00717362 | 1 |
| 2 | 52.7325 | 3.3038 | 0.4379046 | 0.00716311 | 1/4 |
| 3 | 52.8576 | 3.2770 | 0.4317953 | 0.00639392 | 1/8 |
| 4 | 52.7486 | 3.2552 | 0.4240592 | 0.00565256 | 2^-6 |
| 5 | 52.7421 | 3.2525 | 0.4230206 | 0.00555761 | 2^-11 |
| 6 | 52.7422 | 3.2524 | 0.4229881 | 0.00555465 | 2^-12 |
| 7 (rejected) | 52.7423 | 3.2523 | 0.4229718 | 0.00555317 | - |

The `freeze` CL column is identical to the `advance` column to 8e-7 relative over the whole
run, so the residual difference (52.74 vs 3.25) is a reporting artefact of the 26-chord rows,
not physics.

Reading:

* **Freezing the truncation rows is the honest residual metric.** The residual drops from 22.98
  to 3.50 at iteration 0 and then moves by 1.5 % per update, instead of growing to 52.7. The
  26-chord rows are unattached downstream panels; their contribution to the loads is below
  1e-6 relative, so they inflate the residual without changing the physics.
* **The loads still drift.** `CL 0.441355 -> 0.422972` (-4.2 %), `CD_p 0.007127 -> 0.005553`
  (-22 %), and the drift is still live when the gate blocks: iteration 6 is an accepted
  2.4e-4-chord update that moves CL by 3.3e-5, so the last three updates are converged in
  displacement but not in load. The big drop is at iterations 3-4 (1.5e-3 and 6.9e-4 chord of
  motion buying 4 % and 2 % of CL): the load is sensitive to the wake geometry, and a
  fixed-point map that admits a step per iteration is not a contracting iteration.
* **The residual cannot be driven to zero with the current wake mesh.** `freeze` measures
  3.25 chord of remaining request on a sheet whose median request is 0.03 chord; the remaining
  3.25 comes from the coarse panels and the TE-gap pairs, not from a velocity error.

## 7. Reproduction

```powershell
$env:PYTHONPATH='D:/3D/Projects/panel/runs/fix_master;D:/3D/Projects/panel/runs;D:/3D/Projects/panel/runs/torch_site'
$py='D:/3D/Projects/panel/.conda_env/python.exe'
& $py tools/issue84/t5_gate_ab.py new_gate_per_node        8 docs/issue84/t5_gate_ab_final_advance.json 0.25
& $py tools/issue84/t5_gate_ab.py new_gate_per_node_frozen 8 docs/issue84/t5_gate_ab_final_frozen.json  0.25
& $py tools/issue84/t5_gate_ab.py legacy_gate_per_node     8 docs/issue84/t5_gate_ab_final_legacy.json  0.25
```

The fourth positional argument is `GeometryGates.max_step_fraction` (default 0.25); the driver
passes `truncation_policy='freeze'` whenever the condition name contains `frozen`, and
`_geometric` / `_nocert` suffixes are no longer needed (the geometric cap and the certificate
are the defaults; the suffix paths remain for the historical runs).

## 8. Open items

1. Wake mesh refinement at the trailing edge: replace the 26-chord row with resolved panels.
   That removes the truncation row instead of freezing it and is the only way to bring the
   `freeze` residual below the 3.25-chord plateau, and it would also decide whether the two
   TE-gap pairs at iteration 7 are a real crossing or a sliver artefact.
2. Convergence of the coupled iteration itself: with the gate and the amplitude fixed, the
   loads drift monotonically. Candidate causes to falsify next are the pressure-Kutta update
   and the arclength marcher's tangent averaging, measured as a function of the number of
   updates rather than of accepted gate steps.
3. Whether `truncation_policy='freeze'` should become the coupling default. It changes only
   the residual metric (loads are identical to 1e-6), so it is a reporting decision, not a
   physics decision, and it is left at `'advance'` here.
4. Nothing in this document changes `physical_accuracy_accepted`; the G2 case is still not a
   converged free-wake solution.