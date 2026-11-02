# T5: declared-contact classification and per-node step control

Status: research-only, in the PR #91 branch. Defaults unchanged
(`velocity_method=analytic_edges`, `gate_mode=audit`);
`physical_accuracy_accepted=false`; no threshold or tolerance was widened.

## 1. What T4 left open

T4 measured a blocked first geometry update on G2 and attributed it to a
declared-contact false positive: `panel/p1_coupled_material_wake.py::geometry_failure`
used a boolean SAT overlap over the swept hull `old ∪ new`, so a wake triangle
that shares two vertices with a wall triangle (a declared TE contact) could
never move, and every other update was scaled down by a *global* backtracking
factor until the sheet was effectively frozen.

T5 asks for (A) topology-aware contact classification that still refuses real
protrusions, (B) per-node step caps instead of global backtracking, (C) a
gate-mode audit that does not silently freeze nodes, and (D) exact reproduction
of the T4 baseline plus old/new A/B evidence.

## 2. A: contact classification

`panel/p1_geometry_contact.py` replaces the boolean SAT test with a signed
plane-distance classification:

* `declared_contact_pairs(wake_faces, wall_faces, group_labels=None)` derives the
  declared pairs from **vertex identity only** (group label, wake face id, wall
  face id) -> `('vertex'|'edge', shared vertex ids)`. Coordinates are never used
  to invent an identity, so translating a mesh cannot create contact.
* `classify_swept_wall(hull, wall_tri, *, tol, declared, declared_region, graze_tol)`
  classifies a pair as `disjoint`, `contact`, `transverse`, `penetration` or
  `coplanar_overlap`, with a signed depth. A declared pair may touch
  (|d| <= tol) and may leave its own attachment only inside a graze corridor:
  `graze = max(20*tol, 0.02*shortest_wall_edge)` measured to the **wall** half of
  the shared simplex (`mesh.xyz[shared_ids]`), not to the moving wake node.
* A pair that already violates in the geometry the step *starts* from is
  attributed (`pre_existing`) rather than excused when it keeps the same class
  and does not get deeper by more than `10*tol`. Any new class, any deeper
  protrusion and any undeclared contact still blocks. Inherited defects are
  reported every iteration; nothing is permanently exempt.

Two implementation bugs were found while testing:

1. An AABB early-exit in `_classify_set` returned `disjoint` for a triangle that
   sits *entirely behind* the wall plane (deep penetration has a disjoint AABB
   along the normal). The early exit is gone; callers already filter far pairs.
2. `if not offending:` on a numpy mask raised
   `ValueError: The truth value of an array ... is ambiguous`; the filter now
   tests `len(offending) == 0`.

## 3. B: per-node step control

Update 2026-11-01: the cap unit is the **shortest incident edge** (mesh resolution).
A link-length cap was tried and refuted: 0.25 * a 26-chord truncation link = 6.5 chord does
not bound strain and the next update is refused by the area/fold gate
(`docs/issue84/t5_gate_ab_contact_v8.json`). Truncation rows are still identified by a
*relative* criterion and reported/frozen (`truncation_policy`). The TE-gap false positive of
section A is fixed (per-point depth plus a sliver enclosure certificate), and the A/B now
admits eight consecutive updates instead of zero. See
`docs/archive/issue/ISSUE84_STREAMLINE_STEP_UNIT.md` sections 3.1, 4 and 5 for the readings; the remaining
blocker is the wake mesh, not the gate.

`node_step_scale(wake_xyz, wake_faces)` gives every node its shortest incident
edge; `clip_step_update(old, target, caps, roots=..., reference=...)` caps the
displacement node by node and returns a ledger (`mode`, `clipped`,
`clipped_fraction`, `raw_max`, `accepted_max`, `cap_min`, `clipped_ids`).
The cap is applied with a `(1-1e-9)` margin so the accepted step is *strictly*
inside `max_step_fraction` (a rounded cap used to read back as
0.25000000000000794 and fail the step gate).

The update loop clips first and only then falls back to a bounded global scale
if the clipped candidate still fails a gate; every trial writes its own `gate`
detail, `control_nodes` comes from the **last** refusing trial, and `control`
reports `none` / `per_node_step_cap` / `per_node_step_cap+bounded_global_fallback`.
A freeze is never silent: it is `blocked_geometry` with the refusing gate and
node ids.

## 4. C: gate-mode semantics

| mode | admitted nodes | counter semantics | production_safety |
| --- | --- | --- | --- |
| `enforce` | sampling-converged only (legacy freeze) | `rejected = ~eligible` | `sampling_converged` |
| `safe` | sampling-converged only | `rejected = convergence` | `sampling_converged` |
| `audit` (default) | every finite-speed node | `audit_rejected = convergence`, `advanced_despite_audit` | `sampling_not_gated` |
| `off` | every finite-speed node, no refined quadrature | same as audit | `sampling_not_gated` |

`rejected` is zero in `audit`/`off` and `advanced_despite_audit` is zero in
`safe`/`enforce`; tests assert both directions, so "audit rejection" can never be
read as "sampled pass". Gating is still needed for production accuracy, so
`audit` keeps `production_safety='sampling_not_gated'` in its counters.

## 5. Tests

`test/test_p1_geometry_contact.py` (12) + `test/test_p1_gate_contact.py` (13):

* declared pairs by vertex id only; shared-vertex and shared-edge tangency is
  allowed; leaving the shared vertex by 5e-4 inside a 1e-3 corridor is contact;
  the same motion without a corridor, a 4e-3 push, and an off-simplex
  penetration 0.2 away from the attachment are all refused;
* undeclared pairs get no corridor; deep penetration behind the wall plane is
  still caught (the AABB regression);
* `clip_step_update` caps every node, pins roots, and keeps the step strictly
  below `max_step_fraction`;
* inherited crossings are attributed (`pre_existing`, no block) while a new
  class or a deeper protrusion blocks;
* `audit` advances sampling-rejected nodes and reports `advanced_despite_audit`;
  `safe` freezes them and reports `rejected == convergence`.

Full T5-affected set (`test_p1_gate_contact`, `test_p1_geometry_contact`,
`test_gate_mode`, `test_p1_coupled_material_wake`): **73 passed / 30 s**.

## 6. D: A/B on G2

Setup: G2 assembly (`half_cad_wingbody_v1` main wing + `make_wing(4,4,span 1.8,
chord .55, wake 3)` raised tail at (1.6, 0, .35)), alpha 5 deg,
`kutta='signed_speed'`, `velocity_method='analytic_edges'`, `gate_mode=audit`, 8 iterations,
`--max-backtracks 12`, default `GeometryGates`. 765 wake nodes, 95 roots,
175 free nodes declared sampling-rejected by the end.

| condition | outcome | seconds | accepted updates | trials at the last iteration |
| --- | --- | --- | --- | --- |
| `legacy_gate_global` (pre-T5) | `blocked_geometry` at iter 0 | 10.8 | 0 | 12 `step` + 2 `wall` |
| `legacy_gate_per_node` | `blocked_geometry` at iter 0 | 28.5 | 0 | 14 `wall` |
| `new_gate_global` | `blocked_geometry` at iter 1 | 19.8 | 1 (at 2^-11) | 12 `step` + 2 `attachment` |
| `new_gate_per_node` | `blocked_geometry` at iter 7 | 261.3 | 7 | 12 `wall` (+ 2 `area`) |

`legacy_gate_global` reproduces the T4 baseline exactly (block at iter 0, the
same 12 `step` refusals followed by 2 `wall` refusals at 2^-11/2^-12). Per-node
caps alone (`legacy_gate_per_node`) remove the `step` refusals but the SAT gate
still refuses the declared TE contact at every scale.

`new_gate_per_node` per-iteration ledger (relaxation, accepted step, loads):

| iter | relaxation | dmax | clipped | CL | CL_gamma | CD_p | CM_y | map_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2^-6 | 1.75e-03 | 349/765 | 0.441355 | 0.446374 | 0.007127 | -0.12214 | 22.982 |
| 1 | 1 | 1.10e-01 | 353/765 | 0.441462 | 0.446486 | 0.007174 | -0.12201 | 52.015 |
| 2 | 1/4 | 7.39e-03 | 350/765 | 0.437905 | 0.442527 | 0.007163 | -0.12048 | 52.733 |
| 3 | 1/8 | 3.80e-03 | 346/765 | 0.431795 | 0.436231 | 0.006394 | -0.11998 | 52.858 |
| 4 | 2^-6 | 6.87e-04 | 343/765 | 0.424059 | 0.428560 | 0.005653 | -0.11780 | 52.749 |
| 5 | 2^-11 | 2.22e-05 | 342/765 | 0.423021 | 0.427538 | 0.005558 | -0.11749 | 52.742 |
| 6 | 2^-12 | 1.11e-05 | 342/765 | 0.422988 | 0.427499 | 0.005555 | -0.11748 | 52.742 |
| 7 | blocked | 0 | 342/765 | 0.422972 | 0.427487 | 0.005553 | -0.11748 | 52.742 |

Clip ledger throughout: 342-353 of 765 nodes clipped (44.7-46.1 %), raw target
displacement 22.98-52.86 chord reduced to an accepted maximum of
2.9e-2-3.1e-2 chord, `cap_min = 2.31e-3`. `wall_rms` stays at
1.216e-03 -> 1.235e-03 (+1.6 %).

### 6.1 Why it still blocks

The final refusals are two declared pairs, wake face 299 against wall face 19
(shared edge) and wall face 415 (shared vertex):

* depths 5.56e-04 and 3.33e-02 with `old_kind='disjoint'` -> the crossing is
  **new** (the old triangle's behind-plane points project outside the wall
  footprint; the moved ones do not), so the inherited-defect attribution
  correctly does not excuse them;
* they persist from factor 2^-1 to 2^-12 because the depth is inherited from the
  *static* geometry and only the footprint crossing is new, i.e. shrinking the
  step does not un-cross the footprint.

`pre_existing` is 0 in every iteration: G2 never enters the inherited branch,
which is exercised only by the synthetic unit tests. The starting mesh has
0 violations at identity (1130 declared pairs, 54 wall pairs in bounds, 52
declared / 2 undeclared / 0 violations).

### 6.2 What actually limits the update

`map_residual_chord` is the norm of the streamline displacement target. It is
22.98 chord at iteration 0 and 52.0-52.9 chord afterwards, while the per-node cap
is 2.31e-3 chord: the sheet creeps at 1/64-1/4096 of a target it cannot follow
and `map_residual_chord` sits on a plateau (52.742 with 1e-5 changes) from
iteration 1 on. This reproduces T4's "accepted != progress": acceptance is a
gate property, progress is not. The design point that follows is that the
*streamline target* must be repaired (arc-length/pseudo-time normalisation
rather than a raw `u/|Vx|` displacement) before geometry updates can be
evaluated on physics; T5 does not change that formulation.

## 7. Verdict

* **A (repair the false positives): done.** Declared contacts are recognised by
  vertex identity, may graze inside an explicit corridor, and every refusal is
  now a signed classification with a depth and a class rather than a boolean SAT
  hit. The T4 iter-0 false positive is gone (`new_gate_global` accepts the first
  update the legacy gate refused).
* **B (per-node control): implemented, but not sufficient.** Caps, ledgers and
  last-trial control-node attribution work and the step gate never fires in the
  production path; the sequence still collapses to 1/4096 because the target is
  52 chord away. No default or tolerance was changed.
* **C (gate-mode audit): done.** `audit` is not a freeze (sampling-rejected
  nodes advance and are counted in `advanced_despite_audit`) and `safe` is the
  enforcer; `production_safety` states the accuracy status in the counters.
* **D (A/B): done** for the 2x2 matrix above, with the T4 baseline reproduced
  bit-for-bit in load values at iteration 0
  (CL 0.441355 / CD_p 0.007127 / CM_y -0.12214).
* **Test ledger on this head**: targeted T5 set 73 passed / 30 s; whole
  `test/` tree **772 passed, 12 skipped, 0 failed** (182 s) with
  `runs/torch_site` on `PYTHONPATH` (7 collection errors without it, all torch
  imports). No test was skipped or xfailed to obtain that ledger.

Out of scope but now the blocking item: the streamline target magnitude and the
TE attachment that lets the sheet drift 3.3e-2 chord into the wall footprint.

## 8. Reproduction

```powershell
$env:PYTHONPATH='D:/3D/Projects/panel/runs/fix_master;D:/3D/Projects/panel/runs'
$py = 'D:/3D/Projects/panel/.conda_env/python.exe'
& $py tools/issue84/t5_gate_ab.py legacy_gate_global    8 docs/issue84/t5_gate_ab_legacy_gate_global_v2.json
& $py tools/issue84/t5_gate_ab.py legacy_gate_per_node  8 docs/issue84/t5_gate_ab_legacy_gate_per_node.json
& $py tools/issue84/t5_gate_ab.py new_gate_global       8 docs/issue84/t5_gate_ab_new_gate_global.json
& $py tools/issue84/t5_gate_ab.py new_gate_per_node     8 docs/issue84/t5_gate_ab_new_gate_per_node_v4.json
& $py tools/issue84/t5_static_contacts.py
```
