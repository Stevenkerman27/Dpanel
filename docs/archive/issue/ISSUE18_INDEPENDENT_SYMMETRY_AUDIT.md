# Independent symmetry audit of the saved half-model fixtures (issue #18, PR #19)

Scope: verify the half-model symmetry gate repair and the blocked verdict in
`panel/audit_half_mesh.py` with a *separate* NumPy implementation, and pin down
what the old trailing-edge "64/80 unpaired" number actually measured.

Nothing here solves a flow, changes CAD, relaxes a tolerance or claims physical
or AD acceptance. `status == "ok"` would only mean the discrete surface is
mirror invariant.

## 1. PR #19 acceptance (this machine)

| item | result |
| --- | --- |
| head | `2465a729fe9cac36f8579989f88a13ee09a5ca1f` (branch `fix/half-model-symmetry-gate-audit`) |
| `pytest -q test/test_half_mesh_audit.py test/test_matrix_free_symmetry_gate.py` | **10 passed, 0.66 s, exit 0** |
| `python -m panel.audit_half_mesh` on the three fixtures | **exit 0, 0.34-0.40 s each, `status=blocked`** |

Environment: Windows, Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, JAX 0.10.2 (CPU,
x64 disabled by default), 32 cores. `TMP/TEMP/TMPDIR` were redirected because
the sandbox blocks `tempfile.mkdtemp` (this is environment, not repository).

The audit rejects all three fixtures for six reasons: no bijective vertex
reflection, no reflected wall connectivity, wall triangles crossing the
symmetry plane, no reflected wake connectivity, TE banks not matched, and
non-mirror-equivariant component labels.

## 2. Independent implementation

`panel/symmetry.py` (NumPy only, no JAX, no imports from the audit) plus
`panel/verify_symmetry_independent.py` (CLI) and
`test/test_symmetry_diagnostics.py` (**5 passed, 0.07 s**). Layers are reported
separately: vertex map (bijection, involution, residual), wall connectivity
with winding verdict, wake connectivity, TE edge and bank pairing, circulation
stations, and a geometry split into true on-plane caps versus panels that
merely straddle or nearly touch the plane.

Cross-check at the same relative tolerance as the audit (1e-10):

| case | nodes paired | wall verified | crossing | plane caps | wake verified | TE edges mirrored | TE bank-matched | stations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_wing | 399/531 | 26/484 | 17 | 0 | 268/600 | 80/80 | 2 | 41/41 |
| two_bodies_coarse | 345/649 | 56/616 | 18 | 0 | 180/708 (audit: 148) | 72/72 | 2 | 45/45 |
| forward_body | 406/637 | 72/696 | 30 | 0 | 268/600 | 80/80 | 14 | 41/41 |

Vertex, wall, crossing, plane-cap, TE-edge and station counts agree exactly with
`panel/audit_half_mesh.py`. Two counts differ by construction:

- **wake.** The audit counts faces whose vertices are individually paired
  (partial test); this implementation additionally requires the mirrored
  centroid to land within tolerance. The two agree on single_wing (268) and
  forward_body (268) and differ on two_bodies_coarse (180 vs 148).
- **TE bank pairs.** The audit reports 0 fully auditable pairs (because the wall
  face map is not bijective) plus 2 partial; this implementation reports 2, 2
  and 14 edge pairs whose two bank panels both have verified mirror faces. Both
  readings say the bank layer is broken.

## 3. Set membership is not a mirror test

The old gate compared a mirrored upper-TE face id against the *set* of lower-TE
face ids. The same mistake shape applies to any vertex-set lookup: on
`single_wing` the vertex map is not injective (399 of 531 vertices paired,
132 unpaired, `bijective=False`), so applying it to a triangle frequently lands
on an unrelated face that happens to exist:

| case | wall vertex-set hits | hits that are geometric mirrors | worst spurious distance |
| --- | --- | --- | --- |
| single_wing | 376 / 484 | 26 (350 spurious) | 9.50e-02 |
| two_bodies_coarse | 407 / 616 | 56 (351 spurious) | 5.66e-02 |
| forward_body | 506 / 696 | 72 (434 spurious) | 1.27e-01 |

So a connectivity-only audit can report 376/484 "paired" wall panels when the
true count is 26/484. The geometric check (mirrored centroid within tolerance)
is mandatory, and it is what the repaired TE gate also needs. A first pass of
this module without that check produced 376/484 and was discarded for exactly
that reason.

## 4. Was "64/80 TE edges unpaired" a false diagnosis?

Yes. All trailing edges are geometrically mirrored in every fixture: 80/80,
72/72, 80/80. The old 64/80 figure came from asking for a z-bank swap (mirror
of an upper panel against the lower-panel set), which no XZ-symmetric mesh can
satisfy. The genuine TE failure is the bank *pairing*: only 2/80 edge pairs on
single_wing have both bank panels mirrored onto the corresponding partner banks.

## 5. Wake pairing is tolerance dependent

Repeating the two_bodies_coarse wake layer at three relative tolerances:

| relative tolerance | nodes paired | wake verified |
| --- | --- | --- |
| 1e-9 | 393/649 | 248/708 |
| 1e-10 | 345/649 | 180/708 |
| 1e-11 | 295/649 | 112/708 |

A pairing count that changes by a factor of two when the tolerance changes by
one decade is not evidence of symmetry; the wake layer must be judged on a
strict map, not on a near-miss count.

## 6. What this does not establish

- The five production Gmsh fixtures remain `blocked`; `--half` still raises
  `ValueError: Need bijective mirrored vertices, including wake nodes`
  (`panel/iterative.py:101`).
- No statement about loads, gradients or `physical_accuracy_accepted`, which
  stays `false`.
- The gate repair was verified through its unit tests and the audit only; this
  report does not re-run the full GMRES path on a symmetric mesh, because no
  such mesh exists yet (issue #20).

## 7. Reproduction

```
python -m pytest -q test/test_half_mesh_audit.py test/test_matrix_free_symmetry_gate.py
python -m pytest -q test/test_symmetry_diagnostics.py
python -m panel.audit_half_mesh --surface cases/panel_cut/single_wing.json --out runs/half_audit_single_v1.json
python -m panel.verify_symmetry_independent --surface cases/panel_cut/single_wing.json
```

Both audit commands exit 0 on a completed audit and report the verdict in JSON;
`--fail-on-blocked` turns a blocked surface into exit 1 for CI use.
