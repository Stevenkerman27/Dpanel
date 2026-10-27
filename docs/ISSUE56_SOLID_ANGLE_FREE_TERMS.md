# Issue #56, part A: the node free term is a 3D solid angle, and the sector rule is wrong

Research only. `physical_accuracy_accepted=false`; no production default, rank gate or
`panel/kernel.py` line was touched. Numbers below are produced by `test/test_p1_solid_angle.py`
and `panel/p1_solid_angle.py` (run on the 396-panel wing/body half mesh only where stated).

## 1. What was wrong

`panel/p1_gate_checks.py` treated the sum of *in-plane* sector angles over the incident fan,
divided by `4 pi`, as a node's solid-angle free term, and PR54 reported the resulting
`c = 0.2343 .. 0.5044` as if it were the P1 diagonal. It is not: each incident panel was
evaluated in its own normal limit and the results were added. Two independent failures follow
from that, both measured here.

## 2. Analytic corner oracle (demand A)

The interior cone of a trihedral corner subtends `Omega = alpha + beta + gamma - pi`, where the
angles are the dihedral angles between the outward face normals. `cone_solid_angle` implements
that; `sector_angle_free_term` implements the old rule:

| corner | `Omega` | inside `Omega/4pi` | outside `1 - Omega/4pi` | sector rule | error of the sector rule |
| --- | --- | --- | --- | --- | --- |
| right tetrahedron, origin corner | 1.570796327 | **0.125000000** | **0.875000000** | 0.375000000 | 0.500 (133 percent of the true interior value) |
| generic tetrahedron (`e_x`, `(0.35, 1.1, 0)`, `(0.2, 0.25, 0.9)`) | 2.227410421 | 0.177251689 | 0.822748311 | 0.308377733 | 0.131 |

The right-tetrahedron row reproduces the issue's numbers exactly (`1/8` and `7/8`), and the
dihedral angles come out at `pi/2` to machine precision, so this is an independent check rather
than a restatement.

Single-panel behaviour is unchanged and still correct, which is why the defect is specifically
in `assembled_free_terms`: a flat triangle seen from an edge midpoint gives `Omega/4pi =
0.25000`, from a vertex `0.12500` (its in-plane angle), both to `1e-5`.

## 3. The only free-term statement that survived: the constant-density identity

Summing the signed solid angles of **all** panels with the probe stepped off the node gives
exactly `+1` from the interior side and `0` from the exterior side, for every closed test body:

| body | all panels, interior limit | all panels, exterior limit | offset range |
| --- | --- | --- | --- |
| right tetrahedron, corner | 1.000000000 | 0.000000000 | 1e-4 .. 1e-6 |
| generic tetrahedron, corner | 1.000000000 | 0.000000000 | 1e-4 .. 1e-6 |
| icosphere level 2, nodes 0/5/17 | 1.000000000 | 0.000000000 | 1e-3 .. 1e-4 |
| two-triangle open sheet | 0.125000000 (no identity) | -- | 1e-6 |

This is the same `-1` (up to the kernel's sign convention) that the panel-centroid assembly
reproduces on the real half mesh once the mirrored block is subtracted, and it is the check a
node-collocation assembly should be accepted against. It is *not* a formula for a diagonal
entry: the diagonal is whatever the probe limit assigns, and the row is then automatically
consistent.

## 4. Incident-fan sums are not coefficients either

| quantity | right tetra corner | generic tetra corner | icosphere L2 node 0 |
| --- | --- | --- | --- |
| sector rule | 0.375 | 0.308 | 0.493 .. 0.494 |
| incident fan, exterior probe | -0.125 | -0.073 | -0.42 |
| incident fan, interior probe | 0.875 | -- | -- |
| true `1 - Omega/4pi` | 0.875 | 0.823 | 0.5 (flat limit) |

Three separate lessons:

1. The right-tetrahedron fan value `0.875` coincides with the coefficient only because the corner
   is orthogonal; on the generic corner the same recipe gives `-0.073` where the answer is
   `0.823`. Signed fan sums cancel whole faces at a sharp corner.
2. On a triangulated sphere both fan rules are biased at finite `h` (`0.42` versus the flat limit
   `0.5`), because the incident fan does not cover the ring of neighbours the continuum integral
   sees. Refinement, not tuning, is what closes that gap.
3. Winding matters: with the tetrahedron written in mixed orientation the same probe flips side
   and the fan sum reads `-0.875`. `outward_orient` now normalises orientation before any of
   these measurements, and each test states its convention.

## 5. Demand B: the audit tautologies are gone

`node_dof_audit` no longer reports `n_equations = len(vertices)` or
`equations_match_raw_unknowns = bool(len(vertices) == len(vertices))`. Every count is now
measured, and the two claims that cannot be measured from a mesh alone are delegated:

| key | meaning |
| --- | --- |
| `n_wall_used_vertices` | distinct merged vertices actually referenced by faces (the compact DOF) |
| `n_continuous_node_unknowns` | `= n_wall_used_vertices`; one potential unknown per geometric node |
| `n_orphan_vertices` | vertices no face references (the 12290-panel mesh has 1212 wake-only ones) |
| `n_distinct_physical_cut_traces` | merged groups that are used and hold more than one original vertex |
| `n_test_equations` | supplied by the caller once the test space is chosen; `None` until then |
| `square_system` | `None` unless `n_test_equations` was supplied, then the actual comparison |
| `surface_has_boundary` | descriptive count, **no** longer called `gauge_required` |
| `nullspace_probe_required` | `True`: the gauge question is answered on the assembled operator |

Fixture check: a two-triangle sheet gives `n_wall_used_vertices = 4`, `n_orphan_vertices = 0`,
`n_distinct_physical_cut_traces = 0`; duplicating one shared node gives `n_unique_vertices = 4`
from 5 coordinates, `n_distinct_physical_cut_traces = 1`, `split_nodes_present = True`, and
`square_system = True` only because the caller passed `n_test_equations = 4`.

`assembled_free_terms` is kept but relabelled: it returns `rule = "sector_angle_not_free_term"`
and its docstring points at `cone_solid_angle` and `assembled_row_sum` instead. PR54's
`c = 0.2343 .. 0.5044` column must not be quoted as a P1 diagonal anywhere.

## 6. Not done yet (explicitly, so it is not mistaken for done)

- **C** rank and gauge on three topologies (closed sphere/tetrahedron, small open sheet, full
  wing plus wake), with `||A 1||`, the measured null space, `sigma_min` and the effect of an
  explicit constraint. Only the descriptive counts exist so far.
- **D** the well-posed small G0 P1 discretisation. G1's linear Kutta row remains a definition
  rather than a condition (see the issue #49 receipt of commit `6b5f780`), so there is no
  lifting P1 result to report.
- Concave corners and corners with more than three incident faces: `cone_solid_angle` raises
  instead of guessing, and no concave oracle has been written.

Tests: `test/test_p1_solid_angle.py` 8 cases; P1 subset (solid angle, gate checks, G0 sphere,
wake, G1, doublet moment) **61 passed / 15.4 s**.