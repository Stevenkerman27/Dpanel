# Issue #56, part C: rank and gauge, measured on four topologies

Research only. `physical_accuracy_accepted=false`; no production default, rank gate or
`panel/kernel.py` line is touched. Data: `docs/issue49_vspaero/p1/rank_gauge.json`, produced by
`tools/vsp_evidence/p1/p1_rank_gauge_report.py`; implementation `panel/p1_rank_gauge.py`; contract
`test/test_p1_rank_gauge.py` (6 cases).

## 1. The claim being tested

`node_dof_audit` used to return `gauge_required = True` whenever the merged mesh had a boundary
edge, with the reasoning that "an open surface admits a constant doublet shift". That is a
statement about a specific *operator*, not about a mesh, so it was removed in part B and is now
replaced by three measurements on the operator that is actually assembled:

1. numerical rank and the singular-value floor;
2. whether the constant density vector is in the null space (`||A 1||`, and the row sums);
3. what an explicit pin does to the least-squares residual (unchanged = a gauge, worse = a
   different discrete problem).

## 2. The four topologies

| operator | rows | cols | rank | nullity | `sigma_min` | `||A 1||` | row sums |
| --- | --- | --- | --- | --- | --- | --- | --- |
| icosphere L1 (closed) | 80 | 42 | 42 | 0 | 0.1430 | 8.944 | -1.000000 (5.6e-16) |
| icosphere L2 (closed) | 320 | 162 | 162 | 0 | 0.0676 | 17.889 | -1.000000 (1.2e-15) |
| tetrahedron (closed) | 4 | 4 | 4 | 0 | 0.0921 | 2.000 | -1.000000 (0) |
| open sheet, 2 panels | 2 | 4 | 2 | **2** | 0.1667 | 1.414 | not constant |
| half wing 396 panels, wall only | 396 | 213 | 213 | 0 | 0.0652 | 17.158 | -0.856709 (2.9e-01) |
| half wing, wall minus mirror | 396 | 213 | 213 | 0 | 0.0652 | 19.900 | **-1.000000 (3.0e-15)** |
| half wing, wall plus mirror | 396 | 213 | 213 | 0 | 0.0651 | 19.900 | -0.713417 (5.7e-01) |

`constant_response` is `||A 1||`, not a normalised number; for a closed body it equals `sqrt(m)`
because every row sum is exactly `-1`, which is the constant-density identity already used in
part A, now seen from the matrix side.

## 3. Findings

**3.1 The constant is never a null vector of the pure-doublet collocation operator.** For every
closed body `A 1 = -1`, and for the open sheet and the open half wing the row sums are not even
constant (`-0.856709` with spread `0.286` on the wall alone). So this formulation has no constant
null space anywhere, and the classical "open surface needs a gauge" reasoning does not transfer
to it.

**3.2 Pinning changes the fit, so it is not a gauge here.** On the 396-panel wall-minus-mirror
operator, pinning node 0 to zero moves the least-squares residual from `0.024139` to `0.231993`
(+861 percent) while `sigma_min` barely moves (`0.0652` to `0.0653`). A genuine gauge leaves the
residual untouched by construction. Any pin added to this system is therefore a modelling
decision that has to be justified physically, not bookkeeping.

**3.3 The only null space seen on a real mesh was a bookkeeping artefact.** Assembling the same
wing on all 400 coordinates (187 of which are wake-only and referenced by no wall face) gives
nullity `187` exactly: `rank 213` of `400` columns. That is the `n_orphan_vertices` count from
part B, not a gauge, and it is why the DOF audit now builds its unknowns from the referenced wall
nodes.

**3.4 The mirror block must be subtracted, and the operator sees it too.** Wall-only row sums are
`-0.856709` with spread `0.286`; subtracting the mirrored block gives `-1.000000` with spread
`3.0e-15`; adding it gives `-0.713417` with spread `0.572`. This reproduces, on the operator
itself, the sign that G1 first established from the loads (`A - B` row sum `-1.0`, spread
`2.9e-15`), so the mirror convention is now confirmed in two independent ways.

**3.5 The trailing-edge cut does not open a null space in this formulation.** The 213-DOF
wall-minus-mirror operator has `sigma_min = 0.0652` and nullity `0`, i.e. the open TE is not what
makes the system singular. In G1 the cut mattered for *Kutta coupling* (a linear `mu_up - mu_lo -
Gamma = 0` row is a definition, so the least-squares solution takes `Gamma = 0`), not for rank.

## 4. Consequences for a gauge policy

- State the operator. Rank, null space and gauge are properties of the assembled matrix, not of
  the mesh; `surface_has_boundary` stays descriptive and `nullspace_probe_required` stays `True`.
- The G0 sphere and the G1 wing solve uniquely with no pin; that is measured, not assumed.
- If a square boundary-integral operator with an explicit `+-1/2` jump term is ever assembled,
  these three measurements have to be repeated for *that* matrix before any gauge or pin is
  claimed. The present results say nothing about it.
- A pin is admissible only when the residual is unchanged to solver precision; otherwise it
  belongs in the physics discussion (wake cut, TE trace), not in the linear algebra.

## 5. Not done

- **D**, the well-posed small G0 discretisation with the analytic-sphere multi-offset `Cp`,
  no-penetration and load-convergence acceptance: still open, and the natural next step.
- The square jump-form operator (a `+-1/2` diagonal) is not implemented, so no gauge statement
  is made about it.
- Concave corners and corners with more than three incident faces remain unsupported;
  `cone_solid_angle` raises rather than guessing.

Tests: `test/test_p1_rank_gauge.py` 6 cases; P1 subset (solid angle, gate checks, rank/gauge,
G0 sphere, wake, G1, doublet moment) 67 passed.