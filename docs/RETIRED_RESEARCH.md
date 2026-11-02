# Retired research code (S12)

`research-p1-line-archive` is the last commit where the modules in the table below live in
`panel/`. S12 removes them from the live tree. Nothing here was refuted: every result they
produced is still in this repository as a document plus, where a measurement was made, a JSON
artifact next to it. What is removed is the code that would let you re-run those measurements
without checking out the tag.

## Why remove working code

Three reasons, in order of weight.

1. **It is not reachable.** No production entry point imports any of these modules; the
   reachability walk in `test/test_live_path_has_no_numpy_lapack.py` starts at twelve live
   entries and never arrives here. They are alive only because their own tests import them.
2. **They duplicate the same discretization.** Six of the modules solve the same P1 potential
   problem with a different Kutta closure, gauge, or quadrature. S10 removed exactly this kind of
   duplication from the NumPy chain because two live implementations drift apart silently; a
   variant solver that no one calls has the same failure mode with a slower feedback loop, since
   nothing forces it to track changes to the geometry or wake data structures.
3. **Their measurements are already frozen.** The Galerkin convergence table, the edge exponent
   study, the sphere acceptance gates and the recovery comparisons are recorded as committed JSON
   or tables; re-deriving them is a tag checkout away.

## What was removed

| Module | What it did | Where its result lives |
| --- | --- | --- |
| `panel/p1_g1.py` | the first P1 collocation lifting solve (Dirichlet potentials, least squares) | docs/INTEGRATION_P1.md, docs/P1_G1_END_TO_END.md, docs/ISSUE49_P1_G1_DESIGN.md |
| `panel/p1_dirichlet.py` | the Dirichlet-collocation wall operator that p1_g1 and the G0 work built on | docs/ISSUE49_P1_G0_SPHERE.md |
| `panel/p1_galerkin_wall.py` | P1 Galerkin mass/weak-form wall operator | docs/ISSUE71_GALERKIN_WALL_PROTOTYPE.md |
| `panel/p1_galerkin_solver.py` | the weak-form Galerkin solve behind the issue 71 convergence result | docs/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md |
| `panel/p1_rank_aware_kutta.py` | the rank-aware Kutta variant and its rank-gap diagnostics | docs/ISSUE69_P1_DIRICHLET_RANK_KUTTA.md |
| `panel/p1_rank_gauge.py` | percentage-rank gauge experiment | docs/ISSUE56_RANK_GAUGE.md |
| `panel/p1_gate_checks.py` | the G0/G1 acceptance gate suite | docs/ISSUE49_P1_GATES.md |
| `panel/p1_doublet_moment.py` | doublet moment bookkeeping used by the gate work | docs/ISSUE49_P1_G0_SPHERE.md |
| `panel/p1_g0_acceptance.py` | G0 sphere acceptance harness | docs/ISSUE56_G0_ACCEPTANCE.md |
| `panel/p1_solution_status.py` | a status enum the Galerkin tools shared | docs/ISSUE77_CONTACT_TRACE_FIX.md |
| `panel/p1_te_jump_integral.py` | trailing-edge jump integral audit | docs/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md |
| `panel/p1_te_trace_audit.py` | TE trace audit for the collocation line | docs/ISSUE71_P1_GALERKIN_TE_TRACE_AUDIT.md |
| `panel/p1_solid_angle.py` | an early solid-angle helper predating panel/p1_analytic_sheet_velocity.py | docs/ISSUE56_SOLID_ANGLE_FREE_TERMS.md |
| `panel/p1_wake.py` | an early wake-sheet helper predating panel/wake_velocity.py | docs/INTEGRATION_P1.md |
| `panel/p1_edge_singularity.py` | the wedge exponent study (PR 74) | docs/ISSUE71_EDGE_SINGULARITY_DERIVATION.md, docs/ISSUE71_EDGE_EXPONENT_MEASUREMENT.md |
| `panel/probe_pressure_reconstruction.py` | the first pressure-reconstruction probe | docs/PRESSURE_RECONSTRUCTION_PROBE.md |
| `panel/affine_consistent_recovery.py` | affine-consistent pressure recovery variant | docs/ISSUE49_PHASE0_BASELINE.md |
| `panel/manufactured_surface_recovery.py` | manufactured-surface recovery oracle | docs/MANUFACTURED_SURFACE_RECOVERY.md |
| `panel/relax_wake.py` | an early wake relaxation driver | docs/WAKE_ACCURACY.md |
| `panel/analytic_sphere.py` | analytic sphere reference solution | docs/ISSUE49_PHASE0_BASELINE.md |
| `panel/audit_reference.py` | the sphere reference audit | docs/ISSUE49_PHASE0_BASELINE.md |
| `panel/run_p1_rank_aware_kutta.py` | CLI driver for the rank-aware variant | docs/ISSUE69_P1_DIRICHLET_RANK_KUTTA.md |

Also removed: 19 test modules that existed to test the modules above, and 13 tools that drove
them (`tools/issue71/*.py`, four `_galerkin_*` helpers plus `_bridge_diagnose.py` under
`tools/grass_fan_convergence/`, and all five scripts in `tools/vsp_evidence/p1/`).

## What was deliberately kept

| Kept | Why |
| --- | --- |
| `panel/p1_cp_diagnostics.py` | `test/test_p1_wake_contact.py` uses its surface gradient in a live-code assertion |
| `panel/p1_closed_jax.py`, `run_p1_closed.py` | the closed-body line is a separate demo still used by its own tests |
| `panel/pressure_stencil.py`, `panel/pressure_recovery_ab.py`, `compare_pressure_recovery.py` | imported by the half-model and recovery comparison modules that stay |
| `panel/solve.py`, `matrix_free.py`, `iterative.py`, `half_native*.py`, `kernel.py` | the VLM and half-model lines; they back issue 16, the half-model scaling note and the VSPAERO comparisons, and are a separate decision |

## How to run the retired measurements again

```
git checkout research-p1-line-archive
python -m pytest test/test_p1_galerkin_solver.py -q        # issue 71 convergence
python -m tools.issue71.galerkin_ab --help                 # the A/B driver
```

The frozen outputs are still in the live tree, for example `docs/issue71/galerkin_ab/*.json` and
`docs/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md`.

## Test baseline

| Tree | Result |
| --- | --- |
| before S12 (commit `aedadd5`) | 1000 passed, 21 skipped, 2 failed |
| after S12 | 827 passed, 21 skipped, 2 failed |

The two failures are the pre-existing jax 0.12 jaxpr recursion failures in
`test/test_half_native_panel.py` and `test/test_iterative_panel.py`; they fail on clean master too.

## What this does not claim

- It does not claim the retired solvers are wrong. It claims nothing in the live tree depends on
  them and their answers are recorded.
- It does not remove the documents that cite them. Each of those documents now carries a one-line
  archive note pointing at the tag; the numbers in them are unchanged.

## Where the receipt helpers went

`tools/issue71/galerkin_cad_sweep.py` was not pure solver code: three of its functions turn a
solved state into a receipt, and three live tests asserted them. They moved to
`panel/p1_contact_report.py` as `trace_audit_summary`, `contact_jump_diagnostics` and
`component_pressure_loads`; `test/test_p1_wake_contact.py` and `test/test_p1_galerkin_cad.py` now
import them from there. The fourth helper, `status(state, cap)`, mapped a state to a status string
and depended on the retired `panel/p1_solution_status.py`; its single test was dropped rather than
reimplemented, because the live acceptance predicate lives in `panel/p1_lifting_g2.py` and is
covered by its own tests.
