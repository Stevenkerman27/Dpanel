# Issue #65 / B2: coupled material-coordinate wake experiment

Date: 2026-10-28. Parent: #65. Acceptance discussion: https://github.com/Stevenkerman27/Dpanel/issues/65#issuecomment-6466403840.

**不宣称稳态；physical_accuracy_accepted=false。** This is an opt-in integration experiment, not a replacement of the existing solver and not a GRASS training backend.

## Ownership and branch lifecycle

Cloud owns implementation, numerical diagnosis, tests and corrections. The local agent checks out the PR, executes the commands, reports unedited evidence and independently reviews it; local is NOT assigned to implement the B2 solver. One branch from `master`, one PR to `master`, no stacked dependency. Do not merge before the pending full-repository tests and an execution receipt have been reviewed. A physically non-convergent but correctly reported experiment is not automatically a software defect; it must not be described as an accepted physical free wake.

Base: `9480516c5dd80c24d9346ee077956cc48c8bd920`.
Implementation/test commit: `0f6be3e463d2849a847224da000d220757a86aaf` (subsequent documentation-only commits do not change these source blobs).

## What is implemented

`panel/p1_coupled_material_wake.py` computes the same total potential representation as `_potentials_at`: freestream plus the shared wall doublet field plus every independent wake. Wall values are `mu[dof_face]`; each wake's nodal values are `gamma_hat @ Gamma_g`. Per-face centroids and tangential gradients are reconstructed on the CURRENT geometry, not carried as fixed world-coordinate gradients and not reduced to P0 strengths.

`TotalVelocity` vectorizes the formula in `wake_velocity.sheet_velocity`. Production `influence_block` uses the opposite double-layer sign, so the adapter applies one global minus sign to BOTH wall and wake contributions. The 48-point edge quadrature is preserved; it is not mislabeled as an analytically integrated edge kernel. The implementation measures both the last epsilon-ladder change and the change on doubling edge quadrature order (48 to 96 by default). Neither finite check proves a Cauchy principal value, especially at a free edge or a TE cut.

Each accepted candidate updates only W; A/J/K/wall geometry are reused. `solve_multi` then re-solves shared mu and ALL groups' Gamma before diagnostics or a final result are returned. Iteration 0 is separately compared with the canonical fixed-wake solve, including W and CLp/CLgamma/CDp/CMy. The gate is `abs(delta)/max(1,abs(reference)) <= 1e-10` (a mixed absolute/relative gate, intentionally well-defined at zero load).

The geometry map is rooted fixed-material-arclength Picard iteration, not unsteady material advection:

```
X_target[root] = X_reference[root]
X_target[i] = X_target[pred(i)] + ds_reference[i] * unit(V_endpoint_average)
X_trial = X_current + relaxation * (X_target - X_current)
```

At a TE root, the first downstream node's tangent is used rather than pretending the fixed root is a freely moving sheet node. An aligned uniform wake is a fixed point; xyz can all change, with no division by Vx. Connectivity, material interpolation and roots remain fixed. Root/path discovery still inherits `prepare_wake_material`'s original upstream-x graph assumption: this is NOT a new automatic TE detector for arbitrary geometry.

A node with unresolved epsilon/order sensitivity or a vanishing tangent stays in place and remains a rejected node for convergence purposes. Whole candidate geometries are validated atomically. Backtracking responds only to geometry safety, not to desired CL or VSPAERO agreement. Area changes are checked against both the previous and reference geometry; swept convex hulls test wall and non-neighbour wake intersections. Exhausting the collision-pair budget rejects the update rather than silently skipping checks. Swept hulls are conservative and may reject noncolliding motions; declared boundary contact is tolerated by a 1e-8 relative face shrink and a 1e-10 chord projection tolerance. Topological neighbours use the local fold gate. These are explicit limitations, not a universal CAD collision certificate.

## Recorded quantities and interpretation

Every solved iteration includes CLp, CLgamma, CD_pressure, CM_y, signed and relative CLp-CLgamma gap, wall/jump RMS, both Kutta RMS values, linear-seed rank/singular minimum, and the rank/singular minimum of the selected EXACT linear constraints. The inherited `linear_constrained_rank`/`linear_singular_min` describe the signed-speed linear seed, not the final nonlinear pressure Jacobian. Default Kutta is `pressure`; `signed_speed` remains an explicit diagnostic control.

Face velocity samples are evaluated at face centres, not by averaging singular nodal values. Slip, Cp jump, normal jump and Cp-identity residual have area-weighted RMS/max. All faces are included, even when nearby nodes are refused. Three fixed bands use relative ORIGINAL material arclength, not absolute chord cutoffs or moving quantiles; each reports face count, area and total area fraction. Current solver freestream speed is one, so these quantities are already normalized by U or U^2.

For finite offsets the measured normal velocity difference is not forced to zero. The identity residual is `dCp + 2*Vmean.dot(Vplus-Vminus)`, not dCp itself. This identity is an algebraic check, not an independent physical-accuracy test.

`incoming_update` belongs to the motion which produced this solved state; `outgoing_update` belongs to the candidate built from it. Rejection units: root/speed/convergence count nodes; geometry reason counts count refused backtracking trials. Per-group `eligibility.rejected` counts refused free nodes, not faces. Displacement RMS includes all used nodes, including fixed roots. Trends explicitly retain nonmonotone slip and rejected-node counts. No fixed point found in N steps is not a proof that no mathematical steady solution exists.

`converged` and `physical_accuracy_accepted` always remain false. A `kinematic_candidate` only means this discretization met finite sampling and kinematic gates; pressure continuity, mesh/epsilon convergence and external validation remain separate work.

NPZ snapshots preserve mu/Gamma/Cp, wall and wake geometry, material bases, roots, predecessor graph, samples, eligibility and sampling controls. A content SHA covers array names, shapes, dtypes and bytes independently of ZIP timestamps. Frozen replay re-evaluates the saved total field without re-solving strengths and reports bitwise equality; identical bits are not promised across environments.

## Actual cloud execution — not a full checkout

The cloud container could not clone GitHub (DNS resolution failure). The exact three NEW source files were tested locally; their blob SHAs were checked against the remote branch. No claim is made that the complete repository or generated/CAD integration passed.

Environment: Linux x86_64/glibc 2.41; Python 3.13.5; NumPy 2.3.5; SciPy 1.17.0; pytest 9.0.2.

| Command / check actually executed | Exit | Observed result |
| --- | --- | --- |
| `python -m py_compile panel/p1_coupled_material_wake.py panel/run_p1_coupled_material_wake.py test/test_p1_coupled_material_wake.py` | 0 | All three compile |
| `python -m pytest test/test_p1_coupled_material_wake.py -q -k 'not integration'` | 0 | 27 passed, 3 deselected |
| CLI frozen-snapshot replay of the test's manufactured two-triangle P1 field | 0 | bitwise_equal=true, all maximum differences=0 |
| Full-repository integration tests | not run | Pending local execution; NOT counted as passed |
| Generated G1/G2 physical iteration histories | not run | Pending local execution; no invented iteration table |
| Real CAD | not run | Requires an existing local input and its SHA |

The 27 passing tests include independent P1 area-integral comparison, field superposition without double counting, gradient transport, root/path invariants, reverse Vx, uniform-flow fixed point, rejection gates, area/band accounting and replay. Four tests use MOCKED solves to verify orchestration/blocked-state behavior; those are not physical or production-solver validation.

Source blobs:

```
panel/p1_coupled_material_wake.py       7949ce1924d3a74cc23f1d5d20fe9c7bbad212c8
panel/run_p1_coupled_material_wake.py   08f4f5d1167d682d283d6a1eaed0d412e0c6b9c1
test/test_p1_coupled_material_wake.py   631563dbf33ab6524690327652161ca2be81aa85
```

Failure retained: the first unit run was 19 passed / 1 failed / 3 deselected, exit 1. A near-zero interpolation assertion had `atol=0`: expected 6.66e-18, observed 2.78e-17. The assertion now uses atol=1e-14; no physical kernel, convergence gate or solver was adjusted to hide that failure. An auxiliary cloud receipt script also initially collided with Python's `test` package when importing the fixture; using `runpy.run_path` corrected the harness without changing submitted code.

## Local execution and acceptance — no implementation assignment

Use a clean checkout of this PR, record `git rev-parse HEAD`, and fix thread counts. Every output directory must be new; existing evidence is never overwritten. Record shell exit codes even when a run is blocked.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest test/test_wake_velocity.py test/test_wake_nodes.py test/test_p1_coupled_material_wake.py -q

python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta signed_speed --iterations 0 --chord-panels 4 --span-panels 4 --wake-panels 3 --out runs/issue65_b2_g1_i0
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 4 --chord-panels 4 --span-panels 4 --wake-panels 3 --out runs/issue65_b2_g1_pressure
python -m panel.run_p1_coupled_material_wake --case g2 --alpha 5 --kutta pressure --iterations 4 --chord-panels 4 --span-panels 4 --wake-panels 3 --out runs/issue65_b2_g2_pressure
```

Only after these receipts are inspected, refine the epsilon ladder and quadrature separately, retaining the initial geometry, strength snapshot and all failed results. A finer sampling run is a sensitivity check, not permission to tune away a failing physics metric. The optional CAD path must actually exist:

```bash
python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 5 --kutta pressure --iterations 4 --out runs/issue65_b2_cad_pressure
```

The CAD adapter owns that input's wake discretisation; `--wake-panels` only affects generated wings. Do not invent `cases/half_cad_wingbody_v1/surface.json`.

For a saved iteration, copy `state_sha256` from its JSON and run:

```bash
python -m panel.run_p1_coupled_material_wake --replay-snapshot runs/issue65_b2_g1_pressure/iteration_000.npz --expected-state-sha THE_RECORDED_SHA --out runs/issue65_b2_g1_replay
```

Exit codes: 0 means execution completed, NOT physical acceptance; 1 means an execution exception; 2 means a blocked geometry/velocity/nonlinear run with evidence; 3 means frozen replay was not bitwise equal. Return the logs, receipt/report JSON, snapshots or their accessible artifact locations, HEAD SHA and exit codes in #65 and the PR. Do not patch the algorithm locally; cloud owns any code corrections after the failure is localized.

## Audit notes on the historical B1 evidence

At the pinned base, `wake_nodes.grid_sheet` numbers nodes `i + j*(nx+1)` but sets `roots=np.arange(nx+1)`, which fixes the first spanwise row rather than the intended i=0 TE column. Its `sheet_diagnostics` per-band `audit_rms` aggregates the audit expression rather than the identity residual (the separate top-level maximum residual is different). Also `WakeSheet.with_vertices` carries a world-coordinate gradient unchanged. These are source-review findings, not newly executed B1 results; historical B1 tables must not be promoted to free-wake acceptance evidence without correction and re-execution. This B2 route does not use those routines: roots come from the declared material graph, P1 gradients are reconstructed, and identity residuals are computed explicitly. The historical B1 files are not silently rewritten in this PR.

## Exit criteria

First establish same-flow/sign/basis correctness and accepted-step re-solving with full-repository tests. Then obtain and review G1, G2 and an available CAD history. Reject convergence claims with frozen nodes, unresolved epsilon/order checks, nonconverged Kutta solves, untracked geometry rejection, or divergent CLp/CLgamma disagreement. Do not choose a method because CL increases, drag decreases or a VSPAERO value is approached. Do not close #65 merely because this integration code merges. Analytic/controlled near-edge limits, pressure-continuous steady acceptance, implicit wake VJP and unsteady Kelvin-history shedding remain separate milestones.
