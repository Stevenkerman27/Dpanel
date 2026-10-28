# Issue #65 / B2: coupled material-coordinate wake experiment

Date: 2026-10-28. Parent: #65. Acceptance discussion: https://github.com/Stevenkerman27/Dpanel/issues/65#issuecomment-6466403840.

**不宣称稳态；physical_accuracy_accepted=false。** This is an opt-in integration experiment, not a replacement of the existing solver and not a GRASS training backend.

## Ownership and branch lifecycle

Cloud owns implementation, numerical diagnosis, tests and corrections. The local agent checks out the PR, executes the commands, reports unedited evidence and independently reviews it; local is NOT assigned to implement the B2 solver. One branch from `master`, one PR to `master`, no stacked dependency. Do not merge before the pending full-repository tests and an execution receipt have been reviewed. A physically non-convergent but correctly reported experiment is not automatically a software defect; it must not be described as an accepted physical free wake.

Base: `9480516c5dd80c24d9346ee077956cc48c8bd920`.
Implementation/test commit: `0f6be3e463d2849a847224da000d220757a86aaf` (subsequent documentation-only commits do not change these source blobs).

## What is implemented

`panel/p1_coupled_material_wake.py` computes the same total potential representation as `_potentials_at`: freestream plus the shared wall doublet field plus every independent wake. Wall values are `mu[dof_face]`; each wake's nodal values are `gamma_hat @ Gamma_g`. Per-face centroids and tangential gradients are reconstructed on the CURRENT geometry, not carried as fixed world-coordinate gradients and not reduced to P0 strengths.

`TotalVelocity` vectorizes the formula in `wake_velocity.sheet_velocity`. Production `influence_block` uses the opposite double-layer sign, so the adapter applies one global minus sign to BOTH wall and wake contributions. At the original submitted HEAD, 48-point Gauss edge quadrature was used and 48→96 sensitivity was reported. **After the in-place PR #68 numerical repair below, the opt-in B2 default is analytic straight-edge integration of the P1 linear strength (including its first edge moment).** Opposite-oriented traces are combined per actual topological edge before singular contributions are evaluated; genuine cut discontinuities are retained. The legacy Gauss evaluator remains available only as `--velocity-method legacy_gauss` for frozen A/B comparisons. The epsilon ladder still measures near-sheet convergence; neither method proves a principal-value limit at a singular TE or tip.

Each accepted candidate updates only W; A/J/K/wall geometry are reused. `solve_multi` then re-solves shared mu and ALL groups' Gamma before diagnostics or a final result are returned. Iteration 0 is separately compared with the canonical fixed-wake solve, including W and CLp/CLgamma/CDp/CMy. The **original** gate compared loads to `1e-10`. Local integration revealed that it incorrectly compared two independent nonlinear solver termination paths (5 versus 6 `nfev`) even when all fixed operators matched and W differed by only ~3.47e-18. The **repaired** iteration 0 explicitly checks A/J/K/grad operator identity and W relative difference <=1e-12, then reuses the canonical fixed-wake state instead of solving the same unchanged geometry twice. Every accepted *changed* wake geometry still re-solves shared mu and each Gamma.

The geometry map is rooted fixed-material-arclength Picard iteration, not unsteady material advection:

```
X_target[root] = X_reference[root]
X_target[i] = X_target[pred(i)] + ds_reference[i] * unit(V_endpoint_average)
X_trial = X_current + relaxation * (X_target - X_current)
```

At a TE root, the first downstream node's tangent is used rather than pretending the fixed root is a freely moving sheet node. An aligned uniform wake is a fixed point; xyz can all change, with no division by Vx. Connectivity, material interpolation and roots remain fixed. Root/path discovery still inherits `prepare_wake_material`'s original upstream-x graph assumption: this is NOT a new automatic TE detector for arbitrary geometry.

A node with unresolved epsilon/order sensitivity or a vanishing tangent stays in place and remains a rejected node for convergence purposes. Whole candidate geometries are validated atomically. Backtracking responds only to geometry safety, not to desired CL or VSPAERO agreement. Area changes are checked against both the previous and reference geometry; swept convex hulls test wall and non-neighbour wake intersections. Exhausting the collision-pair budget rejects the update rather than silently skipping checks. Swept hulls are conservative and may reject noncolliding motions; declared boundary contact is tolerated by a 1e-8 relative face shrink and a 1e-10 chord projection tolerance. Topological neighbours use the local fold gate. These are explicit limitations, not a universal CAD collision certificate.

## Recorded quantities and interpretation

Every solved iteration includes CLp, CLgamma, CD_pressure, CM_y, signed and relative CLp-CLgamma gap, wall/jump RMS, both Kutta RMS values, linear-seed rank/singular minimum, and the rank/singular minimum of the selected EXACT linear constraints. The inherited `linear_constrained_rank`/`linear_singular_min` describe the signed-speed linear seed, not the final nonlinear pressure Jacobian. An initial pressure-Kutta nonlinear failure is now reported as `blocked_initial_nonlinear_solve` (versus `blocked_updated_nonlinear_solve` after an accepted move); the initial state and diagnostics remain in the receipt. A larger `max_nfev` is not a substitute for a physically compatible Kutta closure. Default Kutta is `pressure`; `signed_speed` remains an explicit diagnostic control.

Face velocity samples are evaluated at face centres, not by averaging singular nodal values. Slip, Cp jump, normal jump and Cp-identity residual have area-weighted RMS/max. All faces are included, even when nearby nodes are refused. Three fixed bands use relative ORIGINAL material arclength, not absolute chord cutoffs or moving quantiles; each reports face count, area and total area fraction. Current solver freestream speed is one, so these quantities are already normalized by U or U^2.

For finite offsets the measured normal velocity difference is not forced to zero. Analytic edge integration removes one source of spurious normal jump (underresolved edge quadrature), but normal jump sampled at distinct offset points also includes genuine O(epsilon) variation of the smooth wall/cross-wake field. Its zero-offset limit requires a separate convergence analysis; this code does not clamp it. The identity residual is `dCp + 2*Vmean.dot(Vplus-Vminus)`, not dCp itself. This identity is an algebraic check, not an independent physical-accuracy test.

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


## In-place repair following genuine local G1/G2/CAD feedback (2026-10-28)

This is a substantive numerical change to **the same PR #68** and not a
branch-stacked follow-up. The original local receipts and their failures are
retained at [PR #68 local G1](https://github.com/Stevenkerman27/Dpanel/pull/68#issuecomment-6468145085),
[G2](https://github.com/Stevenkerman27/Dpanel/pull/68#issuecomment-6468224999), and
[real CAD](https://github.com/Stevenkerman27/Dpanel/pull/68#issuecomment-6469452045).
These numbers were measured on the PRE-FIX source, not the repaired source.

| Failure in pre-fix receipt | Source-derived evidence | Repair / remaining boundary |
| --- | --- | --- |
| G1 exit 1 before physical data | A/J/K/grad equal exactly, W max delta 3.47e-18; nonlinear nfev 5 vs 6 changed CL by 5.514e-10 | Gate the actual operator. Reuse iteration-0 canonical state, preserving exact loads; all changed geometries still re-solve. |
| G1 and G2 report no wake fixed point | G1 node epsilon error max 3.66e-2 vs tol 2e-2; G2 main slip 5.25% → 4.99% → 5.20%, repeated node rejections | Retain all epsilon/slip gates and non-convergence verdicts. New exact-edge kernel may improve estimates but does not guarantee a fixed point. |
| G1/G2 signed pressure drag negative | G1 CDp≈−0.0201; G2 CDp≈−0.0235 | CLI now emits an explicit warning. These signed, cancellation-prone near-field integrals are NOT validated induced or total drag. |
| CAD iteration-0 pressure Kutta failed | 792 full-wall triangles; nonlinear nfev 80/200/600 all ended without success; wall RMS ~1.49e-3; CL drift 0.3036→0.3100 | Distinguish **initial** nonlinear failure from failure after moving wake. Do not silently switch Kutta or increase nfev. |
| CAD fixed 48/96-point Gauss sheet velocity breaks on slender wake | 770 wake faces, aspect p90=260, node quadrature discrepancy 1.23, epsilon discrepancy 0.695; 258/468 nodes refused by convergence | **Replace the numerical edge integrals by analytic straight-segment antiderivatives, including the P1 first moment.** Cancel continuous shared-edge traces before evaluation. Retain the true singular-edge/epsilon refusal. |

### Mathematics and verification of the new kernel

Let a source edge be parameterized by x(l)=a+l*t, l∈[0,L],
s=(p−a)·t, rho=p−a−s*t, q=s−L, and
R0=sqrt(|rho|²+s²), R1=sqrt(|rho|²+q²). Then

- I1 = ∫ dl/R = asinh(s/|rho|) − asinh(q/|rho|),
- I3 = ∫ dl/R³ = [s/R0−q/R1]/|rho|² (with the algebraically stable outside-segment form),
- Il3 = ∫ l dl/R³ = s*I3 + 1/R0−1/R1.

For a linearly varying edge strength mu(l)=mu_a+(mu_b−mu_a)*l/L,
its exact Biot–Savart boundary contribution is

    (t × rho) [ mu_a*I3 + (mu_b−mu_a)*Il3/L ].

For the P1 in-plane-strength-gradient term the corresponding exact edge
integral is I1(t × n). The −Omega*n component is retained exactly, as
required by the independent Phase A mathematical identity. Combining these
produces the same P1 off-sheet potential gradient as before, without any
48-point near-singular edge quadrature. The cancellation is performed only
within one physical mesh using actual shared vertex IDs; a TE/cut jump is
never spuriously smoothed or averaged.

**Independent isolated mathematical tests actually executed** on Python
3.13.5 / NumPy 2.3.5 / SciPy 1.17.0:

    PYTHONPATH=/mnt/data/dpanel_issue65_fix python -m pytest -q \
      /mnt/data/dpanel_issue65_fix/test/test_p1_analytic_sheet_velocity.py
    # exit 0, 6 passed, 0.49s

This test harness executed an independently authored local prototype with the
same stated closed-form equations, **not a complete checkout of the posted Git
branch**. It verifies adaptive SciPy line integration, independent high-order
Duffy area integration for linear doublets, continuous shared-edge cancellation
and physical cut retention, scale and winding, and aspect-ratio 260 slivers.
A matching standalone regression file is committed to this PR.

On an off-sheet triangular sliver with longitudinal length 26, transverse
width 0.1 and normal evaluation offset 0.000625, the local oracle measured
norm errors relative to the analytic edge integral of approximately **1.91**
for whole-edge 48-point Gauss and **1.62** for 96-point Gauss (unit velocity
normalization). This is a controlled fixed-strength numerical counterexample,
not a new CAD convergence result. For non-near-sheet independent linear P1
area quadrature the analytic evaluator agreed at near machine precision.

### A/B and failure states after repair

The repaired opt-in B2 command uses analytic edges by default. To compare
finite-edge quadrature on identical geometry, alpha and Gamma solve, add
\`--velocity-method legacy_gauss\` and use a different \`--out\` directory:

\`\`\`bash
python -m pytest test/test_p1_analytic_sheet_velocity.py test/test_p1_coupled_material_wake.py test/test_wake_nodes.py test/test_wake_velocity.py -q
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 4 --chord-panels 4 --span-panels 4 --wake-panels 3 --out runs/issue65_b2_analytic_g1_r2
python -m panel.run_p1_coupled_material_wake --case g2 --alpha 5 --kutta pressure --iterations 4 --chord-panels 4 --span-panels 4 --wake-panels 3 --out runs/issue65_b2_analytic_g2_r2
python -m panel.run_p1_coupled_material_wake --case g1 --alpha 5 --kutta pressure --iterations 0 --chord-panels 4 --span-panels 4 --wake-panels 3 --velocity-method legacy_gauss --out runs/issue65_b2_legacy_g1_i0_r2
\`\`\`

Run the CAD input **only where it actually exists** using the same new branch
HEAD, separate output directories and original fixture SHA. Its pressure-Kutta
iteration 0 will still be explicitly blocked if the unchanged underlying
physical nonlinear solve fails. Only a *new* actual run can establish whether
the improved off-sheet speed estimates reduce epsilon errors or allow more
eligible material nodes on the real CAD.

In analytic mode \`quadrature_error=0\` means **not applicable** (no numerical
edge quadrature); it does **not** mean the near-sheet velocity limit is exact.
\`edge_quadrature_error_interpretation\` explicitly records this. The
\`epsilon_error\`, normal-jump, slip, Kutta and mesh-quality outputs remain
necessary and cannot be suppressed to claim convergence.

Snapshots now record \`sampling_kernel\` so that the selected A/B method is
replayed exactly. Pre-repair snapshots lacking that marker are replayed using
the preserved legacy Gauss evaluator. As before, a bitwise frozen-state
replay is guaranteed only within a reproducible implementation/environment.

**Full-repository integration of this latest PR HEAD and real CAD experiments
after the repair remain pending local execution.** In particular, this
change makes no claim that pressure Kutta, a finite-epsilon PV at a tip, or a
steady free-wake physical fixed point has been solved. Keep PR Draft until
tested/reviewed; keep issue #65 open and physical_accuracy_accepted=false.


## Independent wall Cp / velocity-trace diagnostics (added 2026-10-28)

New \`panel/p1_cp_diagnostics.py\` intentionally does NOT modify \`state['cp']\`,
clip it, smooth it, or update the solution. It separates:

1. \`p1_surface_gradient(xyz, faces, dof_face, mu)\`: independently rebuild the
   per-triangle P1 basis gradient and report shortest edge, aspect ratio,
   edge-matrix condition number, and gradient-operator norm.
2. \`potential_gradient_fd4(potential, points, step)\`: a fourth-order centered
   finite-difference gradient of a caller-supplied scalar potential.
3. \`compare_p1_to_offsurface(...)\`: compare the current \`grad(mu)\` Cp with
   an independent total-potential gradient sampled on a declared fluid-side
   normal-offset ladder. It also reports the alternative \`U_t+grad(mu)\`
   convention, so a missing/double-counted freestream convention cannot be
   silently assumed. Offsets default to fractions of each face's shortest
   edge, not a global length scale.
4. \`diagnose_g2_state(mesh, state)\`: adapter for the existing G2 solve state;
   it uses \`p1_steady_free_wake._potentials_at\`, i.e. freestream + shared wall
   + every P1 wake, and reports exact recomputation consistency against the
   solver's Cp plus independent off-surface Cp/velocity traces.
5. \`kutta_pressure_diagnostic\` and \`pressure_load_diagnostic\`: per-station
   Cp/speed-squared jump and the unclipped signed pressure-force/drag
   cancellation budget. Equal-face Kutta averaging matches the current G1/G2
   residual; area-weighted averaging is a separately selected alternative.

This tool requires the caller to verify outward normals / fluid-side sign and
that all finite-difference shifted points remain in the exterior. It does not
automatically certify CAD winding, detect crossing nearby surfaces, or prove a
pointwise principal-value limit. It is an audit path, not a solver change.

### Local isolated execution (not full-repository integration)

\`\`\`bash
PYTHONPATH=. python -m pytest test/test_p1_analytic_sheet_velocity.py \\
  test/test_p1_cp_stability_manufactured.py test/test_p1_cp_diagnostics.py \\
  -q -p no:cacheprovider
# 20 passed / exit 0 in the isolated sandbox
\`\`\`

A manufactured harmonic potential \`phi=x+0.1 sin(x) cosh(z)\` has exact
zero normal velocity on the flat wall \`z=0\`. Across 12/24/48/96 chordwise
panels, surface P1 Cp RMS error vs the analytic wall trace is respectively
0.01239 / 0.00618 / 0.00309 / 0.001545; independent fourth-order
off-surface-gradient Cp error is 1.97e-5 / 1.22e-5 / 3.04e-6 / 7.60e-7.
This is a controlled smooth manufactured solution, NOT a real-aircraft
validation. It shows the two methods have distinct, measurable errors and
that the comparison converges with refinement. The generated JSON is saved
outside the repo as \`dpanel_cp_diagnostic_demo_20261028.json\`.

### Required next actual-repository run

Once a local checkout of this exact branch is available, run the existing
G1/G2 solve at fixed geometry, then call:

\`\`\`python
from panel.p1_cp_diagnostics import diagnose_g2_state
audit = diagnose_g2_state(mesh, state, offsets=(0.2, 0.1, 0.05),
                          offset_sign=+1.0, local_scale="edge_min")
\`\`\`

Save the report and compare: solver-Cp recomputation error (should be near
roundoff), off-surface Cp vs solver Cp for each offset, tangential-velocity
discrepancy for both potential conventions, normal-velocity residual, local
edge conditioning, per-TE pressure residual, and signed drag cancellation.
Repeat with a mesh-refined geometry and a verified outward-normal sign. Do not
run this adapter on the true CAD until full source checkout and unit tests pass.
The sandbox cannot resolve GitHub DNS, so **the 20 isolated tests do not claim
that this adapter has run on a real G1/G2 solve or real CAD**.
