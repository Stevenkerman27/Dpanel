# Issue 77 / PR 78 — wake/body contact trace repair and measured acceptance

## Actual cause

The independent local review correctly located the remaining Cp extremes on the wake/body contact curve, downstream of the wing TE. The CAD surface already declares these edges as `smooth_wall_contact`, alongside `trailing_edge` cuts. The P1 adapter nevertheless split potential DOFs only on the TE; both banks of each smooth contact edge shared one value of mu. A nonzero wake potential jump was therefore attached to an artificially continuous wall trace. The resulting near-contact gradient and pressure were not a physical body pressure distribution.

A second interface defect was found at the root: geometric-neighbour Kutta sampling selected body triangles. At one root, the two upper samples had CAD tags [20,30] (body,wing) and both lower samples tag20. Kutta must use the actual declared TE-bank faces, not whichever same-bank centroid is nearby.

The correction does NOT round the body, remesh, clip Cp, tune a target CL/CD, add regularization or replace impermeability with Robin data. It completes the existing cut topology and preserves the declared physical model.

## Method

For every declared wall/wake shared edge, require two oppositely wound wall faces and one wake boundary face. Split each geometric vertex's local incident face fan only across the declared cut edges. The plus wall bank has opposite edge winding to the wake; the minus bank has the same winding. Impose

    mu_plus - mu_minus = H_contact Gamma

using the EXACT wake interpolation at the shared geometric vertex. This cancels the artificial edge vortex. Junction nodes shared by TE and contact get only one jump row. Contact compatibility adds no pressure/speed Kutta equation. The original 46 TE Kutta rows and 18 Gamma unknowns remain; conv gains 67 contact trace DOFs and 67 compatibility rows. Only the exterior wing tips are zero-Gamma endpoints; the off-centre wing roots remain free.

The two-bank convention follows edge orientation, not the sign of global normal_z. A rotation-invariance oracle verifies this. For the contact CAD, true TE face pairs are recovered from declared cut-edge incidence. All three no-contact fixtures return the original Wing object without modification.

Primary methodological reference: MachLine Modeling, wake shedding and body/wake intersection potential-jump cancellation: https://github.com/usuaero/MachLine/wiki/Modeling . This motivates the compatibility relation; the DPanel tests and controlled ablation establish its implementation here.

## Executed sandbox evidence (2026-10-29)

The full source snapshot was exported by GitHub Actions, not re-created from isolated formulas. Snapshot merge ref: `bb6fd7e59492a615d79750de2aeafabc5af2a943`, branch head at export `d5e11d6d34bfa09e089b318cb3f97b7810645d32`. The modified source files are committed in this PR and their blob hashes can be compared directly with the sandbox files.

Input: `cases/geom_robust/surfaces/conv_s075/surface.json`; SHA256 `3fa433761a98e6eb3bf708124783c2a1a7b138de418abaa4105c99a04e6ce172`. All runs use the same 6224 wall triangles, unchanged wake geometry and Gamma basis, alpha5deg, target quadrature order2, and single-threaded BLAS. Python3.13, NumPy2.3.5, SciPy1.17.0. The standard Cp expression and force integration are unchanged.

| Variant | CLp | CLgamma | CDp total | CDp body | body max abs Cp | constrained condition |
|---|---:|---:|---:|---:|---:|---:|
| Earlier PR78: TE-only split | 0.4060144343 | 0.4114293439 | 0.0998550679 | 0.0959382662 | 15.47497247 | 353.0713 |
| Contact split only; old Kutta sampling | 0.4746695635 | 0.4799008306 | 0.0129259786 | 0.0057766002 | 1.05599135 | 306.8311 |
| Contact split + declared true TE sampling | 0.4746467377 | 0.4798586233 | 0.0129238569 | 0.0057749460 | 1.05598401 | 308.9496 |
| Same corrected operator, pressure Kutta | 0.4658279948 | 0.4709025655 | 0.0121672079 | 0.0056696821 | 1.05159291 | not comparable to signed-Kutta nullspace |

The first three rows use signed-speed Kutta. The contact-only ablation removes the body pressure anomaly before changing the sampler, establishing the main causal mechanism. The final signed solve has contact jump RMS1.647e-15, max4.108e-15, signed-Kutta RMS1.165e-13. The pressure solve succeeds in 7 evaluations (`ftol`), pressure-Kutta RMS9.265e-9, contact jump RMS1.674e-15. A small Kutta residual alone is still not aerodynamic validation.

The full related suite was executed on the modified snapshot: **83 passed / exit0**, final rerun4.15s. It includes contact rotation/sign tests, missing-cut rejection, manufactured piecewise potential with correct bankwise gradients, real-CAD topology/TE sampling, G1/G2 coupling, shared-status tests, shadowed-test detection, prior TE/Galerkin/edge tests and Cp diagnostics. The CP diagnostics module also had a pre-existing mismatched bracket; this was corrected and its tests are now included.

## Review hygiene corrections

Duplicate CAD test definitions were removed; an AST guard rejects future shadowed tests. Solver and sweep use one `assess_solution` implementation. The column-norm ratio is explicitly a warning/proxy: the full original conv case had min/median0.21 and was detected by its enormous loads, NOT the0.1 support threshold. A same-case CD-reference warning is available, but different aircraft are never used as a universal drag limit. A component with no drag contributions now has ratio=null and status=no_drag_contributions; complete cancellation is separately labelled.

## Reproduction and next gate

    python -m pytest test/test_p1_wake_contact.py test/test_p1_solution_status.py test/test_p1_edge_singularity.py test/test_p1_te_trace_audit.py test/test_p1_te_jump_integral.py test/test_p1_galerkin_wall.py test/test_p1_galerkin_solver.py test/test_p1_galerkin_cad.py test/test_p1_lifting_g1.py test/test_p1_lifting_g2.py test/test_p1_cp_diagnostics.py -q
    python -m tools.issue71.wake_contact_ab --order 2 --out runs/issue77_contact_ab.json
    python -m tools.issue71.wake_contact_ab --variants fixed --methods signed_speed pressure --order 2 --max-nfev 20 --out runs/issue77_contact_fixed_both.json

Local execution/review should verify the same results on the final PR head and merge through PR78. No additional test-only PR is required. Follow-up work is quadrature/grid refinement with the CORRECT contact trace space, additional contact-CAD cases, and then free-wake coupling. This finite set of runs fixes the identified topology defect; it does not establish all-geometry physical accuracy, zero body drag, or convergence of pressure drag under every refinement.

## Follow-up review fixes on PR #78

The independent review also found validation/reporting gaps. The latest tested code snapshot `f856a5f4f2b34aa9effd0714157e16a8e29958ec` closes them:

- **91 regression tests passed in 2.91 s**. Five additional rejection-path tests cover a wake edge that is not a boundary, malformed bank orientation, branched cuts, open cut endpoints with nonzero Gamma support, and changed mirrored geometry. The existing missing-cut test covers the declared-vs-actual edge mismatch.
- `contact_jump` is now recomputed and written into the CAD receipt for both Galerkin and collocation states; `mesh_input.trace_audit` and `trace_audit_by_group` record `split_nodes`, `contact_jump_rows`, `unsplit_zero_gamma_tips`, TE/contact edge counts and the fact that contact rows are not Kutta rows.
- `pressure_load_diagnostic` and the CAD sweep use the same explicit null convention for undefined drag-cancellation ratios: `null` plus `no_drag_contributions` or `complete_cancellation`.
- The duplicate-test-name AST guard remains active, and catastrophic-solution screening is centralized in `assess_solution`; the weak Gamma-column ratio remains diagnostic only.

The new test suite initially caught a fixture-construction error (`np.r_` interprets strings as special directives); this was corrected by using `np.concatenate`. The final 91-test run is green. The two CAD solve jobs for this follow-up are being rerun on the same code snapshot; see the current Actions runs linked from PR #78 before merge. `physical_accuracy_accepted=false` remains intentional.