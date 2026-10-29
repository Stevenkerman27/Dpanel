# Issue #77 — Gamma support mismatch at the physical TE root

## Root cause

`conv_s075` has a fuselage-contact interval in the half-wake mesh at y = 0 and 0.2, but the actual exposed main-wing TE begins at y = 0.249996. The exported `gamma_stations` describe the wake/contact discretization, not necessarily the independent circulation basis for the exposed TE.

`from_dpanel_half_surface` previously copied every station into the full-span Gamma basis. This created independent basis functions at y = 0 and ±0.2 with nearly zero wall influence, while the TE/Kutta constraints still acted on those DOFs. The constrained least-squares problem remained full rank and its wall residual was small, but three near-null directions were strongly excited: `cond(B@N) ≈ 4.7e5`, `||mu|| ≈ 228`, `||gamma|| ≈ 298`, `CL ≈ 1328`.

## Fix

The loader now derives the physical half-span support from the declared upper/lower TE face pairs. It removes inboard contact stations from the solver's independent Gamma interpolation knots, but only if the exported `gamma_station_dof` explicitly ties those stations to the actual TE-root DOF. If an inboard station is an independent DOF, loading fails with a clear `ValueError` rather than silently discarding it.

The physical wake geometry is unchanged: its y=0/0.2 contact nodes remain and are interpolated across the symmetric root gap between -y_root and +y_root. Both mirrored TE-root stations are retained. This matters because the standard construction `[-half_y[1:], half_y]` assumes the half grid starts at zero; when the physical TE root is off-centre, that construction would silently drop the negative root and create an asymmetric Gamma space. A regression asserts symmetric stations and both root banks.

Other three CAD fixtures have their TE starting at the symmetry plane, so their Gamma basis is unchanged.

## Solution plausibility gate

The Galerkin solve now reports the constrained singular-value condition estimate, minimum/maximum singular value, `||mu||`, `||gamma||`, min/median wake-column norm, and a separate `solution_status`. It labels nonfinite states, |CL| > 20, |CDp| > 1, `||gamma|| > 50 ||mu||`, condition > 1e6, or a near-zero-support Gamma column as `implausible_solution`. The CAD sweep applies the same gross-load/norm gate to collocation and propagates implausible status to the model-level report. This is a catastrophic-failure gate, not an aerodynamic-accuracy acceptance test.

## Validation required

Run the related test suite and all four full CAD solves through `.github/workflows/issue77-gamma-support.yml`. The expected result is that conv no longer has zero-support independent Gamma columns, the constrained condition returns to the same order of magnitude as the other models, and its loads are finite and plausible. `CDp ≈ 0.106` from the station-clipping experiment remains a separate open accuracy issue and is not claimed fixed here.

`physical_accuracy_accepted=false`.
## Status after the later wake/body contact-trace repair

**This document records the intermediate Gamma-support-only stage. Its statement that `CDp ≈ 0.106` remains an open issue is superseded by [`ISSUE77_CONTACT_TRACE_FIX.md`](ISSUE77_CONTACT_TRACE_FIX.md).** That value came from a solver with the physical TE Gamma support corrected but the `smooth_wall_contact` potential trace still artificially continuous.

On the same frozen 6224-face `conv_s075` geometry, adding the declared contact jump and using the actual TE-bank faces gives `CL=0.4746467377`, `CL_gamma=0.4798586233`, `CDp=0.0129238569`, body `CDp=0.0057749460`, body `max|Cp|=1.055984`, and constrained condition estimate `308.95`. The controlled contact-only ablation is the causal evidence; do not treat the old `CL=0.4060` / `CDp=0.099855` row as the final contact-compatible solution.

This is still not an all-geometry physical-accuracy certification. Contact-jump compatibility, true-TE Kutta, quadrature/grid convergence, and additional contact-CAD models remain separate acceptance checks.