# Standard-model calibration (issue #115)

Issue #115 asks for a calibration suite that walks from analytic potential flow to
the NACA 0012 experiment, in this order: C01 cylinder, C02 sphere, C03 Joukowski,
C04 thin wing, C05 elliptic wing, C06 NACA 0012. This module implements that
suite as `tools/calibration.py`; the frozen reference numbers live in
`benchmarks/calibration/reference.json` and were extracted from the issue text
(analytic rows are exact, the NACA rows are experimental and carry their
Reynolds/Mach/transition conditions).

    python -m tools.calibration --case C02 --out runs/verification/calibration
    python -m tools.calibration --out runs/verification/calibration

Every case writes `summary.json` following the field template of the issue
(section 7.2): case id, geometry source, solver commit, solver mode, flow
conditions, mesh, residuals, Cp errors, CL/CL_circulation/CD/CDi, and
`physical_acceptance = "not_assessed"`. No thresholds are applied anywhere: the
report states the numbers and the difference from the reference.

## Status

| case | status | what runs |
| --- | --- | --- |
| C01 cylinder | blocked | no 2D panel path in the repository; reference table frozen |
| C02 sphere | ran | P1 closed-body solver, icosphere levels 1-3 |
| C03 Joukowski | blocked | needs a 2D path with a sharp trailing edge |
| C04 thin wing | ran | 3D thin wing AR 16 against the lifting-line limit |
| C05 elliptic wing | ran | elliptic planform AR 4/6/8 against LLT and Trefftz |
| C06 NACA 0012 | blocked | needs the 2D path for a like-for-like Cp comparison |

C01, C03 and C06 are blocked for the same structural reason: this repository
solves 3D surfaces only (P1 doublet on wing or closed body). A long capped
cylinder mid-section must not be substituted for the 2D analytic answer, and the
NACA 0012 pressure data are 2D upper-surface data. The blocked records keep the
reference tables so the cases run as soon as a 2D path exists.

## Readings on 2026-11-10

C02 sphere, Cp error at the face centre against Cp = 1 - 9/4 sin^2(theta):

| level | faces | Cp L2 | Cp Linf | L2 ratio to previous | max radius error | net force |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 | 5.762e-02 | 9.169e-02 | - | 3.6e-02 | 8.6e-16 |
| 2 | 320 | 2.847e-02 | 8.535e-02 | 0.494 | 1.6e-02 | 7.4e-16 |
| 3 | 1280 | 1.317e-02 | 3.433e-02 | 0.462 | 4.5e-03 | 8.6e-16 |

The L2 error halves when the mesh halves, so the closed-body path is first order
in h on this body, and the pressure force cancels to machine zero (zero net force
is the analytic result for a closed body in uniform inviscid flow). The residual
gap to the exact Cp is the faceted geometry plus the affine face model, not an
algebra failure: the algebraic system is solved by untruncated QR.

C04 thin wing (thickness 0.02, AR 16, 8 x 64 panels, potential-jump Kutta):

| quantity | value |
| --- | --- |
| CL at 1, 2, 3 deg | 0.090398, 0.180638, 0.270562 |
| CL_alpha (per rad) | 5.16133 |
| 2*pi (2D limit) | 6.28319 (-17.9%) |
| lifting line 2 pi AR/(AR+2) | 5.58505 (-7.6%) |

The wing is below the 2D limit as it must be; the remaining 7.6% gap to the
finite-AR lifting line is the finite-span discretisation error, reported here
rather than hidden behind a threshold.

C05 elliptic wing (span 8, 9 sections, 6 chordwise panels per section, alpha 4 deg,
potential-jump Kutta, Trefftz drag):

| AR | CL | CL_alpha | LLT CL_alpha | rel | CL at 4 deg | LLT CL | CDi | e | Gamma shape RMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.27267 | 3.9057 | 4.1888 | -6.8% | 0.27267 | 0.29243 | 5.613e-03 | 0.9804 | 8.1% |
| 6 | 0.33280 | 4.7670 | 4.7124 | +1.2% | 0.33280 | 0.32899 | 5.460e-03 | 0.9917 | 6.2% |
| 8 | 0.36949 | 5.2925 | 5.0265 | +5.3% | 0.36949 | 0.35092 | 4.999e-03 | 0.9970 | 5.1% |

Span efficiency approaches one as the aspect ratio grows (0.980 -> 0.997), and
the spanwise circulation approaches the elliptic shape (RMS 8.1% -> 5.1%). AR 4
is where the slender-wing assumption of lifting-line theory is weakest, which is
why the issue flags it as a stress case rather than a reference.

## Known gaps

1. No 2D path, so C01, C03 and C06 cannot run. This is the next structural gap
   to close if the calibration order of the issue is to be followed.
2. C04 reports the wing CL_alpha, not a section lift coefficient; extracting
   mid-span section lift (2 Gamma / (U c)) is the correct 2D-to-3D bridge and is
   not wired in yet.
3. `max_normal_velocity_ratio` is left null for C02: the P1 recovery velocity is
   in-plane by construction and cannot diagnose wall leakage. An exterior-probe
   audit through `velocity_fd` is the honest measurement and is not wired in.
4. Mesh convergence is reported for C02 only; C04 and C05 still need their three
   refinement levels.
