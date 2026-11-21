# Standard-model calibration

Run the calibration suite from the repository root in the project `myml`
environment. Theory and experimental tables are in
`benchmarks/calibration/reference.json`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m tools.calibration --out runs/verification/current_area_calibration
python -m tools.calibration --case C05 --refine --out runs/verification/calibration_refine
```

The first command writes one `summary.json` with the reported solver, geometry,
conditions, and measurements. The second includes a three-grid elliptic-wing
study. The suite reports differences without setting acceptance thresholds.
For the current worktree status, see [STATUS.md](../../STATUS.md).

## Measurements on 2026-11-21

Windows, Python 3.13.14, Torch 2.13.0+cu132, NumPy 2.4.6, SciPy 1.18.0,
single-threaded CPU. C04 and C05 use fixed wake with `Galerkin + pressure + area`.
The first suite run stopped while constructing the C05 report because its
`refinement` field was unsupported. The corrected run completed all six records
and solved C02, C04, and C05. The optional C05 refinement was not run.

| Case | Comparison | Result |
| --- | --- | --- |
| C02 sphere | Analytic \(C_p=1-\frac94\sin^2\theta\) | Area-weighted \(C_p\) RMS error was 0.05762, 0.02847, and 0.01316 on 80, 320, and 1280 faces. Finest-grid net force coefficient norm was \(1.80\times10^{-15}\). |
| C04 thin wing | Finite-span lifting-line estimate for AR 16 | At 1°, 2°, and 3°, \(C_L\) was 0.08960, 0.17900, and 0.26806. The fitted slope was 5.1126/rad versus 5.5851/rad (−8.46%). |
| C05 elliptic wing | Lifting-line \(C_L\), elliptical circulation, and ideal induced drag | See the 4° results below. |

| C05 target AR | Mesh AR | \(C_L\) | LLT \(C_L\) | Relative difference | Trefftz \(C_{D_i}\) | \(e\) | Normalized circulation RMS error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 4.0403 | 0.26474 | 0.29341 | −9.77% | 0.005415 | 0.9716 | 7.89% |
| 6 | 6.0605 | 0.32217 | 0.32981 | −2.32% | 0.005232 | 0.9819 | 6.04% |
| 8 | 8.0806 | 0.35681 | 0.35162 | +1.48% | 0.004764 | 0.9869 | 4.94% |

C05 uses 1560 faces per wing and a finite tip chord. Mesh AR is \(b^2/S_{\rm ref}\).
The reported \(e\) uses circulation-derived lift and Trefftz induced drag.
C02 measures surface pressure. An exterior probe is still needed to measure
normal-velocity leakage.

## Two-dimensional cases

| Case | Evidence |
| --- | --- |
| C01 cylinder | The P1 suite lists the analytic reference. The separate [potential-flow study](../../potential/RESULTS.md) reports cylinder pressure errors. |
| C03 Joukowski | The separate [P2 airfoil study](../../airfoil2d/RESULTS.md) reports finest-grid \(C_L=0.586420\) versus 0.588180 analytically, with \(C_p\) RMSE 0.01127. |
| C06 NACA 0012 | The suite stores experimental conditions and fits the Ladson small-angle slope (0.107994/deg at \(Re=6\times10^6\), \(M=0.15\), 80-grit). Gregory upper-surface \(C_p\) data are at \(Re=2.88\times10^6\). The solver has yet to be compared against either dataset at matched conditions. |
