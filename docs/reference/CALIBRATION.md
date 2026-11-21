# Standard-model calibration

Run the calibration suite from the repository root in the project `myml`
environment. Reference tables are in
`benchmarks/calibration/reference.json`.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m tools.calibration --out runs/verification/macro_calibration
python -m tools.calibration --case C05 --refine --out runs/verification/macro_calibration_refine
```

The first command writes one `summary.json` with the three P1 cases, solver, geometry,
conditions, and measurements. The second includes an AR 6 elliptic-wing study
on 272, 740, and 1560 faces. Pytest applies the limits below to small, independent
reference cases.
For the current worktree status, see [STATUS.md](../../STATUS.md).

## Automated Reference Checks

| Case | Mesh and condition | Check |
| --- | --- | --- |
| Sphere | 80, 320, 1280 faces | Successive \(C_p\) RMS error ratios below 0.70, fine-grid RMS below 0.02, net force coefficient norm below \(10^{-10}\). The measured ratios were 0.49 and 0.46. |
| Thin wing | AR 16, 4×16 and 6×24, 1° and 3° | The slope approaches the lifting-line estimate with refinement. The 6×24 error is within 12%. Measured slope errors were −15.03% and −10.45%. A separate 2° probe on 8×32 and 10×40 gave −8.66% and −8.02%. |
| Elliptic wing | Nominal AR 6, 272 faces, 4° | Within 6% of the lifting-line lift, normalized circulation RMS below 0.09, and span efficiency above 0.94 with the ideal upper bound. On 272, 740 and 1560 faces, measured \(e\) was 0.9737, 0.9808, 0.9819. |
| Free wake | Same 272-face thin wing, 0°, four updates | Require convergence, \(\lvert C_L\rvert<10^{-3}\), raw mapping residual below \(10^{-3}\), normal slip RMS below \(5\times10^{-4}\), and wake \(C_p\) jump RMS below \(5\times10^{-5}\). The 2° probe completed four updates with mapping residual 0.0548 and did not converge. |

## Measurements on 2026-11-21

Windows, Python 3.13.14, Torch 2.13.0+cu132, NumPy 2.4.6, SciPy 1.18.0,
single-threaded CPU. C04 and C05 use fixed wake with `Galerkin + pressure + area`.

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

The archived NACA 0012 measurements remain in `benchmarks/calibration/reference.json`.
