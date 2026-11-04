# Galerkin batched assembly benchmark

Date: 2026-10-29. Commit: `01746bb733dd2e3ff7df249028058dc3676022c1`.
CI run: https://github.com/Stevenkerman27/Dpanel/actions/runs/40820079672

## Method

`tools/issue71/galerkin_batch_benchmark.py` compares the previous per-target-face Galerkin assembly loop against batched target-face assembly. Both paths use the same geometry, P1 analytic source kernel, Duffy quadrature order 3, wake basis and double precision. The report checks relative L2 and maximum absolute differences for A, W and b; every case must have relative L2 error below 1e-11.

Environment: GitHub Actions Ubuntu 24.04, Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, single-threaded BLAS.

| Chord panels | Wall faces | Previous assembly | Batched assembly | Speedup | Equivalent A/W/b |
|---:|---:|---:|---:|---:|---|
| 4 | 144 | 12.428 s | 0.384 s | 32.38× | yes |
| 8 | 288 | 44.985 s | 1.209 s | 37.22× | yes |
| 12 | 432 | 98.088 s | 2.514 s | 39.01× | yes |

These timings are for generated G1 meshes with 8 span panels and 2 wake panels. They demonstrate the effect of batching, not the total cost of the dense solve or the full CAD path.

## Related test result

The complete related suite (edge singularity, TE topology/jump, Galerkin wall/solver, four CAD fixture import tests, G1 and G2 regression tests) passed: **53 passed in 4.58 s**.

## CAD validation status

The linked CI run completed successfully. It assembled all four committed CAD-derived surfaces (`elliptic_s075`, `conv_s075`, `swept30_s075`, `bwb_s055`) and solved two representative cases (`elliptic_s075`, `swept30_s075`) with signed-speed Kutta, comparing collocation and Galerkin on identical geometry. The later Issue #77 workflow additionally runs the repaired `conv_s075` plus unchanged `swept30_s075` control and exports per-model receipts.

`physical_accuracy_accepted=false`: the speed benchmark validates assembly equivalence and performance only; it does not certify absolute aerodynamic accuracy.