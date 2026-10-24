# Measured Results

## Closed Bodies

| Case | N | Cp RMSE | Centroid RMS | Face-interior RMS | Passed |
|---|---:|---:|---:|---:|---|
| sphere_L2 | 128 | 0.0404959 | 3.86e-16 | - | True |
| sphere_L3 | 512 | 0.0188443 | 7.01e-16 | - | True |
| sphere_L4 | 2048 | 0.00880907 | 1.35e-15 | 0.01394 | True |
| prolate_ellipsoid_L2 | 128 | 0.0260086 | 1.66e-16 | - | True |
| prolate_ellipsoid_L3 | 512 | 0.0118979 | 3.12e-16 | - | True |
| prolate_ellipsoid_L4 | 2048 | 0.00548349 | 5.83e-16 | 0.006529 | True |
| sphere_L3_scale_0p001 | 512 | 0.0188443 | 7.01e-16 | - | True |
| sphere_L3_scale_1000 | 512 | 0.0188443 | 7.01e-16 | - | True |
| sphere_L3_translate | 512 | 0.0188443 | 6.59e-16 | - | True |
| sphere_L3_rotate_axis_1_2_3_deg37 | 512 | 0.0188443 | 7.05e-16 | - | True |

Cp reference: radial projection of the flat centroid onto the analytic smooth surface.

## Rejected Lifting Prototype

| Case | N | CL pressure | CL wake diagnostic | Normal RMS | Source flux | Kutta max |
|---|---:|---:|---:|---:|---:|---:|
| wing_t33_a-5_w50_constrain | 2150 | 0.0325743 | 0.0384455 | 0.001516 | -1.29e-17 | 7.55e-15 |
| wing_t33_a-5_w50_report | 2150 | 0.0328735 | 0.0384397 | 9.749e-16 | -0.001522 | 7.55e-15 |
| wing_t33_a0_w50_constrain | 2150 | 0.000384373 | -8.95921e-05 | 0.001911 | 1.084e-19 | 9.99e-16 |
| wing_t33_a0_w50_report | 2150 | 0.000380682 | -9.70309e-05 | 3.552e-16 | -0.001919 | 1.11e-15 |
| wing_t33_a5_w50_constrain | 2150 | -0.0313234 | -0.038624 | 0.002291 | -3.524e-18 | 4.33e-15 |
| wing_t33_a5_w50_report | 2150 | -0.0317843 | -0.038633 | 9.355e-16 | -0.0023 | 7.22e-15 |
| wing_t33_a5_w10_constrain | 2150 | -0.0313412 | -0.0386473 | 0.002291 | -1.298e-17 | 7.66e-15 |
| wing_t33_a5_w200_constrain | 2150 | -0.0313226 | -0.038623 | 0.002291 | -2.523e-17 | 6.44e-15 |
| wing_t17_a0_w50_constrain | 550 | 0.0013627 | -0.000763674 | 0.002933 | 7.589e-19 | 2.22e-16 |
| wing_t17_a5_w50_constrain | 550 | -0.0273494 | -0.0688508 | 0.003908 | -1.334e-17 | 1.55e-15 |
