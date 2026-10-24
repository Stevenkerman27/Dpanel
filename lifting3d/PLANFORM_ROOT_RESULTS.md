# Planform Root Geometry Matrix

## Protocol

Five legal merged-root planforms run serially in this fixed order:

| taper ratio | quarter-chord sweep |
| ---: | ---: |
| 1.0 | 20 degrees |
| 0.5 | 0 degrees |
| 0.5 | 20 degrees |
| 1.0 | -10 degrees |
| 1.5 | 0 degrees |

Shared inputs: thickness 0.24 chord, body length 4 chords, body radius 0.35
chord, body aft coordinate 1.5 chords, mesh size 0.25 chord, curvature 16,
16 requested base wake strips, merged root wake, weak Kutta, Schur linear solver, Mach 0,
angle of attack 5 degrees, NCPU 8. Each mesh is limited to 90 seconds and
each solve to 180 seconds. A failed mesh, mesh test, or solve is preserved
and stops the batch; later cases are not silently substituted or skipped.
For merged-root geometry the builder inserts two analytic root-contact
endpoints in addition to the requested base partitions, so the observed
`actual_wake_strips` is 18 while the requested `wake_strips` remains 16.

The existing `test_mesh.py` must pass all eight checks before each solve. The
driver records every command, return code, elapsed time, timeout state, logs,
mesh tests, result, raw field files, and SHA-256 hashes. It compares start/end
hashes of the builder, solver, planform root implementation, root-wake module,
Schur implementation, mesh test, and solver/root-wake protocols.

Each mesh must match frozen geometry metadata, include wall tags 20 and 30,
declare constructed root support, and pass all eight independent mesh checks.
Each solve must converge, have finite numerical raw fields, report a strictly
positive weak-Kutta test measure for every reduced Gamma degree of freedom,
and permit complete expansion of the saved station-to-DOF map. Pressure force
is independently reintegrated from raw quadrature data only on physical wall
tags 10, 20, and 30; outer boundary and wake tags 100, 201, and 202 are
excluded. The integrated force must close against the component force sum to
1e-10.

No Cp accuracy, absolute lift accuracy, mesh convergence, or physical
validation is claimed by these geometry and algebra gates.

## Execution Results

The first v1 driver attempt generated the first mesh successfully, then
incorrectly rejected it because the audit expected exactly 16 actual strips
and disallowed optional wall tag 10. The failure snapshot is preserved as
`results/planform_root_v1/attempt_001_progress.json`. The corrected v2 run
reused the first mesh without rewriting it; its SHA-256 remained
`37841fac6e848abe951d691c478d8151a56781efaed7499c0e3ab607567d2460`.
Here, “v2 run” means attempt 2 in the same `results/planform_root_v1/`
directory, started with `run_planform_root.py --resume`; it is not a separate
v2 output directory. The v1 progress snapshot, original mesh, and its hash
remain in that directory.

| taper | sweep | points | tets | requested/actual strips | CL pressure | CL Gamma | CD pressure | mesh SHA-256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 20 | 63,924 | 41,486 | 16/18 | 0.3758680 | 0.3843790 | 0.0118937 | `37841fac6e848abe951d691c478d8151a56781efaed7499c0e3ab607567d2460` |
| 0.5 | 0 | 65,609 | 42,422 | 16/18 | 0.4194812 | 0.4330694 | 0.0120712 | `26ea5ee85698e37cec444cc45da2293bc4662b7f1ea912cfb7749c928c5fcb3e` |
| 0.5 | 20 | 65,920 | 42,608 | 16/18 | 0.4272998 | 0.4364222 | 0.0122807 | `478768ea08d8a241a66e000810b94701d2e667ac9e5888963b1882082181f028` |
| 1.0 | -10 | 64,737 | 41,844 | 16/18 | 0.3632883 | 0.3767888 | 0.0118376 | `20b761e746a0c7aebcdab7508c0fababbefc99c37e4624d292dd5f6c3e3e4f2a` |
| 1.5 | 0 | 63,176 | 41,122 | 16/18 | 0.3356232 | 0.3463123 | 0.0121975 | `604f65d668c3b1d74f5a76bf7a94cf4eb03b24ccf4b8fb10897f47000a3974cf` |

All five meshes passed 8/8 `test_mesh.py` checks. All solves converged in
26-28 seconds with weak-mass residual norms from `5.49e-14` to `5.77e-14`,
Kutta maxima from `1.20e-14` to `2.26e-14`, and strictly positive weak-Kutta
measures (per-case minima 0.00986 to 0.03502). The post-run raw-field audit
reconstructed every station jump from the saved reduced `gamma` and map.
Wall forces were reintegrated from tags 20 and 30 only; tags 100, 201, and
202 were excluded. The largest wall-force/component-sum discrepancy was
`9.94e-15`. Detailed values and gate booleans are in
`results/planform_root_v1/independent_audit.json`.

The source hashes recorded at run start and finish match for every frozen
builder, solver, root-mapping, wake, Schur, test, and protocol file. The
current `run_planform_root.py` now also performs the direct raw-reduced-Gamma
expansion check; `driver.py.snapshot` preserves the exact driver bytes used
by v2. After v2 completed, the current driver was changed only to record
requested/actual strip counts explicitly and strengthen the map gate to
reconstruct station jumps directly from saved reduced `gamma`. Thus the
current source is intentionally not byte-identical to the executed snapshot.
The independently saved post-run audit applied that stronger raw-Gamma check
to all five v2 results and passed. These are algebraic/geometry checks only,
not a Cp or aerodynamic accuracy certification.
