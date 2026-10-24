# Wing-Body Geometry Robustness Results

## Outcome

The authorized 11-case matrix completed in
`results/geometry_robustness_v2/progress.json` with status
`completed_with_case_results`. All 11 cases are `solver_sanity_passed`;
`test_mesh.py` reports 8/8 passing checks for every mesh. This is a
geometry/topology and numerical-sanity result only, not an aerodynamic
accuracy or general lifting-solver validation.

All cases used `--root-wake merged`, NCPU 8, weak Kutta, Mach 0, alpha 5
degrees, requested wake strips 16, `h=.25`, curvature 16, and farfield size
`5/3`. The builder produced 18 actual wake strips in every case to conform
to its merged-root partition. Across cases, meshes contain 60,647-75,101
points and 39,724-47,088 Tet10 elements. The smallest dense-sampled Jacobian
was positive (`4.8871025e-7`). CAD root reconciliation maximum shifts range
from `3.6072345e-10` to `1.9936345e-9`, below the builder-recorded
`1e-8` tolerance.

The final weak residuals range from `5.4457e-14` to `5.6787e-14`.
Component-force closure errors range from `2.22e-16` to `7.77e-15`.
Individual solver times range from 21.2 to 29.6 seconds; complete sequential
campaign elapsed time was 376.1 seconds. Raw wall/junction, Cp, force, wake,
and nonlinear histories remain in each case's JSON artifacts and are
diagnostics, not accuracy gates.

| Case | t | r | aft | Points | Tet10 | Min. Jacobian | CAD max shift | Strips req./actual | Mesh tests | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `t012_r035_a120` | .12 | .35 | 1.2 | 62,615 | 40,700 | 6.0773e-7 | 1.0758e-9 | 16/18 | 8/8 | passed |
| `t024_r035_a120` | .24 | .35 | 1.2 | 62,567 | 40,562 | 2.5371e-6 | 1.0758e-9 | 16/18 | 8/8 | passed |
| `t036_r035_a120` | .36 | .35 | 1.2 | 60,891 | 39,910 | 2.3108e-6 | 1.0758e-9 | 16/18 | 8/8 | passed |
| `t012_r035_a150` | .12 | .35 | 1.5 | 63,187 | 40,992 | 4.9978e-7 | 3.6072e-10 | 16/18 | 8/8 | passed |
| `t024_r035_a150` | .24 | .35 | 1.5 | 65,442 | 42,430 | 2.2563e-6 | 3.6072e-10 | 16/18 | 8/8 | passed |
| `t036_r035_a150` | .36 | .35 | 1.5 | 61,173 | 40,008 | 2.1158e-6 | 3.6072e-10 | 16/18 | 8/8 | passed |
| `t012_r035_a200` | .12 | .35 | 2.0 | 64,319 | 41,748 | 4.8871e-7 | 1.9936e-9 | 16/18 | 8/8 | passed |
| `t024_r035_a200` | .24 | .35 | 2.0 | 65,083 | 42,116 | 3.1123e-6 | 1.9936e-9 | 16/18 | 8/8 | passed |
| `t036_r035_a200` | .36 | .35 | 2.0 | 60,647 | 39,724 | 4.3818e-6 | 1.9936e-9 | 16/18 | 8/8 | passed |
| `t024_r015_a150` | .24 | .15 | 1.5 | 75,101 | 47,088 | 8.6997e-7 | 7.7272e-10 | 16/18 | 8/8 | passed |
| `t024_r050_a150` | .24 | .50 | 1.5 | 61,405 | 40,112 | 4.0903e-6 | 1.6703e-9 | 16/18 | 8/8 | passed |

## Reporting Recovery

`geometry_robustness_v1` contains a complete first-case mesh, field, solver
result, and analysis, but its driver process terminated before serializing
the per-case record. The actual exception was
`NameError: analysis_run is not defined`: the subprocess variable was named
`analyze_run`, but the subsequent guard referenced `analysis_run`. It was not
a NumPy-boolean JSON serialization error. The corrected guard was used for
the successful v2 run.

Version 1's original progress file is preserved as
`results/geometry_robustness_v1/progress.pre_repair.json`. Its progress record
now identifies the reporting failure and points to the surviving first-case
artifacts. Version 2 has 11 distinct case directories and reran all 11 mesh,
test, solve, and analysis stages. It did not reuse or overwrite version 1
fields. The first case therefore exists in both batches as separate evidence.

## Frozen Provenance

The v2 `progress.json` records the source hashes below. They matched the
corresponding source files when the completed batch was audited. Later builder
changes, including work on the newly reported `1.0001` geometry case, do not
alter this recorded provenance or the per-case source snapshots.

| Source | SHA-256 |
|---|---|
| `lifting3d/run_geometry_robustness.py` | `01333c39f254537272e23852d06641b39cfd46055dbd6bebb832866bc00a161a` |
| `lifting3d/build_mesh.py` | `684a01db1cc88841480583d91c46f430d66d655ba2f1bc3c28c8c7130fe2c180` |
| `lifting3d/solve.py` | `e5d0ae2aa0244cf29e2fb7847b20df2a362091aeba1b19321efbf493796a3528` |
| `lifting3d/test_mesh.py` | `70a5fb100bd8ad79fe47a51917211c0cc5997d7b238d58e6168df94cdd445f32` |
| `lifting3d/analyze.py` | `c9fd4705771ca91e9d475b2f8c6fec0654c1663338432b28d8c6a8970eb5c85e` |
| `lifting3d/root_wake.py` | `f2d33ced9df54b8f9cf0035199005c5174dc0bc6f379a7389522dc274ff333ce` |
| `lifting3d/ROOT_WAKE_PROTOCOL.md` | `3816f94fd40476b42d4015712d07585e3ddbf08d54673ba4f31a08a771bcef8d` |
| `lifting3d/schur.py` | `8b352fd6ae5359256f7beb28cb21afa33bbc92e4203b9cb3a225e033c3d07408` |
| `volume/p2.py` | `920f88a737414a3d3a135d91a8373f877dec4619986e7452654f6e33619aa44a` |
| Frozen campaign protocol snapshot | `a5be2776293cdce2f57cbfa43b4a4783a35fa8e2fef59b849ee2e235f38714b1` |

The live campaign protocol was subsequently edited only to remove the
incorrect extra-approval wording; the v2 `protocol.snapshot.md` retains the
exact protocol text used for the run. Each case directory retains its mesh,
test, solver, and analysis artifacts and logs.
