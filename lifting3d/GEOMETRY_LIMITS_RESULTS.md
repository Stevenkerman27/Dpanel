# Geometry-Limits Results

## Scope and disposition

This note separates the eight legal-geometry flow fields from six historical
expected-rejection records. It reports mesh/solver sanity and force
reconciliation only; it is not an aerodynamic-accuracy or general geometry
certification.

The eight audited legal fields are the seven v2 runs that reused the original
v1 meshes, plus the repaired `aft=1.0001` follow-up from
`root_near_repair_v1`. All eight converged and passed the independent wall
force audit. The near-tangent follow-up is a separate repaired-builder mesh,
not a rebuilt or substituted v1 mesh.

| Legal case | Mesh provenance | Gamma stations / reduced DOFs | Wall force error |
|---|---|---:|---:|
| `aft=0.999` | v1 mesh reused | 17 / 15 | 0 |
| `aft=1.0001` | `root_near_repair_v1` follow-up | 19 / 15 | 0 |
| `aft=1.001` | v1 mesh reused | 19 / 15 | 0 |
| `aft=1.01` | v1 mesh reused | 19 / 15 | 0 |
| `aft=3.0` | v1 mesh reused | 19 / 13 | 0 |
| root-knot exact, `aft=1.5` | v1 mesh reused | 17 / 13 | 0 |
| root-knot `radius+1e-9`, `aft=1.5` | v1 mesh reused | 17 / 13 | 0 |
| thickness `0.6`, `aft=1.5` | v1 mesh reused | 19 / 15 | 0 |

Wall force uses only tags `[10, 20, 30]`; farfield tag 100 and wake traces
201/202 are excluded. Across all eight fields, maximum wall-force difference
from `result.json` is 0, maximum component-force closure error is
`1.155e-14`, maximum Gamma expansion error is 0, and every saved surface field
is finite with positive surface weights and positive `min_beta`.

## Historical expected rejections

These are v1 records of the then-current builder/solver behavior. They are not
a statement that all six are current rejection boundaries. In particular,
the sweep rejection records an old builder limitation; sweep/taper root support
has since been expanded.

| v1 case | Historical stage and recorded reason | Current interpretation |
|---|---|---|
| `aft=1.0` tangent | Builder rejected tangent topology at the wing TE | Historical boundary record; not re-executed here |
| `aft=3.0001` | Builder rejected maximum girth downstream of the TE | Historical boundary record; not re-executed here |
| radius `3.05`, `aft=1.5` | Builder rejected full-span body contact/no exposed TE | Historical boundary record; not re-executed here |
| sweep `20`, `aft=1.5` | Old builder required a rectangular unswept wing | Obsolete as a current-capability claim; sweep roots are now supported |
| wake tilt `2`, `aft=1.5` | Builder rejected non-planar cut for merged root | Historical boundary record; not re-executed here |
| node Kutta on pilot root mesh | Solver required weak Kutta equations for merged wall-contact root | Historical solver rejection record; solver source was not changed in this work |

The v1 legal `aft=1.0001` build did not produce a mesh: it stopped at
`Exposed CAD strip knots must coincide with TE nodes`. The repaired
`root_near_repair_v1` mesh and flow are recorded as the separate legal follow-up
above.

## Compressible Nonlinear Sanity

`results/root_compressible_v1/` adds an independent nonlinear sanity run on
the existing `root_wake_pilot_v2` mesh, not another geometry-limit case. The
run used Mach 0.2, alpha 5 degrees, weak Kutta, Schur, and NCPU 8. It converged
in four Newton iterations in 58.54 seconds; the final residual was
`5.984e-14`, maximum local Mach was `0.445398`, and `min_beta` was
`0.969533`. This is a converged subsonic compressible case, not a transonic
flow validation.

The saved mesh hash matches the pilot mesh:
`0cc465d59a3d692576e716ee725c96ac04d66f0b48234093f8bd58c7edb2848d`.
An independent wall-only force integration over the available wall tags
`[20, 30]` matches `result.json` with maximum error 0; component-force closure
error is `2.61e-15`, and reduced-to-full Gamma expansion error is 0. The saved
surface fields are finite and the weights are positive. The field SHA256 is
`e5ecc67be422c73cf439f029448cb5d6c409161e8eb020e3c9c2780e8422783d`.

## Gate corrections

The original v1 topology gate required tags `[10, 20, 30, 100]`. Its merged
body-wing meshes actually contain `[20, 30, 100]`; tag 10 is not present on
those meshes. Independent `test_mesh.py` checks passed, but the v1
`mesh_topology_failed` classifications remain in place as incorrect gate
results. They are not relabeled as v1 passes.

The first v2 force gate integrated every `field` face, including farfield and
wake traces, and consequently reported false failures. The original failed
gate records are retained. A separate corrected audit recomputed force using
only wall tags `[10, 20, 30]`; all eight fields pass that wall-force audit.
Neither gate correction changed the governing equations or solver files.

## Source provenance

The seven reused meshes and the six expected-rejection records retain their
v1 provenance. The original v1 mesh-builder SHA256 is
`684a01db1cc88841480583d91c46f430d66d655ba2f1bc3c28c8c7130fe2c180`.
The current builder SHA256 at report time is
`b0524b2c3f19df7d1008e7eab0427eb6aa264768cc2a175c48d1b6368f2f4c4b`.
The builder hashes differ; this does not invalidate fields solved on the
byte-verified reused v1 meshes. The repaired near-tangent mesh has its own
captured build and solve artifacts under `results/root_near_repair_v1/`.

For all audited runs, captured `solve.py`, `p2.py`, `schur.py`, and
`root_wake.py` snapshots match the current files. `test_mesh.py` matches the
v2-start SHA256 recorded in the replay manifest. Protocol/document hashes are
retained per run rather than treated as campaign-wide invariant hashes. In
particular, the runs captured `ROOT_WAKE_PROTOCOL.md` SHA256
`3816f94fd40476b42d4015712d07585e3ddbf08d54673ba4f31a08a771bcef8d`; the
current document SHA256 is
`f17ad8531baa6b346d80085f678141f9a182180f08108196ea17f91981842514`.
That difference is documentation provenance, not a solver-code change.

## Artifacts

- `results/geometry_limits_v1/result.json`: original mesh runs, gates, and historical rejections.
- `results/geometry_limits_v2/result_replay.json`: reused-mesh replay summary and source provenance.
- `results/geometry_limits_v2/force_gate_audit.json`: independent wall-force audit for the seven v2 fields and near-tangent follow-up.
- `results/geometry_limits_v2/solve_continuation.json`: the four v2 fields completed after preserving the interrupted first batch.
- `results/root_near_repair_v1/`: repaired near-tangent mesh, tests, solve, and captured source snapshots.
- `results/root_compressible_v1/`: same-pilot-mesh Mach 0.2 nonlinear sanity result and field.
