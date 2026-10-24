# Wing-Body Geometry Robustness Protocol

## Scope and execution hold

This is a geometry/topology robustness matrix, not an aerodynamic-accuracy
study. The completed `lifting_adaptation_v1` resource-rejected record is
preserved and is not part of this campaign. The root-merge pilot passed, and
execution of this matrix was authorized by the existing task.

`run_geometry_robustness.py` is dry-run by default. This authorized run used
`--execute` and explicitly passed `--root-wake merged`; the builder's `auto`
default was not relied upon. Before any CAD operation the driver checks that
builder help exposes the merged option. A generated mesh must declare
`root_wake_support.status == "constructed"` and provide finite
`root_interval_y`, `exposed_te_intervals_y`, `station_to_dof`, and
`weak_test_support_integrals`. The driver checks these against the analytic
ellipsoid/TE intersection and saved gamma-station mapping. `constructed` is
not treated as independent validation: the driver then runs `test_mesh.py` and
requires its independently recomputed `root_wake_row_coverage` test to pass,
alongside the dense Jacobian, exterior-face adjacency, and component-tag
checks. It compares generated geometry metadata against every frozen input.
If a contract or test is absent/inconsistent, it records a failure and does
not launch the solver or fall back to the old straight planar wake.

## Frozen matrix

All cases use a symmetric rectangular finite-thickness wing with span `4`,
chord `1`, and a four-chord x-aligned ellipsoid body. The principal matrix is
the full Cartesian product:

- body radius `.35`;
- aft positions `1.2`, `1.5`, `2.0`;
- NACA thickness ratios `.12`, `.24`, `.36`.

Two additional radius controls use thickness `.24`, aft `1.5`, and radii
`.15` and `.50`. There are 11 unique cases total. The driver derives the body
center as `aft - body_length/2` and the body/TE intersection interval from the
ellipsoid equation before meshing. A missed or tangent TE/body intersection,
no exposed TE on either wing side, or a body outside the farfield box is an
explicit geometry rejection, not a mesh or solver failure.

Frozen discretization and flow inputs are `h=.25`, curvature `16`, requested
wake strips `16`, farfield radius `5`, farfield size `5/3`, weak Kutta, Mach
`0`, alpha `5` degrees, NCPU `8`. The builder may adjust the strip partition
to conform to the merged root interval; record both requested and actual strip
counts, and do not reject a mesh solely because the actual count differs from
16. Each individual solver call has a hard 180-second timeout. Cases execute
sequentially. No adaptation, refinement, parameter fitting, or post-hoc gate
adjustment is included.

The aft body overlaps the geometric plane `z=0` downstream of the wing TE.
This matrix specifically requires the integrated root-wake implementation to
route/represent the wake around the fuselage-root intersection. The legacy
builder's assertion that the body tail must be ahead of the TE is not an
acceptable fallback for these inputs.

## Geometry and topology gates

For every case, retain the complete command, source hashes (including
`ROOT_WAKE_PROTOCOL.md`, `schur.py`, and `volume/p2.py`), CAD/build log, mesh
metadata, mesh NPZ hash, test output, solver log/result when available, and
elapsed time. Check, in order:

1. Analytical geometry preflight succeeds and reports finite exposed TE
   intervals and root intervals.
2. CAD meshing completes without invalid Boolean fragments or missing
   components; body and wing boundary tags are both present.
3. The builder's `constructed` root interval and exposed intervals match the
   analytic geometry, station-to-DOF mapping matches the saved mesh, and all
   weak-test support integrals are finite and positive.
4. Independent `test_mesh.py` checks pass: Tet10 dense-sampled Jacobian is
   positive; all nodes are used; exterior face adjacency and wake pairing are
   complete and one-to-one; and root-wake row coverage has no gap or overlap
   over exposed TE intervals with no spurious circulation DOF on the
   root-covered TE segment.
5. Gamma stations and any root-interval station mapping are finite,
   strictly ordered, and match the exposed trailing-edge intervals.

An illegal input is `geometry_rejected`; CAD or mesh-check failures retain
their own failure status. Do not alter geometry to make a case pass.

## Solver sanity gates

Only mesh cases passing all geometry/topology gates proceed to the solver.
Run Mach `0`, alpha `5`, weak Kutta, NCPU `8`, using the explicitly selected
linear backend recorded in the run metadata. Preserve the full Newton and
linear-solver histories. Timeout is `solver_timeout`, distinct from a
non-finite state, factorization failure, or nonlinear non-convergence.

The solver sanity checks are: converged status, all saved raw Cp values finite,
positive finite beta/subsonic state as enforced by the solver, finite pressure
forces, and component-force closure under the existing `1e-10` check. Record
weak residual, raw normal-velocity measures, raw tag-wise mass flux, Kutta
residual, Gamma/root mapping, pressure lift/drag and circulation lift as
diagnostics. Do not use Cp extrema, agreement with a reference lift, wall RMS,
or force magnitude as absolute-accuracy gates for this campaign.

Report per-case status and failure stage, geometry parameters, root/exposed TE
intervals, mesh counts, minimum Jacobian, row coverage, component tags, wall
and junction raw diagnostics, Cp finite count, force closure, solver elapsed
time, and complete resource/runtime costs. Aggregate success means only that
the geometry/topology and numerical-sanity gates passed for the case; it is
not validation of pressure accuracy, lift accuracy, a general lifting solver,
or free-wake physics.
