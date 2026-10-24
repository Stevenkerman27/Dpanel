# Root-Wake Geometry Limits Protocol

## Scope

This is a serialized geometry/topology boundary matrix for the merged
ellipsoid/wing-root wake. It is not an absolute Cp or aerodynamic-accuracy
study. `run_geometry_limits.py --execute` writes only to a new results
directory and refuses to reuse an existing one. Failed artifacts and logs are
retained; no failed case is repaired or overwritten.

The v2 replay is selected explicitly with
`run_geometry_limits.py --execute --reuse-v1 --out results/geometry_limits_v2`.
It does not invoke `build_mesh.py`: each available legal mesh is read from
`geometry_limits_v1`, checked against its v1 SHA256, metadata parameters,
saved independent `tests.json`, boundary tags, and root station map, then
solved into the new v2 directory. The original v1 tree remains untouched.
The legal `aft=1.0001` item has no saved mesh because its v1 builder attempt
failed with `Exposed CAD strip knots must coincide with TE nodes`; v2 records
it as unavailable for reuse rather than silently rebuilding it. Expected
rejects retain their v1 stage/reason as provenance and are not rerun in replay
mode.

`build_mesh.py` is recorded as mesh provenance, not required to match the
current builder during replay. The v1 builder hash comes from the v1 run
record, while the current builder hash is recorded separately. Solver,
mesh-test, root-wake, numerical support, protocol, and replay-driver hashes
remain frozen and are checked throughout v2.

The frozen ordinary-case settings are span 4, chord 1, body length 4,
`size=0.25`, curvature 16, requested wake strips 16, farfield radius 5,
farfield size 5/3, alpha 5 degrees, Mach 0, weak Kutta, Schur solve, and
NCPU 8. Mesh timeout is 90 seconds and solve timeout is 180 seconds. Cases run
sequentially.

## Cases

Legal geometry cases use thickness 0.24, radius 0.35, and aft positions
0.999, 1.0001, 1.001, 1.01, and 3.0. The root-knot pair uses aft 1.5 and
`radius=0.25/sqrt(1-(1.5/2)^2)`, then that radius plus `1e-9`; the analytic
root half-width is 0.25, coincident with a requested uniform wake station.
The thick-wing legal case uses thickness 0.6, radius 0.35, and aft 1.5.

Expected explicit rejects are aft 1.0 (tangent contact), aft 3.0001 (body
maximum girth downstream of the wing TE), radius 3.05/aft 1.5 (root contact
covers the full half-span), sweep 20 degrees/aft 1.5 (merged root does not
support swept geometry), and wake tilt 2 degrees/aft 1.5 (merged root requires
a planar cut). The wrong-Kutta case reuses the successful `root_wake_pilot_v2`
mesh and requests node Kutta; it must be explicitly rejected because merged
root contact requires weak Kutta.

The builder is exercised for every geometry case, including expected rejects.
Only geometries expected to be legal advance to the independent mesh test and
solve. The wrong-Kutta case skips meshing and tests the solver rejection on the
saved pilot mesh.

## Acceptance

Each legal case must preserve the requested geometry metadata, pass all
independent `test_mesh.py` gates, and have finite positive P2 Jacobians and all
required boundary tags. A merged body-wing mesh requires tags `[20, 30, 100]`:
body tags 20/30 and farfield tag 100. Tag 10 is the isolated-wing wall tag and
is not expected on a merged body-wing mesh. For a merged root, the
saved interval must match the ellipsoid/TE intersection, the mesh and metadata
station-to-DOF maps must agree, each reduced DOF must have positive exposed
weak-test support, and root endpoints must be present as stations. The exact
root-knot case additionally checks the 0.25 station coincidence; the
`radius+1e-9` case checks that the perturbed root remains explicitly represented
without silently snapping the requested geometry.

For the completed v2 replay, see `results/geometry_limits_v2/result_replay.json`.
The initial replay was intentionally interrupted after three solves when its
force gate was found to include non-wall faces. Those fields were preserved;
their force gate was recomputed independently in
`results/geometry_limits_v2/force_gate_audit.json`, and only cases without an
existing field were solved in `solve_replay/` as recorded by
`solve_continuation.json`. The finalized replay therefore does not claim a
single uninterrupted driver run.

Every accepted mesh is solved at the frozen flow/Kutta/backend settings. The
driver checks convergence, finite saved Cp/velocity/normal/weights, positive
surface weights, positive beta, exact agreement of field and mesh Gamma maps,
expansion of reduced `field.gamma` to the full station vector in
`result.potential_jump`, and full/component pressure-force closure to absolute
tolerance `1e-10`. Pressure force is independently integrated over wall tags
`[10, 20, 30]` only; farfield tag 100 and wake traces 201/202 are excluded.
No Cp range, force magnitude, CL agreement, or wall-normal
velocity is an absolute pass/fail accuracy threshold.

Each expected reject must fail at its named stage with the recorded rejection
reason. Unexpected acceptance, timeout, meshing failure of a legal case,
topology failure, solve failure, nonfinite output, or a changed frozen source
hash is a failed matrix item and remains in the output record.

## Frozen Sources

At startup the driver hashes itself, this protocol, `build_mesh.py`,
`solve.py`, `root_wake.py`, `test_mesh.py`, `volume/p2.py`, `schur.py`, and the
solver/root-wake/Schur protocols. It stores the starting hashes and checks
them after every case and at completion. A mismatch stops further execution
and is reported as source mutation. Commands, stdout/stderr, elapsed times,
mesh/test outputs, solver results, hashes, and rejection messages are retained
under each case directory.

During v2 replay, builder hash equality is deliberately not an acceptance
condition because no mesh is rebuilt. The v1 builder hash is checked against
the source hash recorded in the v1 manifest and each reused mesh hash is
checked against the corresponding v1 case record. All non-builder numerical
sources are frozen against mutation during replay.

The v2 replay has 13/14 recorded items accepted: seven reused-mesh solves and
six carried-forward expected-rejection records. `legal_aft_1p0001` remains
pending because v1 saved no mesh for that legal request. It must be validated
and solved as a separate repaired-builder case; its result must not be
silently substituted into the v1 mesh-reuse record.

The matrix only probes declared geometric/topological limits and numerical
sanity. Passing it does not establish aerodynamic accuracy, mesh convergence,
or a generally valid lifting solver.
