# Local Gmsh Background-Size Adaptation

## Verified API path

The local bundled Gmsh Python module is version 4.15.2. Its runtime docstring
defines `gmsh.view.addListData(tag, "SS", n, data)` as scalar tetrahedron list
data, with each element encoded as x coordinates, y coordinates, z coordinates,
then one scalar per vertex. The adapter uses the four corner nodes from each
source tetrahedron and a scalar target size at each source vertex. The script
adds a `PostView` mesh field, sets `ViewTag`, calls
`gmsh.model.mesh.field.setAsBackgroundMesh`, and disables point/curvature/
boundary-extension sizing so the background field controls the requested
volume size. No Gmsh installation or external package is used.

The runtime API path is also exercised by a standalone smoke test in
`run_adaptation.py`: it inserts one scalar tetrahedron into a view, attaches
the `PostView` field, and asks Gmsh to generate a tetrahedral mesh on a small
CAD volume. This verifies invocation and data shape, not interpolation-error
or optimality of the Gmsh size field. Background-field CAD meshing is
single-threaded because the local eight-thread PostView curve stage stalled;
the P2 solve and indicator remain at the requested `--ncpu` (default 8).
The implementation follows the local Gmsh 4.15.2 examples
`share/doc/gmsh/examples/api/copy_mesh.py` and
`share/doc/gmsh/tutorials/python/t17.py` for `PostView`, `ViewTag`, and
`setAsBackgroundMesh`.

## One-pass indicator mapping

The current source is the existing curved P2 sphere mesh and its matching
M=0, sphere-exact total-potential field. The Dörfler 50% cell marks are mapped
to source corner vertices. A vertex receives the minimum target size among
incident cells: `0.65 * base_size` if any incident cell is marked, otherwise
`base_size`. The remesh uses the same analytic sphere/farfield CAD, radius,
and nominal base size. This is a single local refinement pass, not an adaptive
convergence loop.

## Pre-registered comparison gates

For each run, `run_adaptation.py` first resolves the unchanged source P2 mesh
with the current `solve_p2.py`, then computes its indicator with
`error_indicator.py`. It verifies that the indicator result's `field_sha256`
exactly matches the just-written baseline field before using its marks. Only
then does it save the source hashes, counts, threshold, and same-run baseline
measurements to `preflight.json`, before launching CAD remeshing.
The remeshed sphere is checked for positive sampled curved Jacobians and
preserved body/farfield tags. The new M=0 `sphere-exact` P2 solution is
compared with the existing coarse result using:

1. Area-weighted analytic sphere Cp RMS and maximum absolute error. Both must
   be strictly lower than baseline to call Cp improvement.
2. Absolute lift and drag errors against zero. Both must be no larger than
   baseline to call force improvement.
3. Raw wall normal-velocity RMS, raw wall normal mass-flux RMS, and absolute
   integrated total-boundary mass closure, each no larger than baseline to call
   raw-flux improvement.
4. The raw weak free-node residual must meet the solver's existing `1e-10`
   convergence tolerance.

These are measured gates, not a tuning objective. A failed gate is retained
and reported as failed; a smaller aggregate indicator is not a substitute for
Cp or load improvement. Results are written to a new output directory, with
source, background, generated-mesh, solver, and indicator hashes retained.

## First measured pass

The first completed pass is recorded in `audit/adaptation_sphere_v7/`. It used
the same source mesh and freshly regenerated source solution for baseline and
marking. Cp RMS and maximum error improved, but drag error and both raw wall
normal RMS values worsened, so the complete physical gate set failed. The
modest element-count change also means this run only verifies the API path and
one adaptation pass; it is not evidence of a converged or generally effective
adaptation strategy. See that audit's `RESULTS.md` for exact values.

The one separately frozen wall-aware comparison and the v1-v7 execution
history are documented in `WALL_ADAPTATION_PROTOCOL.md` and
`ADAPTATION_DELIVERY.md`. The added wall-aware case passed its unchanged
physical gates; it is still a single sphere-only experiment.
