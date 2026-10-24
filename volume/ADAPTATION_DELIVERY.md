# Sphere Adaptation Delivery

## Attempt history

| Run | Outcome |
|---|---|
| v1 | CAD extraction failed before remeshing: OCC reported both concentric sphere surfaces as type `Sphere`; the classifier put both in the farfield group, leaving no body surface. Fixed in the authorized mesh builder by classifying the spherical surfaces against their CAD bounding-box radii. Failure JSON/logs remain in `audit/adaptation_sphere_v1/`. |
| v2 | No completed remesh/result. Its frozen preflight paired the old `p2_initial` field (`6ffbf0...`) with an indicator generated from the different `p2_fluxjump_final` field (`3946af...`). This evidence is invalid for adaptation comparison and was not used. |
| v3 | Recomputed the source field and indicator consistently; both hashes were `3946af...`. Gmsh with eight threads stalled in the PostView 1D curve-meshing stage and was stopped. No remesh or solver result was accepted. |
| v4 | Direct reproduction localized the same eight-thread stall to `gmsh.model.mesh.generate(3)` during 1D curve meshing, after OCC and PostView registration. The partial diagnostic did not produce a mesh/result artifact. |
| v5 | Running PostView CAD meshing single-threaded removed the stall and generated a valid curved sphere mesh (2,917 tets, positive sampled Jacobians). This was a mesh-builder check only; it did not run the matching baseline/indicator/solver comparison. |
| v6 | Full chain completed with a consistent fresh baseline. Physical measurements were valid, but the summary reused the adapted indicator result when writing `baseline_indicator_total`, making the two displayed totals equal. No physical gate logic was changed. |
| v7 | Fixed the indicator-summary variable reuse. The v6/v7 generated `sha256_arrays` are identical (`e0553f...`), and all physical measurements/gates match. Correct totals are baseline 7.281891 and adapted 6.598408. v7 still fails three physical gates, listed below. |
| wall-aware v1 | One separately pre-registered wall-aware pass completed; exact method, hashes, gates and measurements are in `audit/adaptation_sphere_wallrefine_v1/`. |

The v2 mismatch was caught and documented rather than silently reused.
Starting with v3, each run recomputes the source solution and its indicator
inside the new audit directory, then requires exact equality between the
indicator's `field_sha256` and the baseline field file SHA-256 before freezing
preflight or remeshing.
For v7 and wall-aware v1, both hashes are
`3946afabfa1b2869cab266b883b04972af200baeb7a1bc52d46e0a92089e7c32`.

## v7 gate result

The v7 method was Dörfler-50% marked corner vertices at `0.65 * base_size`,
with all other source corner vertices at base size. Seven of ten gates passed.
The failed gates were:

- Absolute drag error increased from `0.01277112` to `0.01581562`.
- Raw wall area-weighted RMS normal velocity increased from `0.15930229` to
  `0.16417519`.
- Raw wall area-weighted RMS normal mass flux increased from `0.15930229` to
  `0.16417519` (M=0 makes density one here).

Cp RMS/max errors, lift error, absolute raw total-boundary mass closure,
weak residual, sampled Jacobians, and boundary tags passed. The lower
integrated closure (`0.20975300` to `0.05508300` absolute) does not erase the
failed wall RMS gates: wall and outer signed fluxes can cancel.

## Single wall-aware comparison

The separate `WALL_ADAPTATION_PROTOCOL.md` was frozen before this run. The
explicit `--wall-refine` mode refines the union of Dörfler-marked cell corners
and all physical-wall-adjacent cell corners to `0.5 * base_size`; farfield
corner vertices are reset to base size. The historical default remains the
v7 0.65 mode. This experiment adds a boundary-residual/curved-wall target; it
does not fit the coefficient, scan parameters, or modify any solver.

All ten unchanged physical gates passed on this one sphere case. The source
mesh marked 50 cells; the marked and wall-adjacent sets contained 61 and 72
corner vertices, with a 79-vertex union. The mesh had 3,746 tets and 6,036
P2 DOFs, minimum sampled curved Jacobian determinant `0.016259` (positive), and
minimum tet `minSICN` `0.1896`.

## Error, size, and time

All rows use M=0, sphere-exact farfield and quadrature order 3. `DOFs` is the
P2 nodal count. Solver time is the saved `solve_p2.py` wall time; mesh time is
the saved mesh-generation metadata time. The global-fine sphere is an
independently generated mesh, not a nested refinement of the coarse source.
These rows are an efficiency comparison, not a strict convergence sequence.
Solver times use the same solver entry point and settings. Mesh times are the
recorded generation-stage times, but the adaptive rows perform direct CAD
remeshing while the pre-existing coarse/fine references include P1-to-P2 mesh
construction; treat mesh-time differences as indicative, not controlled
benchmarks.

| Mesh/method | Cp RMS error | Cp max error | DOFs | Tets | Mesh seconds | Solver seconds |
|---|---:|---:|---:|---:|---:|---:|
| Coarse source baseline | 0.06039244 | 0.19819357 | 4,834 | 2,883 | 0.1685 | 1.1240 |
| Dörfler-only 0.65 (v7) | 0.05725342 | 0.17819864 | 4,844 | 2,917 | 0.7853 | 1.1487 |
| Dörfler + wall union 0.5 | 0.03762046 | 0.17050508 | 6,036 | 3,746 | 0.8600 | 1.5033 |
| Global fine reference | 0.03376881 | 0.13658529 | 21,906 | 14,121 | 0.5834 | 4.7212 |

The wall-aware pass substantially improved over coarse and v7 at lower DOFs
than the global fine reference. The global fine reference still has lower Cp
errors and raw wall RMS (0.08529264 versus 0.09261734). These results support
only this single, sphere-specific comparison. They do not establish a
certified indicator bound, general adaptation efficiency, repeated-mesh
convergence, or 3D lifting/Kutta/wake validity.

## Artifacts

- `volume/run_adaptation.py`: historical 0.65 default and explicit
  `--wall-refine` method.
- `volume/build_curved_mesh.py`: optional PostView field and sphere CAD
  boundary classification fix; ordinary no-background behavior remains.
- `volume/ADAPTATION_PLAN.md`: Gmsh API notes and v7 first-pass findings.
- `volume/WALL_ADAPTATION_PROTOCOL.md`: pre-registered wall-aware method and
  unchanged gates.
- `audit/adaptation_sphere_v7/`: consistent baseline and Dörfler-only run.
- `audit/adaptation_sphere_wallrefine_v1/`: frozen preflight, wall-aware
  mesh/field/indicator, solver diagnostics and complete result JSON.
