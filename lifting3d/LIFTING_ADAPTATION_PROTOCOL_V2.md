# Lifting3D Cost-Bounded Local Adaptation Protocol

## Status and hold point

This protocol defines a proposed single-cycle experiment. It does not launch a
mesh build or solver run. The 523,020-point V1 mesh is a retained
`resource_rejected` result and is not a template for further global refinement.
V1 files and evidence remain unchanged.

Do not build the V2 adapted mesh until the optional exact-Schur backend has
passed both the small-wing and fixed 113,012-point baseline comparisons against
the current direct backend on identical meshes and source revisions. Save the
backend identity, source hashes, residual histories, wall/junction diagnostics,
Cp/Gamma/loads, and peak process/system memory for both. Differences must be
within the existing solver gates; do not loosen any gate to admit a backend.
Use q3 for the registered comparison. The completed q3/q4 same-field check
showed `CL(q3)=0.3792998478824885`, `CL(q4)=0.379299802103342`, a difference
of `4.58e-8`; the observed junction RMS change `.014858` to `.014622` is
quadrature-region sampling sensitivity and is not evidence of a flow change.

The fixed comparison geometry remains thickness `.24`, body length `3`,
radius `.35`, aft `.9`, span `4`, chord `1`; `h=.2`, curvature setting `32`,
24 wake strips, weak Kutta, Mach `0`, alpha `5` degrees, farfield radius `5`,
farfield target size `5/3`, and NCPU `8`. The V1 historical solver hash differs
from the current source. Use the current-source same-mesh baseline solve as
the quantitative reference and retain the `wingbody_matrix_v1` record as
historical context only.

## Frozen indicator and local region

Compute the q3 error indicator from the exact baseline mesh and current-source
baseline `field.npz`. Before marking, require exact points/tets ordering and
require `indicator.field_sha256` to equal the field file hash. Keep the
indicator implementation, Dörfler fraction, and all raw contributions
unchanged.

Mark global Dörfler contribution `50%`. Build one fixed seam KDTree from the
baseline P2 mesh edges shared by physical tags `20` and `30`, including their
quadratic edge nodes. The junction band is the physical distance `<0.1 chord`
from those fixed nodes; do not rebuild the junction definition from the
adapted mesh's own sampling nodes.

Set target size `0.1 = 0.5h` only on vertices belonging to Dörfler-marked
tetrahedra or to tetrahedra incident to junction-band surface faces. Do not
apply `0.5h` to every wall-adjacent tetrahedron. Set all other background
vertices to `5/3`. Resolve overlap with the minimum target size. Keep the
original Gmsh distance and optional tip fields in the builder's existing `Min`
combination. Export only one physical half of the symmetric volume for
PostView, after validating no tetrahedron straddles `z=0` and the two halves
have matching element counts.

This is intentionally a local indicator-plus-junction proposal, not a promise
that Gmsh will realize the requested sizes. Record actual mesh point/element
counts and quality checks before any flow solve.

## Cost gates

Use the baseline mesh counts as hard pre-solve limits:

- points at most `1.5 * 113012 = 169518`;
- tetrahedra at most `1.5 * 72426 = 108639`.

If either limit is exceeded, retain the mesh and mark the attempt
`mesh_budget_rejected`; do not launch the solver. Require the existing complete
mesh checks, positive dense-sampled curved Jacobian, and matching background
field SHA before considering a solve.

For any solver run, sample process working set and system free physical memory
at least once per second. Stop and classify `resource_rejected` if process
working set reaches `8 GiB`, system free physical memory falls below `12 GiB`,
or solver wall time exceeds three times the measured current-backend
fixed-mesh baseline time. These are pre-registered resource gates, not
convergence or physics gates. Keep partial logs and mesh artifacts; do not
describe a resource stop as numerical non-convergence. Report build, mesh-test,
indicator, and solver wall times, peak sampled working set, minimum free RAM,
points/tets, and ratios to baseline.

## Physical comparison

Preserve the current per-mesh `analyze.py` output. Also recompute raw junction
RMS and area on both baseline and adapted raw surface quadrature using the
same baseline-seam KDTree and `<0.1 chord` cutoff. Report the original
per-mesh junction metrics beside this controlled comparison; neither may
replace the other.

Report, without filtering or projection:

- full-wall raw normal-velocity RMS/max and fixed-seam junction RMS/area;
- raw Cp finite count, minimum/maximum, and physical-tag area-weighted mean
  and RMS;
- pressure lift/drag, circulation lift, component loads and their closure;
- Gamma values at common physical stations, interpolating only for the
  comparison table and retaining both original station vectors;
- raw mass flux by physical and wake tag, weak residual/gauge reaction, and
  raw wake/internal-face flux diagnostics when measured;
- raw baseline/adapted indicator terms, marked fractions, mesh counts and
  all runtime/memory costs.

Keep physical acceptance thresholds unchanged. Indicator reduction,
successful mesh generation, or solver convergence alone is not evidence of
improved wall tangency, junction behavior, Cp, circulation, conservation, or
load accuracy. This experiment is one cost-bounded refinement cycle and does
not certify the lifting solver or general mesh convergence.
