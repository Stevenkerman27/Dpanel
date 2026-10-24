# Lifting3D Adaptive Mesh Protocol

## Frozen comparison

This is one pre-registered, single-cycle refinement experiment, not a parameter
sweep or a claim of validated lifting accuracy. The reference is
`results/wingbody_matrix_v1/t024_r035_a090/m0_a5`, with the matching saved mesh
and solution:

- symmetric thick wing-body: NACA thickness `.24`, ellipsoid length `3`,
  radius `.35`, aft location `.9`;
- mesh target size `.2`, curvature setting `32`, 24 conforming wake strips;
- weak Kutta, Mach `0`, alpha `5` degrees, solver `--ncpu 8`;
- farfield radius `5`, farfield target size `5/3`, reference area `4`.

The mesh, field, and result hashes in the baseline solver result and analysis
must agree before adaptation. The indicator is recomputed from that exact
baseline field and mesh. Its recorded `field_sha256` must equal the supplied
baseline field file hash. The same solver/indicator executable sources and
quadrature order are used for the adapted solve and its indicator. A failed
provenance check stops the experiment before background generation.

## Indicator and background sizing

`volume/error_indicator.py` consumes P2 lifting `field.npz` with potential key
`phi`, and requires its `points` and `tets` arrays to exactly match the mesh.
For cut meshes it requires both wake face traces and their adjacent tetrahedra.
It assembles ordinary interior-face pairs by corner-node identity, then handles
the coincident wake cut as a separate upper/lower face pairing. It checks
physical trace coincidence and opposing oriented area vectors before computing
raw mass-flux jumps. Its wall contribution includes every boundary tag other
than `100`; it does not report a separate junction RMS. Those are solver and
surface-analysis diagnostics, not substitutes for the indicator.

The scalar-tetra background NPZ contains exactly `points`, integer `tets`, and
positive `vertex_size`. The saved lifting mesh contains reflected upper/lower
volume halves. Before export the driver verifies that no corner tetrahedron
straddles `z=0` and that upper/lower element counts agree, then sends only
upper-half tetrahedra to PostView. This avoids duplicate volume samples while
retaining the physical upper side of the cut. It does not average or alter the
indicator or solution.

Global Dörfler marking targets cumulative indicator contribution `0.5`.
Vertices incident to any marked tetrahedron get size `0.5h = 0.1`. The same
size is assigned to vertices of tetrahedra adjacent to physical wall faces
(tags `10`, `20`, `30`) and to tetrahedra whose P2 geometry nodes lie within
`.1` chord of the body/wing seam. The seam uses the existing analysis
definition: edges shared by tags `20` and `30`, and their P2 edge nodes.
Unselected vertices retain farfield size `5/3`. Overlaps take the minimum.
The builder combines this PostView with its existing geometric distance and
optional tip fields using `Min`; background meshing is single-threaded by the
builder while the flow solve requests eight CPUs.

This local-size assignment is a deterministic background-field prescription,
not a guarantee that every generated edge realizes the target size. The mesh
metadata and mesh checks are retained to expose that distinction.

## Runs and acceptance record

All new files go under `results/lifting_adaptation_v1`; existing matrix
evidence is read-only. The driver preserves subprocess logs and failure JSON,
and runs sequentially:

1. provenance checks and same-field baseline indicator;
2. background NPZ write and hash;
3. CAD mesh build, mesh checks, weak-Kutta solve, and raw-field analysis;
4. same-mesh adapted indicator and a frozen baseline/adapted comparison.

Report baseline and adapted raw wall normal-velocity RMS, junction RMS and
area, raw Cp finite count/range, pressure lift/drag, circulation lift, Gamma
station values, component-force closure, raw mass flux by physical/wake tag,
weak residual and gauge reaction, and wake flux/pressure jumps. Compare
pressure/Cp and Gamma/load only as the saved raw quadrature/station data allow;
record differences without reweighting, clipping, or fitting. Keep raw wall
and junction errors, weak nodal balance, and raw boundary flux integrals as
separate measures. Any pre-existing acceptance threshold remains unchanged;
the driver does not turn a failed gate into a pass.

The existing `analyze.py` junction mask uses the current run's shared seam P2
nodes, so its sampling cloud can change when the mesh changes. Preserve that
per-mesh analysis unchanged, and additionally run
`run_adaptation.py --fixed-baseline-seam-audit` after both runs finish. This
postprocessor uses one KDTree made from the baseline mesh's tag-20/tag-30 seam
P2 nodes for both raw surface quadrature sets and writes a separate audit JSON.
This controlled region is needed because even two quadrature rules on the same
potential can change junction RMS when samples straddle the distance cutoff.
For example, the frozen q3/q4 check has junction RMS `.014858`/`.014622`;
that difference is not evidence of a changed flow solution.

If the new CAD build or solve fails, retain its failure record and do not relax
the mesh/solver gates or silently alter physics. Indicator reduction alone is
not evidence of improved Cp, wall tangency, junction behavior, circulation, or
load accuracy. One adaptive cycle cannot establish convergence or an error
bound. The underlying fixed-wake, P2 weak-Galerkin lifting method remains an
experimental reference and is not certified here.

The full-wall `.5h` proposal is allowed to fail the resource budget. If sparse
LU is stopped to protect system memory, mark the batch `resource_rejected`,
retain the generated mesh, mesh tests, topology input, and captured logs, and
do not call the outcome solver non-convergence. Record the largest sampled
working set and free-memory level. No refinement size, solver gate, or physics
is changed in response.
