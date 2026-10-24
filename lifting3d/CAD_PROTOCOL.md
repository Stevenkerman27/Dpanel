# Imported CAD with a planar trailing-edge wake

Status on 2026-10-24: development is paused at the user's request. Nine CAD
configurations completed the full pipeline; the repaired two-body mesh has
not completed independent audit or flow solution. Evidence and limitations
are recorded in [CAD_ROBUSTNESS_RESULTS.md](CAD_ROBUSTNESS_RESULTS.md);
the restart checklist is [PAUSE_HANDOFF.md](../PAUSE_HANDOFF.md).

This branch imports closed wing and optional body CAD solids, plus an explicit
trailing-edge polyline. It builds the full fluid volume; it does not reflect
an upper-half mesh. Cambered and geometrically asymmetric solids can therefore
retain their actual upper and lower surfaces. The current wake is fixed,
planar, and downstream in x. This is a step toward general lifting CAD, not
automatic support for arbitrary nonplanar trailing edges or free wakes.

The trailing-edge JSON contains native-coordinate `points` (strictly
increasing y), positive `chord` and `reference_area`, and a three-component
`origin`. All TE points must lie on the origin's z plane. The x coordinate
is piecewise linear in y. CAD is normalized by that explicit origin/chord,
not by a conservative CAD bounding box. Saved mesh coordinates, reference
area, TE points and potential jumps are restored to native units.
Each declared TE segment is checked at its endpoints and midpoint against
trimmed wing CAD boundary curves, including parameter bounds. This is a
finite geometric check, not certification of every point on an arbitrary
curved CAD edge.

The CAD workflow first fragments a fluid box by the declared wake strips.
Both surface boundary curves and mesh-embedded curves are included when
finding shared wall/wake edges; boundary-only discovery missed body contact
in the first imported wing/body attempt. Supplied TE curves are excluded
before constructing smooth-wall contact components. Components are projected onto y using CAD
curve endpoints and stationary points; overlapping projections are merged.
The model Gamma(y) must be constant over each connected projection while
retaining an unknown value. Final strips include each component endpoint.
Discovery and final meshing are separate passes; every tolerance and
projection diagnostic is retained. Unresolved contact topology is rejected.
Stationary points are located by a finite parameter scan and derivative
root finding. The diagnostics do not claim a global extremum error bound
for arbitrary oscillatory CAD splines.

For a node on the internal cut, remove the wake faces from its incident
tetrahedron star. Positive and negative wake traces seed the two remaining
components. Duplicate the node only on the negative component. A global
centroid-z split is insufficient near cambered trailing edges. Tip nodes
remain shared because Gamma is prescribed zero there. The final mesh must
have complete exterior faces, consistent P2 midnodes, positive
dense-sampled Jacobians, and paired, geometrically identical wake traces.
The independent audit also checks that each Tri6 stays within a Gamma
strip and that every reduced circulation test has positive support on the
declared TE. Smooth-body contact edges cannot contribute Kutta support;
a body entirely upstream of the TE is allowed to have no wake contact.
Adaptive minimum-Jacobian cross-checking was proposed but is not implemented.

The existing P2 Galerkin full-potential equations are reused. On imported CAD,
weak Kutta equations integrate the raw streamwise derivative difference
against the mapped Gamma tests with physical TE edge measure. Bank signs
come from cut topology rather than global z. No velocity recovery, pressure
clipping, singular-value truncation, root-lift prescription, or numerical
regularization is introduced. M=0 remains linear; compressible cases remain
strictly sampled-subsonic.

When separate body and wing solids are supplied, wall CAD ancestors provide
component labels (body 20, wing 30). Ambiguous coincident ancestors require
an explicit recorded ownership rule. Wing-only CAD uses tag 10. Pressure
force and moments always integrate the original raw quadrature.

Verification prioritizes successful faithful CAD import, complete topology,
finite fields, actual test support, force closure, explicit rejection, and
coordinate invariance. Raw boundary errors and both wall banks remain
visible; finite pressure is not an absolute-pressure accuracy certificate.
Gmsh warnings from meshing and high-order optimization are preserved.
The two-body coarse-mesh optimization failure and the subsequent
curvature-sized mesh are separate attempts; a successful builder run does
not count as a completed flow or load calculation.
