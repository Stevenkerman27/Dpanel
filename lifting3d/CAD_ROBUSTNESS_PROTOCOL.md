# CAD geometry acceptance

The priority is faithful geometry and well-defined discrete equations. This
experiment does not use an absolute lift or pressure-accuracy target.

## Positive cases

Use independently saved closed STEP solids: symmetric wing, cambered wing,
symmetric wing/body, cambered wing/body, cambered wing scaled by 1000 and
translated, and tapered swept cambered wing. A second contact group tests
disconnected body contacts and a body whose maximum section is downstream of
the declared trailing edge.

Each accepted case must complete import, full-volume meshing, independent mesh
audit, original-equation solve, raw pressure force/moment integration, and
raw surface export. Every stage records its command and return code. Mesh
and solver source snapshots and input hashes identify the actual attempt.
Failed attempts remain in their original directory.

Geometry checks include dense-sampled positive Tet10 Jacobians, complete
exterior-face adjacency, identical Tri6 midside ownership, coincident paired
wake traces, shared zero-circulation tip nodes, conforming Gamma strips,
explicit CAD contact equivalences, and positive trailing-edge test support
for every remaining circulation unknown. Passing a finite sampling test is
not a proof of Jacobian positivity at every point.

Contact discovery must include both boundary curves and embedded curves of
the CAD wall/wake surfaces. Curves matching the declared trailing edge are
excluded before constructing connected smooth-wall contact components.
Disconnected contacts must not be joined through an excluded trailing edge.
Endpoint and stationary-point projection diagnostics are retained. The
uniform derivative scan does not certify all extrema for arbitrary CAD
splines; unresolved features remain a stated limitation.

## Negative cases

Reject an unsorted, nonplanar, out-of-solid, or unresolved declared trailing
edge; missing closed solids; unmatched physical walls; non-manifold cut
stars; nonpositive sampled geometry; or unsupported circulation rows.
Rejection is recorded at the earliest stage with sufficient evidence.
It must not trigger pressure clipping, extra diagonal terms, prescribed
root circulation, or singular-value truncation.

## Coordinate and load checks

The explicit chord and origin define normalized coordinates. The imported
geometry is normalized before CAD booleans and meshing; saved coordinates,
lengths, areas and potential have native units. Compare normalized geometry
and dimensionless fields under scaling/translation, while allowing mesh
triangulation differences to be reported rather than concealed.

Pressure loads use original Gauss-point fields and sum each owned face once.
Body and wing force/moment sums must close to the whole-aircraft integral.
Both wall banks and wake traces remain available for raw residual inspection.
Numerical convergence is separate from geometry acceptance and physical
accuracy acceptance. The fixed planar wake and sampled-subsonic full
potential model remain explicit restrictions.
