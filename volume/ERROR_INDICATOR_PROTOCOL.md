# P2 Full-Potential Residual Indicator

This is a separate, post-processing research script. It reads an existing
tet10 mesh and nodal total-potential field; it does not call, import, or modify
the P2 solver. It is not an estimator with a proven bound on Cp, lift, drag,
or solution error. Its purpose is to expose interpretable residual and flux
contributions for mesh-refinement studies.

## Inputs and conventions

The mesh NPZ uses `points (n,3)`, `tets (m,10)` in the existing P2 edge-node
ordering, `boundary_triangles (b,6)`, `boundary_tags`, and optional
`boundary_tet`. A field NPZ contains coordinates/connectivity plus a nodal
total potential: `Phi_total` for the volume solver or `phi` for lifting3d.
Coordinates, connectivity, and potential must agree in size and node order;
the CLI `--mach` is explicit. Physical wall tags are all boundary tags except
100 (the farfield tag). For lifting3d use its original mesh NPZ, not
`topology_input.npz`, so the physical wall is not confused with the duplicated
wake cut.

The field is the solved total potential, with physical coordinates and
potential in compatible length units. No velocity component is projected,
including on walls or wake interfaces. All density-weighted fluxes are raw
`rho*q`.

## Strong residual from the isoparametric Hessian

For each tet10, reference P2 shape Hessians are constant. At each volume
quadrature point assemble the reference field Hessian `Href(phi)`, the three
reference mapping Hessians `Href(x_k)`, and reference-to-physical Jacobian
`J[k,a] = dx_k/dxi_a`. First obtain `q = grad_x(phi)`, then transform the
Hessian analytically:

```text
Hx(phi) = J^-T [Href(phi) - sum_k q_k Href(x_k)] J^-1
```

For `M = Mach`, `beta = 1 + 0.2 M^2 (1 - q.q)`,
`rho = beta^(1/(gamma-1))`, and

```text
div(rho*q) = rho * (trace(Hx) - M^2/beta * q^T Hx q)
```

Any nonpositive beta, non-subsonic quadrature state, or nonpositive isoparametric
Jacobian is a hard error. No clipping or regularization is used.

## Cell contributions and marking

For each volume cell `K`, use its maximum tet10 nodal-pair distance as `hK`.
Its bulk contribution is

```text
eta_bulk,K^2 = hK^2 * integral_K div(rho*q)^2 dV
```

For each conforming internal face `F`, evaluate both adjacent tet10 traces at
the same physical quadrature points and independently orient each normal
outward from its adjacent cell. Do not average either trace. Its contribution
is `hF * integral_F (rho*q.n0 + rho*q.n1)^2 dS`, split equally between its two
cells for marking. `hF` is the maximum distance between the six face nodes.

The external physical wall contribution is
`hF * integral_F (rho*q.n)^2 dS`, assigned to its adjacent cell. It is the raw
normal mass flux, not velocity and not a projected value. Farfield faces are
not treated as walls and do not enter this term.

For lifting3d, the upper/lower wake cut is a duplicated interior interface.
Pair each stored `wake_upper` / `wake_lower` tri6 by physical node coordinates,
evaluate the two raw traces and outward normals at corresponding quadrature
points, and record its area and flux jump separately from ordinary shared
internal faces. Its jump contribution is split between the adjacent cells.
If the two faces do not match geometrically within tolerance, fail rather than
silently pair unrelated faces. For all shared and paired faces also compare
pointwise area Jacobians and the sum of outward area vectors; reject a
non-closing geometry rather than averaging away a mismatch. Once those checks
pass, integrate the squared jump using one side's surface measure.

The total cell indicator is the sum of bulk, internal-face (including a
measured wake-cut contribution when present), and physical-wall terms. Sort
cells by descending total contribution with stable index tie-breaking. Mark
the smallest prefix whose cumulative contribution reaches at least 50% of
the global sum. Save per-cell component values, total values, sorted rank,
cumulative fraction, and the Boolean mark.

## Verification

Before using solver fields, tests must verify:

1. A physical quadratic polynomial on an affine tet10 is recovered with its
   exact physical Hessian.
2. On a curved isoparametric tet10, nodal `phi=x_k` produces exactly constant
   physical gradient and zero physical Hessian using the mapping-Hessian
   correction.
3. The analytical Hessian and strong divergence agree with independent
   centered finite differences of the physical gradient and mass flux.
4. An affine field has zero internal jump on a two-tetrahedron patch, and a
   perturbed shared P2 degree of freedom gives a nonzero raw jump.

## Execution and outputs

Run the CLI with explicit `--mesh`, `--field`, `--mach`, and `--out`;
`--ncpu` defaults to 8, and quadrature order is selectable as 3 or 4. The
script limits BLAS/OpenMP pools to the requested CPU count and records it.
Write `indicator.npz` with one contribution per tetrahedron and `result.json`
with input hashes, integration counts, residual/flux summaries, Dörfler
threshold and marked cell IDs. Failed runs preserve a failure JSON where the
output directory can be created.

Interpret high contributions as locations where this discrete field has
strong residual, inter-element mass-flux mismatch, or raw wall leakage. They
are refinement guidance, not a certified Cp-error bound and not proof that
refinement alone will fix a modeling or boundary-condition error. Lifting3d
results remain an unvalidated lifting reference; the indicator does not
validate wake/Kutta physics or certify lift.
