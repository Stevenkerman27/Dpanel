# Curved P2 Full-Potential FEM Protocol

## Scope

This is a bounded three-dimensional, closed-body, non-lifting reference. It
does not implement Kutta conditions, circulation, a wake, or sonic/transonic
flow. A wing tag, if present, is an impermeable wall only. This work is
independent of `airfoil2d/` and must not modify the P1 reference solver.

The unknown is total potential `Phi`, with `Phi = Uinf dot x + chi`; the
implementation solves for and stores `Phi`, not `chi`. The method is
continuous isoparametric quadratic tetrahedral Galerkin FEM. Its nonlinear
residual is a weak nodal balance, not a finite-volume shared-face flux.

## Input mesh

The NPZ contains `points (n,3)`, `tets (m,10)`,
`boundary_triangles (b,6)`, `boundary_tags (b,)`, and optionally
`boundary_tet (b,)`. Existing boundary tags retain their meanings:
`100` farfield, `10` sphere/ellipsoid, `20` fuselage, `30` wing.

Tetrahedron node order is exactly
`0,1,2,3,01,12,20,03,13,23`; triangle order is exactly
`0,1,2,01,12,20`. The first four/three nodes define the corner simplex.
Every corner tetrahedron must have positive orientation. The boundary is a
closed two-manifold, each boundary face must have one adjacent tetrahedron,
every point node must be referenced by a tet10, and tag `100` is mandatory.
Boundary orientation is checked against the
adjacent fluid tetrahedron and corrected in memory if needed. A malformed or
mismatched `boundary_tet` is ignored in favor of corner-face adjacency and is
reported. Curved tetrahedron Jacobian determinants must remain positive at
all volume quadrature points.

## Geometry, quadrature, and equation

Use `gamma=1.4`, unit freestream speed, and
`Uinf=(cos(alpha)cos(beta), sin(beta), sin(alpha)cos(beta))`, with input
angles in degrees. The P2 Lagrange shapes are four vertex functions
`Ni=lambda_i(2 lambda_i-1)` and six edge functions
`Nij=4 lambda_i lambda_j` in the node order above. Map coordinates with these
same shapes. Transform reference gradients with the inverse isoparametric
Jacobian and integrate with its positive determinant.

Use tensor-product Gauss-Legendre quadrature mapped by a Duffy transform from
the unit cube to the reference tetrahedron. Pair every point with the point
obtained by swapping barycentric columns 1 and 2, assigning half the original
weight to each. This averages the rule with its exact coordinate-permuted
counterpart: `--quad-order 3` and `4` use 54 and 128 volume points, and 18 and
32 face points, respectively. Pairing preserves the original rule's exactness
and total measure while making the rule invariant to this barycentric
permutation. Record the chosen order and actual point counts in every result
to permit quadrature-sensitivity comparisons.

At each volume quadrature point, define `v=grad(Phi)`, `q2=v dot v`,
`beta=1+0.2 M^2(1-q2)`, `rho=beta^(1/(gamma-1))`, and `F=rho v`.
Assemble
`R_i=sum_e,q wq det(J) grad(N_i) dot F`.
The exact constitutive Jacobian is
`rho [I - M^2/beta v outer v]`. Check `beta>0` before division and reject
any evaluated quadrature state with local Mach `M^2 q2/beta >= 1`; never
clip or regularize.

Evaluate the energy primitive with cancellation-resistant `log1p`/`expm1`
algebra. For M>0 compute
`Cp=2/(gamma M^2) expm1(gamma/(gamma-1) log1p(0.2 M^2(1-q2)))`;
at M=0 use `Cp=1-q2`. Include derivative/finite-difference and low-Mach
limits in the tests.

## Boundary and nonlinear solve

`uniform` prescribes `Phi=Uinf dot x` at farfield tag `100`.
`sphere-exact` requires M=0 and adjacent `mesh.json` with
`shape=="sphere"` and an explicit positive radius. Reject ellipsoid, unknown,
or missing shape/radius metadata. Prescribe the analytic incompressible
spherical total potential at all farfield nodes; this is a reference
condition, not a general farfield model.

Initialize with the isoparametric P2 Laplace problem and the selected
Dirichlet values. For M>0 use uniform farfield data. Apply exact-Jacobian
damped Newton with sparse direct solves and residual-decreasing line search.
Every trial must satisfy strict subsonic admissibility at quadrature points.
Do not add arbitrary diagonal regularization. Report residual and energy for
the Laplace start and each accepted iterate. Failed solves must return nonzero
status and preserve a JSON failure result when the output directory is usable.

## Surface recovery and conservation reporting

Integrate each curved tri6 surface using its isoparametric coordinates and
surface Jacobian. Evaluate the adjacent tetrahedron's full physical P2
gradient at each surface point; do not project out wall-normal velocity.
Integrate Cp loads and raw `v dot n` and `rho v dot n` separately on each
boundary tag. Report signed integrated mass flux for wall tags `10/20/30`,
outer tag `100`, and total boundary.

Report weak nodal balance separately: free-node residual L2/sum,
Dirichlet-node reaction sum, and all-node residual sum. These are not the same
quantity as independently quadrature-integrated raw boundary mass flux.
Report the signed total raw boundary mass-flux closure and its ratio to the sum
of absolute per-tag fluxes as a measured mass-gate residual; do not imply
acceptance without an explicitly selected tolerance. These are not the same
quantity as the weak nodal balance.

For every interior corner face, reconstruct its shared tri6 nodes and
independently evaluate both adjacent tet10 isoparametric maps, physical
gradients, and `rho*v` at the same triangle quadrature coordinates. Integrate
the outward flux jump `F0 dot n0 + F1 dot n1` without averaging/projecting
either trace. Report both signed side integrals, signed and absolute jump
integrals, area-weighted RMS and maximum pointwise jump. Check coordinate
agreement, pointwise area-Jacobian agreement, integrated area on each side,
and cancellation of the two outward area vectors. This is an element-trace
diagnostic; it is distinct from external-boundary closure and weak nodal
balance.

For M=0 sphere metadata, compare area-weighted surface Cp RMS/max against the
analytic unit-direction sphere Cp at curved face quadrature points for either
farfield boundary prescription. Report integrated lift/drag and analytic
zero-load references. Compare uniform and sphere-exact data on the same
farfield-radius mesh as a boundary-condition sensitivity check, not as a
domain-radius truncation study. State that coarse curved geometry,
interpolation, and quadrature remain in the error. For ellipsoid, report loads
and conservation diagnostics but no sphere reference.

## CLI, files, and verification gates

The CLI follows `solve.py`: `--mesh`, `--mach`, `--alpha`, `--beta`,
`--boundary uniform|sphere-exact`, `--sref`, `--ncpu` (default 8), `--out`,
plus `--quad-order 3|4` (default 3). Use NumPy, SciPy, and threadpoolctl only.
Save `field.npz` and `result.json` including mesh/source SHA-256, combined
meshcodehash, node/element/free/Dirichlet DOFs, quadrature order, solver
history, raw fields, surface outputs, loads, weak balance, raw boundary
mass-flux integrals, raw internal-face two-sided flux jumps, and both
geometric area-closure checks.

Reference-mesh gates before body runs:

1. P2 shapes sum to one and their reference gradients sum to zero.
2. A physical quadratic polynomial is reproduced by isoparametric mapping
   and physical gradients on a curved tet10.
3. The assembled Jacobian agrees with independent centered residual
   differences; energy directional derivative agrees with the weak residual.
4. Low-Mach Cp and energy approach their incompressible limits.
5. Compare quadrature orders 3 and 4 on reference tests and record sensitivity.

When the prescribed meshes arrive, run M=0 sphere with `sphere-exact`,
M=0.2 sphere with `uniform`, and M=0 ellipsoid with `uniform`. The sphere
sequence must show Cp error and force diagnostics; the ellipsoid must
converge without a sphere reference. Preserve all outputs under
`volume/results/p2_initial/`. Do not regenerate or install a mesher.
