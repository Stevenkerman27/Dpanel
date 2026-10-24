# Three-Dimensional Full-Potential P1 FEM Protocol

## Scope

This solver is a reference implementation for steady, irrotational, isentropic,
fully subsonic flow around closed bodies in a tetrahedral fluid domain. It is
not a lifting-wing solver: there is no Kutta condition, circulation unknown,
vortex sheet, wake, or sonic/transonic treatment. A tagged wing surface is only
another impermeable body wall. Results must not be presented as general wing
lift predictions.

The discretization is continuous piecewise-linear tetrahedral Galerkin FEM for
the full-potential conservation law. The implementation solves for the total
potential `Phi`, not the perturbation `chi`: `Phi = Uinf dot x + chi`, and
`chi = Phi - Uinf dot x` is only the corresponding decomposition. Its
nonlinear residual is a nodal weak balance. It is not a finite-volume method
and does not enforce one shared conservative flux per tetrahedron face.

## Mesh contract

Input is a NumPy NPZ archive with:

- `points`: `(n, 3)` finite coordinates.
- `tets`: `(m, 4)` integer point indices. Every tetrahedron must be
  nondegenerate and have positive signed volume.
- `boundary_triangles`: `(b, 3)` integer point indices, oriented outward from
  the fluid volume.
- `boundary_tags`: `(b,)` integer labels: `100` farfield, `10` independent
  sphere/ellipsoid, `20` fuselage, `30` wing.
- Optional `boundary_tet`: `(b,)` tetrahedron index adjacent to each boundary
  triangle. If absent, adjacency is inferred from the tetrahedron faces.

The tetrahedral boundary must be a closed two-manifold with exactly one
adjacent tetrahedron per boundary triangle. Boundary tag `100` is mandatory
because the outer Dirichlet condition is applied there. Tags `10`, `20`, and
`30` are natural zero-normal-flux walls. The mesh must describe a connected
fluid region with body surfaces as cavities, not tetrahedra inside the bodies.
The loader checks each supplied orientation against its adjacent tetrahedron
and reorients inconsistent triangles in memory, recording counts by tag;
source mesh files are never modified. Optional `boundary_tet` is treated as a
hint: if its shape, range, or triangle adjacency is inconsistent, adjacency
is inferred from tetrahedron faces and the mismatch count is recorded.

## Governing equation and weak form

Use `gamma = 1.4`, unit freestream speed, and
`Uinf = (cos(alpha) cos(beta), sin(beta), sin(alpha) cos(beta))`, where angles
are in degrees. Define `v = grad(Phi)` and `q2 = dot(v, v)`. Then

```
rho(q2) = [1 + 0.2 M^2 (1 - q2)]^(1 / (gamma - 1))
beta(q2) = 1 + 0.2 M^2 (1 - q2)
F(v) = rho(q2) v
div(F) = 0
```

For test functions vanishing on the farfield Dirichlet boundary, assemble
`R_i = sum_e volume_e * grad(N_i) dot F(grad(Phi_e))`.
Natural wall flux is zero in the weak formulation; do not add a wall penalty
or overwrite the recovered wall-normal gradient.

The exact Jacobian contribution is
`volume_e * grad(N_i) dot [rho (I - M^2 / beta * v outer v)] grad(N_j)`.
Reject any iterate for which `beta <= 0` or the local Mach number is not
strictly below one. Do not clip density, beta, or the local Mach number.
Check `beta > 0` before evaluating `M^2 q2 / beta`. Evaluate the energy
primitive with `log1p`/`expm1`-stable algebra, preserving its derivative `F`.

## Boundary data and nonlinear solve

`uniform` prescribes `Phi = Uinf dot x` at tag `100`.
`sphere-exact` is permitted only when M=0 and an adjacent `mesh.json`
explicitly declares `shape: "sphere"` and a finite positive sphere radius
(`sphere_radius`, `radius`, or `reference_radius`). Missing/unknown metadata,
ellipsoid metadata, and other shapes are rejected. The boundary condition is
the analytic incompressible spherical potential
`Phi = (r + 0.5 R^3/r^2) * dot(Uinf, x)/r` on tag `100`, with center at the
origin. It is a reference boundary condition, not a general farfield model.

Initialize by solving the linear Laplace P1 problem with the selected
Dirichlet data. For M>0, use the uniform boundary data. Then apply damped
Newton iterations to the full nonlinear residual and exact Jacobian. Solve the
free-node sparse system directly; add no arbitrary diagonal regularization.
Backtracking must strictly reduce the original nonlinear residual norm while
retaining `beta > 0` and local Mach `< 1` in every element. Fail explicitly if
no admissible decreasing step exists, a linear solve fails, or tolerances are
not met. Report the residual and energy for the initial field and every
accepted Newton iterate. The energy is the volume integral of the primitive
whose gradient with respect to `v` is `F(v)`.

## Recovery and diagnostics

Recover the constant raw `grad(Phi)` in each original tetrahedron. Compute
surface velocity, density, Cp, and pressure loads by assigning each boundary
triangle the gradient of its adjacent original tetrahedron. Never project away
the wall-normal velocity. For M=0 use `Cp = 1 - |v|^2`; for M>0 evaluate
`Cp = 2/(gamma M^2) * expm1(gamma/(gamma-1) * log1p(0.2 M^2 (1-|v|^2)))`
to preserve the low-Mach limit without subtractive cancellation.
Integrate each non-farfield tag separately. Report force vector, drag along
`Uinf`, and lift along the projection of global +z perpendicular to `Uinf`,
all divided by `sref` (default one).

Report both:

- Internal-face flux jump: evaluate `F(v)` independently on each side, take
  the outward-normal sum, and report area-weighted RMS and maximum absolute
  jump. This is a diagnostic only; it is not constrained to zero by the
  continuous Galerkin method.
- Raw boundary tetrahedron-gradient normal velocity `v dot n` and mass flux
  `rho v dot n` on every boundary tag, including farfield. Name and report
  each separately with area-weighted RMS and maximum absolute value. Wall
  values remain unmodified so they reveal natural-boundary recovery error.
- Signed integrated raw mass flux per tag, total wall tags `10/20/30`, outer
  tag `100`, and total boundary. Report this raw boundary mass balance
  separately from the weak nodal balance: free-node residual sum/L2 norm,
  Dirichlet-node reaction sum, and all-node residual sum. Do not equate nodal
  reactions to boundary flux quadrature.

For a sphere with available analytic reference, additionally report
area-weighted Cp RMS and maximum error on body tag `10`, and analytic lift and
drag reference errors. State the geometric and farfield truncation errors;
do not interpret the exact-boundary test as validation of general body
farfield conditions.

## CLI and artifacts

The command-line interface accepts `--mesh`, `--mach`, `--alpha`, `--beta`,
`--boundary uniform|sphere-exact`, `--sref`, `--ncpu` (default `8`), and
`--out`. Use NumPy, SciPy, and threadpoolctl only. Limit BLAS worker threads
with threadpoolctl. Save `field.npz` with mesh geometry, raw potential,
element gradient, element local-Mach/beta, and recovered surface data.
The stored nodal potential key is `Phi_total`; it is not `chi`.
Save `result.json` with input parameters, convergence/history, termination
status, total/free/Dirichlet degrees of freedom, mesh SHA-256, solver
source SHA-256 available even on input failure, a combined meshcodehash when
the mesh can be read, force/Cp data, and all diagnostics.
Create output directories as needed. A failed solve must return a nonzero
process status and preserve a machine-readable failure result when output
creation was possible.

## Verification gates

The self-contained tests construct a tetrahedral cube mesh and check:

1. An affine potential is reproduced exactly at nodes and in every element.
2. Element gradients recover an affine field to floating-point tolerance.
3. The assembled global weak residual balances for an affine harmonic field
   with matching Dirichlet data.
4. A centered finite difference independently checks the assembled Jacobian
   and the energy directional derivative.
5. Low-Mach Cp and energy evaluations approach their incompressible limits
   without cancellation.

When the independent sphere mesh becomes available, run M=0 coarse and fine
cases with `sphere-exact`, and a coarse M=0.2 case with `uniform`. Check
Newton convergence, strict subsonic admissibility, decreasing sphere Cp error
with refinement for the exact-boundary incompressible sequence, and record
all raw-gradient and flux-jump metrics. A missing mesh is a skipped run, not a
reason to regenerate it or install a mesher.
