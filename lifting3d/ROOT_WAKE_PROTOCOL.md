# Aft-Body Wake Root

## Scope and status

This research branch connects `root_wake.py` and `planform_root.py` to the
lifting3d builder and solver for symmetric straight-semispan-TE planforms,
a planar `z=0` symmetry-plane wake cut, and an x-aligned, span-centered
ellipsoid. Rectangular integrated tests have completed; swept/tapered root
tests are being run separately. This is not a general solid/wake
intersection method. Integration is **not physically validated**, and a
successful mesh/topology check is not an aerodynamic acceptance. NCPU is 8.

The builder supplies `gamma_station_dof` and root-contact metadata. The
solver reduces the Γ unknowns through that station map and requires weak
Kutta when a merged root is declared. Other geometry, wake, or Kutta paths
remain outside this candidate's claims.

## Geometry

Let the ellipsoid center be `x_c = x_aft - L/2`, with axial semiaxis `a=L/2`
and transverse radius `r`. At the rectangular wing TE plane, the ellipsoid
section is

```text
((x_TE-x_c)/a)^2 + (y/r)^2 <= 1
```

When `|x_TE-x_c|<a`, its y interval is `[-y_r,+y_r]`, where

```text
y_r = r * sqrt(1 - ((x_TE-x_c)/a)^2)
```

For this interval to contain the entire body/wake contact arc downstream of
the TE, restrict the geometry to

```text
x_c <= x_TE < x_aft
y_r < b
```

Thus the body's maximum-girth station must be at or ahead of the TE, while
the aft endpoint must be strictly downstream. If `x_c>x_TE`, the downstream
contact arc can extend beyond the TE-plane section and the computed root
interval is not a sufficient plateau. If `x_aft=x_TE`, contact is tangent
and has zero spanwise width; reject it. If `y_r>=b`, no exposed TE remains on
both sides and this candidate rejects the case.

The default candidate uses `x_TE=1`, `x_aft=1.5`, `L=4`, `r=.35`, and wing
half-span `b=2`, giving `x_c=-.5` and `y_r=0.231503...`. This is the section
of the body at the TE plane. It is not a general body/wake intersection
algorithm: the ellipsoid is assumed axis-aligned, the TE is straight, and the
wake cut is the planar symmetry plane `z=0`.

For a swept/tapered wing, `x_TE(y)=1+k*abs(y)`, where
`k=tan(sweep)+.75*(taper-1)/(span/2)`. With `d=1-x_c`, its positive root
solves

```text
A*y_root^2 + B*y_root + C = 0
A = k^2/a^2 + 1/r^2
B = 2*d*k/a^2
C = d^2/a^2 - 1 < 0
```

The preconditions become `x_aft>1`, `x_c<=min(x_TE)`, positive tip chord,
and `0<y_root<span/2`. There is one positive root. The body contact remains
connected and entirely represented by the central plateau; see
`PLANFORM_ROOT_ANALYSIS.md` for the aft-branch proof and counterexamples
outside this scope. `planform_root.py` uses the appropriate algebraic root
expression for either sign of B, without iteration or fitted parameters.

## Local continuity argument

Across an ideal potential jump sheet, write the two traces as `phi+` and
`phi-`, with `Gamma = phi+ - phi-`. Along a smooth connected solid-wall
contact curve parameterized by arclength `s`, the difference between the two
tangential trace velocities is

```text
(u+ - u-) dot t = d(phi+ - phi-)/ds = dGamma/ds
```

If both traces describe one physical wall flow and there is no physical vortex
sheet on that wall, their tangential velocities must agree. Thus `dGamma/ds=0`
is a local necessary compatibility condition along that contact curve.
For this condition alone, it is also sufficient to eliminate the
jump-induced tangential-velocity difference along that curve. It requires Γ
to be constant on each connected wall-contact component. Under the geometric
restrictions above, the body-contact arc is contained in `|y|<=y_r`, so the
piecewise-linear Γ plateau imposes that compatibility.

The plateau is **not** a substitute for impermeability. The upper and lower
body-wall banks must each satisfy their own no-through-flow condition
`u_plus dot n_body = 0` and `u_minus dot n_body = 0`; matching Γ does not set
either wall-normal velocity to zero. Keep both banks topologically distinct
in the wall boundary assembly, and validate raw wall-normal velocity and
normal mass flux separately on each bank and on the contact neighborhood.
An aggregate signed wall flux can cancel between banks and is not this check.
This local constraint also does not prove correct wall pressure, global
potential consistency, Kelvin circulation, wake pressure continuity, or a
valid discretized flow. It does not demand that the common root value be
zero.

## Reduced basis data contract

Before constructing the basis, insert `-y_r` and `+y_r` exactly into the
strictly increasing wake station array. The two wing tips remain its first and
last entries and retain prescribed Γ=0. Every interior station has a base
piecewise-linear nodal circulation coefficient. All station coefficients in
the closed root interval, including both interval endpoints and any interior
stations there, map to one shared root coefficient. All exposed stations
outside the interval retain distinct coefficients.

The returned `station_to_dof` array uses `-1` for prescribed zero tip values
and nonnegative integers for free Γ coefficients. Expanding free coefficients
through this map gives one constant Γ on the root interval. The map does not
set or regularize that common coefficient.

For a weak Kutta row, integrate the ordinary piecewise-linear station hat
functions after applying `station_to_dof`. On an exposed TE segment adjacent
to either root endpoint, the central test function is the endpoint hat and
has positive support. Contributions from the left and right exposed roots
accumulate into the same reduced central row. Do not add a `Γ''=0`
regularization or an independent body-root equation merely to replace this
shared degree of freedom.

The independent support utility computes exact one-dimensional integrals of
the mapped linear hats against `dy` on listed exposed TE intervals. The
solver uses normalized span integrals of raw x-derivative traces. For the
straight symmetric TE in scope, `ds=sqrt(1+k^2)*dy` on both semispans, so
using arc length gives the same normalized rows. Orientation and actual P2
trace gradients are assembled from the real mesh, not the support utility.

## Preconditions and explicit rejection

- The TE plane must cut the ellipsoid in a positive-width interval. A missed
  plane or tangent contact is rejected rather than silently treated as a
  zero-width root.
- The interval must leave exposed TE on both sides. If the body covers the
  entire wing span, this candidate rejects the case; there is no two-sided
  exposed-TE support to test.
- Both root endpoints must be inserted as stations before mapping. The tool
  does not silently alter or resample a caller's station array.
- Only symmetric straight-semispan-TE wings, a centered x-aligned ellipsoid,
  and the `z=0` wake cut are in scope. Body incidence, off-center body,
  nonsymmetric contact sets, curved TE, and nonplanar wakes require a
  geometric intersection and contact-component analysis first.

## Verification

Run the small algebraic checks from the project root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python lifting3d\test_root_wake.py
```

Print the default geometry and basis summary with:

```powershell
python lifting3d\root_wake.py --ncpu 8
```

The algebraic test checks the analytic ellipsoid root interval, zero tip
values, constant root plateau and zero discrete spanwise slope, positive
shared-root weak-test support on both exposed sides, and explicit rejection
of a full-span body, a TE plane that misses the ellipsoid, and missing root
stations. It does not establish physical accuracy.

## Integration Pilot

`results/root_wake_pilot_v1/` records the first integrated mesh and solve
attempt. The mesh completed in about four seconds. Dense P2 Jacobians were
positive (minimum `2.268512e-6`); cut pairing, wake-face pairing, reflection,
exterior-face topology, and root weak-test support checks passed. The TE
station-pairing check failed. Gmsh root-endpoint y coordinates differed from
the analytic root knots by `2.02e-10` and `3.61e-10` chord; x differences
were below `9e-14`, and other stations matched exactly. The pilot solver
therefore stopped at the TE-station match assertion before a flow solve.

The current bounded reconciliation is applied before reflection, to shared
global nodes near the analytic root-strip planes on `z=0` downstream of the
root TE. It restores their span coordinate and the root-TE vertex x
coordinate. Incident elements share these nodes. The selection uses
coordinates rather than OCC entity identity, so independent contact-curve
and strip-conformity audits remain important. Original coordinates and every
change are saved; all original mesh checks are rerun after reconciliation.
The geometry budget and near-tangent limitation are documented in
`ROOT_GEOMETRY_TOLERANCE.md`. No flow equation or geometric gate is relaxed.

`root_wake_pilot_v2` completed all 8 mesh checks and a nonzero-root
circulation solve. The rectangular 11-case matrix also completed all mesh
and finite-flow gates; see `GEOMETRY_ROBUSTNESS_RESULTS.md`. The first
near-tangent `aft=1.0001` attempt failed at CAD station pairing; the
condition-aware coordinate budget then passed all 8 original mesh checks
and a flow solve in `root_near_repair_v1`. These are geometry and algebra
results, not wall impermeability, Cp, or mesh-convergence certification.
