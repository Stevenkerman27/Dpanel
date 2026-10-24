# Swept/Tapered Aft-Body Root Analysis

## Scope

This is a read-only mathematical extension study for a symmetric trapezoidal
wing, straight quarter-chord sweep, an x-aligned span-centered ellipsoid, and
a fixed planar `z=0` wake cut. No solver, builder, or `root_wake.py` code is
changed. The frozen core matrices remain untouched. This derivation does not
claim that the current builder supports swept or tapered aft-body roots.

## Geometry and contact set

Let the wing half-span be `b=span/2`. With root chord one, tip/root chord
ratio `lambda>0`, and quarter-chord sweep angle `Lambda`, the local chord and
trailing-edge coordinate on either semispan are

```text
c(y)   = 1 + (lambda - 1) |y|/b
x_TE(y) = 1 + k |y|
k       = tan(Lambda) + 0.75 (lambda - 1)/b
```

This formula assumes straight leading and trailing edges on each semispan,
no twist, and a chordwise direction parallel to global x. The fuselage
ellipsoid is

```text
(x-x_c)^2/a^2 + y^2/r^2 + z^2/r^2 = 1
x_aft = x_c + a
```

On its aft branch in the symmetry plane, its surface is

```text
x_body(y) = x_c + a sqrt(1 - y^2/r^2),  |y| <= r
```

For `y>=0`, define `d=1-x_c` and `q(y)=d+k y`. If

```text
x_aft > 1
x_c <= min_(0<=y<=b) x_TE(y) = 1 + min(k,0) b
```

then `0<=d<a` and `q(y)>=0` over the semispan. The second condition keeps the
TE plane on or aft of the ellipsoid maximum-girth station, including when
`k<0` and the minimum TE is at the tip.

The lower bound is non-strict: `x_c=1` gives `d=0` and is a valid finite
intersection when `x_aft>1`. For `k=0`, its positive root is exactly `r`;
for nonzero `k`, the same quadratic and unique-positive-root argument apply.
The separate maximum-girth condition still applies and may impose the
stronger bound `d>=-k b` when `k<0`.

The body/wake contact arc is the aft ellipsoid branch where
`x_body(y)>=x_TE(y)`. Its endpoint solves

```text
(q(y)/a)^2 + (y/r)^2 = 1
```

or

```text
A y^2 + B y + C = 0
A = k^2/a^2 + 1/r^2 > 0
B = 2 d k/a^2
C = d^2/a^2 - 1 < 0
```

Since the product of the two roots is `C/A<0`, there is exactly one positive
root and one negative root. The positive root is

```text
y_root = (-B + sqrt(B^2 - 4 A C))/(2 A)
       = -2 C/(B + sqrt(B^2 - 4 A C))
```

The second form avoids subtractive cancellation; its denominator is positive
because `C<0` implies `sqrt(B^2-4AC)>|B|`. Symmetry gives the contact interval
`[-y_root,+y_root]`. The set is connected: on each semispan the body aft
branch `x_body(y)` is strictly decreasing for `y>0`, while the TE is affine;
equivalently, under the stated branch condition the quadratic is negative
between its two roots.

For a usable lifting TE, additionally require

```text
0 < y_root < b
c(y) > 0 for 0 <= y <= b
```

If `y_root>=b`, the body covers the full span at the TE and there is no
two-sided exposed TE support for this method. If `x_aft=1`, the root contact
collapses to zero width for the root chord station; reject the tangent
topology instead of creating a zero-width plateau.

## Necessity of the maximum-girth condition

The condition `x_c<=min(x_TE)` is sufficient to identify the quadratic's
positive root with the aft-branch contact endpoint. It is not a general
condition for connectedness of the contact set. Concavity of
`x_body(y)-x_TE(y)` also makes its nonnegative set connected on one
semispan whenever the body aft tip lies beyond the root TE. However, if
`x_c>min(x_TE)`, the squared equation can select the ellipsoid's forward
branch or miss the actual contact endpoint, so the proposed `y_root` formula
is not reliable without a branch-aware geometric intersection.

For example, with `k=0`, `x_c=1.2`, `a=2`, `r=.35`, and `x_TE=1`, the
quadratic at the TE plane gives about `0.3482`, but the wake begins ahead of
the ellipsoid maximum girth. The actual aft body/wake contact reaches
`|y|=r=.35`. The contact remains connected, but the TE-plane root is the
wrong boundary for the plateau.

If `x_aft<1`, the root-chord TE is not crossed, but a negative `k` can put
outboard local TE stations farther forward; that case needs a separate
branch-aware intersection test and is outside this proof. If `x_aft=1`, the
root-TE contact is tangent. These cases should not be conflated with a finite
crossing under the stated assumptions.

## Circulation constraint

Parameterize the connected body/wake contact arc by `y`, with
`x=x_body(y)` and `z=0`. For a potential jump `Gamma(y)` across the planar
wake, its derivative along the contact tangent is

```text
d Gamma/ds = Gamma'(y) dy/ds
```

For a nondegenerate arc, `dy/ds` is nonzero away from its center, so a
constant Γ on `[-y_root,+y_root]` removes this jump-induced tangential
velocity difference along the complete contact arc. A piecewise-linear basis
can impose this by inserting both root endpoints and mapping every station
inside the closed interval, including its endpoints, to the same free root
DOF. The shared value remains unknown; do not impose zero circulation there
or add an artificial `Gamma''=0` term. The weak Kutta test for that DOF must
retain support on both exposed root-adjacent TE intervals after the mapping.

This is a local compatibility condition, not a complete wall boundary
condition. The upper and lower body-wall banks still require separate
no-through-flow conditions. Neither a constant Γ nor a weak Kutta equation
sets either bank's normal velocity to zero. A future integrated validation
must report raw wall-normal velocity and mass flux independently by bank;
an aggregate signed flux can hide cancellation. The contact-endpoint
neighborhood also needs explicit mesh and trace checks.

## Applicability to sweep and taper

The same one-root/plateau construction is geometrically valid for the
symmetric linear `x_TE(y)=1+k|y|` family when all of these hold:

- The body is an axis-aligned ellipsoid centered on the wing symmetry plane.
- The TE is straight on each semispan and `k` is constant there.
- `x_c<=min(x_TE)` and `x_aft>1`, so the full downstream contact arc is on
  the aft branch represented by the equation.
- The positive root is strictly inside the half-span and local chord remains
  positive.
- The wake cut is planar and its circulation coordinate is physical spanwise
  `y`; the wake/body intersection is conformingly represented in the mesh.
- Both root stations are exact mesh/TE stations and every reduced weak-Kutta
  basis function has nonzero support on exposed TE edges.

For these straight semispan TE edges, physical edge measure is
`ds=sqrt(1+k^2) dy`. The factor is constant and equal on both sides, so it
does not change a normalized symmetric support test; the production weak
equation must still use its declared physical edge measure and actual trace
derivative. The central reduced test is one Galerkin row with support on both
root-adjacent exposed intervals, not two pointwise Kutta conditions.

This extension is not valid as stated for curved or nonsymmetric TE, varying
sweep, an off-center or tilted body, a nonplanar/free wake, or a nonlinear
material-span coordinate. Those cases need the actual body/wake intersection
components first; a scalar `y_root` may become multiple intervals or cease to
describe the contact set.

## Verification state

The first rectangular pilot (v1) failed exact TE/root station pairing. The
corrected pilot (v2) completed the 11-case matrix, including mesh and solver
cases. Its two CAD root endpoints were reconciled to the analytic strip
planes with a maximum coordinate shift of `3.607234511093793e-10`; the
existing geometric and aerodynamic gates were not relaxed. This is evidence
for the rectangular integrated case only, not for swept/tapered roots.

The builder's reconciliation budget is interpreted as a first-order bound:
if the CAD absolute positional tolerance is `epsilon` and `n_y` is the
absolute y component of the unit normal to the analytic root intersection,
then a y correction `delta_y` produces normal displacement approximately
`|n_y| delta_y`, so `delta_y <= epsilon/|n_y|`. It is further restricted to
less than one percent of both the root half-width and exposed half-span
length. This explains the scale of an allowed correction, but does not alone
prove that every selected mesh node is the intended OCC intersection vertex
or that the adjusted point remains on the CAD surface. The implementation
selects by cut-plane, downstream-x, and proximity conditions rather than
explicit OCC entity identity. Therefore the recorded original coordinates,
node count, maximum shift, analytic ellipsoid residual, reflected-pair
consistency, and post-adjustment P2 Jacobian remain necessary audit evidence.
No swept/tapered aft-body mesh or solver run is claimed here. Existing
aerodynamic precision gates remain unchanged and are not inferred from this
derivation.
