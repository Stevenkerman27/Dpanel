# Root intersection geometry budget

The first near-tangent test (`aft=1.0001`, `L=4`, `r=.35`) failed the original
TE station check. The captured CAD vertices differ from the analytic root
stations by `1.07e-8` and `1.25e-8` chords. The original failed mesh and the
unmodified-builder diagnostic are retained in `results/geometry_limits_v1`
and `results/root_near_probe_v1`. This is a geometry failure before a flow
solve, not an ill-conditioned aerodynamic system.

At the analytic ellipsoid/TE intersection `(1,y_root,0)`, the normal is
parallel to `((1-xc)/a^2, y_root/r^2, 0)`. A surface displacement changes its
intersection with the fixed TE plane with first-order sensitivity
`delta_y = delta_normal / abs(n_y)`. Near tangency, `abs(n_y)` approaches zero.
The builder uses the explicit CAD tolerance `epsilon=1e-8` and a coordinate
budget `epsilon/abs(n_y)` for reconciling the two declared root strip planes.
This is a first-order resolution budget, not a theorem about OCC's errors.

Before building CAD, the budget must be below one percent of both the root
half-width and remaining exposed semispan. Smaller unresolved contact sets
are explicitly rejected. A uniform wake knot within 100 coordinate budgets
of a root knot is replaced by the analytic root knot, avoiding a strip whose
width is comparable to geometric uncertainty. Wing-tip knots are preserved.
This adjusts the discretization partition, not the solid shape.

After meshing, only `z=0` nodes downstream of the TE and within the coordinate
budget of a root strip plane are restored to that analytic y coordinate.
Every original moved coordinate, node index, maximum displacement, normal
component, CAD tolerance, and coordinate budget is saved. Dense P2 Jacobian,
one-to-one cut pairing, exterior-face adjacency, and the original TE station
checks must still pass after this operation. The operation cannot remove a
bad Jacobian or relax an acceptance gate.

For a straight swept/tapered TE `x=1+k*y` on the positive semispan, a root
coordinate correction follows `(delta_x,delta_y)=(k,1)*delta_y`.
The corresponding budget is `epsilon/abs(n_y+k*n_x)`. The exact root vertex
is restored in both x and y; other downstream strip nodes retain x and only
restore y. Saved metadata distinguishes the y-coordinate budget from the
maximum coordinate displacement. Rectangular wings retain `k=0`.

The flow equations, Kutta test space, pressure samples, and circulation
unknowns are unchanged by this tolerance policy. There is no pressure
clipping, SVD truncation, vortex-core adjustment, or prescribed root lift.
The policy does not support an exact tangent contact or general CAD.
