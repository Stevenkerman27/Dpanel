# Cartesian Two-Dimensional Full-Potential E-O Kernel Protocol

## Scope and Theory Gate

This protocol tests an independent Cartesian grid-point/finite-volume residual
prototype. It does not modify or certify the existing airfoil solver, solve a
transonic airfoil, or equate the scalar E-O Riemann study with a 2D result.
The previously tested centered-density bias remains rejected.

NASA TM-85751 Section VI, printed pages 38-40 (PDF pages 39-41), defines
`J_inv=det(d(xi)/d(x))`, `U_i=a_ij phi_xj`, and the conservation flux
`rho U_i/J_inv`. `U_i` is the general contravariant velocity without a
Jacobian. For a forward map, `J_phys=1/J_inv` and the flux is
`J_phys rho U_i`. Section VI places the E-O split on total speed
`q=|grad(phi)|`, multiplied by the signed directional projection `U_i/q`;
it does not say to decide the split solely from the physical face-normal
Mach number. Equation (6.2) is a directional difference scheme with
one-sided metric-gradient coefficients. The shared-face construction below
is a conservative Cartesian FV interpretation to be tested, not a claim of
algebraic identity with every stencil term of (6.2).

For a unit physical face normal `n`, define the averaged face gradient `g_f`,
`q_f=|g_f|`, and `c_f=(g_f dot n)/q_f`. Let

```text
h_EO(q1,q0) = -m(q1)+s(q1)-s(q0)
m(q) = rho(q) q
s(q) = 0                         when q <= q*
s(q) = m(q)-m(q*)                when q > q*
```

The candidate shared Cartesian face flux is

```text
F_f = |S_f| c_f [-h_EO(q_R,q_L)]     if c_f >= 0
F_f = |S_f| c_f [-h_EO(q_L,q_R)]     if c_f < 0
```

where `q_L,q_R` are the magnitudes of the two potential-derived face-side
gradients. Each face flux is evaluated once and enters neighboring cell
balances with opposite signs. This is a proposed directional extension; the
scalar E-flux result alone is not a multidimensional entropy proof.

The separate normal-only diagnostic freezes tangential velocity and splits
with the normal scalar state. It is retained only as a comparator. On an
oblique supersonic flow, total Mach can exceed one while normal Mach is below
one; that demonstrates the normal sensor's blindness to the total sonic
switch used in Section VI, but is not by itself a proof that a face-normal
characteristic split violates a multidimensional entropy inequality.

## Frozen Acceptance Criteria

Use `gamma=1.4`, `M_inf=0.5,0.8`, double precision, and set numerical-library
thread variables to 8. No clipping, added viscosity, filtering, or sign
changes are allowed.

1. Constant-gradient free stream: maximum periodic control-volume mass
   residual divided by `max(1, rho |grad(phi)|)` is at most `2e-12`.
2. Discrete potential gradients are obtained by commuting centered
   derivatives of one scalar potential. The maximum discrete curl must be at
   most `2e-12` times the gradient scale.
3. A rigid rotation of gradients and all physical face area vectors must
   leave each face flux and the assembled residual invariant to `2e-12`
   relative scale.
4. A potential depending on x only must give zero transverse flux and reduce
   the x-face value to the one-dimensional E-O mass split within `2e-12`.
   Repeat after reversing the flow and face orientation.
5. For the periodic smooth manufactured potential, compare the residual to
   the analytic `div(rho grad(phi))`. On `N=32,64,128,256`, the L-infinity
   error must decrease at every refinement and the last two observed orders
   must each be at least `0.7`. This is a consistency gate, not a claim of
   second-order accuracy.
6. Construct oblique total-supersonic states whose face-normal Mach is
   subsonic and explicitly report whether the normal-only switch is inactive
   while the Section-VI total-speed switch is active. Then test the proposed
   projected total-speed face flux against the scalar normal E-flux inequality
   with frozen tangential states, dense endpoint pairs and 101 interior
   states. The maximum positive violation must be no greater than `2e-10`.
   Also report a Rankine-Hugoniot-matched planar compression state; do not
   mistake the numerical flux at a discontinuity for the physical RH flux.

Any failed numerical threshold remains in the result JSON and makes the
candidate status `FAIL`. A failure of the frozen-tangent scalar diagnostic
does not prove the full 2D entropy theorem false; it does block this face
construction pending a derivation of an appropriate multidimensional entropy
flux. Passing all listed tests still does not certify a 2D transonic weak
solution or an airfoil calculation.

## Result Classification

Results distinguish:

- `PASS`: an algebraic/geometric test met its stated threshold.
- `FAIL`: a frozen threshold was exceeded.
- `DIAGNOSTIC`: measured evidence without an acceptance claim.
- `NOT_CERTIFIED`: the prototype has not established a fully-discrete
  multidimensional entropy inequality, nonlinear convergence, shock
  admissibility, or transonic-airfoil accuracy.

Store implementation/test/protocol hashes, all mesh errors, sensor
counterexamples, shock states, and each gate status under
`potential/results/transonic_cartesian_v1/`.

## Round-1 Outcome and Theory Revision

The round-1 shared-face candidate failed the frozen-tangential oblique E-flux
gate and is rejected. Its maximum positive violations were `0.3645056192`
for `M_inf=0.5` and `0.3936720150` for `M_inf=0.8`; the corresponding
normal-frozen-tangent scalar E-O reference violations were
`2.81e-16` and `1.67e-16`. This is not fixed by changing the flux sign: the
candidate and reference use the same `h=-m+s(q1)-s(q0)` convention and the
same physical normal mass-flux entropy comparison. Retain the counterstates
in the result JSON.

The total-speed sonic switch was active while the normal-only switch was
inactive in both oblique sensor cases. The total/normal Mach-squared pairs
were approximately `(1.47,0.37)` and `(1.44,0.36)`. This confirms the
normal-only sensor does not see the full-potential type switch in that face
direction. It does not, alone, make the normal E-O scalar flux incorrect:
the face-normal scalar characteristic is controlled by its normal derivative.

The residual, discrete-curl, rigid-rotation, free-stream and both 1D
degeneration checks passed. Smooth manufactured residual errors decreased
approximately first order: `M_inf=0.5` observed orders
`1.026,1.007,1.002`; `M_inf=0.8` orders `1.063,1.017,1.004`. The RH-matched
oblique planar compression states had normal mass mismatch below `1.8e-14`.
These checks do not turn the numerical flux at a discontinuous face into a
steady shock solution. Overall round-1 status remains `FAIL`.

The matrix multiplying second derivatives is

```text
A(grad(phi)) = rho [ I - (M_inf^2/beta) grad(phi) outer grad(phi) ]
```

Its eigenvalues are `rho` transverse to `grad(phi)` and
`rho(1-M_local^2)` along `grad(phi)`. Thus a Cartesian mesh can have
subsonic normal characteristic coefficients on each coordinate face while
the full matrix is already indefinite. The failed construction
`(U_n/q) * scalar_EO(q_R,q_L)` does not satisfy even the frozen-tangential
normal E-flux comparison for the tested oblique states. It is rejected as a
2D shared-face flux.

The normal-frozen-tangent E-O flux is retained only as a scalar face baseline:
it passes that local inequality, but coordinate-face scalar inequalities do
not control the cross-derivative/flow-aligned negative eigenmode when no
coordinate direction resolves it. The next admissible theoretical step is a
flow-aligned or genuinely multidimensional entropy-compatible stencil/flux
whose shared conservation balance includes the full matrix above. It must be
derived and independently tested for rotation covariance and a discrete
entropy inequality before any 2D nonlinear solve. No extra dissipation is
authorized as a substitute for that derivation.
