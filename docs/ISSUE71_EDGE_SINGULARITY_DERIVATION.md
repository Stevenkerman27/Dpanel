# Issue #71 — derive the edge singularity before enriching the basis

Status: analytical oracle and tests only. No production solver change.

## 1. Start from the physical boundary condition

For inviscid incompressible irrotational flow, the total potential satisfies

    Laplace(Phi) = 0 in the fluid,
    d Phi / d n = 0 on an impermeable stationary body.

DPanel's double-layer unknown is the physical surface-potential trace. The
boundary integral equation is an indirect representation of this Neumann
problem; do not interpret the unknown as a constant-potential Dirichlet
boundary condition. The smooth-sphere manufactured solution already used in
Issue #71 is an important control.

## 2. Local straight-edge wedge

Assume locally that an edge is straight and its cross-section is a fluid wedge

    0 < r < r0, 0 < theta < Omega,

with homogeneous Neumann conditions on theta=0 and theta=Omega. Seek a
separated harmonic solution

    phi(r,theta) = r^lambda f(theta).

The polar Laplacian gives

    f'' + lambda^2 f = 0.

The face condition is d phi/d theta = 0, so f'(0)=f'(Omega)=0.
The nonconstant eigenmodes are

    f(theta)=cos(lambda theta),
    lambda Omega = n*pi, n=1,2,...

The leading nonconstant mode therefore has

    lambda = pi/Omega,
    phi_s = a r^(pi/Omega) cos(pi theta/Omega),
    |grad(phi_s)| = |a| (pi/Omega) r^(pi/Omega - 1).

For a convex solid whose interior cross-sectional angle is beta, the
surrounding fluid angle is Omega=2*pi-beta. Thus pi<Omega<2*pi and
1/2<lambda<1. In the zero-thickness plate/crack limit beta->0, Omega->2*pi,
lambda->1/2 and velocity behaves as r^(-1/2). For a 90-degree solid corner,
Omega=3*pi/2, lambda=2/3 and velocity behaves as r^(-1/3).

This is a local asymptotic mode for a homogeneous Neumann wedge, not a claim
that every LE, TE and 3-D wing tip has the same exponent. A 3-D edge may have
variation along its edge, multiple interacting edges, and different local
boundary conditions. The cross-sectional angle and physical edge type must
be identified separately.

## 3. What P1 enrichment would mean

On the two wedge faces, the leading mode's traces are

    phi_s(r,0)       = +a r^lambda,
    phi_s(r,Omega)   = a r^lambda cos(pi) = -a r^lambda.

So a candidate trace enrichment is a cutoff version of r^lambda on each
edge bank, with the correct signed pairing. In a real 3-D mesh this requires
a geometric edge coordinate, a cutoff along the edge, and compatibility at
edge intersections (LE/TE/tip). It cannot be inserted as one global scalar
basis without specifying which edge and bank it belongs to.

The velocity itself is singular and is not a pointwise bounded target at
r=0. Do not demand convergence of max|V| including the mathematical corner.
Use fixed physical points with r>0, convergence of the local field away from
the singular set, and suitable weighted/energy norms. Report the corner
exponent separately.

## 4. Decision gates before implementation

1. Identify each edge's solid interior angle beta from geometry, and hence
   Omega and lambda. Do not fit lambda to make computed CL improve.
2. Validate the mode against the exact Laplacian and both no-penetration
   face conditions (tests in test_p1_edge_singularity.py).
3. Use the current P1 solution and the singular oracle to measure the
   residual in edge-distance bins. A fitted exponent is diagnostic only; it
   must be compared with the geometrically predicted exponent.
4. Build an opt-in enriched trial/test space only after the edge classification
   and exponent are stable. Use a cutoff and partition of unity; keep the
   original P1 coefficients and enrichment coefficients separately observable.
5. Compare against a separate Galerkin implementation, not by switching the
   production solver in place. Both routes must pass the sphere oracle and
   identical frozen geometry/TE/wake conditions.

## 5. Current implementation

panel/p1_edge_singularity.py implements the analytical exponent and the
Neumann wedge mode. test/test_p1_edge_singularity.py checks the crack limit,
a 90-degree solid corner, exact no-penetration on both wedge faces, trace
signs and input validation. These are analytical tests; they do not claim
that the G1/G2 wing has already been enriched or that its CL has converged.

## 6. Local measured traces: LE supports the mode; TE exposes a topology/discrete-space issue

Local PR #74 receipt (G1 closed-body/no-cut/no-wake/Gamma=0, alpha=5 degrees, mid-span) reports the LE upper-minus-lower trace difference over x in [0.005,0.05] with fitted slopes +0.527 at c=16 and +0.447 at c=24. Each fit has only two points, so this is suggestive and not a robust exponent regression, but it is consistent with the common LE geometric vertex and the crack-limit r^(1/2) antisymmetric trace.

The TE behaves differently: the upper-minus-lower trace remains approximately O(1) over the chord and collapses to zero only in the last chordwise cell, because the closed-body/no-cut mesh shares the geometric TE potential DOF. The inferred collapse width is one cell, so the gradient over that cell scales like O(1/h). This is evidence of a trial-space/topology mismatch for a lifting sharp TE in the no-cut control, not proof of a particular TE velocity exponent.

The local report explicitly retracted velocity log-log slope as an acceptance quantity: fitting windows s<0.2 versus s in [0.02,0.15] changed the slope substantially and even changed its sign at some refinements. Use fixed physical off-surface points for field convergence; use potential trace shape and the declared local wedge model for singularity analysis. Do not fit an exponent to improve CL.

Implication: a physical lifting TE must use a declared cut and a compatible Gamma jump space. Enrichment and cut topology are coupled, but the Galerkin baseline should first be validated without enrichment. Then add a cutoff r^lambda edge basis on each bank with independent coefficients and compare on the same frozen cut/wake.