# Issue #71 — first P1 Galerkin wall-operator prototype

Status: opt-in research only. The existing P1 collocation solver is unchanged.
This first step assembles the weak wall operator; it does not yet solve the
lifting problem or define a new Kutta condition.

## Formulation

Let the physical surface-potential trace be expanded in the existing
continuous P1 trial basis N_j (with only the declared physical TE cut
duplicated). The wall equation is the interior trace of D_S(mu)+Phi_inf,
plus the wake contribution D_W(Gamma). Testing against N_i gives

    A_G[i,j] = integral_S N_i(x) D_S^-(N_j)(x) dS_x
    W_G[i,g] = integral_S N_i(x) D_W(H_g)(x) dS_x
    b_G[i]   = -integral_S N_i(x) Phi_inf(x) dS_x.

The source-triangle P1 double-layer integral is evaluated by the current
analytic _doublet_matrix; the target-face test integration is evaluated by
a Duffy-transformed tensor Gauss rule. The exact own-face interior trace
is supplied by _doublet_matrix as -1/2 times the full barycentric vector.
Its weak integral is therefore the exact local P1 mass contribution
-1/2 M_T, with

    M_T = area(T)/12 * [[2,1,1],[1,2,1],[1,1,2]].

The implementation also assembles the wake coupling W_G by integrating the
wake source influence over the target tests. This is not a row-weighted
collocation matrix: each row is a basis test integral.

## Files

- panel/p1_galerkin_wall.py: exact P1 mass matrix and experimental A_G, W_G, b_G assembly.
- test/test_p1_galerkin_wall.py: mass symmetry/positive definiteness,
  partition-of-unity RHS integral, exact local self-jump mass, and input guards.
- The operator uses the actual production P1 source kernel. It deliberately
  keeps physical_accuracy_accepted=false.

## Limitations / gates before use

1. The outer target integral currently uses a fixed Duffy tensor rule.
   Shared-edge/near-edge interactions may need adaptive subdivision and an
   order study; increasing order is not automatically a proof of accuracy.
2. The existing analytic source integral can itself fail on extreme
   near-singular geometry. Those cases must be isolated and handled with a
   verified singular/near-singular strategy, not silently clipped.
3. No Kutta, J constraint, pressure closure, nonlinear solve or loads are
   assembled here. A square wall matrix alone is not the coupled lifting
   formulation.
4. Before solving, test the closed-body constant-mode identity
   A_G @ ones = -M @ ones under refinement, compare independent source
   coefficients, and verify sphere manufactured convergence.
5. Compare collocation and Galerkin on identical frozen geometry, freestream,
   physical trace space, and prescribed Gamma. Only then add the cut/wake
   and study Kutta. The wedge singular enrichment remains a separate follow-up.

## Local commands

    python -m pytest test/test_p1_galerkin_wall.py test/test_p1_edge_singularity.py -q

Do not switch production G1/G2 to this operator or claim physical convergence
until the operator identities and mesh-refinement checks pass.
## Integrated framework update

The branch now includes an end-to-end experimental coupled solver, not just a wall matrix:

- panel/p1_galerkin_solver.py assembles the true P1 weak wall matrix, wake doublet coupling, exact P1 mass, and freestream moments for one or multiple lifting groups. It reuses the same geometry, Gamma basis, TE jump matrix, velocity gradients and Kutta sampling pairs as G2.
- solve_galerkin_multi uses the mass-dual norm ||M^(-1/2) r_G|| for the weak residual. It can enforce the existing signed-speed Kutta equations exactly, or solve the pressure-Kutta nonlinear problem while preserving the TE jump exactly. It reports raw wall RMS, dual wall RMS, J residual, signed Kutta, pressure Kutta, loads, constrained rank and nonlinear termination separately.
- panel/p1_te_jump_integral.py checks full-edge compatibility mu_upper-mu_lower=Gamma on the common refinement of wall TE stations and Gamma material knots, including exact integral, L2/max mismatch and weak moments. This is not pressure Kutta.
- tools/issue71/galerkin_ab.py runs collocation versus Galerkin on identical generated G1/G2 geometries and both Kutta variants, recording Git HEAD, Python/NumPy/SciPy, thread environment, CL/CLgamma/CD/CM, wall residuals, closure residuals, state differences, TE-jump audit and nonlinear status into one JSON receipt.
- test/test_p1_galerkin_solver.py checks end-to-end solve, exact J/K closure, pressure Jacobian against finite differences and nonlinear termination. test/test_p1_te_jump_integral.py checks knot alignment, interior-knot mismatch, integral cancellation, weak moments and G1/G2 adapters.

## Full local acceptance commands

First run the full related suite:

    python -m pytest test/test_p1_edge_singularity.py test/test_p1_galerkin_wall.py test/test_p1_galerkin_solver.py test/test_p1_te_jump_integral.py test/test_p1_lifting_g1.py test/test_p1_lifting_g2.py -q

Then execute the reproducible A/B sweep:

    python tools/issue71/galerkin_ab.py --chord-panels 4 6 8 --span-panels 4 --methods signed_speed pressure --order 3 --max-nfev 80 --out runs/issue71_galerkin_ab_r1.json

For a coupled multi-surface case add --g2. Preserve the JSON even when a pressure solve reaches its iteration cap; such a row is labelled as not converged and must not be included as a valid solution.

## What a successful run does and does not establish

The test suite establishes assembly identities and code-path consistency. The A/B report establishes what this discretization does on generated meshes. Neither alone establishes physical accuracy. Acceptance still requires quadrature-order stability, A_G @ ones = -M @ ones, the smooth-sphere manufactured control, fixed-physical-point velocity convergence, and stable loads/Gamma under mesh refinement. The edge-enrichment basis is deliberately not yet coupled into this solver; add it only after the Galerkin baseline passes these gates, to avoid confounding basis and operator changes.