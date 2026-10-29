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