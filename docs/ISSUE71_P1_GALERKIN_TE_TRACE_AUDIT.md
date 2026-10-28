# Issue #71: derivation of P1 Galerkin boundary integrals and TE-cut review

Date: 2026-10-29. Source baseline: `master` and latest local #70/#71 review.
This is a **read-only mathematical/topological audit**, not a new solver,
not a replacement of P1 G1/G2 or free-wake #68. Physical acceptance remains false.

## 1. Correct the local evidence before designing a new formulation

Local PR #70 feedback confirmed the original #69 wall equation is
**Dirichlet potential trace**, not Neumann flux. On the CAD
`B @ null(J)` is full column rank, with condition number approximately 243
and maximum adjacent singular-value ratio only 1.46. There are **no** 88
separable weak circulation modes; the local `blocked_no_wall_nullspace`
result is a valid negative result for #70, not a solver bug.

The local suspicion that upper/lower **interior chord** nodes were artificially
duplicated was explicitly retracted: those points are physically distinct
coordinates `chord*(zcam +/- thickness)`. The local trial of merging them
destroyed the body geometry. Do NOT repeat that fix.

The remaining unresolved observation is physical-grid nonconvergence:
generated G1 chord panels 4/6/8, span=8, camber=.02 had CL
0.457263/0.388562/0.244532, while the raw wall potential LS residual
decreased 2.14e-3/1.78e-3/1.68e-3. For a near-zero-thickness symmetric
wing, the c8 lift even became negative at positive incidence. Both
signed-speed and pressure Kutta show the chordwise trend; rank and Kutta
alone do not establish an explanation. See Issue #71 for exact local receipts.

## 2. Start with the SAME physical representation and sign

Let `S` be the physical closed wetted surface, `W` the attached wake cut,
`G(x,y)=1/(4*pi*|x-y|)`, with normals and doublet signs consistent with
`p1_lifting_g1._doublet_matrix`. Write the total potential as

    Phi(x) = Phi_inf(x) + D_S(mu)(x) + D_W(Gamma)(x),

where `D_S` and `D_W` use the same P1 double-layer kernel. The existing
face-centre P1 equation takes the **interior body limit**

    (-1/2 I + K_S) mu + K_W Gamma = -Phi_inf          (1)

on interiors of smooth flat wall triangles. `K_S` is the finite-part/average
double-layer boundary operator, and `K_W` is the off-wake potential
influence evaluated on the wall. The sign of the jump must be checked
against `D=-Omega/(4*pi)` rather than imported from another textbook
convention. The source-free pure doublet model is a *model assumption*;
Galerkin alone does not validate it.

Let the actual P1 trial functions be `N_j` with **two independent
DOF traces only at a true sharp TE cut**. Thus

    mu_h = sum_j mu_j * N_j   on S, with broken TE trace,
    Gamma_h = sum_g Gamma_g * H_g   on W, in frozen material coordinates.

## 3. True Galerkin of (1), not row reweighting or vertex collocation

Use the **same P1 functions** `N_i` as test functions, with the
trace space broken only at declared TE physical edges. Require for each i

    int_S N_i(x) [
        (-1/2 mu_h + K_S mu_h)(x)
        + K_W Gamma_h(x) + Phi_inf(x)
    ] dS_x = 0.

The exact matrices are

    M_ij = int_S N_i(x) N_j(x) dS_x
    A^G_ij = -1/2 M_ij
           + int_S int_S N_i(x) k_D(x,y) N_j(y) dS_y dS_x
    W^G_ig = int_S int_W N_i(x) k_D(x,y) H_g(y) dS_y dS_x
    b^G_i = -int_S N_i(x) Phi_inf(x) dS_x.

For any planar triangle of area A, the exact local P1 mass matrix is

                    [ 2 1 1 ]
    M_T = (A/12) *  [ 1 2 1 ]
                    [ 1 1 2 ]

where the local DOF scatter **must use** `dof_face`, not just geometric
vertex IDs. A manufactured quadratic integral with three degree-2
triangle points recovers this matrix to roundoff. The self jump is
`-1/2 M_T`, NOT simply a diagonal `-1/2`. For an interior target point
on a planar source face, its own principal-value `k_D` contribution
vanishes; the nonzero self contribution is the exact jump distribution.

In the EXISTING collocation kernel, the exact own-face trace is
`-.5*B`, where `B=1/3+(x-centroid)@shape_grad` **is the full P1
barycentric shape vector**. It is NOT merely a constant P0 approximation.
At a centroid, it reduces to `[-1/6,-1/6,-1/6]`; at another strict
interior point all three barycentric fractions are retained. The local
hypothesis that the code discards the linear part of its own-face
`-.5*B` term is therefore contradicted by static code inspection.
The remaining source shape-moment / cross-element coefficients still need
an independent P1 oracle and mesh-refinement test.

For an initial quadrature-Galerkin implementation:
- integrate the OUTER wall-test variable `x` on each triangle with
  degree-2, then higher-order and near-edge adaptive quadrature;
- for each target `x`, use the existing analytic P1
  `_doublet_matrix` to integrate `y` over the source face/wake;
- replace a source face's OWN `D^-` contribution by the known analytic
  local jump, integrated exactly as `-1/2 M_T`;
- use explicit same-face/shared-edge/adjacent-near-edge panels and
  compare 3/7/13 or adaptive outer quadrature; never claim that a
  three-point rule alone resolves the near-singular wall/wake interactions.

The resulting weak matrix has one wall TEST row per wall P1 DOF, not one
row per face. **But** Gamma, jump, and Kutta add further unknowns/equations.
A square wall matrix does not automatically make the FULL lifting system
square or ensure physical lift convergence.

## 4. TE traces: current implementation versus genuine edge constraints

A sharp attached wake is a branch cut. The potentials `mu^+` and `mu^-`
may differ on the SAME geometric TE point; the jump must equal the wake
strength, according to the established code sign convention:

    mu_h^+(s) - mu_h^-(s) = Gamma_h(s) on the physical TE edge.

Current `J=[C,-H]` enforces this at selected TE nodes only.
It does NOT enforce pressure Kutta or prove equality along every TE edge.
A weak/mortar-compatible TE condition, if meshes are not matching, is

    int_TE q_a(s) [mu_h^+(s)-mu_h^-(s)-Gamma_h(s)] ds = 0.

With matching piecewise-affine TE and Gamma material knot meshes, nodal
equality yields full edge equality; otherwise it can fail BETWEEN nodes.
If a `gamma_y` knot lies strictly inside an unsplit geometric TE edge,
`Gamma_h` can kink there while `mu^+-mu^-` remains affine. The audit
reports this as `te_edge_crosses_gamma_knot`; one remedy is a compatible
TE mesh subdivision or edge-integral mortar projection, not more Kutta
iterations.

**Source audit — generated geometry:** `make_wing` uses distinct
upper/lower geometry at interior chord stations, common LE/TE geometry,
and duplicates only the lower POTENTIAL DOFs at internal-span TE nodes.
Tip DOFs remain unsplit and material Gamma is zero at tips. This
construction is mathematically coherent as a sharp-TE representation;
it does not prove the complete lift solver is consistent.

**Source audit — real CAD adapter:** `from_dpanel_half_surface`
does identify TE from `upper_te_faces/lower_te_faces`, but:
1. it accepts `len(intersect(face_up,face_down))>0`, even if the
   intersection is ONLY ONE VERTEX rather than a physical TE edge;
2. it loops over `zip(up_full,lo_full)`, which silently truncates if
   the two input arrays differ in length;
3. it infers bank from `normal_z` sign, not from a persistent physical
   upper/lower TE entity label. This may split unrelated contact/fuselage
   facets sharing TE vertices or miss a valid near-vertical TE bank;
4. its Kutta face search falls back to ANY face with the requested bank
   if the exact TE potential DOF is absent. That fallback can make a
   seemingly small pressure residual unrelated to the declared TE station.

These are **verified code-path risks**, NOT observed violations in the
private real CAD fixture. Do not modify the adapter or merge those
branches before a real-input audit confirms them.

## 5. Read-only executable audit

`panel/p1_te_trace_audit.py` implements:

- `audit_declared_cad_te(parsed_source_json)`: equal upper/lower array
  lengths; paired faces share EXACTLY one declared TE edge (two vertices);
  consistent opposite edge winding; no repeated TE face edge.
- `audit_te_trace(wing)`: one P1 potential DOF per geometric vertex away
  from the TE cut; valid upper/lower supports for every TE pair; no
  extra cut crossing an ordinary wall edge; no branched TE seam; each
  seam edge matches an actual geometric wake-root edge; interpolated
  `gamma_hat` matches the TE jump hat basis; no unrepresented interior
  gamma knot across a geometric TE edge.

It returns `ok`, explicit error codes, counts and
`pressure_kutta_verified=False`. It never edits the geometry, potential,
Gamma or original G1/G2 solver.

Cloud sandbox has actually run **14 manufactured/negative tests** in
`/mnt/data/dpanel_issue71` (Python 3.13.5, NumPy 2.3.5, pytest):
`14 passed / exit 0`. The new GitHub test also includes one
**full-repository generated-G1 integration test** (NOT run in cloud)
which checks its three interior TE pairs and four attached seam edges.
Real CAD data are not in the repo, so NO real-CAD cut validation is claimed.

## 6. Separate *operator* and *physical* convergence tests

Before implementing Galerkin, preserve the same frozen geometry and
`gamma_hat`, compare analytical per-source P1 potential values with
independent, correctly Jacobian-weighted high-order triangle integral
for constant and LINEAR strengths. The local #71 constant-density
midpoint oracle already agreed to ~1e-4, limited by its quadrature.

A *manufactured operator* test may prescribe a P1 strength vector
`mu*` and Gamma* then compute a reference RHS from independent
integration; it tests code and boundary trace signs, but does NOT
establish a physical no-penetration solution. Physical convergence
requires a separately known harmonic lifting/no-lift reference and a
mesh-refinement series. In particular, prescribing an elliptical
Gamma(y) alone does NOT construct a full exact harmonic body solution.

Only then introduce a separate Galerkin research implementation with
a documented constraint strategy. Naively stacking all `N_mu`
Galerkin wall equations, `N_J` nodal TE jump rows and `N_K`
pressure-Kutta rows gives MORE equations than `N_mu+N_Gamma` unknowns
in many real cases; a mathematically justified constraint selection,
mixed formulation, or constrained least-squares problem is still needed.
Do not re-label a reweighted collocation LS or a forced square vertex
system "Galerkin".

Exit: the correct TE topology and analytic self jump are prerequisites;
they do not solve the observed G1 chordwise CL collapse. Keep
`physical_accuracy_accepted=false`, PR #68/#70 experimental, and
production defaults unchanged.
