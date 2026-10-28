# Issue #69 — P1 Dirichlet rank audit and pressure-Kutta selection

Scope: independent opt-in research from master; **does not** modify
\`p1_lifting_g2.solve_multi\`, old G1/G2 defaults, PR #68, or GRASS.
\`physical_accuracy_accepted=false\` throughout.

## Mathematical correction to the original issue

The wall equation implemented in \`panel/p1_lifting_g2.py\` is

\`\`\`
A @ mu + W @ Gamma = -centroid @ Uinf
J @ [mu, Gamma] = 0
\`\`\`

This is a **Dirichlet total-potential trace** boundary-integral equation,
not a Neumann normal-velocity collocation equation. The face-centre test
space has 792 rows on the reported real CAD, while the continuous nodal P1
trial space has fewer DOFs. The resulting residual is generally a nonzero
best-fit discretization residual. Therefore, "square the system" or
"wall_rms must be <1e-10" does not by itself establish correct physics.
A vertex collocation also requires unambiguous 3-D corner solid-angle free
terms and split TE traces; replacing centres with nodes is not a trivial fix.

The local #69 evidence reports 447 free DOFs after J and a numerical rank
359 for the weighted wall operator on those DOFs (threshold-dependent);
88 apparent weak modes remain. This is **a hypothesis about a numerical
near-nullspace**, not proof of exactly 88 physical circulation modes.

## Implemented opt-in algorithm

\`panel/p1_rank_aware_kutta.py\` introduces a strict two-stage experiment:

1. Form the existing area-weighted wall matrix \`B\`, RHS \`b\`,
   enforce the original TE jump exactly using \`x=N_J z\`.
2. Form \`E=B N_J\` and compute its full SVD; report **every** singular
   value, cutoff, inferred rank/nullity, the gap at the cutoff and
   \`||B T||/||B||\`.
3. If and ONLY if there is a clear singular gap, and the inferred weak
   mode count matches the number of pressure Kutta constraints, solve the
   minimum-norm wall LS optimum \`x_wall\`.
4. With \`T=N_J V_weak\`, solve
   \`min_t ||pressure_Kutta(x_wall+T t)||²\` using the *original*
   upper/lower P1 gradient speed-squared difference, exact Jacobian and
   a signed-speed Kutta seed.
5. Reject if the wall residual deteriorates >1%, if
   \`pressure_kutta_rms>1e-6\`, if the nonlinear solver fails, if the
   final pressure Kutta Jacobian is rank deficient, or if the spectral
   separation is ambiguous. The original face-centre wall residual,
   TE jump residual, pressure Kutta residual and CLp/CLgamma/CDp/CMy
   remain visible and are not clipped, smoothed or set to zero.

Passing this method is **only** a rank-conditional numerical pressure-Kutta
candidate, not a claim that the Dirichlet BIE is exactly satisfied or that
the free wake has converged.

## Executed sandbox evidence — exact scope

In this cloud sandbox (Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0),
an independently authored reference implementation of the same two-stage
algorithm was executed with pytest: **8 passed / exit 0**. The tests
exercise exact TE jump, overdetermined wall LS with nonzero optimum,
analytic pressure Jacobian vs centred difference, independence from
wall-equation scaling, no wall nullspace, a missing singular gap, nullity
mismatch and invalid inputs. This is **not** a full DPanel checkout.

The current repository branch also contains its own
\`test/test_p1_rank_aware_kutta.py\`. It needs full-repository execution;
cloud GitHub DNS and the prescribed proxy are unavailable from this sandbox.
No G1/G2 or real CAD post-fix result is claimed.

## Full checkout commands — local executes only

Select the PR #69 head and record the precise commit SHA and environment.
Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1.

\`\`\`bash
python -m pytest test/test_p1_rank_aware_kutta.py test/test_p1_lifting_g1.py test/test_p1_lifting_g2.py -q
python -m panel.run_p1_rank_aware_kutta --case g1 --compare-legacy-pressure --out runs/issue69_g1_r1
python -m panel.run_p1_rank_aware_kutta --case g2 --compare-legacy-pressure --out runs/issue69_g2_r1
python -m panel.run_p1_rank_aware_kutta --case g1 --main-half-surface runs/half_cad_wingbody_v1/surface.json --compare-legacy-pressure --out runs/issue69_cad_r1
\`\`\`

Only the CAD input that actually exists on the machine should be used; its
SHA is captured by the command. Retain separate evidence from pressure-Kutta
before the experiment. Report: test exit, raw stderr, numeric singular
spectrum, rank gap, inferred nullity, wall-fit RMS before/after, pressure
Kutta residual, pressure Jacobian rank, CLp−CLgamma, signed CDp and status.
For the **same frozen CAD geometry**, scan \`--wall-rcond\` at 1e-9,
1e-11, 1e-13. Changing rcond is a **sensitivity experiment**, never
a tuning pass to match VSPAERO or produce a preferred CL.

Exit 0 = rank-conditional *experimental* candidate; exit 2 = classified
numerical/physical blocker; exit 1 = execution failure. Exit 0 does NOT
indicate physical acceptance.

## Next decision

If a stable 88-mode gap exists and pressure constraints can be fulfilled
without spoiling the wall fit, integrate this constrained selection as a
separately reviewed optional solver. If the nullity/solutions vary strongly
with cutoff, or pressure Jacobian lacks rank, reject the hypothesis and
move to a correctly derived P1 **weak/Galerkin boundary-integral method**
with explicit TE trace DOFs. Do not replace the existing collocation model
with vertex collocation simply to force a square matrix.
