# Continuous P1 G2 — coupled main wing and elevated horizontal tail

Research implementation based on P1 G1 (`research/p1-g1-te-pairing-fix@4051ee8`).
**This is an actual second independent lifting surface and wake**, not a sum of
separately computed wing loads. Old `panel/kernel.py`, P0 production solver,
existing G1 source and all CAD inputs are unchanged.

## Method / implementation

`panel/p1_lifting_g2.py` builds one shared P1 doublet boundary-integral
system for every wetted triangle of all lifting components:

    [D_wall  W_main  W_tail] [mu; gamma_main; gamma_tail] = -phi_inf

There is one continuous wall doublet potential field on each connected lifting
component, plus independent TE trace DOF splits for each declared physical
wake. Coupling is real because `D_wall` includes cross-component influences and
both `W_main` and `W_tail` act on the walls of every component.

Each lifting group has its own span stations, P1 Gamma hat basis, triangulated
physical wake, TE trace jumps, tip Gamma=0, and an independent Kutta condition:

    Jump_g: mu_up_g - mu_down_g = H_g gamma_g
    Kutta_g: signed downstream tangential velocity equality (linear), OR
             Cp_upper - Cp_lower = 0 (nonlinear, analytic Jacobian)

The TE jump constraint is **enforced exactly** by null-space parameterisation;
it is not a penalty residual. The linear Kutta mode is also a null-space exact
constraint. In the pressure Kutta mode the Kutta condition is a nonlinear
least-squares residual, with an explicit independent loss from the wall BIE.

Reference conventions: velocity magnitude 1, inviscid, uniform alpha rotation
about y, no beta, force coefficients from triangle face Cp and normals.
`Sref` defaults to the **sum of main and tail reference planforms for the
self-contained demonstration only**. For a real aircraft's published
main-wing Sref use `--reference-area 4.0` (or the correct actual value) so
that CL/CM comparisons have common normalization.

## Run directly (no CAD/Gmsh needed)

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_lifting_g2 \
      --alpha 0 5 --kutta pressure --out runs/g2_generated_new

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_lifting_g2 \
      --alpha 5 --kutta signed_speed --out runs/g2_linear_new

**Optional actual DPanel half wing/body CAD:** in a worktree based on the
corrected G1 `research/p1-g1-te-pairing-fix` branch, whose adapter uses
`upper_te_faces`/`lower_te_faces` and not the TE edge polyline as face banks:

    python -m panel.run_p1_lifting_g2 \
      --main-half-surface runs/half_cad_wingbody_v1/surface.json \
      --tail-span 1.8 --tail-chord .55 --tail-x 1.6 --tail-z .35 \
      --reference-area 4.0 --alpha 0 5 --kutta pressure \
      --out runs/g2_real_wingbody_new

The CAD input remains byte-for-byte unchanged; it is mirrored to a full
physical wall, and a generated **independent tail** is added at the specified
position. This is research geometry, not a certified aircraft CAD assembly.
There is no need for new Gmsh/OCC mesh generation for the original main wing.

`report.json` contains per-group CL_p, CL_gamma, Gamma arrays, solve and Kutta
residuals. `mesh.npz` and each `alpha_*.npz` contain all actual coordinates,
faces, wet-wall P1 nodal solution, per-group wake geometry, Gamma, and Cp.

## Actual sandbox computations

Machine: Python 3.13.5, NumPy 2.3.5, SciPy 1.17 CPU x64.
Fixture: cambered main (span 4, chord 1, 4x6 chord-span panels),
symmetric raised tail (span 1.8, chord 0.55, 4x4 panels) at
`(x,y,z)=(1.6,0,0.35)`, 2-streamwise-panel wake per group.
192 true wet-wall triangular panels; combined P1 doublet influence matrix
192x108, two wake blocks with 5 and 3 independent gamma DOFs (8 total).
The two wake sheets lie at different z, so they are not geometrically one sheet.
The demonstration common Sref is 4.99 (=4 + 0.99).

|alpha|Kutta|CL_p total|CL_gamma total|CL_p main|CL_p tail|wall RMS|jump RMS|
|---:|---|---:|---:|---:|---:|---:|---:|
|0°|pressure|0.083623|0.092742|0.098794|-0.015171|0.001734|7.9e-17|
|5°|pressure|0.365472|0.342333|0.360551|+0.004921|0.001918|1.4e-16|
|5°|signed speed|0.373501|0.350698|0.3698|+0.0037|—|<1e-12|

**Measured interference, not superposition:** solve the two wings separately
at 5°, normalise them with the same total Sref: 
main CL_p=0.360955, tail CL_p=0.056555, sum 0.417510.
The **coupled** CL_p is 0.365472; actual body/wake influence reduces the
tail load to 0.004921. The components have separate Gamma vectors and the
other component sees nonzero influence from each wake block. This is an
example of coupled downwash, not a validated real-aircraft prediction.

Four focused research tests, including single-group G1 reproduction (to
machine precision), G2 off-diagonal wake coupling, actual nonlinear Kutta
solve and fixed-geometry alpha sensitivity vs finite difference, passed.

## Known numerical/physics limits (not work blockers)

- Both pressure CL and Gamma CL remain different (G2 5°: 0.36547 vs 0.34233).
- Coarse near-field pressure CD can be negative; cannot represent real total CD.
- All present wakes are fixed planar; tail wake only samples prescribed
  placement and not free wake roll-up.
- Need eventual JAX geometry VJP, scalable solve, and generalized independent
  G2 real CAD shed group input. This implementation is NumPy/SciPy CPU.
- The real 396-half-wing CAD was unavailable in the sandbox. The corrected
  `from_dpanel_half_surface` adapter is wired, but this particular full CAD
  + generated tail scenario was **not run** in the sandbox.
- VSPAERO is not a truth oracle. Actual physical acceptance remains false;
  research experiments should continue rather than waiting for a gate sweep.
