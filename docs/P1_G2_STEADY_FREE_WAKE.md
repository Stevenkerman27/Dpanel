# Continuous-P1 G2: steady free-wake relaxation (research)

This is a **running end-to-end STEADY** material-wake relaxation loop, layered
on top of PR #60's coupled lifting main wing and raised tail. It does not
introduce an unsteady simulation, Kelvin-history release, time integration,
quad/hex meshes, viscosity, or a VSPAERO calibration target. Original P0
production is untouched. The local real-CAD 396-half-face fixture is not in
this sandbox; the actual executed cases below use the generated two-wing G2.

## State and equations

At wake geometry `X_w` each iteration solves **both** wing and tail in one
system, not a post-processing wake deflection:

```
wall BIE:  A_wet(mu) + W_main(X_main) Gamma_main
                    + W_tail(X_tail) Gamma_tail = -phi_infinity
TE jump:   mu_up - mu_down = H_TE Gamma, exactly constrained
Kutta:     equal signed TE tangent speed (linear), or Cp upper=lower
           (nonlinear wall pressure, analytic Jacobian)
```

For each P1 wake face, at its centroid `x`, evaluate complete velocity on the
`+n` and `-n` sides using the **same analytic P1 doublet potential** as the
wall/wake BIE:

```
phi(x) = U_inf . x + D_wall(x) mu
          + sum_g D_wake_g(x, X_g) Gamma_g(material_coordinate)
v_mean(x) = (grad phi(x + h n) + grad phi(x - h n))/2
```

Each gradient uses second-order centered spatial differences at points with
`h=0.055 sqrt(panel area)` and `delta=0.009 sqrt(panel area)` by default.
The two-sided mean approximates the principal-value vortex-sheet velocity;
finite probe distance is a **numerical small-scale regularization**, not a
physical vortex-core radius. The normal-slip and pressure jumps are recorded
explicitly: the discrete wake is **not** presumed dynamically equilibrated.

Area-weight face-centroid velocities onto existing wake vertices, and march
material streamlines downstream on the fixed ORIGINAL x sections:

```
y_{j+1} = y_j + delta_x * mean_over_edge(V_y / V_x)
z_{j+1} = z_j + delta_x * mean_over_edge(V_z / V_x)
X_w^{next} = (1-omega) X_w + omega X_w^{streamline}, omega=0.25
```

The TE/root wake nodes are frozen; all other vertices move in y/z, while
original x-grid and **original wake connectivity are preserved**. Wake
predecessor connectivity comes from triangle edges and the original (frozen)
material-y coordinate. The P1 `gamma_hat` mapping is **never recomputed from
the deformed global y**: each material vortex filament retains the appropriate
spanwise Gamma label. If local V_x falls below the forward-flow threshold,
the x-marching method stops instead of pretending it handles recirculation.

**Efficiency:** cache the full wet-wall A, geometry gradients, TE jump/Kutta
matrices and material Gamma mapping. Only reassemble the wake blocks W as
wake positions change. At identical geometry the reassembled W agrees with
original G2 to 1.1e-16 absolute. Every new W triggers a new mu/Gamma solve.

## Actual sandbox computation (both wakes move)

Machine: Python 3.13.5, NumPy 2.3.5, SciPy 1.17, x64 CPU;
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
Generated research fixture from existing PR #60: 192 wet-wall triangles,
108 shared wall potential DOFs, main wing 5 Gamma DOFs and raised tail 3 Gamma
DOFs; main wake **72** triangles and tail wake **48** triangles, six
streamwise panels for each. Reference area is the G2 demonstration convention
`Sref = 4.99` (not an engineering aircraft Sref).

|alpha|method|iteration 0 CL_p|last CL_p|initial CL_gamma|last CL_gamma|status|
|--:|--|--:|--:|--:|--:|--|
|0°|pressure Kutta|0.083623|0.083258|0.092742|0.092481|geometry stable, normal-slip threshold met|
|5°|pressure Kutta|0.365472|0.365192|0.342333|0.341845|geometry stable, normal-slip **not** converged|
|5°|signed-speed Kutta (5 iterations)|0.373501|0.373267|0.350698|0.350309|short exploratory run, not converged|

|alpha|wake|initial normal slip RMS / U|last normal slip RMS / U|initial wake Cp jump RMS|last wake Cp jump RMS|
|--:|--|--:|--:|--:|--:|
|0°|main|0.028276|0.009915|0.004838|0.004800|
|0°|tail|0.013577|0.004909|0.002496|0.002491|
|5°|main|0.057137|0.026794|0.015190|0.015973|
|5°|tail|0.018557|0.005168|0.002068|0.001598|

At 5° after the 18-iteration budget, the **main wake tip** rises by as much
as ~0.49 chord lengths at the far downstream end; the tail wake moves by up
to ~0.08 chord. Thus both wake surfaces genuinely deform in 3D. The total CL
changes little (0.365472→0.365192) in this PARTICULAR geometry; free wake
need not change CL monotonically or by a large amount. It can strongly affect
local downwash, pressure, moment, and wake-root/tip conditions in other layouts.

**Strict interpretation:** "geometry stable" is not equivalent to an exact
steady flow surface. At 5° the wake normal slip is still 2.68% U on the main
sheet, and its inferred Cp jump slightly worsens. The present algorithm is a
**steady free-wake research prototype with kinematic relaxation**; it has not
yet solved the complete pressure-continuity (dynamic sheet equilibrium) and
roll-up singularity problem. Negative coarse-mesh pressure CD remains.

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_steady_wake \
    --alpha 0 5 --iterations 18 --relaxation 0.25 --wake-panels 6 \
    --kutta pressure --out runs/g2_steady_free_wake_NEW

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_steady_wake \
    --alpha 5 --iterations 5 --wake-panels 6 --kutta signed_speed \
    --out runs/g2_steady_free_wake_linear_NEW
```

Optional existing DPanel **actual half-CAD** (adapted using corrected TE
`upper_te_faces/lower_te_faces` from PR #59's follow-up 
`research/p1-g1-te-pairing-fix@4051ee8`):

```bash
python -m panel.run_p1_steady_wake \
    --main-half-surface runs/half_cad_wingbody_v1/surface.json \
    --alpha 5 --iterations 12 --reference-area 4.0 \
    --out runs/g2_real_wingbody_steady_wake_NEW
```

That real CAD entry has **NOT** been executed in this sandbox because the
private local `surface.json` isn't mounted here; only the generated two-wing
cases above were run. CAD's wake topology may require additional generalized
streamwise material pairing if the wake is strongly swept, irregular, or has
multiple distinct sheets within one LiftingGroup.

`report.json` contains every iteration's loads and both sheet conditions;
`wet_wall.npz` contains fixed triangles; and each
`alpha_*/iteration_NNN.npz` contains full `mu`, all `Gamma`, `Cp`, both
wake geometries, connectivity and **unchanged material Gamma bases**. No local
agent test campaign is requested: these outputs already exist in sandbox.

## Deliberate next implementation seam

- Replace the finite two-sided probe with a more precise P1 / Biot–Savart
  principal-value velocity (including sheet-strength gradient) near roll-up.
- Couple kinematic sheet tangency with dynamic equal-pressure across the wake;
  then solve steady geometry plus Gamma equilibrium to both tolerances.
- Generalize wake material predecessor topology for arbitrary multi-shed OCC
  half models; retain the same `LiftingGroup`/`MultiLiftingMesh` interface.
- Make the final steady state implicitly differentiable in JAX; do **not**
  reverse through a large arbitrary number of relaxation steps.

This is exploratory steady research, not an engineering-validated unsteady or
viscous solver. `physical_accuracy_accepted=false` remains a label of evidence,
not a prohibition on using the implementation experimentally.
