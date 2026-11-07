# Fixed-Wake CST Gradient

## Four-section optimization experiment

The Web gradient no longer imposes a 600-wall-panel limit. This does not
remove the dense differentiable assembly and exact Hessian's memory and
runtime costs; gradients above 600 panels have not been verified.

On Windows, myml Python, CPU with OpenBLAS/OMP/MKL threads set to 1, the
default Web root and tip sections were linearly interpolated into four
sections. The 76 variables are four chords, four twists and 17 CST
coefficients per section. Leading-edge positions and trailing-edge thickness
were fixed. The objective was a half-infinite planar-wake ideal Trefftz CDi
estimate, with fixed area and CL no lower than the initial value. This CDi
estimate is not the exact momentum drag of the finite 10c three-dimensional
wake; the production pressure CD is reported separately.

```powershell
python -m tools.grass_section_optimize --out runs/verification/cst_gradient/four_section_ideal_304_v1 --chord-panels 4 --span-panels-per-segment 3 --iterations 40
python -m tools.grass_opt_grid_check --result runs/verification/cst_gradient/four_section_ideal_304_v1/result.json --out runs/verification/cst_gradient/four_section_ideal_304_grid_v1.json
python -m tools.grass_mesh_gradient_check --surface runs/verification/cst_gradient/four_section_ideal_304_v1/final_surface.json --out runs/verification/cst_gradient/four_section_ideal_304_gradient.json --ideal-drag --four-sections
```

Paths must be new. `--case` accepts a four-section GRASS JSON input. The
304-panel run reached the 40-iteration limit rather than converging:
CL `0.358456385 -> 0.358452644`, ideal CDi `0.007509433 -> 0.006551700`,
pressure CD `0.008012325 -> 0.021689673`; CL constraint residual is
`-3.74e-6`. On 600 panels the baseline and candidate gave CL
`0.35734495 / 0.32382857`, ideal CDi `0.00828346 / 0.00620735`, and
pressure CD `0.00898502 / 0.01924745`. Thus the candidate fails the lift
constraint on the finer mesh and does not establish useful drag reduction.
At the 304-panel candidate, centered differences with step `1e-5` for each
section's chord, twist and first CST coefficient gave maximum absolute errors
`7.87e-8`, `1.00e-8`, and `2.78e-9` for CL, pressure CD, and ideal CDi.
This checks discrete derivatives, not aerodynamic accuracy.
An earlier 156-panel pressure-CD optimization produced negative pressure CD;
the 156-panel ideal-CDi optimization reached its iteration limit and its
600-panel reevaluation increased ideal CDi from `0.00828346` to `0.00865372`.
These are failed or unfinished numerical experiments, not physical accuracy
or grid-convergence acceptance. Results and histories are under
`runs/verification/cst_gradient/four_section_*`.

This is a small-grid differentiability experiment, not a CAD-mesh gradient or
physical accuracy acceptance. `panel.p1_cst_gradient.fixed_wake_cst` accepts the GRASS
18-value direct Kulfan code and returns torch CL and pressure CD with a
gradient. It uses the sibling `GRASS-plane/cst_airfoil_codec.py` directly.
The previous GRASS wing builder imports `panel3d_geometry`, which is absent
from the current sibling checkout; this experiment does not use that builder.

The 8 upper weights, 8 lower weights and leading-edge modification are
independent. The last code entry (trailing-edge thickness) must be zero and
is held fixed: the P1 topology requires a shared sharp trailing edge. The
same section is applied across a fixed rectangular wing; span, planform,
topology, wake and reference values are fixed. This does not yet expose
root/tip CST distributions, other GRASS variables, CAD/Gmsh or free wakes.

`fixed_wake_sections` extends the research path to the actual GRASS
`leading_edge_xyz`, `chord`, `twist_radians`, and per-section `cst` tensors,
using GRASS's `wing_section_points`. It creates a fixed-topology symmetric
thick wing and x-extruded 10c fixed wake. Wake nodes and the reference area
remain in the computation graph. The root position is constrained to zero,
each section's trailing-edge thickness is constrained to zero, and the
reference area is twice the trapezoidal integral of chord over the input
section y coordinates. For a GRASS wing without a body, Web now saves this
same full-wing mesh and the production P1 solver reads it directly. The
gradient reconstructs the graph at its saved resolution and checks its nodes,
connectivity and wake against that mesh. STEP is a separate B-spline
approximation and is not used to generate this fixed-topology mesh. A two-section case
with sweep, dihedral, taper and twist matches the production P1 forward
solve; six representative variables (tip x/y/z, chord, twist and one CST
coefficient, plus root chord and twist) agree with centered differences in
the regression test. A three-section case checks the middle chord derivative.
In one full-suite perturbation run the pressure solver exhausted its iteration
budget with stationarity infinity norm 1.53e-11; the old success-flag guard
rejected it. The current guard reports that status separately and accepts the
implicit derivative only after the independent stationary/Hessian checks above.

The production Galerkin source kernel and target quadrature are reproduced in
torch, including the geometry-dependent mass matrix and wall/wake operators.
The pressure-Kutta forward solve uses the existing structured solver. The
backward pass differentiates its exact stationary equation, including the
pressure residual's second-order term; it requires an infinity-norm
stationarity residual at most 1e-9 and a positive-definite exact Hessian.
It retains the forward solver's separate success/status flag: a solver
iteration-limit status can still satisfy this stricter mathematical check,
which does not constitute aerodynamic acceptance. The trailing-edge face selections and constraint
topology are fixed. The implementation is dense and intended for small
verification meshes, not large-scale performance.

`fixed_wake_surface` is a separate intermediate bridge that loads the actual
production half-surface and differentiates its wall and wake x/z coordinates
with y and the circulation basis fixed. It does not map GRASS parameters to
CAD mesh nodes, and it rejects wake contact constraints. It is not connected
to the Web gradient button. A real 98-panel half mesh produced forward CL/CD
`0.365751306574/0.0106128483026`, matching the production solve to `3e-12`;
for x/z uniform scaling and a correspondingly scaled reference area, its
derivative was `-0.1488348883/-0.00932994593` versus production centered
differences `-0.1488352074/-0.00932997468` at step `1e-3`. This one deformation
does not validate arbitrary mesh motion, changing y, or GRASS derivatives.

Windows, myml Python 3.13, CPU, OpenBLAS/OMP/MKL threads 1, rectangular wing
span 4c, alpha 4 degrees, 10c fixed wake, two streamwise wake panels,
order-3 Galerkin and pressure Kutta. All 17 independent parameters were
checked individually:

| wall grid | CL | pressure CD | FD step | maximum absolute error CL / CD |
| --- | ---: | ---: | ---: | --- |
| 3x3 | 0.20770818 | -0.00569669 | 1e-4 | 2.94e-10 / 3.91e-10 |
| 3x3 | 0.20770818 | -0.00569669 | 1e-5 | 8.74e-10 / 6.29e-11 |
| 4x4 | 0.23237660 | -0.00278479 | 1e-4 | 5.59e-10 / 1.42e-9 |
| 4x4 | 0.23237660 | -0.00278479 | 1e-5 | 1.56e-9 / 1.12e-10 |

The 3x3 forward CL/CD and assembled A/W/RHS/mass are independently checked
against `solve_multi` and `assemble_galerkin_multi` in
`test/test_p1_cst_gradient.py`. A cambered section also checks an upper weight
against centered differences. A small discrete derivative error does not
prove mesh convergence, aerodynamic accuracy, or a useful drag prediction;
in particular these coarse-grid pressure CD values are negative.

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -m tools.cst_gradient_check --out runs/verification/cst_gradient/rect_3x3.json
python -m tools.cst_gradient_check --out runs/verification/cst_gradient/rect_4x4.json --chord-panels 4 --span-panels 4
python -m pytest -q test/test_p1_cst_gradient.py -p no:cacheprovider
```

The commands require fresh output paths. JSON records all derivative and
finite-difference entries. The shared Web mesh can also be checked against
the production solver, keeping its panel counts fixed:

```powershell
python -m tools.grass_mesh_gradient_check --surface runs/web/20261106-210317-9c3bab/mesh/surface.json --out runs/verification/cst_gradient/new_check.json
```

Alternatively reproduce the 84-panel default case without a prior Web run:

```powershell
python -m tools.grass_mesh_gradient_check --default-size 0.7 --out runs/verification/cst_gradient/new_default_check.json
```

Windows/myml, CPU, single thread, alpha 4 degrees, 10c fixed wake:
for 10 root/tip CST, chord, twist and tip x/y/z entries, step `1e-5`,
maximum absolute CL/CD gradient errors against production centered differences
were `1.69e-9/1.34e-10` on the 84-panel mesh and `1.94e-7/1.48e-8`
on the 156-panel mesh. The JSON outputs are `unified_mesh_84_h1e5.json`
and `unified_mesh_156_h1e5.json` under `runs/verification/cst_gradient/`.
An automated three-section case also checks intermediate-section x/y/z,
chord, twist and CST against production centered differences. These results
do not establish mesh convergence or aerodynamic accuracy. CAD/Gmsh meshes,
body intersections and free-wake gradients are not covered. There is no
panel-count gate; large dense differentiable solves have not been validated
and can exhaust memory.
