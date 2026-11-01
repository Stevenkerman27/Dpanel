# Issue 92 Phase 0: production baseline and solver crossover

Status: measured on 2026-11-01; branch worktree D:/3D/Projects/panel/runs/fix_master;
physical_accuracy_accepted = false (this document makes no accuracy claim).

Phase 0 of issue 92 is a measurement unit, not a port. Its job is to answer three
questions with numbers before any PyTorch code is written:

1. What does the production NumPy/SciPy path actually spend its time on?
2. At which unknown count does a dense LU stop being the right kernel, per backend?
3. Which stage should the native backend reproduce first, and with which tolerance?

The answers below are reproducible from the three scripts in tools/issue92 and the
frozen fixtures in docs/issue92/fixtures.

## 1. Method

* Eight geometry-only fixtures (docs/issue92/fixtures/*.npz) are built once and then
  frozen by SHA256: each array is hashed in C order and the sorted (name, hash) list is
  hashed again into one digest recorded in docs/issue92/fixtures_manifest.json. Every
  measurement re-verifies that digest before it runs, so the claim that a number belongs
  to a given geometry is checked rather than assumed.
* Fixtures carry topology and reference data only. They do not store solver settings,
  so the same file can be replayed by NumPy/SciPy, PyTorch CPU and CUDA.
* Timing is cProfile on the production entry points (panel.p1_lifting_g1.assemble and
  solve, panel.p1_lifting_g2.assemble_multi and solve_multi, and
  panel.p1_coupled_material_wake.solve_coupled_material_wake) plus wall clock.
* The solver crossover runs each backend family in its own process. This is not
  cosmetic: mixing SciPy/OpenBLAS with a CUDA torch build in one process produced a
  silent Windows fatal exception: access violation (exit -1073741819) with no Python
  traceback at n=385. Isolated processes are stable, and a crash can then only lose one
  column instead of the whole table.

## 2. Environment

| item | value |
| --- | --- |
| Python | 3.11.17 (.conda_env) |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| JAX | 0.10.2 (installed, unused by Phase 0) |
| CPU | AMD64 Family 26 Model 68 Stepping 0, 32 logical cores |
| OS | Windows 10.0.26200 |
| torch (CPU parity) | 2.15.0+cpu, loaded from runs/torch_site (not inside .conda_env) |
| GPU | NVIDIA GeForce RTX 5060, 8.52 GB |
| CUDA torch | 2.13.0+cu132, CUDA 13.2, FP64 supported |

Note the split: the production interpreter has no torch, and the CUDA interpreter (myml)
has no JAX. Phase 1 therefore has to be importable from .conda_env through an explicit
site directory, exactly like runs/torch_site, or Phase 1 must run in its own process.

## 3. Fixture battery

Unknowns are mu dofs + gamma dofs. A is the wall (Dirichlet trace) block only; the
Kutta rows are imposed as constraints on the null space, not appended to A, so the
matrix in the table is (faces x mu) and the solved system is
[faces + kutta_rows] x [mu + gamma] in the constrained formulation.

| fixture | wall faces | wall nodes | mu | gamma | kutta rows | unknown | A (MB) | A nnz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1_wing_rect | 112 | 58 | 63 | 5 | 5 | 68 | 0.054 | 1.0 |
| g1_wing_camber | 112 | 58 | 63 | 5 | 5 | 68 | 0.054 | 1.0 |
| g2_conventional | 192 | 108 | 108 | 8 | 8 | 116 | 0.158 | 1.0 |
| g2_tail_low | 192 | 108 | 108 | 8 | 8 | 116 | 0.158 | 1.0 |
| g1_wingbody_cad | 792 | 765 | 497 | 44 | 88 | 541 | 3.003 | 0.99998 |
| g1_wingbody_cad_3x | 2410 | 1925 | 1308 | 44 | 88 | 1352 | 24.05 | 0.99994 |
| g1_swept30 | 2764 | 1928 | 1447 | 15 | 63 | 1462 | 30.51 | 0.99997 |
| g1_elliptic | 3220 | 2368 | 1695 | 33 | 83 | 1728 | 41.64 | 1.0 |

The influence matrices are completely dense (nnz fraction 0.99994-1.0) even for the
CAD bodies. Sparsity is not available; the only cheap representation is a matrix-free
or blockwise kernel, and per issue #16 a matrix-free matvec currently costs 1.6 s, so
it saves memory and not time.

## 4. Wall clock of the production path

alpha = 5 deg, kutta = pressure. The solve column is solve()/solve_multi() end to end,
and that function re-assembles internally, so the solve column contains a second
assembly (the linear algebra alone is quantified in section 5).

| fixture | faces | assemble (s) | solve (s) | CL |
| --- | --- | --- | --- | --- |
| g1_wing_rect | 112 | 0.029 | 0.037 | 0.319473 |
| g1_wing_camber | 112 | 0.029 | 0.037 | 0.450291 |
| g2_conventional | 192 | 0.053 | 0.076 | 0.365472 |
| g2_tail_low | 192 | 0.055 | 0.075 | 0.362833 |
| g1_wingbody_cad | 792 | 0.775 | 1.929 | 0.536112 |
| g1_wingbody_cad_3x | 2410 | 4.745 | 12.694 | 0.553572 |
| g1_swept30 | 2764 | 5.570 | 10.309 | 0.467421 |
| g1_elliptic | 3220 | 7.700 | 19.287 | 0.347274 |

Assembly grows as roughly N^1.65 over 112 to 3220 faces (0.029 s to 7.70 s, 265x for
28.8x faces). Peak RSS reaches 1116 MB for the CAD battery, which is dominated by the
dense A blocks plus the least-squares workspace, not by the fixtures (306 KB total).

## 5. Where the time actually goes

cProfile cumulative seconds for the elliptic fixture (3220 faces, 1728 unknowns):

| function | calls | cum (s) | share of 26.99 s |
| --- | --- | --- | --- |
| _doublet_matrix (p1_lifting_g1.py:369) | 1529 | 12.21 | 45% |
| scipy svd | 13 | 9.54 | 35% |
| scipy lstsq (gelsd) | 1 | 0.51 | 2% |
| scipy null_space | 2 | 0.18 | 1% |

The SVD attribution is the important one. Splitting by Kutta branch on the same
geometry separates the two contributions:

| kutta | total (s) | least_squares cum (s) | svd inside least_squares | lstsq | null_space |
| --- | --- | --- | --- | --- | --- |
| signed_speed | 13.01 | 0 | 0 | 0.47 | 0.10 |
| pressure | 22.89 | 9.84 | 9.26 (13 calls, 0.71 each) | 0.47 | 0.18 |

So for the pressure-Kutta production path:

* influence assembly is about half the wall clock (0.6-0.8 ms per wake triangle);
* scipy.optimize.least_squares spends 9.3 s in dense SVD of the Jacobian, i.e. 43% of
  the run, at every accepted step of the nonlinear Kutta solve;
* the actual forward solve (one gelsd on a 3220 x 1728 dense system) is 2%.

Consequently the dense linear solve is NOT the bottleneck at production sizes. A dense
LU of the same size (0.064 s at n=1537, 0.41 s at n=6145, see section 7) is 0.3% of a
19 s run. Porting a solver first would optimise the wrong 2%.

## 6. Free-wake (B2) stage

One coupled material-wake iteration on the CAD wing-body case, gate_mode = off,
velocity_method = analytic_edges, order 48:

| quantity | value |
| --- | --- |
| wall faces / wake nodes / wake triangles | 872 / 785 / 794 |
| unknowns | 589 |
| assemble_multi | 0.83 s |
| initial solve_multi | 6.55 s |
| one coupled iteration (sample + update) | 57.73 s |
| batched velocity evaluations | 32 calls, 18992 points (593 points per call) |
| status after one update | blocked_updated_nonlinear_solve |
| map_residual_chord | 22.05 (initial) to 52.01 (after one update) |

That is 3.0 ms per wake-point evaluation end to end, 30x the cost of the wall solve
per unknown. The free wake, not the wall system, is where Phase 2 has to spend its
acceleration budget, and the blocked status after a single update is the streamline
target saturation already documented in docs/ISSUE84_STREAMLINE_STEP_UNIT.md, not a
new defect.

## 7. Dense solver crossover

Synthetic dense matrix (n x n, diagonally dominant, seed 20261101), right-hand side
b = A @ ones, residual verified after every timing. Warm medians in seconds; n is the
number of unknowns.

| n | scipy LU | torch CPU LU | torch CUDA LU | dense GMRES |
| --- | --- | --- | --- | --- |
| 96 | 0.0000 | 0.0000 | 0.0002 | 0.0260 |
| 192 | 0.0004 | 0.0002 | 0.0005 | 0.0019 |
| 385 | 0.0122 | 0.0005 | 0.0014 | 0.0024 |
| 769 | 0.0449 | 0.0017 | 0.0048 | 0.0063 |
| 1537 | 0.0636 | 0.0065 | 0.0175 | 0.0074 |
| 3073 | 0.1217 | 0.0392 | 0.0907 | 0.0259 |
| 6145 | 0.4114 | 0.2487 | 0.6146 | 0.1443 |

All residuals are at or below 2.24e-15. Verdicts:

* torch CPU LU overtakes scipy LU at n = 192 (2.0x) and stays ahead (1.65x at 6145);
* CUDA LU overtakes scipy LU only in n = 385 to 3073 (8.7x at 385, 1.34x at 3073) and
  is 1.49x slower than scipy LU at 6145;
* CUDA LU never beats torch CPU LU in this range (2.5x slower at 6145): with 32 cores,
  a single dense FP64 LU of a 6145 system is a poor GPU fit, because the 0.25 s CPU
  time is already below the launch and synchronisation granularity;
* dense GMRES is the fastest kernel from n = 1537 upward (8.6x faster than scipy LU at
  6145) but its 29 matvecs are dense BLAS-2/3 products; the production matrix-free
  matvec is 1.6 s, so this column does not transfer to the AIC without an
  assembly-free kernel.

The crossover is therefore backend- and size-dependent, and the production sizes
(541-1728 unknowns) sit exactly where the answer is torch CPU, not GPU. This is the
reason issue 92 sequences CPU parity before partial CUDA: the CUDA win only exists for
the assembly and the induced velocity, which are the actual hot spots.

## 8. What Phase 1 has to reproduce, and how it is checked

The fingerprint below is the parity target for each fixture (recorded in
docs/issue92/phase0_profile.json). Example for g1_elliptic:

| quantity | value |
| --- | --- |
| CL | 0.3472736155437683 |
| CL_gamma | 0.3400753819393768 |
| CD_pressure | 0.0049227372732021 |
| CM_y | -0.07531996207332511 |
| wall_rms | 2.970747257516866e-04 |
| pressure_kutta_rms | 7.64093243994016e-07 |
| signed_kutta_rms | 0.008780662154406798 |
| jump_rms | 3.71418561938156e-16 |
| constrained_rank | 1562 |

Phase 1 acceptance should be stated against these numbers with an explicit tolerance
per quantity (and gradcheck / directional VJP for the autograd path), not against a
re-run of the SciPy path in the same process.

## 9. Consequences for the next units

1. Do not port a solver first. Port the influence assembly and the constrained least
   squares reduction, which together are 80-98% of the run.
2. The single largest algorithmic win is independent of torch: the pressure-Kutta
   least_squares call spends 9.3 s of 22.9 s in dense SVD of its Jacobian. Replacing
   that SVD (structured or reduced variable or iterative Levenberg-Marquardt) is worth
   more than any backend swap at 3220 faces.
3. Assembly is O(N^1.65) in the current Python-loop kernel; it must become batched
   (block over wake triangles, like the PR #76 block=512 change, but on the influence
   side) before it can be judged against torch.
4. The free wake is 30x more expensive per unknown than the wall solve, so Phase 2
   should put CUDA on the induced-velocity and geometry-update kernels, not on LU.
5. Selecting a backend by size must be explicit: below roughly 1500 unknowns torch CPU
   is the best dense-LU host on this machine, CUDA only wins in a narrow band, and
   neither matters until the assembly is fixed.

## 10. Reproduction

    cd D:/3D/Projects/panel/runs/fix_master
    set PYTHONPATH=D:/3D/Projects/panel/runs/fix_master
    set PANEL_ROOT=D:/3D/Projects/panel/runs/fix_master

    python -m tools.issue92.fixtures --out docs/issue92/fixtures
    python -m tools.issue92.profile_production --wake --wake-iterations 1
    python -m tools.issue92.solver_crossover --backends cpu --out docs/issue92/solver_crossover_cpu.json
    python -m tools.issue92.solver_crossover --backends torch_cpu --out docs/issue92/solver_crossover_torchcpu.json
    python -m tools.issue92.solver_crossover --backends cuda --out docs/issue92/solver_crossover_cuda.json
    python -m tools.issue92.solver_crossover --merge docs/issue92/solver_crossover_cpu.json docs/issue92/solver_crossover_torchcpu.json docs/issue92/solver_crossover_cuda.json --out docs/issue92/solver_crossover.json

The CUDA line must run in an interpreter that has a CUDA torch build (myml, with
C:/Users/zyx20/anaconda3/envs/myml/python.exe on this machine); the other lines run in
.conda_env. Do not combine them in one process (section 1).

## 11. Limitations and non-claims

* The crossover matrix is synthetic and diagonally dominant. Real AIC matrices are
  dense and non-symmetric and their LU does not have a better constant; treating the
  synthetic numbers as a performance promise would be wrong. They are used only to
  locate the crossover, which is a property of BLAS/LAPACK and the device, not of the
  panel kernel.
* GMRES iterations on the synthetic matrix (29) are not representative of the AIC
  spectrum.
* Timings are single-machine, warm-cache medians of 3 runs, on an idle desktop; the
  absolute values are not portable, the ratios and the ordering are the finding.
* No accuracy statement is made here: this unit measures the existing production
  path. physical_accuracy_accepted stays false.
