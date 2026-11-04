# Issue 92 Phase 1 step 1: batched P1 influence kernel

Status: measurement + opt-in backend delivered. Defaults do not move, no physical-accuracy claim, no solver change.

## 1. Why this kernel, and why not the LU

Phase 0 (docs/archive/issue/ISSUE92_PHASE0_BASELINE.md) attributed the cost of the reference G1
pressure-Kutta path on the elliptic fixture (3220 wall faces, 1728 unknowns):

| item | seconds | share |
| --- | --- | --- |
| total | 22.89 | 100% |
| scipy.optimize.least_squares (13 dense Jacobian SVDs) | 9.84 | 43% |
| _doublet_matrix assembly (1529 calls) | 12.21 | ~45-50% |
| lstsq (gelsd) | 0.51 | 2% |
| null_space | 0.18 | 1% |

So the first code change is not a solver port. It is the influence assembly: every
wall x wall and wall x wake doublet block is currently produced by a Python loop over
source triangles, one allocation and three edge-log terms at a time. The FP64 device
probe (docs/archive/issue/ISSUE92_PHASE0_BASELINE.md section 7b) also showed that a dense FP64 LU
stays on the CPU on this host (CUDA 2.4-2.6x slower, 0.46x FP64 peak), so the GPU is
only useful where the work is target x source pairwise - which is exactly this kernel.

## 2. The kernel

New module panel/p1_batched_influence.py. Pure PyTorch float64, imported lazily so the
NumPy/SciPy reference never depends on torch being installed.

* source_invariants(tris, *, device, dtype) precomputes, once per assembly: triangle
  vertices, unit normal, the three gradient vectors G = ((-g1-g2, g1, g2)), the
  centroid, the three edge lengths and the outward edge normals.
* block_rows(targets, src, index) evaluates the reference closed form for one
  target block x source block:
  * solid angle D = -2 atan2(a.(b x c), L0L1L2 + (a.b)L2 + (b.c)L0 + (c.a)L1) / (4 pi)
  * edge terms gd_k -= outward_k * log1p(2 ll_k / gap_k), with the same
    product/dot branch and the same gap expression as the reference
  * rows = D/3 + ((X - centroid) D + h gd / (4 pi)) G^T
  * self-panel rows replaced by -0.5 (1/3 + (X - centroid) G^T)
* influence_matrix(...) writes those rows into an (n_target, ndof) tensor with
  index_add_ over face_dofs, i.e. exactly the reference scatter of the three vertex
  values of each source triangle into its three DOFs.
* wake_matrix(...) is the same call with face_dofs = wake_faces and ndof = number of
  wake nodes; multiplying by gamma_hat gives W.

Bounded memory is a design constraint, not an afterthought: the largest temporary is
target_block x source_block x 3 x 3 float64 (512 x 256 x 3 x 3 = 9 MB). No [N, N, 3]
tensor is ever formed, and both block sizes are arguments.

Guards, all raising rather than silently degrading: zero-area source triangle, target
coincident with a source vertex, near-singular edge geometry (gap <= 0), malformed or
out-of-range face_dofs, non-positive block sizes, and source invariants that do not
match the triangle array they are passed with.

Summation order inside each scatter is the kernel order, so equality with the reference
is a float64 rounding statement, measured below, not an algebraic identity.
## 3. Blocking and threads

Self-panel A on the elliptic fixture (3220 faces, 1728 unknowns), best of 3 warm runs,
seconds (docs/issue92/block_sweep.json):

| threads | 128x64 | 256x128 | 512x128 | 256x256 | 512x256 | 1024x256 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.017 | 6.230 | 6.109 | 3.964 | 3.949 | 4.135 |
| 8 | 3.961 | 1.522 | 1.054 | 0.996 | 0.867 | 0.954 |
| 16 | 4.580 | 1.974 | 1.294 | 1.281 | 0.933 | 0.810 |
| 32 | 6.318 | 2.679 | 1.688 | 1.589 | 1.128 | 0.926 |

Readings:

* One thread is arithmetic-bound at about 4 s regardless of blocking, so the speedup
  here is thread scaling of an already vectorised kernel, not a block trick.
* Small blocks are not neutral: 128x64 is 4.0-7.0 s at every thread count, because the
  per-block bookkeeping dominates. The default 512x256 is 0.87-0.93 s.
* The recorded minimum is 16 threads x 1024x256 at 0.810 s, i.e. 7% below 512x256. The
  ordering between 512x256 and 1024x256 is not stable across repeated sweeps (an
  earlier sweep had 8 threads x 512x256 at 0.766 s), so the module default stays at
  512x256 with a 9 MB temporary rather than chasing a 7% difference that does not
  reproduce. Both values are reported here instead of one.

Defaults used everywhere below: DEFAULT_TARGET_BLOCK = 512, DEFAULT_SOURCE_BLOCK = 256,
torch.set_num_threads(8) in the parity runner (the library itself does not set it, so
an embedder keeps its own thread policy).

## 4. Parity on the eight frozen fixtures

docs/issue92/influence_parity.json, all fixtures rebuilt from docs/issue92/fixtures with
their SHA256 verified first. A and W are compared as max abs difference over the whole
matrix; relative scale is at most 1.4 for A and 0.32 for W here, so 1e-15 is about 1e-15
relative.

| fixture | A max abs | W max abs | ref A (s) | batched A (s) | x | ref W (s) | batched W (s) | x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1_elliptic | 4.66e-16 | 2.19e-16 | 9.60 | 1.22 | 7.87 | 4.55 | 0.55 | 8.22 |
| g1_swept30 | 3.06e-16 | 5.04e-17 | 7.13 | 0.89 | 8.03 | 2.84 | 0.37 | 7.71 |
| g1_wing_camber | 1.44e-15 | 1.04e-17 | 0.04 | 0.00 | 9.35 | 0.01 | 0.00 | 4.30 |
| g1_wing_rect | 1.85e-15 | 4.55e-18 | 0.03 | 0.00 | 8.04 | 0.01 | 0.00 | 4.58 |
| g1_wingbody_cad | 7.53e-16 | 4.74e-16 | 0.65 | 0.08 | 8.00 | 0.69 | 0.07 | 9.88 |
| g1_wingbody_cad_3x | 7.41e-16 | 8.88e-16 | 4.67 | 0.68 | 6.86 | 3.21 | 0.41 | 7.82 |
| g2_conventional | 1.73e-15 | 1.67e-16 | 0.04 | 0.01 | 8.36 | 0.01 | 0.00 | 2.39 |
| g2_tail_low | 1.75e-15 | 1.67e-16 | 0.04 | 0.00 | 11.29 | 0.01 | 0.00 | 2.43 |

The reference W column is the per-wake-triangle loop, which is what G1 actually runs;
its 2.4x speedup on the two tiny G2 fixtures is not a kernel statement, the absolute
times there are 10 ms.

Solving with the batched matrices (the same reference solver, matrices replaced):

| fixture | mu max abs | gamma max abs | cp max abs | worst load delta |
| --- | --- | --- | --- | --- |
| g1_elliptic | 1.5e-11 | 1.9e-11 | 1.0e-09 | 1.73e-14 |
| g1_swept30 | 1.2e-14 | 2.2e-14 | 4.2e-13 | 1.20e-14 |
| g1_wing_camber | 4.7e-09 | 1.9e-09 | 2.6e-06 | 1.25e-09 |
| g1_wing_rect | 2.4e-14 | 4.8e-14 | 5.7e-13 | 6.11e-15 |
| g1_wingbody_cad | 4.4e-13 | 7.8e-13 | 2.3e-11 | 1.99e-13 |
| g1_wingbody_cad_3x | 1.1e-13 | 1.9e-13 | 1.4e-12 | 1.04e-13 |
| g2_conventional | n/a | n/a | n/a | 1.42e-13 |
| g2_tail_low | n/a | n/a | n/a | 1.67e-13 |

The wing_camber line is the honest worst case: its mu is 4.7e-9 off and cp 2.6e-6 off.
That fixture is the most ill-conditioned of the small set (stagnation cells on a cambered
section), so the change in summation order is amplified by the solve; the loads still
move by 1.25e-9, which is the number that matters for the acceptance in step 5.

Singular values where the matrix is small enough to afford an SVD:

| fixture | sigma max | sigma min | condition | max abs change |
| --- | --- | --- | --- | --- |
| g1_wing_camber | 1.382 | 2.747e-2 | 50.30 | 8.88e-16 |
| g1_wing_rect | 1.381 | 2.750e-2 | 50.22 | 1.33e-15 |
| g1_wingbody_cad | 1.452 | 1.763e-2 | 82.33 | 5.55e-15 |

Fingerprints preserved (elliptic, pressure Kutta): CL 0.3472736155, CL_gamma 0.3400753819,
CD_pressure 0.0049227373, CM_y -0.0753199621, wall_rms 2.9707472575e-04,
pressure_kutta_rms 7.64093244e-07, constrained_rank 1562. These are the Phase 0 numbers,
so the batched kernel has not moved the reference solution.
## 5. Production entry points (opt-in)

The backend is wired, but it is not the default:

* panel/p1_lifting_g1.py: assemble(wing, *, backend='numpy'), solve(..., backend='numpy').
  backend='torch' routes A through influence_matrix and W through wake_matrix @ gamma_hat.
  Unknown values raise ValueError("backend must be 'numpy' or 'torch'").
* panel/p1_lifting_g2.py: assemble_multi(mesh, *, backend='numpy'),
  solve_multi(..., backend='numpy'), same contract.

End-to-end assembly through the public entry point, same process, same geometry:

| fixture | numpy (s) | torch (s) | x |
| --- | --- | --- | --- |
| g1_elliptic | 14.23 | 1.80 | 7.92 |
| g1_swept30 | 9.97 | 1.23 | 8.09 |
| g1_wingbody_cad_3x | 8.68 | 1.12 | 7.72 |
| g1_wingbody_cad | 1.37 | 0.16 | 8.69 |
| g1_wing_camber | 0.05 | 0.01 | 6.36 |
| g1_wing_rect | 0.04 | 0.01 | 6.40 |
| g2_conventional | 0.05 | 0.01 | 6.15 |
| g2_tail_low | 0.05 | 0.01 | 6.17 |

These totals are larger than the kernel rows because they include the remaining
non-kernel work of assemble (TE bank matrices C and H, the K gradient map, contact
pairs, group bookkeeping). On the elliptic fixture the Phase 0 profile put about 22 s of
the 22.9 s pressure-Kutta total in kernel plus SVD; with the kernel at 1.8 s the assembly
is no longer the dominant term, which is the point of doing this before touching the
solver.

Absolute times are not portable. The same elliptic reference assembly measured 5.19 s
in one run and 9.60 s in another on this host under different background load, so only
ratios measured inside one process are quoted as speedups.

## 6. The same kernel on CUDA (float64)

docs/issue92/influence_cuda.json, myml interpreter (torch 2.13.0+cu132, CUDA 13.2,
NVIDIA GeForce RTX 5060, 8124 MB), same kernel, same frozen fixtures, sources
precomputed once and left on the device, gamma contraction on the device:

| fixture | faces | wake tris | warm A (s) | warm W (s) | cold (s) | transfer (MB) | peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g1_elliptic | 3220 | 1528 | 0.278 | 0.116 | 15.460 | 0.9 | 131 |
| g1_swept30 | 2764 | 1110 | 0.202 | 0.083 | 0.213 | 0.8 | 113 |
| g1_wingbody_cad_3x | 2410 | 1466 | 0.156 | 0.083 | 0.162 | 0.7 | 109 |
| g1_wingbody_cad | 792 | 770 | 0.024 | 0.021 | 0.025 | 0.2 | 80 |
| g1_wing_rect | 112 | 24 | 0.003 | 0.003 | 0.003 | 0.0 | 35 |

Parity against the NumPy reference on the device: A max abs 3.82e-16 to 1.37e-15,
W max abs 3.83e-17 to 7.47e-16, the same float64 level as the CPU torch kernel.

Against the CPU torch kernel measured in the same sweep: elliptic A 1.22 -> 0.278 s
(4.4x), W 0.55 -> 0.116 s (4.7x); swept30 4.4x / 4.5x; cad_3x 4.4x / 4.9x; cad 3.3x /
3.3x; rect 1.3x / 1.3x. The GPU is worth about 4.4x on the fixtures that matter and
nothing on the toy ones; the cold number (15.5 s on the first fixture of the process) is
CUDA context and kernel image initialisation, not kernel time.

Two operational facts that cost time to find:

* Importing tools/issue92/fixtures into a CUDA process pulls SciPy/OpenBLAS through the
  Wing builders, and a BLAS call in a process with a live CUDA context dies with
  "Windows fatal exception: access violation" and no Python traceback. The CUDA runner
  therefore reads the npz files and the manifest directly and re-verifies the file
  digest. The same failure appears when the coefficients are taken back to NumPy and
  multiplied there, so the contraction stays on the device: influence_matrix grew
  return_tensor=True.
* 8.1 GB of VRAM is far above what this kernel needs (131 MB peak at the largest
  fixture), so VRAM is not the step 4 constraint; the constraint is the 4.4x.

## 7. What this does not claim

* No default changed: backend defaults to 'numpy' in both solvers, and the CLI does not
  expose the torch backend yet.
* No physical accuracy claim: physical_accuracy_accepted stays false, and the accepted
  convergence statements remain the ones in docs/MESH_CONVERGENCE.md (CL, Trefftz CD).
* Not a solver result: the dense LU stays on the CPU and solve time is unchanged. Step 2
  targets the 43% spent in least_squares' dense Jacobian SVD.
* The GPU numbers are kernel measurements, not workflow measurements: the production
  assemble() path has no device argument yet, so a CUDA end-to-end assembly is not
  claimed.

## 8. Reproduction

From the repository root, with PYTHONPATH containing the tree, runs and runs/torch_site,
and TMP/TEMP/TMPDIR pointing at runs/pipwork:

* python -m tools.issue92.fixtures --out docs/issue92/fixtures
* python -m tools.issue92.influence_parity --repeat 3 --threads 8 --save-reference runs/issue92/reference --out docs/issue92/influence_parity.json
* python -m tools.issue92.block_sweep --out docs/issue92/block_sweep.json --repeat 3
* the myml interpreter with -m tools.issue92.influence_cuda --fixtures g1_elliptic,g1_swept30,g1_wingbody_cad_3x,g1_wingbody_cad,g1_wing_rect --reference runs/issue92/reference --repeat 3
* python -m pytest test/test_batched_influence.py -q

## 9. Next

In the order the plan requires: (step 2) remove the dense Jacobian SVD from the
pressure-Kutta nonlinear solve using the structure of J, with baseline parity on the same
eight fixtures; (step 3) profile and vectorise the projection-fan wake evaluator at the
real 18992-target size; (step 4) move this kernel and the wake evaluator to CUDA behind
an explicit device argument; (step 5) full G1/G2 CPU vs torch vs CUDA parity, and only
then the matching VSPAERO comparison.
