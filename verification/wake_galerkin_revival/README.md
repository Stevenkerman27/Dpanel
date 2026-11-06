# P1 weak-wall revival and Web leading edge

2026-11-06. Windows 11, Python 3.13.14 in `myml`, NumPy 2.4.6,
Torch 2.13, single-thread OpenBLAS/OMP/MKL for the P1 scans.
The restored Galerkin implementation uses SciPy and remains an independent
optional research path. It has not been connected to the Web default fixed
solver, the free-wake outer iteration, or a geometry-AD path.
Physical accuracy is not accepted.

## Why restore it

The S12 retirement was because the research modules were not called by live
entry points and duplicated active paths, not because the weak formulation
was found incorrect. Historical issue 71 data showed chordwise collocation
divergence even with fixed circulation and without a wake or cut. The
Galerkin formulation improved that particular study; its former code and
basic tests are restored from `research-p1-line-archive` (`aedadd5`).

## Same-case refinement

The fixed wake is 5c long, two streamwise panels; NACA0012 rectangular wing,
span 4c, alpha 4 degrees, pressure Kutta, quarter-chord moment reference.
Chordwise spacing is cosine and spanwise spacing uniform. Geometry, wall mesh,
Gamma degrees of freedom and wake span faces vary together on the joint
sequence; this is not an isolated wall, Gamma or wake-length convergence test.

In the original collocation scan, fixed span 8 and chord 8/12/14 give
CL 0.092451/-0.202085/-0.325661. The same sequence with weak wall and
order-3 target quadrature gives 0.250807/0.248258/0.248158. For the joint
weak sequence:

| Chord x span | Wall faces | CL | CL Gamma | CD pressure | CD Trefftz | CM y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26 x 28 | 3016 | 0.26208748 | 0.26520551 | 0.00591327 | 0.00532827 | 0.00406007 |
| 28 x 30 | 3472 | 0.26243323 | 0.26552548 | 0.00588011 | 0.00535638 | 0.00394342 |
| 30 x 32 | 3960 | 0.26272780 | 0.26579456 | 0.00585277 | 0.00538096 | 0.00384298 |

Relative changes for these two transitions are, respectively:
CL 0.132% / 0.112%, CL Gamma 0.121% / 0.101%,
CD pressure 0.564% / 0.467%, CD Trefftz 0.525% / 0.457%,
CM y 2.958% / 2.614%. These satisfy the previously declared five-coefficient
3% screen twice in this joint sequence, but are not proof of physical accuracy.
At 20x22, changing target quadrature order 3 to 5 changes the five quantities
by 0.098%, 0.104%, 0.276%, 0.165%, and 1.967%, respectively. This one
quadrature comparison does not certify every finer mesh. The 30x32
order-3 run took 104.4 s while the host was not isolated; it is not a
benchmark. Straight fixed wake, the 5c truncation and the free-wake update
remain independently unvalidated.

Raw JSON files here contain runs and nonlinear termination details. The
unmodified collocation data are in `verification/wake_grid_length/chord.json`
and `grid.json`. Weak and collocation wall residuals use different test
spaces/norms and must not be compared as the same physical error.

## Web geometry

The prior STEP builder used sampled section points as B-spline control points.
The first control-polygon edges on the upper and lower root section had a
154.72-degree opening; a smooth nose has a 180-degree opening under this
measurement. Setting both first control edges along the local thickness axis
gives matched tangents. In an independent 1001-sample Gmsh section check,
the maximum vertical error relative to the analytic default root section
changed from 0.001941c to 0.000516c. The STEP regression tests check the
tangents, trailing-edge endpoints, and root section error for default and
twisted/dihedral geometry. Meshes and STEP files generated before this
change retain their old geometry and must be regenerated.

On the default Web wing at 5 degrees, physical mesh size 0.35 m produced
204 half-wall faces; 0.05 m produced 4594. At 0.05 m the existing Web
half-domain P0 fixed-wake solver completed at CL 0.6773520, CD pressure
0.0142760 and relative linear residual 2.84e-15 (two threads).
That is a single run, not a grid convergence or validation result, and is
not the P1/Galerkin calculation above.

## Reproduction

Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `1`,
and use `C:/Users/zyx20/anaconda3/envs/myml/python.exe`. All output paths
must be new.

```powershell
python -m tools.wake_grid_length_study --wall-method galerkin --levels 4x8,6x8,8x8,10x8,12x8,14x8 --out <chord.json>
python -m tools.wake_grid_length_study --wall-method galerkin --levels 26x28,28x30,30x32 --out <fine.json>
python -m tools.wake_grid_length_study --wall-method galerkin --order 5 --levels 20x22 --out <order5.json>
python -m tools.web_nose_check --size 0.05 --alpha 5 --ncpu 2 --out <fresh-web-directory>
python -m pytest -q test/test_p1_galerkin_wall.py test/test_p1_galerkin_solver.py test/test_web_step_leading_edge.py -p no:cacheprovider
```
# 正式 P1 接入（2026-11-06）

弱壁面算子现在可通过 `panel.p1_lifting_g2.solve_multi(...,
wall_formulation="galerkin", galerkin_order=3)` 使用；默认仍为
`collocation`。求解和质量矩阵 Cholesky 均使用 Torch；弱形式矩阵组装使用
NumPy 解析核，不依赖 SciPy LAPACK。历史 `solve_galerkin_multi` 仍是独立
SciPy 求解对照，两者共享弱算子组装。`wall_rms` 是未缩放的弱残差，
`wall_dual_rms` 是质量矩阵对偶范数对应的 RMS，不能直接与配点 `wall_rms`
比较。压力 Kutta 权重仍以对偶范数构造的壁面残差为准。

稳态自由尾迹同样支持 `wall_formulation="galerkin"`，每步缓存壁面矩阵、
质量矩阵和约束，仅重新积分当前尾迹矩阵。CLI 示例：

```powershell
python -m panel.run_p1_steady_wake --wall-formulation galerkin --galerkin-order 2 --kutta signed_speed --iterations 1 --main-chord-panels 3 --main-span-panels 3 --tail-chord-panels 3 --tail-span-panels 3 --wake-panels 2 --out runs/scratch/galerkin_cli_20261106
```

Windows myml，三种 BLAS 线程变量均为 1：该命令退出 0，CL
`0.245367 -> 0.245135`，`iteration_limit`，未收敛。实际运行
`python -m pytest -q test/test_p1_galerkin_live.py test/test_p1_galerkin_wall.py test/test_p1_galerkin_solver.py test/test_p1_steady_free_wake.py test/test_live_path_has_no_numpy_lapack.py -p no:cacheprovider`：
30 passed。覆盖 CPU/CUDA、两种 Kutta 与 SciPy 对照、单/双翼变形尾迹的
缓存与全重装配、自由尾迹初始步和固定尾迹 8/12 弦向加密。
`python -m pytest -q test/test_p1_galerkin_live.py test/test_p1_galerkin_solver.py test/test_p1_lifting_g1.py test/test_p1_lifting_g2.py test/test_p1_solve_torch.py test/test_p1_wake_contact.py test/test_p1_steady_free_wake.py test/test_live_path_has_no_numpy_lapack.py -p no:cacheprovider`：
53 passed、3 skipped（在添加最后一项加密测试前）。
全量 `python -m pytest -q test -p no:cacheprovider --tb=no` 为 11 failed、
850 passed、21 skipped，失败 ID 与 `runs/verification/native_blas/fixed_pytest.log`
一致。全量测试启动于最终将 `leggauss` 改成 Torch 特征分解之前；
最后改动后上面的 30 项针对性测试重新通过。修改原因是 NumPy `leggauss`
间接调用 NumPy LAPACK，而静态导入检查不能检测该内部调用。

Web 固定尾迹调用半域 P0 稠密求解，Web 自由尾迹调用 P1 配点稳态外循环；
它们是活动路径，但不是同一离散模型，Web 尚未选择弱形式。弱形式目前
不是几何自动微分路径，矩阵组装和返回结果经过 NumPy，不能据此宣称
几何敏感性可用；自由尾迹、独立网格/尾迹长度与物理精度仍待验收。
