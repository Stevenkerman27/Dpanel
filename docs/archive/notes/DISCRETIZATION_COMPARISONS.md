# 历史离散化对照

本页保存旧 `face`、壁面离散和 Torch 移植对照。下文的测试与读取方描述沿用归档时的口径，
旧冻结回放测试保留在 `93abddb`。
现役物理验证见 [VALIDATION.md](../../VALIDATION.md)，项目状态见 [STATUS.md](../../../STATUS.md)。

## 1 验证层级

| 层级 | 定义 | 判据来源 | 当前状态 |
|---|---|---|---|
| L1 代数一致性 | 同一离散的两种实现（numpy/torch、单翼/多翼、解析/数值核）给出同一结果 | 冻结基准 parity 测试，差值 ≤ 双精度舍入 | 通过 |
| L2 数值收敛 | 网格/求积加密后解与积分量稳定 | 网格收敛研究、闭包残差 | 部分通过（有量级证据，无统一阈值） |
| L3 物理精度 | 与实验/高保真 CFD 的定量一致 | 标模参照（C01–C06） | **未验收**：`physical_accuracy_accepted = False` |

结论：DPanel 的通过项只覆盖 L1 与部分 L2；任何"精度已验收"的表述都与 `STATUS.md` 矛盾。

## 2 冻结基准（verification/issue92）

冻结包是**测试输入**，不是文档：npz 存算子与参照解，json 存 parity 记录。

| 包 | 内容 | 读取方 |
|---|---|---|
| `s1_reference_{base,hard,stiff,backtrack}.npz` + `s1_parity_*.json` | 压力型结构化最小二乘的问题与参照解（4 个难度档） | `test/test_p1_pressure_kutta_torch.py` |
| `s2_reference.npz` + `s2_rank_parity.json` | 秩感知最小二乘（null_space/lstsq 替换）的参照 | `test/test_p1_rank_aware_torch.py` |
| `s3_reference_g1_wing_rect.npz` + `s3_parity.json` | 单翼矩形 pressure 驱动参照 | `test/test_p1_pressure_kutta_torch_driver.py` |
| `s4_assembly_parity.json` | 装配层 numpy/torch 一致性 | `test/test_p1_galerkin_wall.py` 等 |
| `s5_reference_g2.npz` + `s5_loop_parity.json` | G2 多组求解循环参照 | `test/test_p1_solve_torch.py` |
| `s6_parity.json` | 迁移期 parity 记录 | 人工复核 |
| `s7_rm_parity*.json` + `s7_reference_index.json` | 正则化面片速度求值器 parity（含 CUDA 档） | `test/test_p1_regularized_sheet_velocity_torch.py` |
| `s8_adjoint.json` | 伴随/梯度参照 | `test/test_p1_cst_gradient.py` |
| `s9_inventory.json` | 迁移期 NumPy/SciPy 调用清单 | 人工复核 |
| `fixtures/` | 冻结网格与几何输入 | parity 测试 |
| `solver_crossover_*.json`、`pressure_solver_bench.json` | 求解器切换点与耗时基准 | 人工复核 |

parity 测试在输入缺失时**失败**（不 skip），以保证基准不可被静默绕过；只有可选的 CUDA 档
按 `PANEL_TORCH_PARITY` 环境变量启用。

## 3 后缘构型参照表

条件：固定尾迹、α=4°、AR=6、单线程；列 = CL / Trefftz CDi / e / Σ|ΔΓ| / Γ 变号数。
数据源 `runs/verification/route_check/`，复算命令见 §6。

后掠 30°（弦向 6 段 × 展向 4 段，782 节点，1560 三角形）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ|ΔΓ| | 变号 |
|---|---|---|---|---:|---:|---:|---:|
| collocation | face | pressure | 0.278962 | 4.218e-3 | 0.898 | 0.454 | 0 |
| collocation | face | potential_jump | 0.345508 | 3.984e-2 | 0.147 | 6.972 | 4 |
| galerkin | face | pressure | 0.298891 | 4.936e-3 | 0.903 | 0.367 | 0 |
| galerkin | face | potential_jump | 0.343703 | 6.017e-2 | 0.099 | 8.762 | 6 |
| galerkin | area | pressure | 0.299383 | 4.979e-3 | 0.898 | 0.251 | 0 |
| galerkin | area | potential_jump | 0.299370 | 4.980e-3 | 0.898 | 0.250 | 0 |

矩形（782 节点）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ|ΔΓ| |
|---|---|---|---|---:|---:|---:|
| galerkin | face | pressure | 0.319943 | 5.369e-3 | 0.945 | 0.887 |
| galerkin | face | potential_jump | 0.321231 | 6.324e-3 | 0.809 | 2.214 |
| galerkin | area | pressure | 0.320731 | 5.253e-3 | 0.971 | 0.311 |
| galerkin | area | potential_jump | 0.320719 | 5.253e-3 | 0.971 | 0.310 |

椭圆（782 节点）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ|ΔΓ| | 变号 |
|---|---|---|---|---:|---:|---:|---:|
| collocation | face | pressure | 0.305888 | 5.038e-3 | 0.907 | 0.599 | 2 |
| collocation | face | potential_jump | 0.332800 | 5.460e-3 | 0.992 | 0.524 | 0 |
| collocation | area | pressure | 0.306466 | 5.003e-3 | 0.918 | 0.561 | 2 |
| collocation | area | potential_jump | 0.332932 | 5.463e-3 | 0.992 | 0.523 | 0 |
| galerkin | face | pressure | 0.322037 | 5.229e-3 | 0.992 | 0.515 | 0 |
| galerkin | face | potential_jump | 0.321988 | 5.228e-3 | 0.991 | 0.515 | 0 |
| galerkin | area | pressure | 0.322166 | 5.232e-3 | 0.992 | 0.512 | 0 |
| galerkin | area | potential_jump | 0.322121 | 5.230e-3 | 0.992 | 0.512 | 0 |

由表可读出的定量事实（不是精度结论）：

1. Galerkin + area：三个平面形状上两闭合的 CL 差 ≤1.3e-5、CDi 差 ≤1e-6、Σ|ΔΓ| 差 ≤1e-3。
2. Galerkin + face：椭圆两闭合一致（e 0.991–0.992）；矩形 e 从 0.971 掉到 0.809；后掠 e 从
   0.903 掉到 0.099，CDi 涨到 6.0e-2。
3. collocation：pressure 在椭圆上 e = 0.907（换 area 也只到 0.918），pj 为 0.992——差异来自
   壁面离散，不是恢复方式。

### 3.1 弦向细化（后掠 30°，Galerkin，固定尾迹）

| 弦向段 | 节点 | face+pressure (CL/e/Σ|ΔΓ|) | face+pj | area+pressure | area+pj |
|---|---:|---|---|---|---|
| 4 | 522 | 0.299623 / 0.879 / 0.678 | 0.300518 / 0.575 / 3.444 | 0.300505 / 0.891 / 0.241 | 0.300445 / 0.890 / 0.240 |
| 6 | 782 | 0.298891 / 0.903 / 0.367 | 0.343703 / 0.099 / 8.762 | 0.299383 / 0.898 / 0.251 | 0.299370 / 0.898 / 0.250 |
| 8 | 1042 | 0.296543 / 0.905 / 0.267 | 0.296535 / 0.905 / 0.267 | 0.296831 / 0.900 / 0.255 | 0.296829 / 0.900 / 0.255 |

### 3.2 展向细化（后掠 30°，弦向 6 段，Galerkin）

| 每段展向段 | 节点 | face+pressure | face+pj | area+pj |
|---|---:|---|---|---|
| 3 | 590 | 0.300017 / 0.871 / 0.710 | 0.318495 / 0.377 / 3.403 | 0.298829 / 0.895 / 0.224 |
| 4 | 782 | 0.298891 / 0.903 / 0.367 | 0.343703 / 0.099 / 8.762 | 0.299370 / 0.898 / 0.250 |
| 5 | 974 | 0.299281 / 0.903 / 0.277 | 0.198734 / 0.0028 / 252.2（CDi 16.39，56 次变号） | 0.299704 / 0.899 / 0.269 |

face + potential_jump 的 e 随展向加密单调下降（0.377 → 0.099 → 0.0028），而同一批网格的
area + potential_jump 为 0.895–0.899、0 次变号。该组合的残差诊断（c6/span4）：
`pressure_kutta_rms` = 9.27e-15、`wall_rms` = 4.15e-6、`signed_kutta_rms` = 0.0508
（span5 时 1.351）。即后缘压力残差与壁面残差都小，而 Γ 呈锯齿——后缘残差型判据不能识别该解，
属于 [DEFECTS.md](DEFECTS.md) D-4 的证据。

### 3.3 椭圆弦向细化（Galerkin，area）

| 弦向段 | area + pressure (CL/e) | area + pj |
|---|---|---|
| 4（522 节点） | 0.324650 / 0.9935 | 0.324579 / 0.9935 |
| 6（782 节点） | 0.322166 / 0.9918 | 0.322121 / 0.9918 |
| 8（1042 节点） | 0.318697 / 0.9915 | 0.318663 / 0.9914 |

两闭合在每档重合（CL 差 ≤8e-5），但 CL 随弦向加密单调下降且不趋近升力线值 0.328987
（AR 6 椭圆、α=4°、无粘）。该趋势与闭合无关，属固定尾迹/远场离散的偏差，不得作为收敛到
真值的证据。

## 4 其它验证记录

| 主题 | 记录 | 文档 |
|---|---|---|
| 网格收敛（NACA4412） | 加密到某档后 CL 变化约 5%（2026-10-26 记录，无统一阈值） | `docs/reference/MESH_CONVERGENCE.md` |
| 前缘网格特征尺寸 | 0.220 → 0.120 m 的对照 | `docs/reference/MESH_FEATURE_SIZE.md` |
| 几何鲁棒性 | 程序化几何批测（JAX 0.10.2 时代） | `docs/reference/GEOMETRY_ROBUSTNESS.md` |
| 翼身接触尾迹鲁棒性 | 37/37、180/180 通过（接触分类与尾迹） | `docs/reference/PANEL_ROBUSTNESS.md` |
| 半模缩比 | LU vs GMRES 的半模一致性 | `docs/reference/HALF_MODEL_SCALING.md` |
| 误差分离 | 四类误差（离散/迭代/模型/几何）的分解（2026-10-25 记录，未复跑） | `docs/reference/ERROR_SEPARATION.md` |
| 标模校准 | C01–C06 与参照表 | `docs/reference/CALIBRATION.md`、`benchmarks/calibration/reference.json` |

## 5 回归基线

| 项 | 值 |
|---|---|
| 套件 | `python -m pytest -q test -p no:cacheprovider` |
| 环境 | Python 3.13.14、torch 2.13.0+cu132、numpy 2.4.6、scipy 1.18.0，单线程 BLAS |
| 结果 | 674 passed / 17 skipped / 2 xfailed / 2 failed，248.97 s |
| 2 个失败 | 冻结回归所需翼身 CAD 输入不存在：`runs/cases/wingbody/half_cad_wingbody_3x/surface.json`（求解未开始） |
| 2 个 xfail | pressure 配点闭合的升力斜率与弦向细化问题 |

## 6 复算

下列历史对照含旧 `face` 分支，使用 `8807b61` 版本。

    # 三平面形状 × 壁面 × 恢复 × 闭合（§3 主表）
    python runs/verification/route_check/run.py      # → swept_elliptic.json
    python runs/verification/route_check/rect.py     # → rect_colloc.json
    # 弦向细化（§3.1）
    python runs/verification/route_check/refine.py   # → refine.json
    # 展向细化与坏解诊断（§3.2）
    python runs/verification/route_check/root_probe.py
    python runs/verification/route_check/span_face_pressure.py

单线程（`torch.set_num_threads(1)`），12 个解约 3 分钟。JSON 中偶有 `NaN`（`span_e`），解析前
替换为 `null`。

## 7 不声明项

1. 不声明气动精度：`physical_accuracy_accepted` 恒为 `false`，L3 未验收。
2. 不声明后缘闭合的实验正确性：表中差异都是离散层面的对照，不是与实验的比较。
3. 不声明网格无关：§3.3 显示 CL 随弦向加密单调下降，未收敛。
4. 不声明自由尾迹的跨构型适用性：无耦合伴随、无 BWB/大迎角验收。
5. 不声明 VSPAERO/WBCAero 的比对结论：它们是外部对照，不是真值。
