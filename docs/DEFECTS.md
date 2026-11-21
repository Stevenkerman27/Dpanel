# 缺陷登记

本文登记求解器的已知缺陷与已完成的规格修改。每条给出：症状的**可判定定义**、根因、修复后的
不变量、验证数据与残余风险。状态只有三种：

| 状态 | 定义 |
|---|---|
| fixed | 修复已进入默认链，且有回归或冻结数据复核 |
| open | 现象可复现，尚无判据把它与可接受解区分开，或修复未进入默认链 |
| external | 属外部对照/环境问题，不修代码，只记录边界 |

## D-1 Γ 支撑在翼身接触站位上的退化（#77，fixed）

**症状定义**：翼身组合（conv_s075，6224 面，α=5°）上出现 `cond(B N) ≈ 4.7e5`、
`‖μ‖ ≈ 228`、`‖Γ‖ ≈ 298`、`CL ≈ 1328`。判据：CL 超出灾变门（>20）且 Γ 列范数失衡。

**根因**：机身接触区间位于 y=0 与 y=0.2，真实外露主翼后缘从 y≈0.249996 起。旧装配把每个
站位都变成独立 Γ 基，于是在 y=0/±0.2 出现"壁面影响≈0 的独立基"，而 TE/Kutta 约束仍作用
其上，约束系统因此近奇异。

**修复规格**：从声明的上/下后缘面 pair 推导真实半展支撑，剔除内侧接触站位作为独立 Γ 基；
仅当 `gamma_station_dof` 明确把该站位绑到真实后缘根 DOF 时才绑定，否则抛 `ValueError`。
尾迹几何不变；镜像后缘根两站保留。新增 `solution_status` 灾变门（[NUMERICS.md](NUMERICS.md) §6）。

**修复后不变量**：Γ 基的支撑只覆盖外露后缘；根部非零 Γ 由真实根 DOF 承担，翼尖 Γ 为 0；
`implausible_solution` 在非有限、|CL|>20、|CDp|>1、‖Γ‖>50‖μ‖、cond>1e6、零支撑 Γ 列时触发。

**验证**（6224 面，单线程）：

| 配置 | CLp | CL_γ | CDp | body max\|Cp\| | cond |
|---|---:|---:|---:|---:|---:|
| TE-only split（修复前基线） | 0.4060144343 | 0.4114293439 | 0.0998550679 | 15.47497247 | 353.0713 |
| 接触迹 split + 旧采样 | 0.4746695635 | 0.4799008306 | 0.0129259786 | 1.05599135 | 306.8311 |
| 接触迹 split + 声明 TE 采样（最终） | 0.4746467377 | 0.4798586233 | 0.0129238569 | 1.05598401 | 308.9496 |
| 同算子 pressure Kutta | 0.4658279948 | 0.4709025655 | 0.0121672079 | 1.05159291 | — |

相关套件 83 passed；重放数据 `verification/issue77/`。

## D-2 尾迹-机身接触迹缺少分裂势自由度（#77，fixed）

**症状定义**：接触迹处近场梯度与压力非物理：body `max|Cp| = 15.47`、`CDp = 0.0999`
（修复后分别为 1.056 与 0.0129）。

**根因**：`smooth_wall_contact`（尾迹与机身壁面的接触曲线）在 P1 适配器里没有分裂势 DOF，
两片壁面共用一个 μ，于是非零的尾迹势跳变被挂在一条人为连续的壁面迹上。同时根部 Kutta
采样按"同岸最近质心"取面，会取到机身三角形（tag 20）而不是声明的后缘岸面。

**修复规格**：对每条声明的 cut 边要求两个反向绕的壁面 + 一个尾迹边界面，只沿 cut 边分裂
顶点扇；接触处约束 `μ_plus − μ_minus = H_contact Γ`（用接触顶点上精确的尾迹插值）；junction
点只写一条 jump 行；不新增压力/速度 Kutta 行。Kutta 采样改用声明的 TE 岸面。双岸约定按
边定向而非法向 z 符号（旋转不变性 oracle）。

**修复后不变量**：原 46 条 TE Kutta 行与 18 个 Γ 不变；conv_s075 多 67 个接触迹 DOF 与
67 条兼容行；根部不再强制零 Γ，只有外翼尖为零。

**残余**：接触迹分类依赖几何声明的 cut 边；未声明 cut 的几何不会自动分裂（见
`docs/reference/PANEL_ROBUSTNESS.md` 的分类规则）。

## D-3 NumPy/SciPy 链的 torch 移植（#92，fixed）

**定义**：迁移目标为：活跃求解路径不再持有 NumPy LAPACK 决策点，默认后端为 torch，冻结
基准保证两种实现逐位一致。

| 阶段 | 内容 |
|---|---|
| 基线 | c953ac5；AST 扫描可达 14 个模块、45 处 np.linalg/scipy 调用；真正决策点 2 个 least_squares、6 个 null_space、5 个 lstsq、3 个 svd、2 个 qr、2 个 solve、2 个 solve_triangular |
| S9（2026-11-03） | 默认后端切 torch |
| S10 | NumPy 求解链退役，冻结于 tag `numpy-path-frozen-s8` |
| S11 | 活跃路径清零 LAPACK；证据脚本仍可调用退役链做对照 |

**约束（进程级）**：同一解释器内 torch 之后再调 SciPy LAPACK 可能触发 0xC0000005；证据
脚本必须分进程运行（`docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md` §6f）。

**历史验证**：`verification/issue92/` 的 s1–s9 冻结包及移植记录见 [历史离散化对照](archive/notes/DISCRETIZATION_COMPARISONS.md) §2。

## D-4 face 恢复与 potential_jump 闭合的系统性不一致（生产路径已移除）

**症状定义**：Galerkin + face + potential_jump 在 30° 后掠翼上，随展向加密 e 单调下降
（每段 3/4/5 等分：0.377 / 0.099 / 0.0028），c6/span5 时 CDi = 16.39（pressure 口径 5e-3）、
Σ|ΔΓ| = 252.2、Γ 变号 56 次、max|dΓ/dy| = 5.60。

**判据缺口**：这些解的 `pressure_kutta_rms` = 9.27e-15、`wall_rms` = 4.15e-6、
`signed_kutta_rms` = 0.0508（span5 为 1.351）——后缘残差与壁面残差都小，现有
`pressure_kutta_acceptance` 与灾变门都不报警。即：**后缘残差型判据不能区分收敛良好的坏解**。

**已知范围**：同一网格的 area + potential_jump 稳定（e 0.895–0.899、0 次变号）；face +
pressure 稳定（e 0.871–0.905）。因此缺陷属于 (face 恢复 × pj 闭合) 这一组合，而不是 pj 方程
本身，也不是后掠翼本身。

**待定义**：需要一条含 Γ 展向锯齿的判据（如 Σ|ΔΓ| 或 max|dΓ/dy| 的上界），在闭合求解后
评估；在此之前 pj 不进入默认链。数据见 [历史离散化对照](archive/notes/DISCRETIZATION_COMPARISONS.md) §3.1–3.2。

## D-5 默认值与文档字符串不一致（fixed）

固定尾迹梯度入口从 `PRODUCTION_KUTTA` 读取默认闭合，默认恢复为 `area`。见 [INTERFACE.md](INTERFACE.md) §3。

## D-6 研究口径与生产口径分叉（open）

**定义**："生产口径"在不同文件中指不同组合：

| 位置 | 壁面 | 恢复 | 闭合 |
|---|---|---|---|
| Web 前向（`tools/web/solve_runner.solve_fixed`） | galerkin | area | pressure |
| `tools/wake_closure_study.py` `run()` | collocation（显式） | area | 显式传参 |
| `test/test_p1_cst_gradient.py`、`test/test_lifting_line_studies.py` | galerkin | area | 显式 |
| `tools/calibration.py:205`、`tools/acceptance.py:70` | collocation | 显式 | 显式 |

2026-11-10 的"pj 在后掠翼上 6 倍阻力 → 默认保持 pressure"决策用的是 collocation + face 口径；
在生产口径（galerkin + area）下该结论不成立（两闭合重合）。缺陷不是某次决策错，而是口径
没有随结论一起记录。

**修复规格**：任何跨入口的结论必须标注 (wall_formulation, pressure_recovery, kutta) 三元组；
离散诊断报告应记录这三项。默认恢复已统一为 `area`。
配点研究与 Galerkin 生产结果仍需注明壁面离散。

## 开放 issue

| issue | 内容 | 状态 |
|---|---|---|
| #113 | 自由尾迹在部分构型上崩塌（CL 下降约 77%） | open，需在默认口径下按 E1–E4 重测 |
| #115 | 标模校准 C01–C06 与参照表的偏差 | open，`benchmarks/calibration/reference.json` |
| #103 | P1 几何可微的边界（顶点级 VJP、接触约束） | open，见 [MODEL.md](MODEL.md) §8 |
| PR #114、#98 | 待合并的改动 | open |
