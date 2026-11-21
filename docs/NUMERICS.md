# 数值实现定义

本文规定 DPanel 低速主线的离散装配、线性代数、非线性求解与数值判据。模型本身（未知量、
方程、闭合、力的定义）见 [MODEL.md](MODEL.md)。

## 1 算子装配

| 项 | 定义 | 实现 |
|---|---|---|
| 双层位势核 | 源三角形上 P1 偶极的解析核；奇异性按解析积分处理 | `panel/p1_lifting_g1.py` `_doublet_matrix` |
| 目标求积 | Duffy 变换的张量 Gauss（默认 order=3，可调 `galerkin_order`）；权重含 (1−u) Jacobian | `panel/p1_galerkin_wall.py:15` `_duffy_rule` |
| 批处理 | 目标面按批组装，核在每批内遍历一次源面 | `assemble_galerkin_multi` |
| 质量矩阵 | 精确 P1 质量矩阵：每个三角面 area/12·[[2,1,1],[1,2,1],[1,1,2]] | `p1_mass_matrix`（:31） |
| 尾迹影响 W | 每个 group 的尾迹面偶极对每个壁面目标点；无尾迹时为 0 列 | `assemble_multi` / Galerkin 版 |
| 约束 J | 每站一行：Σⱼ cⱼ μⱼ − Σₖ hₖ Γₖ = 0（上下岸 μ 差 = 站位 Γ 的 hat 插值） | `panel/p1_lifting_g2.py:190` |
| 线性速度差 K | 每站使用上/下岸最近面平均 P1 梯度的差。用于 `signed_speed` 和非线性闭合的初始化 | `assemble_te_constraints` |

装配是确定性的：同样的网格、同样的 order 给出逐位相同的矩阵（无随机抽样、无并行归约次序
依赖）。

## 2 约束处理与降维

壁面方程与后缘约束组成约束最小二乘：

    min_s ‖L⁻¹(A s − b)‖²   s.t.  [J; K] s = 0,   s = (μ, Γ)

其中 L 是质量矩阵的 Cholesky 因子（Galerkin）或面积开方权重（配点）。实现顺序：

1. 约束矩阵 C = [J; K]；零空间 N = null(C, rcond=1e-11)。若 N 为空列直接报错
   `All independent mu/Gamma DOFs consumed by constraints`。
2. 白化：Galerkin 时 M = L Lᵀ，B = L⁻¹[A|W]，b_w = L⁻¹b；配点时 B = diag(√(A_f/Ā))[A|W]。
3. 线性最小二乘：解 B N z = b_w（`lstsq`），s = N z；输出 `linear_constrained_rank`、
   `linear_singular_min`。

## 3 pressure 闭合的求解器

结构：外层只对 Γ 的**零空间补空间**做非线性迭代，内层是带预分解的线性最小二乘。

| 项 | 定义 |
|---|---|
| 残差 | r(s) = [B s − b_w ; w · p(s)]，w = `kut_weight` = 5，p 为逐站 ΔCp（恢复模型给出） |
| 迭代 | 结构化 Gauss-Newton；B 部分预分解（QR/Cholesky），Γ 方向用 Woodbury 恒等式修正 |
| 判据 | `pressure_kutta_acceptance`（`panel/p1_lifting_g2.py:269`）：相对梯度 ≤1e-6；wall_rms ≤ 1.05×该算例下界；pressure_rms ≤ 1e-4 |
| 输出 | `pressure_status`、`pressure_success`、`pressure_message`、`pressure_nfev`、`pressure_seconds` |
| 失败 | 不抛异常；状态字段记录，并保留当前迭代解 |

判据中的三个量按相对量比较（相对于该算例自身的梯度范数与下界），不使用绝对阈值。

## 4 potential_jump 闭合的求解器

| 项 | 定义 |
|---|---|
| 结构 | 方阵：J s = 0 精确成立；对 Γ 做 Newton，求 p(Γ) = 0 |
| 内层 | μ 对 Γ 仿射：一次最小二乘得 μ = μ₀ + MΓ，之后每步只解 n_Γ × n_Γ 系统 |
| 收敛 | Newton 收敛即接受；无专用的后缘残差判据 |
| 多根 | 粗网格上 Δp_TE(Γ)=0 有多个根，不同初值收敛到不同根（同一网格 CL 可差 3.4%） |
| 失败 | 抛 `potential-jump Kutta ...`（区别于返回坏状态） |
| 成本 | 782 面、6 段弦向、4 段展向：2–8 步，秒级；比 pressure 快，但没有罚项的鲁棒性 |

## 5 回退规则

仅当（a）调用方未 pinned 闭合，（b）`kutta="potential_jump"` 抛出 `potential-jump Kutta ...`
时，用同一组算子重解 pressure 闭合，并在输出写 `kutta_requested`、`kutta_effective`、
`kutta_fallback`。回退永不静默：`kutta_effective` 与实际使用的闭合一致。
可微路径（`p1_cst_gradient`）不设回退，失败即报错。

## 6 灾变门（不是精度门）

`panel/p1_solution_status.py` 定义 `PlausibilityControls`：

| 判据 | 阈值 | 后果 |
|---|---|---|
| 非有限（解或条件数） | — | implausible_solution |
| |CL| | > 20 | implausible_solution |
| |CD_pressure| | > 1 | implausible_solution |
| ‖Γ‖ / ‖μ‖ | > 50 | implausible_solution |
| 约束系统条件数 | > 1e6 | implausible_solution |
| 最小 Γ 列范数 / 中位数 | < 0.1 | 仅警告 `weak_gamma_column_support_proxy` |
| 同算例 CD_pressure 回归 | > 5× 参考 | 仅警告 `same_case_pressure_drag_regression` |
| 非线性未收敛 | `nonlinear.success = False` | `nonlinear_not_converged` |

状态取 `solver_success` / `implausible_solution` / `nonlinear_not_converged`。这些阈值是尺度与
基相关的代理量，用于拦截灾变，不构成精度验收；`physical_accuracy_accepted` 恒为 `False`。

## 7 条件数与已知数值事实

| 算例 | 量 | 值 | 用途 |
|---|---|---|---|
| conv_s075 翼身（6224 面）修复前 | cond(B N) | ≈ 4.7e5 | 触发 #77 的支撑修复（[DEFECTS.md](DEFECTS.md) D-1） |
| 椭圆 3096 面纯 pj | cond | 1.63e6，97.2% 质量集中 2 站 | 说明 pj 的近零模态 |
| 椭圆 2328 面 | cond | 3.98e3，top3 73.9% | 同上 |
| 同 3096 面 pressure | Kutta 残差 | 4.84e-14，γ 2 次变号 | 罚项抑制近零模态 |

## 8 成本与硬件

| 项 | 数值 |
|---|---|
| 单次求解（782 壁面节点，固定尾迹） | 2–30 s（CPU 单线程，双精度） |
| 12 组 A/B（[VALIDATION.md](VALIDATION.md) §3） | ≈ 3 min |
| 设备 | CPU torch 为默认；CUDA fp64 可选（`device` 参数），无 CUDA 时须显式回退 |
| 线程 | 证据脚本固定单线程（`torch.set_num_threads(1)`，OMP/MKL/OPENBLAS=1），保证可比 |
| 内存 | 显存/内存随 DOF 增长；隐式可微路径的内存对照见 `docs/reference/DIFFERENTIABLE_MEMORY.md` |

## 9 自由尾迹迭代

`solve_steady_free_wake`（`panel/p1_steady_free_wake.py:537`）把固定尾迹解作为第 0 次迭代，
之后按下式迭代：

| 参数 | 默认 | 定义 |
|---|---|---|
| iterations | 6 | 尾迹更新次数（第 0 次为固定尾迹解） |
| relaxation | 0.25 | 顶点位移的松弛因子 |
| probe_fraction, step_fraction | 0.055, 0.009 | 尾迹面探针位置与推进步长（按弦长比例） |
| geometry_tol, slip_tol | 2e-3, 1e-2 | 几何收敛与滑移（法向速度）门限 |
| min_axial | 0.08 | 轴向速度下界，防止尾迹回卷 |
| frozen_guard | True | 几何门禁：某步若不合法则阻止该步更新 |

每次迭代在当前尾迹几何上重算 Γ 与壁面解；收敛只在最终状态上报告，不提前终止。该路径
**没有**耦合伴随（几何随解变化不影响梯度），不进入生产默认链。

## 10 进程约束

| 约束 | 原因 |
|---|---|
| 同一进程内 torch 之后再调 SciPy LAPACK 可能 0xC0000005 | 见 `docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md` §6f；证据脚本必须分进程跑 |
| 单一解释器环境 | `C:/Users/zyx20/anaconda3/envs/myml/python.exe`（Python 3.13，torch 2.13，numpy/scipy） |
| 不使用 JAX 于生产路径 | JAX 仅存在于 `core/` 与历史实验中（`docs/reference/GEOMETRY_DIFFERENTIABILITY.md`） |
