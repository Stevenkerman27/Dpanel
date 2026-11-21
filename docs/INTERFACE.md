# 接口与仓库契约

本文规定代码入口、参数默认值、输出字段、数据位置与测试契约。模型定义见
[MODEL.md](MODEL.md)，数值判据见 [NUMERICS.md](NUMERICS.md)。

## 1 目录职责

| 目录 | 职责 | 是否生产路径 |
|---|---|---|
| `panel/` | P1 面元求解器、几何/网格、可微包装、证据脚本 | 是（唯一生产实现） |
| `torch_panel/` | torch 迁移期的对照实现 | 否（历史） |
| `potential/` | 全势/P2 求解器 | 否（受限前向证据） |
| `lifting3d/` | 升力线 + CAD 处理 | 否（对照） |
| `core/` | JAX 面元实验 | 否 |
| `volume/` | 体网格相关工具 | 否 |
| `airfoil2d/` | 二维翼型工具 | 否 |
| `vspapi/` | OpenVSP/VSPAERO 客户端（唯一入口） | 否（外部对照） |
| `tools/` | CLI 脚本；`tools/web/` 浏览器工作台；`tools/issue92/` 迁移期工具 | 部分是（`tools/web/` 调生产链） |
| `test/` | pytest 套件（含冻结基准 parity） | — |
| `verification/` | 冻结基准与重放数据（`issue92/`、`issue77/`、尾迹研究） | — |
| `benchmarks/calibration/` | 标模参照数据（C01–C06） | — |
| `runs/` | 运行产物（gitignore），按 `verification/`、`cases/`、`scratch/` 组织 | — |
| `cases/` | 几何与算例输入（json） | — |
| `docs/` | 人读文档（见 [README.md](README.md)） | — |
| `audit/` | 第三方许可与外部代码审计材料 | — |

## 2 生产调用链

    tools/web/solve_runner.solve_fixed            (Web 前向)
    tools/web/grass_runner                        (Web 梯度)
            ↓
    panel/p1_cst_gradient.fixed_wake_geometry     (可微包装，固定尾迹)
            ↓ 或直接
    panel/p1_lifting_g1.solve                     (单翼适配器)
            ↓
    panel/p1_lifting_g2.solve_multi               (求解核)
            ↓
    panel/p1_solve_torch.solve_multi_torch        (torch 前向)

| 函数 | 签名（关键参数） | 位置 |
|---|---|---|
| `solve_multi` | `(mesh, alpha_deg=5., *, kutta=None, kut_weight=5., max_nfev=80, operators=None, pressure_solver=None, backend=None, device='cpu', wall_formulation=None, galerkin_order=3, pressure_recovery="area")` | `panel/p1_lifting_g2.py:333` |
| `solve` | `(wing, alpha_deg=5., *, kutta=None, wall_weight=1., jump_weight=6., kutta_weight=5., seed=None, max_nfev=100, pressure_solver=None, backend=None, device='cpu', wall_formulation='galerkin', pressure_recovery="area")` | `panel/p1_lifting_g1.py:499` |
| `solve_multi_torch` | `(mesh, alpha_deg=5., *, kutta="pressure", kut_weight=5., max_nfev=80, operators=None, device="cpu", pressure_recovery="area")` | `panel/p1_solve_torch.py:77` |
| `fixed_wake_geometry` | `(wing, xyz, wake, reference_area, alpha_deg, max_nfev, ..., kutta=None, pressure_recovery="area")` | `panel/p1_cst_gradient.py:502` |
| `solve_steady_free_wake` | `(mesh, alpha_deg=5., *, kutta='pressure', iterations=6, relaxation=.25, ...)` | `panel/p1_steady_free_wake.py:537` |

`panel/p1_lifting_g1.solve` 只做 `combine_wings([wing])` 后转调 `solve_multi`；单翼 torch 核
`solve_single_torch` 已删除。`wall_weight != 1` 或 `jump_weight != 6` 直接 `ValueError`（旧单翼
加权已退役）。

## 3 默认值（按入口）

| 入口 | wall_formulation | kutta | pressure_recovery |
|---|---|---|---|
| `solve_multi`（默认源） | galerkin | pressure（`PRODUCTION_KUTTA`） | area |
| `solve` | galerkin | pressure | area |
| `solve_multi_torch` | —（沿用算子） | pressure | area |
| `fixed_wake_geometry` | galerkin | pressure | area |
| `tools/web/solve_runner.solve_fixed` | galerkin | production → pressure | area（不暴露参数） |
| `tools/web/grass_runner` | 继承 | production → pressure | area |
| `tools/web/optimize_runner` | — | `--kutta` 默认 pressure | area |
| `tools/acceptance`、`tools/wake_closure_study` | collocation（脚本内显式） | 显式传参 | area |
| `test/test_p1_cst_gradient.py`、`test/test_lifting_line_studies.py` | galerkin | 显式 | area |

Web 参数表（`tools/web/params.py:21` `SCHEMA`）只有几何量：origin_x、span、root_chord、
tip_chord、sweep_deg、dihedral_deg、twist_root_deg、twist_tip_deg、thickness_root、
thickness_tip、camber_root、camber_tip、camber_position、sections。没有闭合与恢复开关。

## 4 输出字段契约

`solve_multi_torch` 返回 dict，字段分组：

| 组 | 字段 |
|---|---|
| 输入回显 | `alpha_deg`、`kutta`、`wall_formulation`、`pressure_recovery`、`backend`、`device` |
| 解 | `mu`（torch tensor）、`gamma`、`cp`、`group_results`（每组 CL_pressure/CL_gamma/gamma/gamma_y/Gamma_integral/wall_faces/wake_triangles） |
| 力 | `CL`、`CL_gamma`、`CD_pressure`、`CM_y`、`force` |
| 诊断 | `jump_rms`、`signed_kutta_rms`、`pressure_kutta_rms`、`wall_rms`、`wall_dual_rms`、`linear_constrained_rank`、`linear_singular_min` |
| 状态 | `pressure_status`、`pressure_success`、`pressure_message`、`pressure_nfev`、`pressure_seconds`、`pressure_history`、`nonlinear`、`solve_seconds` |
| 声明 | `physical_accuracy_accepted = False` |

残差定义（均除以 √行数）：`jump_rms` = ‖J s‖/√n_TE；`signed_kutta_rms` = ‖K s‖/√n_TE；
`pressure_kutta_rms` = ‖p(s)‖/√n_TE；`wall_rms` = ‖A μ + W Γ − b‖/√n_μ；
`wall_dual_rms` = ‖B s − b_w‖/√n_μ（仅 Galerkin）。

## 5 异常语义

| 条件 | 行为 |
|---|---|
| `kutta` 非法值 | `ValueError("kutta=pressure, signed_speed or potential_jump")` |
| `pressure_recovery` 非法值（含 `face`） | `ValueError("pressure_recovery must be area or quadratic")` |
| pj 求解失败且未 pinned | 回退 pressure，写 `kutta_requested/effective/fallback`（不静默） |
| pj 求解失败且 pinned（可微路径） | 直接抛错 |
| 约束吃掉全部自由度 | `ValueError("All independent mu/Gamma DOFs consumed by constraints")` |
| TE 恢复缺同岸入射面 | `ValueError("TE recovery is missing bank-specific incident faces")` |
| 秩不足的 quadratic 恢复 | `ValueError("TE recovery patch is rank deficient after %d rings")` |
| 可微路径遇到接触约束 | `ValueError("Wake contact constraints are not supported by this derivative path")` |

## 6 数据位置

| 数据 | 位置 | 读取方 |
|---|---|---|
| #92 移植档案及 VSPAERO 对照输入 | `verification/issue92/` | 历史研究工具与 VSPAERO 对照工具 |
| #77 接触修复摘要 | `verification/issue77/` | 文档与 CI 工作流 |
| 尾迹研究重放 | `verification/wake_*/` | 对应工具脚本 |
| CST 梯度证据 | `verification/cst_gradient/` | `tools/cst_gradient_check.py` |
| 标模参照表 | `benchmarks/calibration/reference.json` | `tools/calibration.py`（C01–C06） |
| 运行产物 | `runs/verification/`、`runs/cases/`、`runs/scratch/`（gitignore） | 人工复核 |

实验与理论参照由校准程序直接读取。

## 7 测试契约

| 测试 | 固定下来的契约 |
|---|---|
| `test/test_p1_cst_gradient.py` | `area` 下可微与生产算子的 CL/CD 差 ≤2e-9，批量 VJP 与逐载荷 VJP 一致 |
| `test/test_web_solve_threads.py` | `--kutta production` 映射到 `kutta=None` |
| `test/test_p1_te_recovery.py` | `area` 与 `quadratic` 的形状、同位置性质与秩判定 |
| `test/test_p1_pressure_kutta_torch_driver.py` | 后缘压力定义、有限差分 Jacobian 与输入校验 |
| `test/test_p1_rank_aware_torch.py` | 已知零空间、秩截断、最小范数解与输入校验 |
| `test/test_p1_solution_status.py` | 灾变门的阈值语义 |
| `test/test_wake_velocity_methods.py` | 尾迹速度方法文档路径常量指向现存文件 |

全量基线见 `STATUS.md`。

## 8 仓库约定

| 项 | 规定 |
|---|---|
| 分支 | `master` 为可复现基线；长期并行才另开分支并在 `STATUS.md` 记一行 |
| 状态 | 唯一状态源 `STATUS.md`；文档不写"当前状态" |
| 文档 | 一个模块一篇入口（见 [README.md](README.md)）；被取代的归档并登记 `RETIRED_RESEARCH.md` |
| 环境 | 只维护 `C:/Users/zyx20/anaconda3/envs/myml/python.exe`；OpenVSP 用 `PYTHONPATH` 指向安装目录 |
| 第三方数据 | 不复制进仓库（`WBCAERO/`、`OpenVSP-3.56.0-win64/`、`audit/` 为外部材料） |
| 提交 | 一条命令可复现；commit message 写测试命令、环境、结果与限制 |
