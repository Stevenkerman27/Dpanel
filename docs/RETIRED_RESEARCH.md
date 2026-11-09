# 已退役的研究代码（S12）

2026-11-09：`tools/issue71/bisect_cut.py` 的固定环量/无切口实验无效，已停用并给出替代命令。
旧 `null_space(J)` 构造既改变 Gamma 又未抵消初始跳跃；无切口算例仍含切口/尾迹自由度。
用 `tools/lifting_line_study.py` 重做严格固定 Gamma 与物理位置探针；历史结论限制见
[Kutta 条件 §10](KUTTA_CONDITION.md)。`fixed_gamma.py` 的固定环量求解已修正，
旧探针仅保留历史用途。

**2026-11-06 更新：** 用户要求重新研究弦向加密不收敛后，
`p1_galerkin_wall.py`、`p1_galerkin_solver.py`、`p1_solution_status.py`
及其基础测试已从归档恢复。弱形式现已成为正式 P1 求解器、自由尾迹
外循环和 Web 的默认壁面离散；历史 SciPy 求解器仍用于独立对照。
下表这三项的“已移除”状态不再适用。弱形式尚未完成自由尾迹或物理精度验收。

### 2026-11-06 退役项复查

- `p1_rank_aware_kutta.py`：真实 CAD 上未发现方案所需的可分离弱模态
  或壁面零空间，不是已解决问题的替代算法；见
  [后缘迹审计](archive/issue/ISSUE71_P1_GALERKIN_TE_TRACE_AUDIT.md)。
- `p1_rank_gauge.py`：得到的结论是当时的算子不应随意钉住自由度，
  属于约束设计证据，不是可直接恢复的修复方案；
  见 [秩与规范](archive/issue/ISSUE56_RANK_GAUGE.md)。
- `manufactured_surface_recovery.py`、`analytic_sphere.py` 及门禁、后缘迹、
  跳跃与边缘指数审计：有独立基准和诊断价值，适合后续恢复并适配，
  但它们不直接修复活动求解器。制造势结果使用精确输入并非真实流动真值，
  见 [本地验收](archive/pr/PR36_LOCAL_ACCEPTANCE.md)。
- `affine_consistent_recovery.py` 和压力重构探针是不同恢复方法的研究候选；
  历史证据不足以认定它们全面解决真实流动中的压力误差。
  见 [压力探针](archive/pr/PRESSURE_RECONSTRUCTION_PROBE.md)。
- `relax_wake.py` 相关早期线/面更新改善过法向残差，却恶化近尾迹压力跳跃，
  且未收敛，不能当作自由尾迹已解决的实现；见
  [尾迹精度记录](archive/notes/WAKE_ACCURACY.md)。

此复查基于该清单及相关归档，不能证明其他全部历史分支都没有可用成果。

`research-p1-line-archive` 是下表所列模块仍位于
`panel/` 下的最后一个提交。S12 把它们从活动树中移除。这里没有任何结论被推翻：它们
产出的每个结果都仍以文档形式留在本仓库中，并且在做过测量的地方，
旁边还有 JSON 产物。被移除的是让你无需检出该 tag
就能重新运行那些测量的代码。

## 为什么要移除可用的代码

三条理由，按权重排序。

1. **它不可达。** 没有任何生产入口导入这些模块；
   `test/test_live_path_has_no_numpy_lapack.py` 中的可达性遍历从十二个活动
   入口出发，永远不会到达这里。它们之所以还活着，只是因为它们自己的测试导入了它们。
2. **它们重复了同一种离散化。** 六个模块用不同的 Kutta 闭合、规范（gauge）或积分
   求解同一个 P1 势问题。S10 正是从 NumPy 链路中移除了这类
   重复，因为两个活动实现会悄然漂移；一个无人调用的变体求解器
   具有同样的失效模式，只是反馈回路更慢，因为
   没有任何东西迫使它跟踪几何或尾迹数据结构的变化。
3. **它们的测量结果已经冻结。** Galerkin 收敛表、边缘指数
   研究、球体验收门禁与恢复对比都已记录为已提交的 JSON
   或表格；重新推导它们只差一次 tag 检出。

## 移除了什么

| 模块 | 它的作用 | 其结果所在位置 |
| --- | --- | --- |
| `panel/p1_g1.py` | 首个 P1 配点升力求解（Dirichlet 势、最小二乘） | docs/archive/notes/INTEGRATION_P1.md, docs/archive/notes/P1_G1_END_TO_END.md, docs/archive/issue/ISSUE49_P1_G1_DESIGN.md |
| `panel/p1_dirichlet.py` | p1_g1 与 G0 工作所基于的 Dirichlet 配点壁面算子 | docs/archive/issue/ISSUE49_P1_G0_SPHERE.md |
| `panel/p1_galerkin_wall.py` | P1 Galerkin 质量/弱形式壁面算子 | docs/archive/issue/ISSUE71_GALERKIN_WALL_PROTOTYPE.md |
| `panel/p1_galerkin_solver.py` | issue 71 收敛结果背后的弱形式 Galerkin 求解 | docs/archive/issue/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md |
| `panel/p1_rank_aware_kutta.py` | 秩感知 Kutta 变体及其秩间隙诊断 | docs/archive/issue/ISSUE69_P1_DIRICHLET_RANK_KUTTA.md |
| `panel/p1_rank_gauge.py` | 百分比秩规范实验 | docs/archive/issue/ISSUE56_RANK_GAUGE.md |
| `panel/p1_gate_checks.py` | G0/G1 验收门禁套件 | docs/archive/issue/ISSUE49_P1_GATES.md |
| `panel/p1_doublet_moment.py` | 门禁工作使用的偶极子矩记账 | docs/archive/issue/ISSUE49_P1_G0_SPHERE.md |
| `panel/p1_g0_acceptance.py` | G0 球体验收装置 | docs/archive/issue/ISSUE56_G0_ACCEPTANCE.md |
| `panel/p1_solution_status.py` | Galerkin 工具共用的状态枚举 | docs/archive/issue/ISSUE77_CONTACT_TRACE_FIX.md |
| `panel/p1_te_jump_integral.py` | 后缘跳跃积分审计 | docs/archive/issue/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md |
| `panel/p1_te_trace_audit.py` | 配点线的 TE 迹审计 | docs/archive/issue/ISSUE71_P1_GALERKIN_TE_TRACE_AUDIT.md |
| `panel/p1_solid_angle.py` | 早于 panel/p1_analytic_sheet_velocity.py 的早期立体角辅助模块 | docs/archive/issue/ISSUE56_SOLID_ANGLE_FREE_TERMS.md |
| `panel/p1_wake.py` | 早于 panel/wake_velocity.py 的早期尾迹面辅助模块 | docs/archive/notes/INTEGRATION_P1.md |
| `panel/p1_edge_singularity.py` | 楔形指数研究（PR 74） | docs/archive/issue/ISSUE71_EDGE_SINGULARITY_DERIVATION.md, docs/archive/issue/ISSUE71_EDGE_EXPONENT_MEASUREMENT.md |
| `panel/probe_pressure_reconstruction.py` | 首个压力重构探针 | docs/archive/pr/PRESSURE_RECONSTRUCTION_PROBE.md |
| `panel/affine_consistent_recovery.py` | 仿射一致的压力恢复变体 | docs/archive/issue/ISSUE49_PHASE0_BASELINE.md |
| `panel/manufactured_surface_recovery.py` | 制造曲面恢复基准（oracle） | docs/archive/notes/MANUFACTURED_SURFACE_RECOVERY.md |
| `panel/relax_wake.py` | 早期尾迹松弛驱动脚本 | docs/archive/notes/WAKE_ACCURACY.md |
| `panel/analytic_sphere.py` | 解析球体参考解 | docs/archive/issue/ISSUE49_PHASE0_BASELINE.md |
| `panel/audit_reference.py` | 球体参考审计 | docs/archive/issue/ISSUE49_PHASE0_BASELINE.md |
| `panel/run_p1_rank_aware_kutta.py` | 秩感知变体的 CLI 驱动脚本 | docs/archive/issue/ISSUE69_P1_DIRICHLET_RANK_KUTTA.md |

同时移除：19 个为测试上述模块而存在的测试模块，以及驱动它们的 13 个工具
（`tools/issue71/*.py`、`tools/grass_fan_convergence/` 下的四个 `_galerkin_*` 辅助脚本
外加 `_bridge_diagnose.py`，以及 `tools/vsp_evidence/p1/` 中的全部五个脚本）。

## 刻意保留的内容

| 保留项 | 原因 |
| --- | --- |
| `panel/p1_cp_diagnostics.py` | `test/test_p1_wake_contact.py` 在活动代码断言中使用了它的表面梯度 |
| `panel/p1_closed_jax.py`, `run_p1_closed.py` | 闭合体这条线是独立示例，仍被其自身测试使用 |
| `panel/pressure_stencil.py`, `panel/pressure_recovery_ab.py`, `compare_pressure_recovery.py` | 被保留的半模型与恢复对比模块导入 |
| `panel/solve.py`, `matrix_free.py`, `iterative.py`, `half_native*.py`, `kernel.py` | VLM 与半模型这条线；它们支撑 issue 16、半模型缩放说明与 VSPAERO 对比，属于另一项单独决策 |

## 如何重新运行已退役的测量

```
git checkout research-p1-line-archive
python -m pytest test/test_p1_galerkin_solver.py -q        # issue 71 convergence
python -m tools.issue71.galerkin_ab --help                 # the A/B driver
```

冻结输出仍在活动树中，例如 `docs/issue71/galerkin_ab/*.json` 与
`docs/archive/issue/ISSUE71_GALERKIN_CONVERGENCE_MEASUREMENT.md`。

## 测试基线

| 树 | 结果 |
| --- | --- |
| S12 之前（提交 `aedadd5`） | 1000 passed, 21 skipped, 2 failed |
| S12 之后 | 827 passed, 21 skipped, 2 failed |

这两个失败是既有的 jax 0.12 jaxpr 递归失败，位于
`test/test_half_native_panel.py` 与 `test/test_iterative_panel.py`；它们在干净的 master 上同样失败。

## 回执辅助函数去了哪里

`tools/issue71/galerkin_cad_sweep.py` 并非纯求解器代码：它的三个函数把一个
已求解状态变成回执，且有三个活动测试对它们做了断言。它们被移到
`panel/p1_contact_report.py`，命名为 `trace_audit_summary`、`contact_jump_diagnostics` 与
`component_pressure_loads`；`test/test_p1_wake_contact.py` 与 `test/test_p1_galerkin_cad.py` 现在
从那里导入它们。第四个辅助函数 `status(state, cap)` 把状态映射为状态字符串，
并依赖已退役的 `panel/p1_solution_status.py`；它的唯一测试被删除而不是
重新实现，因为活动验收谓词位于 `panel/p1_lifting_g2.py`，并
由其自身测试覆盖。
