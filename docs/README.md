# 文档索引

`docs/` 只放两样东西：**能长期引用的结论**（本层）和**当时的证据**（子目录）。一次性的验收报告、
PR 报告、过程分析一律进 `archive/`。目标是：新人打开这一页，十分钟内知道该读哪一篇，不需要浏览 500 个文件。

## 1. 按问题查（现役文档，17 篇）

| 你想知道 | 读这篇 |
| --- | --- |
| 求解链现在是什么、为什么默认 torch、旧 NumPy 链去哪了 | [ISSUE92_TORCH_PORT_MAP.md](ISSUE92_TORCH_PORT_MAP.md) |
| 尾迹模型有哪些、怎么选、哪些只是文档 | [ARCHITECTURE_WAKE_MODELS.md](ARCHITECTURE_WAKE_MODELS.md) |
| 数值到底准不准（几何/离散/势重构/尾缘四类误差预算） | [ERROR_SEPARATION.md](ERROR_SEPARATION.md) |
| 网格加密后哪些量收敛、哪些不收敛 | [MESH_CONVERGENCE.md](MESH_CONVERGENCE.md) |
| 半模 396/805/1205 壁面的成本与载荷敏感性 | [HALF_MODEL_SCALING.md](HALF_MODEL_SCALING.md) |
| 不同布局（后掠/椭圆/BWB/翼身）的鲁棒性矩阵 | [GEOMETRY_ROBUSTNESS.md](GEOMETRY_ROBUSTNESS.md)、[PANEL_ROBUSTNESS.md](PANEL_ROBUSTNESS.md) |
| 网格质量门禁与自适应尺寸 | [MESH_QUALITY_GATES.md](MESH_QUALITY_GATES.md)、[MESH_FEATURE_SIZE.md](MESH_FEATURE_SIZE.md) |
| 耦合自由尾迹（G2）的现状与诚实结论 | [P1_G2_COUPLED.md](P1_G2_COUPLED.md) |
| 几何可微性与灵敏度 | [GEOMETRY_DIFFERENTIABILITY.md](GEOMETRY_DIFFERENTIABILITY.md) |
| 精确 Trefftz 阻力 | [EXACT_TREFFTZ_ENERGY.md](EXACT_TREFFTZ_ENERGY.md) |
| 测试基线、已知失败、环境矩阵 | [TEST_HEALTH_2026-10-30.md](TEST_HEALTH_2026-10-30.md) |
| 某个模块被删了，代码去哪找 | [RETIRED_RESEARCH.md](RETIRED_RESEARCH.md)（含归档 tag 与 checkout 方法） |
| VSPAERO 对照工具怎么用、有哪些坑 | [VSPAPI.md](VSPAPI.md) |
| 协作与验收流程（回执、先声明阈值） | [WORKFLOW.md](WORKFLOW.md) |

## 2. 证据包（原始数据，按 issue 编号）

结论在正文，原始数据在子目录，命名 `issueNN/`；一个目录对应一个 issue，里面的 JSON/CSV 就是表里的数。

| 目录 | 覆盖 |
| --- | --- |
| `issue92/` | torch 移植（S1–S12）与 VSPAERO 四几何：parity、bench、bundle 索引 |
| `issue84/` | RM-P1 尾迹求值器：节点拒绝、投影扇区、门槛 A/B、G2 ledger |
| `issue77/`、`issue71/` | Γ 支撑修正；P1 Galerkin 收敛测量 |
| `issue49_vspaero/`、`issue49_phase0/` | VSPAERO 外部对照全量原始 polar/lod/history/日志 + SHA-256 |
| `issue28_wake_audit/` | 半模尾迹审计 |
| `geometry_robustness/`、`mesh_*_verify/`、`panel_robustness_evidence*/` | 几何与网格矩阵的逐案例数据 |
| `pr26/pr29/pr32/pr34/pr36_acceptance/`、`shed_groups_cad/`、`wake_*_20261026/` | 早期 PR 的验收快照 |

## 3. 归档（`archive/`，只读）

- `archive/pr/`（19 篇）：PR 报告，PR 合并即归档。
- `archive/issue/`（54 篇）：单文件 issue 报告（证据包目录仍留在上一层）。
- `archive/notes/`（34 篇）：一次性分析、阶段总结、历史交接。

归档不等于失效：这些文件保留了当时的读数与判断，路径改过但内容没改；用 `git log --follow` 仍可追溯。

## 4. 维护规则

1. 新文档先自问"半年后还有人读吗"。会 → 放本层并在上表加一行；只是这次验收 → `archive/`，数据进 `issueNN/`。
2. 每篇结论文档必须写清：数据文件路径、复现命令、以及**不声明**的部分。
3. 先声明阈值再看数：阈值写在文档里、写死在被测脚本里，事后改写要留更正记录。
4. 一个模块只留一个入口文档；被取代的入口文档归档并在 `RETIRED_RESEARCH.md` 记一行。

## 5. 仓库与目录约定

- 真正的 git 仓库在 `runs/pushrepo`（远程 `Stevenkerman27/Dpanel`），工作树以 `runs/fix_master` 为主；
  根目录 `D:/3D/Projects/panel` 是工作快照，不同步 `runs/`、`audit/` 等大件。
- 不回仓库的大件：`runs/`、`audit/`、`lifting3d/results/*.npz`（2026-11-03 清理）、`.conda_env/`。
- 模块入口文档：`panel/README.md`、`torch_panel/README.md`、`monitor/README.md`、`test/README.md`、`lifting3d/README.md`、`potential/README.md`、`vspapi/`（见 `VSPAPI.md`）。
