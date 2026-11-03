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
| 测试基线、已知失败、环境矩阵 | [TEST_STATUS.md](TEST_STATUS.md)（历史 PR 口径见 [archive/notes/TEST_HEALTH_2026-10-30.md](archive/notes/TEST_HEALTH_2026-10-30.md)） |
| 某个模块被删了，代码去哪找 | [RETIRED_RESEARCH.md](RETIRED_RESEARCH.md)（含归档 tag 与 checkout 方法） |
| VSPAERO 对照工具怎么用、有哪些坑 | [VSPAPI.md](VSPAPI.md) |
| 协作与验收流程（回执、先声明阈值） | [WORKFLOW.md](WORKFLOW.md) |

## 2. 证据包（原始数据，按 issue 编号）

结论在正文，原始数据在子目录，命名 `issueNN/`；一个目录对应一个 issue，里面的 JSON/CSV 就是表里的数。
历史 PR/研究线的证据目录已移入 `archive/evidence/`，本层只留现役文档直接引用的数据。

| 目录 | 覆盖 |
| --- | --- |
| `issue92/` | torch 移植（S1–S12）与 VSPAERO 四几何：parity、bench、bundle 索引 |
| `issue84/` | RM-P1 尾迹求值器：节点拒绝、投影扇区、门槛 A/B、G2 ledger |
| `issue77/`、`issue71/` | Γ 支撑修正；P1 Galerkin 收敛测量 |
| `issue49_vspaero/` | VSPAERO 外部对照的汇总表、polar、截面与 SHA-256；`raw/`、`ladders/` 原始日志已从工作树移除，需要时按该目录 README 第 8 节从 `ce91e0a` 检出 |
| `geometry_robustness/`、`panel_robustness_evidence_v2/` | 几何鲁棒性矩阵与当前面板鲁棒性证据 |

## 3. 归档（`archive/`，只读）

- `archive/pr/`（19 篇）：PR 报告，PR 合并即归档。
- `archive/issue/`（54 篇）：单文件 issue 报告。
- `archive/notes/`（35 篇）：一次性分析、阶段总结、历史交接。
- `archive/evidence/`：已退役 PR/研究线的证据目录（pr26–pr36、wake_*_20261026、mesh_*_verify、
  issue28/issue49_phase0、wingbody、shed_groups_cad 等），以及 `p1_receipts/`。
  两个打包件（`_moved_2026-11-03.tar.gz`、`issue49_vspaero_raw.tar.gz`）只在作者本机，不入库；
  被移除的内容在 `ce91e0a` 之前的提交里都能取回。

归档不等于失效：这些文件保留了当时的读数与判断，路径改过但内容没改；用 `git log --follow` 仍可追溯。

## 4. 维护规则

1. 新文档先自问"半年后还有人读吗"。会 → 放本层并在上表加一行；只是这次验收 → `archive/`，数据进 `issueNN/`。
2. 每篇结论文档必须写清：数据文件路径、复现命令、以及**不声明**的部分。
3. 先声明阈值再看数：阈值写在文档里、写死在被测脚本里，事后改写要留更正记录。
4. 一个模块只留一个入口文档；被取代的入口文档归档并在 `RETIRED_RESEARCH.md` 记一行。

## 5. 仓库与目录约定

- 真正的 git 仓库就在根目录 `D:/3D/Projects/panel`（远程 `Stevenkerman27/Dpanel`，分支 `work/root-2026-11-03`）。
  旧检出 `runs/pushrepo`、`runs/fix_master` 已于 2026-11-03 删除；它们检出的分支
  （`research/issue49-vspaero-evidence`、`research/issue92-phase1-wake-evaluator`）都在本仓库历史里，
  `git checkout <branch>` 即可取回，不需要另存一份工作树。
- 不回仓库的大件：`runs/`、`audit/`、`lifting3d/results/*.npz`（2026-11-03 清理）、`.conda_env/`。
- `runs/` 与 `audit/` 已按功能分层（`cases/`、`benchmarks/`、`studies/`、`verification/`、`scratch/`），
  不再按 issue/PR 命名；目录说明与旧路径映射见 [../runs/README.md](../runs/README.md)、
  [../runs/MIGRATION.md](../runs/MIGRATION.md)。
- 模块入口文档：`panel/README.md`、`torch_panel/README.md`、`monitor/README.md`、`test/README.md`、`lifting3d/README.md`、`potential/README.md`、`vspapi/`（见 `VSPAPI.md`）。
