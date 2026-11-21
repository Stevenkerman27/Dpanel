# 文档索引

打开这一页应该能在十分钟内知道**该读哪一篇**。`docs/` 顶层只放人读的文档，
过程材料（日志、移植清单、协议、原始数据）一律不在顶层：

```
docs/
├── README.md              这一页（分层索引）
├── ROUTE.md               路线层 ①  实际求解链 + 各入口默认值
├── TRAILING_EDGE.md       路线层 ②  后缘闭包 × 速度恢复
├── ISSUE77_92.md          路线层 ③  两次大修留下了什么
├── RETIRED_RESEARCH.md    退役代码与降级文档清单
├── reference/             15 篇：分模块的验收/标定/模型入口，按主题查
└── archive/               历史：pr/ issue/ notes/ root/ evidence/ + 旧数据包
```

- **状态层只有一处**：[STATUS.md](../STATUS.md)。
- **测试重放用的数据不在 `docs/`**：`verification/issue92/`（65 个文件，含 `fixtures/`）与
  `verification/issue77/`，由 `test/test_frozen_bundle_parity.py` 等直接读取。

> 规则：文档与 `STATUS.md` 冲突时以 `STATUS.md` 为准；文档与代码冲突时以代码为准，
> 并且那本身就是一个要修的 bug（见 `ROUTE.md` §5）。

## 1. 路线层：先读这三篇

| 你想知道 | 读这篇 |
| --- | --- |
| 求解链每一步是什么、代码在哪、各入口默认值（壁面/闭包/恢复）、哪里还不一致 | **[ROUTE.md](ROUTE.md)** |
| pressure 与 potential_jump 有什么区别、为什么默认是 pressure；face 与 area 恢复差在哪、什么时候必须用 area | **[TRAILING_EDGE.md](TRAILING_EDGE.md)** |
| issue #77（翼身接触迹 + Γ 支撑）和 #92（Torch 移植）到底改了什么、现在还有哪些部分在生效 | **[ISSUE77_92.md](ISSUE77_92.md)** |

读法建议：先 `ROUTE.md` §1–§2（十分钟），再按需要跳到 `TRAILING_EDGE.md` 的对照表。
这两篇里的每个数字都能用文中给的命令复算。

## 2. 状态

- [STATUS.md](../STATUS.md)：当前主线、测试基线（全量回归数字）、后缘恢复对照表、
  开放 issue/PR、下一步。**只在这里写状态**，其他文档不复制状态。

## 3. 参考层：按主题查

### 验证与标定

| 文档 | 内容 | 口径 |
| --- | --- | --- |
| [ACCEPTANCE.md](reference/ACCEPTANCE.md) | `tools/acceptance.py`：同工况多设置并排比较，打印绝对/相对差与解析参考 | 刻意**不设阈值** |
| [TEST_STATUS.md](reference/TEST_STATUS.md) | 测试基线、已知失败、环境矩阵（2026-11-03 口径） | 历史基线，最新数字看 STATUS.md |
| [CALIBRATION.md](reference/CALIBRATION.md) | #115 标模链 C01 圆柱 → C06 NACA0012，参考值在 `benchmarks/calibration/reference.json` | 验收套件 |
| [ERROR_SEPARATION.md](reference/ERROR_SEPARATION.md) | 几何/离散/势重构/尾缘四类误差预算（2026-10-25，未复跑） | 误差量级记录 |

### 几何与网格

| 文档 | 内容 |
| --- | --- |
| [GEOMETRY_ROBUSTNESS.md](reference/GEOMETRY_ROBUSTNESS.md) | 后掠/椭圆/BWB/翼身的鲁棒性矩阵（程序化生成，JAX 0.10.2 时代） |
| [PANEL_ROBUSTNESS.md](reference/PANEL_ROBUSTNESS.md) | 翼身接触尾迹面元：37/37 主矩阵、180/180 标量比较（不认证物理精度） |
| [MESH_CONVERGENCE.md](reference/MESH_CONVERGENCE.md) | 半模 NACA4412 翼身 5° 的网格收敛（2026-10-26） |
| [MESH_FEATURE_SIZE.md](reference/MESH_FEATURE_SIZE.md) | 前缘局部加密（平均边长 0.220 → 0.120 m） |
| [MESH_QUALITY_GATES.md](reference/MESH_QUALITY_GATES.md) | #42 OCC 网格预算与质量门禁（可选补丁，不改默认） |
| [HALF_MODEL_SCALING.md](reference/HALF_MODEL_SCALING.md) | 半模 396/805/1205 壁面的 LU vs 预条件 GMRES 成本 |

### 模型、可微与力

| 文档 | 内容 |
| --- | --- |
| [ARCHITECTURE_WAKE_MODELS.md](reference/ARCHITECTURE_WAKE_MODELS.md) | 尾迹模型三轴（表示/求值器/更新）、材料线推进、B3 实验与合并边界 |
| [DIFFERENTIABLE_MEMORY.md](reference/DIFFERENTIABLE_MEMORY.md) | 反传内存 15,273 → 1,810 MB（1720 面板），梯度一致性 ~7e-9 |
| [GEOMETRY_DIFFERENTIABILITY.md](reference/GEOMETRY_DIFFERENTIABILITY.md) | 固定拓扑几何 VJP 的接口设计 + 最小 JAX 数值证明（未进生产） |
| [EXACT_TREFFTZ_ENERGY.md](reference/EXACT_TREFFTZ_ENERGY.md) | 分段线性 Γ 的解析理想 Trefftz 积分（#49），生产 CDi 用的就是它 |
| [VSPAPI.md](reference/VSPAPI.md) | `vspapi/`：仓库内唯一 OpenVSP/VSPAERO 客户端，外部对照用 |

### 证据日志（曾经是入口，现在只当证据读）

这些是按时序追加的长日志，结论已提炼进路线层；留作复查用，不要当“现在是什么”来读。

| 文档 | 性质 |
| --- | --- |
| [KUTTA_CONDITION.md](archive/notes/KUTTA_CONDITION.md) | 814 行后缘闭包实验日志（§3 线性闭包 vs ITKUT、§6 五层网格、§10 恢复 A/B、§11 平面形状对比）→ 结论见 [TRAILING_EDGE.md](TRAILING_EDGE.md) |
| [ISSUE92_TORCH_PORT_MAP.md](archive/issue/ISSUE92_TORCH_PORT_MAP.md) | torch 移植 AST 清单与 S1–S12 切片 → 摘要见 [ISSUE77_92.md](ISSUE77_92.md) |
| [ISSUE92_B3_VSPAERO_PROTOCOL.md](archive/issue/ISSUE92_B3_VSPAERO_PROTOCOL.md) | 自由尾迹/B3 与 VSPAERO 的对照协议（600 行）；VSPAERO 不是真值 |
| [P1_G2_COUPLED.md](archive/notes/P1_G2_COUPLED.md) | 部分过时：“NumPy/SciPy CPU”“基座 G1 已退役”已不成立；G2 耦合实测数字仍可用 |

## 4. 归档与证据包

- `archive/pr/`（19 篇）PR 报告；`archive/issue/`（53 篇）单文件 issue 报告；
  `archive/notes/`（35 篇）一次性分析、阶段总结、历史交接；`archive/evidence/` 已退役研究线的原始数据。
- 结论在正文、原始数据在同名子目录（`issue84/`、`issue71/`、`issue49_vspaero/`、
  `geometry_robustness/`、`panel_robustness_evidence_v2/`）。#92 与 #77 的数据已搬到
  `verification/issue92/`、`verification/issue77/`（测试直接读，不能放在文档目录）。
- 被删掉的代码去哪找：[RETIRED_RESEARCH.md](RETIRED_RESEARCH.md)（S12 退役清单 + tag `research-p1-line-archive`）。
- 归档不等于失效：路径改过，内容没改，`git log --follow` 可追溯。

## 5. 维护规则

1. **新文档先自问“半年后还有人读吗”。** 会 → 放本层并加一行；只是这次验收 → `archive/`；被测试重放的数据进 `verification/issueNN/`。
2. **一篇只回答一个问题。** 过程按时间追加的日志不算结论文档，写完当天就该把结论搬进路线层。
3. 每篇结论文档写清：数据文件路径、复现命令、以及**不声明**的部分。
4. 先声明阈值再看数；阈值写进文档也写死在被测脚本里，事后改写要留更正记录。
5. **路线变了先改 `ROUTE.md`**，再改代码注释；一个模块只留一个入口文档，被取代的归档并在
   `RETIRED_RESEARCH.md` 记一行。
6. 文档不承载测试数据：测试只读 `json`/`npz`，不从 Markdown 正文抓关键词当契约。

## 6. 仓库约定

- 仓库根就是 `D:/3D/Projects/panel`（远程 `Stevenkerman27/Dpanel`，主分支 `master`）。
- 不入库的大件：`runs/`、`audit/`、`lifting3d/results/*.npz`。
  `runs/` 按功能分层（`cases/`、`benchmarks/`、`studies/`、`verification/`、`scratch/`），
  每个子目录自带脚本 + `*.json`/`*.npz` 结果，没有统一索引文件；复现入口写在对应文档的
  “复算”一节（如 [TRAILING_EDGE.md](TRAILING_EDGE.md) §7）。
- 模块入口文档：`panel/README.md`、`torch_panel/README.md`、`monitor/README.md`、
  `test/README.md`、`lifting3d/README.md`、`potential/README.md`、`vspapi/`（见 `VSPAPI.md`）。
- 协作与验收流程：先声明阈值、给回执、可复现命令进 `tools/`（原 `WORKFLOW.md` 已废止，
  规则并入本节与 `AGENTS.md`）。
