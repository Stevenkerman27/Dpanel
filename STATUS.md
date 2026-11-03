# 项目状态

更新：2026-11-03。本文件是根目录唯一的状态入口；2026-10-26 的旧状态文档已移到 `docs/archive/root/`。

## 一句话

JAX 可微气动面元求解器（解析源/偶极子 + 真实流体侧接触尾迹），另有全势/P2 对照线；几何与离散导数在声明范围内通过，**物理精度尚未验收**（`physical_accuracy_accepted=false`）。

## 代码基线

- 分支 `research/issue92-phase1-wake-evaluator`，提交 `ce91e0a`（master + 27 个提交）；根目录现在是唯一工作树。
- torch 移植（issue #92）当前默认是 S9–S11；架构判断与当前默认见 [docs/ISSUE92_TORCH_PORT_MAP.md](docs/ISSUE92_TORCH_PORT_MAP.md)，S1–S8 的交付记录在 [docs/archive/issue/ISSUE92_DELIVERY_LOG.md](docs/archive/issue/ISSUE92_DELIVERY_LOG.md)。
- B3 的 DPanel/VSPAERO 对照（48 项判定，4 项失败）连同共同参考点力矩、剖面 Cp、固定→自由尾迹见 [docs/ISSUE92_B3_VSPAERO_PROTOCOL.md](docs/ISSUE92_B3_VSPAERO_PROTOCOL.md)。自由尾迹只有 rect 跑完，其余三个算例在 `combine_wings` 后原生崩溃（0xC0000005），属未解决阻塞。
- 已退役的研究代码（P1 配点/Galerkin 变体等）见 [docs/RETIRED_RESEARCH.md](docs/RETIRED_RESEARCH.md)；它们的结果仍作为文档与 JSON 保留。

## 文档入口

| 想知道 | 看 |
| --- | --- |
| 协作、分支、PR 与验收规范 | [AGENTS.md](AGENTS.md) |
| 现役文档索引与证据包 | [docs/README.md](docs/README.md) |
| 历史报告（PR/issue/notes/evidence） | [docs/archive/README.md](docs/archive/README.md) |
| 案例怎么跑 | [docs/README.md](docs/README.md) 的索引，每篇开头有复现命令 |
| 验收口径（代数 vs 物理） | [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| 测试现状与已知失败 | [docs/TEST_STATUS.md](docs/TEST_STATUS.md) |

## 本轮文档清理（PR #99）

- 交付分支 `work/root-2026-11-03`，PR：https://github.com/Stevenkerman27/Dpanel/pull/99 （base 为 `research/issue92-phase1-wake-evaluator`，所以 PR 只含一个提交）。
- 内容：废弃证据与旧状态文档移入 `docs/archive/`（顶层保留 17 篇）；ISSUE92 只留当前默认 S9–S11 相关章节；新增本文件与测试基线文档。
- 未包含：`.github/workflows/galerkin-cad-validation.yml` 里的文档路径过滤——更新它需要带 workflow 权限的 token。

## 旧状态文档的处理

2026-10-26 的 `README`、`DEVELOPMENT`、`PAUSE_HANDOFF`、`SOLVER_ROUTE`、`WBCAERO_STUDY`、`cloud_agent_prompt`
都写着“本地/云端暂停”，与当前 [AGENTS.md](AGENTS.md) 的协作规范冲突，已归档到 `docs/archive/root/`。
其中仍然有效的技术判断（路线选择、全势对照、WBCAero 研究边界）保留在归档原文里，不再作为现行状态。
