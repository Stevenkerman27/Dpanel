# 归档说明

这里的文档按"当时的交付单元"分组，不再更新：

- `pr/`：PR 验收报告（`PRnn_*`）。
- `issue/`：单文件 issue 报告（`ISSUEnn_*`）。对应的原始数据仍在该 issue 的证据目录
  （`docs/issueNN/`），本层只放结论文字。
- `notes/`：一次性分析、阶段总结、历史交接（`PAUSE_HANDOFF_*`、`CLEANUP_*`、`STAGE1_*` 等）。
- `evidence/`：已退役 PR/研究线的原始证据目录（pr26–pr36、wake_*_20261026、mesh_*_verify、
  issue28/issue49_phase0、wingbody、shed_groups_cad），以及 `p1_receipts/`、
  `issue49_vspaero_raw.tar.gz`（VSPAERO 原始日志）与 `_moved_2026-11-03.tar.gz`（搬迁前备份）。

## 怎么用

- 要知道"结论是什么"，先看 [../README.md](../README.md) 的现役表；归档只用于追溯"当时为什么这么判"。
- 检索：`grep -rn "关键词" docs/archive`。
- 历史：路径改过，`git log --follow -- docs/archive/issue/X.md` 能看到移动前的全部提交。
- 归档里如果某篇又被引用为现行结论，把它提升回 `docs/` 并在现役表登记，不要在两处维护同一份读数。
