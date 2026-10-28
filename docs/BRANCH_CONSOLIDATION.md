# 47 个分支如何收敛成一条 master

日期：2026-10-28。数据：`runs/branch_audit.json`、`runs/consolidation_plan.json`，规划/执行器：`tools/consolidate_master.py`。

## 1. 先说结论

**不要逐支 `git merge`。** 47 个分支不是 47 条独立特性线，而是同一段研究历史的 47 个**快照**：每个都自带一份当时的 `panel/`，且都从 master 唯一那个 `7b806439`（Initial commit，222 文件、**没有 `panel/` 目录**）派生。

实测（`git merge-tree --write-tree` 对整合分支逐支试合）：

| 分组 | 分支数 | 与整合分支的内容冲突 | 典型差异 |
| --- | --- | --- | --- |
| 大快照（issue49 证据、网格研究、shed-groups、local/panel-jax） | 7 | **6 个冲突文件** | 各带 143–400 个整合分支没有的 docs/tools 文件 |
| P1 链与验收分支 | 30 | 0–2 | panel/ 里 6–8 个模块版本不同 |
| 早期 stage1 探针 | 4 | 0–2 | 只动 aero.py / 少量脚本 |

逐支合并 = 47 次三方合并，其中 20 次会在 `panel/kernel.py`、`panel/half_native.py` 这类**同一文件的不同演进版本**上停下等人裁决，而且裁决结果几乎总是「保留最新那份」。

## 2. 可行做法：一个基线 + 文件级导入

`tools/consolidate_master.py` 实现三段式，默认 dry-run：

```powershell
python tools/consolidate_master.py           # 只出计划：runs/consolidation_plan.json
python tools/consolidate_master.py --apply   # 在 runs/consol_git 工作树里真做
```

**Phase 1 基线**：把 `research/master-integration-p1`（PR #63，含 P0+P1 的 43 个 `panel/` 模块、8 个 P1 测试、`docs/INTEGRATION_P1.md`）合到 master 线上。因为 master 是它的祖先，这一步是干净的 fast-forward 语义合并。

**Phase 2 文件级导入**：分支里存在、而基线里没有的路径（文档回执、`tools/` 脚本、`cases/`、`vspapi/`、`monitor/`、`torch_panel/`），按「谁最后改谁优先」（分支 committerdate）为每个路径挑唯一来源，再按来源分支成批提交。实测 569 个路径，来自 8 个主来源：

| 来源分支 | 导入路径数 |
| --- | --- |
| `research/issue49-vspaero-evidence` | 351 |
| `research/p1-free-wake-realcad-fix` | 85 |
| `local/shed-groups-stepb-20261026` | 21 |
| `local/pr36-acceptance-20261026` | 13 |
| 其余 18 个分支 | 99 |

按顶层目录：`docs` 366、`test` 70、`tools` 36、`panel` 35、`cases` 34、`vspapi` 10、`torch_panel` 7、`monitor` 6。

**Phase 3 内容冲突只报告**：同一路径在基线与分支里都有但内容不同 → 21 个，脚本不覆盖，留给人工：

- 根文件 5 个：`.gitignore`、`DEVELOPMENT.md`、`PAUSE_HANDOFF.md`、`README.md`、`SOLVER_ROUTE.md`
- `panel/` 16 个：`kernel.py`、`half_native.py`、`run_half_native.py`、`differentiable.py` 等

策略：`panel/` 一律保留基线（基线取自研究线 tip `0bffd14`，是最新演进）；5 个根文件人工并一次（README 要把 P1 quickstart 段与旧主线段拼起来）。

## 3. 已经跑出来的结果（本地 `runs/consol_git`，未推远端）

```
phase1 merge: 0
imported 21 branches, 569 paths, 24 commits total
tracked_files=855  panel_modules=76  HEAD=f11f64a
python -c 'import panel.kernel' -> exit 0
```

即：master 从「222 文件、无 panel/」变成「855 文件、76 个 panel 模块、保留全部回执与工具」。

## 4. 怎么变成真正的 master

两条路，都需要你点头（会改默认分支）：

1. **推荐**：从 `consolidat/master` 开一个正式 PR → master，走正常审阅合并，历史里留下「基线 + 21 个 import 提交」。
2. 直接 `git push origin consolidat/master:master`（需要 fast-forward 或 force；master 只有 1 个 commit，force 代价很小，但会重写默认分支）。

合并后收尾：把 46 个分支按处置删掉——`absorbed`（内容已在基线里）、`imported`（文件已导入）、`superseded`（同链后一支取代前一支），只保留 `research/issue49-vspaero-evidence`（P1 审计模块的家）与 `research/master-integration-p1`（回滚点）。分支列表与建议见 `runs/branch_audit.json` 的 `changes_vs_integ` / `files_missing_in_integ` 两列。

## 5. 复现与验收

```powershell
# 规划（只读）
python tools/consolidate_master.py
# 执行（在 runs/consol_git 里建 consolidat/master，不动远端）
python tools/consolidate_master.py --apply
# 验收
cd runs/consol_git; python -c "import panel.kernel"
python -m pytest test/test_p1_*.py -q -p dsh_tmp_plugin   # 120 passed, 4 skipped
```

注意：整仓 `pytest test` 里有 7 个 `test_torch*` 文件需要 `runs/torch_site`（torch 2.14.1+cpu），基础环境没装，收集阶段就会报错——这是环境依赖，不是整合损坏。
## 6. 实际落地结果（2026-10-28）

- 仓库设置为 **squash-only**（`PUT /pulls/64/merge` 用 `merge`/`rebase` 都返回 405 `Merge commits are not allowed`）→ PR #64 以 squash 合并，23 个提交压成 1 个：master `d7f8db92170573f4e1017517f92720231b3bd2b6`（855 blobs / 78 个 panel 文件 / 378 个 docs 文件）。逐个来源分支的导入记录保留在分支 `consolidat/master` 上。
- PR #63 关闭（被 #64 取代），正文改成超集说明。
- 45 个分支删除，删除前把每个分支 tip 打成 tag `archive/<分支名>`（46 个 tag，全部 201 成功）——commit 仍可达，历史不丢。保留 `master`、`consolidat/master`（导入记录）、`research/issue49-vspaero-evidence`（P1 审计模块的家）、`research/master-integration-p1`（回滚点）。
- 教训：`monitor/net.mjs` 的 `apiJson({body})` 只接受**普通对象**，传字符串会二次编码 → GitHub 422 `"{...}" is not an object`。第一次清理时因此 tag 全部失败而分支已删，事后用本地 `refs/remotes/origin` 的完整 SHA 补建 46 个 tag 才恢复；脚本现在应先断言 tag 成功再删分支。
