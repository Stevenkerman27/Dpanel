# 协作工作流

本仓库是研究型项目：每个功能都要走"实现 → 测试 → 合并进 master"。这份文档记录 2026-10-28 分支整合之后确定的做法，以及它要修的到底是什么问题。

## 1. 为什么有这份文档

整合前的实际状态（可复核）：

| 指标 | 数值 |
| --- | --- |
| 分支 | 48 |
| PR | 37 |
| 真正合并的 PR | **1**（#64） |
| 关闭但未合并 | 36 |
| base 指向 master 的 PR | 4 |
| 不同的 base 分支 | 19 |
| master 的提交数 | 1（`Initial commit`，222 个文件，没有 `panel/`） |

根因有四个，规则就是针对这四个：

1. **PR 被当成交付回执，不是合并请求**——发完评论就结束了，没人执行合并。
2. **master 没有承担集成点**——`panel/` 从未存在过，任何 PR 的 base 只能是别的研究分支，于是天然不会被合并。
3. **分支堆叠 + 全树快照**——每次推送把整棵工作树推成一个新分支，47 个分支是同一棵树的 47 个版本，不是 47 个增量；逐支合并时每支都撞同样的 6 个文件。
4. **没有收口动作**——分支不删，链条越堆越长。

## 2. 五条规则

1. **master 是唯一集成点。** 功能分支一律从 `origin/master` 切；PR 的 `base` 只能是 `master`；测试通过当天合并。
2. **一个功能 = 一个分支 = 一个 PR。** 合并后立刻删分支；要留档就打 `archive/<分支名>` tag（tag 保留 commit 可达性，分支列表保持干净）。
3. **不叠长链。** 若 A 依赖 B，先把 B 合并进 master，再从 master 切 A。链条最多叠一层，且必须同一天合完。
4. **证据/数据随代码走。** 放进同一个 PR 的 `docs/<issue>/` 与 `tools/` 里；不单开"只放数据"的分支。
5. **PR 是合并请求，不是回执。** 回执 = issue 里一条评论 + 指向 commit/PR 的链接。PR 正文写四项：改了什么、验收命令、实测输出、未解决项。

## 3. 标准循环

仓库是 **squash-only**（`merge` / `rebase` 都返回 405），所以合并一定压成 1 个提交；需要保留逐提交结构时，把分支留在远端当记录，并在 squash 正文里指明。

```bash
# 0) 起分支（绝对路径；relative 会相对 -C 的仓库根解析）
git -C runs/integ_git worktree add "D:/3D/Projects/panel/runs/<name>_git" -B <branch> origin/master

# 1) 在本机实现 + 跑测试（sandbox 下 pytest 需 -p dsh_tmp_plugin + DSH_TMP_ROOT）
python -m pytest test/test_<area>.py -q -p dsh_tmp_plugin

# 2) 提交并推送（自动 proxy→direct 切换与重试）
git -C runs/<name>_git add -A
git -C runs/<name>_git commit -m "<type>: <what>"
node monitor/gh_push.mjs push runs/<name>_git HEAD:<branch>

# 3) 开 PR（base=master，正文四项）
node monitor/gh_api.mjs POST repos/Stevenkerman27/Dpanel/pulls runs/pr_<n>.json
node monitor/pr_ready.mjs Stevenkerman27/Dpanel <n>      # Draft → Ready（REST 的 draft:false 会被忽略）

# 4) 合并（squash）+ 删分支（先打 tag 再删）
node monitor/gh_api.mjs PUT repos/Stevenkerman27/Dpanel/pulls/<n>/merge runs/merge_<n>.json   # {"merge_method":"squash"}
node monitor/gh_api.mjs POST  repos/Stevenkerman27/Dpanel/git/refs runs/tag_<n>.json          # {"ref":"refs/tags/archive/<branch>","sha":"<tip>"}
node monitor/gh_api.mjs DELETE repos/Stevenkerman27/Dpanel/git/refs/heads/<branch>

# 5) 回执
node monitor/post_comment.mjs repos/Stevenkerman27/Dpanel/issues/<n>/comments runs/receipt.md
```

### 命令级注意事项

- `monitor/net.mjs` 的 `apiJson({body})` **只接受普通对象**；传字符串会二次编码，GitHub 回 422 `"{...}" is not an object`。而 `DELETE` 不带 body，会照常成功——所以**先确认 tag 建成（201）再删分支**。
- `monitor/pr_ready.mjs` 需要两个参数：`<owner/repo> <number>`。
- `PATCH /pulls/<n>` 只带 `body` 会改正文、不会关闭；关闭要显式 `{"state":"closed"}`。
- 沙箱下 node 的 `spawn` 用管道会 `EPERM`：需要 git 时用 pwsh 把输出写文件、node 再读；Windows 下 git 用绝对路径 `C:\Program Files\Git\cmd\git.exe`。

## 4. 什么情况可以例外

- **研究记录**：结论作废或失败的实验，保留成分支/标签而不是删除，但必须在 issue 里写明"已作废 + 为什么"。
- **证据包**：外部工具（VSPAERO 等）的原始输出体积大、可重复生成，但与代码强相关时仍放同一个 PR 的 `docs/<issue>/` 下，不单独立项。
- **回滚点**：基线快照可以留一条分支，但要在 README 或 issue 里写明它的用途和删除条件。

## 5. 合并前检查清单

- [ ] 分支从 `origin/master` 切出，PR base = `master`。
- [ ] 验收命令在 PR 正文里可复制执行，且贴了实际输出（不是"应该通过"）。
- [ ] 结论区分"实测"与"推测"；未解决项单独列出。
- [ ] `physical_accuracy_accepted` 之类的默认值没有被悄悄改掉。
- [ ] 合并后分支已删（或已打 `archive/` tag）。
