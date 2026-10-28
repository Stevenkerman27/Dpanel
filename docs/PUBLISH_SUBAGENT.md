# 把 GitHub 发布交给子智能体（2026-10-28）

> **现状（2026-10-28 之后）**：发布已自动化 —— 主会话写文件 + `runs/outbox/<name>.json` 清单，守护进程 `monitor/autopublish.mjs --daemon` 自动 stage/发布/校验/记账，见 `monitor/README.md` 的「出箱自动发布」。本文档保留为**手动或回退路径**（守护进程不在、或需要一次性的非常规发布时照此派人）。

为什么要有这份文档：`github.com:443` 的 git 传输和 `api.github.com` 会**独立**故障，
手拼 `-c http.proxy=... -c http.sslBackend=openssl -c http.extraHeader=...` 再反复重试，
每次要花掉主会话十几个步骤。现在发布是一条命令 + 一个低思考子智能体，主会话只读一行回执。

## 1. 发布通道

`monitor/publish.mjs` 一次完成「暂存 → 提交 → 推送 → 校验」：

```powershell
node monitor/publish.mjs <repoDir> <branch> `
  --paths a,b,c --message-file <abs.txt>
```

- 先走 git 传输（保留真实 commit 对象）；失败自动回退 git data API：只为 `--paths` 建 blob
  → tree（`base_tree` = 远端 head 的 tree）→ commit（父节点 = 远端 head）→ PATCH ref → GET ref 校验。
- 输出一行：`PUSHED <sha> <branch> via git` 或
  `API_PUSH <old>-><new> <branch> files=N verified=true`。
- 成功后在 `monitor/state/own_actions.json` 追加 `{kind:'push', refspec}`，监听器据此把这次
  提交标成「本人账号·已核对为我的操作」。
- `--dry-run` 只报本地/远端 HEAD 和将发布的路径。

## 2. 派单模板（直接复制，只改方括号部分）

```text
你是负责 GitHub 发布的子智能体。工作目录 D:\3D\Projects\panel（Windows，shell 用 pwsh）。

任务：[一句话，例如 把 monitor/publish.mjs 发布到 master]。只做这一件事。

步骤：
1. Copy-Item '<源文件>' '<runs/master_git 下的目标路径>' -Force
2. 提交信息写文件（PowerShell 不支持 heredoc）：用 Set-Content -Encoding utf8 写 [msg 路径]
   ⚠ 本机是 PowerShell 5.1，没有 `utf8NoBOM` 枚举；`-Encoding utf8` 会写 BOM，`publish.mjs` 会剥掉，所以直接用它即可。要完全避开 BOM 就用 `[System.IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false)))`。
3. Set-Location 'D:\3D\Projects\panel'; $ErrorActionPreference='Continue'
   node monitor\publish.mjs runs/master_git master --paths [路径] --message-file [msg 路径]
4. 核对：node monitor\gh_api.mjs GET repos/Stevenkerman27/Dpanel/git/ref/heads/[branch]
   返回的 object.sha 必须与第 3 步输出的新 sha 一致。

约束：不碰别的分支，不发 PR/评论，不 force，不用 bash/curl/Start-Process，不改脚本内容。
单条命令可能超过 100 秒时用 run_in_background 再收结果。

报告不超过 6 行：
RESULT: success|failed
SHA: <7 位>
ROUTE: git|api
VERIFIED: yes|no
ERROR: <仅失败时原文前 300 字符>
```

## 3. 主会话（我）只做三件事

1. 把要发布的内容放进 `runs/master_git`（master 的 worktree）并写提交信息文件；
2. 派单（`subagent`，不带 reasoning effort）；
3. 读它回的那 6 行，必要时 `gh_api` 再核一次。

## 4. 边界：子智能体**不能**做这些

- 不能合并 PR（squash-only 仓库，`PUT /pulls/{n}/merge {merge_method:'squash'}`，merge/rebase 都是 405）；
- 不能转正 Draft（`node monitor/pr_ready.mjs Stevenkerman27/Dpanel <n>` 走 GraphQL，REST 的
  `PATCH /pulls/{n} {"draft": false}` 会被静默忽略）；
- 不能删分支/打 tag（顺序必须是**先**建 `refs/tags/archive/<name>` 且断言 201，**再** DELETE 分支，
  2026-10-28 就因为 body 二次编码导致 45 个 tag 全 422 而分支已被删，靠本地 refs 才补回来）；
- 不能改 `monitor/*.mjs` 的内容——它只搬运，不创作。

## 5. 为什么这些细节必须写死

- 沙箱下 node `spawn` 的 `stdio:'pipe'` 直接 `EPERM`，子进程输出只能重定向到文件再读回；
  Windows 上 `spawn` 也不走 `PATHEXT`，git 必须写 `C:/Program Files/Git/cmd/git.exe`；
- `apiJson` 的 body 必须是普通对象，传字符串会二次编码 → 422；
- 代理 `/probe` 返回 `proxy ECONNRESET / direct 200` 时按 AGENTS.md 约定只告知用户，不 debug。

## 6. 已验证的两次派单

| 内容 | 结果 |
| --- | --- |
| `monitor/publish.mjs` + `monitor/README.md` | `API_PUSH ee9c98c->90b2c4f master files=2 verified=true` |
| `publish.mjs` 补账本 | `API_PUSH 90b2c4f->9d6c04e master files=1 verified=true` |
