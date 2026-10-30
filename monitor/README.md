# GitHub 监听

按 [AGENTS.md](../AGENTS.md) 的要求，后台监听 `Stevenkerman27/Dpanel` 的 commit、PR、issue、discussion，并过滤掉本项目自己的提交（作者邮箱 `zyx20031020@gmail.com` / 账号 `Stevenkerman27`）。

2026-10-25更新：监听请求已改为经 `config.json` 的HTTP CONNECT代理，再建立验证证书的TLS连接。默认代理 `192.168.3.9:7890`；代理失败停止本轮，不尝试直连或排查代理。脚本只读取GitHub，云端协同保持暂停。首次运行只建立基线，不把历史提交当成新事件。

当前后台调度由本地会话心跳任务 `Dpanel GitHub只读监听`（ID `dpanel-github`）每15分钟运行一次 `--once`；无实质更新保持安静。下文常驻进程/DSH唤醒方式属于早期手工运行方案，当前不需要并行启动第二个监听进程。

## 用法

```bash
node monitor/github_watch.mjs --once      # 轮询一次，新事件写入 monitor/state/
node monitor/github_watch.mjs --wait      # 静默轮询（900 秒），发现新事件立即退出并通知 agent
node monitor/github_watch.mjs --loop      # 常驻监听，默认 900 秒一次
node monitor/github_watch.mjs --report    # 输出上次汇报后的未读事件，并标记已读
node monitor/github_watch.mjs --status    # 状态：基线、上次轮询、常驻进程、已见数量、API 余量
node monitor/github_watch.mjs --reset --yes  # 清空状态，下次重建基线
```

## 产出

| 文件 | 内容 |
|---|---|
| `monitor/state/state.json` | 基线、ETag、已见 sha/编号、过滤作者、API 余量 |
| `monitor/state/events.jsonl` | 每条新事件一行 JSON（逐次追加，超过 5000 行截到 3000 行） |
| `monitor/state/latest_report.md` | 最近一次轮询的可读报告 |
| `monitor/state/watcher.json` | 常驻进程 pid 与心跳 |

`monitor/state/` 不入库。ETag 命中 304 时不消耗 GitHub API 额度。默认 15 分钟一轮、三个 REST 端点；未认证额度只有 60 次/小时，因此脚本会读取 `.agents/github_token`（不入库）并带 `Authorization: Bearer`，额度升到 5000 次/小时。

## 过滤规则

- commit：`commit.author.email` 命中 `selfEmails`，或 `author.login` 命中 `selfLogins` 时丢弃。
- PR / issue：保留状态更新并标记本人账号，避免漏掉同账号下的云端PR更新；不会自动发评论。discussion公开页只识别新增讨论编号。
- 已知账号可从提交里自动学习：作者邮箱属于自己、且提交带 GitHub 账号时，把该账号加入过滤表。

## 令牌与"自己的评论"

- 令牌文件：.agents/github_token（已被 .gitignore 的 .agents/ 规则忽略）。脚本读不到或格式可疑时自动退回匿名模式；令牌只进请求头，不写日志、不进报告。
- 写评论需要细粒度令牌对该仓库开 **Issues: Read and write**（PR 评论走 Issues 权限）；只读令牌 POST 会返回 403 Resource not accessible by personal access token。
- **发完评论立刻跑一次 --once**：云端 agent 和我共用同一个账号，评论作者相同，监听器无法区分"我发的"和"云端发的"，不马上吸收就会把自己的评论当成新事件唤醒自己。

### 发评论：用 `monitor/post_comment.mjs`

```powershell
node monitor\post_comment.mjs repos/Stevenkerman27/Dpanel/issues/7/comments .tmp\report.md
node monitor\post_comment.mjs <apiPath> <bodyFile> --delete <commentId>   # 顺手删掉临时评论
```

- 走代理隧道（CONNECT 192.168.3.9:7890）再 TLS 到 api.github.com；代理下线时直连也能用，脚本会打印 `ERR socket hang up`。
- **踩过的坑（重要）**：`https.request` 的 `path` 必须以 `/` 开头。传 `repos/...` 这种不带斜杠的路径会发出畸形请求行，服务器直接断连，现象是 `ERR socket hang up`——
  看起来像"长正文被中断""限流""代理坏了"，其实全是路径少一个斜杠。脚本现在会自动补前导斜杠。
- 发完仍要立刻 `--once` 吸收，理由见上一节。

## 已知限制

- discussions 没有 REST 接口，未认证也没有 GraphQL 额度，只能抓公开页面；仓库当前返回 404（未启用 Discussions），脚本按"未启用"记录并退避 6 小时再试，不报错。`github.com` 页面直连偶发超时，失败后退避 30 分钟。
- 只监听仓库级事件，不展开 issue/PR 评论正文；需要评论时看 `html_url`。
- 2026-10-25实测代理轮询成功：commit、PR、issue与discussion均经代理；未切换直连。更早的直连和代理失败记录不再代表当前实现。
- 未认证额度 60 次/小时，且实测 304 仍然计数。每轮 3 个 REST 请求，15 分钟一轮约 12 次/小时，余量足够；不要在同小时内反复手动 `--once`。

## 运行环境注意（本机实测）

| 事项 | 结果 |
|---|---|
| `ops_bash` 前台命令 | 正常 |
| `ops_bash run_in_background` | 失败：cygwin `bash.exe: fatal error - CreateFileMapping ... Win32 error 5`，且会连带后续 bash 命令一起挂掉 |
| `pwsh run_in_background` | 正常，正是当前常驻监听用的方式 |
| `git ls-remote/push` 走 HTTPS | 失败：schannel `SEC_E_NO_CREDENTIALS`；`http.sslBackend=openssl` 则 `Connection was reset`。沙箱内 git 远端操作不可用，GitHub 交互改用 Node HTTPS API |

因此常驻监听启动命令（在 `D:\3D\Projects\panel`）：

```powershell
& 'C:\Program Files\nodejs\node.exe' 'monitor\github_watch.mjs' '--loop' '--interval' '900'
```

## 唤醒方式：事件驱动，不轮询 agent

当前常驻进程用 `--wait` 模式：它自己每 900 秒检查一次 GitHub，**只有真的出现新事件才退出**（退出码 0，标准输出以 `WAKE: N 条新事件` 开头）。它作为 DSH 后台任务运行，进程退出会触发一次会话通知，于是 agent 被叫醒、读事件、汇报，然后重新挂起等待。

因此没有周期性定时器，没有更新就不会有会话唤醒。代价：

- 唤醒依赖 DSH 后台任务存活。DSH 或宿主机重启后进程消失，不会有任何通知；下次用户发消息时用 `--status` 检查并重启即可。
- 连续 24 次轮询失败（约 6 小时断网）时进程会以退出码 3 结束并打印 `WAKE-ERROR`，同样触发通知，避免静默死亡。

## 网络层（2026-10-28 重写）

代理会周期性 ECONNRESET，而 `api.github.com` 直连往往还通；git 推送还会偶发 `Recv failure: Connection was reset`。传输因此不再各脚本各写一套，统一到两个模块：

| 文件 | 作用 | 用法 |
|---|---|---|
| `monitor/net.mjs` | HTTPS 请求：代理/直连自动交替、指数退避（`min(15s, 0.8*2^i)`）、路由记忆 `state/route.json`、ETag/304、`postComment()` | `node monitor/net.mjs --probe`、`node monitor/net.mjs GET <apiPath>`，或 `import { apiJson } from './net.mjs'` |
| `monitor/gh_push.mjs` | git push/fetch/ls-remote：两条路由（proxy 带 `http.proxy=`，direct 显式清空），`http.sslBackend=openssl`（本机 schannel 报 `SEC_E_NO_CREDENTIALS`），重试 8 次，认证走 `http.extraHeader` | `node monitor/gh_push.mjs fetch runs/integ_git`、`node monitor/gh_push.mjs push runs/x HEAD:branch` |

`github_watch.mjs`、`post_comment.mjs`、`gh_api.mjs` 已全部改为调用 `net.mjs`，不再各自实现 CONNECT 隧道。实测：`--probe` 两条路由都 200；一次 push 中代理 `Recv failure` 后**自动切直连一次成功**；`fetch` 一次拉全 49 个远端分支。

沙箱限制（踩过，写死在这里）：DSH 沙箱禁止 node 用管道捕获子进程输出，`spawn(git)` 默认 `stdio:'pipe'` 直接 EPERM。`gh_push.mjs` 因此把子进程 stdout/stderr 重定向到 `state/git-*.log` 再读回；且 Windows 下 spawn 不走 PATHEXT，必须用 `C:\Program Files\Git\cmd\git.exe` 绝对路径。

### 自身操作账本（`state/own_actions.json`）

云端 agent 与本地用的是**同一个 GitHub 账号**，所以只按 actor 过滤会把云端工作一并吞掉。现在的做法：本地每次通过 `post_comment.mjs` / `gh_api.mjs` 做写操作（POST/PUT/PATCH/DELETE）都记一条 `{kind, number, at}` 到账本；发评论时再额外记一条精确的 `{kind:"comment", number, commentId, at}`（`commentId` 取 API 响应体的 `id`）。监听器把本人账号的事件分成「已核对为我的操作」与「**可能是你手动操作**」两类。

配对规则（2026-10-30 收紧）：

| 事件 | 认作「我的」的条件 |
|---|---|
| `comment`（`id = "#<commentId>"`）| 账本里存在**同一个 commentId**，且 30 分钟内 |
| `issue` / `pull`（`id = "#<number>"`）| 账本里有同号记录，且 30 分钟内 |
| `commit` | 账本里有 `kind:"push"`，且 30 分钟内 |

粗粒度记账（合并、关闭、改标签）**不能替任何一条评论作证**：评论事件只认 commentId。

`gh_api.mjs` 记录 PR 创建/合并/关闭、issue 关闭等（`number` 从路径取）；`gh_push.mjs` 在 push 成功时记一条 `kind: "push"`，监听器把它与 30 分钟内的 commit 事件配对。提交本身**不再按作者丢弃**（同账号，云端也这样推），只是打标签。

## 发布通道：`publish.mjs`（2026-10-28）

`github.com:443`（git 传输）和 `api.github.com` 会**独立**故障：代理正常时 git 也可能
`Recv failure: Connection was reset`，而 REST 一直 200。以前每次推送要手拼 `-c` 参数并
反复重试，现在一条命令：

```powershell
# 只提交本地已有改动并推送（先走 git，失败自动改走 git data API）
node monitor/publish.mjs <repoDir> <branch>

# 指定要发布的文件与提交信息（推荐；API 回退只上传这些文件）
node monitor/publish.mjs runs/master_git master `
  --paths monitor/publish.mjs,monitor/README.md `
  --message-file D:/3D/Projects/panel/runs/msg.txt
```

输出一行机器可读结果：`PUSHED <sha> <branch> via git` 或
`API_PUSH <old>-><new> <branch> files=N verified=true`；`--dry-run` 只报本地/远端
HEAD 与将发布的路径。

三条注意：

- 回退路径**只上传 `--paths` 里的文件**（逐个建 blob），所以给它准确的路径，别留空；空
  `--paths` 时用它推断 `git show --name-only HEAD`。
- API 回退产生的 commit **父节点是远端 HEAD**，与本地 commit 不是同一个对象（内容相同）。
  代理恢复后本地分支需要 `git fetch` 才能和远端对齐。
- 子进程输出必须重定向到文件再读回：沙箱下 node 的 `stdio:'pipe'` 直接 `EPERM`
  （Windows 上 `spawn` 也不走 `PATHEXT`，所以写死 `C:/Program Files/Git/cmd/git.exe`）。

## 唤醒策略（2026-10-28）

`--wait` 只在「**不是我刚做的事**」时退出并唤醒 agent：

- 事件的 actor 是本人账号、且能在 `state/own_actions.json` 里配对上（评论按 **commentId** 精确配对，issue/PR 按编号，推送按 refspec，均限 30 分钟窗口）→ 只记账，打印 `本人操作 N 条（已核对），不唤醒；累计抑制 M 条`，继续轮询。
- 本人账号但账本里没有 → 照样唤醒（可能是你手动在网页上操作的），标签 `（本人账号·可能是你手动操作）`。
- 其他人 → 照样唤醒。
- `--wake-all` 恢复旧行为（连本人已核对也唤醒），用来排查监听器本身。

**为什么不是加延时**：轮询间隔默认 900 s，加 1 s 延时不改变任何东西；真正的问题不是「太快」，而是「这件事我已经知道」。判定依据必须是账本配对（谁做的），不是时间（多近做的）。

### 评论级事件（2026-10-30，修一次真实漏报）

**事故**：2026-10-30 14:20:53Z 我在 PR #85 发回执（记账 `{kind:"api", number:85}`），云端 14:35:28Z 在同一 PR 上给出下一阶段指令。旧实现里 watcher **没有评论级事件**，只比较 issue/PR 的 `updated_at`，生成 `kind:"issue", id:"#85"` 事件；`isLocalAction` 又只按「编号 + 30 分钟窗口」匹配，于是这条云端评论被判成「本人已核对」→ 只记账不唤醒，**并且 seen 已前移**，之后 `--once` 永远报「0 条新事件」。直到用户手动提示才发现。

**修法**：

1. `net.mjs` 的 `recordMutation` 对 `POST /issues/{n}/comments`、`POST /pulls/{n}/comments` 额外记 `{kind:"comment", number, commentId}`；
2. watcher 在 issue/PR 的 `updated_at` 变化时拉 `/issues/{n}/comments?since=…`，为每条**未见过的**评论发一个 `kind:"comment", id:"#<commentId>"` 事件（带作者、正文摘要），并把见过的 id 存在 `state.seen.comments[number]`（每号最多 200 条）；
3. `isLocalAction` 对 `kind:"comment"` 要求 **commentId 精确相等**，粗粒度 `kind:"api"` 记录不得替任何评论作证。

回归测试：`node monitor/test_self_filter.mjs`（9 条断言，含「云端 14.6 分钟后的评论不得被吞」）。

**`post_comment.mjs --delete` 语义**：带 `--delete` 就是删除动作，删完直接退出，不再读 body file 重发（早期版本会「删了又发一条」，2026-10-28 修）。

## 出箱自动发布：autopublish.mjs（2026-10-28）

发布是程序化动作，不再需要子智能体。主会话只做两件事：把文件写进工作区、写一个清单，然后等。

```json
// runs/outbox/<name>.json
{"branch": "master",
 "paths": ["monitor/autopublish.mjs", "monitor/README.md"],
 "message": "monitor: add autopublish outbox"}
```

- `node monitor/autopublish.mjs --once`：处理完当前清单退出（有失败时 exit 1）。
- `node monitor/autopublish.mjs --daemon --interval 15`：常驻，每 15 秒扫一次出箱。
- 每个清单：把 `paths` 从工作区复制进 `runs/master_git` → 调 `publish.mjs`（git 传输失败自动回退 API）→ 校验 → 结果写 `runs/outbox/results/<name>.json`，日志写 `runs/outbox/logs/`，成功清单移进 `done/`、失败移进 `failed/`。
- 发布成功会写 `own_actions.json`（`kind:'push'`），所以监听器不会为这次发布唤醒 agent。
- `"dryRun": true` 只跑 `--dry-run`，用来验证清单路径是否干净。
- 清单里 `paths` 必须相对仓库根且已存在于工作区；`message` 里的 BOM 会被剥掉。
