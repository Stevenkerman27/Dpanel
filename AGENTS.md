目标：鲁棒，快速，方法论清晰的可微气动求解器，方法不限于VLM，全势流等

openvsp，WBCAero已放在项目中供你参考。

git地址为https://github.com/Stevenkerman27/Dpanel.git，使用192.168.3.9:7890代理连github，如果代理下线了不需要尝试debug，告诉用户即可
云端起**指导**作用（方向、优先级、评审），主要开发由本机 agent 完成。不再"完成一项就开 PR/issue"：工作单元做完直接提交并推送；只有需要云端判断、或需要外部复现的问题才开 issue。现有的 PR/issue 定期检查，已完成或已被取代的关闭。

## 协作与项目管理规范

- **分工。** 云端给方向与评审意见，agent 负责实现、实测、文档与提交；讨论结论写进 STATUS.md 或对应文档，不停留在聊天里。
- **Issue 只在需要时开。** 触发条件：物理/数值缺陷需要云端独立判断、依赖外部资源（代理、另一台机器、实验数据）、或用户明确要求。issue 正文必须含现象、最小复现命令、实际输出、期望行为、已排除的假设；修好后由 agent 关闭并写结论。
- **PR 不再逐项开。** 旧的已合并或被取代的 PR 检查后关闭，保留有价值的研究记录。
- **分支纪律。** `master` 保持可复现、可运行的基线；只有需要长期并行研究时才另开分支，并在 STATUS.md 记一行说明假设、下一步和退出条件。
- **提交必须可追溯。** commit message 与对应文档要写实际测试命令、运行环境、结果和已知限制；严格区分代数收敛、数值验证与物理精度验收，未实际执行的测试不得声称通过，失败结果必须保留。
- **主动维护状态。** 每轮工作结束时更新 `STATUS.md`（已完成、当前阻塞、下一步），检查开放 issue/PR 与分支依赖，优先完成可验收的闭环而不是增加新分支。
- **单一环境。** 全项目只维护一个 Python 环境：`C:/Users/zyx20/anaconda3/envs/myml/python.exe`（Python 3.13 + torch 2.13 cu132 + numpy/scipy/meshio/pytest + Gmsh 4.15.2）。OpenVSP 不装包，用 `PYTHONPATH=D:/3D/Projects/OpenVSP-3.55.0-win64/python/openvsp` 指向安装目录。仓库内不再新建 `.conda_env` 之类的环境副本。


## 工程结构规范（目录 / 复现链 / 文档）

> 目录和复现链按**功能**组织。按 issue/PR 组织只适用于 GitHub 上的跟踪，不进入文件系统。

1. **目录按功能分层，不按 issue/PR 命名。** 证据、算例、基准分放在 `cases/`、`benchmarks/`、`studies/`、`verification/`、`scratch/`，用功能词命名（`geometry_robustness`、`vsp_convergence`）；issue 号只出现在文档标题、commit message 和 PR 里。对应的人读索引是 `runs/README.md`。
2. **一份数据只存一份。** 第三方库、Python 绑定、网格、结果不得在仓库里存第二份副本，也不得复制成 `xxx_v2`；引用外部安装用路径。复制前先比 SHA-256，相同就删副本。
3. **复现链 = 脚本 + 一条命令 + 输入。** 可复用的进 `tools/`（带 `--case/--out` 参数，一条 `python -m` 命令可跑完）；一次性的进 `runs/scratch/`，可随时整目录删除；禁止硬编码 worktree 绝对路径（`runs/xxx_git`），用 `Path(__file__)` 定位仓库根。
4. **文档一模块一入口，状态只有一个来源。** 一个模块一篇入口文档；当前状态只写 `STATUS.md`，历史交接进 `docs/archive/`；被取代的文档归档并在 `RETIRED_RESEARCH.md` 记一行，不另开第二份状态页。
5. **文档不承载测试数据。** 测试只读数据文件（`json`/`npz`），不从 Markdown 正文抓关键词当契约——否则一次翻译或改写就会弄坏测试。

## 工具使用约定（省 token、省时间）

一个 run_code 里做完「量 → 做 → 验 → 只打印结论」四件事，不要拆成四次模型往返。

**判据：这一步的结论会不会改变我下一步做什么？**
- 不会 → 合并进上一步。
- 会（要分支、要用户拍板、要换方案）→ 才拆成两步。

1. 单步工具时间控制在 60–90 秒内，超了转后台作业（见下）。
2. 每个 run_code 只打印下一步真正需要的（一行结论 + 必要 id/路径），不要整段回传清单——它会被下一轮重新重读。
3. 写操作先用一条命令验证前置条件（存在性、大小、git 状态），验证与改动可放同一个 run_code，验证失败要提前退出。

### 长任务：后台启动 + 事后收，禁止 job_output(wait) 死等

2. 必须后台的：`run_in_background: true` 起完立刻去做别的独立工作，最后统一收（`job_output` 不带 `wait`，或只给 5–15 秒探一次）。
3. 要看进度的：让作业把日志写进文件，需要时读文件尾部，而不是反复 `job_output`。

### 推送：提权 + 仓库内 token（沙箱内 TLS 后端不可用）

沙箱内 git 连匿名 `git ls-remote` 都失败：schannel 报 `SEC_E_NO_CREDENTIALS`，openssl 后端被报 unavailable；把 `.agents/github_token` 拼进 URL 也不管用（失败发生在 TLS 初始化，不是认证）。可用路径只有一条：

    ops_bash(command="bash tools/push_github.sh", sandbox_permissions="danger-full-access", justification="...")

`tools/push_github.sh` 从 `.agents/github_token`（93 字符 PAT）拼 `https://x-access-token:<tok>@github.com/Stevenkerman27/Dpanel.git` 后直接 `git push <url> HEAD:master`（可传分支名作参数），不写进 git config、不进 remote。token 不要打印、不要进 commit 或文档。`runs/` 被 gitignore，所以推送脚本放 `tools/` 而不是 `runs/scratch_opt/`。

### 每步开工前的自检三问

1. 这一步的结论会改变我下一步吗？不会 → 合并。
2. 我是不是在等一个本来可以稍后再收的东西？是 → 别 wait。
3. 我要打印的东西，下一步真的需要吗？不需要 → 只打印一行。
