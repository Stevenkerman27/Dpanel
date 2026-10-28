
目标：鲁棒，快速，方法论清晰的可微气动求解器，方法不限于VLM，全势流等

openvsp，WBCAero已放在项目中供你参考。

git地址为https://github.com/Stevenkerman27/Dpanel.git，使用192.168.3.9:7890代理连github，如果代理下线了不需要尝试debug，告诉用户即可
使用github和云端进行协作，开一个后台任务监控github的更新，包括新的PR, commit, issue, discussions等。注意筛选掉你自己的提交

## 协作流程（GitHub）
- master 是唯一集成点：功能分支从 master 切，PR base 只能是 master，测试通过当天就合并（仓库为 squash-only），合并后立即删分支；要留档就打 `archive/<名字>` tag。
- 一个功能 = 一个分支 = 一个 PR；不叠长链，依赖项先合并进 master 再切下一层。
- 证据/数据文件随代码放进同一个 PR（`docs/<issue>/`、`tools/`），不单开"只有数据"的分支。
- PR 是合并请求，不是交付回执；回执 = issue 里一条评论 + 指向 commit/PR 的链接。
- PR 正文必须写：改了什么、验收命令、实测输出、未解决项。细节见 `docs/WORKFLOW.md`。
