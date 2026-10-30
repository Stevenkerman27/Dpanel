import assert from "node:assert/strict";
import { isLocalAction } from "./github_watch.mjs";

const T0 = "2026-10-30T14:20:53Z";
const plus = (min) => new Date(Date.parse(T0) + min * 60000).toISOString();

const ledger = [
  { kind: "api", number: 85, at: T0, path: "/repos/x/issues/85/comments" },
  { kind: "comment", number: 85, commentId: 6525069685, at: T0, path: "/repos/x/issues/85/comments" },
];
const cloudComment = (id, at) => ({ kind: "comment", id: "#" + id, number: 85, at });

// 1. 我自己的评论（commentId 精确相等）属于本人操作。
assert.equal(isLocalAction(cloudComment(6525069685, plus(0)), ledger), true);

// 2. 云端 15 分钟后在同一 PR 上的评论不得被粗粒度记账吞掉（这就是这次漏报的回归测试）。
assert.equal(isLocalAction(cloudComment(6525345926, plus(14.6)), ledger), false);

// 3. 超出 30 分钟窗口后，连我自己的评论也不再认作本人操作。
assert.equal(isLocalAction(cloudComment(6525069685, plus(45)), ledger), false);

// 4. 精确认账不替另一条评论作证，即使编号相同、时间窗内。
assert.equal(isLocalAction(cloudComment(7000000000, plus(1)), [ledger[1]]), false);

// 5. 粗粒度事件（issue 关闭、改标签等）仍按编号 + 窗口匹配，避免每次都把自己叫醒。
assert.equal(isLocalAction({ kind: "issue", id: "#85", at: plus(2) }, ledger), true);
assert.equal(isLocalAction({ kind: "issue", id: "#86", at: plus(2) }, ledger), false);

// 6. push 记账只认 commit 事件。
const pushLedger = [{ kind: "push", at: T0 }];
assert.equal(isLocalAction({ kind: "commit", id: "#abc", at: plus(1) }, pushLedger), true);
assert.equal(isLocalAction(cloudComment(1, plus(1)), pushLedger), false);

// 7. 没有账本时，任何事件都不是本人操作。
assert.equal(isLocalAction(cloudComment(6525345926, plus(1)), []), false);

console.log("self-filter tests: 9 assertions passed");