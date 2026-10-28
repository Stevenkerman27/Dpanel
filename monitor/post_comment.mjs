#!/usr/bin/env node
// 向 GitHub 发 issue/PR 评论（也可 --delete 删除评论）。
// 传输全部走 monitor/net.mjs：代理/直连自动切换、指数退避重试、路由记忆。
// 用法: node monitor/post_comment.mjs <apiPath> <bodyFile> [--delete <commentId>...]
//   例: node monitor/post_comment.mjs repos/Stevenkerman27/Dpanel/issues/7/comments .tmp/report.md
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { apiRequest } from "./net.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const cfg = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));
const OWN_ACTIONS = path.join(HERE, "state", "own_actions.json");

// 记录本地发出的评论：监听器靠这本账把我自己发的帖子与云端同账号发的区分开。
function recordOwnAction(apiPath) {
  const m = /repos\/[^/]+\/[^/]+\/(?:issues|pulls)\/(\d+)\/comments/.exec(apiPath);
  const entry = { kind: "comment", number: m ? Number(m[1]) : null, at: new Date().toISOString() };
  let list = [];
  try {
    list = JSON.parse(fs.readFileSync(OWN_ACTIONS, "utf8"));
  } catch {
    list = [];
  }
  const cutoff = Date.now() - 3 * 24 * 3600 * 1000;
  list = list.filter((a) => Date.parse(a.at) > cutoff).concat([entry]);
  fs.mkdirSync(path.dirname(OWN_ACTIONS), { recursive: true });
  fs.writeFileSync(OWN_ACTIONS, JSON.stringify(list, null, 2), "utf8");
}

const argv = process.argv.slice(2);
const delIdx = argv.indexOf("--delete");
const positional = delIdx >= 0 ? argv.slice(0, delIdx) : argv;
const delIds = delIdx >= 0 ? argv.slice(delIdx + 1) : [];
const [apiPathRaw, bodyFile] = positional;

for (const id of delIds) {
  const res = await apiRequest("https://api.github.com/repos/" + cfg.repo + "/issues/comments/" + id, { method: "DELETE" });
  console.log("delete", id, "->", res.status, res.route || "");
}

const bodyPath = bodyFile ? path.resolve(ROOT, bodyFile) : null;
if (!apiPathRaw || !bodyPath || !fs.existsSync(bodyPath)) {
  if (delIds.length) process.exit(0);
  console.error("usage: node monitor/post_comment.mjs <apiPath> <bodyFile> [--delete <id>...]");
  process.exit(2);
}

const body = fs.readFileSync(bodyPath, "utf8");
const payload = Buffer.from(JSON.stringify({ body }), "utf8");
const res = await apiRequest("https://api.github.com/" + apiPathRaw.replace(/^\/+/, ""), { method: "POST", body: payload });
if (res.status === 201) {
  let j = {};
  try {
    j = JSON.parse(res.body);
  } catch {}
  recordOwnAction(apiPathRaw);
  console.log("OK", j.html_url, "id:", j.id, "bytes:", Buffer.byteLength(body, "utf8"), "route:", res.route);
  process.exit(0);
}
console.error("FAILED", res.status, res.error || "", String(res.body || "").slice(0, 300));
process.exit(1);
