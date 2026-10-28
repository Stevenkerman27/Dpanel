#!/usr/bin/env node
// 通用 GitHub REST 调用。传输全部走 monitor/net.mjs（代理/直连自动切换 + 重试 + 路由记忆）。
// 用法: node monitor/gh_api.mjs <METHOD> <apiPath> [bodyFile]
// 例: node monitor/gh_api.mjs GET /repos/Stevenkerman27/Dpanel
//     node monitor/gh_api.mjs POST /repos/Stevenkerman27/Dpanel/pulls .tmp/pr.json
// 环境: GH_LIMIT 输出字符上限（默认 4000，完整 JSON 常需 4000000）
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { apiRequest } from "./net.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

const argv = process.argv.slice(2);
const [method, apiPathRaw, bodyFile] = argv;
if (!method || !apiPathRaw) {
  console.error("usage: node monitor/gh_api.mjs <METHOD> <apiPath> [bodyFile]");
  process.exit(2);
}
const apiPath = apiPathRaw.startsWith("/") ? apiPathRaw : "/" + apiPathRaw;
const body = bodyFile ? fs.readFileSync(path.resolve(ROOT, bodyFile), "utf8") : null;
const res = await apiRequest("https://api.github.com" + apiPath, { method: method.toUpperCase(), body });
const LIMIT = Number(process.env.GH_LIMIT || 4000);
console.log("STATUS", res.status, "route", res.route);
console.log(String(res.body || res.error || "").slice(0, LIMIT));
process.exit(res.status >= 200 && res.status < 300 ? 0 : 1);
