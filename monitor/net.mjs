#!/usr/bin/env node
// 统一 GitHub 传输层：代理隧道 / 直连自动切换 + 重试退避 + 路由记忆。
// 代理（config.json 的 proxy）会周期性地 ECONNRESET，而 api.github.com 直连往往还是通的；
// 反过来白天代理也常常更快。所以这里不猜，直接两条路都试，并把「上次成功的那条」记在
// monitor/state/route.json 里，下次先用它，避免每次都先撞一遍坏路。
//
// 用法:
//   node monitor/net.mjs --probe                     探针：两条路各打一次 /rate_limit
//   node monitor/net.mjs GET  <apiPath> [outFile]    读 JSON
//   node monitor/net.mjs POST <apiPath> <bodyFile>   发 JSON（bodyFile 可为 .json 或纯文本正文）
//
// 作为库使用（monitor/*.mjs 都走这里）:
//   import { apiRequest, apiJson, postComment, probe } from "./net.mjs";
import https from "node:https";
import http from "node:http";
import net from "node:net";
import tls from "node:tls";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, "..");
const cfg = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));
const ROUTE_FILE = path.join(HERE, "state", "route.json");
const TOKEN_FILE = path.join(ROOT, ".agents", "github_token");

const ROUTES = ["proxy", "direct"];
const DEFAULT_TIMEOUT = 25000;

function token() {
  try {
    return fs.readFileSync(TOKEN_FILE, "utf8").trim();
  } catch {
    return "";
  }
}

function readRouteState() {
  try {
    return JSON.parse(fs.readFileSync(ROUTE_FILE, "utf8"));
  } catch {
    return { preferred: "proxy", ok: {}, fail: {} };
  }
}

function writeRouteState(s) {
  try {
    fs.mkdirSync(path.dirname(ROUTE_FILE), { recursive: true });
    fs.writeFileSync(ROUTE_FILE, JSON.stringify(s, null, 2), "utf8");
  } catch {
    /* 记不住就算了，不影响请求 */
  }
}

function noteResult(route, ok) {
  const s = readRouteState();
  const now = Date.now();
  if (ok) {
    s.ok[route] = now;
    s.preferred = route; // 最近成功的那条优先
  } else {
    s.fail[route] = now;
  }
  writeRouteState(s);
}

function routeOrder() {
  const s = readRouteState();
  const first = ROUTES.includes(s.preferred) ? s.preferred : "proxy";
  return [first, ...ROUTES.filter((r) => r !== first)];
}

function proxyAgent(target) {
  const proxy = new URL(cfg.proxy || "http://192.168.3.9:7890");
  const agent = new https.Agent({ keepAlive: false });
  agent.createConnection = (_options, callback) => {
    const connect = http.request({
      hostname: proxy.hostname,
      port: Number(proxy.port || 80),
      method: "CONNECT",
      path: target.hostname + ":443",
      timeout: DEFAULT_TIMEOUT,
    });
    connect.on("connect", (res, socket, head) => {
      if (res.statusCode !== 200) {
        socket.destroy();
        callback(new Error("proxy CONNECT HTTP " + res.statusCode));
        return;
      }
      if (head.length) socket.unshift(head);
      callback(null, tls.connect({ socket, servername: target.hostname }));
    });
    connect.on("timeout", () => connect.destroy(new Error("proxy timeout")));
    connect.on("error", (error) => callback(error));
    connect.end();
  };
  return agent;
}

function once(url, { method = "GET", headers = {}, body = null, timeoutMs = DEFAULT_TIMEOUT, route }) {
  return new Promise((resolve) => {
    const target = new URL(url);
    const opts = { method, headers, timeout: timeoutMs };
    if (route === "proxy") opts.agent = proxyAgent(target);
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      resolve(v);
    };
    const req = https.request(url, opts, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () =>
        done({ status: res.statusCode || 0, headers: res.headers, body: Buffer.concat(chunks).toString("utf8"), error: null, route })
      );
    });
    req.on("timeout", () => req.destroy(new Error("timeout")));
    req.on("error", (err) => done({ status: 0, headers: {}, body: "", error: err.code || err.message, route }));
    if (body) req.write(body);
    req.end();
  });
}

const TRANSIENT = /ECONNRESET|ETIMEDOUT|ECONNREFUSED|EPIPE|EHOSTUNREACH|ENETUNREACH|EAI_AGAIN|socket hang up|timeout|proxy CONNECT/i;

// 每个请求最多尝试 CONTENT 次，路由交替；网络类错误才重试（4xx/5xx 直接返回给调用方）。
export async function apiRequest(url, { method = "GET", headers = {}, body = null, timeoutMs = DEFAULT_TIMEOUT, attempts = 6 } = {}) {
  const auth = token();
  const base = {
    "User-Agent": "dpanel-agent",
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    ...headers,
  };
  if (auth) base.Authorization = "Bearer " + auth;
  if (body) {
    base["Content-Type"] = base["Content-Type"] || "application/json";
    base["Content-Length"] = Buffer.byteLength(body);
  }
  let last = { status: 0, error: "no attempt", body: "", headers: {} };
  const order = routeOrder();
  for (let i = 0; i < attempts; i++) {
    const route = order[i % order.length];
    last = await once(url, { method, headers: base, body, timeoutMs, route });
    if (!last.error) {
      noteResult(route, true);
      return last;
    }
    noteResult(route, false);
    if (!TRANSIENT.test(last.error)) return last;
    const wait = Math.min(15000, 800 * 2 ** Math.min(i, 4)) + Math.floor(Math.random() * 400);
    await new Promise((r) => setTimeout(r, wait));
  }
  return last;
}

// 记录经由本层发出的写操作，供监听器区分「我的操作」与「云端同账号的操作」。
// 放在 apiJson 里而不是各脚本里：任何调用方（含临时脚本）直接 apiJson 写 GitHub 都会被记上，
// 否则监听器会把自己刚做的合并/改引用当成「可能是你手动操作」把自己叫醒。
function recordMutation(apiPath, method, json) {
  if (method === "GET" || method === "HEAD") return;
  const at = new Date().toISOString();
  const m = /(?:pulls|issues)\/(\d+)/.exec(apiPath);
  let number = m ? Number(m[1]) : null;
  if (number == null && json && Number.isFinite(json.number)) number = Number(json.number);
  const entries = [{ kind: "api", number, at, path: apiPath }];
  // merge 与 git/refs 写入会产生新 commit：同时记一条 push，让监听器把 commit 事件也认作我的。
  if (/\/merge$/.test(apiPath) || /\/git\/refs/.test(apiPath)) entries.push({ kind: "push", refspec: apiPath, at });
  try {
    const ledger = path.join(HERE, "state", "own_actions.json");
    let list = [];
    try {
      list = JSON.parse(fs.readFileSync(ledger, "utf8"));
    } catch {}
    const cutoff = Date.now() - 3 * 24 * 3600 * 1000;
    list = list.filter((a) => Date.parse(a.at) > cutoff).concat(entries);
    fs.mkdirSync(path.dirname(ledger), { recursive: true });
    fs.writeFileSync(ledger, JSON.stringify(list, null, 2), "utf8");
  } catch {}
}
export async function apiJson(apiPath, { method = "GET", body = null, etag = null, attempts = 6 } = {}) {
  const p = apiPath.startsWith("/") ? apiPath : "/" + apiPath;
  const headers = etag ? { "If-None-Match": etag } : {};
  const payload = body === null ? null : Buffer.from(JSON.stringify(body), "utf8");
  const res = await apiRequest("https://api.github.com" + p, { method, headers, body: payload, attempts });
  let json = null;
  if (res.body && res.status !== 304) {
    try {
      json = JSON.parse(res.body);
    } catch {
      json = null;
    }
  }
  if (res.status >= 200 && res.status < 300) recordMutation(p, method.toUpperCase(), json);
  return { ...res, json };
}

// 发 issue/PR 评论：成功返回 {ok, id, url}。正文按 UTF-8 计数。
export async function postComment(apiPath, text) {
  const res = await apiJson(apiPath, { method: "POST", body: { body: text }, attempts: 8 });
  if (res.status === 201 && res.json) return { ok: true, id: res.json.id, url: res.json.html_url, route: res.route };
  return { ok: false, status: res.status, error: res.error || (res.body || "").slice(0, 200), route: res.route };
}

export async function probe() {
  const out = [];
  for (const route of ROUTES) {
    const r = await once("https://api.github.com/rate_limit", {
      headers: { "User-Agent": "dpanel-agent", Accept: "application/vnd.github+json" },
      timeoutMs: 12000,
      route,
    });
    out.push({ route, status: r.status, error: r.error });
    noteResult(route, !r.error && r.status > 0);
  }
  return out;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const argv = process.argv.slice(2);
  if (argv[0] === "--probe" || argv.length === 0) {
    for (const r of await probe()) console.log(r.route.padEnd(7), r.error ? "FAIL " + r.error : "HTTP " + r.status);
    console.log("preferred:", readRouteState().preferred);
  } else {
    const [method, apiPath, file] = argv;
    let body = null;
    if (file) {
      const raw = fs.readFileSync(path.resolve(ROOT, file), "utf8");
      try {
        const j = JSON.parse(raw);
        body = j && typeof j === "object" && "body" in j ? j : j;
      } catch {
        body = { body: raw };
      }
    }
    const res = await apiJson(apiPath, { method: method.toUpperCase(), body });
    console.log("STATUS", res.status, "route", res.route);
    console.log((res.body || "").slice(0, 4000));
  }
}/**
 * 发一个 GitHub REST 请求。注意：`body` 必须是**普通对象**（内部只 JSON.stringify 一次）；
 * 传字符串会被二次编码，GitHub 回 422 `\"{...}\" is not an object`。
 * （2026-10-28 踩过：创建 tag 的 45 次调用全 422，而 DELETE 不带 body 因此照常成功。）
 */

