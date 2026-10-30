#!/usr/bin/env node
/**
 * 监听 Stevenkerman27/Dpanel 的 GitHub 更新：commit / PR / issue / discussions。
 * 零第三方依赖，只用 Node 内置模块；首次运行只建立基线，不把历史当新事件。
 *
 * 用法：
 *   node monitor/github_watch.mjs --once            立即轮询一次
 *   node monitor/github_watch.mjs --wait            静默轮询，发现「非本人已核对」事件立即退出（用于唤醒 agent）
 *   node monitor/github_watch.mjs --wait --wake-all 连本人已核对的事件也唤醒（旧行为）
 *   node monitor/github_watch.mjs --loop            常驻轮询（默认 900 秒）
 *   node monitor/github_watch.mjs --loop --interval 300
 *   node monitor/github_watch.mjs --report          输出上次汇报后的未读事件并标记已读
 *   node monitor/github_watch.mjs --report --peek   只输出，不标记已读
 *   node monitor/github_watch.mjs --status          输出状态（不发请求）
 *   node monitor/github_watch.mjs --reset --yes     清空状态，下次运行重建基线
 */
import fs from "node:fs";
import path from "node:path";
import https from "node:https";
import http from "node:http";
import tls from "node:tls";
import { fileURLToPath } from "node:url";
import { apiRequest } from "./net.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const CONFIG_PATH = path.join(__dirname, "config.json");
const STATE_DIR = path.join(__dirname, "state");
const STATE_PATH = path.join(STATE_DIR, "state.json");
const EVENTS_PATH = path.join(STATE_DIR, "events.jsonl");
const REPORT_PATH = path.join(STATE_DIR, "latest_report.md");
const WATCHER_PATH = path.join(STATE_DIR, "watcher.json");
const LOCK_PATH = path.join(STATE_DIR, "poll.lock");
const LOG_PATH = path.join(STATE_DIR, "watcher.log");

const DEFAULT_CONFIG = {
  repo: "Stevenkerman27/Dpanel",
  intervalSeconds: 900,
  proxy: "http://192.168.3.9:7890",
  selfEmails: ["zyx20031020@gmail.com"],
  selfLogins: ["Stevenkerman27"],
  maxCommits: 300,
  maxThreads: 200,
  tokenFile: ".agents/github_token",
};

const argv = process.argv.slice(2);
const has = (f) => argv.includes(f);
const argValue = (f, d) => {
  const i = argv.indexOf(f);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : d;
};

function loadConfig() {
  let cfg = { ...DEFAULT_CONFIG };
  try {
    cfg = { ...cfg, ...JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8")) };
  } catch {}
  return cfg;
}

let cachedToken;
function authToken(cfg) {
  if (cachedToken !== undefined) return cachedToken;
  cachedToken = "";
  const rel = cfg.tokenFile || DEFAULT_CONFIG.tokenFile;
  if (!rel) return cachedToken;
  try {
    const p = path.isAbsolute(rel) ? rel : path.join(ROOT, rel);
    const t = fs.readFileSync(p, "utf8").trim();
    if (t && !/\s/.test(t)) cachedToken = t;
  } catch {}
  return cachedToken;
}

function ensureStateDir() {
  fs.mkdirSync(STATE_DIR, { recursive: true });
}

function loadState() {
  try {
    return JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
  } catch {
    return null;
  }
}

function emptyState() {
  return {
    version: 1,
    baselineAt: null,
    lastPollAt: null,
    lastReportAt: null,
    etags: {},
    seen: { commits: [], pulls: {}, issues: {}, discussions: {} },
    selfLogins: null,
    selfEmails: null,
    lastError: null,
    failures: 0,
    rate: null,
  };
}

function saveJson(file, obj) {
  ensureStateDir();
  const tmp = file + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2), "utf8");
  fs.renameSync(tmp, file);
}

// Comments and pushes I make locally are logged to this ledger, because the cloud
// agent uses the SAME GitHub account: without it every cloud comment would be
// filtered as "mine", which is exactly how cloud work went unnoticed.
const OWN_ACTIONS_PATH = path.join(STATE_DIR, "own_actions.json");

function loadOwnActions() {
  try {
    return JSON.parse(fs.readFileSync(OWN_ACTIONS_PATH, "utf8"));
  } catch {
    return [];
  }
}

function isLocalAction(ev, actions) {
  const raw = String(ev.id || "").replace("#", "");
  const at = Date.parse(ev.at || ev.ts || "");
  if (!Number.isFinite(at)) return false;
  const isComment = ev.kind === "comment";
  return actions.some((a) => {
    const delta = Math.abs(at - Date.parse(a.at || ""));
    if (!Number.isFinite(delta) || delta > 30 * 60 * 1000) return false;
    if (a.kind === "push") return ev.kind === "commit";
    // 评论事件带精确 commentId：只有账本里同一个 commentId 才能认作我的。
    // 粗粒度 api 记账（合并、关闭、改标签等）不得替任何一条评论作证，否则云端在同一
    // PR/issue 上随后发的评论会被当成「本人已核对」吞掉。
    if (isComment) return a.commentId != null && raw === String(a.commentId);
    if (a.commentId != null) return false;
    return a.number != null && String(a.number) === raw;
  });
}

function appendEvents(events) {
  if (!events.length) return;
  ensureStateDir();
  fs.appendFileSync(EVENTS_PATH, events.map((e) => JSON.stringify(e)).join("\n") + "\n", "utf8");
}

function note(line) {
  console.log(line);
  try {
    ensureStateDir();
    if (fs.existsSync(LOG_PATH) && fs.statSync(LOG_PATH).size > 2000000) fs.writeFileSync(LOG_PATH, "");
    fs.appendFileSync(LOG_PATH, line + "\n", "utf8");
  } catch {}
}

function readEvents() {
  try {
    return fs
      .readFileSync(EVENTS_PATH, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((l) => {
        try {
          return JSON.parse(l);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

function trimEventsFile() {
  const events = readEvents();
  if (events.length <= 5000) return;
  ensureStateDir();
  fs.writeFileSync(EVENTS_PATH, events.slice(-3000).map((e) => JSON.stringify(e)).join("\n") + "\n", "utf8");
}

/* ---------------- HTTP ---------------- */

// 传输统一走 net.mjs：代理/直连自动切换 + 指数退避 + 路由记忆，这里只保留 API 语义。
function request(url, headers, timeoutMs = 20000, attempts = 6) {
  return apiRequest(url, { headers, timeoutMs, attempts });
}

async function apiGet(url, etag, cfg) {
  const headers = {
    "User-Agent": "dpanel-github-watch",
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };
  if (etag) headers["If-None-Match"] = etag;
  const token = authToken(cfg);
  if (token) headers.Authorization = "Bearer " + token;
  return request(url, headers);
}

function rateInfo(headers) {
  const remaining = parseInt(headers["x-ratelimit-remaining"] ?? "-1", 10);
  const reset = parseInt(headers["x-ratelimit-reset"] ?? "0", 10) * 1000;
  return { remaining, reset };
}

function fmtRate(rate) {
  if (!rate) return "";
  const t = rate.reset ? new Date(rate.reset).toLocaleString("zh-CN", { hour12: false }) : "?";
  return "API 剩余 " + rate.remaining + " 次，重置于 " + t;
}

/* ---------------- 采集 ---------------- */

function isSelfCommit(item, cfg, selfEmails, selfLogins) {
  const email = item?.commit?.author?.email || "";
  const login = item?.author?.login || "";
  return (email && selfEmails.includes(email)) || (login && selfLogins.includes(login));
}

function firstLine(s, max = 90) {
  const line = String(s || "").split("\n")[0].trim();
  return line.length > max ? line.slice(0, max - 1) + "…" : line;
}

// 拉取某个 issue/PR 上「上次轮询之后」的评论，返回未见过的那些（自己发的也算，交给 self 过滤）。
// 没有它，watcher 只能看到「issue/PR 有更新」这种粗事件，无法区分云端评论与我自己的评论。
async function fetchNewComments(cfg, state, number, prevUpdatedAt) {
  const key = String(number);
  state.seen.comments = state.seen.comments || {};
  const known = new Set(state.seen.comments[key] || []);
  const prevMs = Date.parse(prevUpdatedAt || "");
  const sinceMs = known.size
    ? Date.now() - 7 * 86400 * 1000
    : Number.isFinite(prevMs)
      ? prevMs
      : Date.now() - 86400 * 1000;
  const url =
    "https://api.github.com/repos/" + cfg.repo + "/issues/" + number +
    "/comments?per_page=100&since=" + encodeURIComponent(new Date(sinceMs).toISOString());
  const res = await apiGet(url, null, cfg);
  if (res.status !== 200) return null;
  let list = [];
  try {
    list = JSON.parse(res.body);
  } catch {
    return null;
  }
  if (!Array.isArray(list)) return null;
  const fresh = list.filter((c) => !known.has(c.id));
  state.seen.comments[key] = list.map((c) => c.id).slice(-200);
  return fresh;
}

async function pushCommentEvents(cfg, state, events, number, prevUpdatedAt, title, fallbackUrl, selfLoginSet) {
  const fresh = await fetchNewComments(cfg, state, number, prevUpdatedAt);
  if (!fresh || !fresh.length) return false;
  for (const c of fresh) {
    events.push({
      kind: "comment",
      id: "#" + c.id,
      number,
      title: firstLine(title),
      author: c.user?.login || "unknown",
      url: c.html_url || fallbackUrl,
      at: c.created_at,
      action: "新评论",
      excerpt: firstLine(String(c.body || "").replace(/\r/g, "").replace(/\n+/g, " "), 160),
      self: selfLoginSet.has(String(c.user?.login || "").toLowerCase()),
      ts: new Date().toISOString(),
    });
  }
  return true;
}

function learnSelfLogins(commits, cfg, selfEmails, selfLogins) {
  const learned = new Set(selfLogins);
  for (const c of commits) {
    const email = c?.commit?.author?.email || "";
    const login = c?.author?.login || "";
    if (email && selfEmails.includes(email) && login) learned.add(login);
  }
  return [...learned];
}

async function pollOnce(state, cfg) {
  const selfEmails = (state.selfEmails?.length ? state.selfEmails : cfg.selfEmails).map((s) => s.toLowerCase());
  let selfLogins = (state.selfLogins?.length ? state.selfLogins : cfg.selfLogins).slice();
  const selfLoginSet = new Set(selfLogins.map((s) => s.toLowerCase()));
  const base = "https://api.github.com/repos/" + cfg.repo;
  const events = [];
  const statuses = [];
  const isBaseline = !state.baselineAt;

  // commits
  const commitsRes = await apiGet(base + "/commits?per_page=50", state.etags.commits, cfg);
  if (commitsRes.error) throw new Error("请求失败（代理与直连均不可用）：" + commitsRes.error);
  let commits = [];
  if (commitsRes.status === 200) {
    commits = JSON.parse(commitsRes.body);
    state.etags.commits = commitsRes.headers.etag;
    selfLogins = learnSelfLogins(commits, cfg, selfEmails, selfLogins);
    selfLogins.forEach((l) => selfLoginSet.add(l.toLowerCase()));
  } else if (commitsRes.status === 304) {
    statuses.push("commits 304 未变化");
  } else {
    statuses.push("commits 失败: HTTP " + commitsRes.status + (commitsRes.error ? " " + commitsRes.error : ""));
  }

  const seenCommits = new Set(state.seen.commits);
  if (commitsRes.status === 200) {
    for (const c of commits) {
      if (seenCommits.has(c.sha)) continue;
      seenCommits.add(c.sha);
      if (isBaseline) continue;
      // My own commits are reported too, flagged rather than dropped: the cloud
      // agent pushes with this same account, so dropping by author would hide
      // exactly the work the ledger is meant to distinguish.
      events.push({
        kind: "commit",
        id: c.sha.slice(0, 7),
        title: firstLine(c.commit?.message),
        author: c.author?.login || c.commit?.author?.name || "unknown",
        url: c.html_url,
        at: c.commit?.author?.date || new Date().toISOString(),
        action: "新提交",
        self: isSelfCommit(c, cfg, selfEmails, selfLogins),
        ts: new Date().toISOString(),
      });
    }
    state.seen.commits = [...seenCommits].slice(0, cfg.maxCommits);
  }

  // pulls
  const pullsRes = await apiGet(base + "/pulls?state=all&sort=updated&direction=desc&per_page=30", state.etags.pulls, cfg);
  if (pullsRes.status === 200) {
    state.etags.pulls = pullsRes.headers.etag;
    const pulls = JSON.parse(pullsRes.body);
    for (const p of pulls) {
      const key = String(p.number);
      const prev = state.seen.pulls[key];
      const bySelf = selfLoginSet.has(String(p.user?.login || "").toLowerCase());
      if (!prev) {
        if (!isBaseline) {
          events.push({
            kind: "pull",
            id: "#" + p.number,
            title: firstLine(p.title),
            author: p.user?.login || "unknown",
            url: p.html_url,
            at: p.created_at,
            action: "新 PR" + (p.draft ? "（草稿）" : ""),
            self: bySelf,
            ts: new Date().toISOString(),
          });
        }
        state.seen.pulls[key] = { updated_at: p.updated_at, state: p.state, merged_at: p.merged_at, title: firstLine(p.title) };
      } else if (prev.updated_at !== p.updated_at || prev.state !== p.state) {
        if (!isBaseline) {
          const commented = await pushCommentEvents(cfg, state, events, p.number, prev.updated_at, p.title, p.html_url, selfLoginSet);
          const act = p.merged_at ? "PR 已合并" : p.state === "closed" ? "PR 已关闭" : "PR 有更新";
          if (!commented) events.push({
            kind: "pull",
            id: "#" + p.number,
            title: firstLine(p.title),
            author: p.user?.login || "unknown",
            url: p.html_url,
            at: p.updated_at,
            action: act,
            self: bySelf,
            ts: new Date().toISOString(),
          });
        }
        state.seen.pulls[key] = { updated_at: p.updated_at, state: p.state, merged_at: p.merged_at, title: firstLine(p.title) };
      }
    }
    statuses.push("PR " + pulls.length + " 条");
  } else if (pullsRes.status === 304) {
    statuses.push("PR 304 未变化");
  } else {
    statuses.push("PR 失败: HTTP " + pullsRes.status + (pullsRes.error ? " " + pullsRes.error : ""));
  }

  // issues（排除 PR）
  const issuesRes = await apiGet(base + "/issues?state=all&sort=updated&direction=desc&per_page=30", state.etags.issues, cfg);
  if (issuesRes.status === 200) {
    state.etags.issues = issuesRes.headers.etag;
    const issues = JSON.parse(issuesRes.body).filter((i) => !i.pull_request);
    for (const i of issues) {
      const key = String(i.number);
      const prev = state.seen.issues[key];
      const bySelf = selfLoginSet.has(String(i.user?.login || "").toLowerCase());
      if (!prev) {
        if (!isBaseline) {
          events.push({
            kind: "issue",
            id: "#" + i.number,
            title: firstLine(i.title),
            author: i.user?.login || "unknown",
            url: i.html_url,
            at: i.created_at,
            action: "新 issue",
            self: bySelf,
            ts: new Date().toISOString(),
          });
        }
        state.seen.issues[key] = { updated_at: i.updated_at, state: i.state, title: firstLine(i.title) };
      } else if (prev.updated_at !== i.updated_at || prev.state !== i.state) {
        if (!isBaseline) {
          const commented = await pushCommentEvents(cfg, state, events, i.number, prev.updated_at, i.title, i.html_url, selfLoginSet);
          if (!commented) {
            events.push({
              kind: "issue",
              id: "#" + i.number,
              title: firstLine(i.title),
              author: i.user?.login || "unknown",
              url: i.html_url,
              at: i.updated_at,
              action: i.state === "closed" ? "issue 已关闭" : "issue 有更新",
              self: bySelf,
              ts: new Date().toISOString(),
            });
          }
        }
        state.seen.issues[key] = { updated_at: i.updated_at, state: i.state, title: firstLine(i.title) };
      }
    }
    statuses.push("issue " + issues.length + " 条");
  } else if (issuesRes.status === 304) {
    statuses.push("issue 304 未变化");
  } else {
    statuses.push("issue 失败: HTTP " + issuesRes.status + (issuesRes.error ? " " + issuesRes.error : ""));
  }

  // discussions：REST 不提供，抓取公开页面；未启用时 404。失败后退避，避免每次轮询都等超时。
  const discUrl = "https://github.com/" + cfg.repo + "/discussions";
  const dstat = state.discussionsStatus || {};
  const retryAt = dstat.retryAt ? Date.parse(dstat.retryAt) : 0;
  if (retryAt > Date.now()) {
    statuses.push("discussion 暂缓（" + dstat.reason + "，重试于 " + new Date(retryAt).toLocaleString("zh-CN", { hour12: false }) + "）");
  } else {
  const discRes = await request(discUrl, { "User-Agent": "dpanel-github-watch", Accept: "text/html" }, 12000);
  if (discRes.status === 200) {
    const re = new RegExp('href="/' + cfg.repo + '/discussions/(\\d+)"[^>]*>\\s*([^<]{1,160})', "g");
    const found = new Map();
    let m;
    while ((m = re.exec(discRes.body))) found.set(m[1], m[2].trim());
    for (const [num, title] of found) {
      if (!state.seen.discussions[num]) {
        if (!isBaseline) {
          events.push({
            kind: "discussion",
            id: "#" + num,
            title: firstLine(title),
            author: "",
            url: discUrl + "/" + num,
            at: new Date().toISOString(),
            action: "新 discussion",
            ts: new Date().toISOString(),
          });
        }
        state.seen.discussions[num] = { title: firstLine(title) };
      }
    }
    const keys = Object.keys(state.seen.discussions);
    if (keys.length > cfg.maxThreads) {
      const keep = keys.slice(-cfg.maxThreads);
      state.seen.discussions = Object.fromEntries(keep.map((k) => [k, state.seen.discussions[k]]));
    }
    statuses.push("discussion " + found.size + " 条");
    state.discussionsStatus = { reason: "ok", retryAt: null, checkedAt: new Date().toISOString() };
  } else if (discRes.status === 404) {
    statuses.push("discussion 未启用（404）");
    state.discussionsStatus = { reason: "未启用", retryAt: new Date(Date.now() + 21600000).toISOString(), checkedAt: new Date().toISOString() };
  } else {
    statuses.push("discussion 失败: " + (discRes.error || "HTTP " + discRes.status));
    state.discussionsStatus = { reason: "网络失败", retryAt: new Date(Date.now() + 1800000).toISOString(), checkedAt: new Date().toISOString() };
  }
  }

  if (commitsRes.headers["x-ratelimit-remaining"]) state.rate = rateInfo(commitsRes.headers);
  state.selfEmails = selfEmails;
  state.selfLogins = selfLogins;
  state.meta = { repo: cfg.repo, home: "https://github.com/" + cfg.repo };
  return { events, statuses, isBaseline, commitsCount: commits.length };
}

/* ---------------- 报告 ---------------- */

const KIND_LABEL = { comment: "评论", commit: "提交", pull: "PR", issue: "Issue", discussion: "Discussion" };

function renderEvents(events) {
  if (!events.length) return "无新事件。";
  const groups = new Map();
  for (const e of events) {
    if (!groups.has(e.kind)) groups.set(e.kind, []);
    groups.get(e.kind).push(e);
  }
  const lines = [];
  for (const kind of ["comment", "pull", "issue", "commit", "discussion"]) {
    const list = groups.get(kind);
    if (!list) continue;
    lines.push("#### " + (KIND_LABEL[kind] || kind) + "（" + list.length + "）");
    for (const e of list) {
      const author = e.author ? " @" + e.author : "";
      const tag = e.self ? (e.localAction ? "（本人账号·已核对为我的操作）" : "（本人账号·可能是你手动操作）") : "";
      const excerpt = e.excerpt ? " ｜ " + e.excerpt : "";
      lines.push("- " + e.action + " " + e.id + " " + e.title + author + tag + excerpt + " — " + e.url);
    }
    lines.push("");
  }
  return lines.join("\n");
}

function pollGuard() {
  ensureStateDir();
  try {
    const lock = JSON.parse(fs.readFileSync(LOCK_PATH, "utf8"));
    if (lock && lock.pid && Date.now() - lock.at < 600000) return false;
  } catch {}
  fs.writeFileSync(LOCK_PATH, JSON.stringify({ pid: process.pid, at: Date.now() }), "utf8");
  return true;
}

function releaseGuard() {
  try {
    fs.unlinkSync(LOCK_PATH);
  } catch {}
}

function writeReport(events, statuses, extra) {
  ensureStateDir();
  const stamp = new Date().toLocaleString("zh-CN", { hour12: false });
  const body = [
    "# GitHub 监听报告 " + stamp,
    "",
    "仓库：https://github.com/" + loadConfig().repo,
    "状态：" + statuses.join("；") + (extra ? "；" + extra : ""),
    "",
    renderEvents(events),
  ].join("\n");
  fs.writeFileSync(REPORT_PATH, body + "\n", "utf8");
  return body;
}

/* ---------------- 主流程 ---------------- */

async function runOnce(cfg, { quiet = false } = {}) {
  if (!pollGuard()) {
    console.log("跳过：另一个轮询正在进行（" + LOCK_PATH + "）。");
    return { skipped: true, events: [] };
  }
  let state = loadState() || emptyState();
  const started = new Date().toISOString();
  try {
    const { events: rawEvents, statuses, isBaseline, commitsCount } = await pollOnce(state, cfg);
    const ownActions = loadOwnActions();
    // 不再按 actor 静默丢弃：同账号事件照常上报，只用本机 action 账本标注"是否我的操作"。
    const events = rawEvents.map((e) => (e.self ? { ...e, localAction: isLocalAction(e, ownActions) } : e));
    const mineEvents = events.filter((e) => e.self);
    const mineVerified = mineEvents.filter((e) => e.localAction).length;
    state.lastPollAt = started;
    state.failures = 0;
    state.lastError = null;
    if (!state.baselineAt) state.baselineAt = started;
    if (isBaseline) {
      state.lastReportAt = started;
    } else {
      appendEvents(events);
    }
    saveJson(STATE_PATH, state);
    writeReport(events, statuses, isBaseline ? "首次运行，已建立基线" : "");
    trimEventsFile();
    if (!quiet) {
      const head = isBaseline
        ? "[基线] 首次运行，记录当前 " + commitsCount + " 条最近提交，不视为新事件。"
        : "[轮询] " + new Date().toLocaleString("zh-CN", { hour12: false }) + " 新事件 " + events.length + " 条" + (mineEvents.length ? "（本人账号 " + mineEvents.length + " 条，其中 " + mineVerified + " 条已与本地操作核对，其余可能是你手动操作）" : "");
      note(head);
      note("  " + statuses.join("；") + (state.rate ? "；" + fmtRate(state.rate) : ""));
      if (events.length) note("\n" + renderEvents(events));
    }
    return { isBaseline, events, statuses };
  } catch (err) {
    state.failures = (state.failures || 0) + 1;
    state.lastError = String(err && err.message ? err.message : err);
    saveJson(STATE_PATH, state);
    console.error("轮询失败：" + state.lastError);
    return { error: state.lastError, events: [] };
  } finally {
    releaseGuard();
  }
}

function printStatus() {
  const state = loadState();
  if (!state) {
    console.log("尚未建立状态（还没跑过）。");
    return;
  }
  let watcher = null;
  try {
    watcher = JSON.parse(fs.readFileSync(WATCHER_PATH, "utf8"));
  } catch {}
  const alive = watcher && Date.now() - watcher.heartbeatAt < 3 * 1000 * (watcher.intervalSeconds || 900);
  console.log("仓库：" + (state.meta?.repo || loadConfig().repo));
  console.log("基线：" + (state.baselineAt || "无"));
  console.log("上次轮询：" + (state.lastPollAt || "无"));
  console.log("上次汇报：" + (state.lastReportAt || "无"));
  console.log(
    "常驻监听：" +
      (alive
        ? "运行中 pid " + watcher.pid + "（" + (watcher.mode === "wait" ? "事件唤醒" : "常驻轮询") + "），心跳 " +
          new Date(watcher.heartbeatAt).toLocaleString("zh-CN", { hour12: false })
        : "未运行")
  );
  console.log("已见：commit " + state.seen.commits.length + "，PR " + Object.keys(state.seen.pulls).length + "，issue " + Object.keys(state.seen.issues).length + "，discussion " + Object.keys(state.seen.discussions).length);
  console.log("过滤作者：" + (state.selfLogins || []).join(", ") + " / " + (state.selfEmails || []).join(", "));
  if (state.rate) console.log(fmtRate(state.rate));
  if (state.lastError) console.log("最近错误：" + state.lastError + "（连续失败 " + state.failures + " 次）");
}

async function report(opts) {
  const state = loadState();
  const events = readEvents().filter((e) => !state?.lastReportAt || e.ts > state.lastReportAt);
  const text = renderEvents(events);
  console.log(text);
  if (!events.length) return;
  if (!opts.peek && state) {
    state.lastReportAt = new Date().toISOString();
    saveJson(STATE_PATH, state);
  }
}

async function waitForEvents(cfg, intervalSeconds) {
  ensureStateDir();
  const watcher = { pid: process.pid, startedAt: Date.now(), heartbeatAt: Date.now(), intervalSeconds, mode: "wait" };
  saveJson(WATCHER_PATH, watcher);
  note("[等待] 监听启动 pid " + process.pid + "，间隔 " + intervalSeconds + " 秒；发现新事件即退出并通知。");
  const stop = () => {
    try {
      fs.unlinkSync(WATCHER_PATH);
    } catch {}
    note("监听已停止。");
    process.exit(0);
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  let failures = 0;
  let suppressedCount = 0;
  for (;;) {
    const res = await runOnce(cfg, { quiet: true });
    watcher.heartbeatAt = Date.now();
    saveJson(WATCHER_PATH, watcher);
    if (res.error) {
      failures++;
      note("[等待] " + new Date().toLocaleString("zh-CN", { hour12: false }) + " 轮询失败（连续 " + failures + " 次）：" + res.error);
      if (failures >= 24) {
        console.log("WAKE-ERROR: 连续 24 次轮询失败，需要人工检查：" + res.error);
        process.exit(3);
      }
    } else {
      failures = 0;
      note("[等待] " + new Date().toLocaleString("zh-CN", { hour12: false }) + " 轮询完成，新事件 " + (res.events?.length || 0) + " 条");
    }
    if (res.events && res.events.length) {
      // 只唤醒「不是我刚做的事」：本人账号且已在本机账本里核对过的事件只记账，
      // 不打断会话（否则每次发评论/推送都会把自己叫醒一次）。
      const wakeAll = process.argv.includes("--wake-all");
      const unverified = wakeAll ? res.events : res.events.filter((e) => !(e.self && e.localAction));
      if (!unverified.length) {
        suppressedCount += res.events.length;
        note("[等待] " + new Date().toLocaleString("zh-CN", { hour12: false }) + " 本人操作 " + res.events.length + " 条（已核对），不唤醒；累计抑制 " + suppressedCount + " 条");
      } else {
        const skipped = res.events.length - unverified.length;
        console.log("WAKE: " + unverified.length + " 条新事件 " + new Date().toLocaleString("zh-CN", { hour12: false })
          + (skipped ? "（另有本人已核对 " + skipped + " 条不重复通知）" : ""));
        console.log(renderEvents(unverified));
        process.exit(0);
      }
    }
    await new Promise((r) => setTimeout(r, Math.max(60, intervalSeconds) * 1000));
  }
}

async function loop(cfg, intervalSeconds) {
  ensureStateDir();
  const watcher = { pid: process.pid, startedAt: Date.now(), heartbeatAt: Date.now(), intervalSeconds };
  saveJson(WATCHER_PATH, watcher);
  note("常驻监听启动 pid " + process.pid + "，间隔 " + intervalSeconds + " 秒，仓库 " + cfg.repo);
  const stop = () => {
    try {
      fs.unlinkSync(WATCHER_PATH);
    } catch {}
    note("监听已停止。");
    process.exit(0);
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  for (;;) {
    const res = await runOnce(cfg);
    watcher.heartbeatAt = Date.now();
    saveJson(WATCHER_PATH, watcher);
    if (res.error && res.error.includes("403")) {
      note("触发 API 限流，等待重置后再试。");
    }
    await new Promise((r) => setTimeout(r, Math.max(60, intervalSeconds) * 1000));
  }
}

export { isLocalAction };

async function main() {
  const cfg = loadConfig();
  ensureStateDir();
  if (has("--reset")) {
    if (!has("--yes")) {
      console.log("清空状态需要同时加 --yes。");
      return;
    }
    for (const f of [STATE_PATH, EVENTS_PATH, REPORT_PATH, LOCK_PATH]) {
      try {
        fs.unlinkSync(f);
      } catch {}
    }
    console.log("状态已清空，下次运行会重建基线。");
    return;
  }
  if (has("--status")) return printStatus();
  if (has("--report")) return report({ peek: has("--peek") });
  if (has("--wait")) return waitForEvents(cfg, Number(argValue("--interval", cfg.intervalSeconds)) || cfg.intervalSeconds);
  if (has("--loop")) return loop(cfg, Number(argValue("--interval", cfg.intervalSeconds)) || cfg.intervalSeconds);
  const res = await runOnce(cfg);
  if (res.error) process.exitCode = 2;
}

const isMainModule = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMainModule) main();
