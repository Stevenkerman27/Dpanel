#!/usr/bin/env node
// 出箱式自动发布：主会话只写「文件 + 一个清单」，其余全自动。
// 目的：发布是程序化动作，不需要子智能体、也不需要人盯推送步骤。
//
// 清单 runs/outbox/<name>.json：
//   { "branch": "master",
//     "paths": ["monitor/autopublish.mjs", "monitor/README.md"],
//     "message": "monitor: add autopublish outbox",
//     "dryRun": false }
// paths 相对仓库根；文件必须先存在于工作区，脚本会把它复制进 runs/master_git。
//
// 用法：
//   node monitor/autopublish.mjs --once            处理完当前清单就退出（失败时 exit 1）
//   node monitor/autopublish.mjs --daemon --interval 15   常驻，每 15 秒扫一次出箱
//
// 结果落盘：runs/outbox/results/<name>.json，日志 runs/outbox/logs/<name>.{out,err}.log，
// 成功清单移进 done/，失败移进 failed/（含错误摘要）。
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const OUTBOX = path.join(ROOT, "runs", "outbox");
const DIR = {
  inbox: OUTBOX,
  done: path.join(OUTBOX, "done"),
  failed: path.join(OUTBOX, "failed"),
  logs: path.join(OUTBOX, "logs"),
  results: path.join(OUTBOX, "results"),
  msg: path.join(OUTBOX, "msg"),
};
const WORKTREE = path.join(ROOT, "runs", "master_git");
const PUBLISH = path.join(HERE, "publish.mjs");

function stamp() {
  return new Date().toLocaleString("zh-CN", { hour12: false });
}
function log(msg) {
  console.log("[" + stamp() + "] " + msg);
}
function ensureDirs() {
  for (const d of Object.values(DIR)) fs.mkdirSync(d, { recursive: true });
}
function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}
function writeJson(p, v) {
  fs.writeFileSync(p, JSON.stringify(v, null, 2) + "\n", "utf8");
}
function pending() {
  ensureDirs();
  return fs
    .readdirSync(DIR.inbox)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({ name: f.replace(/\.json$/, ""), file: path.join(DIR.inbox, f) }));
}

// publish.mjs 在子进程里跑；沙箱下 stdio 管道会 EPERM，必须重定向到文件再读回。
function runPublish(name, branch, paths, messagePath) {
  const args = [PUBLISH, WORKTREE, branch, "--paths", paths.join(","), "--message-file", messagePath];
  const out = path.join(DIR.logs, name + ".out.log");
  const err = path.join(DIR.logs, name + ".err.log");
  const fo = fs.openSync(out, "w");
  const fe = fs.openSync(err, "w");
  return new Promise((resolve) => {
    const child = spawn(process.execPath, args, { cwd: ROOT, stdio: ["ignore", fo, fe] });
    child.on("error", (e) => {
      fs.closeSync(fo);
      fs.closeSync(fe);
      resolve({ code: -1, text: String(e && e.message ? e.message : e), sha: null, route: null, verified: false });
    });
    child.on("exit", (code) => {
      fs.closeSync(fo);
      fs.closeSync(fe);
      const text = (fs.existsSync(out) ? fs.readFileSync(out, "utf8") : "") + (fs.existsSync(err) ? fs.readFileSync(err, "utf8") : "");
      const api = /API_PUSH ([0-9a-f]{7,40})->([0-9a-f]{7,40}) (\S+) files=(\d+) verified=(\w+)/.exec(text);
      const git = /PUSHED ([0-9a-f]{7,40}) (\S+)/.exec(text);
      const dry = /DRY_RUN head=(\S+) remote=(\S+)/.exec(text);
      resolve({
        code,
        text,
        sha: api ? api[2] : git ? git[1] : null,
        route: api ? "api" : git ? "git" : dry ? "dry-run" : null,
        verified: api ? api[5] === "true" : Boolean(git),
      });
    });
  });
}

function stageFiles(paths) {
  for (const rel of paths) {
    const src = path.join(ROOT, rel);
    if (!fs.existsSync(src)) throw new Error("missing in workspace: " + rel);
    const dst = path.join(WORKTREE, rel);
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.copyFileSync(src, dst);
  }
}

function messagePath(name, message) {
  const p = path.join(DIR.msg, name + ".txt");
  fs.writeFileSync(p, String(message).replace(/^\uFEFF/, ""), "utf8");
  return p;
}

async function processOne(entry) {
  let manifest;
  try {
    manifest = readJson(entry.file);
  } catch (e) {
    log("SKIP " + entry.name + " (bad json: " + (e && e.message) + ")");
    fs.renameSync(entry.file, path.join(DIR.failed, entry.name + ".json"));
    return false;
  }
  const paths = (manifest.paths || []).map(String);
  const branch = String(manifest.branch || "master");
  const fn = entry.name + ".json";
  if (!paths.length) {
    log("SKIP " + entry.name + " (empty paths)");
    fs.renameSync(entry.file, path.join(DIR.failed, fn));
    return false;
  }
  log("publish " + entry.name + " -> " + branch + " (" + paths.length + " file(s))");
  let msgFile;
  try {
    stageFiles(paths);
    const msg = manifest.message || (manifest.messageFile ? fs.readFileSync(path.join(ROOT, manifest.messageFile), "utf8") : "update " + paths.join(", "));
    msgFile = messagePath(entry.name, msg);
  } catch (e) {
    const reason = String(e && e.message ? e.message : e);
    log("FAIL " + entry.name + ": " + reason);
    writeJson(path.join(DIR.results, entry.name + ".json"), { ok: false, phase: "stage", error: reason, at: new Date().toISOString(), paths });
    fs.renameSync(entry.file, path.join(DIR.failed, fn));
    return false;
  }
  let args = paths;
  if (manifest.dryRun) {
    const dry = path.join(DIR.msg, entry.name + ".dry.txt");
    fs.writeFileSync(dry, "", "utf8");
    const r = await runPublish(entry.name, branch, args, msgFile);
    writeJson(path.join(DIR.results, entry.name + ".json"), { ok: false, phase: "dry-run", route: r.route, at: new Date().toISOString(), paths });
    log("DRY " + entry.name + " -> " + (r.route || "unknown"));
    fs.renameSync(entry.file, path.join(DIR.done, fn));
    return true;
  }
  const r = await runPublish(entry.name, branch, args, msgFile);
  const ok = r.code === 0 && (r.route === "api" || r.route === "git");
  writeJson(path.join(DIR.results, entry.name + ".json"), {
    ok,
    phase: ok ? "published" : "publish",
    sha: r.sha,
    route: r.route,
    verified: r.verified,
    exitCode: r.code,
    at: new Date().toISOString(),
    paths,
    tail: ok ? undefined : String(r.text || "").slice(-600),
  });
  if (ok) {
    log("OK " + entry.name + " " + (r.sha || "").slice(0, 7) + " via " + r.route + " verified=" + r.verified);
    moveManifest(entry.file, path.join(DIR.done, fn));
    return true;
  }
  log("FAIL " + entry.name + " exit=" + r.code + " " + String(r.text || "").slice(-200));
  moveManifest(entry.file, path.join(DIR.failed, fn));
  return false;
}

// 清单可能在处理过程中被另一个实例（守护 + 手动 --once）搬走 —— 那不是错误。
function moveManifest(from, to) {
  try {
    fs.renameSync(from, to);
  } catch (e) {
    if (e.code !== "ENOENT") throw e;
    log("SKIP manifest moved by another instance: " + path.basename(from));
  }
}

// 单实例锁：守护进程与手动 --once 同时跑会各自读到同一清单，
// 两个 publish 都会针对同一个远端 HEAD 建 commit，后到的 PATCH 得到
// 422 "Update is not a fast forward"，看起来像失败其实已经发布成功。
const LOCK_PATH = path.join(OUTBOX, ".autopublish.lock");
function acquireLock() {
  try {
    fs.writeFileSync(LOCK_PATH, String(process.pid), { flag: "wx" });
    return true;
  } catch (e) {
    if (e.code !== "EEXIST") throw e;
    let pid = 0;
    let age = Infinity;
    try {
      pid = Number(fs.readFileSync(LOCK_PATH, "utf8").trim()) || 0;
      age = Date.now() - fs.statSync(LOCK_PATH).mtimeMs;
    } catch {}
    const alive = pid > 0 && age < 600000;
    if (alive) {
      log("SKIP another autopublish holds the lock (pid " + pid + ", " + Math.round(age / 1000) + "s)");
      return false;
    }
    log("STALE lock (pid " + pid + ", " + Math.round(age / 1000) + "s) removed");
    try {
      fs.unlinkSync(LOCK_PATH);
    } catch {}
    return acquireLock();
  }
}
function releaseLock() {
  try {
    fs.unlinkSync(LOCK_PATH);
  } catch {}
}

async function drain() {
  if (!acquireLock()) return { total: 0, ok: 0, skipped: true };
  try {
    const list = pending();
    if (!list.length) return { total: 0, ok: 0 };
    let ok = 0;
    for (const entry of list) {
      if (await processOne(entry)) ok++;
    }
    return { total: list.length, ok };
  } finally {
    releaseLock();
  }
}

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};

if (argv.includes("--once")) {
  const r = await drain();
  log("drain done: " + r.ok + "/" + r.total);
  process.exit(r.total === 0 || r.ok === r.total ? 0 : 1);
} else if (argv.includes("--daemon")) {
  const interval = Math.max(3, Number(arg("--interval", 15)) || 15);
  ensureDirs();
  log("出箱守护启动 pid " + process.pid + "，扫描 " + OUTBOX + "，间隔 " + interval + " 秒");
  for (;;) {
    const r = await drain();
    if (r.total) log("本轮 " + r.ok + "/" + r.total);
    await new Promise((res) => setTimeout(res, interval * 1000));
  }
} else {
  console.error("usage: node monitor/autopublish.mjs --once | --daemon [--interval 15]");
  process.exit(2);
}