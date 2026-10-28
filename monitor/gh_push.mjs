#!/usr/bin/env node
// git 传输层：代理/直连交替 + 重试退避 + 记住上次成功的路由。
// 本机两个坑都编码在这里，别再手工拼 -c 参数：
//   1) schannel 后端在本机拿不到凭据（SEC_E_NO_CREDENTIALS），必须 http.sslBackend=openssl；
//   2) 走代理时若进程环境里残留 http_proxy 或全局配置有代理，直连会失败，所以直连分支显式 http.proxy=。
//
// 用法:
//   node monitor/gh_push.mjs push  <repoDir> <refspec>         例: push runs/integ_git HEAD:branch
//   node monitor/gh_push.mjs fetch <repoDir> [refspec...]      默认 --prune origin +refs/heads/*:refs/remotes/origin/*
//   node monitor/gh_push.mjs ls-remote <repoDir>
// 环境: GH_GIT_ATTEMPTS（默认 8）、GH_GIT_QUIET=1
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const cfg = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));
const STATE = path.join(HERE, "state", "gitroute.json");
const REMOTE = cfg.remote || "https://github.com/" + cfg.repo + ".git";
const ATTEMPTS = Number(process.env.GH_GIT_ATTEMPTS || 8);

function token() {
  try {
    return fs.readFileSync(path.join(ROOT, ".agents", "github_token"), "utf8").trim();
  } catch {
    return "";
  }
}

function readState() {
  try {
    return JSON.parse(fs.readFileSync(STATE, "utf8"));
  } catch {
    return { preferred: "proxy" };
  }
}

function writeState(s) {
  try {
    fs.mkdirSync(path.dirname(STATE), { recursive: true });
    fs.writeFileSync(STATE, JSON.stringify(s, null, 2), "utf8");
  } catch {
    /* ignore */
  }
}

function routeOrder() {
  const s = readState();
  const first = s.preferred === "direct" ? "direct" : "proxy";
  return first === "proxy" ? ["proxy", "direct"] : ["direct", "proxy"];
}

function configArgs(route) {
  const t = token();
  const args = ["-c", "http.sslBackend=openssl", "-c", "http.version=HTTP/1.1"];
  if (route === "proxy") {
    args.push("-c", "http.proxy=" + (cfg.proxy || "http://192.168.3.9:7890"));
  } else {
    args.push("-c", "http.proxy="); // 显式清空，覆盖全局/环境代理
  }
  if (t) {
    const b64 = Buffer.from("x-access-token:" + t, "utf8").toString("base64");
    args.push("-c", "http.extraHeader=AUTHORIZATION: basic " + b64);
  }
  return args;
}

const TRANSIENT = /Connection was reset|Recv failure|Failed to connect|Could not resolve|early EOF|unexpected disconnect|RPC failed|The remote end hung up|schannel|SSL|TLS|timeout|timed out|remote end hung up|HTTP 5\d\d/i;
const PERMANENT = /Authentication failed|Repository not found|not authorized|403|non-fast-forward|\[rejected\]/i;

// Windows 下 node 的 spawn 不走 PATHEXT，PATH 里只有 cmd shim 时会 ENOENT，所以先定位 git.exe。
function gitExe() {
  if (process.env.GH_GIT) return process.env.GH_GIT;
  const candidates = [
    "C:\\Program Files\\Git\\cmd\\git.exe",
    "C:\\Program Files (x86)\\Git\\cmd\\git.exe",
    "C:\\Program Files\\Git\\bin\\git.exe",
    "git",
  ];
  for (const c of candidates) {
    if (!c.includes("\\")) return c;
    if (fs.existsSync(c)) return c;
  }
  return "git";
}

// DSH 沙箱禁止 node 用管道捕获子进程输出（spawn 默认 stdio: 'pipe' 直接 EPERM），
// 所以这里把 stdout/stderr 重定向到临时文件再读回来，不建管道。
function git(args, cwd) {
  return new Promise((resolve) => {
    const dir = path.join(HERE, "state");
    fs.mkdirSync(dir, { recursive: true });
    const stamp = process.pid + "-" + Date.now() + "-" + Math.floor(Math.random() * 1e6);
    const outFile = path.join(dir, "git-" + stamp + ".out");
    const errFile = path.join(dir, "git-" + stamp + ".err");
    const outFd = fs.openSync(outFile, "w");
    const errFd = fs.openSync(errFile, "w");
    const child = spawn(gitExe(), args, {
      cwd,
      env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
      windowsHide: true,
      stdio: ["ignore", outFd, errFd],
    });
    child.on("error", (e) => {
      fs.closeSync(outFd);
      fs.closeSync(errFd);
      resolve({ code: 1, out: "", err: "spawn failed: " + e.message });
    });
    child.on("close", (code) => {
      fs.closeSync(outFd);
      fs.closeSync(errFd);
      let out = "";
      let err = "";
      try {
        out = fs.readFileSync(outFile, "utf8");
      } catch {}
      try {
        err = fs.readFileSync(errFile, "utf8");
      } catch {}
      try {
        fs.unlinkSync(outFile);
        fs.unlinkSync(errFile);
      } catch {}
      resolve({ code, out, err });
    });
  });
}

// 记录成功的 push：监听器据此把「我推的提交」与云端同账号的提交区分开。
function recordPush(refspec) {
  const ledger = path.join(HERE, "state", "own_actions.json");
  const entry = { kind: "push", refspec: refspec || "", at: new Date().toISOString() };
  let list = [];
  try {
    list = JSON.parse(fs.readFileSync(ledger, "utf8"));
  } catch {}
  const cutoff = Date.now() - 3 * 24 * 3600 * 1000;
  list = list.filter((a) => Date.parse(a.at) > cutoff).concat([entry]);
  fs.mkdirSync(path.dirname(ledger), { recursive: true });
  fs.writeFileSync(ledger, JSON.stringify(list, null, 2), "utf8");
}

async function withRetry(cwd, build) {
  const order = routeOrder();
  let last = { code: 1, out: "", err: "no attempt" };
  for (let i = 0; i < ATTEMPTS; i++) {
    const route = order[i % order.length];
    const args = [...configArgs(route), ...build()];
    const res = await git(args, cwd);
    const text = res.out + res.err;
    if (res.code === 0) {
      const s = readState();
      s.preferred = route;
      s[route + "OkAt"] = Date.now();
      writeState(s);
      if (!process.env.GH_GIT_QUIET) console.error("[git] ok via " + route + " (attempt " + (i + 1) + ")");
      return { ...res, route };
    }
    last = { ...res, route };
    if (PERMANENT.test(text) && !TRANSIENT.test(text)) {
      console.error("[git] permanent failure via " + route + ": " + text.trim().split("\n").slice(-1)[0]);
      return last;
    }
    const wait = Math.min(20000, 1000 * 2 ** Math.min(i, 4)) + Math.floor(Math.random() * 500);
    if (!process.env.GH_GIT_QUIET) {
      console.error("[git] " + route + " failed (attempt " + (i + 1) + "/" + ATTEMPTS + "): " + text.trim().split("\n").slice(-1)[0]);
      console.error("[git] retry in " + Math.round(wait / 1000) + "s");
    }
    await new Promise((r) => setTimeout(r, wait));
  }
  return last;
}

const argv = process.argv.slice(2);
const [cmd, repoDir, ...rest] = argv;
if (!cmd || !repoDir) {
  console.error("usage: node monitor/gh_push.mjs {push|fetch|ls-remote} <repoDir> [refspec...]");
  process.exit(2);
}
const cwd = path.resolve(ROOT, repoDir);

let res;
if (cmd === "push") {
  const refspec = rest[0];
  if (!refspec) {
    console.error("push needs a refspec, e.g. HEAD:research/my-branch");
    process.exit(2);
  }
  res = await withRetry(cwd, () => ["push", REMOTE, refspec]);
} else if (cmd === "fetch") {
  res = await withRetry(cwd, () =>
    rest.length
      ? ["fetch", "--prune", REMOTE, ...rest]
      : ["fetch", "--prune", REMOTE, "+refs/heads/*:refs/remotes/origin/*"]
  );
} else if (cmd === "ls-remote") {
  res = await withRetry(cwd, () => ["ls-remote", REMOTE, ...rest]);
} else {
  console.error("unknown command " + cmd);
  process.exit(2);
}

if (res.code === 0) {
  if (cmd === "push") recordPush(rest[0]);
  if (cmd === "ls-remote") console.log(res.out);
  process.exit(0);
}
console.error(res.err || res.out);
process.exit(res.code || 1);
