#!/usr/bin/env node
// Publish local work to GitHub in one call: stage, commit, push, and if the git
// transport is down, replay the same commit onto the remote head through the git
// data API. Prints one machine-readable line so a caller does not need to poll.
//
//   node monitor/publish.mjs <repoDir> <branch> [--paths a,b,c] [--message-file f]
//                                                  [--dry-run] [--attempts 4]
//
// Why: github.com:443 (git transport) fails independently of api.github.com, and a
// failed push used to cost a long retry loop. The API path sends only the files in
// --paths, so it stays cheap and rate-limit friendly.
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { apiJson } from "./net.mjs";

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));

function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  if (i < 0) return dflt;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith("--") ? true : v;
}

const GIT = "C:/Program Files/Git/cmd/git.exe";
const repoDir = process.argv[2];
const branch = process.argv[3];
if (!repoDir || !branch) {
  console.error("usage: node monitor/publish.mjs <repoDir> <branch> [--paths a,b,c] [--message-file f] [--dry-run]");
  process.exit(2);
}
const paths = String(arg("paths", "") || "").split(",").map((s) => s.trim()).filter(Boolean);
const messageFile = arg("message-file", null);
const dryRun = Boolean(arg("dry-run", false));
const attempts = Number(arg("attempts", 4));

let seq = 0;
const TMP = path.join(HERE, "state");

// 沙箱下 node 的 stdio:'pipe' 直接 EPERM（见 monitor/README.md），所以把子进程输出
// 重定向到文件再读回。
function git(args) {
  return new Promise((resolve) => {
    const tag = "pub-" + process.pid + "-" + seq++;
    const o = path.join(TMP, tag + ".out");
    const e = path.join(TMP, tag + ".err");
    const fo = fs.openSync(o, "w");
    const fe = fs.openSync(e, "w");
    const finish = (code) => {
      const read = (f) => {
        try {
          return fs.readFileSync(f, "utf8").trim();
        } catch {
          return "";
        }
      };
      const out = read(o);
      const err = read(e);
      try { fs.closeSync(fo); fs.closeSync(fe); fs.unlinkSync(o); fs.unlinkSync(e); } catch {}
      resolve({ code, out, err });
    };
    let task;
    try {
      task = spawn(GIT, ["-C", repoDir, ...args], { stdio: ["ignore", fo, fe] });
    } catch (err) {
      finish(-1);
      return;
    }
    task.on("error", () => finish(-1));
    task.on("close", (code) => finish(code));
  });
}

// 记录成功的发布：监听器据此把「我推的提交」与云端同账号的提交区分开。
// 与 gh_push.mjs 的 recordPush 同一账本、同一条目形状。
function recordPublish(refspec) {
  const ledger = path.join(TMP, "own_actions.json");
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

async function apiHead(target) {
  const r = await apiJson("/repos/" + target + "/git/ref/heads/" + branch);
  if (r.status !== 200) throw new Error("ref " + r.status + " " + (r.body || "").slice(0, 200));
  return r.json.object.sha;
}

// 1. stage + commit locally
if (paths.length) {
  await git(["add", ...paths]);
}
const dtype = await git(["diff", "--cached", "--name-only"]);
const staged = dtype.out ? dtype.out.split("\n").filter(Boolean) : [];
if (staged.length) {
  if (!messageFile) {
    console.error("DIRTY_STAGED " + staged.join(",") + " (需要 --message-file)");
    process.exit(1);
  }
  const c = await git(["-c", "user.name=Stevenkerman27", "-c", "user.email=zyx20031020@gmail.com",
    "commit", "-F", path.resolve(messageFile)]);
  if (c.code !== 0) {
    console.error("COMMIT_FAILED " + (c.err || c.out).slice(0, 400));
    process.exit(1);
  }
}
const head = await git(["rev-parse", "HEAD"]);
if (head.code !== 0) {
  console.error("REV_FAILED " + head.err);
  process.exit(1);
}
const headSha = head.out;
const touched = paths.length ? paths : (await git(["show", "--pretty=format:", "--name-only", "HEAD"])).out.split("\n").filter(Boolean);

const target = (process.env.GH_REPO || "Stevenkerman27/Dpanel").trim();
const remoteSha = await apiHead(target);

if (dryRun) {
  console.log("DRY_RUN head=" + headSha.slice(0, 7) + " remote=" + remoteSha.slice(0, 7) + " paths=" + touched.join(","));
  process.exit(0);
}
if (headSha === remoteSha) {
  console.log("ALREADY " + headSha.slice(0, 7));
  process.exit(0);
}

// 2. try the git transport first (it preserves the exact commit object)
for (let i = 0; i < attempts; i++) {
  const p = await git(["push", "origin", "HEAD:refs/heads/" + branch]);
  if (p.code === 0) {
    recordPublish("refs/heads/" + branch);
    console.log("PUSHED " + headSha.slice(0, 7) + " " + branch + " via git (attempt " + (i + 1) + ")");
    process.exit(0);
  }
  const msg = (p.err || p.out);
  if (!/Recv failure|Connection was reset|Could not connect|Connection timed out|early EOF|timed out/i.test(msg) || i === attempts - 1) break;
  await new Promise((r) => setTimeout(r, 2000 * (i + 1)));
}

// 3. fall back to the git data API: one commit whose parent is the remote head
const base = await apiJson("/repos/" + target + "/git/commits/" + remoteSha);
if (base.status !== 200) throw new Error("base commit " + base.status);
const entries = [];
for (const p of touched) {
  const abs = path.resolve(repoDir, p);
  if (!fs.existsSync(abs)) {
    entries.push({ path: p, mode: "100644", type: "blob", sha: null });
    continue;
  }
  const blob = await apiJson("/repos/" + target + "/git/blobs", {
    method: "POST",
    body: { content: fs.readFileSync(abs).toString("base64"), encoding: "base64" },
  });
  if (blob.status !== 201) throw new Error("blob " + p + " " + blob.status);
  const mode = await git(["ls-files", "--stage", p]);
  const fileMode = mode.out.startsWith("100755") ? "100755" : "100644";
  entries.push({ path: p, mode: fileMode, type: "blob", sha: blob.json.sha });
}
const msg = messageFile ? fs.readFileSync(path.resolve(messageFile), "utf8") : (await git(["log", "-1", "--pretty=%B"])).out;
const tree = await apiJson("/repos/" + target + "/git/trees", {
  method: "POST",
  body: { base_tree: base.json.tree.sha, tree: entries },
});
if (tree.status !== 201) throw new Error("tree " + tree.status + " " + (tree.body || "").slice(0, 300));
const commit = await apiJson("/repos/" + target + "/git/commits", {
  method: "POST",
  body: { message: msg, tree: tree.json.sha, parents: [remoteSha] },
});
if (commit.status !== 201) throw new Error("commit " + commit.status + " " + (commit.body || "").slice(0, 300));
const patch = await apiJson("/repos/" + target + "/git/refs/heads/" + branch, {
  method: "PATCH",
  body: { sha: commit.json.sha, force: false },
});
if (patch.status !== 200) throw new Error("ref " + patch.status + " " + (patch.body || "").slice(0, 300));
recordPublish("refs/heads/" + branch);
const check = await apiHead(target);
console.log("API_PUSH " + remoteSha.slice(0, 7) + "->" + commit.json.sha.slice(0, 7) + " " + branch +
  " files=" + entries.length + " verified=" + (check === commit.json.sha));
