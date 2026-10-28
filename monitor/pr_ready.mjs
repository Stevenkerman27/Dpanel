#!/usr/bin/env node
// Mark a draft PR ready for review. The REST PATCH /pulls/{n} {"draft": false} is
// silently ignored by GitHub; the supported route is the GraphQL mutation
// markPullRequestReadyForReview. Reuses the proxy tunnel from post_comment.mjs.
// 用法: node monitor/pr_ready.mjs <owner/repo> <number>
import https from "node:https";
import net from "node:net";
import tls from "node:tls";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const cfg = JSON.parse(fs.readFileSync(path.join(HERE, "config.json"), "utf8"));
const proxy = new URL(cfg.proxy || "http://192.168.3.9:7890");
const token = fs.readFileSync(path.resolve(ROOT, ".agents/github_token"), "utf8").trim();

function tunnel() {
  return new Promise((res, rej) => {
    const s = net.connect(Number(proxy.port || 7890), proxy.hostname, () => {
      s.write("CONNECT api.github.com:443 HTTP/1.1\r\nHost: api.github.com:443\r\n\r\n");
    });
    let buf = "";
    const onData = (d) => {
      buf += d.toString("latin1");
      if (!buf.includes("\r\n\r\n")) return;
      s.removeListener("data", onData);
      if (!/^HTTP\/1\.[01] 200/.test(buf)) return rej(new Error("proxy refused: " + buf.slice(0, 80)));
      const t = tls.connect({ socket: s, servername: "api.github.com" }, () => res(t));
      t.on("error", rej);
    };
    s.on("data", onData);
    s.on("error", rej);
    setTimeout(() => rej(new Error("proxy timeout")), 20000);
  });
}

async function request(method, apiPath, body) {
  const t = await tunnel();
  return new Promise((res) => {
    const data = body === null ? null : Buffer.from(JSON.stringify(body), "utf8");
    const headers = {
      "User-Agent": "dpanel-probe",
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      Authorization: "Bearer " + token,
    };
    if (data) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = data.length;
    }
    const r = https.request(
      { host: "api.github.com", path: apiPath, method, headers, createConnection: () => t, agent: false, timeout: 60000 },
      (resp) => {
        const c = [];
        resp.on("data", (d) => c.push(d));
        resp.on("end", () => res({ status: resp.statusCode, text: Buffer.concat(c).toString() }));
      },
    );
    r.on("timeout", () => r.destroy(new Error("timeout")));
    r.on("error", (e) => res({ status: 0, text: "ERR " + e.message }));
    if (data) r.write(data);
    r.end();
  });
}

const [repo, number] = process.argv.slice(2);
if (!repo || !number) {
  console.error("usage: node monitor/pr_ready.mjs <owner/repo> <number>");
  process.exit(2);
}
const pr = await request("GET", "/repos/" + repo + "/pulls/" + number, null);
if (pr.status !== 200) {
  console.error("STATUS " + pr.status + " " + pr.text.slice(0, 400));
  process.exit(1);
}
const info = JSON.parse(pr.text);
if (!info.draft) {
  console.log("ALREADY_READY " + info.html_url);
  process.exit(0);
}
const mutation = "mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{number isDraft url}}}";
const out = await request("POST", "/graphql", { query: mutation, variables: { id: info.node_id } });
const parsed = JSON.parse(out.text);
if (out.status !== 200 || parsed.errors) {
  console.error("GRAPHQL " + out.status + " " + JSON.stringify(parsed.errors || parsed).slice(0, 600));
  process.exit(1);
}
const prOut = parsed.data.markPullRequestReadyForReview.pullRequest;
console.log("READY " + prOut.number + " draft=" + prOut.isDraft + " " + prOut.url);
