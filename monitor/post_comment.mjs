#!/usr/bin/env node
// 通过 HTTP 代理隧道向 GitHub 发 issue/PR 评论。
// 直连 api.github.com 发长正文会被中途断开（ERR socket hang up），走代理则稳定，见 README「发评论」。
// 用法: node monitor/post_comment.mjs <apiPath> <bodyFile> [--delete <commentId>...]
//   例: node monitor/post_comment.mjs repos/Stevenkerman27/Dpanel/issues/7/comments .tmp/report.md
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

const argv = process.argv.slice(2);
const delIdx = argv.indexOf("--delete");
const positional = delIdx >= 0 ? argv.slice(0, delIdx) : argv;
const delIds = delIdx >= 0 ? argv.slice(delIdx + 1) : [];
const [apiPathRaw, bodyFile] = positional;
// https.request 的 path 必须以 "/" 开头，否则 Node 发出畸形请求行，服务器直接断连（表现为 ERR socket hang up）
const apiPath = apiPathRaw && !apiPathRaw.startsWith("/") ? "/" + apiPathRaw : apiPathRaw;

if (!apiPath || !bodyFile) {
  console.error("usage: node monitor/post_comment.mjs <apiPath> <bodyFile> [--delete <id>...]");
  process.exit(2);
}

for (const id of delIds) {
  const d = await request("DELETE", "/repos/" + cfg.repo + "/issues/comments/" + id, null);
  console.log("delete", id, "->", d.status);
}

const body = fs.readFileSync(path.resolve(ROOT, bodyFile), "utf8");
let last;
for (let attempt = 1; attempt <= 3; attempt++) {
  last = await request("POST", apiPath, { body });
  if (last.status === 201) {
    const j = JSON.parse(last.text);
    console.log("OK", j.html_url, "id:", j.id, "bytes:", Buffer.byteLength(body, "utf8"));
    process.exit(0);
  }
  console.error("attempt", attempt, "->", last.status, last.text.slice(0, 120));
  await new Promise((r) => setTimeout(r, 3000));
}
console.error("FAILED");
process.exit(1);
