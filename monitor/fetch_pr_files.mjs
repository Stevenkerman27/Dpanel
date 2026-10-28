import fs from "node:fs";
import path from "node:path";
import https from "node:https";

const ROOT = path.resolve(import.meta.dirname, "..");
const token = (() => {
  try { return fs.readFileSync(path.join(ROOT, ".agents", "github_token"), "utf8").trim(); }
  catch { return ""; }
})();
const REPO = "Stevenkerman27/Dpanel";

function httpsGet(url, raw = false) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request({
      hostname: u.hostname, path: u.pathname + u.search, method: "GET",
      headers: {
        "User-Agent": "dpanel-local",
        Accept: raw ? "application/vnd.github.raw" : "application/vnd.github+json",
        ...(token ? { Authorization: "Bearer " + token } : {}),
      },
    }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const buf = Buffer.concat(chunks);
        if (res.statusCode !== 200) return reject(new Error("HTTP " + res.statusCode + " " + buf.toString("utf8").slice(0, 200)));
        resolve(raw ? buf.toString("utf8") : JSON.parse(buf.toString("utf8")));
      });
    });
    req.on("error", reject);
    req.end();
  });
}

const pr = process.argv[2];
const dest = process.argv[3] || ".";
const info = await httpsGet("https://api.github.com/repos/" + REPO + "/pulls/" + pr);
const files = await httpsGet("https://api.github.com/repos/" + REPO + "/pulls/" + pr + "/files?per_page=100");
console.log("PR #" + pr, info.title);
console.log("head", info.head.sha, "base", info.base.sha, "draft", info.draft, "files", files.length);
for (const f of files) {
  const target = path.join(ROOT, dest, f.filename);
  if (f.status === "removed") { console.log("SKIP removed", f.filename); continue; }
  const body = await httpsGet("https://api.github.com/repos/" + REPO + "/contents/" + encodeURI(f.filename) + "?ref=" + info.head.sha, true);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, body, "utf8");
  console.log(f.status.padEnd(9), String(body.length).padStart(7), f.filename);
}
