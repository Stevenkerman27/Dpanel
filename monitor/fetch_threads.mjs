import fs from "node:fs";
import path from "node:path";
import https from "node:https";

const ROOT = path.resolve(import.meta.dirname, "..");
const token = (() => {
  try { return fs.readFileSync(path.join(ROOT, ".agents", "github_token"), "utf8").trim(); }
  catch { return ""; }
})();

function api(url, { method = "GET", body } = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const data = body ? JSON.stringify(body) : null;
    const req = https.request({
      hostname: u.hostname, path: u.pathname + u.search, method,
      headers: {
        "User-Agent": "dpanel-local",
        Accept: "application/vnd.github+json",
        ...(token ? { Authorization: "Bearer " + token } : {}),
        ...(data ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) } : {}),
      },
    }, (res) => {
      let buf = "";
      res.on("data", (c) => (buf += c));
      res.on("end", () => {
        let json = null;
        try { json = JSON.parse(buf); } catch { json = buf; }
        resolve({ status: res.statusCode, json });
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

const targets = process.argv.slice(2);
const outDir = path.join(ROOT, ".tmp");
fs.mkdirSync(outDir, { recursive: true });

for (const t of targets) {
  const [kind, num] = [t.startsWith("p") ? "pulls" : "issues", t.replace(/[^0-9]/g, "")];
  const main = await api("https://api.github.com/repos/Stevenkerman27/Dpanel/" + kind + "/" + num);
  if (main.status !== 200) { console.log(t, "ERR", main.status, JSON.stringify(main.json).slice(0, 200)); continue; }
  const d = main.json;
  const comments = await api("https://api.github.com/repos/Stevenkerman27/Dpanel/issues/" + num + "/comments?per_page=100");
  const cs = Array.isArray(comments.json) ? comments.json : [];
  const md = [
    "# " + t + " " + d.title,
    "- state: " + d.state + (d.draft ? " (draft)" : ""),
    "- user: " + d.user?.login + "  created " + d.created_at + "  updated " + d.updated_at,
    "- url: " + d.html_url,
    d.head ? "- head: " + d.head.ref + " @ " + d.head.sha + "  base: " + d.base.ref : "",
    d.merged_at ? "- merged_at: " + d.merged_at : "",
    "", "## body", "", d.body || "(empty)", "",
    "## comments (" + cs.length + ")",
    ...cs.map((c) => "\n---\n### @" + c.user?.login + " " + c.created_at + "  (id " + c.id + ")\n\n" + (c.body || "")),
  ].join("\n");
  const f = path.join(outDir, "gh_" + t + ".md");
  fs.writeFileSync(f, md, "utf8");
  console.log(t, "ok", d.state, (d.title || "").slice(0, 60), "comments", cs.length, "->", path.relative(ROOT, f));
}
