#!/usr/bin/env node
// Mark a draft PR ready for review. REST PATCH /pulls/{n} {"draft": false} is
// silently ignored by GitHub; the supported route is the GraphQL mutation
// markPullRequestReadyForReview. Network retries and the proxy/direct failover
// come from net.mjs, so a reset does not lose the mutation.
// 用法: node monitor/pr_ready.mjs <owner/repo> <number>
import { apiRequest } from "./net.mjs";

const [repo, number] = process.argv.slice(2);
if (!repo || !number) {
  console.error("usage: node monitor/pr_ready.mjs <owner/repo> <number>");
  process.exit(2);
}

async function post(url, body, attempts) {
  return apiRequest(url, {
    method: "POST",
    body: Buffer.from(JSON.stringify(body), "utf8"),
    attempts,
  });
}

const pr = await apiRequest("https://api.github.com/repos/" + repo + "/pulls/" + number, { attempts: 6 });
if (pr.status !== 200) {
  console.error("STATUS " + pr.status + " " + (pr.body || pr.error || "").slice(0, 400));
  process.exit(1);
}
const info = JSON.parse(pr.body);
if (!info.draft) {
  console.log("ALREADY_READY " + info.html_url);
  process.exit(0);
}

const mutation =
  "mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{number isDraft url}}}";
const out = await post("https://api.github.com/graphql", { query: mutation, variables: { id: info.node_id } }, 8);
let parsed = null;
try {
  parsed = JSON.parse(out.body);
} catch {
  parsed = null;
}
if (out.status !== 200 || !parsed || parsed.errors) {
  console.error("GRAPHQL " + out.status + " " + JSON.stringify(parsed ? parsed.errors || parsed : out.body).slice(0, 600));
  process.exit(1);
}
const prOut = parsed.data.markPullRequestReadyForReview.pullRequest;
console.log("READY " + prOut.number + " draft=" + prOut.isDraft + " " + prOut.url);
