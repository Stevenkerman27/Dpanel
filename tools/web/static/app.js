"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  schema: null, defaults: null, params: null, job: null,
  mesh: null, result: null, camera: { yaw: -0.62, pitch: 0.42, zoom: 1.0 },
  poll: null, busy: false,
};

function setBadge(text, cls) {
  const el = $("state");
  el.textContent = text;
  el.className = "badge" + (cls ? " " + cls : "");
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ error: "非 JSON 响应" }));
  if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
  return data;
}

function buildForm() {
  const host = $("params");
  host.innerHTML = "";
  for (const item of state.schema) {
    const label = document.createElement("label");
    label.textContent = item.label;
    const input = document.createElement("input");
    input.type = "number";
    input.id = "p-" + item.key;
    input.min = item.min; input.max = item.max; input.step = item.step;
    input.value = state.defaults[item.key];
    input.addEventListener("change", () => { state.params[item.key] = parseFloat(input.value); });
    const unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = item.unit;
    host.append(label, input, unit);
  }
  const spare = document.createElement("span");
  host.append(spare);
}

function readParams() {
  const out = {};
  for (const item of state.schema) {
    const value = parseFloat($("p-" + item.key).value);
    out[item.key] = Number.isFinite(value) ? value : item.default;
  }
  return out;
}

function setBusy(busy, text, cls) {
  state.busy = busy;
  $("build").disabled = busy;
  $("solve").disabled = busy || !state.job;
  $("cancel").disabled = !busy;
  setBadge(text || (busy ? "运行中" : (state.job ? "已构建" : "未构建")),
           cls || (busy ? "busy" : (state.job ? "ok" : "")));
}

/* ---------- 3D 视图（正交投影 + 画家算法，无外部依赖） ---------- */

const view = { canvas: null, ctx: null, faces: [], vertices: [], cache: null };

function resize() {
  const canvas = $("scene");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.round(rect.width * ratio));
  canvas.height = Math.max(240, Math.round(rect.height * ratio));
  view.ctx = canvas.getContext("2d");
  view.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function rotate(point) {
  const { yaw, pitch } = state.camera;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = point[0] * cy - point[1] * sy;
  const y0 = point[0] * sy + point[1] * cy;
  const y = y0 * cp - point[2] * sp;
  const z = y0 * sp + point[2] * cp;
  return [x, y, z];
}

function project(point, center, scale, width, height) {
  const r = rotate(point);
  return [width * 0.5 + (r[0] - center[0]) * scale,
          height * 0.5 - (r[2] - center[2]) * scale, r[1]];
}

function prepare() {
  if (!state.mesh) { view.faces = []; view.vertices = []; return; }
  const vertices = state.mesh.vertices;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) for (let i = 0; i < 3; i++) {
    lo[i] = Math.min(lo[i], v[i]); hi[i] = Math.max(hi[i], v[i]);
  }
  const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  view.vertices = vertices;
  view.center = center;
  view.extent = extent;
  view.faces = state.mesh.faces.map((f) => {
    const a = vertices[f[0]], b = vertices[f[1]], c = vertices[f[2]];
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const len = Math.hypot(n[0], n[1], n[2]) || 1;
    return { f, n: [n[0] / len, n[1] / len, n[2] / len],
             c: [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3] };
  });
}

function draw() {
  const ctx = view.ctx;
  if (!ctx) return;
  const canvas = $("scene");
  const width = canvas.width / (window.devicePixelRatio || 1);
  const height = canvas.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0e1116";
  ctx.fillRect(0, 0, width, height);
  drawGrid(ctx, width, height, view.center || [0, 0, 0]);
  if (!view.faces.length) {
    ctx.fillStyle = "#5f6c80";
    ctx.font = "13px system-ui";
    ctx.fillText("尚未生成几何", width / 2 - 46, height / 2);
    return;
  }
  const scale = Math.min(width, height) / (view.extent * 1.9) * state.camera.zoom;
  const light = rotate([0.45, -0.6, 0.66]);
  const items = [];
  for (const face of view.faces) {
    const n = rotate(face.n);
    const p0 = project(view.vertices[face.f[0]], view.center, scale, width, height);
    const p1 = project(view.vertices[face.f[1]], view.center, scale, width, height);
    const p2 = project(view.vertices[face.f[2]], view.center, scale, width, height);
    const depth = (p0[2] + p1[2] + p2[2]) / 3;
    items.push({ p0, p1, p2, depth, shade: Math.max(0, n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) });
  }
  items.sort((a, b) => a.depth - b.depth);
  for (const item of items) {
    const level = 40 + Math.round(150 * item.shade);
    ctx.fillStyle = "rgb(" + Math.round(level * 0.62) + "," + Math.round(level * 0.78) + "," + level + ")";
    ctx.strokeStyle = "rgba(10,14,20,0.35)";
    ctx.beginPath();
    ctx.moveTo(item.p0[0], item.p0[1]);
    ctx.lineTo(item.p1[0], item.p1[1]);
    ctx.lineTo(item.p2[0], item.p2[1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = "#7d8ba1";
  ctx.font = "12px system-ui";
  const mesh = state.mesh;
  ctx.fillText("截面 " + mesh.sections + " · 每截面 " + mesh.points_per_section + " 点 · 三角面 " + mesh.faces.length,
               12, 20);
}

function drawGrid(ctx, width, height, center) {
  const scale = Math.min(width, height) / ((view.extent || 6) * 1.9) * state.camera.zoom;
  ctx.strokeStyle = "rgba(72,86,106,0.25)";
  ctx.lineWidth = 1;
  const size = 6;
  for (let i = -size; i <= size; i++) {
    const a = project([i * 0.5, -3, 0], center, scale, width, height);
    const b = project([i * 0.5, 3, 0], center, scale, width, height);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    const c = project([-3, i * 0.5, 0], center, scale, width, height);
    const d = project([3, i * 0.5, 0], center, scale, width, height);
    ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
  }
}

/* ---------- 结果 ---------- */

function renderResult(result) {
  const body = $("result").querySelector("tbody");
  body.innerHTML = "";
  if (!result) {
    body.innerHTML = '<tr><td class="muted">尚未求解</td></tr>';
    drawPlot(null);
    return;
  }
  if (result.warning) {
    const tr = document.createElement("tr");
    tr.className = "warn";
    const td0 = document.createElement("td");
    td0.textContent = "警告";
    const td1 = document.createElement("td");
    td1.textContent = result.warning;
    tr.append(td0, td1);
    body.append(tr);
  }
  const rows = [
    ["CL", result.CL], ["CD (压力)", result.CD],
    ["CM", result.CM ? result.CM.join(", ") : "-"],
    ["Cp 范围", result.cp ? result.cp.map((v) => v.toFixed(2)).join(" … ") : "-"],
    ["线性残差", result.residual === null || result.residual === undefined ? "-" : result.residual.toExponential(2)],
    ["未知量", result.unknowns || "-"], ["壁面面板", result.panels || "-"],
    ["求解耗时", result.seconds ? result.seconds.toFixed(1) + " s" : "-"],
    ["收敛", result.converged === undefined ? "-" : (result.converged ? "是" : "否")],
    ["设备", result.options ? result.options.device_used : "-"],
  ];
  for (const [name, value] of rows) {
    const tr = document.createElement("tr");
    const td0 = document.createElement("td");
    td0.textContent = name;
    const td1 = document.createElement("td");
    td1.textContent = typeof value === "number" ? value.toFixed(6) : value;
    tr.append(td0, td1);
    body.append(tr);
  }
  drawPlot(result);
}

function drawPlot(result) {
  const canvas = $("plot");
  const ctx = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#10141b";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#2a3342";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  let series = null, label = "";
  if (result && result.history && result.history.length) {
    series = result.history.map((h, i) => [i / Math.max(1, result.history.length - 1), h.CL]);
    label = "自由尾迹 CL 迭代";
  } else if (result && result.group_rows && result.group_rows.length) {
    series = result.group_rows[0].gamma;
    label = "展向环量 γ（第一组）";
  }
  ctx.fillStyle = "#7d8ba1";
  ctx.font = "11px system-ui";
  ctx.fillText(label || "无曲线数据", 8, 14);
  if (!series || !series.length) return;
  const xs = series.map((p) => p[0]), ys = series.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 1e-9) { y0 -= 1e-3; y1 += 1e-3; }
  const pad = 22;
  const px = (v) => pad + (v - x0) / (x1 - x0 || 1) * (width - pad - 8);
  const py = (v) => height - pad - (v - y0) / (y1 - y0) * (height - pad - 22);
  ctx.strokeStyle = "#4b8fe0";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  series.forEach(([x, y], i) => { const X = px(x), Y = py(y); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
  ctx.stroke();
  ctx.fillStyle = "#5f6c80";
  ctx.fillText(y1.toFixed(4), 2, 24);
  ctx.fillText(y0.toFixed(4), 2, height - 6);
}

/* ---------- 交互 ---------- */

function bindView() {
  const canvas = $("scene");
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", (event) => {
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    state.camera.yaw += (event.clientX - lastX) * 0.008;
    state.camera.pitch = Math.max(-1.4, Math.min(1.4, state.camera.pitch + (event.clientY - lastY) * 0.006));
    lastX = event.clientX; lastY = event.clientY;
    draw();
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.camera.zoom = Math.max(0.2, Math.min(6, state.camera.zoom * (event.deltaY < 0 ? 1.12 : 0.9)));
    draw();
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    state.camera = { yaw: -0.62, pitch: 0.42, zoom: 1.0 };
    draw();
  });
}

function appendLog(lines) {
  const box = $("log");
  box.textContent = lines.join("\n");
  box.scrollTop = box.scrollHeight;
}

async function doBuild() {
  setBusy(true, "构建中", "busy");
  try {
    state.params = readParams();
    const data = await api("/api/build", { params: state.params });
    state.job = data.job;
    state.mesh = data.mesh;
    state.result = null;
    $("job-id").textContent = data.job;
    prepare();
    draw();
    renderResult(null);
    $("solve").disabled = false;
    $("overlay").textContent =
      "参考面积 " + data.reference.area.toFixed(3) + " m² · 参考弦 " + data.reference.chord.toFixed(3) + " m";
    setBusy(false, "已构建", "ok");
  } catch (error) {
    setBusy(false, "构建失败", "err");
    $("log").textContent = String(error.message || error);
  }
}

async function poll() {
  if (!state.job) return;
  try {
    const data = await api("/api/job?id=" + encodeURIComponent(state.job));
    appendLog(data.log || []);
    if (data.state === "done") {
      state.result = data.result;
      renderResult(data.result);
      setBusy(false, "完成", "ok");
      clearInterval(state.poll); state.poll = null;
    } else if (data.state === "error") {
      appendLog(["[错误] " + (data.error || "未知错误")]);
      setBusy(false, "失败", "err");
      clearInterval(state.poll); state.poll = null;
    } else if (data.state === "cancelled") {
      setBusy(false, "已取消", "err");
      clearInterval(state.poll); state.poll = null;
    }
  } catch (error) {
    appendLog([String(error.message || error)]);
  }
}

async function doSolve() {
  if (!state.job) return;
  setBusy(true, "求解中", "busy");
  try {
    await api("/api/solve", {
      job: state.job,
      alpha: parseFloat($("alpha").value),
      wake: $("wake").value,
      device: $("device").value,
      size: parseFloat($("size").value),
      iterations: parseInt($("iterations").value, 10),
    });
    state.poll = setInterval(poll, 700);
    poll();
  } catch (error) {
    setBusy(false, "失败", "err");
    appendLog([String(error.message || error)]);
  }
}

async function doCancel() {
  if (!state.job) return;
  await api("/api/cancel", { job: state.job }).catch(() => {});
}

async function main() {
  const schema = await api("/api/schema");
  state.schema = schema.params;
  state.defaults = schema.defaults;
  state.params = schema.defaults;
  buildForm();
  $("overlay").textContent = "产物目录 " + schema.runs;
  $("build").addEventListener("click", doBuild);
  $("reset").addEventListener("click", () => {
    Object.entries(state.defaults).forEach(([key, value]) => { $("p-" + key).value = value; });
  });
  $("solve").addEventListener("click", doSolve);
  $("cancel").addEventListener("click", doCancel);
  bindView();
  window.addEventListener("resize", resize);
  resize();
}

main().catch((error) => {
  setBadge("启动失败", "err");
  $("log").textContent = String(error.message || error);
});
