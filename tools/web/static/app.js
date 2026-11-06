'use strict';

const el = (id) => document.getElementById(id);
const state = {
  schema: [], bodySchema: [], bodyDefaults: {},
  parts: [], selected: 0,
  build: null, result: null, mesh: null, meshLoading: false, jobId: null,
  tab: 'geom', busy: false, dirty: true, poll: null,
};

/* ---------------- API ---------------- */
async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
  return data;
}

function logLine(text) {
  const box = el('log');
  box.textContent += text + '\n';
  box.scrollTop = box.scrollHeight;
}
function clearLog() { el('log').textContent = ''; }

/* ---------------- parts ---------------- */
function wingDefaults() {
  const out = {};
  state.schema.forEach((item) => { out[item.key] = item.default; });
  return out;
}
function addPart(kind) {
  if (kind === 'wing' && state.parts.some((p) => p.kind === 'wing')) return;
  const params = kind === 'wing' ? wingDefaults() : Object.assign({}, state.bodyDefaults);
  const count = state.parts.filter((p) => p.kind === kind).length + 1;
  state.parts.push({ kind, name: (kind === 'wing' ? '主翼' : '机身 ') + (kind === 'wing' ? '' : count), params });
  state.selected = state.parts.length - 1;
  renderParts();
  renderEditor();
  markDirty();
}
function renderParts() {
  const list = el('partList');
  list.innerHTML = '';
  state.parts.forEach((part, index) => {
    const li = document.createElement('li');
    if (index === state.selected) li.className = 'sel';
    const label = document.createElement('span');
    label.textContent = part.name;
    const kind = document.createElement('span');
    kind.className = 'kind';
    kind.textContent = part.kind === 'wing' ? '升力面' : '机身';
    li.appendChild(label);
    li.appendChild(kind);
    li.addEventListener('click', () => { state.selected = index; renderParts(); renderEditor(); });
    list.appendChild(li);
  });
  el('addWing').disabled = state.parts.some((p) => p.kind === 'wing');
  el('delPart').disabled = !state.parts.length;
  if (!state.parts.length) {
    const li = document.createElement('li');
    li.textContent = '（还没有部件）';
    li.style.color = '#5f6b7c';
    list.appendChild(li);
  }
}

function renderEditor() {
  const part = state.parts[state.selected];
  const box = el('editor');
  box.innerHTML = '';
  if (!part) { el('editorTitle').textContent = '部件参数'; return; }
  el('editorTitle').textContent = part.name + ' · ' + (part.kind === 'wing' ? '升力面' : '机身');
  const schema = part.kind === 'wing' ? state.schema : state.bodySchema;
  schema.forEach((item) => {
    const label = document.createElement('label');
    label.textContent = item.label + (item.unit && item.unit !== '-' ? ' (' + item.unit + ')' : '');
    const input = document.createElement('input');
    input.type = 'number';
    input.value = part.params[item.key];
    input.min = item.min; input.max = item.max; input.step = item.step;
    input.addEventListener('input', () => {
      const value = Number(input.value);
      if (!Number.isFinite(value)) return;
      part.params[item.key] = value;
      markDirty();
    });
    label.appendChild(input);
    box.appendChild(label);
  });
}

function markDirty() { state.dirty = true; updateHint(); }

function updateHint() {
  const hint = el('viewHint');
  if (state.busy) hint.textContent = '计算中…';
  else if (state.tab === 'geom') hint.textContent = '左键旋转 · 滚轮缩放';
  else if (state.tab === 'mesh') {
    if (!state.mesh) hint.textContent = '尚无计算网格';
    else if (state.dirty || state.mesh.size !== Number(el('size').value) ||
             state.mesh.alpha !== Number(el('alpha').value)) hint.textContent = '网格参数已变化，请重新生成';
    else hint.textContent = '壁面 ' + state.mesh.faces.length + ' 面板 · 尾迹 ' + state.mesh.wake_panels + ' 面板';
  } else hint.textContent = state.result && state.result.field ? 'Cp 云图 · 半域镜像显示' : '先求解一次';
}

/* ---------------- viewer ---------------- */
const view = el('view');
const ctx = view.getContext('2d');
const cam = { yaw: -0.9, pitch: 0.35, zoom: 1 };
/* Fixed world->screen fit, computed once per scene. The old code refitted the
   projected bounds on every frame, so rotating also rescaled the model. */
const viewFit = { center: [0, 0, 0], radius: 1, key: '' };

function fitScene(scene) {
  const vertices = scene.vertices;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const used = scene.fitFacesOnly ? new Array(vertices.length).fill(false) : null;
  if (used) {
    for (let i = 0; i < scene.faces.length; i += 1) {
      for (let k = 0; k < scene.faces[i].length; k += 1) used[scene.faces[i][k]] = true;
    }
  }
  for (let i = 0; i < vertices.length; i += 1) {
    if (used && !used[i]) continue;
    for (let k = 0; k < 3; k += 1) {
      if (vertices[i][k] < lo[k]) lo[k] = vertices[i][k];
      if (vertices[i][k] > hi[k]) hi[k] = vertices[i][k];
      if (scene.mirror && k === 1) {
        if (-vertices[i][k] < lo[k]) lo[k] = -vertices[i][k];
        if (-vertices[i][k] > hi[k]) hi[k] = -vertices[i][k];
      }
    }
  }
  const center = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
  let radius = 1e-6;
  for (let i = 0; i < vertices.length; i += 1) {
    if (used && !used[i]) continue;
    const d = Math.hypot(vertices[i][0] - center[0], vertices[i][1] - center[1],
                         vertices[i][2] - center[2]);
    if (d > radius) radius = d;
    if (scene.mirror) {
      const mirroredDistance = Math.hypot(vertices[i][0] - center[0], -vertices[i][1] - center[1],
                                          vertices[i][2] - center[2]);
      if (mirroredDistance > radius) radius = mirroredDistance;
    }
  }
  viewFit.center = center;
  viewFit.radius = radius;
}

function drawAxes(width, height) {
  const origin = [64, height - 64];
  const length = 34;
  const axes = [['X', [1, 0, 0], '#e06c75'], ['Y', [0, 1, 0], '#98c379'], ['Z', [0, 0, 1], '#61afef']];
  ctx.save();
  ctx.lineWidth = 1.4;
  ctx.font = '11px sans-serif';
  axes.forEach((item) => {
    const p = viewMatrix(item[1]);
    const ex = origin[0] + p.sx * length;
    const ey = origin[1] + p.sy * length;
    ctx.strokeStyle = item[2];
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.fillStyle = item[2];
    ctx.fillText(item[0], origin[0] + p.sx * (length + 9) - 3, origin[1] + p.sy * (length + 9) + 4);
  });
  ctx.fillStyle = '#3d4756';
  ctx.beginPath();
  ctx.arc(origin[0], origin[1], 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function resize() {
  const ratio = window.devicePixelRatio || 1;
  const box = view.getBoundingClientRect();
  view.width = Math.max(1, Math.round(box.width * ratio));
  view.height = Math.max(1, Math.round(box.height * ratio));
  draw();
}
window.addEventListener('resize', resize);

let dragging = null;
view.addEventListener('mousedown', (event) => { dragging = { x: event.clientX, y: event.clientY }; });
window.addEventListener('mouseup', () => { dragging = null; });
window.addEventListener('mousemove', (event) => {
  if (!dragging) return;
  cam.yaw += (event.clientX - dragging.x) * 0.01;
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch - (event.clientY - dragging.y) * 0.01));
  dragging = { x: event.clientX, y: event.clientY };
  draw();
});
view.addEventListener('wheel', (event) => {
  event.preventDefault();
  cam.zoom = Math.max(0.25, Math.min(8, cam.zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
  draw();
}, { passive: false });

function viewMatrix(point) {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const x = point[0] * cy - point[1] * sy;
  const y = point[0] * sy + point[1] * cy;
  const z = point[2];
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  return { sx: x, sy: -(z * cp - y * sp), depth: y * cp + z * sp };
}

function shade(normal) {
  const light = [0.35, -0.75, 0.55];
  const dot = Math.abs(normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]);
  return 0.42 + 0.58 * dot;
}

function normalOf(a, b, c) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const len = Math.hypot(n[0], n[1], n[2]) || 1;
  return [n[0] / len, n[1] / len, n[2] / len];
}

/* colour ramp: blue (suction, low Cp) -> red (high Cp) */
function cpColor(value, lo, hi, alpha) {
  const t = hi > lo ? Math.max(0, Math.min(1, (value - lo) / (hi - lo))) : 0.5;
  const stops = [[0.0, 43, 63, 191], [0.25, 35, 165, 216], [0.5, 57, 196, 107],
                 [0.75, 232, 210, 58], [1.0, 224, 69, 47]];
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i += 1;
  const a = stops[i], b = stops[i + 1];
  const f = (t - a[0]) / (b[0] - a[0] || 1);
  const c = [0, 1, 2].map((k) => Math.round(a[k + 1] + (b[k + 1] - a[k + 1]) * f));
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (alpha === undefined ? 1 : alpha) + ')';
}

function sceneGeometry() {
  if (!state.build || !state.build.mesh) return null;
  return { vertices: state.build.mesh.vertices, faces: state.build.mesh.faces, mode: 'plain', mirror: false };
}
function sceneMesh() {
  if (!state.mesh) return null;
  return { vertices: state.mesh.coordinates, faces: state.mesh.faces,
           mode: 'mesh', mirror: true, fitFacesOnly: true };
}
function sceneField() {
  const field = state.result && state.result.field;
  if (!field || !field.cp || !field.cp.length) return null;
  return { vertices: field.vertices, faces: field.faces, cp: field.cp,
           wake: field.wake_faces || [], mode: 'cp', mirror: true, fitFacesOnly: true };
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = view.width / ratio, height = view.height / ratio;
  ctx.fillStyle = '#0d1116';
  ctx.fillRect(0, 0, width, height);
  let scene = sceneGeometry();
  if (state.tab === 'mesh') scene = sceneMesh();
  if (state.tab === 'result') scene = sceneField() || sceneGeometry();
  el('viewEmpty').style.display = scene ? 'none' : 'flex';
  if (!scene) {
    el('viewEmpty').textContent = state.tab === 'mesh' ? '点击「生成网格」' : '先点「生成几何」';
    drawAxes(width, height);
    return;
  }
  const key = state.tab + '|' + (state.build ? state.build.job : '') + '|' + scene.mode + '|' + scene.vertices.length;
  if (viewFit.key !== key) { fitScene(scene); viewFit.key = key; }
  renderScene(scene, width, height);
  drawAxes(width, height);
  el('legend').classList.toggle('hidden', !(scene.mode === 'cp'));
}

function renderScene(scene, width, height) {
  const vertices = scene.vertices;
  const count = vertices.length;
  const center = viewFit.center;
  const toLocal = (p) => [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
  const projected = new Array(count);
  for (let i = 0; i < count; i += 1) projected[i] = viewMatrix(toLocal(vertices[i]));
  const scale = 0.46 * Math.min(width, height) / viewFit.radius * cam.zoom;

  const faces = [];
  const push = (vertices2, faces2, offset, mirror) => {
    for (let f = 0; f < faces2.length; f += 1) {
      const face = faces2[f];
      const a = vertices2[face[0]], b = vertices2[face[1]], c = vertices2[face[2]];
      const n = normalOf(a, b, c);
      if (mirror) n[1] = -n[1];
      const depth = (projected[offset + face[0]].depth + projected[offset + face[1]].depth + projected[offset + face[2]].depth) / 3;
      faces.push({ face: face, offset: offset, depth: depth, normal: n, index: f });
    }
  };
  push(vertices, scene.faces, 0, false);
  let mirrored = null;
  if (scene.mirror) {
    mirrored = vertices.map((p) => [p[0], -p[1], p[2]]);
    const offset = count;
    for (let i = 0; i < count; i += 1) projected.push(viewMatrix(toLocal(mirrored[i])));
    push(mirrored, scene.faces, offset, true);
  }
  faces.sort((a, b) => b.depth - a.depth);

  if (scene.cp) {
    const field = state.result.field;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < field.cp.length; i += 1) {
      lo = Math.min(lo, field.cp[i]); hi = Math.max(hi, field.cp[i]);
    }
    el('cpMax').textContent = lo.toFixed(2);
    el('cpMin').textContent = hi.toFixed(2);
  }
  const cpLo = scene.cp ? Math.min.apply(null, scene.cp) : 0;
  const cpHi = scene.cp ? Math.max.apply(null, scene.cp) : 1;

  const toScreen = (p) => [width / 2 + p.sx * scale, height / 2 + p.sy * scale];

  for (let i = 0; i < faces.length; i += 1) {
    const item = faces[i];
    const pts = item.face.map((index) => toScreen(projected[item.offset + index]));
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < pts.length; k += 1) ctx.lineTo(pts[k][0], pts[k][1]);
    ctx.closePath();
    if (scene.mode === 'cp') {
      ctx.fillStyle = cpColor(scene.cp[item.index], cpLo, cpHi, 0.95);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.18)';
      ctx.lineWidth = 0.4;
      ctx.stroke();
    } else {
      const s = shade(item.normal);
      ctx.fillStyle = 'rgb(' + Math.round(238 * s) + ',' + Math.round(242 * s) + ',' + Math.round(248 * s) + ')';
      ctx.fill();
      ctx.strokeStyle = 'rgba(120,135,155,0.35)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  if (scene.mirror && scene.wake && scene.wake.length) {
    const wake = scene.wake;
    const wakeFaces = [];
    const collectWake = (vertices2, offset, mirror) => {
      for (let f = 0; f < wake.length; f += 1) {
        wakeFaces.push({ face: wake[f], offset: offset, depth: 0, mirror: mirror });
      }
    };
    collectWake(vertices, 0, false);
    if (mirrored) collectWake(mirrored, count, true);
    ctx.save();
    for (const item of wakeFaces) {
      const pts = item.face.map((index) => toScreen(projected[item.offset + index]));
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < pts.length; k += 1) ctx.lineTo(pts[k][0], pts[k][1]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(90,170,255,0.10)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(90,170,255,0.28)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.restore();
  }
}

/* ---------------- results ---------------- */
function setResultRow(label, value, warn) {
  const table = el('resultTable');
  const tr = document.createElement('tr');
  if (warn) tr.className = 'warn';
  if (warn) {
    const cell = document.createElement('td');
    cell.colSpan = 2;
    cell.textContent = label ? label + ' ' + value : value;
    tr.appendChild(cell);
  } else {
    const a = document.createElement('td');
    a.textContent = label;
    const b = document.createElement('td');
    b.textContent = value;
    tr.appendChild(a);
    tr.appendChild(b);
  }
  table.appendChild(tr);
}

function renderResult() {
  const table = el('resultTable');
  table.innerHTML = '';
  const result = state.result;
  if (!result) { el('resultMode').textContent = ''; el('resultMode').className = 'badge'; return; }
  el('resultMode').textContent = result.mode === 'free' ? '自由尾迹' : '固定尾迹';
  el('resultMode').className = 'badge' + (result.mode === 'free' ? ' free' : '');
  const solveSeconds = result.runner_seconds === undefined ? result.seconds : result.runner_seconds;
  el('resultTime').textContent = solveSeconds ? Number(solveSeconds).toFixed(1) + ' s' : '';
  const fmt = (v, digits) => (v === null || v === undefined ? '-' : Number(v).toFixed(digits === undefined ? 5 : digits));
  setResultRow('CL', fmt(result.CL));
  if (result.wall_formulation) setResultRow('壁面离散', result.wall_formulation === 'galerkin' ? 'P1 弱形式' : 'P1 配点');
  setResultRow('CD', fmt(result.CD));
  if (Array.isArray(result.CM)) setResultRow('CM_y', fmt(result.CM[1]));
  else setResultRow('CM_y', fmt(result.CM));
  setResultRow('壁面面板', result.panels === undefined ? '-' : String(result.panels));
  setResultRow('未知量', result.unknowns === undefined ? '-' : String(result.unknowns));
  setResultRow('残差', result.residual === undefined || result.residual === null ? '-' : Number(result.residual).toExponential(2));
  setResultRow('求解耗时', solveSeconds ? Number(solveSeconds).toFixed(2) + ' s' : '-');
  const options = result.options || {};
  const used = options.device_used;
  const requested = options.device_requested;
  const deviceText = !used ? '-' : (requested && requested !== used
    ? used + '（请求 ' + requested + '，不可用已回退）' : used);
  setResultRow('设备', deviceText, Boolean(used && requested && requested !== used));
  if (result.ncpu !== undefined) setResultRow('CPU 线程', String(result.ncpu));
  if (result.warning) setResultRow('', result.warning, true);
  if (result.device_note) setResultRow('', result.device_note, true);
  drawPlot();
}

function drawPlot() {
  const canvas = el('plot');
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  if (box.width < 10 || box.height < 10) return;
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  const c = canvas.getContext('2d');
  c.setTransform(ratio, 0, 0, ratio, 0, 0);
  c.clearRect(0, 0, box.width, box.height);
  const result = state.result;
  if (!result) return;
  const rows = result.group_rows || [];
  const width = box.width, height = box.height;
  const pad = 26;
  c.strokeStyle = '#2a3441';
  c.lineWidth = 1;
  c.beginPath();
  c.moveTo(pad, 8); c.lineTo(pad, height - 18); c.lineTo(width - 6, height - 18);
  c.stroke();
  c.fillStyle = '#6b7789';
  c.font = '10px sans-serif';

  if (result.mode === 'free' && result.history && result.history.length) {
    const values = result.history.map((r) => r.CL).filter((v) => v !== null && v !== undefined);
    if (!values.length) return;
    const lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    const span = Math.max(hi - lo, 1e-6);
    c.strokeStyle = '#5ea8ff';
    c.beginPath();
    values.forEach((value, i) => {
      const x = pad + (width - pad - 8) * (values.length === 1 ? 0.5 : i / (values.length - 1));
      const y = (height - 18) - (height - 30) * (value - lo) / span;
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    });
    c.stroke();
    c.fillStyle = '#8d9aab';
    c.fillText('CL 迭代历史', pad + 4, 12);
    c.fillText(hi.toFixed(3), 2, 16);
    c.fillText(lo.toFixed(3), 2, height - 20);
    return;
  }
  if (!rows.length) return;
  const gamma = rows[0].gamma || [];
  if (!gamma.length) return;
  let lo = Infinity, hi = -Infinity;
  gamma.forEach((pair) => { lo = Math.min(lo, pair[1]); hi = Math.max(hi, pair[1]); });
  const span = Math.max(hi - lo, 1e-9);
  c.strokeStyle = '#58c98a';
  c.beginPath();
  gamma.forEach((pair, i) => {
    const x = pad + (width - pad - 8) * pair[0];
    const y = (height - 18) - (height - 30) * (pair[1] - lo) / span;
    if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
  });
  c.stroke();
  c.fillStyle = '#8d9aab';
  c.fillText('Γ 展向分布', pad + 4, 12);
  c.fillText(hi.toExponential(2), 2, 16);
  c.fillText(lo.toExponential(2), 2, height - 20);
}

/* ---------------- build / solve ---------------- */
function setBusy(value) {
  state.busy = value;
  ['build', 'mesh', 'solve', 'addWing', 'addBody', 'delPart'].forEach((id) => { el(id).disabled = value; });
  if (!value) {
    renderParts();
    el('addWing').disabled = state.parts.some((p) => p.kind === 'wing');
  }
  updateHint();
}

async function buildGeometry() {
  try {
    const payload = { parts: state.parts.map((p) => ({ kind: p.kind, params: p.params })) };
    const data = await api('/api/build', payload);
    state.build = data;
    state.jobId = data.job;
    state.result = null;
    state.mesh = null;
    state.dirty = false;
    renderResult();
    logLine('几何已生成：体积 ' + Number(data.step.volume).toFixed(4) + ' m³，' +
            data.step.sections + ' 个截面' + (data.bodies && data.bodies.length ? '，' + data.bodies.length + ' 个机身' : ''));
    if (data.clamped && data.clamped.length) logLine('参数越界已截断：' + data.clamped.join(', '));
    draw();
    return true;
  } catch (error) {
    logLine('[错误] ' + error.message);
    return false;
  }
}

async function doBuild() {
  clearLog();
  setBusy(true);
  try {
    await buildGeometry();
  } finally {
    setBusy(false);
  }
}

async function doMesh() {
  if (!state.parts.some((p) => p.kind === 'wing')) { logLine('[错误] 需要至少一个升力面'); return; }
  setBusy(true);
  try {
    if (!state.jobId || state.dirty) {
      clearLog();
      const ok = await buildGeometry();
      if (!ok) return;
    }
    const body = {
      job: state.jobId,
      alpha: Number(el('alpha').value),
      size: Number(el('size').value),
    };
    state.result = null;
    state.mesh = null;
    renderResult();
    clearLog();
    logLine('网格生成中：尺寸=' + body.size + ' m，迎角=' + body.alpha + '°');
    await api('/api/mesh', body);
    await pollMesh();
  } catch (error) {
    logLine('[错误] ' + error.message);
  } finally {
    setBusy(false);
  }
}

async function doSolve() {
  if (!state.parts.some((p) => p.kind === 'wing')) { logLine('[错误] 需要至少一个升力面'); return; }
  if (!state.mesh || state.dirty || state.mesh.size !== Number(el('size').value) ||
      state.mesh.alpha !== Number(el('alpha').value)) {
    logLine('[错误] 请先为当前几何、网格尺寸和迎角生成网格');
    return;
  }
  setBusy(true);
  try {
    const body = {
      job: state.jobId,
      alpha: Number(el('alpha').value),
      size: Number(el('size').value),
      wake: el('wake').value,
      wall_formulation: el('wallFormulation').value,
      iterations: Number(el('iterations').value),
      device: el('device').value,
      ncpu: Number(el('ncpu').value),
    };
    clearLog();
    logLine('求解中：α=' + body.alpha + '°，尾迹=' + (body.wake === 'free' ? '自由' : '固定'));
    await api('/api/solve', body);
    await pollJob();
  } catch (error) {
    logLine('[错误] ' + error.message);
  } finally {
    setBusy(false);
  }
}

function pollMesh() {
  return new Promise((resolve) => {
    const tick = async () => {
      let data;
      try {
        data = await api('/api/job?id=' + encodeURIComponent(state.jobId));
      } catch (error) {
        logLine('[错误] ' + error.message);
        resolve();
        return;
      }
      renderLog(data.log || []);
      if (data.state === 'meshed') {
        await loadMesh();
        state.tab = 'mesh';
        document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'mesh'));
        updateHint();
        draw();
        resolve();
        return;
      }
      if (data.state === 'error' || data.state === 'cancelled') {
        logLine(data.error || '已取消');
        resolve();
        return;
      }
      setTimeout(tick, 700);
    };
    tick();
  });
}

function pollJob() {
  return new Promise((resolve) => {
    const tick = async () => {
      let data;
      try {
        data = await api('/api/job?id=' + encodeURIComponent(state.jobId));
      } catch (error) {
        logLine('[错误] ' + error.message);
        resolve();
        return;
      }
      renderLog(data.log || []);
      if (data.state === 'done') {
        if (!state.mesh) await loadMesh();
        state.result = data.result;
        state.tab = 'result';
        document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'result'));
        renderResult();
        draw();
        resolve();
        return;
      }
      if (data.state === 'solving' && !state.mesh) await loadMesh();
      if (data.state === 'error' || data.state === 'cancelled') {
        if (!state.mesh) await loadMesh();
        logLine(data.error || '已取消');
        resolve();
        return;
      }
      setTimeout(tick, 700);
    };
    tick();
  });
}

async function loadMesh() {
  if (!state.jobId || state.meshLoading) return;
  state.meshLoading = true;
  try {
    state.mesh = await api('/api/mesh?id=' + encodeURIComponent(state.jobId));
    draw();
    updateHint();
  } catch (error) {
    state.meshError = error.message;
  } finally {
    state.meshLoading = false;
  }
}

let logCache = [];
function renderLog(lines) {
  if (lines.length === logCache.length && lines.slice(-1)[0] === logCache.slice(-1)[0]) return;
  logCache = lines.slice();
  el('log').textContent = lines.join('\n');
  el('log').scrollTop = el('log').scrollHeight;
}

async function doCancel() {
  if (!state.jobId) return;
  try {
    await api('/api/cancel', { job: state.jobId });
    logLine('已请求取消…');
  } catch (error) {
    logLine('[错误] ' + error.message);
  }
}

/* ---------------- tabs ---------------- */
document.querySelectorAll('.tab').forEach((button) => {
  button.addEventListener('click', () => {
    state.tab = button.dataset.tab;
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b === button));
    if (state.tab === 'mesh' && !state.mesh) loadMesh();
    updateHint();
    draw();
  });
});

el('addWing').addEventListener('click', () => addPart('wing'));
el('addBody').addEventListener('click', () => addPart('body'));
el('delPart').addEventListener('click', () => {
  if (!state.parts.length) return;
  const part = state.parts[state.selected];
  if (part.kind === 'wing') { logLine('升力面是求解器必需部件，不能删除'); return; }
  state.parts.splice(state.selected, 1);
  state.selected = Math.max(0, Math.min(state.selected, state.parts.length - 1));
  renderParts(); renderEditor(); markDirty();
});
el('build').addEventListener('click', () => doBuild());
el('mesh').addEventListener('click', () => doMesh());
el('solve').addEventListener('click', () => doSolve());
el('cancel').addEventListener('click', () => doCancel());
['alpha', 'size'].forEach((id) => el(id).addEventListener('input', updateHint));

async function boot() {
  try {
    const schema = await api('/api/schema');
    state.schema = schema.params || [];
    state.bodySchema = schema.body_params || [];
    state.bodyDefaults = schema.body_defaults || {};
    el('ncpu').value = schema.solve_threads || 8;
    el('ncpu').max = schema.cpu_count || 128;
    const devices = schema.devices || [{ id: 'cpu', label: 'torch CPU', available: true }];
    const select = el('device');
    select.innerHTML = '';
    devices.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.label + (item.available ? '' : '（不可用）');
      option.disabled = !item.available;
      select.appendChild(option);
    });
    select.value = (devices.filter((item) => item.available)[0] || { id: 'cpu' }).id;
    state.parts = [{ kind: 'wing', name: '主翼', params: wingDefaults() }];
    state.selected = 0;
    renderParts(); renderEditor();
    el('server').textContent = '就绪 · runs: ' + (schema.runs || '').split(/[\\/]/).slice(-2).join('/');
    el('server').className = 'status ok';
    resize();
    doBuild();
  } catch (error) {
    el('server').textContent = '无法连接：' + error.message;
    el('server').className = 'status bad';
  }
}
boot();
