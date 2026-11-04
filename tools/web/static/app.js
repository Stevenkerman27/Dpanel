'use strict';

const el = (id) => document.getElementById(id);
const state = {
  schema: [], bodySchema: [], bodyDefaults: {},
  parts: [], selected: 0,
  build: null, result: null, jobId: null,
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
  else hint.textContent = state.result && state.result.field ? 'Cp 云图 · 半域镜像显示' : '先求解一次';
}

/* ---------------- viewer ---------------- */
const view = el('view');
const ctx = view.getContext('2d');
const cam = { yaw: -0.85, pitch: 0.32, zoom: 1 };

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
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + (event.clientY - dragging.y) * 0.01));
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
function sceneField() {
  const field = state.result && state.result.field;
  if (!field || !field.cp || !field.cp.length) return null;
  return { vertices: field.vertices, faces: field.faces, cp: field.cp,
           wake: field.wake_faces || [], mode: 'cp', mirror: true };
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = view.width / ratio, height = view.height / ratio;
  ctx.fillStyle = '#0d1116';
  ctx.fillRect(0, 0, width, height);
  const scene = state.tab === 'result' ? (sceneField() || sceneGeometry()) : sceneGeometry();
  el('viewEmpty').style.display = scene ? 'none' : 'flex';
  if (!scene) return;
  renderScene(scene, width, height);
  el('legend').classList.toggle('hidden', !(scene.mode === 'cp'));
}

function renderScene(scene, width, height) {
  const vertices = scene.vertices;
  const count = vertices.length;
  const projected = new Array(count);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i += 1) {
    const p = viewMatrix(vertices[i]);
    projected[i] = p;
    if (p.sx < minX) minX = p.sx;
    if (p.sx > maxX) maxX = p.sx;
    if (p.sy < minY) minY = p.sy;
    if (p.sy > maxY) maxY = p.sy;
  }
  const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6);
  const scale = 0.88 * Math.min(width / spanX, height / spanY) * cam.zoom;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;

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
    for (let i = 0; i < count; i += 1) {
      const p = viewMatrix(mirrored[i]);
      projected.push(p);
    }
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

  const toScreen = (p) => [width / 2 + (p.sx - cx) * scale, height / 2 + (p.sy - cy) * scale];

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
  el('resultTime').textContent = result.seconds_total ? result.seconds_total.toFixed(1) + ' s' : '';
  const fmt = (v, digits) => (v === null || v === undefined ? '-' : Number(v).toFixed(digits === undefined ? 5 : digits));
  setResultRow('CL', fmt(result.CL));
  setResultRow('CD', fmt(result.CD));
  if (Array.isArray(result.CM)) setResultRow('CM_y', fmt(result.CM[1]));
  else setResultRow('CM_y', fmt(result.CM));
  setResultRow('壁面面板', result.panels === undefined ? '-' : String(result.panels));
  setResultRow('未知量', result.unknowns === undefined ? '-' : String(result.unknowns));
  setResultRow('残差', result.residual === undefined || result.residual === null ? '-' : Number(result.residual).toExponential(2));
  setResultRow('求解耗时', result.seconds ? Number(result.seconds).toFixed(2) + ' s' : '-');
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
  ['build', 'solve', 'addWing', 'addBody', 'delPart'].forEach((id) => { el(id).disabled = value; });
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

async function doSolve() {
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
      wake: el('wake').value,
      iterations: Number(el('iterations').value),
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
        state.result = data.result;
        state.tab = 'result';
        document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'result'));
        renderResult();
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
el('solve').addEventListener('click', () => doSolve());
el('cancel').addEventListener('click', () => doCancel());

async function boot() {
  try {
    const schema = await api('/api/schema');
    state.schema = schema.params || [];
    state.bodySchema = schema.body_params || [];
    state.bodyDefaults = schema.body_defaults || {};
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
