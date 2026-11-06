'use strict';

const el = (id) => document.getElementById(id);
const state = {
  schema: [], bodySchema: [], bodyDefaults: {},
  parts: [], selected: 0,
  build: null, result: null, mesh: null, meshLoading: false, jobId: null,
  tab: 'geom', busy: false, dirty: true, poll: null, grassJobId: null,
  sections: null, sectionIndex: 0, optJobId: null, optResult: null,
};
const SECTION_FIELDS = ['leading_edge_xyz', 'chord', 'twist_radians', 'cst'];
function deepClone(value) { return JSON.parse(JSON.stringify(value)); }
function cloneSections(source) {
  const copy = {};
  SECTION_FIELDS.forEach((key) => { copy[key] = deepClone(source[key]); });
  return copy;
}

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
  return { grass_sections: cloneSections(state.sections) };
}
function addPart(kind) {
  if (kind === 'wing' && state.parts.some((p) => p.kind === 'wing')) return;
  const params = kind === 'wing' ? wingDefaults() : Object.assign({}, state.bodyDefaults);
  const count = state.parts.filter((p) => p.kind === kind).length + 1;
  state.parts.push({ kind, name: (kind === 'wing' ? '主翼' : '机身 ') + (kind === 'wing' ? '' : count), params });
  state.selected = state.parts.length - 1;
  renderParts();
  renderEditor();
  drawSectionView();
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
    li.addEventListener('click', () => {
      state.selected = index; renderParts(); renderEditor(); drawSectionView();
    });
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
  if (part.kind === 'wing') {
    renderSectionEditor(box);
    return;
  }
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

/* ---------------- section editor ---------------- */
function numberField(labelText, get, set, step, options) {
  const label = document.createElement('label');
  label.textContent = labelText;
  const input = document.createElement('input');
  input.type = 'number';
  input.step = step;
  input.value = formatNumber(get());
  if (options && options.disabled) { input.disabled = true; label.title = options.title || ''; }
  // 输入过程中只在值合法时立刻生效；清空、半截数字或非法值都不动文本，
  // 等 change（回车/失焦）才回退。否则删一个字符就 Number('')===0 被判非法，
  // 光标被踢回末尾，前缘 y 这类带递增校验的框根本改不动。
  const commit = (final) => {
    const raw = String(input.value).trim();
    const next = Number(raw);
    const result = (raw !== '' && Number.isFinite(next)) ? set(next) : { ok: false };
    if (result === undefined || result.ok !== false) return;
    if (!final) return;
    input.value = formatNumber(get());
    if (result.message) logLine('[提示] ' + result.message);
  };
  input.addEventListener('input', () => commit(false));
  input.addEventListener('change', () => commit(true));
  label.appendChild(input);
  return label;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return '';
  return String(Number(value.toFixed(6)));
}

function cstGroup(title, weightStart, box) {
  const head = document.createElement('div');
  head.className = 'cst-head';
  head.textContent = title;
  box.appendChild(head);
  const grid = document.createElement('div');
  grid.className = 'cst-grid';
  for (let k = 0; k < 8; k += 1) {
    grid.appendChild(numberField('w' + k,
      () => state.sections.cst[state.sectionIndex][weightStart + k],
      (value) => {
        state.sections.cst[state.sectionIndex][weightStart + k] = value;
        markDirty();
        drawSectionView();
        return { ok: true };
      }, 0.005));
  }
  box.appendChild(grid);
}

function renderSectionEditor(box) {
  const sections = state.sections;
  const count = sections.chord.length;
  state.sectionIndex = Math.max(0, Math.min(state.sectionIndex, count - 1));
  const index = state.sectionIndex;

  const tabs = document.createElement('div');
  tabs.className = 'sectabs';
  for (let i = 0; i < count; i += 1) {
    const button = document.createElement('button');
    button.className = 'sectab' + (i === index ? ' active' : '');
    const name = document.createElement('b');
    name.textContent = '截面 ' + (i + 1);
    const meta = document.createElement('span');
    meta.textContent = 'y=' + formatNumber(sections.leading_edge_xyz[i][1]) + ' · c=' +
      formatNumber(sections.chord[i]);
    button.appendChild(name);
    button.appendChild(meta);
    button.addEventListener('click', () => {
      state.sectionIndex = i;
      renderEditor();
      drawSectionView();
    });
    tabs.appendChild(button);
  }
  box.appendChild(tabs);

  const fields = document.createElement('div');
  fields.className = 'editor-fields';
  box.appendChild(fields);

  const locked = index === 0;
  ['前缘 x (m)', '前缘 y (m)', '前缘 z (m)'].forEach((label, axis) => {
    fields.appendChild(numberField(label,
      () => sections.leading_edge_xyz[index][axis],
      (value) => setLeadingEdge(index, axis, value),
      0.01, locked ? { disabled: true, title: '根部前缘固定在原点' } : null));
  });
  fields.appendChild(numberField('弦长 (m)',
    () => sections.chord[index],
    (value) => {
      if (!(value > 0.01)) return { ok: false, message: '弦长必须大于 0.01 m' };
      sections.chord[index] = value;
      markDirty();
      drawSectionView();
      refreshSectionTabs();
      return { ok: true };
    }, 0.01));
  fields.appendChild(numberField('扭转 (°)',
    () => sections.twist_radians[index] * 180 / Math.PI,
    (value) => {
      sections.twist_radians[index] = value * Math.PI / 180;
      markDirty();
      drawSectionView();
      return { ok: true };
    }, 0.1));

  cstGroup('上表面 CST 权重', 0, box);
  cstGroup('下表面 CST 权重', 8, box);
  const leadingWeight = document.createElement('div');
  leadingWeight.className = 'cst-grid single';
  leadingWeight.appendChild(numberField('前缘权重',
    () => sections.cst[index][16],
    (value) => {
      sections.cst[index][16] = value;
      markDirty();
      drawSectionView();
      return { ok: true };
    }, 0.005));
  box.appendChild(leadingWeight);

  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = locked
    ? '根部前缘固定在原点（改展长请切到截面 2 改前缘 y）；尾缘厚度固定为 0（尖尾缘）。'
    : '前缘 y 必须严格递增；尾缘厚度固定为 0（尖尾缘）。';
  box.appendChild(note);

  const row = document.createElement('div');
  row.className = 'row';
  const add = document.createElement('button');
  add.textContent = '+ 截面';
  add.disabled = count >= 4;
  add.addEventListener('click', () => mutateSectionCount(1));
  const remove = document.createElement('button');
  remove.className = 'danger';
  remove.textContent = '− 截面';
  remove.disabled = count <= 2;
  remove.addEventListener('click', () => mutateSectionCount(-1));
  row.appendChild(add);
  row.appendChild(remove);
  box.appendChild(row);
}

function setLeadingEdge(index, axis, value) {
  const sections = state.sections;
  if (index === 0) return { ok: false, message: '根部前缘固定在原点' };
  if (axis === 1) {
    const lower = sections.leading_edge_xyz[index - 1][1] + 1e-3;
    const upper = index + 1 < sections.chord.length
      ? sections.leading_edge_xyz[index + 1][1] - 1e-3 : Infinity;
    if (!(value > lower && value < upper)) {
      return { ok: false, message: '前缘 y 必须严格递增，可用区间 (' + lower.toFixed(3) + ', ' +
        (Number.isFinite(upper) ? upper.toFixed(3) : '∞') + ')' };
    }
  }
  sections.leading_edge_xyz[index][axis] = value;
  markDirty();
  drawSectionView();
  refreshSectionTabs();
  return { ok: true };
}

function refreshSectionTabs() {
  const nodes = document.querySelectorAll('#editor .sectab');
  nodes.forEach((node, i) => {
    const meta = node.querySelector('span');
    if (!meta) return;
    meta.textContent = 'y=' + formatNumber(state.sections.leading_edge_xyz[i][1]) + ' · c=' +
      formatNumber(state.sections.chord[i]);
  });
}

function mutateSectionCount(delta) {
  const sections = state.sections;
  const count = sections.chord.length;
  if (delta > 0) {
    if (count >= 4) return;
    const last = count - 1;
    sections.leading_edge_xyz.push([sections.leading_edge_xyz[last][0],
      sections.leading_edge_xyz[last][1] + 1, sections.leading_edge_xyz[last][2]]);
    sections.chord.push(sections.chord[last]);
    sections.twist_radians.push(sections.twist_radians[last]);
    sections.cst.push(deepClone(sections.cst[last]));
    state.sectionIndex = count;
  } else {
    if (count <= 2) return;
    sections.leading_edge_xyz.splice(state.sectionIndex, 1);
    sections.chord.splice(state.sectionIndex, 1);
    sections.twist_radians.splice(state.sectionIndex, 1);
    sections.cst.splice(state.sectionIndex, 1);
    state.sectionIndex = Math.max(0, state.sectionIndex - 1);
  }
  markDirty();
  renderEditor();
  drawSectionView();
}

/* ---------------- section side view ---------------- */
const cstView = el('sectionView');
const cstCtx = cstView.getContext('2d');

function cstValue(weights, leadingWeight, teThickness, x, sign) {
  let bernstein = 0;
  const y = 1 - x;
  let binomial = 1;
  for (let k = 0; k < 8; k += 1) {
    if (k > 0) binomial = binomial * (7 - (k - 1)) / k;
    bernstein += binomial * Math.pow(x, k) * Math.pow(y, 7 - k) * weights[k];
  }
  const shape = Math.sqrt(Math.max(x, 0)) * y * bernstein;
  const leading = leadingWeight * x * Math.pow(y, 7.5);
  return shape + leading + sign * x * teThickness / 2;
}

/* Section frames follow GRASS-plane/wing_geometry_encoder.py: the airfoil is
   rotated by (twist - root twist), the section z is rotated by the mean
   dihedral, and the whole wing is rotated by the root twist about the span axis.
   The side view drops the span coordinate and keeps (x, z). */
function sectionFrames(sections) {
  const le = sections.leading_edge_xyz;
  const count = sections.chord.length;
  const dihedral = [];
  const cumZ = [0];
  let y = 0;
  let z = 0;
  for (let i = 1; i < count; i += 1) {
    const span = le[i][1] - le[i - 1][1];
    const angle = Math.atan2(le[i][2] - le[i - 1][2], span);
    dihedral.push(angle);
    y += span * Math.cos(angle);
    z += span * Math.sin(angle);
    cumZ.push(z);
  }
  const frame = [0];
  for (let i = 1; i < count - 1; i += 1) frame.push((dihedral[i - 1] + dihedral[i]) / 2);
  if (count > 1) frame.push(dihedral[dihedral.length - 1]);
  return { frame, cumZ, rootTwist: sections.twist_radians[0] };
}

function projectSection(sections, frames, index, xLocal, zLocal) {
  const relative = sections.twist_radians[index] - frames.rootTwist;
  const cosR = Math.cos(relative);
  const sinR = Math.sin(relative);
  const localX = cosR * xLocal + sinR * zLocal;
  const localZ = -sinR * xLocal + cosR * zLocal;
  const x1 = sections.leading_edge_xyz[index][0] + localX;
  const z1 = frames.cumZ[index] + Math.cos(frames.frame[index]) * localZ;
  const cosT = Math.cos(frames.rootTwist);
  const sinT = Math.sin(frames.rootTwist);
  return [cosT * x1 + sinT * z1, -sinT * x1 + cosT * z1];
}

function sectionOutline(sections, frames, index, count) {
  const code = sections.cst[index];
  const chord = sections.chord[index];
  const upper = [];
  const lower = [];
  for (let i = 0; i <= count; i += 1) {
    const x = 0.5 * (1 - Math.cos(Math.PI * i / count));
    const yu = cstValue(code.slice(0, 8), code[16], code[17], x, 1);
    const yl = cstValue(code.slice(8, 16), code[16], code[17], x, -1);
    upper.push(projectSection(sections, frames, index, chord * x, chord * yu));
    lower.push(projectSection(sections, frames, index, chord * x, chord * yl));
  }
  return { upper, lower, leading: projectSection(sections, frames, index, 0, 0) };
}

function drawSectionView() {
  const part = state.parts[state.selected];
  const ratio = window.devicePixelRatio || 1;
  const width = cstView.clientWidth || 360;
  const height = cstView.clientHeight || 180;
  if (cstView.width !== Math.round(width * ratio) || cstView.height !== Math.round(height * ratio)) {
    cstView.width = Math.round(width * ratio);
    cstView.height = Math.round(height * ratio);
  }
  cstCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  cstCtx.clearRect(0, 0, width, height);
  cstCtx.fillStyle = '#0d1116';
  cstCtx.fillRect(0, 0, width, height);
  if (!part || part.kind !== 'wing' || !state.sections) {
    el('sectionCount').textContent = '';
    return;
  }
  const sections = state.sections;
  const count = sections.chord.length;
  const frames = sectionFrames(sections);
  const outlines = [];
  for (let i = 0; i < count; i += 1) outlines.push(sectionOutline(sections, frames, i, 61));
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  outlines.forEach((outline) => {
    outline.upper.concat(outline.lower).forEach((point) => {
      minX = Math.min(minX, point[0]); maxX = Math.max(maxX, point[0]);
      minZ = Math.min(minZ, point[1]); maxZ = Math.max(maxZ, point[1]);
    });
  });
  const pad = 14;
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanZ = Math.max(maxZ - minZ, 1e-6);
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanZ);
  const originX = pad + (width - 2 * pad - spanX * scale) / 2;
  const originY = pad + (height - 2 * pad - spanZ * scale) / 2;
  const map = (point) => [originX + (point[0] - minX) * scale, originY + (maxZ - point[1]) * scale];

  cstCtx.strokeStyle = 'rgba(90,110,135,.5)';
  cstCtx.setLineDash([3, 4]);
  cstCtx.beginPath();
  const [ax0, ay0] = map([minX, 0]);
  const [ax1, ay1] = map([maxX, 0]);
  cstCtx.moveTo(ax0, ay0); cstCtx.lineTo(ax1, ay1);
  cstCtx.stroke();
  cstCtx.setLineDash([]);

  outlines.forEach((outline, i) => {
    const current = i === state.sectionIndex;
    cstCtx.beginPath();
    outline.upper.forEach((point, k) => {
      const [x, y] = map(point);
      if (k === 0) cstCtx.moveTo(x, y); else cstCtx.lineTo(x, y);
    });
    for (let k = outline.lower.length - 1; k >= 0; k -= 1) {
      const [x, y] = map(outline.lower[k]);
      cstCtx.lineTo(x, y);
    }
    cstCtx.closePath();
    if (current) {
      cstCtx.fillStyle = 'rgba(79,195,247,.16)';
      cstCtx.fill();
      cstCtx.strokeStyle = '#4fc3f7';
      cstCtx.lineWidth = 2;
    } else {
      cstCtx.strokeStyle = 'rgba(150,166,186,.55)';
      cstCtx.lineWidth = 1;
    }
    cstCtx.stroke();
    const [lx, ly] = map(outline.leading);
    cstCtx.fillStyle = current ? '#8fdcff' : 'rgba(150,166,186,.75)';
    cstCtx.font = '10px Consolas, monospace';
    cstCtx.fillText(String(i + 1), lx - 2, ly + (i % 2 ? 12 : -6));
  });

  el('sectionCount').textContent = '共 ' + count + ' 个截面 · 当前 ' + (state.sectionIndex + 1) +
    ' · 弦长 ' + formatNumber(sections.chord[state.sectionIndex]) + ' m · 扭转 ' +
    formatNumber(sections.twist_radians[state.sectionIndex] * 180 / Math.PI) + '°';
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
window.addEventListener('resize', () => { resize(); drawSectionView(); });

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
           mode: 'mesh', mirror: state.mesh.mirror !== false, fitFacesOnly: true };
}
function sceneField() {
  const field = state.result && state.result.field;
  if (!field || !field.cp || !field.cp.length) return null;
  return { vertices: field.vertices, faces: field.faces, cp: field.cp,
           wake: field.wake_faces || [], mode: 'cp', mirror: field.mirror !== false, fitFacesOnly: true };
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
  if (state.tab !== 'result' && state.build && state.build.mesh_features) {
    const features = state.build.mesh_features.features || [];
    for (const feature of features) {
      const points = feature.points.map((point) => toScreen(viewMatrix(toLocal(point))));
      if (points.length < 2) continue;
      ctx.save();
      ctx.strokeStyle = '#ffb454';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i][0], points[i][1]);
      ctx.stroke();
      ctx.restore();
    }
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
  el('resultMode').textContent = result.mode === 'free'
    ? (result.wake_method === 'filament' ? '涡线自由尾迹' : '面元自由尾迹') : '固定尾迹';
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
  ['build', 'mesh', 'solve', 'grassSolve', 'optRun', 'optApply', 'addWing', 'addBody', 'delPart'].forEach((id) => { el(id).disabled = value; });
  if (!value) {
    renderParts();
    el('addWing').disabled = state.parts.some((p) => p.kind === 'wing');
    el('optApply').disabled = !state.optResult;
  }
  updateHint();
}

async function buildGeometry() {
  try {
    const payload = { parts: state.parts.map((p) => ({
      kind: p.kind, params: p.kind === 'wing' ? wingDefaults() : p.params,
    })) };
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
    const wakeLabel = {fixed: '固定', free: '自由（面元）', free_filament: '自由（涡线）'};
    logLine('求解中：α=' + body.alpha + '°，尾迹=' + (wakeLabel[body.wake] || body.wake));
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
  const job = state.optJobId || state.grassJobId || state.jobId;
  if (!job) return;
  try {
    await api('/api/cancel', { job });
    logLine('已请求取消…');
  } catch (error) {
    logLine('[错误] ' + error.message);
  }
}

el('grassSolve').addEventListener('click', async () => {
  if (!state.mesh || state.dirty || state.mesh.size !== Number(el('size').value) ||
      state.mesh.alpha !== Number(el('alpha').value)) {
    logLine('[错误] 请先为当前参数、网格尺寸和迎角生成网格');
    return;
  }
  if (el('wake').value !== 'fixed' || el('wallFormulation').value !== 'galerkin') {
    logLine('[错误] GRASS 梯度目前要求固定尾迹和 P1 弱形式');
    return;
  }
  setBusy(true);
  el('grassResult').textContent = '计算中…';
  try {
    const data = {job: state.jobId, alpha: Number(el('alpha').value)};
    const started = await api('/api/grass-gradient', data);
    state.grassJobId = started.job;
    while (true) {
      const job = await api('/api/job?id=' + encodeURIComponent(started.job));
      if (job.state === 'done') {
        el('grassResult').textContent = JSON.stringify({
          CL: job.result.CL, CD: job.result.CD, gradients: job.result.gradients,
          solver_success: job.result.success, solver_status: job.result.status,
          stationarity: job.result.stationarity, stationary_verified: job.result.stationary_verified,
          constraints: job.result.gradient_constraints,
        }, null, 2);
        break;
      }
      if (job.state === 'error' || job.state === 'cancelled') {
        throw new Error(job.error || '已取消');
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  } catch (error) {
    el('grassResult').textContent = '错误：' + error.message;
  } finally {
    state.grassJobId = null;
    setBusy(false);
  }
});

/* ---------------- section optimization ---------------- */
function optimizationReport(report) {
  const initial = report.initial || {};
  const final = report.final || {};
  const fmt = (value) => (value === null || value === undefined || !Number.isFinite(Number(value)))
    ? '-' : Number(value).toFixed(6);
  const change = (a, b) => {
    if (!Number.isFinite(Number(a)) || !Number.isFinite(Number(b)) || Number(a) === 0) return '';
    const ratio = (Number(b) - Number(a)) / Math.abs(Number(a));
    return ' (' + (ratio >= 0 ? '+' : '') + (100 * ratio).toFixed(1) + '%)';
  };
  const labels = { induced: '诱导阻力 CDi', pressure: '压差阻力 CD', lift: '升力 CL（最大化）' };
  const lines = [];
  lines.push('状态: ' + (report.success ? '收敛' : '未收敛') + ' — ' + (report.message || ''));
  lines.push('目标: ' + (labels[report.objective] || report.objective) + ' ' +
             fmt(initial.objective) + ' → ' + fmt(final.objective) +
             change(initial.objective, final.objective));
  lines.push('CL ' + fmt(initial.CL) + ' → ' + fmt(final.CL) +
             '   CD ' + fmt(initial.CD) + ' → ' + fmt(final.CD));
  lines.push('CDi ' + fmt(initial.CDi) + ' → ' + fmt(final.CDi) +
             '   参考面积 ' + fmt(initial.area) + ' → ' + fmt(final.area));
  lines.push('评估 ' + (report.evaluations || 0) + ' 次 · SLSQP 迭代 ' +
             (report.iterations || 0) + ' · 面板 ' + (report.panel_count || '-'));
  lines.push('气动精度尚未验收：结果只表示在当前离散网格上的改进量。');
  return lines.join('\n');
}

async function runOptimize() {
  if (!state.mesh || state.dirty || state.mesh.size !== Number(el('size').value) ||
      state.mesh.alpha !== Number(el('alpha').value)) {
    el('optResult').textContent = '请先为当前参数、网格尺寸和迎角生成网格';
    return;
  }
  const clText = el('optClMin').value.trim();
  const body = {
    job: state.jobId,
    alpha: Number(el('alpha').value),
    objective: el('optObjective').value,
    cl_min: clText === '' ? null : Number(clText),
    keep_area: el('optKeepArea').checked,
    iterations: Number(el('optIterations').value),
  };
  if (clText !== '' && !Number.isFinite(body.cl_min)) {
    el('optResult').textContent = '升力约束必须是数值';
    return;
  }
  if (!(body.iterations >= 1)) {
    el('optResult').textContent = '迭代次数至少为 1';
    return;
  }
  state.optResult = null;
  setBusy(true);
  el('optResult').textContent = '启动优化…';
  try {
    const started = await api('/api/optimize', body);
    state.optJobId = started.job;
    let last = -1;
    while (true) {
      const job = await api('/api/job?id=' + encodeURIComponent(started.job));
      renderLog(job.log || []);
      if ((job.log || []).length !== last) {
        last = (job.log || []).length;
        const tail = (job.log || []).slice(-1)[0];
        if (tail) el('optResult').textContent = tail;
      }
      if (job.state === 'done') {
        state.optResult = job.result;
        el('optResult').textContent = optimizationReport(job.result);
        el('optApply').disabled = false;
        break;
      }
      if (job.state === 'error' || job.state === 'cancelled') {
        throw new Error(job.error || '已取消');
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  } catch (error) {
    el('optResult').textContent = '错误：' + error.message;
  } finally {
    state.optJobId = null;
    setBusy(false);
  }
}

el('optRun').addEventListener('click', runOptimize);
el('optApply').addEventListener('click', () => {
  if (!state.optResult || !state.optResult.sections_final) return;
  state.sections = cloneSections(state.optResult.sections_final);
  state.sectionIndex = Math.min(state.sectionIndex, state.sections.chord.length - 1);
  state.optResult = null;
  el('optApply').disabled = true;
  markDirty();
  renderEditor();
  drawSectionView();
  logLine('已载入优化截面：请重新生成几何与网格后再求解');
});

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
  renderParts(); renderEditor(); drawSectionView(); markDirty();
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
    state.sections = cloneSections(schema.grass_defaults);
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
    renderParts(); renderEditor(); drawSectionView();
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
