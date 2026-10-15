/* Chương 4 — giao diện cho 4 trang: index (tổng quan) · kg (phòng thí nghiệm KG) · tieuduong / chungkhoan (RAG).
 * Thuật toán nằm trong core.js; file này chỉ nạp JSON tĩnh trong data/ và vẽ. */
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const PAGE = document.body.dataset.page;
const FWS = ['scratch', 'keras', 'torch'];
const FWN = { scratch: 'NumPy (scratch)', keras: 'Keras', torch: 'PyTorch' };
const FWC = { scratch: '#2a6f97', keras: '#c1121f', torch: '#ee8a2a' };
const KINDS = ['transe', 'distmult', 'complex'];
const KN = { transe: 'TransE', distmult: 'DistMult', complex: 'ComplEx' };
const MODES = ['llm', 'text', 'kg', 'hybrid'];
const MN = { llm: 'Chỉ LLM', text: 'RAG văn bản', kg: 'RAG đồ thị', hybrid: 'RAG lai' };
const MC = { llm: '#9e9e9e', text: '#1982c4', kg: '#e76f51', hybrid: '#2a9d8f' };
const cache = {};
const loadJSON = (p) => (cache[p] ||= fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}));
const fmt = (x, d = 3) => Number(x).toLocaleString('vi-VN', { minimumFractionDigits: d, maximumFractionDigits: d });
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tick = () => new Promise((r) => setTimeout(r, 0));
window.SELFTEST = { done: false, maxDiff: null, errors: [] };

// ---------------------------------------------------------------- giao diện sáng/tối + Chart.js
const CHARTS = {};
(() => {
  try { const t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* bỏ qua */ }
  $('#themeBtn')?.addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', document.documentElement.dataset.theme); } catch (_) { /* bỏ qua */ }
    Object.values(CHARTS).forEach((c) => { styleChart(c); c.update('none'); });
    if (PAGE === 'kg') KGLAB.draw?.();
  });
})();
function styleChart(c) {
  const text = cssVar('--text'), muted = cssVar('--muted'), grid = cssVar('--border');
  c.options.color = muted;
  for (const ax of Object.values(c.options.scales || {})) {
    ax.ticks = { ...(ax.ticks || {}), color: muted };
    ax.grid = { ...(ax.grid || {}), color: grid };
    if (ax.title) ax.title.color = muted;
  }
  if (c.options.plugins?.legend?.labels) c.options.plugins.legend.labels.color = text;
}
function makeChart(id, config) {
  if (typeof Chart === 'undefined' || !$(`#${id}`)) return null;
  CHARTS[id]?.destroy();
  config.options = { responsive: true, maintainAspectRatio: false, animation: false, ...config.options };
  config.options.plugins = { legend: { labels: {} }, ...(config.options.plugins || {}) };
  const c = new Chart($(`#${id}`), config);
  styleChart(c);
  c.update('none');
  CHARTS[id] = c;
  return c;
}
function table(head, rows, cls = 'cmp') {
  return `<table class="${cls}"><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${
    rows.map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.cells.map((c, i) => `<td${i && typeof c === 'number' ? ' class="num"' : ''}>${
      typeof c === 'number' ? fmt(c) : c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

// ================================================================= INDEX
async function initIndex() {
  const S = await loadJSON('data/summary.json');
  const k = S.kpi;
  $('#kpis').innerHTML = [
    ['Đồ thị tri thức', '3', 'CoDEx-S · Hetionet-DM · SP500-KG'],
    ['Lần huấn luyện KGE', '27', '3 mô hình × 3 nền tảng × 3 đồ thị'],
    ['MRR tốt nhất (CoDEx-S)', fmt(k.codex_best, 3), 'bài báo gốc: ComplEx 0,465'],
    ['RAG lai — tiểu đường', fmt(k.dm_hybrid, 2), `F1 câu hỏi KG (chỉ LLM: ${fmt(k.dm_llm, 2)})`],
    ['RAG lai — chứng khoán', fmt(k.sp_hybrid, 2), `điểm TB (chỉ LLM: ${fmt(k.sp_llm, 2)})`],
  ].map(([a, b, c], i) => `<div class="kpi${i === 3 ? ' accent' : ''}"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join('');
  $('#kgNum').innerHTML = `<b>${fmt(k.hetio_best, 3)}</b> MRR Hetionet-DM`;
  $('#dmNum').innerHTML = `<b>${S.n.dm_q}</b> câu hỏi đánh giá`;
  $('#spNum').innerHTML = `<b>${S.n.sp_q}</b> câu hỏi đánh giá`;
  const terms = Object.keys(S.openalex);
  const col = { 'semantic web': '#8d99ae', ontology: '#adb5bd', 'knowledge graph': '#e76f51', 'knowledge graph embedding': '#6a4c93',
    'retrieval-augmented generation': '#1982c4', GraphRAG: '#2a9d8f' };
  makeChart('oaChart', {
    type: 'line',
    data: { labels: S.years, datasets: terms.map((t) => ({ label: t, data: S.openalex[t].map((v) => v || null), borderColor: col[t],
      backgroundColor: col[t], pointRadius: 1.5, borderWidth: t === 'ontology' ? 1.2 : 2.2, borderDash: t === 'ontology' ? [4, 3] : [] })) },
    options: { scales: { y: { type: 'logarithmic', title: { display: true, text: 'số công bố / năm' } } },
      plugins: { legend: { labels: { boxWidth: 12, font: { size: 10 } } } } },
  });
  const kg = S.kge;
  let h = '<h3 style="font-size:.92rem">Nhúng đồ thị — MRR có lọc trên tập test</h3>' + table(
    ['Đồ thị', 'Mô hình', ...FWS.map((f) => FWN[f])],
    Object.keys(kg).flatMap((ds) => KINDS.map((m) => ({ cells: [ds, KN[m], ...FWS.map((f) => kg[ds][`${m}_${f}`])] }))));
  h += '<h3 style="font-size:.92rem;margin-top:18px">Truy hồi — nDCG@10 trên tập test</h3>' + table(
    ['Phương pháp', 'MedQuAD', 'MedQuAD (tiểu đường)', 'FiQA'],
    S.ir.map((r) => ({ cells: [r.method, r.medquad, r.medquad_dm, r.fiqa] })));
  h += '<h3 style="font-size:.92rem;margin-top:18px">Hỏi–đáp với LLM ở 4 chế độ</h3>' + table(
    ['Chế độ', 'Tiểu đường: F1 câu hỏi KG', 'Tiểu đường: cosine MedQuAD', 'Chứng khoán: điểm TB'],
    MODES.map((m) => ({ cells: [MN[m], S.rag.dm[m].f1, S.rag.dm[m].MQ_sem, S.rag.sp[m].score] })));
  $('#mainTables').innerHTML = h;
}

// ================================================================= KG LAB
const TYPE_COL = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261', '#8d99ae', '#b5838d', '#6d597a', '#e9c46a', '#1d3557', '#ffb703',
  '#90be6d', '#43aa8b', '#577590', '#f94144', '#adb5bd'];
const KGLAB = {};
const KGDATA = {};
const getKG = async (ds) => (KGDATA[ds] ||= await loadKG(ds));
async function loadKG(ds) {
  const J = await loadJSON(`data/kg_${ds}.json`);
  if (!J._ready) {
    J.N = J.ents.length; J.nRel = J.rels.length;
    J.T = Int32Array.from(J.triples);
    J.emb = {};
    for (const m of KINDS) J.emb[m] = { E: Core.f16(J.models[m].E), R: Core.f16(J.models[m].R), d: J.models[m].dim };
    J.out = new Map(); J.inn = new Map();
    for (let j = 0; j < J.T.length / 3; j++) {
      const h = J.T[3 * j], r = J.T[3 * j + 1], t = J.T[3 * j + 2];
      (J.out.get(h) || J.out.set(h, []).get(h)).push([r, t]);
      (J.inn.get(t) || J.inn.set(t, []).get(t)).push([r, h]);
    }
    J.typeList = [...new Set(J.types)];
    J.deg = (i) => (J.out.get(i)?.length || 0) + (J.inn.get(i)?.length || 0);
    J.lname = J.names.map((n) => String(n).toLowerCase());
    J._ready = true;
  }
  return J;
}
const shortRel = (s) => { const p = String(s).split(/ [-<>] /); return p.length === 3 ? p[1] : s; };

async function initKG() {
  const st = { ds: 'hetio_dm', kind: 'complex', center: null };
  const DEFAULT = { hetio_dm: 'Disease::DOID:9352', sp500: 'TICKER:NEM', codex: 'Q90' };
  async function setDS(ds) {
    st.ds = ds;
    $('#dsInfo').textContent = 'đang nạp…';
    const J = await getKG(ds);
    $('#dsInfo').textContent = `${J.N.toLocaleString('vi-VN')} thực thể · ${J.nRel} quan hệ · ${(J.T.length / 3).toLocaleString('vi-VN')} bộ ba`;
    $('#legend').innerHTML = J.typeList.length > 1 ? J.typeList.map((t, i) => `<span><span class="dot" style="background:${TYPE_COL[i % 15]}"></span>${esc(t)}</span>`).join('') : '';
    metricView(J);
    const c = J.ents.indexOf(DEFAULT[ds]);
    setCenter(c >= 0 ? c : 0);
  }
  function setCenter(i) {
    st.center = i;
    const J = KGDATA[st.ds];
    $('#entInp').value = J.names[i];
    $('#sugg').innerHTML = '';
    $('#gTitle').textContent = `Lân cận của «${J.names[i]}»` + (J.types[i] && st.ds !== 'codex' ? ` [${J.types[i]}]` : '');
    buildGraph(J, i);
    const rels = new Map();
    for (const [r] of J.out.get(i) || []) rels.set(r, (rels.get(r) || 0) + 1);
    for (const [r] of J.inn.get(i) || []) rels.set(r + J.nRel, (rels.get(r + J.nRel) || 0) + 1);
    const opts = [...rels.keys()].sort((a, b) => rels.get(b) - rels.get(a));
    for (let r = 0; r < 2 * J.nRel; r++) if (!rels.has(r)) opts.push(r);
    $('#relSel').innerHTML = opts.map((r) => `<option value="${r}">${r < J.nRel ? '' : '← '}${esc(J.rel_names[r % J.nRel])}${
      rels.has(r) ? ` (${rels.get(r)} đã biết)` : ''}</option>`).join('');
    predict();
  }
  function predict() {
    const J = KGDATA[st.ds];
    const r = +$('#relSel').value, h = st.center, E = J.emb[st.kind];
    const S = Core.kgeScores(st.kind, E.E, E.R, E.d, h, r);
    const known = new Set((r < J.nRel ? J.out.get(h) || [] : J.inn.get(h) || []).filter(([rr]) => rr === r % J.nRel).map(([, t]) => t));
    const order = Core.argsortDesc(S).filter((i) => i !== h).slice(0, 12);
    $('#predHead').innerHTML = r < J.nRel ? `(${esc(J.names[h])}, ${esc(J.rel_names[r])}, <b>?</b>) — ${KN[st.kind]}`
      : `(<b>?</b>, ${esc(J.rel_names[r - J.nRel])}, ${esc(J.names[h])}) — ${KN[st.kind]}`;
    $('#pred').innerHTML = order.map((i) => `<li><a href="#" data-e="${i}">${esc(J.names[i])}</a>${
      J.types[i] && st.ds !== 'codex' ? ` <small class="muted">${esc(J.types[i])}</small>` : ''}<span class="badge ${known.has(i) ? 'known">đã có' : 'new">mới'}</span><span class="s">${S[i].toFixed(2)}</span></li>`).join('');
  }
  // ----- đồ thị lân cận: bố cục lực đơn giản trên canvas
  function buildGraph(J, c) {
    const nodes = [{ id: c }], edges = [], seen = new Map([[c, 0]]), perRel = new Map();
    const add = (r, o, dir) => {
      const key = `${r}${dir}`;
      if ((perRel.get(key) || 0) >= 6) return;
      perRel.set(key, (perRel.get(key) || 0) + 1);
      if (!seen.has(o)) { seen.set(o, nodes.length); nodes.push({ id: o }); }
      edges.push({ a: dir === '>' ? 0 : seen.get(o), b: dir === '>' ? seen.get(o) : 0, label: shortRel(J.rel_names[r]) });
    };
    for (const [r, t] of J.out.get(c) || []) add(r, t, '>');
    for (const [r, h] of J.inn.get(c) || []) add(r, h, '<');
    const n = nodes.length;
    nodes.forEach((p, i) => { const a = 2 * Math.PI * i / Math.max(1, n - 1); p.x = i ? Math.cos(a) * 0.8 : 0; p.y = i ? Math.sin(a) * 0.8 : 0; });
    for (let it = 0; it < 220; it++) {
      for (let i = 1; i < n; i++) {
        let fx = 0, fy = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y, d2 = dx * dx + dy * dy + 1e-3;
          fx += 0.012 * dx / d2; fy += 0.012 * dy / d2;
        }
        const dc = Math.hypot(nodes[i].x, nodes[i].y) + 1e-6;
        fx += (0.75 - dc) * nodes[i].x / dc * 0.1; fy += (0.75 - dc) * nodes[i].y / dc * 0.1;
        nodes[i].x += Math.max(-0.05, Math.min(0.05, fx)); nodes[i].y += Math.max(-0.05, Math.min(0.05, fy));
      }
    }
    KGLAB.g = { J, nodes, edges };
    KGLAB.draw();
  }
  KGLAB.draw = () => {
    const { J, nodes, edges } = KGLAB.g || {};
    if (!J) return;
    const cv = $('#graph'), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const sx = (x) => W / 2 + x * (W / 2 - 70), sy = (y) => H / 2 + y * (H / 2 - 28);
    const text = cssVar('--text'), muted = cssVar('--muted');
    ctx.font = '10px "Be Vietnam Pro", sans-serif';
    for (const e of edges) {
      const a = nodes[e.a], b = nodes[e.b];
      ctx.strokeStyle = muted; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx(a.x), sy(a.y)); ctx.lineTo(sx(b.x), sy(b.y)); ctx.stroke();
      ctx.globalAlpha = 0.85; ctx.fillStyle = muted;
      ctx.fillText(e.label.slice(0, 20), (sx(a.x) + sx(b.x)) / 2 + 2, (sy(a.y) + sy(b.y)) / 2 - 2);
    }
    ctx.globalAlpha = 1;
    nodes.forEach((p, i) => {
      const ti = J.typeList.indexOf(J.types[p.id]);
      ctx.fillStyle = J.typeList.length > 1 ? TYPE_COL[ti % 15] : cssVar('--accent');
      ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), i ? 6 : 11, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = text;
      ctx.font = `${i ? 400 : 700} ${i ? 10.5 : 12}px "Be Vietnam Pro", sans-serif`;
      ctx.fillText(String(J.names[p.id]).slice(0, 28), sx(p.x) + (i ? 8 : 13), sy(p.y) + 4);
    });
  };
  $('#graph').addEventListener('click', (ev) => {
    const { nodes } = KGLAB.g || {};
    if (!nodes) return;
    const cv = $('#graph'), r = cv.getBoundingClientRect(), W = cv.clientWidth, H = cv.clientHeight;
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 14;
    nodes.forEach((p, i) => { const d = Math.hypot(W / 2 + p.x * (W / 2 - 70) - x, H / 2 + p.y * (H / 2 - 28) - y); if (d < bd) { bd = d; best = i; } });
    if (best > 0) setCenter(nodes[best].id);
  });
  window.addEventListener('resize', () => KGLAB.draw());
  // ----- tìm thực thể
  $('#entInp').addEventListener('input', () => {
    const J = KGDATA[st.ds];
    const q = $('#entInp').value.trim().toLowerCase();
    if (!J || q.length < 2) { $('#sugg').innerHTML = ''; return; }
    const hits = [];
    for (let i = 0; i < J.N && hits.length < 60; i++) if (J.lname[i].includes(q)) hits.push(i);
    hits.sort((a, b) => (J.lname[a].startsWith(q) ? 0 : 1) - (J.lname[b].startsWith(q) ? 0 : 1) || J.deg(b) - J.deg(a));
    $('#sugg').innerHTML = hits.slice(0, 12).map((i) => `<li data-e="${i}">${esc(J.names[i])}<small>${esc(st.ds === 'codex' ? (J.descr?.[i] || '') : J.types[i])} · bậc ${J.deg(i)}</small></li>`).join('');
  });
  $('#sugg').addEventListener('click', (ev) => { const li = ev.target.closest('li'); if (li) setCenter(+li.dataset.e); });
  $('#pred').addEventListener('click', (ev) => { const a = ev.target.closest('a'); if (a) { ev.preventDefault(); setCenter(+a.dataset.e); } });
  $('#relSel').addEventListener('change', predict);
  $('#dsSel').addEventListener('change', () => setDS($('#dsSel').value));
  $$('#kindSeg button').forEach((b) => b.addEventListener('click', () => {
    $$('#kindSeg button').forEach((x) => x.classList.toggle('on', x === b));
    st.kind = b.dataset.k;
    predict();
  }));
  function metricView(J) {
    $('#metricTbl').innerHTML = table(['Mô hình', 'Nền tảng', 'MRR', 'H@1', 'H@10'],
      KINDS.flatMap((m) => FWS.map((f) => ({ cells: [KN[m], FWN[f], J.metrics[`${m}_${f}`].MRR, J.metrics[`${m}_${f}`]['H@1'], J.metrics[`${m}_${f}`]['H@10']] }))));
    makeChart('mrrChart', {
      type: 'bar',
      data: { labels: KINDS.map((m) => KN[m]), datasets: FWS.map((f) => ({ label: FWN[f], backgroundColor: FWC[f], data: KINDS.map((m) => J.metrics[`${m}_${f}`].MRR) })) },
      options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'MRR (test)' } } } },
    });
  }
  await setDS('hetio_dm');
  // ----- tự kiểm: core.js so với điểm Python (cùng nhúng float16)
  let maxd = 0, n = 0, rankOk = 0, tot = 0;
  for (const ds of ['hetio_dm', 'sp500', 'codex']) {
    const J = await getKG(ds);
    for (const m of KINDS) {
      for (const t of J.tests[m]) {
        const S = Core.kgeScores(m, J.emb[m].E, J.emb[m].R, J.emb[m].d, t.h, t.r);
        const top = Core.argsortDesc(S).slice(0, 5);
        t.scores.forEach((s, k) => { maxd = Math.max(maxd, Math.abs(S[t.top[k]] - s)); n++; });
        rankOk += top.every((x, k) => x === t.top[k]) ? 1 : 0; tot++;
      }
    }
  }
  $('#selfTest').textContent = `${n} điểm (3 đồ thị × 3 mô hình × 12 truy vấn × top-5): sai lệch lớn nhất JS ↔ Python = ${maxd.toExponential(2)}\n`
    + `top-5 trùng thứ tự: ${rankOk}/${tot} truy vấn`;
  window.SELFTEST = { done: true, maxDiff: maxd, rankOk, tot, errors: [] };
}

// ================================================================= RAG (tiểu đường / chứng khoán)
async function initRAG() {
  const dom = document.body.dataset.domain;
  const R = await loadJSON(`data/rag_${dom}.json`);
  const K = await loadJSON(`data/kg_${dom === 'dm' ? 'hetio_dm' : 'sp500'}.json`);
  const names = dom === 'sp' ? R.names_en : K.names;
  const T = Int32Array.from(K.triples);
  const literals = {};
  for (const [e, v] of Object.entries(R.literals || {})) literals[+e] = v;
  const KGI = new Core.KGIndex(names, T, R.rel_text, R.rel_keys, literals, R.edge_note || {}, R.edge_weight || {});
  const DE = new Core.DualEncoder(R.de);
  const BM = new Core.BM25(R.chunks);
  const Zc = R.chunks.map((c) => DE.encode(c, DE.lp));
  // liên kết thực thể (giống notebook): tên + bí danh; gen/mã cổ phiếu phải viết HOA
  const al = new Map();
  if (dom === 'dm') {
    K.types.forEach((t, i) => { if (t !== 'Gene') al.set(i, [K.names[i], ...(R.aliases[i] || [])]); });
  } else {
    K.types.forEach((t, i) => { if (!['Decade', 'RiskLevel', 'Index'].includes(t)) al.set(i, R.aliases[i] || [K.names[i]]); });
  }
  const linker = new Core.Linker(al, 3, dom === 'dm' ? R.stop : ['united', 'states', 'the company']);
  const upper = new Map();
  if (dom === 'dm') K.types.forEach((t, i) => { if (t === 'Gene') upper.set(K.names[i], i); });
  else Object.entries(R.tickers).forEach(([s, i]) => upper.set(s, i));
  function link(q) {
    let out = linker.link(q);
    const re = dom === 'dm' ? /\b[A-Z][A-Z0-9]{1,9}\b/g : /\b[A-Z]{2,5}\b/g;
    for (const w of q.match(re) || []) if (upper.has(w) && !out.includes(upper.get(w))) out.push(upper.get(w));
    if (dom === 'sp') out = out.map((i, k) => [i, k]).sort((a, b) => (['Company', 'Asset'].includes(K.types[a[0]]) ? 0 : 1)
      - (['Company', 'Asset'].includes(K.types[b[0]]) ? 0 : 1) || a[1] - b[1]).map((x) => x[0]);
    return out;
  }
  function search(q, k = 4) {
    const sb = BM.scores(q), zq = DE.encode(q);
    const sd = Float64Array.from(Zc, (z) => Core.dot(zq, z));
    const s = R.search === 'bm25' ? sb : R.search === 'de' ? sd : Core.rrf([sb, sd]);   // cách truy hồi chọn ở notebook
    const rb = Core.argsortDesc(sb), rd = Core.argsortDesc(sd);
    return Core.argsortDesc(s).slice(0, k).map((i) => ({ i, bm: rb.indexOf(i) + 1, de: rd.indexOf(i) + 1 }));
  }
  // ----- giao diện
  const byQ = new Map(R.questions.map((x, i) => [x.q, i]));
  const sum = Object.fromEntries(R.summary.map((r) => [r.mode, r]));
  const metric = dom === 'dm' ? 'f1' : 'score';
  $('#ragApp').innerHTML = `
    <section class="card">
      <h2><span class="step">1</span>Câu hỏi</h2>
      <div class="ctrl-row" style="margin-bottom:8px">
        <label class="grow">Chọn câu hỏi đánh giá (${R.questions.length} câu, có câu trả lời LLM tính sẵn)
          <select id="qSel">${R.questions.map((x, i) => `<option value="${i}">[${esc(x.tpl)}] ${esc(x.q)}</option>`).join('')}</select></label>
      </div>
      <div class="ctrl-row">
        <label class="grow">… hoặc tự gõ (tiếng Anh)
          <input class="inp" id="qInp" placeholder="${dom === 'dm' ? 'e.g. Which genes does Metformin bind to?' : 'e.g. Which stocks move with gold prices?'}"></label>
        <button class="btn" id="askBtn" type="button">Truy hồi</button>
      </div>
      <div class="chips" id="chips"></div>
    </section>
    <div class="grid2 mt">
      <section class="card"><h2><span class="step">2</span>Đồ thị: thực thể và bộ ba được lấy</h2>
        <div class="ent-chips" id="ents"></div><p class="label">Bộ ba xếp theo độ khớp quan hệ (cột trái = điểm)</p><ul class="facts" id="facts"></ul></section>
      <section class="card"><h2><span class="step">3</span>Văn bản: đoạn được lấy (${R.search === 'bm25' ? 'BM25' : R.search === 'de' ? 'bộ mã hoá kép' : 'BM25 + bộ mã hoá kép, RRF'})</h2><div id="pass"></div></section>
    </div>
    <section class="card mt"><h2><span class="step">4</span>Câu trả lời</h2><div id="ans"></div>
      <details class="extra prompt"><summary>Prompt của chế độ RAG lai (dán vào LLM bất kỳ)</summary><pre class="mono" id="prompt"></pre></details></section>
    <section class="card mt"><h2><span class="step">5</span>Kết quả đánh giá ${R.questions.length} câu hỏi (${esc(R.llm)})</h2>
      <div class="grid2"><div class="chart-box"><canvas id="modeChart" aria-label="Điểm theo chế độ"></canvas></div><div class="tbl-wrap" id="modeTbl"></div></div>
      ${dom === 'dm' ? '<h3 style="font-size:.92rem;margin-top:14px">ComplEx gợi ý thuốc mới cho tiểu đường type 2 (kèm đường đi giải thích)</h3><div class="tbl-wrap" id="repTbl"></div>' : ''}
    </section>`;
  const sample = dom === 'dm' ? ['Which genes does Metformin bind to?', 'What are the symptoms of type 1 diabetes?', 'How is gestational diabetes treated?', 'Is diabetes inherited?']
    : ['Which stocks move with gold prices?', 'Who founded Nvidia?', 'What is the beta of JPM?', 'Which companies are headquartered in Santa Clara?'];
  $('#chips').innerHTML = sample.map((s) => `<button type="button">${esc(s)}</button>`).join('');
  $$('#chips button').forEach((b) => b.addEventListener('click', () => { $('#qInp').value = b.textContent; run(b.textContent); }));
  $('#askBtn').addEventListener('click', () => run($('#qInp').value));
  $('#qInp').addEventListener('keydown', (e) => { if (e.key === 'Enter') run($('#qInp').value); });
  $('#qSel').addEventListener('change', () => { $('#qInp').value = ''; run(R.questions[+$('#qSel').value].q); });

  function highlight(text, words) {
    let s = esc(text);
    for (const w of words) if (w.length > 3) s = s.replace(new RegExp(`\\b(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'), '<mark>$1</mark>');
    return s;
  }
  function run(q) {
    q = String(q || '').trim();
    if (!q) return;
    const ents = link(q);
    const facts = KGI.facts(ents, q, 30);
    const hits = search(q, 4);
    const qw = Core.tokenize(q);
    $('#ents').innerHTML = ents.length ? ents.slice(0, 8).map((i) => `<span>${esc(names[i])}<small>${esc(K.types[i])}</small></span>`).join('')
      : '<span class="muted" style="background:none">không tìm thấy thực thể nào</span>';
    $('#facts').innerHTML = facts.slice(0, 30).map(([s, f]) => `<li class="${s >= 1.5 ? 'hit' : ''}"><span class="s">${s.toFixed(1)}</span><span>${esc(f)}</span></li>`).join('')
      || '<li class="muted">(trống)</li>';
    $('#pass').innerHTML = hits.map((h, k) => `<div class="passage"><div class="src"><b>#${k + 1}</b><span>${esc(R.src[h.i])}</span>
      <span>hạng BM25 ${h.bm} · hạng DE ${h.de}</span></div>${highlight(R.chunks[h.i], qw)}</div>`).join('');
    $('#prompt').textContent = Core.buildPrompt(q, hits.slice(0, 3).map((h) => R.chunks[h.i]), facts.slice(0, 20).map((f) => f[1]));
    const qi = byQ.get(q);
    if (qi !== undefined) {
      const x = R.questions[qi];
      $('#qSel').value = qi;
      const gold = x.gold ? `<div class="gold"><b>Đáp án (${dom === 'dm' ? 'Hetionet' : 'SP500-KG'}):</b> ${esc(x.gold.join(', ').slice(0, 600))}</div>`
        : x.num ? `<div class="gold"><b>Đáp án (dữ liệu giá):</b> ${x.num[0] === 'ret' ? `${fmt(x.num[1], 1)}%` : fmt(x.num[1], 2)}</div>`
          : `<div class="gold"><b>Đáp án của NIH:</b> ${esc(x.ref.slice(0, 700))}…</div>`;
      $('#ans').innerHTML = gold + `<div class="answers">${MODES.map((m) => `<div class="ans" style="border-color:${MC[m]}"><h3><span class="dot" style="background:${MC[m]}"></span>${MN[m]}</h3>${esc(x.ans[m])}</div>`).join('')}</div>
        <p class="note muted">Câu trả lời sinh bởi ${esc(R.llm)} (giải mã tham lam) với đúng ngữ cảnh lúc chạy notebook.</p>`;
    } else {
      const top = facts.filter(([s]) => s >= 1.5).slice(0, 8).map((f) => f[1]);
      const sents = hits.length ? R.chunks[hits[0].i].split(/(?<=[.!?])\s+/).slice(0, 3).join(' ') : '';
      $('#ans').innerHTML = `<div class="answers" style="grid-template-columns:1fr 1fr">
        <div class="ans" style="border-color:${MC.kg}"><h3><span class="dot" style="background:${MC.kg}"></span>Trích xuất từ đồ thị</h3>${top.length ? esc(top.join('\n')) : 'Không có bộ ba nào khớp quan hệ được hỏi.'}</div>
        <div class="ans" style="border-color:${MC.text}"><h3><span class="dot" style="background:${MC.text}"></span>Trích xuất từ đoạn văn #1</h3>${esc(sents)}</div></div>
        <p class="note muted">Câu hỏi tự gõ: trang không chạy LLM (cần GPU); dùng prompt bên dưới với LLM bất kỳ để có câu trả lời sinh.</p>`;
    }
  }
  run(R.questions[0].q);
  makeChart('modeChart', {
    type: 'bar',
    data: { labels: MODES.map((m) => MN[m]), datasets: [{ label: dom === 'dm' ? 'F1 câu hỏi KG' : 'điểm TB', data: MODES.map((m) => sum[m][metric]), backgroundColor: MODES.map((m) => MC[m]) },
      { label: 'độ phủ ngữ cảnh', data: MODES.map((m) => sum[m].ctx_cov), backgroundColor: MODES.map((m) => `${MC[m]}55`) }] },
    options: { scales: { y: { beginAtZero: true, max: 1 } } },
  });
  $('#modeTbl').innerHTML = table(['Chế độ', dom === 'dm' ? 'F1 (KG)' : 'Điểm', 'Recall', 'Precision', ...(dom === 'dm' ? ['MedQuAD: chính xác từ', 'MedQuAD: cosine'] : []), 'Prompt (token)'],
    MODES.map((m) => ({ cells: [MN[m], sum[m][metric], sum[m].recall, sum[m].precision, ...(dom === 'dm' ? [sum[m].MQ_tokP ?? 0, sum[m].MQ_sem] : []), Math.round(sum[m].prompt_tokens).toString()] })));
  if (dom === 'dm') {
    $('#repTbl').innerHTML = table(Object.keys(R.repurpose[0]), R.repurpose.map((r) => ({ cells: Object.values(r).map((v) => (typeof v === 'number' && !Number.isInteger(v) ? v : esc(String(v)))) })));
  } else {
    const F = R.features;
    const rIn = K.rels.indexOf('in_sector'), sectorOf = new Map();
    for (let j = 0; j < T.length / 3; j++) if (T[3 * j + 1] === rIn) sectorOf.set(K.ents[T[3 * j]], K.names[T[3 * j + 2]]);
    const bySec = {};
    F.forEach((f) => { const sec = sectorOf.get(`TICKER:${f.Symbol}`); if (sec) (bySec[sec] ||= []).push(f.corr_gold); });
    const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const secs = Object.keys(bySec).sort((a, b) => avg(bySec[b]) - avg(bySec[a]));
    makeChart('goldChart', {
      type: 'bar', data: { labels: secs, datasets: [{ label: 'tương quan TB với vàng', data: secs.map((s) => avg(bySec[s])), backgroundColor: '#ffb703' }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false } } },
    });
    const top = [...F].sort((a, b) => b.corr_gold - a.corr_gold).slice(0, 10);
    $('#goldTbl').innerHTML = table(['Mã', 'Tương quan vàng', 'Beta', 'Lợi suất 1 năm'], top.map((f) => ({ cells: [f.Symbol, f.corr_gold, f.beta, `${fmt(100 * f.ret_1y, 1)}%`] })))
      + `<p class="note muted">Giá tới ${esc(R.asof)}; tương quan lợi suất ngày 2 năm với hợp đồng vàng GC=F.</p>`;
  }
  // ----- tự kiểm core.js ↔ Python (bộ mã hoá kép + BM25)
  let maxd = 0, bmOk = 0;
  R.de_test.q.forEach((q, k) => {
    const z = DE.encode(q);
    R.de_test.z[k].forEach((v, j) => { maxd = Math.max(maxd, Math.abs(v - z[j])); });
    const top = Core.argsortDesc(BM.scores(q)).slice(0, 5);
    bmOk += top.every((x, j) => x === R.de_test.bm25_top[k][j]) ? 1 : 0;
  });
  window.SELFTEST = { done: true, maxDiff: maxd, bm25: `${bmOk}/${R.de_test.q.length}`, errors: [] };
}

// ================================================================= khởi động
const INIT = { index: initIndex, kg: initKG, rag: initRAG };
INIT[PAGE]?.().catch((e) => {
  console.error(e);
  window.SELFTEST.errors.push(String(e));
  const m = document.createElement('p');
  m.className = 'alert warn';
  m.textContent = `Lỗi nạp dữ liệu: ${e.message}`;
  $('main').prepend(m);
});
