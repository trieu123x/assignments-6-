/* Tiểu luận AI — logic cho 4 trang: index (Ch1) · ml (Ch2) · cnn (Ch3) · rnn (Ch4).
 * Mọi suy luận dùng NN (nn.js). Không có máy chủ: chỉ fetch các file JSON tĩnh trong models/. */
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const PAGE = document.body.dataset.page;
const FWS = ['scratch', 'keras', 'torch'];
const FWN = { scratch: 'NumPy (scratch)', keras: 'Keras', torch: 'PyTorch' };
const FWC = { scratch: '#2a6f97', keras: '#c1121f', torch: '#ee8a2a' };
const cache = {};
const loadJSON = (p) => (cache[p] ||= fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}));
const fmt = (x, d = 2) => Number(x).toLocaleString('vi-VN', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${fmt(100 * x, d)}%`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------- giao diện sáng/tối
(() => {
  try { const t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* bỏ qua */ }
  $('#themeBtn')?.addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', document.documentElement.dataset.theme); } catch (_) { /* bỏ qua */ }
    Object.values(CHARTS).forEach((c) => { styleChart(c); c.update('none'); });
  });
})();

// ---------------------------------------------------------------- Chart.js
const CHARTS = {};
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
  if (typeof Chart === 'undefined') return null;
  CHARTS[id]?.destroy();
  config.options = { responsive: true, maintainAspectRatio: false, animation: false, ...config.options };
  config.options.plugins = { legend: { labels: {} }, ...(config.options.plugins || {}) };
  const c = new Chart($(`#${id}`), config);
  styleChart(c);
  c.update('none');
  CHARTS[id] = c;
  return c;
}

// ---------------------------------------------------------------- tiện ích dùng chung
const MODELS = {};
function getModel(key, json) { return (MODELS[key] ||= NN.load(json)); }
function barRows(items, topIdx) {
  return `<div class="bars">${items.map((it, i) => `<div class="barrow${i === topIdx ? ' top' : ''}">
    <span class="name" title="${esc(it.name)}">${esc(it.name)}</span>
    <span class="track"><span class="fill" style="width:${(100 * it.p).toFixed(1)}%"></span></span>
    <span class="v">${pct(it.p)}</span></div>`).join('')}</div>`;
}
const TOKEN_RE = /[a-z0-9']+/g;
const tokenize = (s) => s.toLowerCase().replaceAll('<br />', ' ').match(TOKEN_RE) || [];

// ================================================================= INDEX (Chương 1)
const DOM_COL = { Language: '#2f5bd3', Vision: '#2a9d8f', Games: '#e9c46a', Multimodal: '#9b5de5', Speech: '#e76f51',
  'Image generation': '#f15bb5', Biology: '#6a994e', Other: '#8d99ae' };
async function initIndex() {
  try {
    const s = await loadJSON('models/summary.json');
    $('#kMl').textContent = pct(s.ml, 1);
    $('#kCnn').textContent = pct(s.cnn, 1);
    $('#kRnn').textContent = pct(s.rnn, 1);
  } catch (e) { console.warn(e); }
  const H = await loadJSON('models/history.json');
  const deep = Object.values(H.fits)[1];
  $('#kDouble').textContent = fmt(12 * Math.log10(2) / deep.slope, 1);
  const toYear = (d) => { const t = new Date(d); return t.getFullYear() + (t - new Date(t.getFullYear(), 0, 1)) / 3.156e10; };
  const draw = (metric) => {
    const groups = {};
    for (const [name, org, date, dom, lc, lp] of H.models) {
      const y = metric === 'c' ? lc : lp;
      if (y == null) continue;
      const g = DOM_COL[dom] ? dom : 'Other';
      (groups[g] ||= []).push({ x: toYear(date), y, name, org, date });
    }
    const ds = Object.entries(groups).sort((a, b) => b[1].length - a[1].length).map(([g, pts]) => ({
      type: 'scatter', label: g === 'Other' ? 'Khác' : g, data: pts, pointRadius: 2.6, backgroundColor: DOM_COL[g] + 'aa', borderWidth: 0 }));
    if (metric === 'c') {
      for (const [nm, f] of Object.entries(H.fits)) {
        ds.push({ type: 'line', label: `${nm.split('(')[0].trim()}: ×${fmt(10 ** f.slope, 1)}/năm`, borderDash: [6, 4],
          data: [f.from, f.to].map((x) => ({ x, y: f.slope * x + f.intercept })), pointRadius: 0, borderWidth: 2,
          borderColor: nm.includes('học sâu') ? '#c1121f' : '#555' });
      }
    }
    makeChart('histChart', {
      data: { datasets: ds },
      options: { parsing: false,
        scales: { x: { type: 'linear', min: 1950, max: 2027, ticks: { callback: (v) => String(v) }, title: { display: true, text: 'năm công bố' } },
          y: { title: { display: true, text: metric === 'c' ? 'log10(FLOP huấn luyện)' : 'log10(số tham số)' } } },
        plugins: { legend: { labels: { boxWidth: 10 } },
          tooltip: { callbacks: { label: (c) => c.raw.name ? `${c.raw.name} (${c.raw.org}, ${c.raw.date}): 10^${fmt(c.raw.y, 1)}` : c.dataset.label } } } },
    });
  };
  draw('c');
  $$('#histMetric button').forEach((b) => b.addEventListener('click', () => {
    $$('#histMetric button').forEach((x) => x.classList.toggle('on', x === b));
    draw(b.dataset.m);
  }));
  makeChart('ladderChart', {
    type: 'bar',
    data: { labels: H.mnist.map((r) => [String(r[1]), r[0].split(' (')[0]]),
      datasets: [{ label: 'tỉ lệ lỗi (%)', data: H.mnist.map((r) => r[2]),
        backgroundColor: ['#8d99ae', '#457b9d', '#457b9d', '#e9c46a', '#2a9d8f', '#2a9d8f', '#e76f51'] }] },
    options: { scales: { y: { type: 'logarithmic', title: { display: true, text: 'lỗi (%)' } }, x: { ticks: { font: { size: 10 } } } },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 2)}% lỗi` } } } },
  });
  $('#selfTestBtn').addEventListener('click', selfTest);
}

async function selfTest() {
  const out = $('#selfTestOut'), btn = $('#selfTestBtn');
  out.hidden = false;
  btn.disabled = true;
  const log = (s) => { out.textContent += s + '\n'; out.scrollTop = out.scrollHeight; };
  out.textContent = '';
  let worst = 0, n = 0;
  const cmp = (tag, js, py) => {
    let m = 0;
    for (let i = 0; i < py.length; i++) m = Math.max(m, Math.abs(js[i] - py[i]));
    worst = Math.max(worst, m);
    n += py.length;
    log(`${m < 1e-3 ? '✓' : '✗'} ${tag}: ${py.length} giá trị, max |JS − Python| = ${m.toExponential(1)}`);
  };
  try {
    const ml = await loadJSON('models/ml.json');
    for (const [ds, D] of Object.entries(ml)) {
      for (const [key, ref] of Object.entries(D.ref)) {
        const m = getModel(`ml/${ds}/${key}`, D.models[key]);
        const js = [];
        D.samples.X.slice(0, ref.length).forEach((row) => {
          const x = Float64Array.from(row, (v, j) => (v - D.mu[j]) / D.sd[j]);
          js.push(...NN.forward(m, { d: x, s: [x.length] }).d);
        });
        cmp(`Ch2 ${ds} ${key}`, js, ref.flat());
      }
      await tick();
    }
    for (const ds of ['fashionmnist', 'cifar10', 'flowers']) {
      const J = await loadJSON(`models/cnn_${ds}.json`);
      const X = cnnSamples(J);
      for (const fw of FWS) {
        const m = getModel(`cnn/${ds}/${fw}`, J.models[fw]);
        const js = [];
        for (const x of X) { js.push(...NN.forward(m, x).d); await tick(); }
        cmp(`Ch3 ${ds} ${fw}`, js, J.ref[fw].flat());
      }
    }
    const im = await loadJSON('models/rnn_imdb.json');
    const texts = im.examples.concat(im.samples);
    for (const fw of FWS) {
      const m = getModel(`imdb/${fw}`, im.models[fw]);
      cmp(`Ch4 IMDB ${fw}`, texts.map((t) => NN.forward(m, imdbEncode(im, t).x).d[0]), im.ref[fw]);
      await tick();
    }
    const je = await loadJSON('models/rnn_jena.json');
    const Z = jenaZ(je);
    for (const fw of FWS) {
      const m = getModel(`jena/${fw}`, je.models[fw]);
      cmp(`Ch4 Jena ${fw}`, je.ref_ends.map((e) => NN.forward(m, jenaWindow(je, Z, e)).d[0]), je.ref[fw]);
    }
    const sh = await loadJSON('models/rnn_shakes.json');
    const c2i = Object.fromEntries(sh.chars.map((c, i) => [c, i]));
    for (const fw of FWS) {
      const m = getModel(`shakes/${fw}`, sh.models[fw]);
      const ids = [...'ROMEO:\nWhat light'].map((c) => c2i[c]);
      const o = NN.forward(m, { d: Int32Array.from(ids), s: [ids.length] });
      cmp(`Ch4 Shakespeare ${fw}`, o.d.slice(o.d.length - sh.chars.length), sh.ref[fw]);
      await tick();
    }
    log(`\n${worst < 1e-3 ? 'ĐẠT' : 'CÓ SAI LỆCH'}: ${n} giá trị, sai lệch lớn nhất ${worst.toExponential(2)}`);
  } catch (e) {
    log('LỖI: ' + e.message);
  }
  btn.disabled = false;
}

// ================================================================= ML (Chương 2)
const COLVI = { Pregnancies: 'Số lần mang thai', Glucose: 'Glucose (mg/dL)', BloodPressure: 'Huyết áp tâm trương (mmHg)',
  SkinThickness: 'Nếp gấp da cơ tam đầu (mm)', Insulin: 'Insulin 2 giờ (µU/mL)', BMI: 'BMI (kg/m²)',
  DiabetesPedigreeFunction: 'Chỉ số tiền sử gia đình', Age: 'Tuổi', MedInc: 'Thu nhập trung vị (×10.000 USD)',
  HouseAge: 'Tuổi nhà trung vị (năm)', AveRooms: 'Số phòng TB mỗi hộ', AveBedrms: 'Số phòng ngủ TB mỗi hộ',
  Population: 'Dân số khu', AveOccup: 'Số người TB mỗi hộ', Latitude: 'Vĩ độ', Longitude: 'Kinh độ' };
async function initMl() {
  const ML = await loadJSON('models/ml.json');
  let ds = 'Pima';
  const render = () => {
    const D = ML[ds];
    $('#mlForm').innerHTML = D.cols.map((c, j) => `<label>${esc(COLVI[c] || c)}
      <input type="number" step="any" class="inp" data-j="${j}" value="${+D.samples.X[0][j].toPrecision(7)}"></label>`).join('');
    const lab = (y) => (D.task === 'bin' ? (y ? 'bệnh' : 'không') : D.task === 'reg' ? `${fmt(y * 100, 0)}k$` : D.classes[y]);
    $('#sampleChips').innerHTML = D.samples.X.map((_, i) => `<button type="button" data-i="${i}">#${i + 1} (${lab(D.samples.y[i])})</button>`).join('');
    $$('#sampleChips button').forEach((b) => b.addEventListener('click', () => {
      $$('#mlForm input').forEach((inp) => { inp.value = +D.samples.X[b.dataset.i][inp.dataset.j].toPrecision(7); });
      run();
    }));
    run();
  };
  const run = () => {
    const D = ML[ds];
    const raw = $$('#mlForm input').map((i) => parseFloat(i.value) || 0);
    const x = Float64Array.from(raw, (v, j) => (v - D.mu[j]) / D.sd[j]);
    const rows = [];
    for (const kind of ['lin', 'mlp']) {
      for (const fw of FWS) {
        const key = `${kind}_${fw}`;
        const out = NN.forward(getModel(`ml/${ds}/${key}`, D.models[key]), { d: x, s: [x.length] }).d;
        const t = D.test[key];
        let pred, metric;
        if (D.task === 'bin') {
          const p = NN.sig(out[0]);
          pred = `<b>${pct(p)}</b> <span class="pill ${p >= 0.5 ? 'no' : 'ok'}">${p >= 0.5 ? 'nguy cơ tiểu đường' : 'không'}</span>`;
          metric = `AUC ${fmt(t.auc, 3)}`;
        } else if (D.task === 'reg') {
          pred = `<b>${fmt(Math.max(0, out[0]) * 100000, 0)} USD</b>`;
          metric = `RMSE ${fmt(t.rmse * 100, 1)}k$`;
        } else {
          const p = NN.softmax(out);
          const k = p.indexOf(Math.max(...p));
          pred = `<b>${D.classes[k]}</b> <span class="muted">${pct(p[k])}</span>`;
          metric = `acc ${pct(t.acc)}`;
        }
        rows.push(`<tr><td>${kind === 'lin' ? 'Tuyến tính' : 'MLP 64-32'}</td><td><span class="dot" style="background:${FWC[fw]}"></span>${FWN[fw]}</td>
          <td>${pred}</td><td class="num muted">${metric}</td></tr>`);
      }
    }
    $('#mlOut').innerHTML = `<table class="cmp"><thead><tr><th>Mô hình</th><th>Nền tảng</th><th>Dự đoán</th><th>Trên tập test</th></tr></thead>
      <tbody>${rows.join('')}</tbody></table>`;
    $('#mlMeta').textContent = D.task === 'bin' ? 'Ngưỡng 50%. Chỉ minh hoạ thuật toán, không dùng để chẩn đoán.'
      : D.task === 'reg' ? 'Giá trị nhà trung vị của một khu dân cư năm 1990; dữ liệu gốc bị chặn trần ở 500.001 USD.'
        : 'Đặc trưng hình dạng được đo tự động từ ảnh hạt đậu (diện tích theo pixel, độ tròn, …).';
  };
  $('#mlRun').addEventListener('click', run);
  $$('#dsSeg button').forEach((b) => b.addEventListener('click', () => {
    $$('#dsSeg button').forEach((x) => x.classList.toggle('on', x === b));
    ds = b.dataset.ds;
    render();
  }));
  render();
}

// ================================================================= CNN (Chương 3)
function cnnSamples(J) {
  const [C, H, W] = J.in_shape, n = J.samples.y.length, sz = C * H * W;
  const bytes = Uint8Array.from(atob(J.samples.data), (c) => c.charCodeAt(0));
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = new Float64Array(sz);
    for (let k = 0; k < sz; k++) { const c = Math.floor(k / (H * W)); x[k] = (bytes[i * sz + k] / 255 - J.mu[c]) / J.sd[c]; }
    out.push({ d: x, s: [C, H, W], raw: bytes.subarray(i * sz, (i + 1) * sz) });
  }
  return out;
}
function rawToCanvas(raw, C, H, W, cv) {
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d'), im = ctx.createImageData(W, H);
  for (let p = 0; p < H * W; p++) {
    for (let c = 0; c < 3; c++) im.data[4 * p + c] = raw[(C === 1 ? 0 : c) * H * W + p];
    im.data[4 * p + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
async function initCnn() {
  let ds = 'fashionmnist', J = null, X = null, cur = null, lastImg = null;
  const show = async (x, cap) => {
    cur = x;
    const [C, H, W] = J.in_shape;
    rawToCanvas(x.raw, C, H, W, $('#cvIn'));
    $('#cvCap').textContent = cap;
    $('#cnnOut').innerHTML = '<p class="placeholder muted">Đang tính…</p>';
    await tick();
    const res = {};
    const t0 = performance.now();
    for (const fw of FWS) res[fw] = NN.softmax(NN.forward(getModel(`cnn/${ds}/${fw}`, J.models[fw]), x).d);
    const ms = performance.now() - t0;
    const top = (p) => p.indexOf(Math.max(...p));
    const agree = new Set(FWS.map((fw) => top(res[fw]))).size === 1;
    $('#cnnOut').innerHTML = (x.y != null ? `<p class="verdict">Nhãn thật: <b>${esc(J.classes[x.y])}</b></p>` : '')
      + FWS.map((fw) => {
        const p = res[fw], order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]).slice(0, 3);
        const ok = x.y == null ? '' : `<span class="pill ${order[0] === x.y ? 'ok' : 'no'}">${order[0] === x.y ? 'đúng' : 'sai'}</span>`;
        return `<div class="fwblock"><h3><span class="dot" style="background:${FWC[fw]}"></span>${FWN[fw]}: ${esc(J.classes[order[0]])}${ok}
          <small class="muted">acc test ${pct(J.test[fw].acc)}</small></h3>${barRows(order.map((i) => ({ name: J.classes[i], p: p[i] })), 0)}</div>`;
      }).join('') + `<p class="meta ${agree ? 'up' : 'down'}">${agree ? 'Ba mô hình đồng ý.' : 'Ba mô hình KHÔNG đồng ý.'}</p>`;
    $('#cnnMeta').textContent = `Ba lượt suy luận mất ${fmt(ms, 0)} ms trong trình duyệt (trọng số float16).`;
  };
  const fromImage = (img) => {
    lastImg = img;
    const [C, H, W] = J.in_shape;
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    const s = Math.min(img.width, img.height);
    ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, W, H);
    const px = ctx.getImageData(0, 0, W, H).data, inv = $('#invert').checked;
    const raw = new Uint8Array(C * H * W);
    for (let p = 0; p < H * W; p++) {
      if (C === 1) {
        let g = 0.299 * px[4 * p] + 0.587 * px[4 * p + 1] + 0.114 * px[4 * p + 2];
        raw[p] = inv ? 255 - g : g;
      } else for (let c = 0; c < 3; c++) raw[c * H * W + p] = inv ? 255 - px[4 * p + c] : px[4 * p + c];
    }
    const x = new Float64Array(raw.length);
    for (let k = 0; k < raw.length; k++) { const c = Math.floor(k / (H * W)); x[k] = (raw[k] / 255 - J.mu[c]) / J.sd[c]; }
    show({ d: x, s: [C, H, W], raw, y: null }, `ảnh của bạn → ${W}×${H}`);
  };
  const handleFile = (f) => {
    if (!f || !f.type.startsWith('image/')) return;
    const img = new Image();
    img.onload = () => fromImage(img);
    img.src = URL.createObjectURL(f);
  };
  const load = async () => {
    $('#cnnOut').innerHTML = '<p class="placeholder muted">Đang nạp mô hình…</p>';
    J = await loadJSON(`models/cnn_${ds}.json`);
    X = cnnSamples(J);
    X.forEach((x, i) => { x.y = J.samples.y[i]; });
    $('#invert').checked = ds === 'fashionmnist';
    const [C, H, W] = J.in_shape;
    $('#samples').innerHTML = '';
    X.forEach((x, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.ds = ds === 'flowers' ? 'flowers' : '';
      b.title = J.classes[x.y];
      const cv = document.createElement('canvas');
      rawToCanvas(x.raw, C, H, W, cv);
      b.innerHTML = `<img alt="${esc(J.classes[x.y])}" src="${cv.toDataURL()}">`;
      b.addEventListener('click', () => {
        $$('#samples button').forEach((o) => o.classList.toggle('sel', o === b));
        show(x, `ảnh test: ${J.classes[x.y]} (${W}×${H})`);
      });
      $('#samples').append(b);
    });
    const spec = J.models.torch.spec.map((s) => s[0] === 'conv' ? `Conv${s[1]}` : s[0] === 'dense' ? `Dense${s[1]}` : s[0] === 'pool' ? 'Pool' : '').filter(Boolean);
    $('#cnnInfo').innerHTML = `<p>Đầu vào ${C}×${H}×${W} · ${J.classes.length} lớp<br>Kiến trúc: ${spec.join(' → ')}</p>`;
    $('#samples button').click();
  };
  $$('#dsSeg button').forEach((b) => b.addEventListener('click', () => {
    $$('#dsSeg button').forEach((x) => x.classList.toggle('on', x === b));
    ds = b.dataset.ds;
    load();
  }));
  $('#file').addEventListener('change', (e) => handleFile(e.target.files[0]));
  $('#invert').addEventListener('change', () => { if (lastImg && cur && cur.y == null) fromImage(lastImg); });
  const drop = $('#drop');
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') $('#file').click(); });
  ['dragover', 'dragenter'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));
  drop.addEventListener('drop', (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });
  window.addEventListener('paste', (e) => handleFile([...(e.clipboardData?.files || [])][0]));
  load();
}

// ================================================================= RNN (Chương 4)
function imdbEncode(J, text) {
  J._w2i ||= new Map(J.words.map((w, i) => [w, i + 2]));
  const toks = tokenize(text);
  const ids = toks.map((w) => J._w2i.get(w) ?? 1).slice(-J.maxlen);
  const x = new Int32Array(J.maxlen);
  x.set(ids, J.maxlen - ids.length);
  return { x: { d: x, s: [J.maxlen] }, n: toks.length, unk: ids.filter((v) => v === 1).length, used: ids.length };
}
function jenaZ(J) { return NN.decode(J.Z); }
function jenaWindow(J, Z, e) {
  const D = J.Z.shape[1], L = J.lookback;
  return { d: Z.subarray((e - L + 1) * D, (e + 1) * D), s: [L, D] };
}
async function initRnn() {
  // ---- IMDB
  const im = await loadJSON('models/rnn_imdb.json');
  $('#imdbChips').innerHTML = im.examples.map((t, i) => `<button type="button" data-t="e${i}">${esc(t.slice(0, 34))}…</button>`).join('')
    + im.samples.map((t, i) => `<button type="button" data-t="s${i}">bài test #${i + 1}</button>`).join('');
  $$('#imdbChips button').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.t;
    $('#imdbText').value = k[0] === 'e' ? im.examples[+k.slice(1)] : im.samples[+k.slice(1)];
    runImdb();
  }));
  const runImdb = () => {
    const enc = imdbEncode(im, $('#imdbText').value);
    $('#imdbTok').textContent = `${enc.n} từ; mô hình đọc ${enc.used} từ cuối, trong đó ${enc.unk} từ lạ.`;
    $('#imdbOut').innerHTML = FWS.map((fw) => {
      const p = NN.sig(NN.forward(getModel(`imdb/${fw}`, im.models[fw]), enc.x).d[0]);
      return `<div class="gauge ${p >= 0.5 ? 'neg' : 'pos'}"><div class="lbl"><span class="dot" style="background:${FWC[fw]}"></span>${FWN[fw]}</div>
        <div class="p">${pct(p, 0)}</div><div class="track"><div class="fill" style="width:${100 * p}%"></div><div class="thr" style="left:50%"></div></div>
        <div class="lbl">${p >= 0.5 ? 'tích cực' : 'tiêu cực'} · acc test ${pct(im.test[fw].acc)}</div></div>`;
    }).join('');
  };
  $('#imdbRun').addEventListener('click', runImdb);
  runImdb();

  // ---- Jena
  const je = await loadJSON('models/rnn_jena.json');
  const Z = jenaZ(je), N = je.Z.shape[0], D = je.Z.shape[1];
  const t0 = new Date(je.start.replace(' ', 'T'));
  const at = (i) => new Date(t0.getTime() + i * 3600e3);
  const nDays = Math.floor((N - je.lookback - je.horizon - 168) / 24);
  $('#jenaDay').max = nDays;
  const Tof = (i) => Z[i * D + 1] * je.sd[1] + je.mu[1];
  const runJena = async () => {
    const d = +$('#jenaDay').value, e0 = je.lookback - 1 + d * 24;
    $('#jenaDate').textContent = at(e0 + je.horizon).toLocaleDateString('vi-VN');
    await tick();
    const ends = Array.from({ length: 168 }, (_, k) => e0 + k);
    const labels = ends.map((e) => at(e + je.horizon).toLocaleString('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit' }));
    const actual = ends.map((e) => Tof(e + je.horizon)), naive = ends.map((e) => Tof(e));
    const preds = {};
    for (const fw of FWS) {
      const m = getModel(`jena/${fw}`, je.models[fw]);
      preds[fw] = ends.map((e) => NN.forward(m, jenaWindow(je, Z, e)).d[0] * je.sd[1] + je.mu[1]);
    }
    const mae = (p) => p.reduce((s, v, i) => s + Math.abs(v - actual[i]), 0) / p.length;
    makeChart('jenaChart', {
      type: 'line',
      data: { labels, datasets: [
        { label: 'thực tế', data: actual, borderColor: cssVar('--text'), borderWidth: 2.2, pointRadius: 0 },
        { label: 'ngây thơ T(t)', data: naive, borderColor: '#999', borderDash: [4, 4], borderWidth: 1.2, pointRadius: 0 },
        ...FWS.map((fw) => ({ label: FWN[fw], data: preds[fw], borderColor: FWC[fw], borderWidth: 1.3, pointRadius: 0 }))] },
      options: { interaction: { mode: 'index', intersect: false },
        scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: '°C' } } },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw, 1)} °C` } } } },
    });
    $('#jenaKpi').innerHTML = FWS.map((fw) => `<div><span>${FWN[fw]} — MAE tuần này</span><b>${fmt(mae(preds[fw]), 2)} °C</b>
        <small class="muted">cả tập test: ${fmt(je.test[fw].mae, 2)} °C</small></div>`).join('')
      + `<div><span>Ngây thơ — MAE tuần này</span><b>${fmt(mae(naive), 2)} °C</b><small class="muted">cả tập test: ${fmt(je.baseline_mae, 2)} °C</small></div>`;
  };
  let tmr = null;
  $('#jenaDay').addEventListener('input', () => { clearTimeout(tmr); tmr = setTimeout(runJena, 120); });
  runJena();

  // ---- Shakespeare
  const sh = await loadJSON('models/rnn_shakes.json');
  const c2i = Object.fromEntries(sh.chars.map((c, i) => [c, i]));
  $('#shTemp').addEventListener('input', () => { $('#shTempV').textContent = fmt($('#shTemp').value, 1); });
  $('#shLen').addEventListener('input', () => { $('#shLenV').textContent = $('#shLen').value; });
  $('#shPrime').value = 'ROMEO:\n';
  const runSh = async () => {
    const fw = $('#shFw').value, m = getModel(`shakes/${fw}`, sh.models[fw]);
    const [E, L, Dn] = m.layers, Dm = E.w.E.s[1];
    const temp = +$('#shTemp').value, n = +$('#shLen').value;
    const prime = [...$('#shPrime').value.replaceAll('\\n', '\n')].filter((c) => c in c2i).join('') || 'ROMEO:\n';
    let st = NN.initState(L), cur = c2i[prime[0]], text = prime;
    const embedRow = (i) => E.w.E.d.subarray(i * Dm, (i + 1) * Dm);
    for (let i = 1; i < prime.length; i++) { st = NN.cellStep(L, embedRow(cur), st); cur = c2i[prime[i]]; }
    $('#shRun').disabled = true;
    for (let k = 0; k < n; k++) {
      st = NN.cellStep(L, embedRow(cur), st);
      const p = NN.softmax(NN.dense(Dn, { d: st.h, s: [st.h.length] }).d, temp);
      let r = Math.random(), j = 0;
      while (j < p.length - 1 && (r -= p[j]) > 0) j++;
      cur = j;
      text += sh.chars[j];
      if (k % 25 === 0) { $('#shOut').textContent = text; await tick(); }
    }
    $('#shOut').textContent = text;
    $('#shRun').disabled = false;
  };
  $('#shRun').addEventListener('click', runSh);
  $('#shOut').textContent = sh.samples.torch;
}

({ index: initIndex, ml: initMl, cnn: initCnn, rnn: initRnn })[PAGE]?.().catch((e) => {
  console.error(e);
  const m = $('main');
  m?.insertAdjacentHTML('afterbegin', `<div class="alert warn">Không nạp được dữ liệu: ${esc(e.message)}</div>`);
});
