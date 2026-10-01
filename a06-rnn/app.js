/* RNN Lab — Assignment 06. Logic cho 3 trang: index · churn · amzn.
 * Mọi suy luận dùng RNNJS (rnn.js). Không có máy chủ: chỉ fetch các file JSON tĩnh trong models/. */
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const PAGE = document.body.dataset.page;
const CELLS = ['rnn', 'lstm', 'gru'];
const CNAME = { rnn: 'RNN', lstm: 'LSTM', gru: 'GRU' };
const ANN = Math.sqrt(252) * 100;
const cache = {};
const loadJSON = (p) => (cache[p] ||= fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
  return r.json();
}));
const fmt = (x, d = 2) => x.toLocaleString('vi-VN', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${fmt(100 * x, d)}%`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------------------------------------------------------- theme
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

// ---------------------------------------------------------------- Chart.js helpers
const CHARTS = {};
const PALETTE = { rnn: '#d62728', lstm: '#1f77b4', gru: '#2ca02c' };
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
  const c = new Chart($(`#${id}`), config);
  styleChart(c);
  c.update('none');
  CHARTS[id] = c;
  return c;
}

// ================================================================= CHURN
const CH_FEATS = ['CreditScore', 'Geo_Germany', 'Geo_Spain', 'Gender_Male', 'Age', 'Tenure', 'Balance',
  'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'];
const CH_LABEL = { CreditScore: 'Điểm TD', Geo_Germany: 'Đức', Geo_Spain: 'TBN', Gender_Male: 'Nam', Age: 'Tuổi',
  Tenure: 'Năm', Balance: 'Số dư', NumOfProducts: 'Số SP', HasCrCard: 'Thẻ TD', IsActiveMember: 'Tích cực',
  EstimatedSalary: 'Thu nhập' };

async function loadChurn() {
  const ms = await Promise.all(CELLS.map((c) => loadJSON(`models/churn_${c}.json`)));
  const smp = await loadJSON('models/churn_samples.json');
  return { models: Object.fromEntries(CELLS.map((c, i) => [c, ms[i]])), smp };
}
/** x thô (11) -> chuỗi 11 token [z, one-hot(11)] như churn_tokens() trong Python */
function churnTokens(x, meta) {
  return x.map((v, j) => {
    const t = new Float64Array(12);
    t[0] = (v - meta.mean[j]) / meta.std[j];
    t[1 + j] = 1;
    return t;
  });
}
const churnProb = (M, x) => RNNJS.sigmoid(RNNJS.forward(M, churnTokens(x, M.meta)));

function formToX(form) {
  const f = new FormData(form);
  const g = f.get('Geography');
  return [+f.get('CreditScore'), g === 'Germany' ? 1 : 0, g === 'Spain' ? 1 : 0, f.get('Gender') === 'Male' ? 1 : 0,
    +f.get('Age'), +f.get('Tenure'), +f.get('Balance'), +f.get('NumOfProducts'), f.get('HasCrCard') ? 1 : 0,
    f.get('IsActiveMember') ? 1 : 0, +f.get('EstimatedSalary')];
}
function xToForm(x, form) {
  const set = (n, v) => { form.elements[n].value = v; };
  set('CreditScore', x[0]);
  set('Geography', x[1] ? 'Germany' : x[2] ? 'Spain' : 'France');
  set('Gender', x[3] ? 'Male' : 'Female');
  set('Age', x[4]); set('Tenure', x[5]); set('Balance', Math.round(x[6] / 500) * 500);
  set('NumOfProducts', x[7]); set('EstimatedSalary', Math.round(x[10] / 500) * 500);
  form.elements.HasCrCard.checked = !!x[8];
  form.elements.IsActiveMember.checked = !!x[9];
}

async function initChurn() {
  const { models, smp } = await loadChurn();
  const form = $('#chForm');
  let traceCell = 'gru';
  // khách hàng mẫu: 6 rời bỏ + 6 ở lại
  const pos = smp.samples.filter((s) => s.y === 1).slice(0, 6);
  const neg = smp.samples.filter((s) => s.y === 0).slice(0, 6);
  $('#chSamples').innerHTML = [...pos, ...neg].map((s, i) =>
    `<button type="button" data-i="${smp.samples.indexOf(s)}">KH ${i + 1} (${s.y ? 'rời bỏ' : 'ở lại'})</button>`).join('');
  $('#chSamples').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    xToForm(smp.samples[+b.dataset.i].x, form);
    update();
  });
  $('#traceSel').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    traceCell = b.dataset.cell;
    $$('#traceSel button').forEach((x) => x.classList.toggle('on', x === b));
    update();
  });
  form.addEventListener('input', update);

  // thẻ mô hình + độ quan trọng (tĩnh)
  const best = Object.entries(smp.baselines).sort((a, b) => b[1] - a[1]);
  $('#chCard').innerHTML = `<table class="cmp"><thead><tr><th>Mô hình</th><th class="num">AUC</th><th class="num">F1</th>
    <th class="num">Recall</th><th class="num">Ngưỡng</th><th class="num">Tham số</th></tr></thead><tbody>
    ${CELLS.map((c) => { const m = models[c].meta; return `<tr><td>${CNAME[c]} <span class="muted">(seed ${m.seed})</span></td>
      <td class="num">${m.test.roc_auc.toFixed(3)}</td><td class="num">${m.test.f1.toFixed(3)}</td>
      <td class="num">${m.test.recall.toFixed(3)}</td><td class="num">${m.thr.toFixed(2)}</td>
      <td class="num">${m.params.toLocaleString('vi-VN')}</td></tr>`; }).join('')}
    ${best.map(([k, v]) => `<tr class="base"><td>${k} <span class="muted">(đối chứng)</span></td><td class="num">${v.toFixed(3)}</td>
      <td></td><td></td><td></td><td></td></tr>`).join('')}</tbody></table>
    <p class="note muted">Chỉ số trên tập test 1.500 khách hàng. RNN không vượt mô hình cây (HistGB). Hồ sơ khách hàng
      là dữ liệu bảng, không có trục thời gian để mạng hồi quy khai thác (mục 4.3 và 6.2 trong báo cáo).</p>`;
  const imp = Object.entries(smp.importance).sort((a, b) => b[1] - a[1]);
  makeChart('impChart', {
    type: 'bar',
    data: { labels: imp.map(([k]) => CH_LABEL[k] || k),
      datasets: [{ data: imp.map(([, v]) => v), backgroundColor: '#55a868' }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'AUC giảm khi xáo trộn cột' } }, y: {} } },
  });

  function update() {
    $$('output[data-for]', form).forEach((o) => {
      const v = +form.elements[o.dataset.for].value;
      o.textContent = o.dataset.for === 'Balance' || o.dataset.for === 'EstimatedSalary' ? v.toLocaleString('vi-VN') : v;
    });
    const x = formToX(form);
    const P = Object.fromEntries(CELLS.map((c) => [c, churnProb(models[c], x)]));
    const votes = CELLS.filter((c) => P[c] >= models[c].meta.thr).length;
    $('#chOut').innerHTML = `<div class="prob-grid3">${CELLS.map((c) => {
      const thr = models[c].meta.thr, p = P[c], on = p >= thr;
      return `<div class="gauge ${on ? 'pos' : 'neg'}"><div class="lbl">${CNAME[c]}</div>
        <div class="p">${pct(p)}</div>
        <div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div>
          <div class="thr" style="left:${(100 * thr).toFixed(1)}%" title="ngưỡng ${thr.toFixed(2)}"></div></div>
        <div class="lbl">${on ? 'nguy cơ rời bỏ' : 'ở lại'} · ngưỡng ${thr.toFixed(2)}</div></div>`;
    }).join('')}</div>`;
    $('#chAlert').innerHTML = `<div class="alert ${votes >= 2 ? 'warn' : 'info'}">${votes}/3 mô hình xếp khách hàng này vào nhóm
      <b>${votes >= 2 ? 'nguy cơ rời bỏ' : 'ở lại'}</b>. ${votes >= 2 ? 'Nên ưu tiên liên hệ giữ chân.' : ''}</div>`;

    // (3) dấu vết đọc từng bước
    const M = models[traceCell];
    const tr = RNNJS.forwardTrace(M, churnTokens(x, M.meta));
    const gateMean = tr.gates.map((g) => (g ? g.reduce((a, b) => a + b, 0) / g.length : null));
    makeChart('traceChart', {
      data: {
        labels: CH_FEATS.map((f) => CH_LABEL[f]),
        datasets: [
          { type: 'line', label: `${CNAME[traceCell]}: σ(head(h_t))`, data: tr.out.map(RNNJS.sigmoid), yAxisID: 'y',
            borderColor: PALETTE[traceCell], backgroundColor: PALETTE[traceCell], tension: 0.2, pointRadius: 3 },
          ...(traceCell === 'rnn' ? [] : [{ type: 'bar', label: traceCell === 'lstm' ? 'cổng quên f_t (TB)' : 'cổng cập nhật z_t (TB)',
            data: gateMean, yAxisID: 'y', backgroundColor: 'rgba(140,140,140,.28)' }]),
          { type: 'line', label: 'ngưỡng', data: CH_FEATS.map(() => M.meta.thr), borderDash: [4, 4],
            borderColor: '#888', pointRadius: 0, borderWidth: 1 },
        ],
      },
      options: { scales: { y: { min: 0, max: 1, title: { display: true, text: 'xác suất / mức cổng' } }, x: {} },
        plugins: { legend: { labels: { boxWidth: 12 } } } },
    });

    // (4) thử thay đổi tuổi và số sản phẩm
    const ages = Array.from({ length: 38 }, (_, i) => 18 + 2 * i);
    makeChart('whatIfChart', {
      type: 'line',
      data: { labels: ages, datasets: CELLS.map((c) => ({ label: CNAME[c], borderColor: PALETTE[c], backgroundColor: PALETTE[c],
        pointRadius: 0, borderWidth: 2, data: ages.map((a) => { const z = [...x]; z[4] = a; return churnProb(models[c], z); }) })) },
      options: { scales: { y: { min: 0, max: 1, title: { display: true, text: 'P(rời bỏ)' } },
        x: { title: { display: true, text: `tuổi (hiện tại ${x[4]})` }, ticks: { maxTicksLimit: 10 } } } },
    });
    makeChart('prodChart', {
      type: 'bar',
      data: { labels: ['1 SP', '2 SP', '3 SP', '4 SP'], datasets: CELLS.map((c) => ({ label: CNAME[c], backgroundColor: PALETTE[c],
        data: [1, 2, 3, 4].map((n) => { const z = [...x]; z[7] = n; return churnProb(models[c], z); }) })) },
      options: { scales: { y: { min: 0, max: 1, title: { display: true, text: 'P(rời bỏ)' } }, x: {} },
        plugins: { legend: { display: false } } },
    });
  }
  // mặc định: khách hàng mẫu rời bỏ đầu tiên
  xToForm(pos[0].x, form);
  update();
}

// ================================================================= AMZN
async function loadAmzn() {
  const [D, MR, MV] = await Promise.all(['models/amzn_data.json', 'models/amzn_ret.json', 'models/amzn_vol.json'].map(loadJSON));
  return { D, MR, MV };
}
/** Chạy hai mô hình trên mọi cửa sổ có đủ 60 phiên. Trả về mu[i], lnsig[i] (NaN nếu chưa đủ). */
async function amznPredictAll({ D, MR, MV }, onProgress) {
  const n = D.close.length, L = MR.meta.window;
  const Z = D.feat.map((f) => f.map((v, j) => (v - MR.meta.mean[j]) / MR.meta.std[j]));
  const mu = new Float64Array(n).fill(NaN), lns = new Float64Array(n).fill(NaN);
  for (let i = L - 1; i < n; i++) {
    const win = Z.slice(i - L + 1, i + 1);
    mu[i] = RNNJS.forward(MR, win) * MR.meta.y_scale + MR.meta.y_mean;
    lns[i] = RNNJS.forward(MV, win) * MV.meta.y_scale + MV.meta.y_mean;
    if (i % 100 === 0) { onProgress?.(i / n); await new Promise((r) => setTimeout(r, 0)); }
  }
  return { mu, lns };
}
function nextBusinessDay(s) {
  const d = new Date(`${s}T00:00:00Z`);
  do { d.setUTCDate(d.getUTCDate() + 1); } while ([0, 6].includes(d.getUTCDay()));
  return d.toISOString().slice(0, 10);
}
const vnDate = (s) => s.split('-').reverse().join('/');

async function initAmzn() {
  const A = await loadAmzn();
  const { D } = A;
  const n = D.close.length;
  const fc = D.forecast;
  const k80 = fc.k80, k95 = fc.k95;
  $('#kpis').innerHTML = '<div class="progress"><div class="progress-bar" id="pb"></div><span class="progress-txt" id="pt">Đang chạy 2 mô hình trên mọi cửa sổ…</span></div>';
  const { mu, lns } = await amznPredictAll(A, (f) => { $('#pb').style.width = `${(100 * f).toFixed(0)}%`; });
  const r = D.feat.map((f) => f[0]);
  const lastC = D.close[n - 1], nxt = nextBusinessDay(D.date[n - 1]);
  const m = mu[n - 1], s = Math.exp(lns[n - 1]);
  const band = (k) => [lastC * Math.exp(m - k * s), lastC * Math.exp(m + k * s)];
  // HAR trong JS (để vẽ đối chứng): ln σ̂ = b0 + b1 ln σ_1d(Parkinson) + b2 ln σ_5d + b3 ln σ_22d
  const lnRV = (i, w) => { let a = 0; for (let k = i - w + 1; k <= i; k++) a += r[k] ** 2; return 0.5 * Math.log(a / w + 1e-10); };
  const har = (i) => D.har.b0 + D.har.b[0] * 0.5 * Math.log(D.feat[i][1] ** 2 / (4 * Math.log(2)) + 1e-10)
    + D.har.b[1] * lnRV(i, 5) + D.har.b[2] * lnRV(i, 22);
  const fut5 = (i) => (i + 5 < n ? Math.sqrt([1, 2, 3, 4, 5].reduce((a, k) => a + r[i + k] ** 2, 0) / 5) : NaN);
  const [p80, p95] = [band(k80), band(k95)];
  $('#kpis').innerHTML = `
    <div class="kpi"><span>Đóng cửa ${vnDate(D.date[n - 1])}</span><b>${fmt(lastC)}</b><small>USD (ngày cuối của dữ liệu)</small></div>
    <div class="kpi accent"><span>Dự báo phiên ${vnDate(nxt)}</span><b>${fmt(lastC * Math.exp(m))}</b><small>μ̂ = ${m >= 0 ? '+' : ''}${fmt(100 * m, 3)}%</small></div>
    <div class="kpi"><span>Khoảng 80%</span><b>${fmt(p80[0], 0)}–${fmt(p80[1], 0)}</b><small>± ${fmt(k80, 2)}σ̂</small></div>
    <div class="kpi"><span>Khoảng 95%</span><b>${fmt(p95[0], 0)}–${fmt(p95[1], 0)}</b><small>± ${fmt(k95, 2)}σ̂</small></div>
    <div class="kpi"><span>σ̂ 5 phiên tới</span><b>${fmt(100 * s)}%</b><small>/phiên ≈ ${fmt(s * ANN, 0)}%/năm · HAR ${fmt(100 * Math.exp(har(n - 1)))}%</small></div>`;
  const kAmzn = $('#kAmzn'); if (kAmzn) kAmzn.textContent = `${fmt(p95[0], 0)}–${fmt(p95[1], 0)}`;

  // (1) giá + quạt dự báo
  const drawPrice = (N) => {
    const i0 = N ? n - N : 0;
    const labels = [...D.date.slice(i0), nxt];
    const close = [...D.close.slice(i0), null];
    const nullsBut = (v) => labels.map((_, k) => (k === labels.length - 2 ? lastC : k === labels.length - 1 ? v : null));
    makeChart('priceChart', {
      type: 'line',
      data: { labels, datasets: [
        { label: 'Giá đóng cửa', data: close, borderColor: cssVar('--text') || '#333', borderWidth: 1.4, pointRadius: 0 },
        { label: '95% trên', data: nullsBut(p95[1]), borderColor: 'rgba(44,160,44,.5)', pointRadius: 0, fill: false },
        { label: 'Khoảng 95%', data: nullsBut(p95[0]), borderColor: 'rgba(44,160,44,.5)', backgroundColor: 'rgba(44,160,44,.18)', pointRadius: 0, fill: '-1' },
        { label: '80% trên', data: nullsBut(p80[1]), borderColor: 'rgba(44,160,44,.8)', pointRadius: 0, fill: false },
        { label: 'Khoảng 80%', data: nullsBut(p80[0]), borderColor: 'rgba(44,160,44,.8)', backgroundColor: 'rgba(44,160,44,.30)', pointRadius: 0, fill: '-1' },
        { label: 'Điểm dự báo', data: nullsBut(lastC * Math.exp(m)), borderColor: '#2ca02c', borderDash: [4, 3], pointRadius: (c) => (c.dataIndex === labels.length - 1 ? 4 : 0) },
      ] },
      options: { interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { filter: (it) => !/trên$/.test(it.text), boxWidth: 12 } } },
        scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: 'USD' } } } },
    });
  };
  $('#rangeSel').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    $$('#rangeSel button').forEach((x) => x.classList.toggle('on', x === b));
    drawPrice(+b.dataset.n);
  });
  drawPrice(250);

  // chỉ số giai đoạn test
  const iTest = D.date.indexOf(D.test_start);
  const testIdx = []; for (let i = iTest; i < n - 5; i++) testIdx.push(i);

  // (2) quay lại một ngày
  const sl = $('#tSlider');
  sl.max = testIdx.length - 1;
  sl.value = testIdx.findIndex((i) => D.date[i] >= '2020-03-09');
  const showDay = () => {
    const i = testIdx[+sl.value];
    $('#tDate').textContent = vnDate(D.date[i]);
    const sg = Math.exp(lns[i]), rr = r[i + 1], v5 = fut5(i);
    const inside = Math.abs(rr - mu[i]) <= k95 * sg;
    $('#tOut').innerHTML = `<div class="mini-kpis">
      <div><span>μ̂ (r<sub>t+1</sub>)</span><b>${fmt(100 * mu[i], 3)}%</b></div>
      <div><span>r<sub>t+1</sub> thực</span><b class="${rr >= 0 ? 'up' : 'down'}">${fmt(100 * rr, 2)}%</b></div>
      <div><span>khoảng 95%</span><b>±${fmt(100 * k95 * sg, 2)}%</b><small class="${inside ? 'ok' : 'no'}">${inside ? 'nằm trong' : 'vượt khoảng'}</small></div>
      <div><span>σ̂ 5 phiên</span><b>${fmt(sg * ANN, 0)}%/năm</b></div>
      <div><span>σ thực 5 phiên</span><b>${fmt(v5 * ANN, 0)}%/năm</b></div>
      <div><span>HAR σ̂</span><b>${fmt(Math.exp(har(i)) * ANN, 0)}%/năm</b></div></div>`;
    const w0 = i - 59;
    makeChart('winChart', {
      type: 'line',
      data: { labels: D.date.slice(w0, i + 7), datasets: [
        { label: '60 phiên mô hình thấy', data: D.close.slice(w0, i + 1), borderColor: '#1f77b4', pointRadius: 0, borderWidth: 1.6 },
        { label: '6 phiên sau (ẩn với mô hình)', data: [...Array(59).fill(null), ...D.close.slice(i, i + 7)], borderColor: '#aaa', borderDash: [3, 3], pointRadius: 0 },
      ] },
      options: { plugins: { legend: { labels: { boxWidth: 12 } } }, scales: { x: { ticks: { maxTicksLimit: 6 } }, y: {} } },
    });
  };
  sl.addEventListener('input', showDay);
  showDay();

  // (3) dải 95% cho lợi suất trên test
  let cov80 = 0, cov95 = 0;
  testIdx.forEach((i) => { const z = Math.abs(r[i + 1] - mu[i]) / Math.exp(lns[i]); cov80 += z <= k80; cov95 += z <= k95; });
  makeChart('bandChart', {
    type: 'line',
    data: { labels: testIdx.map((i) => D.date[i + 1]), datasets: [
      { label: 'r_{t+1} thực (%)', data: testIdx.map((i) => 100 * r[i + 1]), borderColor: cssVar('--text') || '#333', borderWidth: 0.8, pointRadius: 0 },
      { label: 'trên', data: testIdx.map((i) => 100 * (mu[i] + k95 * Math.exp(lns[i]))), borderColor: 'rgba(44,160,44,.6)', pointRadius: 0, borderWidth: 1 },
      { label: 'khoảng 95%', data: testIdx.map((i) => 100 * (mu[i] - k95 * Math.exp(lns[i]))), borderColor: 'rgba(44,160,44,.6)', backgroundColor: 'rgba(44,160,44,.18)', fill: '-1', pointRadius: 0, borderWidth: 1 },
    ] },
    options: { plugins: { legend: { labels: { filter: (it) => it.text !== 'trên', boxWidth: 12 } } },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: '%' } } } },
  });
  $('#bandNote').innerHTML = `Độ phủ thực tế trên ${testIdx.length} phiên test: <b>${pct(cov80 / testIdx.length)}</b> cho khoảng 80%
    và <b>${pct(cov95 / testIdx.length)}</b> cho khoảng 95%. Hệ số k = ${fmt(k80, 2)} và ${fmt(k95, 2)} được hiệu chỉnh trên tập val,
    vì lợi suất có đuôi dày hơn phân phối chuẩn (k chuẩn là 1,28 và 1,96).`;

  // (4) biến động
  makeChart('volChart', {
    type: 'line',
    data: { labels: testIdx.map((i) => D.date[i]), datasets: [
      { label: 'σ thực 5 phiên tới', data: testIdx.map((i) => fut5(i) * ANN), borderColor: '#bbb', borderWidth: 1, pointRadius: 0 },
      { label: `${CNAME[A.MV.rnn.cell]} σ̂`, data: testIdx.map((i) => Math.exp(lns[i]) * ANN), borderColor: PALETTE[A.MV.rnn.cell], borderWidth: 1.6, pointRadius: 0 },
      { label: 'HAR σ̂', data: testIdx.map((i) => Math.exp(har(i)) * ANN), borderColor: '#8c564b', borderDash: [4, 3], borderWidth: 1.2, pointRadius: 0 },
    ] },
    options: { interaction: { mode: 'index', intersect: false }, plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: '% / năm' } } } },
  });

  // bảng chỉ số
  const M = D.metrics;
  const mean = (v) => (Array.isArray(v) ? v.reduce((a, b) => a + b, 0) / v.length : v);
  const retRows = ['Dự báo 0', 'AR(1)', 'LSTM', 'GRU'].filter((k) => M.ret[k]).map((k) =>
    `<tr><td>${k}${Array.isArray(M.ret[k].rmse) ? ' <span class="muted">(TB 3 seed)</span>' : ''}</td>
     <td class="num">${fmt(100 * mean(M.ret[k].r2_oos), 2)}%</td><td class="num">${M.ret[k].hit == null || Number.isNaN(mean(M.ret[k].hit)) ? '—' : pct(mean(M.ret[k].hit))}</td></tr>`).join('');
  const volKeys = Object.keys(M.vol).filter((k) => M.vol[k] && M.vol[k].rmse !== undefined);
  const volRows = volKeys.map((k) => `<tr><td>${k}${Array.isArray(M.vol[k].rmse) ? ' <span class="muted">(TB 3 seed)</span>' : ''}</td>
     <td class="num">${fmt(mean(M.vol[k].rmse), 3)}</td><td class="num">${fmt(mean(M.vol[k].r2), 3)}</td></tr>`).join('');
  $('#amznTable').innerHTML = `
    <table class="cmp"><thead><tr><th>Lợi suất r<sub>t+1</sub></th><th class="num">R²<sub>oos</sub></th><th class="num">đúng chiều</th></tr></thead><tbody>${retRows}</tbody></table>
    <table class="cmp"><thead><tr><th>Biến động ln σ<sub>5</sub></th><th class="num">RMSE</th><th class="num">R²</th></tr></thead><tbody>${volRows}</tbody></table>
    <p class="note muted">Tỉ lệ phiên tăng trên test là ${pct(M.ret.up_rate)}: chiến lược "luôn đoán tăng" đã đạt mức đúng chiều đó.</p>`;
}

// ================================================================= INDEX
async function initIndex() {
  try {
    const m = await loadJSON('models/churn_gru.json');
    $('#kChurn').textContent = m.meta.test.roc_auc.toFixed(3);
    const D = await loadJSON('models/amzn_data.json');
    $('#kAmzn').textContent = `${fmt(D.forecast.pi95[0], 0)}–${fmt(D.forecast.pi95[1], 0)}`;
  } catch (e) { console.error(e); }
  $('#selfTestBtn')?.addEventListener('click', async () => {
    const out = $('#selfTestOut');
    out.hidden = false;
    out.textContent = 'Đang chạy…\n';
    const rep = await window.__selfTest((l) => { out.textContent += `${l}\n`; });
    out.textContent += rep.ok ? '\n✅ Khớp Python.' : '\n❌ Có sai lệch.';
  });
}

// ================================================================= tự kiểm chứng (mọi trang)
window.__selfTest = async (log = console.log) => {
  const t0 = performance.now();
  const rep = { churn: [], amzn: [], n: 0, maxDiff: 0, labelMismatch: 0 };
  const { models, smp } = await loadChurn();
  for (const c of CELLS) {
    let md = 0, mm = 0;
    for (const s of smp.samples) {
      const p = churnProb(models[c], s.x);
      md = Math.max(md, Math.abs(p - s[`p_${c}`]));
      mm += (p >= models[c].meta.thr) !== (s[`p_${c}`] >= models[c].meta.thr);
    }
    rep.churn.push({ model: c, n: smp.samples.length, maxDiff: md, mismatch: mm });
    log(`churn ${CNAME[c].padEnd(4)} n=${smp.samples.length}  max|Δp| = ${md.toExponential(2)}  lệch quyết định = ${mm}`);
    rep.n += smp.samples.length; rep.maxDiff = Math.max(rep.maxDiff, md); rep.labelMismatch += mm;
  }
  const A = await loadAmzn();
  const { mu, lns } = await amznPredictAll(A);
  const idx = new Map(A.D.date.map((d, i) => [d, i]));
  let dr = 0, dv = 0, k = 0;
  A.D.py.date.forEach((d, j) => {
    const i = idx.get(d);
    if (i === undefined || Number.isNaN(mu[i])) return;
    dr = Math.max(dr, Math.abs(mu[i] - A.D.py.ret[j]));
    dv = Math.max(dv, Math.abs(lns[i] - A.D.py.vol[j]));
    k++;
  });
  rep.amzn.push({ model: 'ret', n: k, maxDiff: dr }, { model: 'vol', n: k, maxDiff: dv });
  log(`amzn  lợi suất n=${k}  max|Δμ̂| = ${dr.toExponential(2)} (đơn vị lợi suất, σ ngày ≈ 2·10⁻²)`);
  log(`amzn  biến động n=${k}  max|Δ ln σ̂| = ${dv.toExponential(2)}`);
  rep.n += 2 * k;
  rep.maxDiff = Math.max(rep.maxDiff, dr, dv);
  rep.ok = rep.maxDiff < 1e-4 && rep.labelMismatch === 0;
  log(`TỔNG ${rep.n} phép so · max|Δ| = ${rep.maxDiff.toExponential(2)} · ${(performance.now() - t0).toFixed(0)} ms`);
  return rep;
};

// ---------------------------------------------------------------- khởi động
const INIT = { churn: initChurn, amzn: initAmzn, index: initIndex }[PAGE];
INIT?.().catch((e) => {
  console.error(e);
  const box = $('main');
  box?.insertAdjacentHTML('afterbegin', `<div class="alert warn">Lỗi khi tải mô hình: ${e.message}</div>`);
});
