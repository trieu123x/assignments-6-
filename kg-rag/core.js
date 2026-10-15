/* Lõi tính toán của web Chương 4 (KG và RAG) — viết lại bằng JavaScript thuần đúng như bản Python:
 *   KGE      điểm TransE / DistMult / ComplEx cho mọi thực thể            (lib/kge.py: np_scores)
 *   tokenize tách từ + từ dừng                                             (lib/retr.py: tokenize)
 *   BM25     chỉ mục và điểm BM25 (k1 = 1,2; b = 0,75)                      (lib/retr.py: BM25)
 *   DE       bộ mã hoá kép: trung bình nhúng có trọng số idf → W·u + b → chuẩn hoá  (lib/retr.py: np_encode)
 *   rrf      Reciprocal Rank Fusion                                        (lib/retr.py: rrf)
 *   Linker   liên kết thực thể theo cụm dài nhất                           (lib/rag.py: EntityLinker)
 *   KGIndex  lấy và xếp hạng bộ ba quanh thực thể                          (lib/rag.py: KGIndex)
 * Không có máy chủ: mọi thứ chạy trong trình duyệt từ các file JSON tĩnh trong data/. */
'use strict';
const Core = (() => {
  // ------------------------------------------------------------ float16 (base64) -> Float32Array
  function f16(b64) {
    const bin = atob(b64);
    const n = bin.length >> 1;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return out;
  }

  // ------------------------------------------------------------ KGE
  /** Điểm của mọi thực thể làm đuôi cho truy vấn (h, r). E: N×d, R: 2R×d (đã gồm quan hệ nghịch đảo). */
  function kgeScores(kind, E, R, d, h, r) {
    const N = E.length / d, q = new Float32Array(d), S = new Float32Array(N);
    const ho = h * d, ro = r * d;
    if (kind === 'transe') for (let k = 0; k < d; k++) q[k] = E[ho + k] + R[ro + k];
    else if (kind === 'distmult') for (let k = 0; k < d; k++) q[k] = E[ho + k] * R[ro + k];
    else {
      const m = d / 2;
      for (let k = 0; k < m; k++) {
        const hr = E[ho + k], hi = E[ho + m + k], rr = R[ro + k], ri = R[ro + m + k];
        q[k] = hr * rr - hi * ri;
        q[m + k] = hr * ri + hi * rr;
      }
    }
    if (kind === 'transe') {
      let qq = 0;
      for (let k = 0; k < d; k++) qq += q[k] * q[k];
      for (let i = 0; i < N; i++) {
        let ee = 0, qe = 0;
        const o = i * d;
        for (let k = 0; k < d; k++) { const x = E[o + k]; ee += x * x; qe += q[k] * x; }
        S[i] = -Math.sqrt(Math.max(qq + ee - 2 * qe, 1e-9));
      }
    } else {
      for (let i = 0; i < N; i++) {
        let s = 0;
        const o = i * d;
        for (let k = 0; k < d; k++) s += q[k] * E[o + k];
        S[i] = s;
      }
    }
    return S;
  }
  const argsortDesc = (S) => Array.from(S.keys()).sort((a, b) => S[b] - S[a] || a - b);

  // ------------------------------------------------------------ văn bản
  const STOP = new Set(`a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves s t don doesn didn isn aren wasn weren won`.split(/\s+/));
  const tokenize = (s) => (String(s).toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !STOP.has(w) && w.length > 1);

  class BM25 {
    constructor(docs, k1 = 1.2, b = 0.75) {
      this.post = new Map();
      const toks = docs.map(tokenize);
      const N = docs.length;
      const dl = toks.map((t) => t.length);
      this.avgdl = dl.reduce((a, x) => a + x, 0) / N;
      const tfs = toks.map((t) => { const c = new Map(); for (const w of t) c.set(w, (c.get(w) || 0) + 1); return c; });
      const df = new Map();
      tfs.forEach((c) => c.forEach((_, w) => df.set(w, (df.get(w) || 0) + 1)));
      tfs.forEach((c, i) => c.forEach((tf, w) => {
        const idf = Math.log(1 + (N - df.get(w) + 0.5) / (df.get(w) + 0.5));
        const wt = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[i] / this.avgdl));
        if (!this.post.has(w)) this.post.set(w, []);
        this.post.get(w).push([i, wt]);
      }));
      this.N = N;
    }
    scores(q) {
      const S = new Float64Array(this.N);
      for (const w of tokenize(q)) for (const [i, wt] of this.post.get(w) || []) S[i] += wt;
      return S;
    }
  }

  class DualEncoder {
    constructor(J) {
      this.d = J.dim; this.lq = J.lq; this.lp = J.lp;
      this.E = f16(J.E); this.W = Float32Array.from(J.W); this.b = Float32Array.from(J.b); this.idf = Float32Array.from(J.idf);
      this.w2i = new Map(J.vocab.map((w, i) => [w, i]));
    }
    ids(text, L) { return tokenize(text).map((w) => this.w2i.get(w) ?? 1).slice(0, L); }
    encode(text, L = this.lq) {
      const d = this.d, u = new Float64Array(d);
      let ws = 0;
      for (const id of this.ids(text, L)) {
        const w = this.idf[id];
        ws += w;
        for (let k = 0; k < d; k++) u[k] += w * this.E[id * d + k];
      }
      const v = new Float64Array(d);
      if (ws > 0) for (let k = 0; k < d; k++) u[k] /= ws;
      for (let j = 0; j < d; j++) {
        let s = this.b[j];
        for (let i = 0; i < d; i++) s += u[i] * this.W[i * d + j];
        v[j] = s;
      }
      let n = 0;
      for (let j = 0; j < d; j++) n += v[j] * v[j];
      n = Math.max(Math.sqrt(n), 1e-12);
      return Float32Array.from(v, (x) => x / n);
    }
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  function ranksOf(S) {
    const order = argsortDesc(S), rk = new Int32Array(S.length);
    order.forEach((i, r) => { rk[i] = r + 1; });
    return rk;
  }
  function rrf(list, k = 60) {
    const out = new Float64Array(list[0].length);
    for (const S of list) { const rk = ranksOf(S); for (let i = 0; i < out.length; i++) out[i] += 1 / (k + rk[i]); }
    return out;
  }

  // ------------------------------------------------------------ đồ thị
  const norm = (s) => ` ${String(s).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()} `;
  class Linker {
    constructor(aliases, minLen = 3, stop = []) {   // aliases: Map(chỉ số -> [tên…])
      this.lut = new Map();
      const st = new Set(stop);
      aliases.forEach((names, i) => names.forEach((n) => {
        const k = norm(n).trim();
        if (k.length >= minLen && !st.has(k)) { if (!this.lut.has(k)) this.lut.set(k, new Set()); this.lut.get(k).add(i); }
      }));
      this.maxw = Math.max(...[...this.lut.keys()].map((k) => k.split(' ').length));
    }
    link(text) {
      const w = norm(text).trim().split(' ').filter(Boolean), spans = [];
      for (let n = Math.min(this.maxw, w.length); n > 0; n--) {
        for (let s = 0; s + n <= w.length; s++) {
          const k = w.slice(s, s + n).join(' ');
          if (this.lut.has(k) && !spans.some(([a, b]) => a <= s && s + n <= b)) spans.push([s, s + n, k]);
        }
      }
      spans.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      const out = [];
      for (const [, , k] of spans) for (const i of [...this.lut.get(k)].sort((a, b) => a - b)) if (!out.includes(i)) out.push(i);
      return out;
    }
  }
  class KGIndex {
    constructor(names, triples, relText, relKeys, literals = {}, note = {}, weight = {}) {
      Object.assign(this, { names, T: triples, relText, relKeys, literals, note, weight });
      this.byEnt = new Map();
      for (let j = 0; j < triples.length / 3; j++) {
        for (const e of [triples[3 * j], triples[3 * j + 2]]) {
          if (!this.byEnt.has(e)) this.byEnt.set(e, []);
          this.byEnt.get(e).push(j);
        }
      }
    }
    verbal(j) {
      const T = this.T, n = this.note[j];
      return `${this.names[T[3 * j]]} — ${this.relText[T[3 * j + 1]]} — ${this.names[T[3 * j + 2]]}` + (n ? ` (${n})` : '');
    }
    facts(ents, question, k = 40) {
      const q = norm(question), scored = [];
      ents.slice(0, 4).forEach((e, rankE) => {
        for (const j of this.byEnt.get(e) || []) {
          const r = this.T[3 * j + 1];
          let s = this.relKeys[r].filter((kw) => q.includes(kw)).length * 2;
          const other = this.T[3 * j] === e ? this.T[3 * j + 2] : this.T[3 * j];
          s += 0.5 * (ents.includes(other) ? 1 : 0);
          s += 0.05 * (this.weight[j] || 0);
          scored.push([s - 0.1 * rankE, j]);
        }
        for (const [kws, sent] of this.literals[e] || []) {
          scored.push([kws.filter((kw) => q.includes(kw)).length * 2 - 0.1 * rankE + 0.25, sent]);
        }
      });
      scored.sort((a, b) => b[0] - a[0]);
      const out = [], seen = new Set();
      for (const [s, j] of scored) {
        const txt = typeof j === 'string' ? j : this.verbal(j);
        if (!seen.has(txt)) { seen.add(txt); out.push([s, txt, typeof j === 'string' ? -1 : j]); }
        if (out.length >= k) break;
      }
      return out;
    }
  }

  // ------------------------------------------------------------ prompt (giống lib/rag.py: build_messages)
  const SYSTEM = 'You are a careful domain assistant. Answer in English, concisely (at most about 120 words). '
    + 'When the question asks for a list, list the specific names.';
  function buildPrompt(question, passages = [], facts = []) {
    if (!passages.length && !facts.length) return `${SYSTEM}\n\nQuestion: ${question}\nAnswer from your own knowledge.`;
    const ctx = [];
    if (facts.length) ctx.push('Knowledge-graph facts (subject — relation — object):\n' + facts.map((f) => `- ${f}`).join('\n'));
    if (passages.length) ctx.push('Passages:\n' + passages.map((p, i) => `[${i + 1}] ${p}`).join('\n'));
    return `${SYSTEM}\n\nUse ONLY the context below to answer. If the context does not contain the answer, say you don't know.\n\n`
      + `${ctx.join('\n\n')}\n\nQuestion: ${question}`;
  }

  return { f16, kgeScores, argsortDesc, tokenize, BM25, DualEncoder, dot, rrf, norm, Linker, KGIndex, buildPrompt };
})();
if (typeof module !== 'undefined') module.exports = Core;
