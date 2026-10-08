/**
 * NN.js — suy luận mạng nơ-ron bằng JavaScript thuần, viết lại từ cùng công thức với lib/scratch.py.
 *
 * Mô hình là JSON do lib/tl.py::export_model xuất: { spec, in_shape, weights } với trọng số theo
 * bố cục NumPy của tiểu luận (Dense W[in,out] · Conv W[cout,cin,k,k] (NCHW) · LSTM Wx[D,4H], Wh[H,4H],
 * cổng i,f,g,o · GRU cổng z,r,n, thêm bias bh phía h). Mảng được mã hoá base64 float32 hoặc float16.
 * Một "tensor" ở đây là { d: Float32Array|Float64Array, s: [kích thước] } (không có chiều batch).
 */
(function (global) {
  'use strict';

  // ------------------------------------------------------------- giải mã trọng số
  function f16ToF32(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  function decode(t) {
    const bin = atob(t.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (t.dtype === 'f16') {
      const u = new Uint16Array(bytes.buffer);
      const out = new Float32Array(u.length);
      for (let i = 0; i < u.length; i++) out[i] = f16ToF32(u[i]);
      return out;
    }
    return new Float32Array(bytes.buffer);
  }
  function load(json) {
    return {
      spec: json.spec, in_shape: json.in_shape,
      layers: json.spec.map((item, i) => {
        const w = {};
        for (const [k, v] of Object.entries(json.weights[i] || {})) w[k] = { d: decode(v), s: v.shape };
        return { kind: item[0], args: item.slice(1), w };
      }),
    };
  }

  const sig = (x) => 1 / (1 + Math.exp(-x));
  const prod = (s) => s.reduce((a, b) => a * b, 1);

  // ------------------------------------------------------------- các lớp
  function dense(L, x) {
    const W = L.w.W.d, b = L.w.b.d, [nin, nout] = L.w.W.s;
    const rows = prod(x.s) / nin;                      // (D) hoặc (T, D): áp dụng cho từng hàng
    const out = new Float64Array(rows * nout);
    for (let r = 0; r < rows; r++) {
      const o = r * nout, xi = r * nin;
      for (let j = 0; j < nout; j++) out[o + j] = b[j];
      for (let i = 0; i < nin; i++) {
        const v = x.d[xi + i];
        if (v === 0) continue;
        const wr = i * nout;
        for (let j = 0; j < nout; j++) out[o + j] += v * W[wr + j];
      }
    }
    return { d: out, s: x.s.slice(0, -1).concat(nout) };
  }

  function conv(L, x) {
    // Y[co,h,w] = b[co] + Σ_ci Σ_u Σ_v W[co,ci,u,v] · X[ci, h+u-p, w+v-p]   (padding 'same', bước 1)
    const [C, H, Wd] = x.s, [CO, , K] = L.w.W.s, p = K >> 1, W = L.w.W.d, b = L.w.b.d;
    const out = new Float64Array(CO * H * Wd);
    for (let co = 0; co < CO; co++) {
      const ob = co * H * Wd;
      out.fill(b[co], ob, ob + H * Wd);
      for (let ci = 0; ci < C; ci++) {
        const xb = ci * H * Wd;
        for (let u = 0; u < K; u++) {
          for (let v = 0; v < K; v++) {
            const wv = W[((co * C + ci) * K + u) * K + v];
            for (let h = 0; h < H; h++) {
              const hh = h + u - p;
              if (hh < 0 || hh >= H) continue;
              const orow = ob + h * Wd, xrow = xb + hh * Wd;
              const w0 = Math.max(0, p - v), w1 = Math.min(Wd, Wd + p - v);
              for (let w = w0; w < w1; w++) out[orow + w] += wv * x.d[xrow + w + v - p];
            }
          }
        }
      }
    }
    return { d: out, s: [CO, H, Wd] };
  }

  function pool(x) {
    const [C, H, W] = x.s, h2 = H >> 1, w2 = W >> 1;
    const out = new Float64Array(C * h2 * w2);
    for (let c = 0; c < C; c++)
      for (let i = 0; i < h2; i++)
        for (let j = 0; j < w2; j++) {
          const a = c * H * W + 2 * i * W + 2 * j;
          out[(c * h2 + i) * w2 + j] = Math.max(x.d[a], x.d[a + 1], x.d[a + W], x.d[a + W + 1]);
        }
    return { d: out, s: [C, h2, w2] };
  }

  function embed(L, x) {
    const E = L.w.E.d, D = L.w.E.s[1], T = x.d.length;
    const out = new Float64Array(T * D);
    for (let t = 0; t < T; t++) out.set(E.subarray(x.d[t] * D, x.d[t] * D + D), t * D);
    return { d: out, s: [T, D] };
  }

  // ------------------------------------------------------------- ô hồi quy
  /** Một bước của ô hồi quy. state = {h, c}; trả state mới. x là mảng D phần tử. */
  function cellStep(L, x, st) {
    const Wx = L.w.Wx.d, Wh = L.w.Wh.d, b = L.w.b.d;
    const [D, GH] = L.w.Wx.s, H = L.w.Wh.s[0];
    const zx = Float64Array.from(b), zh = new Float64Array(GH);
    for (let i = 0; i < D; i++) { const v = x[i]; if (v === 0) continue; const r = i * GH; for (let j = 0; j < GH; j++) zx[j] += v * Wx[r + j]; }
    for (let i = 0; i < H; i++) { const v = st.h[i]; if (v === 0) continue; const r = i * GH; for (let j = 0; j < GH; j++) zh[j] += v * Wh[r + j]; }
    const h = new Float64Array(H);
    if (L.kind === 'rnn') {
      for (let k = 0; k < H; k++) h[k] = Math.tanh(zx[k] + zh[k]);
      return { h, c: null };
    }
    if (L.kind === 'lstm') {                         // c = σ(f)c + σ(i)tanh(g); h = σ(o)tanh(c)
      const c = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        const i = sig(zx[k] + zh[k]), f = sig(zx[H + k] + zh[H + k]);
        const g = Math.tanh(zx[2 * H + k] + zh[2 * H + k]), o = sig(zx[3 * H + k] + zh[3 * H + k]);
        c[k] = f * st.c[k] + i * g;
        h[k] = o * Math.tanh(c[k]);
      }
      return { h, c };
    }
    const bh = L.w.bh.d;                             // GRU: n = tanh(xWn + bn + r⊙(hWhn + bhn)); h = z h + (1-z) n
    for (let k = 0; k < H; k++) {
      const z = sig(zx[k] + zh[k] + bh[k]), r = sig(zx[H + k] + zh[H + k] + bh[H + k]);
      const n = Math.tanh(zx[2 * H + k] + r * (zh[2 * H + k] + bh[2 * H + k]));
      h[k] = z * st.h[k] + (1 - z) * n;
    }
    return { h, c: null };
  }
  function initState(L) {
    const H = L.w.Wh.s[0];
    return { h: new Float64Array(H), c: new Float64Array(H) };
  }
  function recurrent(L, x) {
    const [T, D] = x.s, H = L.w.Wh.s[0], seq = !!L.args[1];
    let st = initState(L);
    const out = seq ? new Float64Array(T * H) : null;
    for (let t = 0; t < T; t++) {
      st = cellStep(L, x.d.subarray(t * D, t * D + D), st);
      if (seq) out.set(st.h, t * H);
    }
    return seq ? { d: out, s: [T, H] } : { d: st.h, s: [H] };
  }

  // ------------------------------------------------------------- mô hình
  function forward(model, x, upto = model.layers.length) {
    let t = x;
    for (let i = 0; i < upto; i++) {
      const L = model.layers[i];
      switch (L.kind) {
        case 'dense': t = dense(L, t); break;
        case 'relu': t = { d: t.d.map((v) => (v > 0 ? v : 0)), s: t.s }; break;
        case 'tanh': t = { d: t.d.map(Math.tanh), s: t.s }; break;
        case 'dropout': break;                       // suy luận: không làm gì (inverted dropout)
        case 'conv': t = conv(L, t); break;
        case 'pool': t = pool(t); break;
        case 'flatten': t = { d: t.d, s: [t.d.length] }; break;
        case 'embed': t = embed(L, t); break;
        case 'lstm': case 'gru': case 'rnn': t = recurrent(L, t); break;
        default: throw new Error('lớp lạ: ' + L.kind);
      }
    }
    return t;
  }

  function softmax(z, temp = 1) {
    let m = -Infinity;
    for (const v of z) m = Math.max(m, v / temp);
    const e = Array.from(z, (v) => Math.exp(v / temp - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  global.NN = { load, forward, cellStep, initState, softmax, sig, decode, dense };
})(typeof window !== 'undefined' ? window : globalThis);
