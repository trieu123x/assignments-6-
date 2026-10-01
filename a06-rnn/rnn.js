/**
 * Suy luận RNN / LSTM / GRU thuần JavaScript — viết thẳng từ công thức Chương 2.
 *
 * Trọng số giữ bố cục PyTorch (xuất bởi export_rnn_json trong lib/rnnlib.py):
 *   W_ih (G·H × D), W_hh (G·H × H), b_ih, b_hh (G·H), head W (1 × H), b (1)
 *   thứ tự khối: LSTM (i, f, g, o) · GRU (r, z, n) · RNN (một khối).
 * Tính bằng Float64 nên khớp PyTorch (float32) tới ~1e-6.
 */
(function (global) {
  'use strict';
  const sig = (x) => 1 / (1 + Math.exp(-x));

  function matvec(W, x, out, off = 0) {
    // out[i] += Σ_j W[off+i][j]·x[j]  cho i = 0..out.length-1
    for (let i = 0; i < out.length; i++) {
      const row = W[off + i];
      let s = 0;
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] += s;
    }
    return out;
  }

  /** Chạy một lớp hồi quy trên chuỗi xs (mảng các vectơ D chiều).
   *  trace = true: trả thêm h_t và giá trị cổng ở từng bước (để vẽ). */
  function runLayer(L, xs, trace = false) {
    const H = L.hidden;
    const G = { rnn: 1, lstm: 4, gru: 3 }[L.cell];
    let h = new Float64Array(H);
    let c = new Float64Array(H);
    const tr = { h: [], gates: [] };
    for (const x of xs) {
      const a = Float64Array.from(L.b_ih);             // W_ih x + b_ih  (G·H)
      matvec(L.W_ih, x, a);
      const b = Float64Array.from(L.b_hh);             // W_hh h + b_hh  (G·H)
      matvec(L.W_hh, h, b);
      const hn = new Float64Array(H);
      let gate = null;
      if (L.cell === 'rnn') {
        for (let k = 0; k < H; k++) hn[k] = Math.tanh(a[k] + b[k]);
      } else if (L.cell === 'lstm') {
        const cn = new Float64Array(H);
        const fg = new Float64Array(H);
        for (let k = 0; k < H; k++) {
          const i = sig(a[k] + b[k]);
          const f = sig(a[H + k] + b[H + k]);
          const g = Math.tanh(a[2 * H + k] + b[2 * H + k]);
          const o = sig(a[3 * H + k] + b[3 * H + k]);
          cn[k] = f * c[k] + i * g;                    // c_t = f ⊙ c_{t-1} + i ⊙ g
          hn[k] = o * Math.tanh(cn[k]);                // h_t = o ⊙ tanh(c_t)
          fg[k] = f;
        }
        c = cn;
        gate = fg;
      } else {
        const zg = new Float64Array(H);
        for (let k = 0; k < H; k++) {
          const r = sig(a[k] + b[k]);
          const z = sig(a[H + k] + b[H + k]);
          const n = Math.tanh(a[2 * H + k] + r * b[2 * H + k]);   // r nhân SAU W_hn h + b_hn
          hn[k] = (1 - z) * n + z * h[k];              // h_t = (1−z) ⊙ n + z ⊙ h_{t−1}
          zg[k] = z;
        }
        gate = zg;
      }
      h = hn;
      if (trace) { tr.h.push(h); tr.gates.push(gate); }
    }
    return { h, trace: tr };
  }

  function head(M, h) {
    let s = M.head.b[0];
    for (let k = 0; k < h.length; k++) s += M.head.W[0][k] * h[k];
    return s;
  }

  /** Đầu ra many-to-one (logit hoặc giá trị chuẩn hoá). */
  function forward(M, xs) { return head(M, runLayer(M.rnn, xs).h); }

  /** Đầu ra nếu "dừng đọc" sau từng bước t (áp lớp đầu ra lên h_t) — chỉ để minh hoạ. */
  function forwardTrace(M, xs) {
    const { trace } = runLayer(M.rnn, xs, true);
    return { out: trace.h.map((h) => head(M, h)), gates: trace.gates };
  }

  global.RNNJS = { forward, forwardTrace, sigmoid: sig, runLayer };
})(typeof window !== 'undefined' ? window : globalThis);
