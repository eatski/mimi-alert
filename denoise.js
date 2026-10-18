// 再生用のノイズ除去（スペクトルゲーティング）
// クリップ内で各周波数の「静かなときの大きさ」をノイズとみなし、それを下回る成分を抑える。
// 話し声専用の除去と違い、インターホンや電子音も残る。

const N = 1024, HOP = 256;
const NOISE_PERCENTILE = 0.2; // 各周波数で下位20%の大きさをノイズの目安にする
const OVER = 1.8;             // ノイズを多めに見積もって確実に消す
const FLOOR = 0.03;           // 完全には消さない（不自然になるため）

const win = new Float32Array(N).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
const LOG2N = Math.log2(N);
const rev = new Uint32Array(N).map((_, i) => {
  let r = 0;
  for (let b = 0; b < LOG2N; b++) r |= ((i >> b) & 1) << (LOG2N - 1 - b);
  return r;
});
const cos = new Float32Array(N / 2).map((_, i) => Math.cos((-2 * Math.PI * i) / N));
const sin = new Float32Array(N / 2).map((_, i) => Math.sin((-2 * Math.PI * i) / N));

// in-place の基数2 FFT（inverse=true で逆変換、1/N は呼び出し側で）
function fft(re, im, inverse) {
  for (let i = 0; i < N; i++) {
    const j = rev[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  const sign = inverse ? -1 : 1;
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, step = N / size;
    for (let start = 0; start < N; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sign * sin[k * step];
        const a = start + k, b = a + half;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
      }
    }
  }
}

export function denoise(x) {
  const bins = N / 2 + 1;
  const frames = Math.max(1, Math.ceil((x.length - N) / HOP) + 1);
  const specRe = [], specIm = [], mag = [];
  for (let f = 0; f < frames; f++) {
    const re = new Float32Array(N), im = new Float32Array(N);
    for (let i = 0; i < N; i++) re[i] = (x[f * HOP + i] ?? 0) * win[i];
    fft(re, im, false);
    const m = new Float32Array(bins);
    for (let k = 0; k < bins; k++) m[k] = Math.hypot(re[k], im[k]);
    specRe.push(re); specIm.push(im); mag.push(m);
  }

  // 周波数ごとのノイズの大きさ
  const noise = new Float32Array(bins);
  const col = new Float32Array(frames);
  for (let k = 0; k < bins; k++) {
    for (let f = 0; f < frames; f++) col[f] = mag[f][k];
    col.sort();
    noise[k] = col[Math.floor((frames - 1) * NOISE_PERCENTILE)] * OVER;
  }

  // 抑えるかどうかのマスク。ブツブツした音（ミュージカルノイズ）を防ぐため前後・隣の周波数でならす
  const mask = mag.map((m) => m.map((v, k) => Math.max(FLOOR, 1 - noise[k] / (v + 1e-9))));
  const smooth = (f, k) => {
    let s = 0, n = 0;
    for (let df = -2; df <= 2; df++) for (let dk = -1; dk <= 1; dk++) {
      const ff = f + df, kk = k + dk;
      if (ff < 0 || ff >= frames || kk < 0 || kk >= bins) continue;
      s += mask[ff][kk]; n++;
    }
    return s / n;
  };

  const out = new Float32Array(x.length), norm = new Float32Array(x.length);
  for (let f = 0; f < frames; f++) {
    const re = specRe[f], im = specIm[f];
    for (let k = 0; k < bins; k++) {
      const g = smooth(f, k);
      re[k] *= g; im[k] *= g;
      if (k > 0 && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; } // 実信号の対称性
    }
    fft(re, im, true);
    for (let i = 0; i < N; i++) {
      const p = f * HOP + i;
      if (p >= x.length) break;
      out[p] += (re[i] / N) * win[i];
      norm[p] += win[i] * win[i];
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = norm[i] > 1e-3 ? out[i] / norm[i] : 0;
  return out;
}
