// In-browser retraining of the fatigue heads, mirroring the Python scanner:
// StandardScaler + LogisticRegression(C, class_weight="balanced") on DINOv2 embeddings,
// with the web photos weighted so that together they count as much as the real photos.
// Runs as a module Web Worker (postMessage {cmd:"retrain", ...}) and is importable in Node for tests.

export function fitLR(X, y, classes, C, sw = null, iters = 400) {
  const n = X.length, d = X[0].length, K = classes.length, K2 = K === 2 ? 1 : K;
  const yi = Int32Array.from(y, v => classes.indexOf(v));
  // StandardScaler: unweighted mean / population std, zero std -> 1.
  const mean = new Float64Array(d), std = new Float64Array(d);
  for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (const r of X) for (let j = 0; j < d; j++) std[j] += (r[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1;
  const Z = new Float64Array(n * d);
  X.forEach((r, i) => { for (let j = 0; j < d; j++) Z[i * d + j] = (r[j] - mean[j]) / std[j]; });
  // Balanced class weights from sample-weighted counts (sklearn >= 1.6), times sample weights.
  const w0 = i => (sw ? sw[i] : 1), cnt = new Float64Array(K); let tot = 0;
  yi.forEach((k, i) => { cnt[k] += w0(i); tot += w0(i); });
  const s = Float64Array.from(yi, (k, i) => w0(i) * tot / (K * cnt[k]));
  const S = s.reduce((a, b) => a + b, 0);
  // Minimise [sum_i s_i CE_i + ||W||^2 / (2C)] / S with Adam.
  const P = K2 * d + K2, th = new Float64Array(P), m = new Float64Array(P), v = new Float64Array(P), gr = new Float64Array(P);
  const z = new Float64Array(K2);
  for (let t = 1; t <= iters; t++) {
    gr.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * d;
      for (let k = 0; k < K2; k++) {
        let a = th[K2 * d + k]; const w = k * d;
        for (let j = 0; j < d; j++) a += th[w + j] * Z[off + j];
        z[k] = a;
      }
      if (K2 === 1) z[0] = 1 / (1 + Math.exp(-z[0])) - (yi[i] === 1 ? 1 : 0);
      else {
        let mx = -Infinity; for (let k = 0; k < K; k++) mx = Math.max(mx, z[k]);
        let se = 0; for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - mx); se += z[k]; }
        for (let k = 0; k < K; k++) z[k] = z[k] / se - (yi[i] === k ? 1 : 0);
      }
      for (let k = 0; k < K2; k++) {
        const e = z[k] * s[i]; if (!e) continue; const w = k * d;
        for (let j = 0; j < d; j++) gr[w + j] += e * Z[off + j];
        gr[K2 * d + k] += e;
      }
    }
    for (let p = 0; p < K2 * d; p++) gr[p] += th[p] / C;
    const lr = 0.05 * Math.min(1, 1.5 - t / iters);
    for (let p = 0; p < P; p++) {
      const g = gr[p] / S;
      m[p] = 0.9 * m[p] + 0.1 * g; v[p] = 0.999 * v[p] + 0.001 * g * g;
      th[p] -= lr * (m[p] / (1 - 0.9 ** t)) / (Math.sqrt(v[p] / (1 - 0.999 ** t)) + 1e-8);
    }
  }
  // Fold the scaler into the weights; binary becomes two softmax rows [-z/2, z/2].
  let W = [], b = [];
  for (let k = 0; k < K2; k++) {
    const row = new Array(d); let bk = th[K2 * d + k];
    for (let j = 0; j < d; j++) { row[j] = th[k * d + j] / std[j]; bk -= row[j] * mean[j]; }
    W.push(row); b.push(bk);
  }
  if (K2 === 1) { W = [W[0].map(x => -x / 2), W[0].map(x => x / 2)]; b = [-b[0] / 2, b[0] / 2]; }
  return { classes, W, b };
}

export function predict(h, e) {
  const z = h.W.map((w, k) => w.reduce((a, v, j) => a + v * e[j], h.b[k]));
  const mx = Math.max(...z), ex = z.map(v => Math.exp(v - mx)), s = ex.reduce((a, c) => a + c, 0);
  return ex.map(v => v / s);
}
const argmax = a => a.indexOf(Math.max(...a));

// GroupKFold-style split: whole days go to one fold, largest days first, into the emptiest fold.
function dayFolds(groups, k) {
  const size = new Map(); for (const g of groups) size.set(g, (size.get(g) || 0) + 1);
  const days = [...size.keys()].sort((a, b) => size.get(b) - size.get(a));
  k = Math.min(k, days.length);
  const load = new Array(k).fill(0), fold = new Map();
  for (const g of days) { const f = load.indexOf(Math.min(...load)); fold.set(g, f); load[f] += size.get(g); }
  return { k, of: groups.map(g => fold.get(g)) };
}

// Accuracy on days the model never saw (C fixed), then the final fit on everything.
export function trainHead(X, y, groups, classes, C, web = null) {
  const fitOn = idx => {
    let Xt = idx.map(i => X[i]), yt = idx.map(i => y[i]), sw = null;
    if (web) { Xt = Xt.concat(web.X); yt = yt.concat(web.y); sw = idx.map(() => 1).concat(web.y.map(() => idx.length / web.y.length)); }
    return fitLR(Xt, yt, classes, C, sw);
  };
  const { k, of } = dayFolds(groups, 5);
  let right = 0;
  for (let f = 0; f < k; f++) {
    const tr = [], te = [];
    of.forEach((g, i) => (g === f ? te : tr).push(i));
    const h = fitOn(tr);
    for (const i of te) right += classes[argmax(predict(h, X[i]))] === y[i];
  }
  const base = Math.max(...classes.map(c => y.filter(v => v === c).length)) / y.length;
  return { ...fitOn(X.map((_, i) => i)), C, acc: right / y.length, baseline: base, n: y.length };
}

// data: {real:{X,y,days}, web:{X,y}, scans:[{e,label,day}], C:{binary,three}}
export function retrain(data, progress = () => {}) {
  const X = data.real.X.concat(data.scans.map(s => s.e));
  const y = data.real.y.concat(data.scans.map(s => s.label));
  const g = data.real.days.concat(data.scans.map(s => 'scan-' + s.day));   // webcam days never share a fold with photo days
  const bi = y.map((v, i) => i).filter(i => y[i] !== 'strained');
  progress('alert vs fatigued');
  const binary = trainHead(bi.map(i => X[i]), bi.map(i => y[i]), bi.map(i => g[i]), ['alert', 'fatigued'], data.C.binary, data.web);
  progress('alert / strained / fatigued');
  const three = trainHead(X, y, g, ['alert', 'fatigued', 'strained'], data.C.three);
  return { binary, three, n: y.length, n_scans: data.scans.length };
}

if (typeof self !== 'undefined' && typeof window === 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = e => {
    try { self.postMessage({ done: retrain(e.data, step => self.postMessage({ step })) }); }
    catch (err) { self.postMessage({ error: String(err && err.message || err) }); }
  };
}
