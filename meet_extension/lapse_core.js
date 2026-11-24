// Detection logic for the extension: a line-by-line port of the Python pipeline
// (icps_capture.extract_face_features, lapse_features.add_session_signals /
// window_features, live_monitor.AlertRules / calibrate / main loop, and the rPPG
// helpers in extract_video_features.py). No browser APIs here, so it can be
// checked against Python (see tests/check_js_port.py).
(function (G) {
  "use strict";

  // ---- constants (same names/values as the Python files) ----
  const BLINK_CLOSURE = 0.5, PERCLOS_CLOSURE = 0.8, BLINK_MIN_MS = 50, BLINK_MAX_MS = 500;
  const WINDOW_S = 10, BUFFER_S = 120, TICK_S = 2;
  const MICROSLEEP_S = 1.5, HEAD_DROP_DEG = 20, PERCLOS_WIN_S = 60, PERCLOS_ALERT = 0.15, ZONE_CONSEC = 2;
  const HR_ABS = 100, HR_REL = 20, HR_SUSTAIN_S = 15;
  const LIVE_HR_N = 3, LIVE_HR_MIN_SNR = -1, LIVE_HR_AGREE_BPM = 8, LIVE_HR_MAX_AGE_S = 8;
  const RPPG_FS = 30, POS_WIN_S = 1.6, HR_BAND = [0.7, 3.0];
  const COOLDOWN = { dozing: 10, drowsy: 60, zoning: 45, stress: 120 };
  const PRESETS = {
    low: { microsleep_s: 2.0, perclos: 0.20, zone_consec: 3, hr_rel: 25 },
    normal: { microsleep_s: MICROSLEEP_S, perclos: PERCLOS_ALERT, zone_consec: ZONE_CONSEC, hr_rel: HR_REL },
    high: { microsleep_s: 1.0, perclos: 0.10, zone_consec: 1, hr_rel: 15 },
  };
  const TITLES = { dozing: "WAKE UP!", drowsy: "You're getting drowsy", zoning: "Zoning out?", stress: "Take a deep breath" };
  const LEFT_EYE = [33, 160, 158, 133, 153, 144], RIGHT_EYE = [362, 385, 387, 263, 373, 380];
  const FOREHEAD = [10, 338, 297, 332, 109, 67, 103, 151, 9, 107, 336, 108, 337, 69, 299];
  const LEFT_CHEEK = [116, 117, 118, 119, 100, 126, 142, 203, 206, 207, 187, 147, 123, 50, 101];
  const RIGHT_CHEEK = [345, 346, 347, 348, 329, 355, 371, 423, 426, 427, 411, 376, 352, 280, 330];
  const SKIN_ROIS = [FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK];
  // scipy.signal.butter(3, [0.7/15, 3.0/15], "band") and lfilter_zi(b, a), fs = 30 Hz
  const BP_B = [0.009109513841753935, 0, -0.027328541525261804, 0, 0.027328541525261804, 0, -0.009109513841753935];
  const BP_A = [1, -4.809547082890498, 9.84778483709453, -11.011829868945608, 7.102549006081874, -2.5062702088808653, 0.3778112693226089];
  const BP_ZI = [-0.009109513841736513, -0.009109513841820304, 0.01821902768361306, 0.01821902768342122, -0.009109513841716849, -0.009109513841760513];

  // ---- numpy-like helpers ----
  const fin = (x) => typeof x === "number" && Number.isFinite(x);
  function percentileSorted(s, p) {               // numpy "linear"
    const n = s.length;
    if (!n) return NaN;
    const idx = (p / 100) * (n - 1), lo = Math.floor(idx), hi = Math.ceil(idx), f = idx - lo;
    return f >= 0.5 ? s[hi] - (s[hi] - s[lo]) * (1 - f) : s[lo] + (s[hi] - s[lo]) * f;
  }
  const sorted = (a) => Float64Array.from(a).sort();
  const median = (a) => percentileSorted(sorted(a), 50);
  function nanmean(a) {
    let s = 0, n = 0;
    for (const v of a) if (fin(v)) { s += v; n++; }
    return n ? s / n : NaN;
  }
  const nanmedian = (a) => median(a.filter(fin));
  function runs(mask) {                            // [start, endExclusive) of each run of true
    const out = [];
    let s = -1;
    for (let i = 0; i <= mask.length; i++) {
      const v = i < mask.length && mask[i];
      if (v && s < 0) s = i;
      else if (!v && s >= 0) { out.push([s, i]); s = -1; }
    }
    return out;
  }

  // ---- per-frame face features (icps_capture.extract_face_features) ----
  function faceFeatures(lm, matrix, W, H) {
    if (!lm) return { face: 0, ear: NaN, pitch: NaN, hx: NaN, hy: NaN };
    const dist = (a, b) => Math.hypot((a.x - b.x) * W, (a.y - b.y) * H);
    const ear = (idx) => {
      const [p1, p2, p3, p4, p5, p6] = idx.map((i) => lm[i]);
      const h = dist(p1, p4);
      return h === 0 ? 0 : (dist(p2, p6) + dist(p3, p5)) / (2 * h);
    };
    let pitch = NaN;
    if (matrix && matrix.data && matrix.data.length === 16) {
      const d = matrix.data;
      // translation sits in d[12..14] when column-major, d[3]/d[7]/d[11] when row-major
      const colMajor = Math.abs(d[12]) + Math.abs(d[13]) + Math.abs(d[14]) >= Math.abs(d[3]) + Math.abs(d[7]) + Math.abs(d[11]);
      const R = (r, c) => (colMajor ? d[c * 4 + r] : d[r * 4 + c]);
      pitch = (Math.atan2(R(2, 1), R(2, 2)) * 180) / Math.PI;
    }
    return { face: 1, ear: (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2, pitch, hx: lm[1].x, hy: lm[1].y };
  }

  // ---- lapse_features.add_session_signals (the parts the model uses) ----
  function sessionSignals(rows) {
    const n = rows.length;
    const S = { t: new Float64Array(n), face: new Uint8Array(n), closure: new Float64Array(n),
                closed: new Uint8Array(n), closed80: new Uint8Array(n), hx: new Float64Array(n), hy: new Float64Array(n) };
    const earFace = [];
    for (const r of rows) if (r.face && fin(r.ear)) earFace.push(r.ear);
    let open = NaN, closedEar = NaN;
    if (earFace.length > 100) {
      const s = sorted(earFace);
      open = percentileSorted(s, 50);
      closedEar = percentileSorted(s, 1);
      if (open - closedEar < 0.05) closedEar = 0.5 * open;
    }
    for (let i = 0; i < n; i++) {
      const r = rows[i], e = fin(r.ear) ? r.ear : NaN;
      let c;
      if (earFace.length > 100) c = Number.isNaN(e) ? NaN : Math.min(1, Math.max(0, (open - e) / (open - closedEar)));
      else c = e < 0.2 ? 1 : 0;
      S.t[i] = r.t; S.face[i] = r.face ? 1 : 0; S.closure[i] = c;
      S.closed[i] = c >= BLINK_CLOSURE && r.face ? 1 : 0;
      S.closed80[i] = c >= PERCLOS_CLOSURE && r.face ? 1 : 0;
      S.hx[i] = fin(r.hx) ? r.hx : NaN; S.hy[i] = fin(r.hy) ? r.hy : NaN;
    }
    return S;
  }

  // ---- lapse_features.window_features (eye_closure group + head_speed_mean) ----
  function windowFeatures(S, tEnd, winMs) {
    const t0 = tEnd - winMs;
    let a = 0;
    while (a < S.t.length && !(S.t[a] > t0)) a++;
    let b = a;
    while (b < S.t.length && S.t[b] <= tEnd) b++;
    const n = b - a, f = { n_frames: n };
    if (n < 5) return f;
    const t = S.t.subarray(a, b);
    const diffs = [];
    for (let i = 1; i < n; i++) diffs.push(t[i] - t[i - 1]);
    const frameMs = n > 1 ? median(diffs) : 33;
    const cl = [], c80 = [];
    let anyFace = false;
    for (let i = a; i < b; i++) if (S.face[i]) { anyFace = true; cl.push(S.closure[i]); c80.push(S.closed80[i]); }
    f.closure_mean = nanmean(cl);
    f.perclos = anyFace ? c80.reduce((x, y) => x + y, 0) / c80.length : NaN;
    // _closure_events
    const durs = [];
    let nLong = 0, mx = 0;
    for (const [s, e] of runs(S.closed.subarray(a, b))) {
      const dur = t[e - 1] - t[s] + frameMs;
      mx = Math.max(mx, dur);
      if (dur >= BLINK_MIN_MS && dur <= BLINK_MAX_MS) durs.push(dur);
      else if (dur > BLINK_MAX_MS) nLong++;
    }
    f.blink_dur_mean = durs.length ? durs.reduce((x, y) => x + y, 0) / durs.length : NaN;
    f.blink_dur_max = durs.length ? Math.max(...durs) : NaN;
    f.long_closures = nLong;
    f.max_closure_ms = mx;
    const sp = [];
    for (let i = a + 1; i < b; i++) {
      const dt = (S.t[i] - S.t[i - 1]) / 1000;
      sp.push(Math.hypot(S.hx[i] - S.hx[i - 1], S.hy[i] - S.hy[i - 1]) / (dt > 0 ? dt : NaN));
    }
    f.head_speed_mean = nanmean(sp);
    return f;
  }

  // ---- live_monitor.final_raw_features ----
  function finalRawFeatures(S, tMs, windows, names, spanMs) {
    const out = {};
    for (const w of windows) {
      const win = spanMs ? Math.trunc(Math.min(w * 1000, spanMs)) : Math.trunc(w * 1000);
      const f = windowFeatures(S, tMs, win);
      for (const c of names) out[`${c}@${w}`] = fin(f[c]) ? f[c] : NaN;
    }
    return out;
  }

  // ---- the trained zoning-out model (sklearn ExtraTreesClassifier, exported to JSON) ----
  class ZoningModel {
    constructor(json) {
      this.j = json;
      this.features = json.features;
      this.windows = json.windows_s;
      this.threshold = json.threshold;
      this.smoothing = json.smoothing_ticks;
      this.base = [...new Set(json.features.map((c) => c.replace(/_pd$/, "").split("@")[0]))];
    }
    proba(x) {                                       // x: {feature: value}
      const v = this.features.map((c, i) => {
        let a = x[c];
        if (!fin(a)) a = this.j.medians[i];
        if (!fin(a)) a = 0;
        return Math.fround(a);                       // sklearn predicts on float32
      });
      let s = 0;
      for (const tr of this.j.trees) {
        let k = 0;
        while (tr.f[k] !== -1) k = v[tr.f[k]] <= tr.t[k] ? tr.l[k] : tr.r[k];
        s += tr.t[k];
      }
      return s / this.j.trees.length;
    }
  }

  // ---- rPPG (extract_video_features: skin_rgb, pos_signal, _bandpass, hr_from_window) ----
  function hull(pts) {                               // monotone chain, int points
    const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }
  function inConvex(h, x, y) {                       // counter-clockwise hull, boundary inclusive
    for (let i = 0; i < h.length; i++) {
      const a = h[i], b = h[(i + 1) % h.length];
      if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) return false;
    }
    return true;
  }
  // Mean R, G, B over forehead + cheeks. img: {width, height, data, x0, y0} (a crop of a w x h frame)
  function skinRoiBox(lm, w, h) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const roi of SKIN_ROIS) for (const i of roi) {
      const x = Math.trunc(lm[i].x * w), y = Math.trunc(lm[i].y * h);
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(w - 1, x1); y1 = Math.min(h - 1, y1);
    return x1 >= x0 && y1 >= y0 ? { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 } : null;
  }
  function skinRGB(lm, w, h, img) {
    const hulls = SKIN_ROIS.map((roi) => hull(roi.map((i) => [Math.trunc(lm[i].x * w), Math.trunc(lm[i].y * h)])));
    let r = 0, g = 0, b = 0, n = 0;
    for (let yy = 0; yy < img.height; yy++) for (let xx = 0; xx < img.width; xx++) {
      const X = xx + img.x0, Y = yy + img.y0;
      if (!hulls.some((hh) => hh.length >= 3 && inConvex(hh, X, Y))) continue;
      const k = (yy * img.width + xx) * 4;
      r += img.data[k]; g += img.data[k + 1]; b += img.data[k + 2]; n++;
    }
    return n < 50 ? null : [r / n, g / n, b / n];
  }
  function interp(x, xp, fp) {                       // np.interp, xp increasing
    const out = new Float64Array(x.length);
    let j = 0;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (v <= xp[0]) { out[i] = fp[0]; continue; }
      if (v >= xp[xp.length - 1]) { out[i] = fp[fp.length - 1]; continue; }
      while (xp[j + 1] < v) j++;
      out[i] = fp[j] + ((fp[j + 1] - fp[j]) * (v - xp[j])) / (xp[j + 1] - xp[j]);
    }
    return out;
  }
  const std = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  function posSignal(R, Gc, B, fs) {
    const n = R.length, l = Math.round(POS_WIN_S * fs), H = new Float64Array(n);
    for (let m = l; m <= n; m++) {
      let mr = 0, mg = 0, mb = 0;
      for (let k = m - l; k < m; k++) { mr += R[k]; mg += Gc[k]; mb += B[k]; }
      mr /= l; mg /= l; mb /= l;
      if (mr === 0 || mg === 0 || mb === 0) continue;
      const s0 = new Array(l), s1 = new Array(l);
      for (let k = 0; k < l; k++) {
        const r = R[m - l + k] / mr, g = Gc[m - l + k] / mg, b = B[m - l + k] / mb;
        s0[k] = g - b; s1[k] = -2 * r + g + b;
      }
      const sd1 = std(s1), alpha = sd1 > 0 ? std(s0) / sd1 : 0;
      const hh = s0.map((v, k) => v + alpha * s1[k]);
      const hm = hh.reduce((x, y) => x + y, 0) / l;
      for (let k = 0; k < l; k++) H[m - l + k] += hh[k] - hm;
    }
    return H;
  }
  function lfilter(b, a, x, zi) {                    // direct form II transposed
    const z = zi.slice(), y = new Float64Array(x.length), m = b.length - 1;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i], yi = b[0] * xi + z[0];
      for (let k = 0; k < m - 1; k++) z[k] = b[k + 1] * xi + z[k + 1] - a[k + 1] * yi;
      z[m - 1] = b[m] * xi - a[m] * yi;
      y[i] = yi;
    }
    return y;
  }
  function bandpass(x) {                             // scipy filtfilt, padtype="odd", padlen=21
    const p = 3 * Math.max(BP_A.length, BP_B.length), n = x.length;
    if (n <= p) return null;
    const ext = new Float64Array(n + 2 * p);
    for (let i = 0; i < p; i++) ext[i] = 2 * x[0] - x[p - i];
    ext.set(x, p);
    for (let i = 0; i < p; i++) ext[n + p + i] = 2 * x[n - 1] - x[n - 2 - i];
    let y = lfilter(BP_B, BP_A, ext, BP_ZI.map((z) => z * ext[0]));
    y = y.reverse();
    y = lfilter(BP_B, BP_A, y, BP_ZI.map((z) => z * y[0]));
    return y.reverse().subarray(p, p + n);
  }
  function hrFromWindow(x, fs) {                     // scipy welch(hann, 50 % overlap, nfft 4096) peak + SNR
    const nper = Math.min(x.length, Math.trunc(fs * 8)), step = nper - Math.trunc(nper / 2), nfft = 4096;
    const nseg = Math.floor((x.length - nper) / step) + 1;
    const win = Array.from({ length: nper }, (_, k) => 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / nper));
    const k0 = Math.ceil((HR_BAND[0] * nfft) / fs), k1 = Math.floor((HR_BAND[1] * nfft) / fs);
    const fb = [], pb = [];
    for (let k = k0; k <= k1; k++) { fb.push((k * fs) / nfft); pb.push(0); }
    for (let s = 0; s < nseg; s++) {
      const seg = x.subarray(s * step, s * step + nper);
      const mean = seg.reduce((u, v) => u + v, 0) / nper;
      const sw = Array.from(seg, (v, j) => (v - mean) * win[j]);
      for (let k = k0; k <= k1; k++) {
        let re = 0, im = 0;
        const w = (-2 * Math.PI * k) / nfft;
        for (let j = 0; j < nper; j++) { re += sw[j] * Math.cos(w * j); im += sw[j] * Math.sin(w * j); }
        pb[k - k0] += re * re + im * im;
      }
    }
    const tot = pb.reduce((u, v) => u + v, 0);
    if (!fb.length || tot === 0) return [NaN, NaN];
    let best = 0;
    for (let i = 1; i < pb.length; i++) if (pb[i] > pb[best]) best = i;
    const f0 = fb[best];
    let sig = 0, noise = 0;
    for (let i = 0; i < fb.length; i++) {
      if (Math.abs(fb[i] - f0) <= 0.1 || Math.abs(fb[i] - 2 * f0) <= 0.2) sig += pb[i]; else noise += pb[i];
    }
    return [60 * f0, noise > 0 ? 10 * Math.log10(sig / noise) : NaN];
  }
  function liveRppg(buf) {                           // live_monitor.live_rppg; buf: [t_ms, r, g, b][]
    if (buf.length < 200) return [NaN, NaN];
    const t = buf.map((q) => q[0]);
    if (t[t.length - 1] - t[0] < 9000) return [NaN, NaN];
    const step = 1000 / RPPG_FS, cnt = Math.ceil((t[t.length - 1] - t[0]) / step);
    const grid = Float64Array.from({ length: cnt }, (_, i) => t[0] + i * step);
    const ch = [1, 2, 3].map((c) => interp(grid, t, buf.map((q) => q[c])));
    const pulse = bandpass(posSignal(ch[0], ch[1], ch[2], RPPG_FS));
    return pulse ? hrFromWindow(pulse, RPPG_FS) : [NaN, NaN];
  }

  // ---- live_monitor.AlertRules ----
  class AlertRules {
    constructor() {
      this.p = { ...PRESETS.normal };
      this.enabled = new Set(Object.keys(COOLDOWN));
      this.last = {};
      this.closedSince = null; this.dropSince = null;
      this.closureHist = []; this.hrHist = []; this.zoneStreak = 0;
    }
    fire(kind, t, msg) {
      if (!this.enabled.has(kind) || t - (this.last[kind] ?? -1e9) < COOLDOWN[kind]) return [];
      this.last[kind] = t;
      return [[kind, msg]];
    }
    onFrame(t, face, closure, pitchDev) {
      let out = [];
      const closed = face && closure != null && closure >= BLINK_CLOSURE;
      this.closedSince = closed ? (this.closedSince ?? t) : null;
      if (this.closedSince != null && t - this.closedSince >= this.p.microsleep_s)
        out = out.concat(this.fire("dozing", t, `Eyes closed for ${(t - this.closedSince).toFixed(1)} s`));
      const drop = closed && pitchDev != null && Math.abs(pitchDev) >= HEAD_DROP_DEG;
      this.dropSince = drop ? (this.dropSince ?? t) : null;
      if (this.dropSince != null && t - this.dropSince >= 0.5)
        out = out.concat(this.fire("dozing", t, "Head dropping with eyes closing"));
      if (face && closure != null) this.closureHist.push([t, closure >= PERCLOS_CLOSURE]);
      let i = 0;
      while (i < this.closureHist.length && this.closureHist[i][0] < t - PERCLOS_WIN_S) i++;
      if (i) this.closureHist.splice(0, i);
      return out;
    }
    perclos() {
      const h = this.closureHist;
      return h.length ? h.reduce((s, q) => s + (q[1] ? 1 : 0), 0) / h.length : NaN;
    }
    onTick(t, zoning, hr, hrBase) {
      let out = [];
      const h = this.closureHist;
      if (h.length > 10 && h[h.length - 1][0] - h[0][0] >= 0.8 * PERCLOS_WIN_S && this.perclos() >= this.p.perclos)
        out = out.concat(this.fire("drowsy", t, `Eyes closed ${Math.round(this.perclos() * 100)}% of the last minute`));
      this.zoneStreak = zoning ? this.zoneStreak + 1 : 0;
      if (this.zoneStreak >= this.p.zone_consec)
        out = out.concat(this.fire("zoning", t, "Looks like your attention drifted - refocus on the call"));
      if (fin(hr)) this.hrHist.push([t, hr]);
      while (this.hrHist.length && this.hrHist[0][0] < t - HR_SUSTAIN_S) this.hrHist.shift();
      const thr = fin(hrBase) ? Math.max(HR_ABS, hrBase + this.p.hr_rel) : HR_ABS;
      if (this.hrHist.length >= (0.7 * HR_SUSTAIN_S) / TICK_S) {
        const med = median(this.hrHist.map((q) => q[1]));
        if (med >= thr) out = out.concat(this.fire("stress", t, `Heart rate is high (${Math.round(med)} bpm). Take a slow deep breath.`));
      }
      return out;
    }
  }

  // ---- live_monitor.calibrate ----
  function calibrate(rows, hrCalib, model) {
    const ear = rows.filter((r) => r.face && fin(r.ear)).map((r) => r.ear);
    const s = sorted(ear);
    const earOpen = ear.length ? percentileSorted(s, 50) : 0.28;
    let earClosed = ear.length ? percentileSorted(s, 1) : 0.1;
    if (earOpen - earClosed < 0.05) earClosed = 0.5 * earOpen;
    const anyFace = rows.some((r) => r.face);
    const base = {
      ear_open: earOpen, ear_closed: earClosed,
      pitch: anyFace ? nanmedian(rows.map((r) => r.pitch)) : 0,
      hr_base: hrCalib.length ? median(hrCalib) : NaN,
      final_base: {},
    };
    if (model && rows.length) {
      const S = sessionSignals(rows);
      const a = rows[0].t, b = rows[rows.length - 1].t;
      const cnt = Math.max(0, Math.ceil((b + 1 - (a + 10000)) / 2000));
      const feats = [];
      for (let i = 0; i < cnt; i++) feats.push(finalRawFeatures(S, a + 10000 + i * 2000, model.windows, model.base, b - a));
      if (feats.length) for (const k of Object.keys(feats[0])) base.final_base[k] = nanmedian(feats.map((f) => f[k]));
    }
    return base;
  }

  // ---- live_monitor.main, one frame at a time ----
  class Monitor {
    constructor(model, calibS) {
      this.model = model; this.calibS = calibS;
      this.rules = new AlertRules();
      this.rows = []; this.rgb = []; this.rppgRows = [];
      this.base = null; this.hrCalib = [];
      this.nextTick = 0; this.zoneScore = NaN; this.zoneEma = null;
      this.hrNow = NaN; this.hrSrc = "-";
      this.status = { running: true, started: true, calibrating: true, calib_left_s: Math.round(calibS), face: false,
                      hr: null, hr_source: "-", perclos: null, zoning_score: null,
                      zoning_threshold: model ? model.threshold : null, alerts: [] };
      this.nextId = 1;
    }
    setSettings(sens, enabledList) {
      this.rules.p = { ...(PRESETS[sens] || PRESETS.normal) };
      this.rules.enabled = new Set(enabledList && enabledList.length ? enabledList : Object.keys(COOLDOWN));
    }
    // tMs: ms since start; f: faceFeatures(); rgb: [r, g, b] or null
    frame(tMs, f, rgb) {
      const t = tMs / 1000;
      this.rows.push({ t: tMs, face: f.face, ear: f.ear, pitch: f.pitch, hx: f.hx, hy: f.hy });
      if (f.face && rgb) this.rgb.push([tMs, rgb[0], rgb[1], rgb[2]]);
      let i = 0;
      while (i < this.rgb.length && this.rgb[i][0] < tMs - 10000) i++;
      if (i) this.rgb.splice(0, i);
      i = 0;
      while (i < this.rows.length && this.rows[i].t < tMs - BUFFER_S * 1000) i++;
      if (i) this.rows.splice(0, i);

      let alerts = [];
      if (t >= this.nextTick) { this.nextTick = t + TICK_S; alerts = alerts.concat(this.tick(t, tMs)); }
      if (this.base) {
        const b = this.base;
        const closure = fin(f.ear) ? Math.min(1, Math.max(0, (b.ear_open - f.ear) / (b.ear_open - b.ear_closed))) : null;
        const pitchDev = fin(f.pitch) ? f.pitch - b.pitch : null;
        alerts = alerts.concat(this.rules.onFrame(t, !!f.face, closure, pitchDev));
      }
      this.status.face = !!f.face;
      for (const [kind, msg] of alerts) {
        this.status.alerts = this.status.alerts.concat([{ id: this.nextId++, kind, title: TITLES[kind], msg, t: Math.round(t * 10) / 10 }]).slice(-20);
      }
      return alerts;
    }
    tick(t, tMs) {
      let alerts = [];
      const [bpm, snr] = liveRppg(this.rgb);
      if (fin(bpm)) { this.rppgRows.push([tMs, bpm, snr]); if (this.rppgRows.length > 60) this.rppgRows.shift(); }
      // live_monitor.LIVE_HR_*: show webcam HR only when the last 3 estimates are clean, recent and agree
      const recent = this.rppgRows.slice(-LIVE_HR_N), bpms = recent.map((q) => q[1]);
      const steady = recent.length === LIVE_HR_N && recent.every((q) => q[2] >= LIVE_HR_MIN_SNR) &&
        tMs - recent[0][0] <= LIVE_HR_MAX_AGE_S * 1000 && Math.max(...bpms) - Math.min(...bpms) <= LIVE_HR_AGREE_BPM;
      if (steady) { this.hrNow = median(bpms); this.hrSrc = "webcam"; } else { this.hrNow = NaN; this.hrSrc = "-"; }

      if (!this.base) {
        if (fin(this.hrNow)) this.hrCalib.push(this.hrNow);
        if (t >= this.calibS) this.base = calibrate(this.rows, this.hrCalib, this.model);
      } else {
        let zoning = false;
        const S = sessionSignals(this.rows);
        if (windowFeatures(S, tMs, WINDOW_S * 1000).n_frames >= 50 && this.model) {
          const raw = finalRawFeatures(S, tMs, this.model.windows, this.model.base);
          const x = { ...raw };
          for (const [k, v] of Object.entries(raw)) {
            const bv = this.base.final_base[k];
            x[k + "_pd"] = fin(v) && fin(bv) ? v - bv : NaN;
          }
          const p = this.model.proba(x);
          const al = 1 - 0.5 ** (1 / this.model.smoothing);
          this.zoneEma = this.zoneEma == null ? p : this.zoneEma + al * (p - this.zoneEma);
          this.zoneScore = this.zoneEma;
          zoning = this.zoneScore >= this.model.threshold;
        }
        alerts = this.rules.onTick(t, zoning, this.hrNow, this.base.hr_base);
      }
      const pc = this.rules.perclos();
      Object.assign(this.status, {
        calibrating: !this.base,
        calib_left_s: this.base ? null : Math.max(0, Math.round(this.calibS - t)),
        hr: fin(this.hrNow) ? Math.round(this.hrNow * 10) / 10 : null, hr_source: this.hrSrc,
        perclos: fin(pc) ? Math.round(pc * 1000) / 1000 : null,
        zoning_score: fin(this.zoneScore) ? Math.round(this.zoneScore * 1000) / 1000 : null,
      });
      return alerts;
    }
  }

  G.LapseCore = { faceFeatures, sessionSignals, windowFeatures, finalRawFeatures, ZoningModel, calibrate,
                  AlertRules, Monitor, skinRoiBox, skinRGB, liveRppg, bandpass, posSignal, hrFromWindow, PRESETS };
})(typeof globalThis !== "undefined" ? globalThis : this);
