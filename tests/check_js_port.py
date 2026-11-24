"""
Checks that the extension's JavaScript (meet_extension/lapse_core.js) behaves like
the Python pipeline. macOS only (uses the built-in JavaScriptCore `jsc`).

    python tests/check_js_port.py

1. Model: sklearn predict_proba vs the exported JSON trees on random inputs.
2. End to end: live_monitor.main() runs on a synthetic 200 s session (fake camera
   and fake MediaPipe that replay made-up face measurements: blinks, a microsleep,
   a head drop, a drowsy phase, a face dropout, a pulse that rises to 120 bpm).
   The same frames go through the JS Monitor. Alerts, per-tick zoning scores and
   heart rate are compared.
No real recordings are used.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"
MODEL_PKL = os.path.join(ROOT, "..", "final_results", "final_model_face_only.joblib")
MODEL_JSON = os.path.join(ROOT, "meet_extension", "model", "zoning_model.json")
CORE_JS = os.path.join(ROOT, "meet_extension", "lapse_core.js")
CALIB_S = 20


def run_js(driver, data):
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "in.json")
        with open(inp, "w") as fh:
            json.dump(data, fh)
        js = os.path.join(d, "driver.js")
        with open(js, "w") as fh:
            fh.write(f"load({json.dumps(CORE_JS)});\nconst MODEL = new LapseCore.ZoningModel(JSON.parse(readFile("
                     f"{json.dumps(MODEL_JSON)})));\nconst IN = JSON.parse(readFile({json.dumps(inp)}));\n{driver}")
        out = subprocess.run([JSC, js], capture_output=True, text=True)
        if out.returncode:
            raise SystemExit("jsc failed:\n" + out.stderr + out.stdout)
        return json.loads(out.stdout)


def check_model():
    import joblib
    m = joblib.load(MODEL_PKL)
    rng = np.random.default_rng(1)
    med = np.array([m["medians"].get(c, 0) for c in m["features"]], float)
    X = med * rng.lognormal(0, 0.6, (400, len(med))) + rng.normal(0, 0.05, (400, len(med)))
    X[rng.random(X.shape) < 0.05] = np.nan                # some missing values -> medians
    import pandas as pd
    Xd = pd.DataFrame(X, columns=m["features"]).fillna(pd.Series(m["medians"])).fillna(0)
    py = m["model"].predict_proba(Xd)[:, list(m["model"].classes_).index(1)]
    rows = [{c: (None if np.isnan(v) else float(v)) for c, v in zip(m["features"], r)} for r in X]
    js = run_js("const o = IN.map(r => { const x = {}; for (const k in r) x[k] = r[k] == null ? NaN : r[k];"
                " return MODEL.proba(x); }); print(JSON.stringify(o));", rows)
    d = np.abs(np.array(js) - py)
    print(f"model:  400 inputs, max |JS - sklearn| = {d.max():.2e}")
    return d.max() < 1e-9


def synthetic_session(seed=0, dur_s=200, fps=30):
    rng = np.random.default_rng(seed)
    n = dur_s * fps
    t = np.round(np.arange(n) * 1000 / fps + rng.uniform(-3, 3, n)).astype(int)
    t[0] = 0
    ts = t / 1000
    ear = 0.30 + rng.normal(0, 0.008, n)
    k = 0
    while k < n:                                          # normal blinks every 3-5 s
        k += int(rng.uniform(3, 5) * fps)
        ear[k:k + int(rng.integers(4, 7))] = 0.07
    ear[(ts >= 60) & (ts < 62.2)] = 0.06                  # microsleep -> dozing
    pitch = 4 + rng.normal(0, 0.5, n)
    drop = (ts >= 90) & (ts < 91.2)                       # head drop with eyes closing
    ear[drop] = 0.08
    pitch[drop] = 4 + 26 * np.minimum(1, (ts[drop] - 90) / 0.3)
    k = int(100 * fps)
    while k < int(165 * fps):                             # drowsy phase: long closures
        ear[k:k + int(rng.uniform(0.7, 1.3) * fps)] = 0.07
        k += int(rng.uniform(2.0, 3.0) * fps)
    hx = 0.5 + np.cumsum(rng.normal(0, 0.0008, n))
    hy = 0.45 + np.cumsum(rng.normal(0, 0.0008, n))
    face = ~((ts >= 40) & (ts < 43))                      # face out of view
    bpm = np.where(ts < 120, 70.0, 120.0)
    phase = 2 * np.pi * np.cumsum(bpm / 60 / fps)
    pulse = np.sin(phase)
    rgb = np.column_stack([150 + 0.15 * pulse, 110 + 0.6 * pulse, 90 + 0.25 * pulse]) + rng.normal(0, 0.05, (n, 3))
    return dict(t=t, face=face, ear=ear, pitch=pitch, hx=hx, hy=hy, rgb=rgb)


def run_python(s, workdir):
    import live_monitor as lm
    import icps_capture as cap
    import pandas as pd
    src = os.path.join(workdir, "synthetic.mp4")
    pd.DataFrame({"timestamp_ms": s["t"]}).to_csv(os.path.join(workdir, "synthetic_data.csv"), index=False)
    n = len(s["t"])
    idx = {"i": -1}

    class FakeVid:
        def __init__(self, *a): pass
        def isOpened(self): return True
        def get(self, p): return 30.0
        def set(self, *a): pass
        def release(self): pass
        def read(self):
            idx["i"] += 1
            return (idx["i"] < n), np.zeros((4, 4, 3), np.uint8)

    class Res:
        def __init__(self, i):
            self.face_landmarks = [[0]] if s["face"][i] else []
            self.face_blendshapes = [[0]] if s["face"][i] else []

    class FakeLandmarker:
        @staticmethod
        def create_from_options(o): return FakeLandmarker()
        def detect_for_video(self, img, t_ms): return Res(idx["i"])
        def close(self): pass

    def fake_extract(res):
        i = idx["i"]
        if not s["face"][i]:
            return dict(face_present=0, ear=None, ear_left=None, ear_right=None, eye_closed=None, eye_x=None,
                        eye_y=None, gaze_h=None, gaze_v=None, head_x=None, head_y=None, yaw=None, pitch=None, roll=None)
        e = float(s["ear"][i])
        return dict(face_present=1, ear=e, ear_left=e, ear_right=e, eye_closed=int(e < 0.2), eye_x=0.5, eye_y=0.4,
                    gaze_h=0.5, gaze_v=0.0, head_x=float(s["hx"][i]), head_y=float(s["hy"][i]),
                    yaw=0.0, pitch=float(s["pitch"][i]), roll=0.0)

    lm.cv2.VideoCapture = FakeVid
    lm.mp.tasks.vision.FaceLandmarker = FakeLandmarker
    cap.extract_face_features = fake_extract
    lm.skin_rgb = lambda frame, lms, w, h: (*s["rgb"][idx["i"]], 1000)
    lm.blendshape_row = lambda cats: ({}, np.zeros(3))
    st = lm.Status()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        lm.main(["--source", src, "--fast", "--port", "", "--calib", str(CALIB_S), "--model", MODEL_PKL,
                 "--log-dir", workdir], status=st)
    scores = pd.read_csv(next(os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("scores_")))
    return st.data["alerts"], scores


def check_end_to_end():
    s = synthetic_session()
    with tempfile.TemporaryDirectory() as d:
        alerts_py, scores_py = run_python(s, d)
    frames = [[int(t), int(f), float(e) if f else None, float(p) if f else None, float(x) if f else None,
               float(y) if f else None, [float(v) for v in c]] for t, f, e, p, x, y, c in
              zip(s["t"], s["face"], s["ear"], s["pitch"], s["hx"], s["hy"], s["rgb"])]
    js = run_js("""
const m = new LapseCore.Monitor(MODEL, %d); const ticks = [];
const nn = (v) => v == null ? NaN : v;
for (const [t, face, ear, pitch, hx, hy, rgb] of IN) {
  const before = m.nextTick;
  m.frame(t, {face, ear: nn(ear), pitch: nn(pitch), hx: nn(hx), hy: nn(hy)}, face ? rgb : null);
  if (m.nextTick !== before) ticks.push([t / 1000, m.status.zoning_score, m.status.hr, m.status.perclos, m.base ? 1 : 0]);
}
print(JSON.stringify({alerts: m.status.alerts, ticks}));""" % CALIB_S, frames)

    ok = True
    pa = [(a["kind"], round(a["t"], 1)) for a in alerts_py]
    ja = [(a["kind"], round(a["t"], 1)) for a in js["alerts"]]
    print(f"alerts: Python {pa}\n        JS     {ja}")
    if pa != ja:
        ok = False
        print("        MISMATCH")
    jt = np.array([[np.nan if v is None else v for v in r] for r in js["ticks"]], float)
    if len(jt) != len(scores_py):
        print(f"ticks:  count differs (Python {len(scores_py)}, JS {len(jt)})")
        return False
    for name, col, jcol, tol in [("zoning score", "zoning_score", 1, 1e-3), ("heart rate", "hr", 2, 0.6),
                                 ("PERCLOS", "perclos", 3, 2e-3)]:
        p = scores_py[col].to_numpy(float)
        j = jt[:, jcol]
        both = ~np.isnan(p) & ~np.isnan(j)
        nan_mismatch = int((np.isnan(p) != np.isnan(j)).sum())
        md = float(np.abs(p[both] - j[both]).max()) if both.any() else 0.0
        print(f"{name:13s} {both.sum()} ticks compared, max diff {md:.4g}, missing in only one: {nan_mismatch}")
        ok &= md <= tol and nan_mismatch == 0
    return ok


if __name__ == "__main__":
    if not os.path.exists(JSC):
        raise SystemExit("needs macOS JavaScriptCore (jsc)")
    r1 = check_model()
    r2 = check_end_to_end()
    print("\nPASS" if r1 and r2 else "\nFAIL")
    sys.exit(0 if r1 and r2 else 1)
