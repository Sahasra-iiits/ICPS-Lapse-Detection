"""
Live monitor with alerts (runs next to Meet / Zoom).

    python live_monitor.py                                # webcam
    python live_monitor.py --model lapse_model.joblib     # + trained zoning-out model
    python live_monitor.py --source some_session.mp4      # replay a recording (demo)
    python live_monitor.py --meet                         # helper for the Google Meet extension

--meet: no camera window, no sound from Python; instead a small status server on
http://127.0.0.1:47813 (this computer only) that the Chrome extension in
meet_extension/ reads to show banners + sounds inside Google Meet.
--model accepts a train_final.py model (e.g. final_model_face_only.joblib) or a
train_lapse_model.py model.

Alerts (beep + on-screen banner + desktop notification; Windows/Linux need `plyer`):
  DOZING OFF   eyes closed >= 1.5 s, or head dropping with eyes closing (microsleep)
  DROWSY       PERCLOS over the last 60 s >= 15 %
  ZONING OUT   the trained lapse model says lapse twice in a row
               (without a model: a simple "staring" rule — no blinks, fixed gaze, still head)
  HIGH HR      heart rate >= max(100, your baseline + 20) for 15 s
               -> "take a slow deep breath" + a breathing guide on screen
               HR from the ESP32 if connected, otherwise from the webcam (rPPG)

The first 60 s are calibration: look at the screen and blink normally.
Keys (in the camera window): q = quit, h = demo: fake +35 bpm (shown as SIM).
Alerts are logged to alerts_<time>.csv, and the scores every 2 s to scores_<time>.csv
(compare them with a PVT-B session using compare_alerts.py).
When replaying a recording, its _data.csv timestamps are used, so alert times
line up exactly with the recording's events.
"""

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sounddevice as sd

import icps_capture as cap
import lapse_features as lf
from extract_video_features import (HR_BAND_HZ, RPPG_FS, _bandpass, blendshape_row,
                                    hr_from_window, pos_signal, skin_rgb)

notification = None
if sys.platform != "darwin":          # plyer's macOS backend needs extra packages
    try:
        from plyer import notification
    except ImportError:
        pass

CALIB_S = 60
BUFFER_S = 120            # frames kept for features
TICK_S = 2.0              # how often the model / HR rules run

MICROSLEEP_S = 1.5
HEAD_DROP_DEG = 20.0
PERCLOS_WIN_S = 60
PERCLOS_ALERT = 0.15
ZONE_CONSEC = 2
HR_ABS, HR_REL, HR_SUSTAIN_S = 100.0, 20.0, 15.0
SIM_HR_OFFSET = 35.0

COOLDOWN = {"dozing": 10, "drowsy": 60, "zoning": 45, "stress": 120}
# sensitivity presets (the extension's popup switches between them)
PRESETS = {
    "low":    dict(microsleep_s=2.0, perclos=0.20, zone_consec=3, hr_rel=25.0),
    "normal": dict(microsleep_s=MICROSLEEP_S, perclos=PERCLOS_ALERT, zone_consec=ZONE_CONSEC, hr_rel=HR_REL),
    "high":   dict(microsleep_s=1.0, perclos=0.10, zone_consec=1, hr_rel=15.0),
}
ZONE_SMOOTH_TICKS = 30    # final model: score smoothed over ~60 s (drowsiness is a state)
STYLE = {   # BGR colour, title, beep pattern [(Hz, seconds)]
    "dozing": ((0, 0, 255), "WAKE UP!", [(1200, .15), (0, .05)] * 4),
    "drowsy": ((0, 140, 255), "You're getting drowsy", [(900, .2), (0, .1), (900, .2)]),
    "zoning": ((255, 170, 0), "Zoning out?", [(660, .25)]),
    "stress": ((200, 90, 255), "Take a deep breath", [(440, .4)]),
}
BREATH_IN, BREATH_OUT, BREATH_CYCLES = 4.0, 6.0, 6
# what the extension's demo / preview buttons show (marked "Demo" in the banner)
DEMO_MSG = {
    "dozing": "Your eyes were closed for 2.1 seconds.",
    "drowsy": "Your eyes were closed 18% of the last minute.",
    "zoning": "Your attention seems to have drifted. Refocus on the call.",
    "stress": "Your heart rate is high (112 bpm). Follow the breathing guide.",
}


# ============================================================
# ALERT RULES
# ============================================================

class AlertRules:
    def __init__(self):
        self.p = dict(PRESETS["normal"])
        self.enabled = {"dozing", "drowsy", "zoning", "stress"}
        self.last = {}
        self.closed_since = None
        self.drop_since = None
        self.closure_hist = collections.deque()   # (t, closed80)
        self.hr_hist = collections.deque()        # (t, hr)
        self.zone_streak = 0

    def _fire(self, kind, t, msg):
        if kind not in self.enabled or t - self.last.get(kind, -1e9) < COOLDOWN[kind]:
            return []
        self.last[kind] = t
        return [(kind, msg)]

    def on_frame(self, t, face, closure, pitch_dev):
        out = []
        closed = face and closure is not None and closure >= lf.BLINK_CLOSURE
        self.closed_since = (self.closed_since if self.closed_since is not None else t) if closed else None
        if self.closed_since is not None and t - self.closed_since >= self.p["microsleep_s"]:
            out += self._fire("dozing", t, f"Eyes closed for {t - self.closed_since:.1f} s")

        drop = closed and pitch_dev is not None and abs(pitch_dev) >= HEAD_DROP_DEG
        self.drop_since = (self.drop_since if self.drop_since is not None else t) if drop else None
        if self.drop_since is not None and t - self.drop_since >= 0.5:
            out += self._fire("dozing", t, "Head dropping with eyes closing")

        if face and closure is not None:
            self.closure_hist.append((t, closure >= lf.PERCLOS_CLOSURE))
        while self.closure_hist and self.closure_hist[0][0] < t - PERCLOS_WIN_S:
            self.closure_hist.popleft()
        return out

    def perclos(self):
        return np.mean([c for _, c in self.closure_hist]) if self.closure_hist else np.nan

    def on_tick(self, t, zoning, hr, hr_base):
        out = []
        h = self.closure_hist
        if len(h) > 10 and h[-1][0] - h[0][0] >= 0.8 * PERCLOS_WIN_S and self.perclos() >= self.p["perclos"]:
            out += self._fire("drowsy", t, f"Eyes closed {self.perclos():.0%} of the last minute")

        self.zone_streak = self.zone_streak + 1 if zoning else 0
        if self.zone_streak >= self.p["zone_consec"]:
            out += self._fire("zoning", t, "Looks like your attention drifted - refocus on the call")

        if hr is not None and np.isfinite(hr):
            self.hr_hist.append((t, hr))
        while self.hr_hist and self.hr_hist[0][0] < t - HR_SUSTAIN_S:
            self.hr_hist.popleft()
        thr = max(HR_ABS, hr_base + self.p["hr_rel"]) if np.isfinite(hr_base) else HR_ABS
        if len(self.hr_hist) >= 0.7 * HR_SUSTAIN_S / TICK_S:
            med = float(np.median([v for _, v in self.hr_hist]))
            if med >= thr:
                out += self._fire("stress", t, f"Heart rate is high ({med:.0f} bpm). "
                                               "Take a slow deep breath.")
        return out


# ============================================================
# NOTIFICATIONS
# ============================================================

class Notifier:
    def __init__(self, sound=True, desktop=True):
        self.sound = sound
        self.desktop = desktop and (notification is not None or sys.platform == "darwin")
        self.banner = None          # (kind, msg, until)
        self.breath_start = None

    def alert(self, t, kind, msg):
        print(f"[{t:7.1f}s] {kind.upper()}: {msg}")
        self.banner = (kind, msg, t + 6)
        if kind == "stress":
            self.breath_start = t
        if self.sound:
            threading.Thread(target=self._beep, args=(STYLE[kind][2],), daemon=True).start()
        if self.desktop:
            threading.Thread(target=self._toast, args=(STYLE[kind][1], msg), daemon=True).start()

    @staticmethod
    def _beep(pattern, sr=44100):
        try:
            x = np.concatenate([0.4 * np.sin(2 * np.pi * f * np.arange(int(sr * d)) / sr)
                                for f, d in pattern]).astype(np.float32)
            sd.play(x, sr)
        except Exception as e:
            print("beep failed:", e)

    @staticmethod
    def _toast(title, msg):
        try:
            if sys.platform == "darwin":
                q = lambda s: s.replace("\\", "").replace('"', "'")
                subprocess.run(["osascript", "-e", f'display notification "{q(msg)}" '
                                f'with title "{q(title)}"'], timeout=5, check=False)
            else:
                notification.notify(title=title, message=msg, app_name="ICPS", timeout=5)
        except Exception:
            pass

    def draw(self, img, t):
        h, w = img.shape[:2]
        if self.banner and t < self.banner[2]:
            kind, msg, _ = self.banner
            color = STYLE[kind][0]
            cv2.rectangle(img, (0, 0), (w, 90), color, -1)
            cv2.putText(img, STYLE[kind][1], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            cv2.putText(img, msg, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if self.breath_start is not None:
            el = t - self.breath_start
            cyc = BREATH_IN + BREATH_OUT
            if el > cyc * BREATH_CYCLES:
                self.breath_start = None
                return
            ph = el % cyc
            inhale = ph < BREATH_IN
            frac = ph / BREATH_IN if inhale else 1 - (ph - BREATH_IN) / BREATH_OUT
            r = int(30 + 70 * frac)
            c = (w - 130, h - 140)
            cv2.circle(img, c, 100, (255, 255, 255), 1)
            cv2.circle(img, c, r, (200, 90, 255), -1)
            cv2.putText(img, "Breathe in" if inhale else "Breathe out", (c[0] - 70, c[1] + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ============================================================
# FEATURES
# ============================================================

NO_EVENTS = pd.DataFrame(columns=["timestamp_ms", "event_type", "rt_ms"])


def live_rppg(rgb_buf):
    """HR from the last ~10 s of skin colour. Returns (bpm, snr) or (nan, nan)."""
    if len(rgb_buf) < 200:
        return np.nan, np.nan
    a = np.array(rgb_buf, float)                     # t_ms, r, g, b
    if a[-1, 0] - a[0, 0] < 9000:
        return np.nan, np.nan
    grid = np.arange(a[0, 0], a[-1, 0], 1000 / RPPG_FS)
    u = np.column_stack([np.interp(grid, a[:, 0], a[:, c]) for c in (1, 2, 3)])
    pulse = _bandpass(pos_signal(u, RPPG_FS), RPPG_FS, *HR_BAND_HZ)
    return hr_from_window(pulse, RPPG_FS)


def staring_rule(f, base):
    """Fallback zoning-out rule when no trained model is given."""
    try:
        return (f["longest_interblink_ms"] >= 0.9 * lf.WINDOW_S * 1000
                and f["gaze_dispersion"] <= base["gaze_dispersion_p25"]
                and f["head_still_frac"] >= 0.7)
    except (KeyError, TypeError):
        return False


# ============================================================
# STATUS SERVER (for the Google Meet extension)
# ============================================================

class Status:
    """Shared state read by the Chrome extension over http://127.0.0.1."""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {"running": True, "calibrating": True, "calib_left_s": None, "face": False,
                     "hr": None, "hr_source": "-", "perclos": None, "zoning_score": None,
                     "zoning_threshold": None, "sim_hr": False, "alerts": []}
        self.next_id = 1
        self.requests = {"sim_hr": None, "test": [], "settings": None}

    def update(self, **kw):
        with self.lock:
            self.data.update(kw)

    def alert(self, kind, msg, t, demo=False):
        with self.lock:
            self.data["alerts"] = (self.data["alerts"] + [{
                "id": self.next_id, "kind": kind, "title": STYLE[kind][1], "msg": msg,
                "t": round(t, 1), "demo": demo}])[-20:]
            self.next_id += 1

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data, default=lambda o: None))


def start_status_server(status, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send({})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            with status.lock:
                if u.path == "/state":
                    if "sens" in q or "en" in q:
                        status.requests["settings"] = (q.get("sens", "normal"),
                                                       set(filter(None, q.get("en", "").split(","))))
                elif u.path == "/sim":
                    status.requests["sim_hr"] = q.get("on", "1") == "1"
                elif u.path == "/test" and q.get("kind") in STYLE:
                    status.requests["test"].append(q["kind"])
                elif u.path != "/":
                    return self._send({"error": "unknown path"}, 404)
            self._send(status.snapshot())

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    except OSError:
        raise SystemExit(f"Port {port} is already in use - is another live_monitor.py --meet running? "
                         f"Close it (or the other program using port {port}) and try again.")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Meet extension helper running on http://127.0.0.1:{port}  (local only)")
    return srv


def is_final_model(model):
    return model is not None and "windows_s" in model


def final_raw_features(fr, t_ms, windows, rp, span_ms=None):
    """Physical features at each window length ending at t_ms (train_final.py format)."""
    out = {}
    for w in windows:
        win = int(min(w * 1000, span_ms)) if span_ms else int(w * 1000)
        f = lf.window_features(fr, None, t_ms, win, None, rp)
        for c in lf.PHYSICAL_FEATURES:
            out[f"{c}@{w}"] = f.get(c, np.nan)
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--model", default="lapse_model.joblib")
    ap.add_argument("--port", default=cap.ESP32_PORT, help="ESP32 serial port ('' = none)")
    ap.add_argument("--calib", type=float, default=CALIB_S)
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-notify", action="store_true", help="no desktop notifications")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--fast", action="store_true", help="file replay: don't wait in real time")
    ap.add_argument("--meet", action="store_true",
                    help="helper for the Google Meet extension (no window/sound, status server)")
    ap.add_argument("--http-port", type=int, default=47813)
    args = ap.parse_args()
    status = None
    if args.meet:
        args.no_window = args.no_sound = args.no_notify = True
        if args.port == cap.ESP32_PORT and sys.platform == "darwin":
            args.port = ""                      # COM ports don't exist on a Mac
        status = Status()
        start_status_server(status, args.http_port)

    model = None
    if args.model and os.path.exists(args.model):
        import joblib
        model = joblib.load(args.model)
        if is_final_model(model):
            if model.get("behaviour"):
                print("Note: this model expects response-time features, which a live call doesn't "
                      "have. Use the face-only model (train_final.py without --behaviour).")
            print(f"Zoning-out model: final model ({len(model['features'])} features, "
                  f"windows {model['windows_s']} s)")
        else:
            print(f"Zoning-out model: {model['model_name']} on {model.get('feature_set', '?')}")
    else:
        print("No trained model found - using the simple staring rule for zoning out.")

    is_file = not args.source.isdigit()
    vid = cv2.VideoCapture(args.source if is_file else int(args.source))
    if not vid.isOpened():
        raise SystemExit("Cannot open video source " + args.source)
    if not is_file:
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    file_fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    file_ts = None
    if is_file:
        data_csv = os.path.splitext(args.source)[0] + "_data.csv"
        if os.path.exists(data_csv):
            file_ts = pd.read_csv(data_csv, usecols=["timestamp_ms"]).timestamp_ms.to_numpy()
            print("Using recording timestamps from", os.path.basename(data_csv))

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=cap.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    if args.port:
        cap.ESP32_PORT = args.port
        threading.Thread(target=cap.esp32_reader, daemon=True).start()

    rules, notifier = AlertRules(), Notifier(sound=not args.no_sound, desktop=not args.no_notify)
    rows = collections.deque()
    rgb_buf = collections.deque()
    rppg_rows = collections.deque(maxlen=60)
    log_path = f"alerts_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log = open(log_path, "w", newline="")
    logw = csv.writer(log)
    logw.writerow(["t_s", "alert", "message", "hr", "perclos", "zoning_score"])
    score_log = open(log_path.replace("alerts_", "scores_"), "w", newline="")
    scorew = csv.writer(score_log)
    scorew.writerow(["t_s", "zoning_score", "perclos", "hr", "hr_source", "calibrated"])

    base = None                 # calibration results
    hr_calib = []
    prev_expr = None
    sim_hr = False
    next_tick = 0.0
    zone_score = np.nan
    hr_now, hr_src = np.nan, "-"
    zone_ema = None
    t0 = time.monotonic()
    i = 0

    while True:
        ok, frame = vid.read()
        if not ok:
            break
        if file_ts is not None and i < len(file_ts):
            t_ms = int(file_ts[i])
        elif is_file:
            t_ms = int(i * 1000 / file_fps)
        else:
            t_ms = int((time.monotonic() - t0) * 1000)
        if is_file and not args.fast:
            time.sleep(max(0.0, t_ms / 1000 - (time.monotonic() - t0)))
        i += 1
        t = t_ms / 1000
        h, w = frame.shape[:2]
        cap.FRAME_W, cap.FRAME_H = w, h

        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
        f = cap.extract_face_features(res)
        row = {"timestamp_ms": t_ms, **f}

        if res.face_landmarks and res.face_blendshapes:
            bs, expr = blendshape_row(res.face_blendshapes[0])
            row.update(bs)
            row["expr_motion"] = float(np.abs(expr - prev_expr).sum()) if prev_expr is not None else np.nan
            prev_expr = expr
            r, g, b, _ = skin_rgb(frame, res.face_landmarks[0], w, h)
            if np.isfinite(r):
                rgb_buf.append((t_ms, r, g, b))
        else:
            prev_expr = None
        while rgb_buf and rgb_buf[0][0] < t_ms - 10000:
            rgb_buf.popleft()

        with cap.sensor_lock:
            fresh = cap.latest_hr_time and time.monotonic() - cap.latest_hr_time < cap.SENSOR_STALE_S
            hr_esp = cap.latest_hr if fresh else np.nan
        row["heart_rate"], row["spo2"] = hr_esp, np.nan
        rows.append(row)
        while rows and rows[0]["timestamp_ms"] < t_ms - BUFFER_S * 1000:
            rows.popleft()

        alerts = []
        if status is not None:
            with status.lock:
                req = status.requests
                if req["settings"]:
                    sens, en = req["settings"]
                    rules.p = dict(PRESETS.get(sens, PRESETS["normal"]))
                    rules.enabled = en or set(STYLE)
                    req["settings"] = None
                if req["sim_hr"] is not None:
                    sim_hr, req["sim_hr"] = req["sim_hr"], None
                tests, req["test"] = req["test"], []
            for kind in tests:
                status.alert(kind, DEMO_MSG[kind], t, demo=True)

        # ---------- every TICK_S: HR, zoning model, PERCLOS ----------
        if t >= next_tick:
            next_tick = t + TICK_S
            bpm, snr = live_rppg(rgb_buf)
            if np.isfinite(bpm):
                rppg_rows.append((t_ms, bpm, snr))
            good = [b for _, b, s in list(rppg_rows)[-5:] if s >= lf.RPPG_MIN_SNR]
            if np.isfinite(hr_esp):
                hr_now, hr_src = hr_esp, "ESP32"
            elif good:
                hr_now, hr_src = float(np.median(good)), "webcam"
            else:
                hr_now, hr_src = np.nan, "-"
            hr_used = hr_now + (SIM_HR_OFFSET if sim_hr else 0)

            if base is None:
                if np.isfinite(hr_now):
                    hr_calib.append(hr_now)
                if t >= args.calib:
                    base = calibrate(pd.DataFrame(list(rows)), hr_calib, rppg_rows, model)
                    print(f"Calibrated: open-eye EAR {base['ear_open']:.3f}, "
                          f"baseline HR {base['hr_base']:.0f} bpm")
            else:
                fr = lf.add_session_signals(pd.DataFrame(list(rows)), NO_EVENTS, 3000)
                rp = pd.DataFrame(list(rppg_rows), columns=["t_ms", "hr_rppg", "rppg_snr"])
                wf = lf.window_features(fr, None, t_ms, int(lf.WINDOW_S * 1000), None, rp)
                if wf.get("n_frames", 0) >= 50:
                    if is_final_model(model):
                        raw = final_raw_features(fr, t_ms, model["windows_s"], rp)
                        x = dict(raw)
                        x.update({k + "_pd": v - base["final_base"].get(k, np.nan) for k, v in raw.items()})
                        X = pd.DataFrame([x]).reindex(columns=model["features"]).astype(float)
                        X = X.fillna(pd.Series(model["medians"])).fillna(0)
                        p = float(model["model"].predict_proba(X)[0, 1])
                        a = 1 - 0.5 ** (1 / ZONE_SMOOTH_TICKS)
                        zone_ema = p if zone_ema is None else zone_ema + a * (p - zone_ema)
                        zone_score = zone_ema
                        zoning = zone_score >= model["thresholds"]["trial"]
                    elif model is not None:
                        x = zscore(wf, base)
                        X = pd.DataFrame([x]).reindex(columns=model["features"]).astype(float)
                        zone_score = float(model["model"].predict_proba(X)[0, 1])
                        zoning = zone_score >= model["threshold"]
                    else:
                        zoning = staring_rule(wf, base)
                        zone_score = float(zoning)
                else:
                    zoning = False
                alerts += rules.on_tick(t, zoning, hr_used, base["hr_base"])
            if status is not None:
                status.update(calibrating=base is None,
                              calib_left_s=None if base else max(0, round(args.calib - t)),
                              hr=round(hr_used, 1) if np.isfinite(hr_used) else None, hr_source=hr_src,
                              perclos=round(float(rules.perclos()), 3) if np.isfinite(rules.perclos()) else None,
                              zoning_score=round(zone_score, 3) if np.isfinite(zone_score) else None,
                              zoning_threshold=(model["thresholds"]["trial"] if is_final_model(model)
                                                else model["threshold"] if model else None),
                              sim_hr=sim_hr)
            scorew.writerow([round(t, 2), round(zone_score, 4) if np.isfinite(zone_score) else "",
                             round(rules.perclos(), 4) if np.isfinite(rules.perclos()) else "",
                             round(hr_used, 1) if np.isfinite(hr_used) else "", hr_src, int(base is not None)])

        # ---------- every frame: dozing off ----------
        if base is not None:
            closure = None
            if f["ear"] is not None:
                closure = float(np.clip((base["ear_open"] - f["ear"]) /
                                        (base["ear_open"] - base["ear_closed"]), 0, 1))
            pitch_dev = f["pitch"] - base["pitch"] if f["pitch"] is not None else None
            alerts += rules.on_frame(t, bool(f["face_present"]), closure, pitch_dev)

        if status is not None:
            status.update(face=bool(f["face_present"]))
        for kind, msg in alerts:
            if kind == "stress" and sim_hr:
                msg += " (simulated)"
            notifier.alert(t, kind, msg)
            if status is not None:
                status.alert(kind, msg, t)
            logw.writerow([round(t, 2), kind, msg, round(hr_now, 1) if np.isfinite(hr_now) else "",
                           round(rules.perclos(), 3), round(zone_score, 3)])
            log.flush()

        # ---------- display ----------
        if not args.no_window:
            disp = frame.copy()
            lines = []
            if base is None:
                lines.append(f"Calibrating {t:.0f}/{args.calib:.0f}s - look at the screen, blink normally")
            hr_txt = f"{hr_now + (SIM_HR_OFFSET if sim_hr else 0):.0f} bpm ({hr_src}{', SIM' if sim_hr else ''})" \
                if np.isfinite(hr_now) else "-"
            lines += [f"HR: {hr_txt}",
                      f"PERCLOS 1 min: {rules.perclos():.0%}" if base else "",
                      f"Zoning score: {zone_score:.2f}" if base and np.isfinite(zone_score) else "",
                      "" if f["face_present"] else "NO FACE"]
            for k, s in enumerate([l for l in lines if l]):
                cv2.putText(disp, s, (20, h - 30 - 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            notifier.draw(disp, t)
            cv2.imshow("ICPS live monitor (q = quit, h = simulate high HR)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                sim_hr = not sim_hr
                print("Simulated +35 bpm:", sim_hr)
        else:
            notifier.draw(frame, t)

    vid.release()
    landmarker.close()
    cv2.destroyAllWindows()
    log.close()
    score_log.close()
    cap.stop_threads.set()
    if status is not None:
        status.update(running=False)
    print("Alerts saved to", log_path)


def calibrate(df, hr_calib, rppg_rows, model):
    face = df.face_present == 1
    ear = df.ear[face].dropna().to_numpy(float)
    ear_open = float(np.percentile(ear, 50)) if len(ear) else 0.28
    ear_closed = float(np.percentile(ear, 1)) if len(ear) else 0.1
    if ear_open - ear_closed < 0.05:
        ear_closed = 0.5 * ear_open
    base = dict(ear_open=ear_open, ear_closed=ear_closed,
                pitch=float(np.nanmedian(df.pitch.astype(float))) if face.any() else 0.0,
                hr_base=float(np.median(hr_calib)) if hr_calib else np.nan)

    fr = lf.add_session_signals(df, NO_EVENTS, 3000)
    rp = pd.DataFrame(list(rppg_rows), columns=["t_ms", "hr_rppg", "rppg_snr"])
    sess = {"frames": fr, "audio": None, "rppg": rp}
    base["med"], base["scale"], _ = lf.session_baseline(sess, int(lf.WINDOW_S * 1000))
    gd = base["med"].get("gaze_dispersion", np.nan)
    base["gaze_dispersion_p25"] = gd * 0.6 if np.isfinite(gd) else np.nan
    if is_final_model(model):             # alert baseline = the calibration period (like the PVT-B rest)
        a, b = float(df.timestamp_ms.min()), float(df.timestamp_ms.max())
        feats = [final_raw_features(fr, t, model["windows_s"], rp, span_ms=b - a)
                 for t in np.arange(a + 10000, b + 1, 2000)]
        base["final_base"] = pd.DataFrame(feats).median().to_dict()
    return base


def zscore(f, base):
    med, scale = base["med"], base["scale"]
    out = dict(f)
    for k, v in f.items():
        if k in lf.Z_SKIP or k not in med.index:
            continue
        s = scale.get(k, np.nan)
        out[k + "_z"] = (v - med[k]) / s if pd.notna(s) else np.nan
    return out


if __name__ == "__main__":
    main()
