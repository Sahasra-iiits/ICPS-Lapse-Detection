"""
Window-level features + labels for cognitive-lapse detection.

Reads the files written by icps_capture.py:
    <session>_data.csv    per-frame signals
    <session>_events.csv  probes / self-reports
    <session>_meta.json   participant, audio offset
    <session>.wav         mic audio (optional)
and, if you ran extract_video_features.py:
    <session>_video.csv   per-frame blendshapes, facial motion, skin colour
    <session>_rppg.csv    webcam heart rate

Each labelled example = the WINDOW_S seconds just before a probe beep,
a PVT-B stimulus (pvt_b.py), or a self-report. Label 1 = lapse.

Every feature belongs to a group (see FEATURE_GROUP) so the trainer can run
an ablation: which groups actually help?

  eye_closure  EAR-based: PERCLOS, blinks, long closures (drowsy lapses)
  gaze         iris-in-eye gaze movement
  head         head pose and movement
  eyes_open    signs of zoning out with eyes OPEN: staring (long blink-free
               stretches, fixed gaze), head "freezing", blank face, looking away
  blendshape   MediaPipe blendshape scores (blink, squint, look down, jaw/yawn)
  rppg         webcam heart rate
  contact      ESP32 finger sensor (HR / SpO2)
  audio        own-mic speech activity

Changes vs. the mid-eval version:
  - Eye closure uses a PER-SESSION threshold (from that person's own open/closed
    EAR) instead of a fixed EAR < 0.20. PERCLOS uses the standard >=80% closure.
  - Gaze and head pose are median-smoothed before taking velocities, so 30 fps
    landmark jitter no longer looks like saccades / nods.
  - Frames during the PREVIOUS probe's beep + keypress are removed from the
    window, so the response movement itself doesn't leak into features.
  - Probe RTs are corrected by `probe_output_latency_ms` from _meta.json (if set).
  - New groups: eyes_open, blendshape, rppg.
"""

import glob
import json
import os
import wave

import numpy as np
import pandas as pd


# ---------------- Tunables ----------------
WINDOW_S = 10.0          # look-back window per example
BASELINE_STEP_S = 2.0    # stride for the per-session baseline windows

# A probe is a lapse if missed, or if RT > max(LAPSE_RT_MIN_MS, LAPSE_RT_FACTOR x
# that session's median RT). Relative, because baseline RT varies a lot between
# people and setups (your pilot median was ~1.3 s, not the textbook 0.4 s —
# mostly audio-output latency; see PROBE_OUTPUT_LATENCY_MS in icps_capture.py).
LAPSE_RT_FACTOR = 1.5
LAPSE_RT_MIN_MS = 1000
REPORT_GAP_MS = 1000     # self-report window ends this long before the keypress
PVT_LAPSE_MS = 355       # PVT-B sessions (pvt_b.py): standard brief-PVT lapse threshold

# Remove frames from [probe onset, response + this] of earlier probes/reports
CLEAN_RESPONSES = True
RESPONSE_PAD_MS = 1000
PROBE_TIMEOUT_MS = 3000

# Eye closure (per session): closure = (EAR_open - EAR) / (EAR_open - EAR_closed)
BLINK_CLOSURE = 0.5      # >= 50 % closed counts as "eye closed" for blinks
PERCLOS_CLOSURE = 0.8    # standard PERCLOS: >= 80 % closed
BS_BLINK_THR = 0.5       # blendshape eyeBlink score for "closed"
BLINK_MIN_MS = 50
BLINK_MAX_MS = 500       # closures longer than this = "long closure" (microsleep-ish)

SMOOTH_FRAMES = 5        # rolling median (~170 ms at 30 fps) for gaze / head
SACCADE_SPEED = 1.5      # gaze speed (eye-widths / s) above which = saccade
NOD_SPEED_DEG = 40.0     # |pitch velocity| above this (deg / s) = nod / head drop
HEAD_STILL_DEG_S = 3.0   # angular speed below this = head "frozen"
OFF_SCREEN_YAW = 20.0    # deg from the session's typical yaw
OFF_SCREEN_PITCH = 15.0
YAWN_JAW = 0.5           # jawOpen score
YAWN_MIN_MS = 1500       # speech opens the jaw briefly; yawns last longer

RPPG_MIN_SNR = -3.0      # dB; rPPG estimates below this are ignored

AUDIO_HOP_MS = 50


# Compact set of physically meaningful eye/face features (durations in ms,
# fractions). On DROZY these beat the full feature set: closure DURATIONS mean
# the same for everyone, while raw EAR levels depend on each person's eye shape.
PHYSICAL_FEATURES = [
    "max_closure_ms", "long_closures", "blink_dur_mean", "blink_dur_max", "perclos",
    "closure_mean", "bs_blink_mean", "perclos_bs", "expr_motion_mean", "head_speed_mean",
]

# Group of every feature (filled while features are computed)
FEATURE_GROUP = {}

# The mid-eval feature set, for the ablation baseline
ORIGINAL_FEATURES = {
    "face_present_frac", "ear_mean", "ear_std", "ear_p10", "perclos",
    "blink_rate", "blink_dur_mean", "long_closures", "max_closure_ms",
    "gaze_h_std", "gaze_v_std", "gaze_speed_mean", "gaze_speed_p90",
    "saccade_rate", "longest_fixation_ms",
    "yaw_mean", "yaw_std", "pitch_mean", "pitch_std", "roll_mean", "roll_std",
    "nods", "head_speed_mean",
    "hr_mean", "hr_std", "hr_slope", "spo2_mean",
    "audio_rms_mean", "speech_frac",
}


# ============================================================
# LOADING
# ============================================================

def list_sessions(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*_data.csv")))
    return [os.path.basename(p)[: -len("_data.csv")] for p in paths]


def load_session(data_dir, session_id):
    base = os.path.join(data_dir, session_id)

    frames = pd.read_csv(base + "_data.csv")
    frames = frames.sort_values("timestamp_ms").reset_index(drop=True)

    if os.path.exists(base + "_video.csv"):
        video = pd.read_csv(base + "_video.csv")
        frames = frames.merge(video, on="timestamp_ms", how="left")

    rppg = pd.read_csv(base + "_rppg.csv") if os.path.exists(base + "_rppg.csv") else None

    ev_path = base + "_events.csv"
    events = pd.read_csv(ev_path) if os.path.exists(ev_path) else \
        pd.DataFrame(columns=["timestamp_ms", "event_type", "rt_ms"])

    meta = {}
    if os.path.exists(base + "_meta.json"):
        with open(base + "_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)

    latency = float(meta.get("probe_output_latency_ms", 0) or 0)
    if latency and len(events):
        is_hit = (events.event_type == "probe") & events.rt_ms.notna()
        events.loc[is_hit, "rt_ms"] = (events.loc[is_hit, "rt_ms"] - latency).clip(lower=0)

    audio = None
    if os.path.exists(base + ".wav"):
        audio = audio_envelope(base + ".wav", meta.get("audio_start_ms", 0))

    timeout = float(meta.get("probe_timeout_s", PROBE_TIMEOUT_MS / 1000)) * 1000
    frames = add_session_signals(frames, events, timeout)

    return {
        "session_id": session_id,
        "participant": meta.get("participant", session_id),
        "frames": frames,
        "events": events,
        "audio": audio,
        "rppg": rppg,
        "augmented": bool(meta.get("augmented_from")),
        "dataset": meta.get("dataset", "own"),
        "pvt_lapse_ms": float(meta.get("lapse_threshold_ms", PVT_LAPSE_MS)),
    }


def audio_envelope(wav_path, offset_ms=0, hop_ms=AUDIO_HOP_MS):
    """RMS envelope on the recording's timeline (DataFrame: t_ms, rms)."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    x = raw.reshape(-1, n_ch).mean(axis=1).astype(np.float32) / 32768.0
    hop = int(sr * hop_ms / 1000)
    n = len(x) // hop
    if n == 0:
        return None
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    t = offset_ms + np.arange(n) * hop_ms
    return pd.DataFrame({"t_ms": t, "rms": rms})


# ============================================================
# SESSION-LEVEL PREPROCESSING
# ============================================================

def _rolling_median(s):
    return s.rolling(SMOOTH_FRAMES, center=True, min_periods=1).median()


def add_session_signals(frames, events, timeout_ms):
    """Per-session calibrated columns used by window_features."""
    fr = frames.copy()
    face = fr.face_present.fillna(0).to_numpy() == 1
    ear = fr.ear.to_numpy(float)

    # --- per-person eye closure ---
    ear_face = ear[face & ~np.isnan(ear)]
    if len(ear_face) > 100:
        ear_open = np.percentile(ear_face, 50)     # eyes are open most of the time
        ear_closed = np.percentile(ear_face, 1)
        if ear_open - ear_closed < 0.05:            # too few blinks to estimate
            ear_closed = 0.5 * ear_open
        closure = np.clip((ear_open - ear) / (ear_open - ear_closed), 0, 1)
    else:
        closure = np.where(ear < 0.20, 1.0, 0.0)   # fallback: old fixed threshold
    fr["closure"] = closure
    fr["closed"] = (closure >= BLINK_CLOSURE) & face
    fr["closed80"] = (closure >= PERCLOS_CLOSURE) & face

    if "bs_blink" in fr:
        fr["closed_bs"] = (fr.bs_blink.to_numpy(float) >= BS_BLINK_THR) & face

    # --- smoothed gaze / head (open-eye, face-present frames only) ---
    open_ok = face & ~fr.closed.to_numpy()
    for c in ("gaze_h", "gaze_v"):
        fr[c + "_s"] = _rolling_median(fr[c].where(open_ok))
    for c in ("yaw", "pitch", "roll"):
        if c in fr:
            fr[c + "_s"] = _rolling_median(fr[c].where(face))

    # session "typical" head pose → off-screen = big deviation from it
    fr.attrs["yaw_med"] = float(np.nanmedian(fr.yaw)) if "yaw" in fr else np.nan
    fr.attrs["pitch_med"] = float(np.nanmedian(fr.pitch)) if "pitch" in fr else np.nan
    if "expr_motion" in fr:
        fr.attrs["expr_p25"] = float(np.nanpercentile(fr.expr_motion, 25)) \
            if fr.expr_motion.notna().any() else np.nan

    # --- frames contaminated by an earlier probe's beep + keypress ---
    blocked = np.zeros(len(fr), bool)
    if CLEAN_RESPONSES and len(events):
        t = fr.timestamp_ms.to_numpy()
        for _, e in events.iterrows():
            if e.event_type == "probe":
                rt = e.rt_ms if pd.notna(e.rt_ms) else timeout_ms
                a, b = e.timestamp_ms, e.timestamp_ms + rt + RESPONSE_PAD_MS
            elif e.event_type in ("self_report", "false_alarm"):
                a, b = e.timestamp_ms - 300, e.timestamp_ms + RESPONSE_PAD_MS
            else:
                continue
            blocked |= (t >= a) & (t <= b)
    fr["blocked"] = blocked
    return fr


# ============================================================
# HELPERS
# ============================================================

def _runs(mask):
    """(start_idx, end_idx_exclusive) for each run of True."""
    m = np.concatenate([[False], np.asarray(mask).astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _nanstat(fn, x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(fn(x)) if len(x) else np.nan


def _slope_per_min(t_ms, y):
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(t_ms[ok]) == 0:
        return np.nan
    return float(np.polyfit(t_ms[ok] / 60000.0, y[ok], 1)[0])


def _longest_run_ms(mask, t, frame_ms):
    best = 0.0
    for s, e in _runs(mask):
        best = max(best, t[e - 1] - t[s] + frame_ms)
    return best


class _F(dict):
    """Feature dict that records each feature's group."""
    def put(self, group, name, value):
        FEATURE_GROUP[name] = group
        self[name] = value


def _closure_events(closed, t, frame_ms):
    """Blink durations, #long closures, max closure, blink start times."""
    durs, starts, n_long, mx = [], [], 0, 0.0
    for s, e in _runs(closed):
        dur = t[e - 1] - t[s] + frame_ms
        mx = max(mx, dur)
        if BLINK_MIN_MS <= dur <= BLINK_MAX_MS:
            durs.append(dur)
            starts.append(t[s])
        elif dur > BLINK_MAX_MS:
            n_long += 1
            starts.append(t[s])
    return durs, starts, n_long, mx


# ============================================================
# FEATURES FOR ONE WINDOW
# ============================================================

def window_features(frames, audio, t_end_ms, win_ms, noise_floor=None, rppg=None):
    """Features over frames with t in (t_end - win, t_end], minus blocked frames."""
    t0 = t_end_ms - win_ms
    w = frames[(frames.timestamp_ms > t0) & (frames.timestamp_ms <= t_end_ms)]
    if "blocked" in w:
        w = w[~w.blocked.to_numpy()]

    f = _F()
    n = len(w)
    f["n_frames"] = n
    if n < 5:
        return f

    t = w.timestamp_ms.to_numpy(float)
    frame_ms = float(np.median(np.diff(t))) if n > 1 else 33.0
    minutes = n * frame_ms / 60000.0          # effective (unblocked) duration
    face = w.face_present.to_numpy(float) == 1
    dt = np.diff(t) / 1000.0
    dt[dt <= 0] = np.nan

    # ---------------- eye_closure ----------------
    G = "eye_closure"
    ear = w.ear.to_numpy(float)
    closed = w.closed.to_numpy(bool)
    f.put(G, "face_present_frac", face.mean())
    f.put(G, "ear_mean", _nanstat(np.mean, ear))
    f.put(G, "ear_std", _nanstat(np.std, ear))
    f.put(G, "ear_p10", _nanstat(lambda x: np.percentile(x, 10), ear))
    f.put(G, "closure_mean", _nanstat(np.mean, w.closure.to_numpy(float)[face]))
    f.put(G, "perclos", w.closed80.to_numpy(bool)[face].mean() if face.any() else np.nan)

    durs, starts, n_long, mx = _closure_events(closed, t, frame_ms)
    f.put(G, "blink_rate", len(durs) / minutes)
    f.put(G, "blink_dur_mean", np.mean(durs) if durs else np.nan)
    f.put(G, "blink_dur_max", np.max(durs) if durs else np.nan)
    f.put(G, "long_closures", n_long)
    f.put(G, "max_closure_ms", mx)

    # ---------------- gaze ----------------
    G = "gaze"
    gh = w.gaze_h_s.to_numpy(float)
    gv = w.gaze_v_s.to_numpy(float)
    f.put(G, "gaze_h_std", _nanstat(np.std, gh))
    f.put(G, "gaze_v_std", _nanstat(np.std, gv))
    speed = np.hypot(np.diff(gh), np.diff(gv)) / dt
    valid = ~np.isnan(speed)
    if valid.sum() > 3:
        sp = speed[valid]
        f.put(G, "gaze_speed_mean", float(np.mean(sp)))
        f.put(G, "gaze_speed_p90", float(np.percentile(sp, 90)))
        f.put(G, "saccade_rate", float((sp > SACCADE_SPEED).sum() / minutes))
        still = valid & (speed < SACCADE_SPEED)
        f.put(G, "longest_fixation_ms", _longest_run_ms(still, t[1:], frame_ms))
        f.put("eyes_open", "fixation_frac", float(still.sum() / valid.sum()))
    else:
        for k in ("gaze_speed_mean", "gaze_speed_p90", "saccade_rate", "longest_fixation_ms"):
            f.put(G, k, np.nan)
        f.put("eyes_open", "fixation_frac", np.nan)

    # ---------------- head ----------------
    G = "head"
    for col in ("yaw", "pitch", "roll"):
        v = w[col].to_numpy(float) if col in w else np.full(n, np.nan)
        f.put(G, f"{col}_mean", _nanstat(np.mean, v))
        f.put(G, f"{col}_std", _nanstat(np.std, v))
    if "pitch_s" in w:
        pv = np.diff(w.pitch_s.to_numpy(float)) / dt
        f.put(G, "nods", float(np.nansum(np.abs(pv) > NOD_SPEED_DEG)))
        f.put(G, "pitch_slope", _slope_per_min(t, w.pitch_s.to_numpy(float)))  # head sinking
    hx, hy = w.head_x.to_numpy(float), w.head_y.to_numpy(float)
    f.put(G, "head_speed_mean", _nanstat(np.mean, np.hypot(np.diff(hx), np.diff(hy)) / dt))

    # ---------------- eyes_open (zoning out with eyes open) ----------------
    G = "eyes_open"
    # staring: longest stretch without any blink / closure
    edges = [t[0]] + sorted(starts) + [t[-1] + frame_ms]
    f.put(G, "longest_interblink_ms", float(np.max(np.diff(edges))))
    # how spread out gaze is (low = fixed stare)
    f.put(G, "gaze_dispersion",
          float(np.sqrt(np.nanvar(gh) + np.nanvar(gv))) if np.isfinite(gh).sum() > 3 else np.nan)
    # head "freezing": angular speed of smoothed pose
    if all(c + "_s" in w for c in ("yaw", "pitch", "roll")):
        ang = np.sqrt(sum(np.diff(w[c + "_s"].to_numpy(float)) ** 2
                          for c in ("yaw", "pitch", "roll"))) / dt
        ok = ~np.isnan(ang)
        f.put(G, "head_ang_speed_mean", _nanstat(np.mean, ang))
        f.put(G, "head_still_frac", float((ang[ok] < HEAD_STILL_DEG_S).mean()) if ok.any() else np.nan)
        f.put(G, "head_longest_still_ms",
              _longest_run_ms(ok & (ang < HEAD_STILL_DEG_S), t[1:], frame_ms))
        # looking away from the screen (relative to this person's usual pose)
        off = ((np.abs(w.yaw.to_numpy(float) - frames.attrs.get("yaw_med", np.nan)) > OFF_SCREEN_YAW) |
               (np.abs(w.pitch.to_numpy(float) - frames.attrs.get("pitch_med", np.nan)) > OFF_SCREEN_PITCH))
        f.put(G, "off_screen_frac", float(off[face].mean()) if face.any() else np.nan)
    # blank face: little facial (non-eye) movement
    if "expr_motion" in w:
        em = w.expr_motion.to_numpy(float)
        f.put(G, "expr_motion_mean", _nanstat(np.mean, em))
        p25 = frames.attrs.get("expr_p25", np.nan)
        okm = ~np.isnan(em)
        f.put(G, "expr_still_frac", float((em[okm] < p25).mean()) if okm.any() and np.isfinite(p25) else np.nan)

    # ---------------- blendshape ----------------
    G = "blendshape"
    if "bs_blink" in w:
        cb = w.closed_bs.to_numpy(bool)
        f.put(G, "perclos_bs", cb[face].mean() if face.any() else np.nan)
        d_bs, _, long_bs, _ = _closure_events(cb, t, frame_ms)
        f.put(G, "blink_rate_bs", len(d_bs) / minutes)
        f.put(G, "long_closures_bs", long_bs)
        for c in ("bs_blink", "bs_squint", "bs_wide", "bs_look_down", "bs_look_up",
                  "bs_brow_down", "bs_brow_inner_up", "bs_jaw_open"):
            f.put(G, c + "_mean", _nanstat(np.mean, w[c].to_numpy(float)))
        f.put(G, "bs_look_h_std", _nanstat(np.std, w.bs_look_h.to_numpy(float)))
        jaw = w.bs_jaw_open.to_numpy(float) > YAWN_JAW
        f.put(G, "yawns", sum(1 for s, e in _runs(jaw) if t[e - 1] - t[s] + frame_ms >= YAWN_MIN_MS))

    # ---------------- rppg ----------------
    G = "rppg"
    if rppg is not None and len(rppg):
        r = rppg[(rppg.t_ms > t0) & (rppg.t_ms <= t_end_ms)]
        r = r[r.rppg_snr >= RPPG_MIN_SNR]
        f.put(G, "hr_rppg_mean", _nanstat(np.mean, r.hr_rppg))
        f.put(G, "hr_rppg_slope", _slope_per_min(r.t_ms.to_numpy(float), r.hr_rppg.to_numpy(float)))
        f.put(G, "rppg_snr_mean", _nanstat(np.mean, r.rppg_snr))

    # ---------------- contact (ESP32) ----------------
    G = "contact"
    hr = w.heart_rate.to_numpy(float)
    f.put(G, "hr_mean", _nanstat(np.mean, hr))
    f.put(G, "hr_std", _nanstat(np.std, hr))
    f.put(G, "hr_slope", _slope_per_min(t, hr))
    f.put(G, "spo2_mean", _nanstat(np.mean, w.spo2.to_numpy(float)))

    # ---------------- audio ----------------
    G = "audio"
    if audio is not None:
        a = audio[(audio.t_ms > t0) & (audio.t_ms <= t_end_ms)].rms.to_numpy()
        if len(a):
            f.put(G, "audio_rms_mean", float(a.mean()))
            if noise_floor is not None:
                speech = a > 3 * noise_floor
                f.put(G, "speech_frac", float(speech.mean()))
                f.put(G, "longest_silence_ms", float(
                    max((e - s for s, e in _runs(~speech)), default=0) * AUDIO_HOP_MS))

    return dict(f)


# ============================================================
# PER-SESSION BASELINE (robust z-scores)
# ============================================================

def session_baseline(sess, win_ms):
    frames, audio = sess["frames"], sess["audio"]
    noise = float(np.percentile(audio.rms, 10)) if audio is not None else None

    t_min = frames.timestamp_ms.min() + win_ms
    t_max = frames.timestamp_ms.max()
    rows = [
        window_features(frames, audio, t, win_ms, noise, sess.get("rppg"))
        for t in np.arange(t_min, t_max + 1, BASELINE_STEP_S * 1000)
    ]
    base = pd.DataFrame(rows)

    med = base.median(numeric_only=True)
    iqr = base.quantile(0.75, numeric_only=True) - base.quantile(0.25, numeric_only=True)
    scale = (iqr / 1.349).where(lambda s: s > 1e-9, other=np.nan)
    return med, scale, noise


# ============================================================
# LABELS
# ============================================================

def lapse_rt_threshold(events):
    hits = events[(events.event_type == "probe") & events.rt_ms.notna()].rt_ms
    med = float(hits.median()) if len(hits) >= 3 else LAPSE_RT_MIN_MS
    return max(LAPSE_RT_MIN_MS, LAPSE_RT_FACTOR * med)


def label_points(events, pvt_lapse_ms=PVT_LAPSE_MS):
    """(t_end_ms, label, source, rt_ms) for every labelled example."""
    thr = lapse_rt_threshold(events)
    pts = []
    for _, e in events.iterrows():
        if e.event_type == "probe":
            rt = e.rt_ms
            miss = pd.isna(rt)
            lapse = int(miss or rt > thr)
            pts.append((int(e.timestamp_ms), lapse, "probe", rt))
        elif e.event_type == "pvt":
            rt = e.rt_ms
            lapse = int(pd.isna(rt) or rt >= pvt_lapse_ms)
            pts.append((int(e.timestamp_ms), lapse, "pvt", rt))
        elif e.event_type == "self_report":
            pts.append((int(e.timestamp_ms) - REPORT_GAP_MS, 1, "self_report", np.nan))
    return pts


# ============================================================
# DATASET
# ============================================================

Z_SKIP = {"n_frames"}


def featurize(sess, t_end_ms, med, scale, noise, win_ms):
    f = window_features(sess["frames"], sess["audio"], t_end_ms, win_ms, noise, sess.get("rppg"))
    for k in list(f):
        if k in Z_SKIP or k not in med.index:
            continue
        s = scale.get(k, np.nan)
        f[k + "_z"] = (f[k] - med[k]) / s if pd.notna(s) else np.nan
    return f


def build_dataset(data_dir, win_s=WINDOW_S, min_frames=50):
    win_ms = int(win_s * 1000)
    rows = []

    for sid in list_sessions(data_dir):
        sess = load_session(data_dir, sid)
        pts = label_points(sess["events"], sess["pvt_lapse_ms"])
        if not pts:
            continue

        med, scale, noise = session_baseline(sess, win_ms)
        t_first = sess["frames"].timestamp_ms.min()

        for t_end, y, src, rt in pts:
            if t_end - win_ms < t_first:
                continue
            f = featurize(sess, t_end, med, scale, noise, win_ms)
            if f.get("n_frames", 0) < min_frames:
                continue
            f.update(session_id=sid, participant=sess["participant"],
                     t_end_ms=t_end, label=y, source=src, rt_ms=rt,
                     augmented=int(sess["augmented"]), dataset=sess["dataset"])
            rows.append(f)

    return pd.DataFrame(rows)


META_COLS = ["session_id", "participant", "t_end_ms", "label", "source",
             "rt_ms", "n_frames", "augmented", "dataset"]


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def base_name(col):
    return col[:-2] if col.endswith("_z") else col


def _register_groups():
    """Fill FEATURE_GROUP by computing features once on a dummy session
    (needed when a saved dataset CSV is loaded without build_dataset)."""
    n = 400
    rng = np.random.default_rng(0)
    t = np.arange(n) * 33
    fr = pd.DataFrame({
        "timestamp_ms": t, "face_present": 1, "ear": 0.3 + 0.01 * rng.standard_normal(n),
        "gaze_h": rng.random(n), "gaze_v": rng.random(n), "yaw": rng.random(n),
        "pitch": rng.random(n), "roll": rng.random(n), "head_x": rng.random(n),
        "head_y": rng.random(n), "heart_rate": 70.0, "spo2": 98.0, "expr_motion": rng.random(n),
        **{c: rng.random(n) for c in ("bs_blink", "bs_squint", "bs_wide", "bs_look_down",
                                       "bs_look_up", "bs_brow_down", "bs_brow_inner_up",
                                       "bs_jaw_open", "bs_look_h")}})
    fr = add_session_signals(fr, pd.DataFrame(columns=["timestamp_ms", "event_type", "rt_ms"]), 3000)
    audio = pd.DataFrame({"t_ms": t, "rms": rng.random(n)})
    rp = pd.DataFrame({"t_ms": t[::30], "hr_rppg": 70.0, "rppg_snr": 1.0})
    window_features(fr, audio, t[-1], 10000, 0.1, rp)


def group_of(col):
    if not FEATURE_GROUP:
        _register_groups()
    return FEATURE_GROUP.get(base_name(col), "other")
