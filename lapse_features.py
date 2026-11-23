"""
Window-level features + labels for cognitive-lapse detection.

Reads the files written by icps_recorder.py:
    <session>_data.csv    per-frame signals
    <session>_events.csv  probes / self-reports
    <session>_meta.json   participant, audio offset
    <session>.wav         mic audio (optional)

Each labelled example = the WINDOW_S seconds just before a probe beep
(or just before a self-report). Label 1 = lapse.
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
# people and setups (your pilot median was ~1.3 s, not the textbook 0.4 s).
LAPSE_RT_FACTOR = 1.5
LAPSE_RT_MIN_MS = 1000
REPORT_GAP_MS = 1000     # self-report window ends this long before the keypress

BLINK_MIN_MS = 50
BLINK_MAX_MS = 500       # closures longer than this = "long closure" (microsleep-ish)

SACCADE_SPEED = 1.5      # gaze speed (eye-widths / s) above which = saccade
NOD_SPEED_DEG = 40.0     # |pitch velocity| above this (deg / s) = nod / head drop

AUDIO_HOP_MS = 50


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

    ev_path = base + "_events.csv"
    events = pd.read_csv(ev_path) if os.path.exists(ev_path) else \
        pd.DataFrame(columns=["timestamp_ms", "event_type", "rt_ms"])

    meta = {}
    if os.path.exists(base + "_meta.json"):
        with open(base + "_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)

    audio = None
    if os.path.exists(base + ".wav"):
        audio = audio_envelope(base + ".wav", meta.get("audio_start_ms", 0))

    return {
        "session_id": session_id,
        "participant": meta.get("participant", session_id),
        "frames": frames,
        "events": events,
        "audio": audio,
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
# HELPERS
# ============================================================

def _runs(mask):
    """(start_idx, end_idx_exclusive) for each run of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _nanstat(fn, x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(fn(x)) if len(x) else np.nan


def _slope_per_min(t_ms, y):
    ok = ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(t_ms[ok]) == 0:
        return np.nan
    return float(np.polyfit(t_ms[ok] / 60000.0, y[ok], 1)[0])


# ============================================================
# FEATURES FOR ONE WINDOW
# ============================================================

def window_features(frames, audio, t_end_ms, win_ms, noise_floor=None):
    """Features over frames with t in (t_end - win, t_end]."""
    t0 = t_end_ms - win_ms
    w = frames[(frames.timestamp_ms > t0) & (frames.timestamp_ms <= t_end_ms)]

    f = {}
    n = len(w)
    f["n_frames"] = n
    if n < 5:
        return f

    t = w.timestamp_ms.to_numpy(float)
    face = w.face_present.to_numpy(float) == 1
    f["face_present_frac"] = face.mean()

    # ---- Eyes: EAR / closures / blinks ----
    ear = w.ear.to_numpy(float)
    closed = (w.eye_closed.to_numpy(float) == 1) & face

    f["ear_mean"] = _nanstat(np.mean, ear)
    f["ear_std"] = _nanstat(np.std, ear)
    f["ear_p10"] = _nanstat(lambda x: np.percentile(x, 10), ear)
    f["perclos"] = closed[face].mean() if face.any() else np.nan

    blink_durs, long_closures, max_closure = [], 0, 0.0
    for s, e in _runs(closed):
        dur = t[e - 1] - t[s] + (t[1] - t[0] if n > 1 else 0)
        max_closure = max(max_closure, dur)
        if BLINK_MIN_MS <= dur <= BLINK_MAX_MS:
            blink_durs.append(dur)
        elif dur > BLINK_MAX_MS:
            long_closures += 1

    minutes = win_ms / 60000.0
    f["blink_rate"] = len(blink_durs) / minutes
    f["blink_dur_mean"] = np.mean(blink_durs) if blink_durs else np.nan
    f["long_closures"] = long_closures
    f["max_closure_ms"] = max_closure

    # ---- Gaze (relative to eye corners) ----
    gh = w.gaze_h.to_numpy(float)
    gv = w.gaze_v.to_numpy(float)
    open_ok = face & ~closed

    f["gaze_h_std"] = _nanstat(np.std, gh[open_ok])
    f["gaze_v_std"] = _nanstat(np.std, gv[open_ok])

    dt = np.diff(t) / 1000.0
    dt[dt <= 0] = np.nan
    speed = np.hypot(np.diff(gh), np.diff(gv)) / dt
    valid = open_ok[1:] & open_ok[:-1] & ~np.isnan(speed)

    if valid.sum() > 3:
        sp = speed[valid]
        f["gaze_speed_mean"] = float(np.mean(sp))
        f["gaze_speed_p90"] = float(np.percentile(sp, 90))
        f["saccade_rate"] = float((sp > SACCADE_SPEED).sum() / minutes)

        still = valid & (speed < SACCADE_SPEED)
        longest = 0.0
        for s, e in _runs(still):
            longest = max(longest, t[e] - t[s])
        f["longest_fixation_ms"] = longest            # "staring" proxy
    else:
        for k in ("gaze_speed_mean", "gaze_speed_p90", "saccade_rate",
                  "longest_fixation_ms"):
            f[k] = np.nan

    # ---- Head ----
    for col in ("yaw", "pitch", "roll"):
        v = w[col].to_numpy(float) if col in w else np.full(n, np.nan)
        f[f"{col}_mean"] = _nanstat(np.mean, v)
        f[f"{col}_std"] = _nanstat(np.std, v)

    if "pitch" in w:
        pv = np.diff(w.pitch.to_numpy(float)) / dt
        f["nods"] = float(np.nansum(np.abs(pv) > NOD_SPEED_DEG))
    hx, hy = w.head_x.to_numpy(float), w.head_y.to_numpy(float)
    hspeed = np.hypot(np.diff(hx), np.diff(hy)) / dt
    f["head_speed_mean"] = _nanstat(np.mean, hspeed)

    # ---- Physiology ----
    hr = w.heart_rate.to_numpy(float)
    f["hr_mean"] = _nanstat(np.mean, hr)
    f["hr_std"] = _nanstat(np.std, hr)
    f["hr_slope"] = _slope_per_min(t, hr)
    f["spo2_mean"] = _nanstat(np.mean, w.spo2.to_numpy(float))

    # ---- Audio (talking vs. silent) ----
    if audio is not None:
        a = audio[(audio.t_ms > t0) & (audio.t_ms <= t_end_ms)].rms.to_numpy()
        if len(a):
            f["audio_rms_mean"] = float(a.mean())
            if noise_floor is not None:
                f["speech_frac"] = float((a > 3 * noise_floor).mean())

    return f


# ============================================================
# PER-SESSION BASELINE (robust z-scores)
# ============================================================

def session_baseline(sess, win_ms):
    frames, audio = sess["frames"], sess["audio"]
    noise = float(np.percentile(audio.rms, 10)) if audio is not None else None

    t_min = frames.timestamp_ms.min() + win_ms
    t_max = frames.timestamp_ms.max()
    rows = [
        window_features(frames, audio, t, win_ms, noise)
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


def label_points(events):
    """(t_end_ms, label, source, rt_ms) for every labelled example."""
    thr = lapse_rt_threshold(events)
    pts = []
    for _, e in events.iterrows():
        if e.event_type == "probe":
            rt = e.rt_ms
            miss = pd.isna(rt)
            lapse = int(miss or rt > thr)
            pts.append((int(e.timestamp_ms), lapse, "probe", rt))
        elif e.event_type == "self_report":
            pts.append((int(e.timestamp_ms) - REPORT_GAP_MS, 1, "self_report", np.nan))
    return pts


# ============================================================
# DATASET
# ============================================================

Z_SKIP = {"n_frames"}


def featurize(sess, t_end_ms, med, scale, noise, win_ms):
    f = window_features(sess["frames"], sess["audio"], t_end_ms, win_ms, noise)
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
        pts = label_points(sess["events"])
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
                     t_end_ms=t_end, label=y, source=src, rt_ms=rt)
            rows.append(f)

    return pd.DataFrame(rows)


META_COLS = ["session_id", "participant", "t_end_ms", "label", "source",
             "rt_ms", "n_frames"]


def feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]