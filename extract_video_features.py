"""
Offline video pass: extra computer-vision features from recordings you
already have. No new data collection needed.

    python extract_video_features.py --data "C:\\Users\\sahas\\ICPS\\recordings"

For every <session>.mp4 that has a matching <session>_data.csv it writes:

    <session>_video.csv   per-frame MediaPipe blendshapes (eye blink / look
                          direction / squint / jaw open ...) + facial motion
                          energy + skin RGB means for rPPG
    <session>_rppg.csv    webcam heart rate (POS algorithm), one estimate per
                          second over a sliding window, with a quality (SNR) score

Frame i of the .mp4 is row i of _data.csv (the recorder writes both in the
same loop iteration), so timestamps are taken from _data.csv and everything
lines up with the probe events.

rPPG = remote photoplethysmography: the tiny periodic colour change of facial
skin with each heartbeat. POS: Wang et al., "Algorithmic Principles of Remote
PPG", IEEE TBME 2017. Good for average HR; NOT reliable for HRV at 30 fps.
"""

import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

MODEL_PATH = "face_landmarker.task"

# ---- rPPG settings ----
RPPG_FS = 30.0             # resample the skin-colour trace to this rate
POS_WIN_S = 1.6            # POS projection window (paper default)
HR_WIN_S = 10.0            # window for each HR estimate
HR_STEP_S = 1.0
HR_BAND_HZ = (0.7, 3.0)    # 42 - 180 bpm
MAX_GAP_S = 0.5            # interpolate face-tracking gaps up to this long

# Skin regions (MediaPipe face-mesh indices). Convex hulls, so order is irrelevant.
FOREHEAD = [10, 338, 297, 332, 109, 67, 103, 151, 9, 107, 336, 108, 337, 69, 299]
LEFT_CHEEK = [116, 117, 118, 119, 100, 126, 142, 203, 206, 207, 187, 147, 123, 50, 101]
RIGHT_CHEEK = [345, 346, 347, 348, 329, 355, 371, 423, 426, 427, 411, 376, 352, 280, 330]
SKIN_ROIS = (FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK)

# Blendshapes kept per frame (averaged over left/right where paired)
BS_PAIRS = {
    "bs_blink": ("eyeBlinkLeft", "eyeBlinkRight"),
    "bs_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "bs_wide": ("eyeWideLeft", "eyeWideRight"),
    "bs_look_down": ("eyeLookDownLeft", "eyeLookDownRight"),
    "bs_look_up": ("eyeLookUpLeft", "eyeLookUpRight"),
    "bs_brow_down": ("browDownLeft", "browDownRight"),
}
BS_SINGLE = {"bs_jaw_open": "jawOpen", "bs_brow_inner_up": "browInnerUp"}
# Horizontal gaze from blendshapes: looking toward the right side of the image
# = left eye looks out + right eye looks in (and vice versa).
# Expressivity: frame-to-frame change of all NON-eye blendshapes.
# A "blank" face during zoning out has low expressivity.
EYE_PREFIXES = ("eyeBlink", "eyeLook", "eyeSquint", "eyeWide", "_neutral")


def video_columns():
    return (["timestamp_ms", "v_face"] + list(BS_PAIRS) + list(BS_SINGLE)
            + ["bs_look_h", "bs_blink_l", "bs_blink_r", "expr_motion",
               "skin_r", "skin_g", "skin_b", "skin_px"])


# ============================================================
# PER-FRAME EXTRACTION
# ============================================================

def skin_rgb(frame_bgr, lm, w, h):
    """Mean R, G, B over forehead + cheek polygons. Returns (r, g, b, n_px)."""
    mask = np.zeros((h, w), np.uint8)
    for roi in SKIN_ROIS:
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in roi], np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    n = int(cv2.countNonZero(mask))
    if n < 50:
        return np.nan, np.nan, np.nan, n
    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return r, g, b, n


def blendshape_row(cats):
    s = {c.category_name: c.score for c in cats}
    row = {k: (s.get(a, np.nan) + s.get(b, np.nan)) / 2 for k, (a, b) in BS_PAIRS.items()}
    row.update({k: s.get(v, np.nan) for k, v in BS_SINGLE.items()})
    right = (s.get("eyeLookOutLeft", 0) + s.get("eyeLookInRight", 0)) / 2
    left = (s.get("eyeLookInLeft", 0) + s.get("eyeLookOutRight", 0)) / 2
    row["bs_look_h"] = right - left
    row["bs_blink_l"] = s.get("eyeBlinkLeft", np.nan)
    row["bs_blink_r"] = s.get("eyeBlinkRight", np.nan)
    expr = np.array([v for k, v in sorted(s.items()) if not k.startswith(EYE_PREFIXES)])
    return row, expr


def extract_session(data_dir, sid, model_path):
    base = os.path.join(data_dir, sid)
    frames_csv = pd.read_csv(base + "_data.csv")
    ts = frames_csv.timestamp_ms.to_numpy()

    cap = cv2.VideoCapture(base + ".mp4")
    if not cap.isOpened():
        print(f"  [{sid}] cannot open video, skipped")
        return None
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(n_video - len(ts)) > 2:
        print(f"  [{sid}] WARNING: video has {n_video} frames, CSV has {len(ts)} rows;"
              " aligning by index up to the shorter one")

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        ))

    rows, prev_expr = [], None
    last_t = -1
    i = 0
    while i < len(ts):
        ok, frame = cap.read()
        if not ok:
            break
        t = int(ts[i])
        mp_t = t if t > last_t else last_t + 1   # MediaPipe needs strictly increasing
        last_t = mp_t
        h, w = frame.shape[:2]
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mp_t)

        row = {c: np.nan for c in video_columns()}
        row["timestamp_ms"] = t
        row["v_face"] = 0
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            row["v_face"] = 1
            if res.face_blendshapes:
                bs, expr = blendshape_row(res.face_blendshapes[0])
                row.update(bs)
                if prev_expr is not None and len(prev_expr) == len(expr):
                    row["expr_motion"] = float(np.abs(expr - prev_expr).sum())
                prev_expr = expr
            r, g, b, n = skin_rgb(frame, lm, w, h)
            row.update(skin_r=r, skin_g=g, skin_b=b, skin_px=n)
        else:
            prev_expr = None
        rows.append(row)

        i += 1
        if i % 3000 == 0:
            print(f"  [{sid}] {i}/{len(ts)} frames")

    cap.release()
    landmarker.close()
    return pd.DataFrame(rows, columns=video_columns())


# ============================================================
# rPPG (POS)
# ============================================================

def _bandpass(x, fs, lo, hi, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def pos_signal(rgb, fs):
    """POS pulse signal from an (N, 3) RGB trace sampled at fs."""
    n = len(rgb)
    l = int(round(POS_WIN_S * fs))
    H = np.zeros(n)
    P = np.array([[0, 1, -1], [-2, 1, 1]], float)
    for m in range(l, n + 1):
        C = rgb[m - l:m].T                       # 3 x l
        mean = C.mean(axis=1, keepdims=True)
        if np.any(mean == 0):
            continue
        S = P @ (C / mean)
        sd2 = S[1].std()
        h = S[0] + (S[0].std() / sd2 if sd2 > 0 else 0) * S[1]
        H[m - l:m] += h - h.mean()
    return H


def hr_from_window(x, fs):
    """(bpm, snr_db) from the spectral peak inside the HR band."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)), nfft=4096)
    band = (f >= HR_BAND_HZ[0]) & (f <= HR_BAND_HZ[1])
    if not band.any() or p[band].sum() == 0:
        return np.nan, np.nan
    fb, pb = f[band], p[band]
    f0 = fb[np.argmax(pb)]
    # SNR (de Haan & Jeanne 2013): power near f0 and its 1st harmonic vs the rest
    near = (np.abs(fb - f0) <= 0.1) | (np.abs(fb - 2 * f0) <= 0.2)
    sig, noise = pb[near].sum(), pb[~near].sum()
    snr = 10 * np.log10(sig / noise) if noise > 0 else np.nan
    return 60.0 * f0, snr


def rppg_from_video_df(vdf):
    t = vdf.timestamp_ms.to_numpy(float)
    rgb = vdf[["skin_r", "skin_g", "skin_b"]].to_numpy(float)
    ok = ~np.isnan(rgb).any(axis=1)
    if ok.sum() < RPPG_FS * HR_WIN_S:
        return pd.DataFrame(columns=["t_ms", "hr_rppg", "rppg_snr", "rppg_valid_frac"])

    # resample to a uniform grid (camera frame times jitter)
    grid = np.arange(t[ok][0], t[ok][-1], 1000.0 / RPPG_FS)
    u = np.column_stack([np.interp(grid, t[ok], rgb[ok, c]) for c in range(3)])

    # mark grid points that fall inside long tracking gaps as invalid
    valid_t = t[ok]
    idx = np.clip(np.searchsorted(valid_t, grid), 1, len(valid_t) - 1)
    gap = np.minimum(np.abs(grid - valid_t[idx - 1]), np.abs(valid_t[idx] - grid))
    valid = gap <= MAX_GAP_S * 1000

    pulse = pos_signal(u, RPPG_FS)
    pulse = _bandpass(pulse, RPPG_FS, *HR_BAND_HZ)

    out = []
    win = int(HR_WIN_S * RPPG_FS)
    step = int(HR_STEP_S * RPPG_FS)
    for e in range(win, len(grid) + 1, step):
        vf = valid[e - win:e].mean()
        bpm, snr = (np.nan, np.nan)
        if vf >= 0.8:
            bpm, snr = hr_from_window(pulse[e - win:e], RPPG_FS)
        out.append((grid[e - 1], bpm, snr, vf))
    return pd.DataFrame(out, columns=["t_ms", "hr_rppg", "rppg_snr", "rppg_valid_frac"])


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="recordings folder")
    ap.add_argument("--session", help="only this session id")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {os.path.abspath(args.model)}")

    sids = [os.path.basename(p)[:-len("_data.csv")]
            for p in sorted(glob.glob(os.path.join(args.data, "*_data.csv")))]
    if args.session:
        sids = [s for s in sids if s == args.session]
    if not sids:
        raise SystemExit("No sessions found.")

    for sid in sids:
        base = os.path.join(args.data, sid)
        if not os.path.exists(base + ".mp4"):
            print(f"[{sid}] no .mp4, skipped")
            continue
        if os.path.exists(base + "_video.csv") and not args.overwrite:
            print(f"[{sid}] _video.csv exists (use --overwrite), recomputing rPPG only")
            vdf = pd.read_csv(base + "_video.csv")
        else:
            print(f"[{sid}] extracting video features...")
            vdf = extract_session(args.data, sid, args.model)
            if vdf is None:
                continue
            vdf.to_csv(base + "_video.csv", index=False)

        rp = rppg_from_video_df(vdf)
        rp.to_csv(base + "_rppg.csv", index=False)
        good = rp[rp.rppg_snr > 0] if len(rp) else rp
        print(f"[{sid}] face in {vdf.v_face.mean():.0%} of frames; "
              f"rPPG: {len(rp)} estimates, {len(good)} with SNR > 0 dB, "
              f"median HR {np.nanmedian(rp.hr_rppg) if len(rp) else float('nan'):.1f} bpm")


if __name__ == "__main__":
    main()
