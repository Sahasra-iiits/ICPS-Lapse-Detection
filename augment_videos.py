"""
Video augmentation: make extra training sessions from each recording, as if it
had been filmed with a different webcam / lighting.

    python augment_videos.py --data recordings --copies 3

For every original session (<session>.mp4 + _data.csv + _events.csv) it writes
<session>_aug1, _aug2, ... with the SAME timestamps, events (labels) and
participant, but features re-extracted from altered frames:
    _data.csv   landmarks / EAR / gaze / head pose   (like icps_capture.py)
    _video.csv  blendshapes / facial motion / skin colour (like extract_video_features.py)
    _rppg.csv   webcam heart rate
    _meta.json  participant, "augmented_from", the augmentation parameters
No .mp4 is written (saves disk space).

Each copy draws ONE random "camera" and keeps it for the whole session:
brightness / contrast / gamma, colour cast, sensor noise, blur, JPEG
compression, lower resolution, small rotation / zoom / shift, mirror image.

Augmented sessions are used for TRAINING ONLY. train_lapse_model.py always
tests on real (non-augmented) sessions of unseen participants, so the scores
stay honest.
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

import icps_capture as cap
from extract_video_features import (blendshape_row, rppg_from_video_df, skin_rgb,
                                    video_columns)


def random_params(rng):
    return dict(
        brightness=float(rng.uniform(-40, 40)),
        contrast=float(rng.uniform(0.7, 1.3)),
        gamma=float(rng.uniform(0.7, 1.4)),
        color_gain=[float(g) for g in rng.uniform(0.85, 1.15, 3)],   # B, G, R
        noise_sd=float(rng.choice([0, rng.uniform(2, 10)])),
        blur_px=int(rng.choice([0, 0, 3, 5])),
        jpeg_quality=int(rng.choice([0, rng.integers(25, 70)])),     # 0 = off
        downscale=float(rng.choice([1.0, 1.0, 0.5, 0.35])),          # low-res webcam
        rotate_deg=float(rng.uniform(-6, 6)),
        zoom=float(rng.uniform(0.9, 1.1)),
        shift=[float(s) for s in rng.uniform(-0.05, 0.05, 2)],
        flip=bool(rng.random() < 0.5),
    )


def augment_frame(img, p, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if p["flip"]:
        out = out[:, ::-1]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), p["rotate_deg"], p["zoom"])
    M[:, 2] += (p["shift"][0] * w, p["shift"][1] * h)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = out * np.array(p["color_gain"], np.float32)
    out = (out - 128) * p["contrast"] + 128 + p["brightness"]
    out = 255 * (np.clip(out, 0, 255) / 255) ** p["gamma"]
    if p["noise_sd"] > 0:
        out += rng.normal(0, p["noise_sd"], out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if p["blur_px"]:
        out = cv2.GaussianBlur(out, (p["blur_px"], p["blur_px"]), 0)
    if p["downscale"] < 1:
        small = cv2.resize(out, (int(w * p["downscale"]), int(h * p["downscale"])),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if p["jpeg_quality"]:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def augment_session(data_dir, sid, k, seed):
    base = os.path.join(data_dir, sid)
    out_sid = f"{sid}_aug{k}"
    out = os.path.join(data_dir, out_sid)
    rng = np.random.default_rng(seed)
    p = random_params(rng)

    orig = pd.read_csv(base + "_data.csv")
    ts = orig.timestamp_ms.to_numpy()
    vid = cv2.VideoCapture(base + ".mp4")
    lmk = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=cap.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    data_rows, video_rows, prev_expr, last = [], [], None, -1
    for i in range(len(ts)):
        ok, frame = vid.read()
        if not ok:
            break
        if frame.ndim == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = augment_frame(frame, p, rng)
        h, w = frame.shape[:2]
        cap.FRAME_W, cap.FRAME_H = w, h
        t = int(ts[i])
        mp_t = max(t, last + 1)
        last = mp_t
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mp_t)
        f = cap.extract_face_features(res)
        data_rows.append({"timestamp_ms": t,
                          "heart_rate": orig.heart_rate.iloc[i] if "heart_rate" in orig else None,
                          "spo2": orig.spo2.iloc[i] if "spo2" in orig else None, **f})
        v = {c: np.nan for c in video_columns()}
        v.update(timestamp_ms=t, v_face=0)
        if res.face_landmarks:
            v["v_face"] = 1
            if res.face_blendshapes:
                bs, expr = blendshape_row(res.face_blendshapes[0])
                v.update(bs)
                if prev_expr is not None:
                    v["expr_motion"] = float(np.abs(expr - prev_expr).sum())
                prev_expr = expr
            r, g, b, n = skin_rgb(frame, res.face_landmarks[0], w, h)
            v.update(skin_r=r, skin_g=g, skin_b=b, skin_px=n)
        else:
            prev_expr = None
        video_rows.append(v)
    vid.release()
    lmk.close()

    pd.DataFrame(data_rows, columns=cap.CSV_COLUMNS).to_csv(out + "_data.csv", index=False)
    vdf = pd.DataFrame(video_rows, columns=video_columns())
    vdf.to_csv(out + "_video.csv", index=False)
    rppg_from_video_df(vdf).to_csv(out + "_rppg.csv", index=False)
    shutil.copy(base + "_events.csv", out + "_events.csv")
    if os.path.exists(base + ".wav"):
        shutil.copy(base + ".wav", out + ".wav")
    meta = {}
    if os.path.exists(base + "_meta.json"):
        with open(base + "_meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
    meta.update(session_id=out_sid, augmented_from=sid, augmentation=p)
    meta.setdefault("participant", sid)
    with open(out + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    face = np.mean([r["face_present"] for r in data_rows]) if data_rows else 0
    print(f"  {out_sid}: face in {face:.0%} of frames  "
          f"(flip={p['flip']}, scale={p['downscale']}, jpeg={p['jpeg_quality']}, noise={p['noise_sd']:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sids = [os.path.basename(p)[:-len("_data.csv")]
            for p in sorted(glob.glob(os.path.join(args.data, "*_data.csv")))]
    sids = [s for s in sids if "_aug" not in s and os.path.exists(os.path.join(args.data, s + ".mp4"))
            and os.path.exists(os.path.join(args.data, s + "_events.csv"))]
    if not sids:
        raise SystemExit("No sessions with .mp4 + _data.csv + _events.csv found.")
    for n, sid in enumerate(sids):
        print(f"[{sid}]")
        for k in range(1, args.copies + 1):
            if os.path.exists(os.path.join(args.data, f"{sid}_aug{k}_data.csv")) and not args.overwrite:
                print(f"  {sid}_aug{k} exists, skipped")
                continue
            augment_session(args.data, sid, k, seed=args.seed * 100003 + n * 101 + k)


if __name__ == "__main__":
    main()
