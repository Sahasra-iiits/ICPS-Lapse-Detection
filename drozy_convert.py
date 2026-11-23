"""
Convert the DROZY database into this project's recording format.

    python drozy_convert.py --drozy path/to/DROZY --out drozy_converted --workers 4

DROZY (Massoz et al., WACV 2016) - 14 people x up to 3 tests, each a 10-min
PVT filmed with a Kinect v2 near-infrared camera (512x424, grayscale, 30 fps;
some dark tests 15 fps). Download: https://orbi.uliege.be/handle/2268/191620
LICENSE: research use only, NO redistribution (never commit DROZY or the
converted files to git), do not show its face images in slides/reports,
and cite: Massoz, Langohr, Francois & Verly, "The ULg Multimodality
Drowsiness Database (called DROZY) and Examples of Use", WACV 2016.

Per test S-T it writes DROZY_S-T_data.csv / _video.csv / _events.csv / _meta.json
(+ a link to the original .mp4 so augment_videos.py can use it):
  - frame time = DROZY elapsed time, same clock as the PVT file
  - PVT trials -> event_type "pvt"; lapse threshold 500 ms (standard 10-min PVT)
  - participant = DROZY_<S>, dataset = "drozy"
Grayscale video => no colour => no webcam heart rate for DROZY.
"""

import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

import icps_capture as cap
from extract_video_features import blendshape_row, video_columns

DETECT_CONF = 0.3        # NIR faces are dim; 0.3 found far more faces than 0.5
LAPSE_MS = 500


def parse_ts(s):
    return datetime.strptime(s.strip(), "%Y-%m-%d_%H.%M.%S.%f")


def read_pvt(path):
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    start = parse_ts(lines[0])
    ev = [{"timestamp_ms": 0, "event_type": "test_start", "rt_ms": None}]
    for l in lines[1:]:
        parts = [p for p in l.split(";")]
        stim = parse_ts(parts[0])
        t = (stim - start).total_seconds() * 1000
        rt = None
        if len(parts) > 1 and parts[1].strip():
            rt = (parse_ts(parts[1]) - stim).total_seconds() * 1000
        ev.append({"timestamp_ms": int(round(t)), "event_type": "pvt",
                   "rt_ms": None if rt is None or rt < 0 else round(rt, 1)})
    return pd.DataFrame(ev)


def convert(job):
    drozy, out, test, kss = job
    sid = f"DROZY_{test}"
    base = os.path.join(out, sid)
    if os.path.exists(base + "_meta.json"):
        return f"{sid}: exists, skipped"
    ts = np.loadtxt(os.path.join(drozy, "timestamps", test + ".txt"))[:, 7].astype(int)
    vid_path = os.path.abspath(os.path.join(drozy, "videos_i8", test + ".mp4"))
    vid = cv2.VideoCapture(vid_path)
    lmk = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=cap.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=DETECT_CONF,
            min_face_presence_confidence=DETECT_CONF,
            min_tracking_confidence=DETECT_CONF,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    data, video, prev, last = [], [], None, -1
    for i, t in enumerate(ts):
        ok, frame = vid.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cap.FRAME_W, cap.FRAME_H = w, h
        mp_t = max(int(t), last + 1)
        last = mp_t
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), mp_t)
        f = cap.extract_face_features(res)
        data.append({"timestamp_ms": int(t), "heart_rate": None, "spo2": None, **f})
        v = {c: np.nan for c in video_columns()}
        v.update(timestamp_ms=int(t), v_face=int(bool(res.face_landmarks)))
        if res.face_landmarks and res.face_blendshapes:
            bs, expr = blendshape_row(res.face_blendshapes[0])
            v.update(bs)
            if prev is not None:
                v["expr_motion"] = float(np.abs(expr - prev).sum())
            prev = expr
        else:
            prev = None
        video.append(v)
    vid.release()
    lmk.close()

    pd.DataFrame(data, columns=cap.CSV_COLUMNS).to_csv(base + "_data.csv", index=False)
    pd.DataFrame(video, columns=video_columns()).to_csv(base + "_video.csv", index=False)
    read_pvt(os.path.join(drozy, "pvt-rt", test + ".csv")).to_csv(base + "_events.csv", index=False)
    link = base + ".mp4"
    if not os.path.exists(link):
        try:
            os.symlink(vid_path, link)
        except OSError:
            pass
    subj, tnum = test.split("-")
    face = float(np.mean([d["face_present"] for d in data])) if data else 0.0
    with open(base + "_meta.json", "w") as fh:
        json.dump({"session_id": sid, "participant": f"DROZY_{subj}", "dataset": "drozy",
                   "task": "PVT (10 min)", "test": int(tnum), "kss": kss,
                   "lapse_threshold_ms": LAPSE_MS, "probe_timeout_s": 10.0,
                   "camera": "Kinect v2 near-infrared 512x424",
                   "face_detected_frac": round(face, 3), "n_frames": len(data)}, fh, indent=2)
    return f"{sid}: {len(data)} frames, face in {face:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drozy", required=True, help="the extracted DROZY folder")
    ap.add_argument("--out", default="drozy_converted")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    kss = np.loadtxt(os.path.join(args.drozy, "KSS.txt")).astype(int)
    tests = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.drozy, "videos_i8"))
                   if f.endswith(".mp4"))
    jobs = [(args.drozy, args.out, t, int(kss[int(t.split("-")[0]) - 1, int(t.split("-")[1]) - 1]))
            for t in tests]
    print(f"Converting {len(jobs)} DROZY tests with {args.workers} workers...")
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(convert, jobs):
            print(" ", msg, flush=True)


if __name__ == "__main__":
    main()
