"""
PVT-B: 3-minute brief Psychomotor Vigilance Test, recorded with the webcam.

    python pvt_b.py --participant P01                 # webcam only
    python pvt_b.py --participant P01 --port COM4     # + ESP32 heart rate

Basner, Mollicone & Dinges (2011), Acta Astronautica 69:949-959.
  - a red millisecond counter appears after a random 1-4 s wait
  - press SPACE as fast as possible; your RT is shown for 1 s
  - lapse = RT >= 355 ms (or no response); false start = press < 100 ms after
    the counter appears, or with no counter on screen

Session flow: KSS sleepiness rating -> 60 s rest (just look at the screen;
this is the person's baseline) -> 3 min PVT-B -> KSS again.

Writes the same files as icps_capture.py, so extract_video_features.py,
train_lapse_model.py and live_monitor.py --source work on them directly:
    <session>.mp4, <session>_data.csv, <session>_events.csv, <session>_meta.json
Trials are logged as event_type "pvt" (rt_ms empty = no response) and
"false_start".
"""

import argparse
import csv
import json
import os
import random
import threading
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

import icps_capture as cap

TEST_S = 180
REST_S = 60
ISI_S = (1.0, 4.0)
LAPSE_MS = 355
FALSE_START_MS = 100
TIMEOUT_MS = 10000          # no response for this long = missed trial
FEEDBACK_S = 1.0
WIN = "PVT-B"


class CameraRecorder(threading.Thread):
    """Records webcam video + per-frame face features on the shared clock."""

    def __init__(self, t0, video_path, csv_path, camera):
        super().__init__(daemon=True)
        self.t0, self.video_path, self.csv_path, self.camera = t0, video_path, csv_path, camera
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.n_frames = 0
        self.last_ts = -1
        self.face_frac = []

    def run(self):
        try:
            self._run()
        except Exception as e:          # surface camera problems to the main thread
            self.error = e
            self.ready.set()

    def _run(self):
        vid = cv2.VideoCapture(self.camera)
        if not vid.isOpened():
            raise RuntimeError("Could not open webcam (camera permission?)")
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        vid.set(cv2.CAP_PROP_FPS, cap.VIDEO_FPS)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.FRAME_W, cap.FRAME_H = w, h
        writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.VIDEO_FPS, (w, h))
        lmk = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=cap.MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
                output_facial_transformation_matrixes=True))
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            out = csv.writer(fh)
            out.writerow(cap.CSV_COLUMNS)
            self.ready.set()
            while not self.stop.is_set():
                ok, frame = vid.read()
                if not ok:
                    break
                ts = int((time.monotonic() - self.t0) * 1000)
                ts = max(ts, self.last_ts + 1)
                self.last_ts = ts
                res = lmk.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                f = cap.extract_face_features(res)
                with cap.sensor_lock:
                    now = time.monotonic()
                    hr = cap.latest_hr if cap.latest_hr_time and now - cap.latest_hr_time < cap.SENSOR_STALE_S else None
                    sp = cap.latest_spo2 if cap.latest_spo2_time and now - cap.latest_spo2_time < cap.SENSOR_STALE_S else None
                row = {"timestamp_ms": ts, "heart_rate": hr, "spo2": sp, **f}
                out.writerow([row[c] for c in cap.CSV_COLUMNS])
                writer.write(frame)
                self.n_frames += 1
                self.face_frac.append(f["face_present"])
        writer.release()
        vid.release()
        lmk.close()


def screen(lines, color=(255, 255, 255), size=(720, 1280)):
    img = np.zeros((*size, 3), np.uint8)
    y = size[0] // 2 - 30 * (len(lines) - 1)
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(img, s, ((size[1] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y += 60
    return img


def counter(ms):
    img = np.zeros((720, 1280, 3), np.uint8)
    s = f"{int(ms):d}"
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, 4.0, 8)
    cv2.putText(img, s, ((1280 - tw) // 2, (720 + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 8)
    return img


def ask_kss(when):
    print(f"\nKarolinska Sleepiness Scale ({when}): how sleepy are you RIGHT NOW?")
    print("  1 extremely alert   3 alert   5 neither alert nor sleepy")
    print("  7 sleepy, no effort to stay awake   9 very sleepy, fighting sleep")
    while True:
        v = input("  Rating 1-9: ").strip()
        if v.isdigit() and 1 <= int(v) <= 9:
            return int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", default="", help="ESP32 serial port, e.g. COM4 (optional)")
    ap.add_argument("--minutes", type=float, default=TEST_S / 60)
    ap.add_argument("--rest", type=float, default=REST_S)
    args = ap.parse_args()

    if not os.path.exists(cap.MODEL_PATH):
        raise SystemExit(f"{cap.MODEL_PATH} not found - run from the repo folder.")
    os.makedirs(args.out, exist_ok=True)
    kss_before = ask_kss("before")

    if args.port:
        cap.ESP32_PORT = args.port
        threading.Thread(target=cap.esp32_reader, daemon=True).start()

    sid = datetime.now().strftime("PVT_%Y%m%d_%H%M%S_") + args.participant
    base = os.path.join(args.out, sid)
    t0 = time.monotonic()
    now_ms = lambda: int((time.monotonic() - t0) * 1000)

    rec = CameraRecorder(t0, base + ".mp4", base + "_data.csv", args.camera)
    rec.start()
    rec.ready.wait(20)
    if rec.error or not rec.ready.is_set():
        raise SystemExit(f"Camera error: {rec.error}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(img, wait=1):
        cv2.imshow(WIN, img)
        return cv2.waitKey(wait) & 0xFF

    # ---- instructions ----
    msg = ["PVT-B  (3 minutes)", "",
           "A red counter will appear at random times.",
           "Press SPACE as fast as you can when it appears.",
           "Don't press before it appears.", "",
           "Click this window, then press SPACE to begin."]
    while show(screen(msg), 30) != 32:
        if rec.error:
            raise SystemExit(f"Camera error: {rec.error}")

    # ---- rest / baseline ----
    rest_start = now_ms()
    events = [{"timestamp_ms": rest_start, "event_type": "rest_start", "rt_ms": None}]
    while (now_ms() - rest_start) / 1000 < args.rest:
        left = args.rest - (now_ms() - rest_start) / 1000
        show(screen(["Relax and look at the screen.", "Blink normally.", "",
                     f"Test starts in {left:.0f} s"]), 30)

    # ---- PVT-B ----
    test_start = now_ms()
    events.append({"timestamp_ms": test_start, "event_type": "test_start", "rt_ms": None})
    blank = screen([])
    rts = []
    while (now_ms() - test_start) / 1000 < args.minutes * 60:
        # wait (ISI): any press here is a false start
        isi_end = time.monotonic() + random.uniform(*ISI_S)
        while time.monotonic() < isi_end:
            if show(blank) == 32:
                events.append({"timestamp_ms": now_ms(), "event_type": "false_start", "rt_ms": None})
        # stimulus
        show(counter(0))
        onset = time.monotonic()
        onset_ms = int((onset - t0) * 1000)
        rt = None
        while True:
            el = (time.monotonic() - onset) * 1000
            if el >= TIMEOUT_MS:
                break
            if show(counter(el)) == 32:
                rt = (time.monotonic() - onset) * 1000
                break
        if rt is not None and rt < FALSE_START_MS:
            events.append({"timestamp_ms": onset_ms, "event_type": "false_start", "rt_ms": round(rt, 1)})
        else:
            events.append({"timestamp_ms": onset_ms, "event_type": "pvt",
                           "rt_ms": round(rt, 1) if rt is not None else None})
            rts.append(rt)
        fb = screen([f"{rt:.0f} ms" if rt is not None else "No response"],
                    color=(0, 0, 255) if rt is None or rt >= LAPSE_MS else (0, 255, 0))
        fb_end = time.monotonic() + FEEDBACK_S
        while time.monotonic() < fb_end:
            show(fb)

    events.append({"timestamp_ms": now_ms(), "event_type": "test_end", "rt_ms": None})
    show(screen(["Done - thank you!", "", "Answer the last question in the terminal."]), 500)
    rec.stop.set()
    rec.join(5)
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    kss_after = ask_kss("after")

    with open(base + "_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp_ms", "event_type", "rt_ms"])
        w.writeheader()
        w.writerows(events)

    valid = [r for r in rts if r is not None]
    n_lapse = sum(1 for r in rts if r is None or r >= LAPSE_MS)
    n_fs = sum(e["event_type"] == "false_start" for e in events)
    dur_s = max(rec.last_ts / 1000, 1e-6)
    summary = dict(trials=len(rts), lapses=n_lapse, false_starts=n_fs,
                   mean_rt_ms=round(float(np.mean(valid)), 1) if valid else None,
                   median_rt_ms=round(float(np.median(valid)), 1) if valid else None,
                   mean_speed_1_per_s=round(float(np.mean([1000 / r for r in valid])), 3) if valid else None)
    with open(base + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "session_id": sid, "participant": args.participant, "task": "PVT-B",
            "start_time": datetime.now().isoformat(),
            "rest_s": args.rest, "test_minutes": args.minutes,
            "lapse_threshold_ms": LAPSE_MS, "false_start_ms": FALSE_START_MS,
            "probe_timeout_s": TIMEOUT_MS / 1000,
            "kss_before": kss_before, "kss_after": kss_after,
            "frame_size": [cap.FRAME_W, cap.FRAME_H],
            "video_fps_header": cap.VIDEO_FPS,
            "video_fps_actual": round(rec.n_frames / dur_s, 2),
            "face_detected_frac": round(float(np.mean(rec.face_frac)), 3) if rec.face_frac else 0,
            "summary": summary,
        }, fh, indent=2)

    print("\n" + "=" * 50)
    print(f" PVT-B done - {sid}")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  KSS before/after: {kss_before} -> {kss_after}")
    print(f"  face detected in {np.mean(rec.face_frac):.0%} of frames, "
          f"{rec.n_frames / dur_s:.1f} fps")
    if rec.face_frac and np.mean(rec.face_frac) < 0.9:
        print("  WARNING: face often not detected - check lighting / camera angle")
    print(f"  files: {base}.mp4 / _data.csv / _events.csv / _meta.json")


if __name__ == "__main__":
    main()
