"""
ICPS online-communication recorder, with lapse probes for labelling.

Changes vs. the previous version:
  - Lapse probes: a short beep at random intervals; the participant taps
    Right Ctrl (global hotkey, works while Meet/Zoom has focus). Slow/missed = lapse.
  - Self-report: Right Shift = "I just zoned out".
  - Test keys + beep first:  python icps_recorder.py --keytest
  - Events saved to <session>_events.csv and an `events` MySQL table.
  - Gaze is now measured relative to the eye corners (gaze_h, gaze_v), so head
    movement no longer leaks into "eye movement".
  - Head pose (yaw / pitch / roll) from MediaPipe's transformation matrix.
  - face_present column; when no face is found, features are written as empty
    instead of repeating the last stale value.
  - Per-eye-closed frames are logged as eye_closed (blinks are computed later
    from runs of closed frames — one blink spans several frames).
  - ESP32 thread starts before recording so HR is ready at t=0.
  - MySQL password comes from the ICPS_MYSQL_PASSWORD env var.

Extra dependency:  pip install pynput
Use headphones, so the probe beep doesn't go out over the meeting.
"""

import cv2
import mediapipe as mp
import serial
import threading
import time
import os
import csv
import json
import math
import random
import wave
from datetime import datetime

import numpy as np
import mysql.connector
import sounddevice as sd
from pynput import keyboard

try:
    import winsound  # Windows only
except ImportError:
    winsound = None


# ============================================================
# CONFIGURATION
# ============================================================

ESP32_PORT = "COM4"
ESP32_BAUD = 115200

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("ICPS_MYSQL_PASSWORD", "")
MYSQL_DATABASE = "icps"

MODEL_PATH = "face_landmarker.task"
CAMERA_INDEX = 0
VIDEO_FPS = 30.0          # your webcam delivers ~30 fps

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1

OUTPUT_DIR = r"C:\Users\sahas\ICPS\recordings"

EAR_THRESHOLD = 0.20

# ---- Lapse probes ----
PROBES_ENABLED = True
PROBE_MIN_S = 10          # random gap between probes
PROBE_MAX_S = 20
PROBE_TIMEOUT_S = 3.0     # no tap within this = "miss"
PROBE_FREQ_HZ = 1000
PROBE_DUR_MS = 250
PROBE_VOLUME = 0.6        # 0..1
# Delay between "beep requested" and sound actually reaching the ear.
# Bluetooth headphones: ~150-300 ms, wired: ~10-40 ms. Measure once (record the
# beep with the laptop mic, or film a key tap + beep with a phone at 240 fps)
# and put it here; lapse_features.py subtracts it from every RT.
PROBE_OUTPUT_LATENCY_MS = 0

# Right Ctrl / Right Shift: no Fn-key weirdness on laptops, harmless in calls.
# Run `python icps_recorder.py --keytest` to see what your keys register as.
PROBE_KEY = keyboard.Key.ctrl_r     # tap when you hear the beep
REPORT_KEY = keyboard.Key.shift_r   # tap when you notice you zoned out
KEY_DEBUG = False                   # True = print every key press


# ============================================================
# SHARED STATE
# ============================================================

sensor_lock = threading.Lock()

latest_hr = None
latest_spo2 = None
latest_esp32_timestamp = None
latest_hr_time = None      # monotonic time of last VALID hr
latest_spo2_time = None
SENSOR_STALE_S = 3.0       # older than this = write blank, not a stale value

stop_threads = threading.Event()
recording_started = threading.Event()
start_monotonic = None

events_lock = threading.Lock()
events = []               # dicts: timestamp_ms, event_type, rt_ms
pending_probe_ms = None   # onset of an unanswered probe


def now_ms():
    return int((time.monotonic() - start_monotonic) * 1000)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def setup_database():
    print("Connecting to MySQL...")

    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
    cur.close()
    conn.close()

    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL UNIQUE,
            video_file VARCHAR(255),
            audio_file VARCHAR(255),
            start_time DATETIME,
            end_time DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            heart_rate FLOAT,
            spo2 FLOAT,
            eye_aspect_ratio FLOAT,
            blink BOOLEAN,
            eye_x FLOAT,
            eye_y FLOAT,
            head_x FLOAT,
            head_y FLOAT,
            FOREIGN KEY (session_id)
                REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            event_type VARCHAR(32) NOT NULL,
            rt_ms FLOAT,
            FOREIGN KEY (session_id)
                REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    cur.close()
    return conn


# ============================================================
# ESP32 THREAD
# ============================================================

def esp32_reader():
    global latest_hr, latest_spo2, latest_esp32_timestamp
    global latest_hr_time, latest_spo2_time

    print("Connecting to ESP32...")
    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        time.sleep(2)
        print("ESP32 connected.")

        while not stop_threads.is_set():
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line.startswith("DATA,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                esp_time = int(parts[1])
                hr = float(parts[2])
                spo2 = float(parts[3])
            except ValueError:
                continue

            with sensor_lock:
                latest_esp32_timestamp = esp_time
                if 30 <= hr <= 220:
                    latest_hr = hr
                    latest_hr_time = time.monotonic()
                if 50 <= spo2 <= 100:
                    latest_spo2 = spo2
                    latest_spo2_time = time.monotonic()

        ser.close()

    except Exception as e:
        print("\nESP32 ERROR:", e)
        print("Check: Serial Monitor closed, ESP32 plugged in, port =", ESP32_PORT)


# ============================================================
# PROBES + HOTKEYS
# ============================================================

def log_event(event_type, timestamp_ms, rt_ms=None):
    with events_lock:
        events.append({
            "timestamp_ms": timestamp_ms,
            "event_type": event_type,
            "rt_ms": rt_ms,
        })


_beep_out_rate = None


def _beep_tone():
    global _beep_out_rate
    if _beep_out_rate is None:
        dev = sd.query_devices(kind="output")
        _beep_out_rate = int(dev["default_samplerate"])
    sr = _beep_out_rate
    t = np.arange(int(sr * PROBE_DUR_MS / 1000)) / sr
    tone = PROBE_VOLUME * np.sin(2 * np.pi * PROBE_FREQ_HZ * t)
    ramp = min(len(t) // 10, int(sr * 0.01))          # 10 ms fade, no clicks
    if ramp:
        tone[:ramp] *= np.linspace(0, 1, ramp)
        tone[-ramp:] *= np.linspace(1, 0, ramp)
    return tone.astype(np.float32), sr


def play_beep():
    """Plays through the default OUTPUT device (headphones).
    winsound.Beep is silent on many laptops, so it's only a fallback."""
    try:
        tone, sr = _beep_tone()
        sd.play(tone, sr)
    except Exception as e:
        print("Beep via sounddevice failed:", e)
        if winsound is not None:
            winsound.Beep(PROBE_FREQ_HZ, PROBE_DUR_MS)


def probe_loop():
    global pending_probe_ms

    recording_started.wait()
    rng = random.Random()

    while not stop_threads.is_set():
        if stop_threads.wait(rng.uniform(PROBE_MIN_S, PROBE_MAX_S)):
            break

        onset = now_ms()
        with events_lock:
            pending_probe_ms = onset
        print(f"[probe] beep at {onset / 1000:.1f}s")
        play_beep()

        stop_threads.wait(PROBE_TIMEOUT_S)

        with events_lock:
            if pending_probe_ms == onset:       # nobody answered
                pending_probe_ms = None
                events.append({
                    "timestamp_ms": onset,
                    "event_type": "probe",
                    "rt_ms": None,               # miss
                })
                print("[probe] MISS")


def on_key_press(key):
    global pending_probe_ms

    if KEY_DEBUG:
        print("[key]", key)

    if not recording_started.is_set():
        return

    t = now_ms()

    if key == PROBE_KEY:
        with events_lock:
            if pending_probe_ms is not None:
                events.append({
                    "timestamp_ms": pending_probe_ms,
                    "event_type": "probe",
                    "rt_ms": t - pending_probe_ms,
                })
                print(f"[probe] hit, RT = {t - pending_probe_ms} ms")
                pending_probe_ms = None
            else:
                events.append({
                    "timestamp_ms": t,
                    "event_type": "false_alarm",
                    "rt_ms": None,
                })
                print("[probe] key pressed but no beep pending (false alarm)")

    elif key == REPORT_KEY:
        log_event("self_report", t)
        print("[self-report] logged")


# ============================================================
# FACE FEATURES
# ============================================================

# Landmarks are normalised to width/height separately, so on a 16:9 frame
# vertical distances come out ~1.8x too large. Work in pixels instead.
FRAME_W, FRAME_H = 1280, 720


def dist(a, b):
    return math.hypot((a.x - b.x) * FRAME_W, (a.y - b.y) * FRAME_H)


def calc_ear(lm, idx):
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    h = dist(p1, p4)
    if h == 0:
        return 0.0
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * h)


def gaze_in_eye(lm, iris, c_a, c_b):
    """Iris position relative to the two eye corners.
    h: 0..1 along the corner line, v: offset from corner midline / eye width."""
    a, b, i = lm[c_a], lm[c_b], lm[iris]
    w = b.x - a.x
    width = dist(a, b)
    if abs(w) < 1e-6 or width < 1e-6:
        return None, None
    h = (i.x - a.x) / w
    v = (i.y - (a.y + b.y) / 2.0) * FRAME_H / width
    return h, v


def head_pose(matrix):
    """Yaw / pitch / roll in degrees from MediaPipe's 4x4 transform."""
    R = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def extract_face_features(result):
    """Returns a dict of per-frame face features; all None if no face."""
    empty = dict(
        face_present=0, ear=None, ear_left=None, ear_right=None,
        eye_closed=None, eye_x=None, eye_y=None, gaze_h=None, gaze_v=None,
        head_x=None, head_y=None, yaw=None, pitch=None, roll=None,
    )
    if not result.face_landmarks:
        return empty

    lm = result.face_landmarks[0]

    ear_l = calc_ear(lm, LEFT_EYE)
    ear_r = calc_ear(lm, RIGHT_EYE)
    ear = (ear_l + ear_r) / 2.0

    gh_l, gv_l = gaze_in_eye(lm, 468, 33, 133)
    gh_r, gv_r = gaze_in_eye(lm, 473, 362, 263)
    hs = [g for g in (gh_l, gh_r) if g is not None]
    vs = [g for g in (gv_l, gv_r) if g is not None]

    yaw = pitch = roll = None
    if result.facial_transformation_matrixes:
        yaw, pitch, roll = head_pose(result.facial_transformation_matrixes[0])

    return dict(
        face_present=1,
        ear=ear, ear_left=ear_l, ear_right=ear_r,
        eye_closed=int(ear < EAR_THRESHOLD),
        eye_x=(lm[468].x + lm[473].x) / 2.0,
        eye_y=(lm[468].y + lm[473].y) / 2.0,
        gaze_h=sum(hs) / len(hs) if hs else None,
        gaze_v=sum(vs) / len(vs) if vs else None,
        head_x=lm[1].x, head_y=lm[1].y,
        yaw=yaw, pitch=pitch, roll=roll,
    )


CSV_COLUMNS = [
    "timestamp_ms", "heart_rate", "spo2", "face_present",
    "ear", "ear_left", "ear_right", "eye_closed",
    "eye_x", "eye_y", "gaze_h", "gaze_v",
    "head_x", "head_y", "yaw", "pitch", "roll",
]


# ============================================================
# MAIN
# ============================================================

def main():
    global start_monotonic

    print("\n" + "=" * 60)
    print(" ICPS ONLINE COMMUNICATION DATA COLLECTION")
    print("=" * 60 + "\n")

    participant = input("Participant ID (e.g. P01): ").strip() or "unknown"

    if PROBES_ENABLED:
        print(f"\nProbes ON: every {PROBE_MIN_S}-{PROBE_MAX_S}s you'll hear a beep.")
        print("  RIGHT CTRL  = tap as fast as you can when you hear the beep")
        print("  RIGHT SHIFT = tap whenever you notice you zoned out")
        print("  (keys work even while the meeting window has focus)")
        print("  Use headphones so the beep isn't sent into the call.\n")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: face_landmarker.task not found at", os.path.abspath(MODEL_PATH))
        return

    try:
        db = setup_database()
        print("MySQL connected.")
    except Exception as e:
        print("MYSQL ERROR:", e)
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        db.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    ))

    # Start sensors early so HR is warm by the time recording starts
    esp_thread = threading.Thread(target=esp32_reader, daemon=True)
    esp_thread.start()

    probe_thread = threading.Thread(target=probe_loop, daemon=True)
    if PROBES_ENABLED:
        probe_thread.start()
    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

    input("When the call is ready, press ENTER to start recording...")

    # ---------------- Session ----------------
    session_id = datetime.now().strftime("ICPS_%Y%m%d_%H%M%S_%f")
    start_datetime = datetime.now()

    video_path = os.path.join(OUTPUT_DIR, session_id + ".mp4")
    audio_path = os.path.join(OUTPUT_DIR, session_id + ".wav")
    csv_path = os.path.join(OUTPUT_DIR, session_id + "_data.csv")
    events_path = os.path.join(OUTPUT_DIR, session_id + "_events.csv")
    meta_path = os.path.join(OUTPUT_DIR, session_id + "_meta.json")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global FRAME_W, FRAME_H
    FRAME_W, FRAME_H = width, height
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height)
    )
    if not video_writer.isOpened():
        print("ERROR: Could not create video.")
        stop_threads.set()
        cap.release()
        landmarker.close()
        db.close()
        return

    audio_data = []

    def audio_callback(indata, frames, callback_time, status):
        if status:
            print("Audio status:", status)
        audio_data.append(indata.copy())

    audio_stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS,
        callback=audio_callback,
    )

    cur = db.cursor()
    cur.execute(
        "INSERT INTO sessions (session_id, video_file, audio_file, start_time) "
        "VALUES (%s, %s, %s, %s)",
        (session_id, os.path.basename(video_path),
         os.path.basename(audio_path), start_datetime),
    )
    db.commit()
    cur.close()

    start_monotonic = time.monotonic()
    audio_stream.start()
    audio_start_ms = now_ms()
    recording_started.set()

    print("\n" + "=" * 60)
    print(" RECORDING STARTED —", session_id)
    print("=" * 60)
    print("Press Q in the camera window to stop.\n")

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_COLUMNS)

    last_db_write = 0.0
    last_ts = -1
    n_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read webcam frame.")
                break

            timestamp_ms = now_ms()
            if timestamp_ms <= last_ts:          # MediaPipe needs strictly increasing
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )
            f = extract_face_features(result)

            with sensor_lock:
                now_t = time.monotonic()
                hr = latest_hr if latest_hr_time and now_t - latest_hr_time < SENSOR_STALE_S else None
                spo2 = latest_spo2 if latest_spo2_time and now_t - latest_spo2_time < SENSOR_STALE_S else None

            row = {"timestamp_ms": timestamp_ms, "heart_rate": hr, "spo2": spo2, **f}

            video_writer.write(frame)
            n_frames += 1
            csv_writer.writerow([row[c] for c in CSV_COLUMNS])

            if time.monotonic() - last_db_write >= 0.1:
                last_db_write = time.monotonic()
                cur = db.cursor()
                cur.execute(
                    """
                    INSERT INTO sensor_data
                    (session_id, timestamp_ms, heart_rate, spo2,
                     eye_aspect_ratio, blink, eye_x, eye_y, head_x, head_y)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, timestamp_ms, hr, spo2, f["ear"],
                     f["eye_closed"], f["eye_x"], f["eye_y"],
                     f["head_x"], f["head_y"]),
                )
                db.commit()
                cur.close()

            # ---------------- Display ----------------
            display = frame.copy()
            elapsed = timestamp_ms / 1000.0
            lines = [("ICPS RECORDING", 0.8, (0, 255, 0)),
                     (f"Time: {elapsed:.1f}s", 0.6, (255, 255, 255))]
            if hr is not None:
                lines.append((f"HR: {hr:.1f}", 0.6, (255, 255, 255)))
            if spo2 is not None:
                lines.append((f"SpO2: {spo2:.1f}", 0.6, (255, 255, 255)))
            if f["ear"] is not None:
                lines.append((f"EAR: {f['ear']:.3f}", 0.6, (255, 255, 255)))
            else:
                lines.append(("NO FACE", 0.6, (0, 0, 255)))
            with events_lock:
                n_probes = sum(e["event_type"] == "probe" for e in events)
            lines.append((f"Probes: {n_probes}", 0.6, (255, 255, 255)))

            for i, (text, scale, color) in enumerate(lines):
                cv2.putText(display, text, (20, 35 + 35 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            cv2.putText(display, "Press Q to stop", (20, height - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("ICPS Online Communication Capture", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    finally:
        stop_threads.set()
        key_listener.stop()
        time.sleep(0.5)

        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        csv_file.close()

        # ---------------- Audio ----------------
        if audio_data:
            audio = np.clip(np.concatenate(audio_data, axis=0), -1, 1)
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        # ---------------- Events ----------------
        with events_lock:
            ev = sorted(events, key=lambda e: e["timestamp_ms"])

        with open(events_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["timestamp_ms", "event_type", "rt_ms"])
            w.writeheader()
            w.writerows(ev)

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "session_id": session_id,
                "participant": participant,
                "start_time": start_datetime.isoformat(),
                "audio_start_ms": audio_start_ms,
                "audio_sample_rate": AUDIO_SAMPLE_RATE,
                "ear_threshold": EAR_THRESHOLD,
                "probe_timeout_s": PROBE_TIMEOUT_S,
                "probe_output_latency_ms": PROBE_OUTPUT_LATENCY_MS,
                "frame_size": [FRAME_W, FRAME_H],
                "video_fps_header": VIDEO_FPS,
                "video_fps_actual": round(n_frames / max(last_ts / 1000.0, 1e-6), 2),
                "probes_enabled": PROBES_ENABLED,
            }, fh, indent=2)

        cur = db.cursor()
        if ev:
            cur.executemany(
                "INSERT INTO events (session_id, timestamp_ms, event_type, rt_ms) "
                "VALUES (%s, %s, %s, %s)",
                [(session_id, e["timestamp_ms"], e["event_type"], e["rt_ms"]) for e in ev],
            )
        cur.execute("UPDATE sessions SET end_time = %s WHERE session_id = %s",
                    (datetime.now(), session_id))
        db.commit()
        cur.close()
        db.close()

        n_probe = sum(e["event_type"] == "probe" for e in ev)
        n_miss = sum(e["event_type"] == "probe" and e["rt_ms"] is None for e in ev)
        n_rep = sum(e["event_type"] == "self_report" for e in ev)

        print("\n" + "=" * 60)
        print(" RECORDING FINISHED —", session_id)
        print("=" * 60)
        print(f"Probes: {n_probe} (missed {n_miss}), self-reports: {n_rep}")
        fps = n_frames / max(last_ts / 1000.0, 1e-6)
        print(f"Actual camera fps: {fps:.1f} (video header {VIDEO_FPS:.0f})")
        if abs(fps - VIDEO_FPS) > 3:
            print(f"  -> set VIDEO_FPS = {round(fps)} so the .mp4 plays at real speed")
        for p in (video_path, audio_path, csv_path, events_path, meta_path):
            print(" ", p)
        print()


def keytest():
    """Quick check: beeps every 3 s and prints every key you press.
    Click into another window (e.g. the browser) to confirm keys still register."""
    print("KEY TEST — beeps every 3 s, prints every key. Ctrl+C to quit.")
    print(f"Probe key = {PROBE_KEY}, report key = {REPORT_KEY}")
    print("Output device:", sd.query_devices(kind="output")["name"])

    def show(key):
        tag = ""
        if key == PROBE_KEY:
            tag = "  <-- PROBE KEY OK"
        elif key == REPORT_KEY:
            tag = "  <-- REPORT KEY OK"
        print("[key]", key, tag)

    listener = keyboard.Listener(on_press=show)
    listener.start()
    try:
        while True:
            print("beep")
            play_beep()
            time.sleep(3)
    except KeyboardInterrupt:
        listener.stop()


if __name__ == "__main__":
    import sys
    if "--keytest" in sys.argv:
        keytest()
    else:
        main()