"""
Native-messaging helper for the ICPS Focus Alerts Chrome extension.

Chrome starts this program itself when you switch Focus Alerts on in Meet
(after the one-time setup: install_helper.command on Mac / install_helper.bat on
Windows). It runs the same detection as live_monitor.py - webcam, MediaPipe,
eye-closure rules, the trained model, heart rate - and streams the state and
alerts to the extension over stdin/stdout (Chrome's native-messaging protocol).
When the extension disconnects (switched off, or Meet closed) it stops and
releases the camera.

Messages from the extension:
    {"cmd": "start", "settings": {"sensitivity": "normal", "enabled": [...]}}
    {"cmd": "stop"}
    {"cmd": "settings", "settings": {...}}
Messages to the extension:
    {"type": "ready", "model": "..."}
    {"type": "state", ...live state incl. the last alerts...}
    {"type": "error", "message": "..."}
"""

import json
import os
import struct
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

if sys.platform == "win32":                     # Chrome's protocol needs raw binary pipes
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

# stdout belongs to Chrome's protocol: keep a private copy of it, and send
# everything else (prints, MediaPipe / TensorFlow logs) to a log file.
_proto_out = os.fdopen(os.dup(1), "wb")
_log = open(os.path.join(LOG_DIR, "focus_host.log"), "a", buffering=1)
os.dup2(_log.fileno(), 1)
os.dup2(_log.fileno(), 2)
sys.stdout = sys.stderr = _log
_send_lock = threading.Lock()


def send(obj):
    data = json.dumps(obj, default=lambda o: None).encode("utf-8")
    with _send_lock:
        _proto_out.write(struct.pack("=I", len(data)) + data)
        _proto_out.flush()


def read_message():
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        return None
    n = struct.unpack("=I", raw)[0]
    return json.loads(sys.stdin.buffer.read(n).decode("utf-8"))


def find_model():
    candidates = [os.path.join(HERE, "models", "final_model_face_only.joblib"),
                  os.path.join(HERE, "..", "final_results", "final_model_face_only.joblib"),
                  os.path.join(HERE, "models", "final_model.joblib")]
    return next((os.path.abspath(p) for p in candidates if os.path.exists(p)), "")


class Monitor:
    def __init__(self):
        self.thread = None
        self.stop_event = None
        self.status = None
        self.error = None

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, settings):
        if self.running():
            self.apply(settings)
            return
        import live_monitor as lm
        self.status = lm.Status()
        self.status.update(started=False)
        self.stop_event = threading.Event()
        self.error = None
        self.apply(settings)
        model = find_model()
        argv = ["--source", str(settings.get("camera", 0)), "--port", "", "--log-dir", LOG_DIR,
                "--calib", str(settings.get("calib_s", 60)), "--model", model]

        def run():
            try:
                lm.main(argv, status=self.status, stop_event=self.stop_event)
            except BaseException as e:            # e.g. SystemExit("Cannot open video source")
                self.error = str(e) or e.__class__.__name__
                print("monitor stopped:", self.error, flush=True)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        print(time.strftime("%H:%M:%S"), "monitor started, model:", model or "(none: staring rule)", flush=True)

    def apply(self, settings):
        if self.status is None or not settings:
            return
        en = set(settings.get("enabled", [])) or {"dozing", "drowsy", "zoning", "stress"}
        with self.status.lock:
            self.status.requests["settings"] = (settings.get("sensitivity", "normal"), en)

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.thread = None
        print(time.strftime("%H:%M:%S"), "monitor stopped", flush=True)

    def snapshot(self):
        if self.status is None:
            return {"running": False}
        snap = self.status.snapshot()
        snap["running"] = self.running()
        return snap


def main():
    mon = Monitor()
    send({"type": "ready", "model": os.path.basename(find_model()) or None})

    def sender():
        last_err = None
        while True:
            time.sleep(0.5)
            try:
                send({"type": "state", **mon.snapshot()})
                if mon.error and mon.error != last_err:
                    last_err = mon.error
                    send({"type": "error", "message": mon.error})
            except (BrokenPipeError, OSError):
                os._exit(0)
    threading.Thread(target=sender, daemon=True).start()

    while True:
        msg = read_message()
        if msg is None:                          # Chrome closed the connection
            mon.stop()
            os._exit(0)
        cmd = msg.get("cmd")
        if cmd == "start":
            mon.start(msg.get("settings") or {})
        elif cmd == "stop":
            mon.stop()
        elif cmd == "settings":
            mon.apply(msg.get("settings") or {})


if __name__ == "__main__":
    main()
