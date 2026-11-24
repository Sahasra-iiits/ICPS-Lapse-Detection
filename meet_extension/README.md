# ICPS Focus Alerts: Google Meet extension

Alerts **you** (only you) inside Google Meet with a cherry-red banner and a short sound:

| Alert | When |
|---|---|
| **WAKE UP!** | eyes closed ≥ 1.5 s, or head dropping with eyes closing |
| **You're getting drowsy** | eyes ≥ 80% closed for ≥ 15% of the last minute (PERCLOS) |
| **Zoning out?** | the trained model flags it twice in a row |
| **Take a deep breath** | heart rate (from the webcam, rPPG) ≥ max(100, your baseline + 20) for 15 s → breathing guide (4 s in, 6 s out) |

**Privacy:** the video is analysed inside Chrome on your computer (MediaPipe + the trained model, bundled in this folder). Nothing is uploaded or recorded.

## Setup (once)
1. In Chrome, open `chrome://extensions`, turn on **Developer mode** (top right), click **Load unpacked**, and choose this `meet_extension` folder.
2. Pin the extension (puzzle icon → pin) so you can see its status badge.

No Python or helper program is needed.

## Every call
1. Join the Meet call and **click once anywhere in the Meet page** (browsers only allow sound after a click).
2. Switch **Focus Alerts** on in the status bar at the bottom left (or turn on *Start automatically* in the popup).
3. The first time, a tab opens and Chrome asks for camera access. Click **Allow**; you're taken back to Meet and it starts.
4. Calibration (20 s): look at the screen and blink normally. Then the status bar shows "On".

**Toolbar badge:** `…` = starting / calibrating, `on` = monitoring, `!` = alert (or an error when off).

## Popup settings
- Turn each alert on or off.
- Sensitivity: low / normal / high (eye-closure time, PERCLOS level, zoning persistence, heart-rate margin).
- Start automatically, sound on/off, and status bar on/off.

## How it works
- `background.js` opens a hidden page (`offscreen.html`) when you join a call, so MediaPipe is already loaded when you switch on. Leaving the call closes it and turns the camera off.
- `detector.js` (in that page) reads the camera, runs MediaPipe Face Landmarker, and feeds each frame to `lapse_core.js`.
- `lapse_core.js` is a JavaScript port of `live_monitor.py`: same features, calibration, alert rules, rPPG and zoning model. `python tests/check_js_port.py` (Mac) checks that it matches the Python version.
- `model/zoning_model.json` is the face-only model from `train_final.py`, exported with `python export_model_js.py`. Re-export after retraining, then click the reload arrow on the extension card.
- `camera.html` asks for camera access once. The hidden page can't show Chrome's prompt itself.
- `lib/` holds MediaPipe's browser library (`@mediapipe/tasks-vision` 1.2.2, Apache 2.0). Chrome extensions can't load code from the internet.

## Troubleshooting
- **Camera blocked:** in the camera tab, click the camera icon at the left of the address bar, set Camera to Allow, then click *Try again*.
- **"The camera is being used by another app":** close other apps using the camera (Meet itself is fine).
- **No sound:** click once in the Meet page, and check that Sound is on in the popup.
- **Status bar missing after installing or updating:** reload the Meet tab.
- The ESP32 heart-rate sensor isn't supported in the extension yet. Heart rate comes from the webcam.
