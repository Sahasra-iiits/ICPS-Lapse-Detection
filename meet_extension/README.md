# ICPS Focus Alerts: Google Meet extension

Alerts **you** (only you) inside Google Meet with a cherry-red banner and a short sound:

| Alert | When |
|---|---|
| **WAKE UP!** | eyes closed ≥ 1.5 s, or head dropping with eyes closing |
| **You're getting drowsy** | eyes ≥ 80% closed for ≥ 15% of the last minute (PERCLOS) |
| **Zoning out?** | the trained model (or a staring rule) flags it twice in a row |
| **Take a deep breath** | heart rate ≥ max(100, your baseline + 20) for 15 s → breathing guide (4 s in, 6 s out) |

**Privacy:** the video is analysed by a small helper program on your own computer. Nothing is uploaded; the extension only talks to `127.0.0.1` (this computer).

## Setup (once)
1. Install the project (see `../TESTING.md`, section 1).
2. In Chrome, open `chrome://extensions`, turn on **Developer mode** (top right), click **Load unpacked**, and choose this `meet_extension` folder.
3. Pin the extension (puzzle icon → pin) so you can see its status badge.
4. If you update the files later, click the reload arrow on the extension card and reload the Meet tab.

## Every call
1. Start the helper in a terminal, from the repo folder with the environment activated:
   ```bash
   python live_monitor.py --meet --model ../final_results/final_model_face_only.joblib
   ```
   - You can leave out `--model` to use the simple staring rule for zoning out.
   - Add `--port COM4` if you use the ESP32 heart-rate sensor.
   - Add `--calib 20` for a shorter calibration (useful for demos).
2. Join the Meet call and **click once anywhere in the Meet page**. Browsers only allow sound after a click.
3. During calibration (60 s by default), look at the screen and blink normally.
4. The status bar at the bottom left of Meet shows: calibrating → "Focus alerts on · HR · eyes closed %".

**Toolbar badge:** `cal` = calibrating, `ok` = monitoring, `!` = alert, `off` = helper not running.

## Demo (for presentations)
- **Popup → Run full demo:** plays all four alerts, 9 s apart, including the breathing guide.
- **Popup → Preview buttons:** show one alert.
- **Keyboard, inside Meet:** `Alt+Shift+1` dozing, `Alt+Shift+2` drowsy, `Alt+Shift+3` zoning out, `Alt+Shift+4` breathing, `Alt+Shift+D` full demo.
- **Popup → Simulate high heart rate:** adds 35 bpm, so the *real* heart-rate rule fires after about 15 s. Its message says "(simulated)".
- Demo banners carry a small **DEMO** tag. Demos work **even when the helper is not running**: the banners are then shown locally.
- For a live demo of the real detection, **close your eyes for 2-3 seconds**, and WAKE UP fires.

## Popup settings
- Turn each alert on or off.
- Sensitivity: low / normal / high (eye-closure time, PERCLOS level, zoning persistence, heart-rate margin).
- Sound on/off, and status bar on/off.

## Troubleshooting
- **Status bar says "helper not running":** start the `live_monitor.py --meet` command; the extension reconnects automatically.
- **"Port 47813 is already in use":** another helper is already running. Close it.
- **No sound:** click once in the Meet page, and check that Sound is on in the popup.
- **Preview button says "Reload the Meet tab":** the Meet tab was open before the extension was installed, so reload it.
- **Camera busy (mostly Windows):** some webcams can't be used by Meet and Python at the same time. Try another camera index (`--source 1`), or a virtual camera.
