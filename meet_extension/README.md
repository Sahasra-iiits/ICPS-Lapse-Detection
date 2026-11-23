# ICPS Focus Alerts: Google Meet extension

Alerts **you** (only you) inside Google Meet when you:

| Alert | When |
|---|---|
| 😴 **WAKE UP!** | eyes closed ≥ 1.5 s, or head dropping with eyes closing |
| 🥱 **You're getting drowsy** | eyes ≥ 80% closed for ≥ 15% of the last minute (PERCLOS) |
| 💭 **Zoning out?** | the trained model (or a staring rule) flags it twice in a row |
| 🫁 **Take a deep breath** | heart rate ≥ max(100, your baseline + 20) for 15 s → breathing guide (4 s in, 6 s out) |

Each alert shows a coloured banner at the top of Meet and plays a short sound.

**Privacy:** the video is analysed by a small helper program on your own computer. Nothing is uploaded; the extension only talks to `127.0.0.1` (this computer).

## Setup (once)
1. Install the project (see `../TESTING.md`, section 1).
2. In Chrome, open `chrome://extensions`, turn on **Developer mode** (top right), click **Load unpacked**, and choose this `meet_extension` folder.
3. Pin the extension (puzzle icon → pin) so you can see its status badge.

## Every call
1. Start the helper in a terminal, from the repo folder with the environment activated:
   ```bash
   python live_monitor.py --meet --model ../final_results/final_model_face_only.joblib
   ```
   You can leave out `--model` to use the simple staring rule for zoning out. Add `--port COM4` if you use the ESP32 heart-rate sensor.
2. Join the Meet call and **click once anywhere in the Meet page**. Browsers only allow sound after a click.
3. For the **first 60 s, look at the screen and blink normally** (calibration).
4. The status pill at the bottom left of Meet shows: calibrating → on · HR · eyes closed %.

**Toolbar badge:** `cal` = calibrating, `ok` = monitoring, `!` = alert, `off` = helper not running.

**Popup (click the icon):**
- live readings;
- turn individual alerts on/off;
- sensitivity (low / normal / high) and mute;
- **Test buttons** for each alert, and **Simulate high heart rate**, for demos.

## Troubleshooting
- **Pill says "helper not running":** start the `live_monitor.py --meet` command; the extension reconnects automatically.
- **"Port 47813 is already in use":** another helper is already running. Close it.
- **No sound:** click once in the Meet page, and check that **Mute** is off in the popup.
- **Camera busy (mostly Windows):** some webcams can't be used by Meet and Python at the same time. Try another camera index (`--source 1`), or a virtual camera.
