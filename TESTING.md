# Testing and data collection

Commands use `python`. On a Mac, use `python3` until the virtual environment is active.

## 1. Setup (once per computer)

```bash
git clone https://github.com/Sahasra-iiits/ICPS-Lapse-Detection.git
cd ICPS-Lapse-Detection
python -m venv .venv
```

Activate the environment. **Windows:** `.venv\Scripts\activate`. **Mac:** `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Upgrade pip first. Older pip versions can't find the MediaPipe and OpenCV packages, and the install fails.
Each new terminal: `cd` into the folder and activate the environment again.

---

## 2. Data collection: PVT-B (about 5 min per session)

The **PVT-B** is the 3-minute brief Psychomotor Vigilance Test (Basner, Mollicone & Dinges, 2011). It is the standard short test for attention lapses. A red counter appears at random moments and the participant presses SPACE as fast as possible. A reaction time of **355 ms or more** (or no response) is a **lapse**. `pvt_b.py` runs the test and records the webcam at the same time, so every lapse is labelled automatically.

### Before you start
- **Consent:** tell the participant their face is being recorded, and get their OK.
- **Room:** quiet, with even light on the face. Don't sit with a window behind the participant.
- **Camera:** at eye level, face fully visible. Glasses are fine, but avoid glare.
- **Laptop:** plugged in, notifications off, other apps closed.
- **Optional:** clip on the ESP32 finger sensor (heart rate).

### Run
```bash
python pvt_b.py --participant P01                 # webcam only
python pvt_b.py --participant P01 --port COM4     # with the ESP32 (Windows port)
```
1. In the terminal, the participant rates their sleepiness from 1 to 9 (Karolinska Sleepiness Scale).
2. A fullscreen window opens. **Click it**, then press SPACE.
3. **60 s rest:** the participant looks at the screen and blinks normally. This is their baseline.
4. **3 min test:** press SPACE whenever the red counter appears.
5. The participant rates their sleepiness again in the terminal.

Files are saved in `recordings/`: `PVT_<date>_<id>.mp4`, `_data.csv`, `_events.csv` and `_meta.json`.
The terminal prints a summary. **If it warns "face often not detected", fix the lighting or camera angle and redo the session.**

### How many sessions
- An alert person makes very few lapses, so **record each participant twice: once alert (morning) and once tired** (late evening, after lunch, or after a long day). The tired sessions give the lapses.
- Aim for **at least 5 participants × 2 sessions**. More is better.
- Use the same participant ID (P01, P02, ...) for both of a person's sessions.
- **Never commit `recordings/` to git.** It contains faces and is already in `.gitignore`. Share recordings through a private drive instead.

Sessions recorded during real calls with `icps_capture.py` (beep probes) still count. Put them in the same folder and they're all used together.

---

## 3. Process the data and train the model

```bash
python extract_video_features.py --data recordings     # extra CV features + webcam heart rate (a few min)
python train_lapse_model.py --data recordings          # trains the model and prints the ablation table
python validate_rppg.py --data recordings              # only if the ESP32 was used: webcam vs ESP32 heart rate
```
Outputs:
- `ablation_results.csv`: which feature groups help.
- `model_comparison.csv`: 5 models (logistic regression, SVM, random forest, gradient boosting, MLP) × {real data, +SMOTE, +augmentation, +both}.
- `lapse_model.joblib`: the best model.
- `rppg_validation.png`: the heart-rate comparison plot.

### Final model (recommended): `train_final.py`
This is the setup that worked best on DROZY:
- 10/30/60 s eye-closure features, each compared against the person's alert baseline (the 60 s rest of the PVT-B);
- tuned extra trees;
- detects **drowsy minutes** (a minute with at least 25% lapses).
```bash
python train_final.py --data recordings --behaviour                         # our data
python train_final.py --data drozy_converted recordings --behaviour         # DROZY + ours
python train_final.py --data drozy_converted --test-data recordings --behaviour   # train DROZY, test ours
```
- `--behaviour` adds recent reaction times (in a call, that's how slowly someone has been responding).
- `--target trial` predicts single trials instead of drowsy minutes.

It prints F1, precision, recall, balanced accuracy, ROC-AUC, PR-AUC and MCC. **Judge F1 against `chance_f1`, and accuracy against `majority_class_accuracy`.**

DROZY results (leave-people-out):

| Setup | F1 (chance) | Balanced acc | ROC-AUC |
|---|---|---|---|
| Single trial | 0.43 (0.27) | 0.68 | 0.72 |
| Drowsy minute | 0.55 (0.37) | 0.72 | 0.80 |
| Drowsy minute + response times | **0.65** (0.37) | **0.78** | **0.85** |

SMOTE/ADASYN made F1 slightly worse on DROZY; class weights are better.

### More training data: augmentation + SMOTE
```bash
python augment_videos.py --data recordings --copies 3     # before train_lapse_model.py
```
- **Augmentation** re-renders each recording 3 times as if from a different webcam (lighting, colour, noise, blur, compression, low resolution, tilt/zoom, mirror) and re-extracts the features.
- **SMOTE** creates synthetic lapse windows to balance the classes.
- Both are used **for training only**. Scores are always on real recordings of people the model hasn't seen, and the trainer reports results with and without them, so you can show whether they helped.

### Public dataset: DROZY
14 people × up to 3 ten-minute PVTs (3,383 trials), filmed with a near-infrared camera.
1. Download `DROZY.zip` (2.5 GB) from https://orbi.uliege.be/handle/2268/191620 and unzip it outside the repo.
2. Convert it to our format:
   ```bash
   python drozy_convert.py --drozy path/to/DROZY --out drozy_converted --workers 4
   python train_lapse_model.py --data drozy_converted                          # models on DROZY
   python train_lapse_model.py --data drozy_converted --test-data recordings   # train on DROZY, test on ours
   ```

**License:** research use only. **Never commit DROZY or `drozy_converted/` to git**, don't show DROZY face images in slides, and cite: Massoz, Langohr, François & Verly, *The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use*, WACV 2016.

Read the results against **chance**: PR-AUC is only meaningful compared with the lapse rate, which the script prints.

---

## 4. Test the live alerts

```bash
python live_monitor.py --port "" --calib 30                            # no trained model: simple staring rule
python live_monitor.py --port "" --calib 30 --model lapse_model.joblib # with the trained model
```
Add `--port COM4` instead of `--port ""` to use the ESP32 for heart rate.

**Keys** (click the camera window first): `q` = quit, `h` = simulate +35 bpm heart rate (shows "SIM").

Wait until "Calibrating" disappears (sit normally and blink normally during calibration), then check each alert:

| # | Do this | Expected alert | Expected after |
|---|---|---|---|
| 1 | Sit and blink normally for 1 min | **none** | — |
| 2 | Close your eyes for 3 s | **WAKE UP!** (red, urgent beeps) | about 1.5 s |
| 3 | Let your head drop forward with eyes half-closed | **WAKE UP!** | about 0.5 s |
| 4 | Check that "HR: xx bpm" shows a number, then press `h` | **Take a deep breath** + breathing circle | about 15 s |
| 5 | Stare at one spot: no blinking, no head movement | **Zoning out?** (blue) | about 10–15 s |
| 6 | Slow, heavy blinks (eyes closed about 1 s each) for 1 min | **You're getting drowsy** (orange) | about 1 min |

Alerts are saved to `alerts_<time>.csv`. Note each test as pass or fail. If one fails, note what the bottom of the window showed (HR, PERCLOS, NO FACE).

Alert rules:
- **WAKE UP!:** eyes closed for 1.5 s or more (microsleeps are defined as 1–15 s closures; 1.5 s leaves a margin for webcam noise).
- **Drowsy:** PERCLOS of 15% or more over 1 min.
- **High HR:** heart rate at or above max(100, baseline + 20) for 15 s.
- **Zoning out:** the model says lapse twice in a row (or, without a model, a staring rule).

---

### Inside Google Meet (extension)
See `meet_extension/README.md`:
1. Load the extension in `chrome://extensions` (Developer mode → Load unpacked → `meet_extension`).
2. Run `python live_monitor.py --meet`.
3. Join a call and click once in the page.

Use the popup's **Test** buttons and **Simulate high heart rate** for a demo.

## 5. How well does the monitor catch real lapses?

Replay a PVT-B recording through the monitor, then compare its alerts and scores with the real lapses:
```bash
python live_monitor.py --source recordings/PVT_<...>_P01.mp4 --calib 60 --fast --no-sound --no-notify --model lapse_model.joblib
python compare_alerts.py --session recordings/PVT_<...>_P01 --scores scores_<time>.csv --alerts alerts_<time>.csv
```
It prints:
- ROC-AUC of the zoning score and PERCLOS vs lapses (0.5 = chance).
- How many lapses had an alert in the 10 s before.
- How many non-lapse trials had an alert (false alarms).

`--calib 60` calibrates on the 60 s rest period.
Use a session the model was **not** trained on, otherwise the result is optimistic.

---

## Troubleshooting
- **`python` / `pip` not found (Mac):** use `python3`, or activate the environment first.
- **`No module named cv2`:** the environment isn't active, or the install failed. Activate it, upgrade pip, and reinstall.
- **Camera won't open or is black (Mac):** System Settings → Privacy & Security → Camera → turn on Terminal, then restart Terminal. On Windows, close Zoom/Meet camera previews if the camera is busy.
- **Keys don't work:** click the test or camera window first.
- **`ESP32 ERROR`:** wrong port or the Serial Monitor is open. Use `--port ""` to skip it.
- **Permission denied when cloning:** the folder is owned by another user. Use a folder you own, or run `sudo chown -R $(whoami) <folder>`.
