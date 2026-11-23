# ICPS Lapse Detection

Detecting transient cognitive lapses and drowsiness during online communication from webcam video: eye closure, blinks, gaze, head pose and facial motion. Includes live alerts for dozing off, zoning out and high heart rate.

How to set up, collect data, train and test: see [TESTING.md](TESTING.md).

## Data

| | People | Trials | Lapses | Not lapses |
|---|---|---|---|---|
| DROZY (public dataset) | 14 | 3,383 | 518 | 2,865 |
| Our recordings (PVT-B) | 8 | 370 | 93 | 277 |
| **Real total** | **22** | **3,753** | **611** | **3,142** |
| Augmented copies (training only) | — | 2,485 | 905 | 1,580 |
| **Total incl. augmentation** | | **6,238** | **1,516** | **4,722** |

- **One trial** = one Psychomotor Vigilance Test (PVT) stimulus, plus the face features from the seconds before it.
- **Lapse** = reaction time ≥ 355 ms or no response (our 3-minute PVT-B), or ≥ 500 ms (DROZY's 10-minute PVT).
- **Drowsy minute** = a minute in which ≥ 25% of trials are lapses. There are 388 real minutes: 91 drowsy and 297 alert.
- **Augmented copies** re-render lapse-rich sessions as if from a different webcam (lighting, noise, blur, compression, resolution, tilt, mirror). They are used for training only and never for testing.

## Model

Implemented in `train_final.py`.

1. **Face tracking:** MediaPipe Face Landmarker (478 landmarks, head pose, blendshapes) on every webcam frame.
2. **Features:** computed over the **10 s, 30 s and 60 s** before each moment.

   | Feature | Meaning |
   |---|---|
   | `max_closure_ms`, `long_closures` | longest eye closure; closures > 500 ms (microsleep-like) |
   | `blink_dur_mean`, `blink_dur_max` | blink durations |
   | `perclos`, `closure_mean` | share of time eyes ≥ 80% closed; mean eye closure |
   | `head_speed_mean` | head movement |
   | `bs_blink_mean`, `perclos_bs`, `expr_motion_mean` | blendshape blink score, facial motion (only when the video was processed) |

   Eye closure is measured per person, relative to their own open/closed eye opening (EAR), not a fixed threshold.
3. **Alert-baseline normalisation:** each feature is also expressed as *difference from the person's alert baseline*, meaning the 60 s rest before the PVT-B (or, for DROZY, the person's least sleepy session). This removes person-to-person differences. The live monitor does the same by calibrating at the start of a call.
4. **Behaviour (optional, `--behaviour`):** recent reaction times (previous trial, mean of last 5, lapses in last 5). In a call, these correspond to response delays.
5. **Classifier:** **Extra Trees**, with 300 trees, `min_samples_leaf=20`, `max_features="sqrt"` and balanced class weights. The hyperparameters were tuned with nested leave-people-out cross-validation on DROZY. Training trials that follow ≥ 3 consecutive false starts get weight 0.25, as they're probably mis-presses.
6. **Smoothing:** the scores are causally averaged over past trials (exponential, half-life 15 trials), because drowsiness is a state that lasts.
7. **Output:** lapse probability per trial. For drowsy minutes, the scores are averaged per minute. The decision thresholds are chosen on training people only.

**Models compared on DROZY (leave-people-out):** logistic regression (L1/L2), linear and RBF SVM, random forest, extra trees, gradient boosting, MLP, balanced random forest and EasyEnsemble. Extra trees was the most consistent.

**Getting the model file:** `final_model.joblib` is not in the repo, because it was trained partly on DROZY, which may not be redistributed. Rebuild it with:
```bash
python train_final.py --data drozy_converted recordings --behaviour
```
`--test-people` evaluates on a fixed test set, and `--rotate-dataset own` tests each of our participants once.

## Results

**Test set:** 8 people never seen in training (4 DROZY + 4 of ours).

| Unit | ROC-AUC | Balanced accuracy | F1 (chance) | Accuracy (always "no") |
|---|---|---|---|---|
| **Drowsy minute** | **0.80** | **0.73** | 0.62 (0.47) | 0.76 (0.70) |
| Single trial | 0.76 | 0.70 | 0.50 (0.34) | 0.75 (0.80) |

- **Balanced accuracy** is the average of the hit rate on drowsy and on alert minutes; 0.5 = chance. We report it instead of plain accuracy, because lapses are rare and always guessing "no lapse" already gives high accuracy.
- On our own recordings alone the model is weaker (single-trial ROC-AUC 0.69, 183 test trials). Our sessions are short (3 min) and participants were mostly alert.
- Synthetic oversampling (SMOTE/ADASYN) did not improve F1. Changing the target to drowsy minutes and adding response-time features did.

## DROZY

DROZY is not included in this repo: its license forbids redistribution. Download it from https://orbi.uliege.be/handle/2268/191620 and convert it with `drozy_convert.py`. Cite:

> Q. Massoz, T. Langohr, C. François, J. G. Verly, "The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use", IEEE WACV 2016.
