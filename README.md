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

## Results

**Model:** extra trees on eye-closure/face features at 10/30/60 s windows, each compared with the person's alert baseline, plus recent response times (`train_final.py --behaviour`).

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
