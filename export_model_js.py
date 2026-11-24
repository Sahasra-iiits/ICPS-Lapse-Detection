"""
Export a train_final.py face-only model to JSON for the Chrome extension, which
runs the detection in the browser (meet_extension/detector.js).

    python export_model_js.py                                   # ../final_results/final_model_face_only.joblib
    python export_model_js.py --model path/to/final_model_face_only.joblib

Writes meet_extension/model/zoning_model.json. Re-run it after retraining, then
reload the extension in chrome://extensions.

The extension evaluates the trees exactly like scikit-learn: inputs are cast to
float32 and compared with `<=` against the stored (float64) thresholds, and the
forest's probability is the mean of the trees' leaf probabilities.
"""

import argparse
import json
import math
import os

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def tree_to_json(est, pos):
    t = est.tree_
    leaf = t.children_left == -1
    val = t.value[:, 0, :]
    prob = val[:, pos] / val.sum(axis=1)           # normalised, like DecisionTreeClassifier.predict_proba
    return {
        "f": [int(f) if not lf else -1 for f, lf in zip(t.feature, leaf)],
        # internal node: threshold (exact float64 repr); leaf: P(lapse)
        "t": [float(p) if lf else float(th) for th, p, lf in zip(t.threshold, prob, leaf)],
        "l": [int(c) for c in t.children_left],
        "r": [int(c) for c in t.children_right],
    }


def num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "final_results", "final_model_face_only.joblib"))
    ap.add_argument("--out", default=os.path.join(HERE, "meet_extension", "model", "zoning_model.json"))
    a = ap.parse_args()

    m = joblib.load(a.model)
    if "windows_s" not in m:
        raise SystemExit("Not a train_final.py model (no windows_s).")
    if m.get("behaviour"):
        raise SystemExit("This model uses response-time features, which a live call doesn't have. "
                         "Export the face-only model instead.")
    est = m["model"]
    pos = list(est.classes_).index(1)
    out = {
        "source": os.path.basename(a.model),
        "features": list(m["features"]),
        "medians": [num(m["medians"].get(c)) for c in m["features"]],
        "threshold": float(m["thresholds"]["trial"]),
        "windows_s": [int(w) for w in m["windows_s"]],
        "smoothing_ticks": 30,                      # live_monitor.ZONE_SMOOTH_TICKS
        "trees": [tree_to_json(e, pos) for e in est.estimators_],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"Wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(out['trees'])} trees, "
          f"{len(out['features'])} features)")


if __name__ == "__main__":
    main()
