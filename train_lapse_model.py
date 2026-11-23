"""
Train + evaluate a cognitive-lapse classifier, with a feature-group ablation.

    python extract_video_features.py --data "C:\\Users\\sahas\\ICPS\\recordings"   # once
    python train_lapse_model.py --data "C:\\Users\\sahas\\ICPS\\recordings"

Evaluation is grouped by participant (falls back to session if there are
fewer than 3 participants), so the score reflects unseen people / sessions,
not memorised ones.

Changes vs. the mid-eval version:
  - The F1 decision threshold is chosen INSIDE each training fold (nested CV)
    and then applied to the held-out fold. Before, it was tuned on the test
    predictions themselves, which inflated F1 / precision / recall.
    (ROC-AUC and PR-AUC don't use a threshold and were already fine.)
  - Ablation: the same CV is run on several feature sets, so you can show
    which groups (eyes-open CV, blendshapes, webcam HR, ...) actually help.
    Results -> ablation_results.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lapse_features as lf


# Feature sets for the ablation (cumulative)
ABLATION = [
    ("1_mid_eval_features", None),   # None = lf.ORIGINAL_FEATURES
    ("2_webcam_basic", {"eye_closure", "gaze", "head"}),
    ("3_+eyes_open", {"eye_closure", "gaze", "head", "eyes_open"}),
    ("4_+blendshapes", {"eye_closure", "gaze", "head", "eyes_open", "blendshape"}),
    ("5_+webcam_hr (webcam only)", {"eye_closure", "gaze", "head", "eyes_open",
                                    "blendshape", "rppg"}),
    ("6_all (+ESP32, +mic)", {"eye_closure", "gaze", "head", "eyes_open",
                              "blendshape", "rppg", "contact", "audio"}),
]


def make_models():
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
        ),
        "gboost": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=10, l2_regularization=1.0,
            class_weight="balanced", random_state=0,
        ),
    }


def best_threshold(y, p):
    grid = np.unique(np.round(p, 3))
    scores = [f1_score(y, p >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(scores))])


def inner_threshold(model, X, y, groups):
    """Pick the F1 threshold using only training data (inner grouped CV)."""
    n_g = groups.nunique()
    if n_g >= 2 and y.sum() >= 2 and (1 - y).sum() >= 2:
        cv = StratifiedGroupKFold(n_splits=min(3, n_g), shuffle=True, random_state=1)
        p = np.full(len(y), np.nan)
        for tr, va in cv.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = clone(model).fit(X.iloc[tr], y[tr])
            p[va] = m.predict_proba(X.iloc[va])[:, 1]
        ok = ~np.isnan(p)
        if ok.sum() and len(np.unique(y[ok])) == 2:
            return best_threshold(y[ok], p[ok])
    # fallback: in-sample (only happens with very little data)
    m = clone(model).fit(X, y)
    return best_threshold(y, m.predict_proba(X)[:, 1])


def evaluate(model, X, y, groups, cv):
    oof = np.full(len(y), np.nan)
    pred = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        thr = inner_threshold(model, X.iloc[tr], y[tr], groups.iloc[tr])
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        pred[te] = oof[te] >= thr

    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return None
    yp = pred[ok].astype(bool)
    return dict(
        roc_auc=roc_auc_score(y[ok], oof[ok]),
        pr_auc=average_precision_score(y[ok], oof[ok]),
        precision=precision_score(y[ok], yp, zero_division=0),
        recall=recall_score(y[ok], yp, zero_division=0),
        f1=f1_score(y[ok], yp, zero_division=0),
        n_eval=int(ok.sum()),
    )


def select(feats, groups):
    if groups is None:
        return [c for c in feats if lf.base_name(c) in lf.ORIGINAL_FEATURES]
    return [c for c in feats if lf.group_of(c) in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="recordings folder")
    ap.add_argument("--out", default="lapse_model.joblib")
    ap.add_argument("--window", type=float, default=lf.WINDOW_S)
    ap.add_argument("--save-dataset", default="lapse_dataset.csv")
    ap.add_argument("--ablation-out", default="ablation_results.csv")
    args = ap.parse_args()

    df = lf.build_dataset(args.data, win_s=args.window)
    if df.empty:
        raise SystemExit("No labelled windows found — record sessions with probes on.")
    df.to_csv(args.save_dataset, index=False)

    y = df.label.to_numpy()
    feats = lf.feature_columns(df)
    # Drop features that are mostly missing (e.g. audio / video pass absent in
    # some sessions) or constant — they break training and can't help anyway.
    Xall = df[feats].astype(float)
    keep = (Xall.notna().mean() >= 0.8) & (Xall.nunique() > 1)
    dropped = [c for c in feats if not keep[c]]
    feats = [c for c in feats if keep[c]]
    if dropped:
        print(f"Dropped {len(dropped)} sparse/constant features: {', '.join(dropped)}")

    n_part = df.participant.nunique()
    groups = df.participant if n_part >= 3 else df.session_id
    group_name = "participant" if n_part >= 3 else "session"

    print(f"\nExamples: {len(df)}  lapses: {y.sum()} ({y.mean():.1%})")
    print(f"Sessions: {df.session_id.nunique()}  participants: {n_part}")
    print(f"Features: {len(feats)}  | CV grouped by {group_name}")
    print(pd.Series([lf.group_of(c) for c in feats]).value_counts().to_string())
    print(df.groupby('source').label.agg(['count', 'sum']).rename(
        columns={'count': 'n', 'sum': 'lapses'}))

    n_groups = groups.nunique()
    if y.sum() < 5 or (1 - y).sum() < 5 or n_groups < 2:
        raise SystemExit("\nNot enough data to evaluate yet "
                         "(need >=5 of each class and >=2 groups).")

    n_splits = min(5, n_groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

    rows = []
    for set_name, set_groups in ABLATION:
        cols = select(feats, set_groups)
        if not cols:
            print(f"{set_name}: no features available (run extract_video_features.py?)")
            continue
        X = Xall[cols]
        for name, model in make_models().items():
            r = evaluate(model, X, y, groups, cv)
            if r is None:
                continue
            rows.append(dict(feature_set=set_name, model=name, n_features=len(cols), **r))

    res = pd.DataFrame(rows)
    print(f"\nCross-validated ({n_splits}-fold, unseen {group_name}s; "
          f"threshold chosen inside training folds):")
    print(f"  chance PR-AUC = lapse rate = {y.mean():.3f}   chance ROC-AUC = 0.5")
    print(res.round(3).to_string(index=False))
    res.to_csv(args.ablation_out, index=False)
    print(f"Saved {args.ablation_out}")

    # Final model: best webcam-only set (the target system), else best overall
    webcam = res[res.feature_set.str.startswith(("2_", "3_", "4_", "5_"))]
    pick = (webcam if len(webcam) else res).sort_values("pr_auc", ascending=False).iloc[0]
    set_groups = dict(ABLATION)[pick.feature_set]
    cols = select(feats, set_groups)
    X = Xall[cols]
    model = make_models()[pick.model]
    thr = inner_threshold(model, X, y, groups)
    final = clone(model).fit(X, y)
    print(f"\nFinal model: {pick.model} on '{pick.feature_set}' "
          f"(PR-AUC {pick.pr_auc:.3f}, ROC-AUC {pick.roc_auc:.3f})")

    imp = permutation_importance(final, X, y, scoring="average_precision",
                                 n_repeats=10, random_state=0)
    top = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    print("\nTop features (training-set permutation importance, indicative only):")
    print(top.head(12).round(4).to_string())

    joblib.dump({
        "model": final,
        "model_name": pick.model,
        "feature_set": pick.feature_set,
        "features": cols,
        "threshold": thr,
        "window_s": args.window,
        "lapse_rt_factor": lf.LAPSE_RT_FACTOR,
        "lapse_rt_min_ms": lf.LAPSE_RT_MIN_MS,
        "cv": pick.to_dict(),
    }, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
