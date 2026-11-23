"""
Train + evaluate a cognitive-lapse classifier.

    python train_lapse_model.py --data "C:\\Users\\sahas\\ICPS\\recordings"

Evaluation is grouped by participant (falls back to session if there are
fewer than 3 participants), so the score reflects unseen people / sessions,
not memorised ones.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="recordings folder")
    ap.add_argument("--out", default="lapse_model.joblib")
    ap.add_argument("--window", type=float, default=lf.WINDOW_S)
    ap.add_argument("--save-dataset", default="lapse_dataset.csv")
    args = ap.parse_args()

    df = lf.build_dataset(args.data, win_s=args.window)
    if df.empty:
        raise SystemExit("No labelled windows found — record sessions with probes on.")
    df.to_csv(args.save_dataset, index=False)

    y = df.label.to_numpy()
    feats = lf.feature_columns(df)
    # Drop features that are mostly missing (e.g. audio absent in some sessions)
    # or constant — they break training and can't help anyway.
    Xall = df[feats].astype(float)
    keep = (Xall.notna().mean() >= 0.8) & (Xall.nunique() > 1)
    dropped = [c for c in feats if not keep[c]]
    feats = [c for c in feats if keep[c]]
    if dropped:
        print(f"Dropped {len(dropped)} sparse/constant features: {', '.join(dropped)}")
    X = Xall[feats]

    n_part = df.participant.nunique()
    groups = df.participant if n_part >= 3 else df.session_id
    group_name = "participant" if n_part >= 3 else "session"

    print(f"\nExamples: {len(df)}  lapses: {y.sum()} ({y.mean():.1%})")
    print(f"Sessions: {df.session_id.nunique()}  participants: {n_part}")
    print(f"Features: {len(feats)}  | CV grouped by {group_name}")
    print(df.groupby('source').label.agg(['count', 'sum']).rename(
        columns={'count': 'n', 'sum': 'lapses'}))

    n_groups = groups.nunique()
    if y.sum() < 5 or (1 - y).sum() < 5 or n_groups < 2:
        raise SystemExit("\nNot enough data to evaluate yet "
                         "(need >=5 of each class and >=2 groups).")

    n_splits = min(5, n_groups)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

    results = {}
    for name, model in make_models().items():
        oof = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            model.fit(X.iloc[tr], y[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]

        ok = ~np.isnan(oof)
        if len(np.unique(y[ok])) < 2:
            print(f"{name}: folds lacked both classes, skipped")
            continue

        thr = best_threshold(y[ok], oof[ok])
        pred = oof[ok] >= thr
        results[name] = dict(
            auc=roc_auc_score(y[ok], oof[ok]),
            pr_auc=average_precision_score(y[ok], oof[ok]),
            thr=thr,
            precision=precision_score(y[ok], pred, zero_division=0),
            recall=recall_score(y[ok], pred, zero_division=0),
            f1=f1_score(y[ok], pred, zero_division=0),
        )

    print(f"\nCross-validated ({n_splits}-fold, unseen {group_name}s):")
    print(f"  chance PR-AUC = lapse rate = {y.mean():.3f}")
    res = pd.DataFrame(results).T
    print(res.round(3).to_string())

    best = res.pr_auc.idxmax()
    print(f"\nBest: {best}")

    final = make_models()[best].fit(X, y)

    imp = permutation_importance(final, X, y, scoring="average_precision",
                                 n_repeats=10, random_state=0)
    top = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    print("\nTop features (training-set permutation importance, indicative only):")
    print(top.head(12).round(4).to_string())

    joblib.dump({
        "model": final,
        "model_name": best,
        "features": feats,
        "threshold": results[best]["thr"],
        "window_s": args.window,
        "lapse_rt_factor": lf.LAPSE_RT_FACTOR,
        "lapse_rt_min_ms": lf.LAPSE_RT_MIN_MS,
        "cv": results[best],
    }, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()