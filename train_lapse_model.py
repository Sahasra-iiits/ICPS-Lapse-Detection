"""
Train + evaluate cognitive-lapse classifiers.

    python extract_video_features.py --data recordings        # once
    python augment_videos.py --data recordings --copies 3     # optional: augmented copies
    python train_lapse_model.py --data recordings             # ablation + model comparison

Evaluation is grouped by participant (falls back to session if there are
fewer than 3 participants), so every score is on people the model has never
seen. Augmented sessions (augment_videos.py) and SMOTE samples are only ever
used for TRAINING; test folds contain real sessions only.

Outputs
  ablation_results.csv   feature groups added one at a time (logreg, gboost)
  model_comparison.csv   6 models x {all webcam features, compact physical set}
                         x {real, +augmentation, +SMOTE, +both}
  lapse_model.joblib     best webcam-only model, trained on everything

Cross-dataset (e.g. train on DROZY, test on your own recordings):
    python train_lapse_model.py --data drozy_converted --test-data recordings
Each session's features are z-scored against that session's own baseline,
which removes much of the camera / person difference between datasets.
"""

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import lapse_features as lf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WEBCAM = {"eye_closure", "gaze", "head", "eyes_open", "blendshape", "rppg"}
ABLATION = [
    ("1_mid_eval_features", None),   # None = lf.ORIGINAL_FEATURES
    ("2_webcam_basic", {"eye_closure", "gaze", "head"}),
    ("3_+eyes_open", {"eye_closure", "gaze", "head", "eyes_open"}),
    ("4_+blendshapes", {"eye_closure", "gaze", "head", "eyes_open", "blendshape"}),
    ("5_+webcam_hr (webcam only)", WEBCAM),
    ("6_all (+ESP32, +mic)", WEBCAM | {"contact", "audio"}),
]


def make_models():
    scaled = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
    return {
        "logreg": scaled(LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000)),
        "svm_rbf": scaled(SVC(C=1.0, gamma="scale", class_weight="balanced",
                              probability=True, random_state=0)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)),
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=0)),
        "gboost": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10,
            l2_regularization=1.0, class_weight="balanced", random_state=0),
        "mlp": scaled(MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=1000,
                                    early_stopping=True, random_state=0)),
    }


# ============================================================
# SMOTE (synthetic minority samples) — training data only
# ============================================================

def smote(X, y, k=5, seed=0):
    """Add synthetic lapse windows by interpolating between real lapse windows
    and their nearest lapse neighbours until both classes are the same size."""
    rng = np.random.default_rng(seed)
    Xf = X.to_numpy(float)
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isnan(med), 0, med)
    Xf = np.where(np.isnan(Xf), med, Xf)
    minority = int(np.sum(y == 1) < np.sum(y == 0))
    idx = np.where(y == minority)[0]
    need = int(np.sum(y != minority) - len(idx))
    if need <= 0 or len(idx) < 2:
        return pd.DataFrame(Xf, columns=X.columns), y
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1
    Z = Xf[idx] / sd
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx))).fit(Z)
    neigh = nn.kneighbors(Z, return_distance=False)[:, 1:]
    a = rng.integers(0, len(idx), need)
    b = neigh[a, rng.integers(0, neigh.shape[1], need)]
    lam = rng.random((need, 1))
    new = Xf[idx[a]] + lam * (Xf[idx[b]] - Xf[idx[a]])
    Xo = np.vstack([Xf, new])
    yo = np.concatenate([y, np.full(need, minority)])
    return pd.DataFrame(Xo, columns=X.columns), yo


def fit(model, X, y, use_smote):
    if use_smote:
        X, y = smote(X, y)
    return clone(model).fit(X, y)


# ============================================================
# EVALUATION
# ============================================================

def best_threshold(y, p):
    grid = np.unique(np.round(p, 3))
    scores = [f1_score(y, p >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(scores))])


def _train_idx(idx, aug, use_aug):
    return idx if use_aug else idx[aug[idx] == 0]


def inner_threshold(model, X, y, groups, aug, use_aug, use_smote):
    """F1 threshold from an inner grouped CV on training data (real validation rows)."""
    n_g = groups.nunique()
    real = aug == 0
    if n_g >= 2 and y[real].sum() >= 2 and (1 - y[real]).sum() >= 2:
        cv = StratifiedGroupKFold(n_splits=min(3, n_g), shuffle=True, random_state=1)
        p = np.full(len(y), np.nan)
        for tr, va in cv.split(X, y, groups):
            tr, va = _train_idx(tr, aug, use_aug), va[aug[va] == 0]
            if len(np.unique(y[tr])) < 2 or len(va) == 0:
                continue
            m = fit(model, X.iloc[tr], y[tr], use_smote)
            p[va] = m.predict_proba(X.iloc[va])[:, 1]
        ok = ~np.isnan(p)
        if ok.sum() and len(np.unique(y[ok])) == 2:
            return best_threshold(y[ok], p[ok])
    tr = _train_idx(np.arange(len(y)), aug, use_aug)
    m = fit(model, X.iloc[tr], y[tr], use_smote)
    return best_threshold(y[tr], m.predict_proba(X.iloc[tr])[:, 1])


def scores(y, p, pred):
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), n_eval=len(y))


def evaluate(model, X, y, groups, aug, cv, use_aug=False, use_smote=False):
    oof = np.full(len(y), np.nan)
    pred = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        tr, te = _train_idx(tr, aug, use_aug), te[aug[te] == 0]
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        thr = inner_threshold(model, X.iloc[tr], y[tr], groups.iloc[tr], aug[tr], use_aug, use_smote)
        m = fit(model, X.iloc[tr], y[tr], use_smote)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        pred[te] = oof[te] >= thr
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return None
    return scores(y[ok], oof[ok], pred[ok].astype(bool))


def select(feats, groups):
    if groups is None:
        return [c for c in feats if lf.base_name(c) in lf.ORIGINAL_FEATURES]
    return [c for c in feats if lf.group_of(c) in groups]


def usable_features(df, feats):
    X = df[feats].astype(float)
    keep = (X.notna().mean() >= 0.8) & (X.nunique() > 1)
    return [c for c in feats if keep[c]]


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training recordings folder")
    ap.add_argument("--test-data", help="cross-dataset: train on --data, test on this folder")
    ap.add_argument("--out", default="lapse_model.joblib")
    ap.add_argument("--window", type=float, default=lf.WINDOW_S)
    ap.add_argument("--save-dataset", default="lapse_dataset.csv")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    df = lf.build_dataset(args.data, win_s=args.window)
    if df.empty:
        raise SystemExit("No labelled windows found.")
    df.to_csv(args.save_dataset, index=False)

    y = df.label.to_numpy()
    aug = df.augmented.to_numpy() if "augmented" in df else np.zeros(len(df), int)
    feats = usable_features(df, lf.feature_columns(df))
    Xall = df[feats].astype(float)

    n_part = df.participant.nunique()
    groups = df.participant if n_part >= 3 else df.session_id
    group_name = "participant" if n_part >= 3 else "session"
    real = aug == 0

    print(f"\nWindows: {len(df)} ({real.sum()} real, {(~real).sum()} augmented)  "
          f"lapses in real data: {y[real].sum()} ({y[real].mean():.1%})")
    print(f"Sessions: {df.session_id.nunique()}  participants: {n_part}  "
          f"features: {len(feats)}  | CV grouped by {group_name}, tested on real sessions only")
    print(df[real].groupby('source').label.agg(['count', 'sum']).rename(
        columns={'count': 'n', 'sum': 'lapses'}).to_string())

    models = make_models()

    # ---------------- cross-dataset ----------------
    if args.test_data:
        cross_dataset(args, df, y, aug, feats, models)
        return

    if y[real].sum() < 5 or (1 - y[real]).sum() < 5 or groups[real].nunique() < 2:
        raise SystemExit("\nNot enough real data to evaluate yet "
                         "(need >=5 of each class and >=2 groups).")
    cv = StratifiedGroupKFold(n_splits=min(5, groups[real].nunique()), shuffle=True, random_state=0)
    chance = y[real].mean()
    print(f"\nChance level: PR-AUC = {chance:.3f} (lapse rate), ROC-AUC = 0.5")

    # ---------------- ablation ----------------
    if not args.no_ablation:
        rows = []
        for set_name, set_groups in ABLATION:
            cols = select(feats, set_groups)
            if not cols:
                continue
            for name in ("logreg", "gboost"):
                r = evaluate(models[name], Xall[cols], y, groups, aug, cv)
                if r:
                    rows.append(dict(feature_set=set_name, model=name, n_features=len(cols), **r))
        res = pd.DataFrame(rows)
        print("\n=== Ablation: feature groups (real data only) ===")
        print(res.round(3).to_string(index=False))
        res.to_csv("ablation_results.csv", index=False)

    # ---------------- model comparison ----------------
    # "physical" = compact eye-closure set (best on DROZY, see diagnose_dataset.py)
    phys = [c for c in lf.PHYSICAL_FEATURES if c in feats]
    feature_sets = {"webcam (all)": select(feats, WEBCAM), "physical (compact)": phys}
    configs = [("real only", False, False), ("+SMOTE", False, True)]
    if (~real).any():
        configs += [("+augmentation", True, False), ("+augmentation +SMOTE", True, True)]
    rows = []
    if not args.no_compare:
        for fname, fcols in feature_sets.items():
            if not fcols:
                continue
            for cname, use_aug, use_smote in configs:
                for mname, model in models.items():
                    r = evaluate(model, Xall[fcols], y, groups, aug, cv, use_aug, use_smote)
                    if r:
                        rows.append(dict(features=fname, training_data=cname, model=mname, **r))
                        print(f"  {fname:18s} {cname:22s} {mname:14s} "
                              f"PR-AUC {r['pr_auc']:.3f}  ROC-AUC {r['roc_auc']:.3f}")
        comp = pd.DataFrame(rows)
        print(f"\n=== Model comparison, unseen {group_name}s ===")
        print(comp.round(3).to_string(index=False))
        comp.to_csv("model_comparison.csv", index=False)
        best = comp.sort_values("pr_auc", ascending=False).iloc[0]
        cfg = {c[0]: c[1:] for c in configs}[best.training_data]
        mname = best.model
        cols = feature_sets[best.features]
    else:
        cfg, mname, best = (False, False), "logreg", None
        cols = feature_sets["webcam (all)"]
    X = Xall[cols]

    # ---------------- final model ----------------
    use_aug, use_smote = cfg
    model = models[mname]
    thr = inner_threshold(model, X, y, groups, aug, use_aug, use_smote)
    tr = _train_idx(np.arange(len(y)), aug, use_aug)
    final = fit(model, X.iloc[tr], y[tr], use_smote)
    print(f"\nFinal model: {mname}, training data: "
          f"{'real' + (' +augmentation' if use_aug else '') + (' +SMOTE' if use_smote else '')}")
    imp = permutation_importance(final, X[real], y[real], scoring="average_precision",
                                 n_repeats=10, random_state=0)
    top = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    print("Top features (permutation importance on real training data, indicative only):")
    print(top.head(12).round(4).to_string())
    joblib.dump({"model": final, "model_name": mname,
                 "feature_set": best.features if best is not None else "webcam (all)",
                 "features": cols, "threshold": thr, "window_s": args.window,
                 "use_augmentation": use_aug, "use_smote": use_smote,
                 "cv": None if best is None else best.to_dict()}, args.out)
    print(f"Saved {args.out}")


def cross_dataset(args, df, y, aug, feats, models):
    test = lf.build_dataset(args.test_data, win_s=args.window)
    if test.empty:
        raise SystemExit("No labelled windows in --test-data.")
    test = test[test.get("augmented", 0) == 0]
    cols = [c for c in select(feats, WEBCAM) if c in test.columns]
    cols = [c for c in cols if test[c].notna().mean() >= 0.5]
    yt = test.label.to_numpy()
    print(f"\nCross-dataset: train {args.data} ({len(df)} windows) -> "
          f"test {args.test_data} ({len(test)} windows, {yt.sum()} lapses, "
          f"chance PR-AUC {yt.mean():.3f}); {len(cols)} shared features")
    if len(np.unique(yt)) < 2:
        raise SystemExit("Test data needs both lapse and non-lapse windows.")
    groups = df.participant if df.participant.nunique() >= 3 else df.session_id
    rows = []
    for use_aug, use_smote in [(False, False), (False, True), (True, False), (True, True)]:
        if use_aug and not (aug == 1).any():
            continue
        for mname, model in models.items():
            thr = inner_threshold(model, df[cols], y, groups, aug, use_aug, use_smote)
            tr = _train_idx(np.arange(len(y)), aug, use_aug)
            m = fit(model, df[cols].iloc[tr], y[tr], use_smote)
            p = m.predict_proba(test[cols].astype(float))[:, 1]
            name = "real" + (" +aug" if use_aug else "") + (" +SMOTE" if use_smote else "")
            rows.append(dict(training_data=name, model=mname, **scores(yt, p, p >= thr)))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv("cross_dataset_results.csv", index=False)


if __name__ == "__main__":
    main()
