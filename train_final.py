"""
Final lapse / drowsiness model: the setup that worked best on DROZY
(see diagnose_dataset.py and the DROZY experiments).

    python train_final.py --data recordings                              # drowsy-minute model
    python train_final.py --data recordings --target trial               # single-trial lapses
    python train_final.py --data recordings --behaviour                  # + recent response times
    python train_final.py --data drozy_converted recordings              # pool datasets
    python train_final.py --data drozy_converted --test-data recordings  # train on one, test on other

What it does
  - eye/face "physical" features (closure durations, PERCLOS, blink score, facial
    motion, head speed) at 10, 30 and 60 s windows
  - each person compared to their own ALERT BASELINE (x - baseline):
      1. the 60 s rest period of a PVT-B session (pvt_b.py), else
      2. the person's least sleepy session (KSS in _meta.json, e.g. DROZY), else
      3. the first 2 minutes of the person's first session
    (the live monitor does the same by calibrating at the start of a call)
  - optional behaviour features: recent reaction times (past trials only).
    In a call these correspond to how slowly someone has been responding.
  - extra trees (tuned on DROZY), causal smoothing of scores over past trials
  - target "minute": a minute is DROWSY if >= 25 % of its trials are lapses.
    This matches how alerts work (a state, not one slow reaction) and scored
    much better than single trials on DROZY. target "trial": single trials.
  - leave-people-out CV; decision threshold chosen on TRAINING people only.
    Augmented sessions (augment_videos.py) are used for training only.
Outputs: final_results.csv, final_per_person.csv, final_model.joblib
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

import lapse_features as lf

WINDOWS = (10, 30, 60)
PHYS = lf.PHYSICAL_FEATURES
BEHAVIOUR = ["prev_rt", "recent_rt_mean5", "recent_lapses5"]
SMOOTH_HALF_LIFE = 15          # trials
DROWSY_MINUTE_RATE = 0.25
CALIB_S = 120
FS_RUN = 3                  # >= this many false starts right before a trial ...
FS_WEIGHT = 0.25           # ... -> that trial gets this weight in TRAINING


# ============================================================
# FEATURES
# ============================================================

def session_rows(sess, w):
    """Physical features at every labelled trial for one window length."""
    win = int(w * 1000)
    fr, au, rp = sess["frames"], sess["audio"], sess.get("rppg")
    noise = float(np.percentile(au.rms, 10)) if au is not None else None
    rows = []
    for t_end, y, src, rt in lf.label_points(sess["events"], sess["pvt_lapse_ms"]):
        f = lf.window_features(fr, au, t_end, win, noise, rp)
        if f.get("n_frames", 0) < 5:
            continue
        rows.append({"t_end_ms": t_end, "label": y, "source": src, "rt_ms": rt,
                     **{f"{c}@{w}": f.get(c, np.nan) for c in PHYS}})
    return rows


def baseline_vector(sess, kind):
    """Median physical features over the session's calibration period."""
    fr, au, rp = sess["frames"], sess["audio"], sess.get("rppg")
    ev = sess["events"]
    t0 = fr.timestamp_ms.min()
    if kind == "rest":
        a = float(ev[ev.event_type == "rest_start"].timestamp_ms.iloc[0])
        b = float(ev[ev.event_type == "test_start"].timestamp_ms.iloc[0])
    else:
        a, b = t0, t0 + CALIB_S * 1000
    out = {}
    for w in WINDOWS:
        win = int(min(w * 1000, b - a))
        ts = np.arange(a + win, b + 1, 2000) if b - a >= 2000 else [b]
        feats = [lf.window_features(fr, au, t, win, None, rp) for t in ts]
        df = pd.DataFrame(feats)
        for c in PHYS:
            out[f"{c}@{w}"] = df[c].median() if c in df else np.nan
    return out


def load_folder(folder):
    # cache in the CURRENT folder (never write into the recordings folder)
    name = os.path.basename(os.path.normpath(os.path.abspath(folder)))
    cache = f"_features_{name}.csv"
    newest = max([os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)] or [0])
    if os.path.exists(cache) and os.path.getmtime(cache) > newest:
        return pd.read_csv(cache)
    parts = []
    for sid in lf.list_sessions(folder):
        sess = lf.load_session(folder, sid)
        meta_p = os.path.join(folder, sid + "_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        per_w = [pd.DataFrame(session_rows(sess, w)) for w in WINDOWS]
        if any(len(p) == 0 for p in per_w):
            continue
        d = per_w[0]
        for p in per_w[1:]:
            d = d.merge(p.drop(columns=["label", "source", "rt_ms"]), on="t_end_ms")
        ev = sess["events"].event_type
        has_rest = (ev == "rest_start").any() and (ev == "test_start").any()
        # false starts since the previous trial (runs of them = likely mis-presses)
        fs_t = sess["events"][ev == "false_start"].timestamp_ms.to_numpy(float)
        trial_t = np.sort(d.t_end_ms.unique())
        prev_t = np.concatenate([[-np.inf], trial_t[:-1]])
        n_fs = {t: int(((fs_t > pt) & (fs_t <= t)).sum()) for t, pt in zip(trial_t, prev_t)}
        d["false_starts_before"] = d.t_end_ms.map(n_fs)
        base = baseline_vector(sess, "rest" if has_rest else "start")
        d = d.assign(session_id=sid, participant=sess["participant"], augmented=int(sess["augmented"]),
                     dataset=sess["dataset"], has_rest=int(has_rest),
                     kss=meta.get("kss", meta.get("kss_before", np.nan)),
                     **{f"base_{k}": v for k, v in base.items()})
        parts.append(d)
        print(f"  {sid}: {len(d)} trials", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(cache, index=False)
    return out


def add_person_baseline(df):
    """x - alert baseline of the same person (rest > least sleepy session > first 2 min)."""
    raw = [f"{c}@{w}" for w in WINDOWS for c in PHYS]
    df = df.copy()
    for p, g in df.groupby("participant"):
        real = g[g.augmented == 0] if (g.augmented == 0).any() else g
        if real.has_rest.any():
            ref = real[real.has_rest == 1][[f"base_{c}" for c in raw]].median()
            ref.index = raw
        elif real.kss.notna().any() and real.session_id.nunique() > 1:
            sid = real.sort_values(["kss", "session_id"]).session_id.iloc[0]
            ref = real[real.session_id == sid][raw].median()
        else:
            first = real.sort_values("session_id").session_id.iloc[0]
            ref = real[real.session_id == first][[f"base_{c}" for c in raw]].median()
            ref.index = raw
        df.loc[g.index, [c + "_pd" for c in raw]] = (g[raw] - ref).to_numpy()
    return df, raw + [c + "_pd" for c in raw]


def add_behaviour(df):
    df = df.sort_values(["session_id", "t_end_ms"]).reset_index(drop=True)
    g = df.groupby("session_id")
    df["prev_rt"] = g.rt_ms.shift(1)
    df["recent_rt_mean5"] = g.rt_ms.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["recent_lapses5"] = g.label.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    return df


# ============================================================
# MODEL / EVALUATION
# ============================================================

def model():
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1, random_state=0)


def train_weights(d):
    """Trials right after a run of false starts are probably mis-presses: down-weight them."""
    fs = d["false_starts_before"].fillna(0).to_numpy() if "false_starts_before" in d else np.zeros(len(d))
    return np.where(fs >= FS_RUN, FS_WEIGHT, 1.0)


def smooth(d, p, half_life=SMOOTH_HALF_LIFE):
    a = 1 - 0.5 ** (1 / half_life)
    s = pd.Series(p, index=d.index)
    return s.groupby(d.session_id.values).transform(lambda x: x.ewm(alpha=a, adjust=True).mean()).to_numpy()


def fit(tr, cols):
    med = tr[cols].median()
    m = model().fit(tr[cols].fillna(med).fillna(0), tr.label, sample_weight=train_weights(tr))
    return m, med


def predict(m, med, te, cols, smooth_on):
    s = m.predict_proba(te[cols].fillna(med).fillna(0))[:, 1]
    return smooth(te, s) if smooth_on else s


def to_units(d, score, unit, min_trials):
    """Per-trial rows, or per-minute rows (mean score, drowsy label)."""
    if unit == "trial":
        return d.label.to_numpy(), score, d.participant.to_numpy()
    m = d.assign(s=score, minute=d.t_end_ms // 60000).groupby(
        ["participant", "session_id", "minute"]).agg(s=("s", "mean"), rate=("label", "mean"),
                                                      n=("label", "size")).reset_index()
    m = m[m.n >= min_trials]
    return (m.rate >= DROWSY_MINUTE_RATE).astype(int).to_numpy(), m.s.to_numpy(), m.participant.to_numpy()


def best_threshold(y, s):
    if len(np.unique(y)) < 2:
        return 0.5
    grid = np.unique(np.quantile(s, np.linspace(0.05, 0.99, 95)))
    return float(grid[np.argmax([f1_score(y, s >= g) for g in grid])])


def scores_of(y, s, pred):
    out = dict(n=len(y), positives=float(y.mean()))
    if len(np.unique(y)) == 2:
        out.update(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
    out.update(f1=f1_score(y, pred, zero_division=0), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               balanced_accuracy=balanced_accuracy_score(y, pred) if len(np.unique(y)) == 2 else np.nan,
               accuracy=accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
               chance_f1=2 * y.mean() / (1 + y.mean()), majority_class_accuracy=max(y.mean(), 1 - y.mean()))
    return out


def thresholds_from_training(tr, cols, min_trials, smooth_on):
    """Decision thresholds (trial + minute) from inner leave-people-out on TRAINING people."""
    real = tr[tr.augmented == 0]
    inner = np.full(len(real), np.nan)
    people = real.participant.unique()
    k = min(3, len(people))
    if k < 2:
        return {"trial": 0.5, "minute": 0.5}
    for i, test_p in enumerate(np.array_split(np.random.default_rng(1).permutation(people), k)):
        trn = tr[~tr.participant.isin(test_p)]
        if trn.label.nunique() < 2:
            continue
        m, med = fit(trn, cols)
        idx = real.participant.isin(test_p).to_numpy()
        inner[idx] = predict(m, med, real[idx], cols, smooth_on)
    ok = ~np.isnan(inner)
    r = real[ok]
    return {u: best_threshold(*to_units(r, inner[ok], u, min_trials)[:2]) for u in ("trial", "minute")}


def evaluate_split(tr, te, cols, min_trials, smooth_on):
    thr = thresholds_from_training(tr, cols, min_trials, smooth_on)
    m, med = fit(tr, cols)
    s = predict(m, med, te, cols, smooth_on)
    return s, thr


def report(te, s, thr, min_trials, title):
    rows = []
    print(f"\n=== {title} ===")
    for unit in ("minute", "trial"):
        for name, mask in [("ALL test people", np.ones(len(te), bool))] + [
                (f"{ds} only", (te.dataset == ds).to_numpy()) for ds in sorted(te.dataset.unique())]:
            if mask.sum() == 0:
                continue
            y, su, _ = to_units(te[mask], s[mask], unit, min_trials)
            if len(y) == 0:
                continue
            r = scores_of(y, su, su >= thr[unit])
            rows.append(dict(evaluation=title, unit=unit, subset=name, **r))
            auc = f"ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  " if "roc_auc" in r else ""
            print(f"  {unit:6s} {name:18s} n={r['n']:4d} pos={r['positives']:.0%}  {auc}"
                  f"F1 {r['f1']:.3f} (chance {r['chance_f1']:.3f})  prec {r['precision']:.3f}  "
                  f"recall {r['recall']:.3f}  bal.acc {r['balanced_accuracy']:.3f}  "
                  f"acc {r['accuracy']:.3f} (majority {r['majority_class_accuracy']:.3f})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more recording folders")
    ap.add_argument("--test-people", nargs="*", help="fixed split: these participants are the TEST set")
    ap.add_argument("--rotate-dataset", help="also rotate which people of this dataset are tested "
                                             "(e.g. 'own'); everyone else always trains")
    ap.add_argument("--behaviour", action="store_true", help="add recent reaction-time features")
    ap.add_argument("--min-trials", type=int, default=5, help="trials needed for a minute to count")
    ap.add_argument("--own-lapse-ms", type=float, help="override the PVT lapse threshold for non-DROZY data")
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--no-augmented", action="store_true", help="ignore augmented sessions")
    ap.add_argument("--out", default="final_model.joblib")
    args = ap.parse_args()
    smooth_on = not args.no_smoothing

    df = pd.concat([load_folder(f) for f in args.data], ignore_index=True)
    if "false_starts_before" not in df:
        df["false_starts_before"] = 0
    df["false_starts_before"] = df.false_starts_before.fillna(0)
    if args.own_lapse_ms:
        own = (df.dataset != "drozy") & (df.source == "pvt")
        df.loc[own, "label"] = (df.loc[own, "rt_ms"].isna() | (df.loc[own, "rt_ms"] >= args.own_lapse_ms)).astype(int)
    if args.no_augmented:
        df = df[df.augmented == 0]
    df, cols = add_person_baseline(df)
    if args.behaviour:
        df = add_behaviour(df)
        cols += BEHAVIOUR
    df = df.sort_values(["session_id", "t_end_ms"]).reset_index(drop=True)

    # only keep features that every dataset actually has (e.g. blendshapes need the video)
    real = df[df.augmented == 0]
    miss = real.groupby("dataset")[cols].apply(lambda g: g.isna().mean()).max()
    dropped = [c for c in cols if miss[c] > 0.5]
    cols = [c for c in cols if c not in dropped]
    if dropped:
        bases = sorted({c.split("@")[0] for c in dropped})
        print(f"Dropped {len(dropped)} features missing in some dataset: {', '.join(bases)}")

    print(f"\n{real.participant.nunique()} people ({', '.join(f'{k}: {v}' for k, v in real.groupby('dataset').participant.nunique().items())}), "
          f"{len(real)} real trials, {real.label.mean():.1%} lapses; "
          f"{int((train_weights(real) < 1).sum())} trials down-weighted after >= {FS_RUN} false starts; "
          f"{df.augmented.sum()} augmented windows")

    all_rows = []
    if args.test_people:
        test_mask = df.participant.isin(args.test_people)
        tr, te = df[~test_mask], df[test_mask & (df.augmented == 0)]
        print(f"TRAIN: {', '.join(sorted(tr.participant.unique()))}\nTEST:  {', '.join(sorted(te.participant.unique()))}")
        s, thr = evaluate_split(tr, te, cols, args.min_trials, smooth_on)
        all_rows += report(te, s, thr, args.min_trials, "Fixed test set")
        pp = []
        for p, g in te.assign(s=s).groupby("participant"):
            if g.label.nunique() == 2:
                pp.append(dict(person=p, dataset=g.dataset.iloc[0], trials=len(g), lapses=int(g.label.sum()),
                               roc_auc=roc_auc_score(g.label, g.s), f1=f1_score(g.label, g.s >= thr["trial"])))
        if pp:
            pp = pd.DataFrame(pp)
            print("\n  per test person (trial level):")
            print(pp.round(3).to_string(index=False))
            pp.to_csv("final_per_person.csv", index=False)

    if args.rotate_dataset:
        rp = real[real.dataset == args.rotate_dataset].participant.unique()
        k = min(4, len(rp))
        units = {"minute": [], "trial": []}
        for test_p in np.array_split(np.random.default_rng(0).permutation(rp), k):
            tr = df[~df.participant.isin(test_p)]
            te = df[df.participant.isin(test_p) & (df.augmented == 0)]
            s, thr = evaluate_split(tr, te, cols, args.min_trials, smooth_on)
            for u in units:                                   # each fold uses ITS OWN threshold
                y, su, _ = to_units(te, s, u, args.min_trials)
                units[u].append((y, su, su >= thr[u]))
        title = f"Rotation: every '{args.rotate_dataset}' person tested once ({k} folds; everyone else trains)"
        print(f"\n=== {title} ===")
        for u, parts in units.items():
            y, su, pr = (np.concatenate(x) for x in zip(*parts))
            r = scores_of(y, su, pr)
            all_rows.append(dict(evaluation=title, unit=u, subset=args.rotate_dataset, **r))
            auc = f"ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  " if "roc_auc" in r else ""
            print(f"  {u:6s} n={r['n']:4d} pos={r['positives']:.0%}  {auc}F1 {r['f1']:.3f} "
                  f"(chance {r['chance_f1']:.3f})  prec {r['precision']:.3f}  recall {r['recall']:.3f}  "
                  f"bal.acc {r['balanced_accuracy']:.3f}  acc {r['accuracy']:.3f} "
                  f"(majority {r['majority_class_accuracy']:.3f})")

    pd.DataFrame(all_rows).to_csv("final_results.csv", index=False)

    # final model on everything
    m, med = fit(df, cols)
    thr = thresholds_from_training(df, cols, args.min_trials, smooth_on)
    joblib.dump({"model": m, "features": cols, "medians": med.to_dict(), "thresholds": thr,
                 "windows_s": WINDOWS, "behaviour": args.behaviour,
                 "smoothing_half_life": SMOOTH_HALF_LIFE if smooth_on else 0,
                 "false_start_rule": {"run": FS_RUN, "weight": FS_WEIGHT}}, args.out)
    print(f"\nSaved {args.out} (trained on all people) and final_results.csv")


if __name__ == "__main__":
    main()
