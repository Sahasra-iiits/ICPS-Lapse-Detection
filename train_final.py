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


def smooth(d, p, half_life=SMOOTH_HALF_LIFE):
    a = 1 - 0.5 ** (1 / half_life)
    s = pd.Series(p, index=d.index)
    return s.groupby(d.session_id.values).transform(lambda x: x.ewm(alpha=a, adjust=True).mean()).to_numpy()


def fit_predict(tr, te, cols, target):
    med = tr[cols].median()
    m = model().fit(tr[cols].fillna(med).fillna(0), tr[target])
    return m.predict_proba(te[cols].fillna(med).fillna(0))[:, 1], m


def to_units(d, score, target_kind, min_trials):
    """Per-trial rows, or per-minute rows (mean score, drowsy label)."""
    if target_kind == "trial":
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
    return dict(n=len(y), positives=float(y.mean()), roc_auc=roc_auc_score(y, s),
                pr_auc=average_precision_score(y, s), f1=f1_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred), accuracy=accuracy_score(y, pred),
                mcc=matthews_corrcoef(y, pred), chance_f1=2 * y.mean() / (1 + y.mean()),
                majority_class_accuracy=max(y.mean(), 1 - y.mean()))


def threshold_from_training(tr, cols, target_kind, min_trials, smooth_on):
    inner = np.full(len(tr), np.nan)
    k = min(3, tr.participant.nunique())
    if k < 2:
        return 0.5
    for a, b in GroupKFold(k).split(tr, tr.label, tr.participant):
        trn = tr.iloc[a]
        trn = trn if trn.label.nunique() == 2 else tr
        inner[b] = fit_predict(trn, tr.iloc[b], cols, "label")[0]
    if smooth_on:
        inner = smooth(tr, inner)
    y, s, _ = to_units(tr, inner, target_kind, min_trials)
    return best_threshold(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more recording folders")
    ap.add_argument("--test-data", nargs="*", help="evaluate on these folders instead of CV")
    ap.add_argument("--target", choices=["minute", "trial"], default="minute")
    ap.add_argument("--behaviour", action="store_true", help="add recent reaction-time features")
    ap.add_argument("--min-trials", type=int, default=5, help="trials needed for a minute to count")
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--use-augmented", action="store_true", help="train on augmented sessions too")
    ap.add_argument("--out", default="final_model.joblib")
    args = ap.parse_args()

    load = lambda folders: pd.concat([load_folder(f) for f in folders], ignore_index=True)
    df = load(args.data)
    test = load(args.test_data) if args.test_data else None
    allp = pd.concat([df, test], ignore_index=True) if test is not None else df
    allp, cols = add_person_baseline(allp)
    if args.behaviour:
        allp = add_behaviour(allp)
        cols += BEHAVIOUR
    allp = allp.sort_values(["session_id", "t_end_ms"]).reset_index(drop=True)
    tr_mask = allp.session_id.isin(df.session_id)
    if not args.use_augmented:
        allp = allp[(allp.augmented == 0) | ~tr_mask].reset_index(drop=True)
        tr_mask = allp.session_id.isin(df.session_id)
    smooth_on = not args.no_smoothing

    real = allp[tr_mask & (allp.augmented == 0)]
    print(f"\nTarget: {args.target}{' + behaviour' if args.behaviour else ''} | "
          f"{real.participant.nunique()} people, {real.session_id.nunique()} sessions, "
          f"{len(real)} trials, {real.label.mean():.1%} lapses")

    per_person, ys, ss, ps = [], [], [], []
    if test is None:
        people = real.participant.unique()
        if len(people) < 2:
            raise SystemExit("Need at least 2 people for leave-people-out CV.")
        k = min(7, len(people))
        for fold, test_people in enumerate(np.array_split(np.random.default_rng(0).permutation(people), k)):
            tr = allp[tr_mask & ~allp.participant.isin(test_people)]
            te = allp[allp.participant.isin(test_people) & (allp.augmented == 0)]
            if tr.label.nunique() < 2 or te.empty:
                continue
            thr = threshold_from_training(tr, cols, args.target, args.min_trials, smooth_on)
            s = fit_predict(tr, te, cols, "label")[0]
            if smooth_on:
                s = smooth(te, s)
            y, s_u, who = to_units(te, s, args.target, args.min_trials)
            ys.append(y); ss.append(s_u); ps.append(s_u >= thr)
            for p in np.unique(who):
                m = who == p
                if len(np.unique(y[m])) == 2:
                    per_person.append(dict(person=p, n=int(m.sum()), positives=int(y[m].sum()),
                                           roc_auc=roc_auc_score(y[m], s_u[m]),
                                           f1=f1_score(y[m], s_u[m] >= thr)))
        label = f"leave-people-out CV ({k} folds)"
    else:
        tr = allp[tr_mask]
        te = allp[~tr_mask & (allp.augmented == 0)]
        thr = threshold_from_training(tr, cols, args.target, args.min_trials, smooth_on)
        s = fit_predict(tr, te, cols, "label")[0]
        if smooth_on:
            s = smooth(te, s)
        y, s_u, _ = to_units(te, s, args.target, args.min_trials)
        ys, ss, ps = [y], [s_u], [s_u >= thr]
        label = f"trained on {args.data}, tested on {args.test_data}"

    y, s, p = np.concatenate(ys), np.concatenate(ss), np.concatenate(ps)
    if len(np.unique(y)) < 2:
        raise SystemExit("Test units contain only one class - not enough data yet.")
    res = scores_of(y, s, p)
    print(f"\n=== {label}, unit = {'drowsy minute' if args.target == 'minute' else 'trial'} ===")
    for k_, v in res.items():
        print(f"  {k_:24s} {v:.3f}" if isinstance(v, float) else f"  {k_:24s} {v}")
    print("  (judge F1 against chance_f1, accuracy against majority_class_accuracy)")
    pd.DataFrame([dict(setup=label, target=args.target, behaviour=args.behaviour, **res)]).to_csv(
        "final_results.csv", index=False)
    if per_person:
        pp = pd.DataFrame(per_person)
        pp.to_csv("final_per_person.csv", index=False)
        print(f"\n  per person: ROC-AUC {pp.roc_auc.mean():.3f} ± {pp.roc_auc.std():.3f}, "
              f"above chance for {(pp.roc_auc > .5).sum()}/{len(pp)} people")

    # final model on all training data
    tr = allp[tr_mask]
    thr = threshold_from_training(tr, cols, args.target, args.min_trials, smooth_on)
    m = fit_predict(tr, tr.head(1), cols, "label")[1]
    joblib.dump({"model": m, "features": cols, "threshold": thr, "target": args.target,
                 "windows_s": WINDOWS, "behaviour": args.behaviour,
                 "smoothing_half_life": SMOOTH_HALF_LIFE if smooth_on else 0,
                 "medians": tr[cols].median().to_dict(), "cv": res}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
