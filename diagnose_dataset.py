"""
Why is lapse prediction hard on this data? Diagnostics on a lapse_dataset.csv
(written by train_lapse_model.py).

    python diagnose_dataset.py --dataset lapse_dataset.csv --out diagnostics

1. Where do lapses come from?  How much is explained by WHO / WHICH SESSION /
   TIME-ON-TASK alone (no face features at all)?
2. Single-feature ROC-AUC, pooled across people vs. within each person.
3. Variance decomposition: share of each feature's variance explained by the
   participant vs. by the lapse label.
4. Clustering (PCA + k-means): do windows cluster by person or by lapse?
5. Personal models (train/test within each session, earlier -> later trials)
   vs. leave-person-out.
Plots + CSVs go to --out.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lapse_features as lf

warnings.filterwarnings("ignore")


def auc(y, s):
    ok = ~np.isnan(s)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], s[ok])


def loo_mean(df, key):
    """Lapse rate of the same session/person, leaving the trial itself out."""
    g = df.groupby(key).label
    s, n = g.transform("sum"), g.transform("count")
    return ((s - df.label) / (n - 1).clip(lower=1)).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="diagnostics")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df = df[df.get("augmented", 0) == 0].reset_index(drop=True)
    y = df.label.to_numpy()
    feats = [c for c in lf.feature_columns(df) if df[c].notna().mean() >= 0.8 and df[c].nunique() > 1]
    print(f"{len(df)} windows, {y.sum()} lapses ({y.mean():.1%}), "
          f"{df.participant.nunique()} people, {df.session_id.nunique()} sessions, {len(feats)} features")

    # ---------- 1. where do lapses come from ----------
    print("\n[1] Lapse signal WITHOUT any face features (ROC-AUC, 0.5 = chance)")
    per_sess = df.groupby("session_id").label.mean().sort_values(ascending=False)
    top = per_sess.head(6)
    share = df[df.session_id.isin(top.index)].label.sum() / y.sum()
    print(f"  lapse rate per session: median {per_sess.median():.0%}, max {per_sess.max():.0%}; "
          f"top 6 of {len(per_sess)} sessions hold {share:.0%} of all lapses")
    t_in = df.groupby("session_id").t_end_ms.transform(lambda t: (t - t.min()) / max(t.max() - t.min(), 1))
    prev_rt = df.sort_values("t_end_ms").groupby("session_id").rt_ms.shift(1).reindex(df.index)
    oracle = {
        "same SESSION's lapse rate (leave-trial-out)": loo_mean(df, "session_id"),
        "same PERSON's lapse rate (leave-trial-out)": loo_mean(df, "participant"),
        "time-on-task (position in the 10 min)": t_in.to_numpy(float),
        "previous trial's RT": prev_rt.to_numpy(float),
    }
    rows = []
    for k, s in oracle.items():
        a = auc(y, s)
        rows.append(dict(signal=k, roc_auc=a))
        print(f"  {k:48s} {a:.3f}")

    # ---------- 2. single features ----------
    res = []
    for c in feats:
        x = df[c].to_numpy(float)
        within = [auc(g.label.to_numpy(), g[c].to_numpy(float)) for _, g in df.groupby("participant")]
        within = [w for w in within if np.isfinite(w)]
        a = auc(y, x)
        res.append(dict(feature=c, group=lf.group_of(c), pooled_auc=a,
                        pooled_strength=abs(a - 0.5) if np.isfinite(a) else np.nan,
                        within_person_auc=np.mean(within) if within else np.nan,
                        sign_agreement=np.mean([np.sign(w - .5) == np.sign(a - .5) for w in within])
                        if within and np.isfinite(a) else np.nan))
    fr = pd.DataFrame(res).sort_values("pooled_strength", ascending=False)
    fr.to_csv(os.path.join(args.out, "feature_auc.csv"), index=False)
    print("\n[2] Best single features (pooled AUC; within-person mean AUC; "
          "fraction of people where the direction agrees)")
    print(fr.head(12)[["feature", "group", "pooled_auc", "within_person_auc", "sign_agreement"]]
          .round(3).to_string(index=False))

    # ---------- 3. variance decomposition ----------
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[feats]), columns=feats)
    Xs = StandardScaler().fit_transform(X)

    def eta2(v, g):
        m = v.mean()
        return sum(len(v[g == k]) * (v[g == k].mean() - m) ** 2 for k in np.unique(g)) / max(((v - m) ** 2).sum(), 1e-12)
    part = df.participant.to_numpy()
    raw = [c for c in feats if not c.endswith("_z")]
    zc = [c for c in feats if c.endswith("_z")]
    print("\n[3] Share of feature variance explained by WHO the person is vs. by LAPSE (eta-squared)")
    for name, cols in (("raw features", raw), ("per-session z-scored", zc)):
        if not cols:
            continue
        ix = [feats.index(c) for c in cols]
        ep = np.mean([eta2(Xs[:, i], part) for i in ix])
        el = np.mean([eta2(Xs[:, i], y) for i in ix])
        print(f"  {name:22s} person: {ep:6.1%}   lapse: {el:6.2%}")

    # ---------- 4. clustering ----------
    pca = PCA(n_components=10, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    k = df.participant.nunique()
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Z)
    sub = np.random.default_rng(0).choice(len(Z), min(3000, len(Z)), replace=False)
    print(f"\n[4] Clustering (PCA 10 comps = {pca.explained_variance_ratio_.sum():.0%} of variance, k-means)")
    print(f"  k={k} clusters vs PERSON:  adjusted mutual info {adjusted_mutual_info_score(part, km):.3f}")
    print(f"  k={k} clusters vs SESSION: adjusted mutual info {adjusted_mutual_info_score(df.session_id, km):.3f}")
    print(f"  k={k} clusters vs LAPSE:   adjusted mutual info {adjusted_mutual_info_score(y, km):.3f}")
    print(f"  k=2  clusters vs LAPSE:   adjusted mutual info {adjusted_mutual_info_score(y, km2):.3f}")
    print(f"  silhouette by person {silhouette_score(Z[sub], part[sub]):.3f}   "
          f"by lapse {silhouette_score(Z[sub], y[sub]):.3f}   (higher = tighter groups)")
    ct = pd.crosstab(km, y, normalize="index")[1].sort_values(ascending=False)
    print(f"  lapse rate inside the {k} clusters: {ct.min():.0%} .. {ct.max():.0%} "
          f"(overall {y.mean():.0%})")

    # ---------- 5. personal vs cross-person ----------
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))
    cols = [c for c in feats if lf.group_of(c) in {"eye_closure", "gaze", "head"}]
    within_p = []
    for _, g in df.groupby("session_id"):
        g = g.sort_values("t_end_ms")
        n = len(g)
        tr, te = g.iloc[: n // 2], g.iloc[n // 2:]
        if tr.label.nunique() < 2 or te.label.nunique() < 2 or tr.label.sum() < 3:
            continue
        p = model.fit(tr[cols], tr.label).predict_proba(te[cols])[:, 1]
        within_p.append(auc(te.label.to_numpy(), p))
    p_cross = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=min(7, df.participant.nunique())).split(df, y, df.participant):
        p_cross[te] = model.fit(df.iloc[tr][cols], y[tr]).predict_proba(df.iloc[te][cols])[:, 1]
    print("\n[5] Logistic regression on eye/gaze/head features")
    print(f"  leave-PERSON-out (new person):              ROC-AUC {auc(y, p_cross):.3f}")
    print(f"  PERSONAL (1st half of session -> 2nd half): mean ROC-AUC {np.nanmean(within_p):.3f} "
          f"over {len(within_p)} sessions with enough lapses")

    # ---------- plots ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        codes = pd.factorize(part)[0]
        ax[0].scatter(Z[sub, 0], Z[sub, 1], c=codes[sub], cmap="tab20", s=6, alpha=.6)
        ax[0].set_title("PCA of features, coloured by PERSON")
        ax[1].scatter(Z[sub, 0], Z[sub, 1], c=np.where(y[sub] == 1, "#d62728", "#c7c7c7"), s=6, alpha=.6)
        ax[1].set_title("Same PCA, coloured by LAPSE (red)")
        for a in ax[:2]:
            a.set_xlabel("PC1"); a.set_ylabel("PC2")
        per_sess.plot.bar(ax=ax[2], color="#1f77b4")
        ax[2].set_title("Lapse rate per session")
        ax[2].tick_params(axis="x", labelsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "diagnostics.png"), dpi=140)
        print(f"\nSaved {args.out}/diagnostics.png and feature_auc.csv")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
