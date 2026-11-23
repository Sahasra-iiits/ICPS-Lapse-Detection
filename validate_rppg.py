"""
Webcam heart rate (rPPG) vs. the ESP32 finger sensor, on your own recordings.

    python extract_video_features.py --data DIR     # makes <session>_rppg.csv
    python validate_rppg.py --data DIR

For every rPPG estimate (a 10 s window) the ESP32 HR is averaged over the same
window. Reports MAE / RMSE / Pearson r / Bland-Altman bias and limits of
agreement, pooled and per session, plus two plots (scatter, Bland-Altman).

Caveat for the slides: the MAX30100 is itself a noisy reference (it is not a
medical ECG), so disagreement is partly the reference's fault.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from extract_video_features import HR_WIN_S

MIN_SNR = -3.0   # same default as lapse_features.RPPG_MIN_SNR


def pair_session(data_dir, sid):
    base = os.path.join(data_dir, sid)
    rp = pd.read_csv(base + "_rppg.csv")
    fr = pd.read_csv(base + "_data.csv", usecols=["timestamp_ms", "heart_rate"])
    t, hr = fr.timestamp_ms.to_numpy(float), fr.heart_rate.to_numpy(float)
    ref = []
    for te in rp.t_ms.to_numpy(float):
        m = (t > te - HR_WIN_S * 1000) & (t <= te)
        v = hr[m]
        v = v[~np.isnan(v)]
        # need the contact sensor to cover most of the window
        ref.append(v.mean() if m.sum() and len(v) >= 0.8 * m.sum() else np.nan)
    rp["hr_ref"] = ref
    rp["session_id"] = sid
    return rp


def stats(d):
    e = d.hr_rppg - d.hr_ref
    bias, sd = e.mean(), e.std()
    return pd.Series(dict(
        n=len(d),
        mae=e.abs().mean(),
        rmse=np.sqrt((e ** 2).mean()),
        pearson_r=d.hr_rppg.corr(d.hr_ref) if len(d) > 2 else np.nan,
        bias=bias,
        loa_low=bias - 1.96 * sd,
        loa_high=bias + 1.96 * sd,
        within_5bpm=(e.abs() <= 5).mean(),
        within_10bpm=(e.abs() <= 10).mean(),
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-snr", type=float, default=MIN_SNR)
    ap.add_argument("--out", default="rppg_validation")
    args = ap.parse_args()

    sids = [os.path.basename(p)[:-len("_rppg.csv")]
            for p in sorted(glob.glob(os.path.join(args.data, "*_rppg.csv")))]
    if not sids:
        raise SystemExit("No _rppg.csv files — run extract_video_features.py first.")

    allp = pd.concat([pair_session(args.data, s) for s in sids], ignore_index=True)
    both = allp.dropna(subset=["hr_rppg", "hr_ref"])
    good = both[both.rppg_snr >= args.min_snr]

    print(f"rPPG estimates: {len(allp)}, with ESP32 reference: {len(both)}, "
          f"passing SNR >= {args.min_snr} dB: {len(good)} ({len(good) / max(len(both), 1):.0%})")
    if len(good) < 3:
        raise SystemExit("Too few paired estimates.")

    summary = pd.DataFrame({
        "all (no quality filter)": stats(both),
        f"SNR >= {args.min_snr} dB": stats(good),
    }).T
    print("\nPooled:")
    print(summary.round(2).to_string())

    per = good.drop(columns="session_id").groupby(good.session_id).apply(stats)
    print("\nPer session (quality-filtered):")
    print(per.round(2).to_string())

    summary.to_csv(args.out + "_summary.csv")
    per.to_csv(args.out + "_per_session.csv")
    good.to_csv(args.out + "_pairs.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(good.hr_ref, good.hr_rppg, s=8, alpha=0.4)
        lo, hi = np.nanmin([good.hr_ref, good.hr_rppg]), np.nanmax([good.hr_ref, good.hr_rppg])
        ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[0].set_xlabel("ESP32 HR (bpm)")
        ax[0].set_ylabel("Webcam rPPG HR (bpm)")
        s = summary.iloc[1]
        ax[0].set_title(f"MAE {s.mae:.1f} bpm, r = {s.pearson_r:.2f}")

        mean = (good.hr_ref + good.hr_rppg) / 2
        diff = good.hr_rppg - good.hr_ref
        ax[1].scatter(mean, diff, s=8, alpha=0.4)
        for v, ls in ((s.bias, "-"), (s.loa_low, "--"), (s.loa_high, "--")):
            ax[1].axhline(v, color="k", ls=ls, lw=1)
        ax[1].set_xlabel("Mean of the two (bpm)")
        ax[1].set_ylabel("rPPG − ESP32 (bpm)")
        ax[1].set_title(f"Bland-Altman: bias {s.bias:.1f}, LoA [{s.loa_low:.1f}, {s.loa_high:.1f}]")
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=150)
        print(f"\nSaved {args.out}.png")
    except ImportError:
        print("(matplotlib not installed — skipped plots)")


if __name__ == "__main__":
    main()
