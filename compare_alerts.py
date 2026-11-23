"""
How well did the live monitor track real lapses? Compares a PVT-B session
(pvt_b.py) with live_monitor.py run on the same recording.

    python pvt_b.py --participant P01 --out recordings
    python live_monitor.py --source recordings/PVT_..._P01.mp4 --calib 60 --fast --no-sound --no-notify
    python compare_alerts.py --session recordings/PVT_..._P01 --scores scores_....csv --alerts alerts_....csv

For every PVT trial after calibration:
  - zoning score / PERCLOS just before the stimulus -> ROC-AUC vs. lapse
    (0.5 = chance, 1.0 = perfect)
  - was there an attention alert in the 10 s before the stimulus (or before the
    response)? -> lapses caught, and alerts on non-lapse trials
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAPSE_MS = 355
LOOKBACK_S = 10.0
ATTENTION_ALERTS = {"zoning", "drowsy", "dozing"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="path prefix, e.g. recordings/PVT_..._P01")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--alerts", required=True)
    args = ap.parse_args()

    ev = pd.read_csv(args.session + "_events.csv")
    trials = ev[ev.event_type == "pvt"].copy()
    trials["t_s"] = trials.timestamp_ms / 1000
    trials["lapse"] = (trials.rt_ms.isna() | (trials.rt_ms >= LAPSE_MS)).astype(int)
    trials["end_s"] = trials.t_s + trials.rt_ms.fillna(10000) / 1000

    sc = pd.read_csv(args.scores).sort_values("t_s")
    sc = sc[sc.calibrated == 1]
    al = pd.read_csv(args.alerts)
    al = al[al.alert.isin(ATTENTION_ALERTS)]

    if sc.empty:
        raise SystemExit("Monitor never finished calibrating - use a longer recording or smaller --calib.")
    trials = trials[trials.t_s >= sc.t_s.min()]
    if trials.empty:
        raise SystemExit("No PVT trials after calibration.")

    # score just before each stimulus
    m = pd.merge_asof(trials.sort_values("t_s"), sc[["t_s", "zoning_score", "perclos"]],
                      on="t_s", direction="backward")
    m["alerted"] = [
        int(((al.t_s >= t - LOOKBACK_S) & (al.t_s <= e)).any())
        for t, e in zip(m.t_s, m.end_s)
    ]

    n, nl = len(m), int(m.lapse.sum())
    print(f"PVT trials after calibration: {n}, lapses (RT >= {LAPSE_MS} ms or none): {nl} ({nl / n:.0%})")
    if nl == 0:
        print("No lapses in this session - nothing to detect (an alert participant). "
              "Try a session when tired (late evening / after lunch).")
    for col in ("zoning_score", "perclos"):
        ok = m[col].notna()
        if ok.sum() and m.lapse[ok].nunique() == 2:
            print(f"  ROC-AUC {col:13s}: {roc_auc_score(m.lapse[ok], m[col][ok]):.2f}   (0.5 = chance)")
    caught = int(m.alerted[m.lapse == 1].sum())
    false_alerts = int(m.alerted[m.lapse == 0].sum())
    if nl:
        print(f"  Lapses with an alert in the {LOOKBACK_S:.0f} s before: {caught}/{nl} ({caught / nl:.0%})")
    print(f"  Non-lapse trials with an alert:       {false_alerts}/{n - nl} "
          f"({false_alerts / max(n - nl, 1):.0%})")
    print(f"  Attention alerts in total: {len(al)}")
    if nl:
        print("  Mean RT, lapse vs non-lapse trials: "
              f"{m.rt_ms[m.lapse == 1].mean():.0f} / {m.rt_ms[m.lapse == 0].mean():.0f} ms")


if __name__ == "__main__":
    main()
