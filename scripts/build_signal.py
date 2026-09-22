"""Compute NLT and run the walk-forward backtest.

    python scripts/build_signal.py --demo            # zero setup, synthetic data
    python scripts/build_signal.py                   # real data from data/*.csv
    python scripts/build_signal.py --demo --plot out.png --out preds.csv

The printed precision/lift lines are the artifact's headline numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from energy_nlt import config as C  # noqa: E402
from energy_nlt import backtest, data, features, synthetic  # noqa: E402
from energy_nlt.signal import apply_flags, fit_thresholds, label_events  # noqa: E402


def _load(args) -> pd.DataFrame:
    if args.demo:
        hourly, capacity = synthetic.make(years=args.years)
        df = hourly.copy()
        df["year"] = df["timestamp"].dt.year
        df = df.merge(capacity, on="year", how="left").drop(columns="year")
        df["dispatchable_mw"] = df["dispatchable_mw"].ffill().bfill()
        return df.sort_values("timestamp").reset_index(drop=True)
    return data.load_hourly(args.data, args.capacity)


def main() -> None:
    ap = argparse.ArgumentParser(description="Net-Load Tightness signal + backtest for DE-LU")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no token/network)")
    ap.add_argument("--years", type=float, default=2.0, help="years of synthetic data (--demo)")
    ap.add_argument("--data", default=C.HOURLY_CSV, help="hourly CSV path")
    ap.add_argument("--capacity", default=C.CAPACITY_CSV, help="capacity CSV path")
    ap.add_argument("--folds", type=int, default=C.FOLDS)
    ap.add_argument("--plot", default=None, help="save a price-vs-NLT scatter PNG here")
    ap.add_argument("--out", default=None, help="write per-hour NLT + flags CSV here")
    args = ap.parse_args()

    df = features.compute_nlt(_load(args))
    label = f"{C.ZONE_LABEL} — {'SYNTHETIC demo' if args.demo else 'ENTSO-E'} — {len(df):,} h"

    metrics = backtest.walk_forward(df, folds=args.folds)
    print(backtest.format_report(metrics, label))
    if args.demo:
        print("  NOTE: synthetic data — proves the pipeline, not the real German market.")

    if args.out or args.plot:
        th = fit_thresholds(df["nlt"].to_numpy(), df["price"].to_numpy())
        sflag, cflag = apply_flags(df, th)
        sevent, cevent = label_events(df, th)
        if args.out:
            df.assign(spike_flag=sflag, cheap_flag=cflag,
                      spike_event=sevent, cheap_event=cevent).to_csv(args.out, index=False)
            print(f"  wrote {args.out}")
        if args.plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("  matplotlib not installed; skipping --plot")
            else:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df["nlt"], df["price"], s=4, alpha=0.25)
                ax.axvline(th["nlt_top"], color="crimson", ls="--", lw=1, label="top-decile NLT (spike watch)")
                ax.axvline(th["nlt_bottom"], color="seagreen", ls="--", lw=1, label="bottom-decile NLT (cheap watch)")
                ax.set_xlabel("Net-Load Tightness"); ax.set_ylabel("Day-ahead price (€/MWh)")
                ax.set_title(f"NLT vs price — {C.ZONE_LABEL}"); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(args.plot, dpi=120)
                print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
