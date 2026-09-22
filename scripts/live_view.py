"""The forward-looking view for the artifact: take the most recent forecast
window, score it with thresholds fit on everything BEFORE it, and print the
flagged cheap / spike-risk hours. This is the "when is power about to be free or
spiky" table a reader sees.

    python scripts/live_view.py --demo            # last 7 days of synthetic as the stand-in
    python scripts/live_view.py --hours 168       # real data/*.csv

TODO(live): wire the real 7-day-ahead pull — ENTSO-E publishes week-ahead load
and day-ahead wind/solar forecasts; fetch those into the same schema and this
same code flags the upcoming window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from energy_nlt import config as C  # noqa: E402
from energy_nlt import data, features, synthetic  # noqa: E402
from energy_nlt.signal import apply_flags, fit_thresholds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Flagged upcoming windows for DE-LU")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--hours", type=int, default=168, help="size of the 'upcoming' window")
    ap.add_argument("--data", default=C.HOURLY_CSV)
    ap.add_argument("--capacity", default=C.CAPACITY_CSV)
    args = ap.parse_args()

    if args.demo:
        hourly, capacity = synthetic.make(years=1.0)
        df = hourly.copy()
        df["year"] = df["timestamp"].dt.year
        df = df.merge(capacity, on="year", how="left").drop(columns="year")
    else:
        df = data.load_hourly(args.data, args.capacity)

    df = features.compute_nlt(df.sort_values("timestamp").reset_index(drop=True))
    history, upcoming = df.iloc[: -args.hours], df.iloc[-args.hours :]
    if history.empty:
        raise SystemExit("Not enough history before the upcoming window; lower --hours.")

    th = fit_thresholds(history["nlt"].to_numpy(), history["price"].to_numpy())
    spike_flag, cheap_flag = apply_flags(upcoming, th)

    view = upcoming[["timestamp", "nlt"]].copy()
    view["flag"] = pd.Series(
        ["🔺 SPIKE-RISK" if s else "🔻 CHEAP/NEG" if c else "" for s, c in zip(spike_flag, cheap_flag)],
        index=view.index,
    )
    flagged = view[view["flag"] != ""]
    print(f"{C.ZONE_LABEL} — next {args.hours} h — thresholds from {len(history):,} h of history")
    print(f"  spike watch NLT ≥ {th['nlt_top']:.3f} | cheap watch NLT ≤ {th['nlt_bottom']:.3f}")
    if flagged.empty:
        print("  no flagged hours in the window.")
    else:
        for _, r in flagged.iterrows():
            print(f"  {r['timestamp']:%Y-%m-%d %H:%M}  NLT {r['nlt']:.3f}  {r['flag']}")
    if args.demo:
        print("  NOTE: synthetic stand-in for the upcoming window.")


if __name__ == "__main__":
    main()
