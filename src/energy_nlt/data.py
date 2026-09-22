"""Local-CSV loader. The web sandbox is firewalled from ENTSO-E, so the
pipeline reads files fetch_data.py wrote on your laptop — same split as the
energy-desk model (fetch on the machine with the token, model anywhere).

Produces one tidy hourly frame with the dispatchable-capacity column already
merged in, ready for features.compute_nlt().
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def load_hourly(path: str = C.HOURLY_CSV, capacity_path: str = C.CAPACITY_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for col in ("price", "load_forecast", "wind_forecast", "solar_forecast"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["wind_forecast"] = df["wind_forecast"].fillna(0.0)
    df["solar_forecast"] = df["solar_forecast"].fillna(0.0)
    df = df.dropna(subset=["timestamp", "price", "load_forecast"])

    cap = pd.read_csv(capacity_path)
    cap["year"] = pd.to_numeric(cap["year"], errors="coerce").astype("Int64")
    df["year"] = df["timestamp"].dt.year.astype("Int64")
    df = df.merge(cap[["year", "dispatchable_mw"]], on="year", how="left")
    # If a year has no capacity row, fall back to the nearest known value.
    df["dispatchable_mw"] = df["dispatchable_mw"].ffill().bfill()

    return df.sort_values("timestamp").drop(columns="year").reset_index(drop=True)
