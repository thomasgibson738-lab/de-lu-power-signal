"""Synthetic DE-LU data so the whole pipeline runs with zero token and zero
network — the same `--demo` discipline as the energy-desk model.

The generator wires price to net load ON PURPOSE (high net load → spikes, deep
renewable surplus → negative prices), so `--demo` shows the pipeline *working*
end to end and a believable hit-rate. It says nothing about the REAL German
market: only a real ENTSO-E pull does that. Clearly synthetic, never shipped as
a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make(years: float = 2.0, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = int(years * 365 * 24)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC").tz_localize(None)
    h = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    season = np.cos(2 * np.pi * (doy - 15) / 365)          # +1 in winter, -1 in summer
    daynight = np.cos(2 * np.pi * (h - 18) / 24)           # peak ~evening

    load = 55000 + 9000 * season + 7000 * daynight + rng.normal(0, 2500, n)
    solar = np.clip((-daynight) * (14000 + 6000 * (-season)) + rng.normal(0, 1500, n), 0, None)
    solar *= (h >= 6) & (h <= 20)
    wind = np.clip(12000 + 9000 * season + rng.normal(0, 6000, n), 0, None)

    dispatchable_mw = 85000.0
    net_load = load - (wind + solar)
    squeeze = net_load / dispatchable_mw  # the same NLT the model will recompute

    # Price rises steeply with the squeeze; deep surplus goes negative; noise + rare shocks.
    price = 15 + 260 * np.clip(squeeze, 0, None) ** 3 - 40 * np.clip(-squeeze + 0.15, 0, None) * 6
    price += rng.normal(0, 12, n)
    price += (rng.random(n) < 0.01) * rng.normal(180, 60, n)  # occasional scarcity shocks

    hourly = pd.DataFrame(
        {
            "timestamp": idx,
            "price": np.round(price, 2),
            "load_forecast": np.round(load, 1),
            "wind_forecast": np.round(wind, 1),
            "solar_forecast": np.round(solar, 1),
        }
    )
    capacity = pd.DataFrame({"year": sorted(set(idx.year)), "dispatchable_mw": dispatchable_mw})
    return hourly, capacity
