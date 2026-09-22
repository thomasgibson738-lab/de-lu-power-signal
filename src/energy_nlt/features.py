"""The computed layer — the part that is NOT a raw mirror of the free feed.

Net-Load Tightness (NLT) is standard power-market economics (net load / residual
load = the "duck curve" quantity) turned into a unitless squeeze ratio:

    net_load = load_forecast - (wind_forecast + solar_forecast)
    NLT      = net_load / available_dispatchable_capacity

All inputs are day-ahead FORECASTS, so NLT is knowable before the market clears —
which is what makes the live 7-day view honest (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_nlt(df: pd.DataFrame, dispatchable_col: str = "dispatchable_mw") -> pd.DataFrame:
    out = df.copy()
    out["net_load"] = out["load_forecast"] - (out["wind_forecast"] + out["solar_forecast"])
    disp = out[dispatchable_col].replace(0, np.nan)
    out["nlt"] = out["net_load"] / disp
    return out
