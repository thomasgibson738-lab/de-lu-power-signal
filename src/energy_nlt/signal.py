"""Turn the NLT index into two decision-ready flags, plus the truth labels the
backtest scores them against. Thresholds are always fit on a TRAINING slice and
passed in — this module never peeks at the data it is scoring (the backtest owns
the split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fit_thresholds(nlt_train: np.ndarray, price_train: np.ndarray) -> dict:
    """Fit signal + label thresholds on training data only (no lookahead)."""
    return {
        "nlt_top": float(np.nanquantile(nlt_train, C.NLT_TOP_Q)),
        "nlt_bottom": float(np.nanquantile(nlt_train, C.NLT_BOTTOM_Q)),
        "spike_price": float(np.nanquantile(price_train, C.SPIKE_PRICE_Q)),
    }


def apply_flags(df: pd.DataFrame, th: dict) -> tuple[pd.Series, pd.Series]:
    """The signal: what NLT predicts. Top-decile NLT → spike watch;
    bottom-decile NLT → cheap/negative watch."""
    spike_flag = df["nlt"] >= th["nlt_top"]
    cheap_flag = df["nlt"] <= th["nlt_bottom"]
    return spike_flag, cheap_flag


def label_events(df: pd.DataFrame, th: dict) -> tuple[pd.Series, pd.Series]:
    """The truth: what the market actually did. Spike = top-decile price;
    cheap = price at or below the negative-price ceiling (default 0 €/MWh)."""
    spike_event = df["price"] >= th["spike_price"]
    cheap_event = df["price"] <= C.NEG_PRICE_CEILING
    return spike_event, cheap_event
