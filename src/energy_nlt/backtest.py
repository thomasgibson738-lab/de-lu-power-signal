"""Walk-forward backtest — the credibility hook. Same rule as the energy-desk
model: never score a row with thresholds fit on its own future. Rows are ordered
by timestamp, split into sequential folds, and fold k is scored using thresholds
fit only on folds 0..k-1.

The headline number for the artifact is PRECISION (hit-rate): when the signal
fires, how often the market actually did the thing — and the LIFT over the base
rate, which is what proves the signal carries information a coin flip doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .signal import apply_flags, fit_thresholds, label_events


@dataclass
class DirMetrics:
    name: str
    n: int
    n_flagged: int
    base_rate: float
    precision: float  # P(event | flagged) — the hit-rate
    recall: float     # P(flagged | event)
    lift: float       # precision / base_rate

    def line(self) -> str:
        return (
            f"  {self.name:<24} flagged {self.n_flagged:>5} h | "
            f"precision {self.precision * 100:5.1f}% | recall {self.recall * 100:5.1f}% | "
            f"base {self.base_rate * 100:4.1f}% | lift {self.lift:4.1f}x"
        )


def _confusion(flag: np.ndarray, event: np.ndarray, name: str) -> DirMetrics:
    flag = np.asarray(flag, dtype=bool)
    event = np.asarray(event, dtype=bool)
    n = len(event)
    n_flagged = int(flag.sum())
    base = float(event.mean()) if n else 0.0
    precision = float(event[flag].mean()) if n_flagged else float("nan")
    recall = float(flag[event].mean()) if event.sum() else float("nan")
    lift = precision / base if base and not np.isnan(precision) else float("nan")
    return DirMetrics(name, n, n_flagged, base, precision, recall, lift)


def walk_forward(df: pd.DataFrame, folds: int = C.FOLDS) -> dict[str, DirMetrics]:
    """Score both signal directions out-of-sample. `df` must carry nlt + price."""
    df = df.dropna(subset=["nlt", "price"]).sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    size = n // folds
    if size == 0:
        raise ValueError(f"Not enough samples ({n}) for {folds} folds")

    sflag, sevent, cflag, cevent = [], [], [], []
    for k in range(1, folds):
        train = df.iloc[0 : k * size]
        test = df.iloc[k * size : (n if k == folds - 1 else (k + 1) * size)]
        th = fit_thresholds(train["nlt"].to_numpy(), train["price"].to_numpy())
        sf, cf = apply_flags(test, th)
        se, ce = label_events(test, th)
        sflag.append(sf.to_numpy()); sevent.append(se.to_numpy())
        cflag.append(cf.to_numpy()); cevent.append(ce.to_numpy())

    return {
        "spike": _confusion(np.concatenate(sflag), np.concatenate(sevent), "spike-risk (top-decile NLT)"),
        "cheap": _confusion(np.concatenate(cflag), np.concatenate(cevent), "cheap/neg (bottom-decile NLT)"),
    }


def format_report(metrics: dict[str, DirMetrics], label: str) -> str:
    lines = [
        f"Net-Load Tightness backtest — {label}",
        "  (walk-forward, thresholds fit on past folds only)",
        metrics["spike"].line(),
        metrics["cheap"].line(),
    ]
    return "\n".join(lines)
