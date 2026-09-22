"""energy_nlt — Net-Load Tightness signal on the DE-LU bidding zone from free
ENTSO-E data. Pipeline: fetch (laptop, token) → data (local CSV) → features
(NLT) → signal (flags) → backtest (walk-forward hit-rate). Run `--demo` for a
zero-setup end-to-end pass on synthetic data.
"""

from __future__ import annotations

__all__ = ["config", "fetch", "data", "synthetic", "features", "signal", "backtest"]
