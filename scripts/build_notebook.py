"""Generate the publishable notebook from source, so the .ipynb stays a build
artifact (regenerable, clean diffs) instead of hand-edited JSON.

    python scripts/build_notebook.py          # writes notebook/de_lu_nlt_backtest.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebook/de_lu_nlt_backtest.ipynb

The notebook itself auto-detects demo vs real data (DEMO flag in the setup cell),
so the same file runs now on synthetic data and publishes for real once you fetch.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebook" / "de_lu_nlt_backtest.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
    md(
        """
# Predicting cheap and spiky power hours in Germany from free ENTSO-E data

**What this is.** A tiny, reproducible model that turns two *free* forecast feeds
(load, wind + solar) into one forward-looking index — **Net-Load Tightness (NLT)** —
and flags the hours most likely to see a **price spike** or **cheap / negative**
prices in the German (DE-LU) day-ahead market. Then it **backtests** those flags
out-of-sample, so the numbers below are earned, not asserted.

**Who it's for.** Anyone modelling DE-LU: traders, battery/EV optimisers, energy-data
folks who currently wrangle ENTSO-E by hand. If it's useful, there's a one-line ask at
the bottom.

**Read the table, not the prose.** The headline is the *precision* (hit-rate when the
signal fires) and the *lift* over the base rate, on data the model never trained on.
"""
    ),
    md(
        """
> ⚠️ **Honesty banner.** If the cell below prints `SYNTHETIC demo data`, every number and
> chart here is from a **synthetic generator that wires price to net load on purpose** — it
> proves the pipeline runs, and says *nothing* about the real German market. The real
> figures appear only after you fetch ENTSO-E data (`scripts/fetch_data.py`) and set
> `DEMO = False`. Don't publish synthetic numbers as if they were real.
"""
    ),
    code(
        """
# --- setup: find the package whether run from repo root or notebook/ ---
import sys, pathlib
here = pathlib.Path.cwd()
for base in [here, *here.parents]:
    if (base / "src" / "energy_nlt").is_dir():
        sys.path.insert(0, str(base / "src")); break

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energy_nlt import config as C, features, backtest, synthetic, data as data_mod
from energy_nlt.signal import fit_thresholds, apply_flags, label_events

# Okabe-Ito colours (colourblind-safe): spike = vermillion, cheap = blue.
SPIKE, CHEAP = "#D55E00", "#0072B2"
"""
    ),
    code(
        """
DEMO = True   # ← set False after: export ENTSOE_API_TOKEN=... && python scripts/fetch_data.py ...

if DEMO:
    hourly, capacity = synthetic.make(years=3.0)
    df = hourly.copy()
    df["year"] = df["timestamp"].dt.year
    df = df.merge(capacity, on="year", how="left").drop(columns="year")
else:
    df = data_mod.load_hourly()

df = features.compute_nlt(df.sort_values("timestamp").reset_index(drop=True))
SOURCE = "SYNTHETIC demo data" if DEMO else "real ENTSO-E data"
span = f"{df['timestamp'].min():%Y-%m-%d} → {df['timestamp'].max():%Y-%m-%d}"
print(f"{C.ZONE_LABEL} · {SOURCE} · {len(df):,} hours · {span}")
if DEMO:
    print("⚠️  SYNTHETIC — pipeline proof only, not the real market.")
"""
    ),
    md(
        """
## The idea in three lines

```
net_load = load_forecast − (wind_forecast + solar_forecast)
NLT      = net_load / available_dispatchable_capacity
```

**Net load** (a.k.a. residual load, the "duck curve" quantity) is what thermal/hydro
plants must actually cover once renewables are in. Divide it by the dispatchable
capacity that *could* cover it and you get a unitless **squeeze**: near 0 the grid is
awash with cheap renewables; above ~1 it's scraping the top of the merit order, where
prices spike. Every input is a **day-ahead forecast**, so NLT is knowable *before* the
market clears — no lookahead.
"""
    ),
    md(
        """
## Does it actually work? (walk-forward, out-of-sample)

Thresholds (the NLT deciles that trigger a flag, and the price level that counts as a
spike) are fit on **past** data only; each fold is scored on the **next** slice it never
saw. Precision = when the flag fires, how often the market really did it. Lift =
precision ÷ base rate — the factor by which the signal beats a coin weighted to the base
rate. **Lift ≈ 1× means no signal; be willing to read that and stop.**
"""
    ),
    code(
        """
m = backtest.walk_forward(df)
print(backtest.format_report(m, f"{C.ZONE_LABEL} — {SOURCE}"))

rows = [
    ("🔺 spike-risk (top-decile NLT)", m["spike"]),
    ("🔻 cheap/negative (bottom-decile NLT)", m["cheap"]),
]
tbl = pd.DataFrame(
    {
        "flagged (h)": [d.n_flagged for _, d in rows],
        "precision": [f"{d.precision*100:.1f}%" for _, d in rows],
        "recall": [f"{d.recall*100:.1f}%" for _, d in rows],
        "base rate": [f"{d.base_rate*100:.1f}%" for _, d in rows],
        "lift": [f"{d.lift:.1f}×" for _, d in rows],
    },
    index=[name for name, _ in rows],
)
tbl
"""
    ),
    md(
        """
## Why it works: tighter grid, higher price

The scatter is every hour: NLT on the x-axis, the day-ahead price it cleared at on the y.
The dashed lines are the decile thresholds the signal uses. The relationship is the whole
thesis — you're not selling the free data, you're selling *this shape*, computed and
maintained.
"""
    ),
    code(
        """
th = fit_thresholds(df["nlt"].to_numpy(), df["price"].to_numpy())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["nlt"], df["price"], s=5, alpha=0.15, color="#555")
ax.axvline(th["nlt_top"], color=SPIKE, ls="--", lw=1.5, label="top-decile NLT → spike watch")
ax.axvline(th["nlt_bottom"], color=CHEAP, ls="--", lw=1.5, label="bottom-decile NLT → cheap watch")
ax.axhline(th["spike_price"], color=SPIKE, ls=":", lw=1, alpha=0.7)
ax.axhline(C.NEG_PRICE_CEILING, color=CHEAP, ls=":", lw=1, alpha=0.7)
ax.set_xlabel("Net-Load Tightness (day-ahead forecast)")
ax.set_ylabel("Day-ahead price (€/MWh)")
ax.set_title(f"Tighter grid → higher price · {C.ZONE_LABEL} · {SOURCE}")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); plt.show()
"""
    ),
    code(
        """
# Monotonicity check: spike rate and negative rate should climb / fall across NLT deciles.
d = df.dropna(subset=["nlt"]).copy()
d["decile"] = pd.qcut(d["nlt"], 10, labels=False, duplicates="drop")
d["is_spike"] = (d["price"] >= th["spike_price"]).astype(float)
d["is_neg"] = (d["price"] <= C.NEG_PRICE_CEILING).astype(float)
by = d.groupby("decile")[["is_spike", "is_neg"]].mean() * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar(by.index, by["is_spike"], color=SPIKE)
a1.set_title("Price-spike rate by NLT decile"); a1.set_xlabel("NLT decile (0 loosest → 9 tightest)")
a1.set_ylabel("% of hours a spike")
a2.bar(by.index, by["is_neg"], color=CHEAP)
a2.set_title("Cheap / negative rate by NLT decile"); a2.set_xlabel("NLT decile")
a2.set_ylabel("% of hours ≤ 0 €/MWh")
fig.suptitle(f"{C.ZONE_LABEL} · {SOURCE}", fontsize=10)
fig.tight_layout(); plt.show()
"""
    ),
    md(
        """
## The forward view: which upcoming hours are flagged

Same signal, pointed at the most recent window, using thresholds fit only on the history
*before* it. In production this window is the next 7 days of published forecasts; here it's
the tail of the dataset as a stand-in.
"""
    ),
    code(
        """
HOURS = 168
hist, upcoming = df.iloc[:-HOURS], df.iloc[-HOURS:]
thL = fit_thresholds(hist["nlt"].to_numpy(), hist["price"].to_numpy())
sflag, cflag = apply_flags(upcoming, thL)
flags = ["🔺 spike-risk" if s else "🔻 cheap/neg" if c else "" for s, c in zip(sflag, cflag)]
live = upcoming[["timestamp", "nlt"]].assign(flag=flags)
live[live["flag"] != ""].reset_index(drop=True).head(30)
"""
    ),
    md(
        """
## What this doesn't do (so you can trust what it does)

- **Forecast-limited.** NLT rides on published load / wind / solar forecasts; when those miss, so does it. That's honest and it's the point — it's a *forward* signal, not hindsight.
- **`available_dispatchable` is v1.** It uses installed non-intermittent capacity. Subtracting real generation-unit outages (ENTSO-E unavailability) is the obvious next refinement.
- **One zone.** DE-LU only. The same schema extends to the other ENTSO-E bidding zones without a rewrite.
- **Not trading advice.** It's a computed data signal with a measured hit-rate, nothing more.
"""
    ),
    md(
        """
## If this is useful to you

I'm turning this into a **clean, maintained API** — normalised ENTSO-E data plus computed
signals like NLT — so you don't rebuild the pipe. Want it for your bidding zone, or a
different computed layer?

**→ Leave your email: `[WAITLIST LINK]`**  ·  code: `[REPO LINK]`

If it's not useful, tell me why — that's worth as much as a signup.
"""
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
