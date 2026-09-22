# DE-LU Net-Load Tightness

A small, reproducible signal for the German (DE-LU) day-ahead electricity market, built from
**free ENTSO-E data**. It turns two public forecast feeds (load, and wind + solar) into one
forward-looking index, **Net-Load Tightness (NLT)**, and flags the hours most likely to see a
**price spike** or **cheap / negative** prices. Then it backtests those flags walk-forward, out
of sample.

## Results

Real DE-LU data, **31 Dec 2022 to 31 Dec 2025 (26,300 hourly rows)**, walk-forward with
thresholds fit only on past folds:

| Signal | Hit rate when it fires | Base rate | Lift |
|---|---|---|---|
| Spike-risk (top-decile NLT) | **57.5%** | 7.8% | **7.4x** |
| Cheap / negative (bottom-decile NLT) | **44.7%** | 6.3% | **7.1x** |

The cheap/negative flag also catches **92% of all negative-price hours** (recall). Full method,
charts, and the forward view are in [`notebook/de_lu_nlt_backtest.ipynb`](notebook/de_lu_nlt_backtest.ipynb).

Lift is the point: a value near 1x would mean the signal carries no information. Both directions
are ~7x on data the model never trained on.

## The idea

```
net_load = load_forecast - (wind_forecast + solar_forecast)
NLT      = net_load / available_dispatchable_capacity
```

Net load (a.k.a. residual load, the "duck curve" quantity) is what dispatchable plants must cover
once renewables are in. Divide it by the dispatchable capacity available to cover it and you get a
unitless **squeeze**: near zero the grid is awash with cheap renewables (prices low or negative);
high, and you are at the top of the merit order, where prices spike. Every input is a **day-ahead
forecast**, so NLT is knowable before the market clears (no lookahead).

## Run it

Zero setup, on synthetic data (no token, no network):

```bash
pip install numpy pandas
python scripts/build_signal.py --demo
```

Real data (free ENTSO-E token from the [Transparency Platform](https://transparency.entsoe.eu/)):

```bash
pip install entsoe-py
export ENTSOE_API_TOKEN=<your token>
python scripts/doctor.py                                       # preflight
python scripts/fetch_data.py --start 2023-01-01 --end 2026-01-01
python scripts/doctor.py                                       # validate the pull
python scripts/build_signal.py                                 # the backtest
python scripts/live_view.py                                    # flagged upcoming hours
```

The notebook auto-detects demo vs real data (a `DEMO` flag in the setup cell).

## Layout

```
src/energy_nlt/
  config.py     # zone, thresholds, paths, token
  fetch.py      # ENTSO-E -> local CSV via entsoe-py
  data.py       # local-CSV loader
  synthetic.py  # --demo generator
  features.py   # net_load, NLT
  signal.py     # decile flags + event labels (train-fit only)
  backtest.py   # walk-forward precision / recall / lift
scripts/        # fetch_data - build_signal - live_view - doctor - build_notebook
notebook/       # the reproducible write-up
```

## Limitations

- **Forecast-limited.** NLT rides on published load / wind / solar forecasts, so it inherits their
  error. That is the honest cost of being a forward signal.
- **`available_dispatchable` is v1:** installed non-intermittent capacity, not yet net of real
  generation-unit outages. That is the obvious next refinement.
- **One bidding zone (DE-LU).** The same schema extends to the other ENTSO-E zones without a rewrite.
- **Not trading advice.** It is a computed data signal with a measured hit rate, nothing more.

## License

MIT, see [LICENSE](LICENSE).
