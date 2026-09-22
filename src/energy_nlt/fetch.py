"""ENTSO-E pull → local CSVs. Runs on YOUR laptop (with a token), not in the
web sandbox (which is firewalled from the ENTSO-E servers, same story as EMI in
the energy-desk model).

We stand on `entsoe-py` for ingestion on purpose: it already solves the XML /
security-token / EIC-code pain, so our effort goes into the computed layer, not
re-scraping a free feed. Everything is resampled to a clean hourly UTC index so
the rest of the pipeline never has to think about 15-minute MTUs.
"""

from __future__ import annotations

import pandas as pd

from . import config as C


def _client(token: str):
    try:
        from entsoe import EntsoePandasClient
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit(
            "entsoe-py not installed. Run: pip install entsoe-py\n"
            "Then export ENTSOE_API_TOKEN=<your free token>."
        ) from e
    if not token:
        raise SystemExit(
            "No ENTSO-E token. Register (free) on the Transparency Platform, "
            "generate a token, then: export ENTSOE_API_TOKEN=<token>."
        )
    return EntsoePandasClient(api_key=token)


def _to_hourly(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Normalise any ENTSO-E return to a tz-naive UTC hourly index (mean over
    sub-hourly steps). DE load/generation is often quarter-hourly; prices moved
    to 15-min in 2025 — hourly means align them all."""
    obj = obj.copy()
    obj.index = pd.to_datetime(obj.index, utc=True)
    obj = obj.resample("1h").mean()
    obj.index = obj.index.tz_convert("UTC").tz_localize(None)
    return obj


def _pick(df: pd.DataFrame, names) -> pd.Series:
    """Sum the columns in `names` that actually exist (ENTSO-E omits absent
    production types), returning 0.0 where none are present."""
    cols = [c for c in df.columns if c in names]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def fetch(start: str, end: str, token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull the five inputs for DE-LU between `start` and `end` (YYYY-MM-DD) and
    return (hourly_df, capacity_df). Writes nothing — the CLI does the I/O."""
    token = C.API_TOKEN if token is None else token
    client = _client(token)
    s = pd.Timestamp(start, tz=C.SOURCE_TZ)
    e = pd.Timestamp(end, tz=C.SOURCE_TZ)

    price = _to_hourly(client.query_day_ahead_prices(C.ZONE, start=s, end=e)).rename("price")
    load = client.query_load_forecast(C.ZONE, start=s, end=e)
    load_fc = _to_hourly(load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load).rename("load_forecast")

    ws = _to_hourly(client.query_wind_and_solar_forecast(C.ZONE, start=s, end=e))
    wind_fc = _pick(ws, ("Wind Onshore", "Wind Offshore")).rename("wind_forecast")
    solar_fc = _pick(ws, ("Solar",)).rename("solar_forecast")

    hourly = (
        pd.concat([price, load_fc, wind_fc, solar_fc], axis=1)
        .dropna(subset=["price", "load_forecast"])
        .rename_axis("timestamp")
        .reset_index()
    )

    # Installed capacity is a yearly table; sum the non-intermittent types.
    cap_raw = client.query_installed_generation_capacity(C.ZONE, start=s, end=e)
    disp = cap_raw.drop(columns=[c for c in cap_raw.columns if c in C.INTERMITTENT_PSR], errors="ignore")
    capacity = pd.DataFrame(
        {
            "year": pd.to_datetime(cap_raw.index).year,
            "dispatchable_mw": disp.sum(axis=1).values,
        }
    ).drop_duplicates("year")

    return hourly, capacity
