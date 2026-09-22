"""Single source of truth for the run config. Same discipline as the
energy-desk model: the knobs live here so the scripts stay clean, and the
values that need a real ENTSO-E pull are marked TODO(token).

Product context: this is the first $0-test artifact for the energy-data
self-serve niche — a Net-Load Tightness (NLT) signal on the German (DE-LU)
bidding zone, off free ENTSO-E data. See
scratchpad/energy-first-artifact-spec-2026-09-15.md.
"""

from __future__ import annotations

import os

# --- Market / zone -------------------------------------------------------
ZONE = "DE_LU"  # entsoe-py country_code for the Germany-Luxembourg bidding zone
ZONE_LABEL = "Germany-Luxembourg (DE-LU)"
SOURCE_TZ = "Europe/Berlin"  # entsoe-py returns tz-aware; we normalise to UTC hourly

# ENTSO-E requires a free security token (email registration on the
# Transparency Platform, then Account Settings → "Generate a new token").
# TODO(token): export ENTSOE_API_TOKEN in your shell before running fetch_data.py.
API_TOKEN = os.environ.get("ENTSOE_API_TOKEN", "")

# --- The computed signal -------------------------------------------------
# net_load = load_forecast - (wind_forecast + solar_forecast)
# NLT      = net_load / available_dispatchable_capacity   (unitless "squeeze")
# available_dispatchable defaults to installed non-intermittent capacity.
# TODO(outages): refine available_dispatchable by subtracting ENTSO-E
# generation-unit unavailability (event-based, so it needs aggregating into an
# MW-offline-per-hour series). The installed-capacity default is the honest v1.
INTERMITTENT_PSR = ("Solar", "Wind Onshore", "Wind Offshore")  # everything else = dispatchable

# Signal thresholds are fit on the TRAINING slice only (no lookahead), as
# quantiles of NLT. Top decile → spike watch; bottom decile → cheap/negative.
NLT_TOP_Q = 0.90
NLT_BOTTOM_Q = 0.10

# Event labels (the truth we score against), also fit on training only.
SPIKE_PRICE_Q = 0.90     # a price "spike" = top-decile day-ahead price (~10% base rate)
NEG_PRICE_CEILING = 0.0  # a "cheap/negative" hour = day-ahead price <= 0 €/MWh

# --- Backtest ------------------------------------------------------------
FOLDS = 5  # walk-forward: fold k scored by thresholds fit on folds 0..k-1

# --- Paths ---------------------------------------------------------------
DATA_DIR = "data"
HOURLY_CSV = f"{DATA_DIR}/de_lu_hourly.csv"      # timestamp, price, load_forecast, wind_forecast, solar_forecast
CAPACITY_CSV = f"{DATA_DIR}/de_lu_capacity.csv"  # year, dispatchable_mw
