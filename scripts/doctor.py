"""Preflight check. Run this before (and after) the ENTSO-E fetch so problems
surface loudly and early, instead of three minutes into a pull or deep in the
backtest.

    python scripts/doctor.py

It auto-detects where you are in the flow:
  - no data yet  -> checks deps + token, tells you you're clear to fetch
  - data present -> validates the pulled CSVs, tells you you're clear to run build_signal

Exit code 0 = ready for the next step, 1 = something needs fixing.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_nlt import config as C  # noqa: E402


class Doctor:
    def __init__(self) -> None:
        self.fails = 0
        self.warns = 0

    def _p(self, tag: str, label: str, detail: str = "") -> None:
        print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))

    def ok(self, label: str, detail: str = "") -> None:
        self._p("PASS", label, detail)

    def warn(self, label: str, detail: str = "") -> None:
        self.warns += 1
        self._p("WARN", label, detail)

    def fail(self, label: str, detail: str = "") -> None:
        self.fails += 1
        self._p("FAIL", label, detail)

    def info(self, label: str, detail: str = "") -> None:
        self._p("··  ", label, detail)


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def check_env(d: Doctor) -> bool:
    print("Environment")
    py_ok = sys.version_info >= (3, 10)
    (d.ok if py_ok else d.warn)(
        f"Python {sys.version_info.major}.{sys.version_info.minor}",
        "" if py_ok else "3.10+ recommended",
    )
    have_np, have_pd = _has("numpy"), _has("pandas")
    d.ok("numpy") if have_np else d.fail("numpy missing", "pip install numpy")
    d.ok("pandas") if have_pd else d.fail("pandas missing", "pip install pandas")
    if _has("entsoe"):
        d.ok("entsoe-py")
    else:
        d.warn("entsoe-py missing", "needed for fetch_data.py: pip install entsoe-py")
    return have_np and have_pd


def check_token(d: Doctor) -> None:
    print("\nENTSO-E token")
    tok = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not tok:
        d.warn("ENTSOE_API_TOKEN not set", "export it before fetching; not needed to validate existing data")
    elif len(tok) < 20 or " " in tok:
        d.warn("ENTSOE_API_TOKEN looks off", "expected a long token with no spaces")
    else:
        d.ok("ENTSOE_API_TOKEN set", f"{tok[:4]}…{tok[-4:]}")


def check_data(d: Doctor, have_pd: bool) -> str:
    """Returns one of: 'no-data', 'bad-data', 'ready'."""
    print("\nData")
    hourly = ROOT / C.HOURLY_CSV
    cap = ROOT / C.CAPACITY_CSV
    if not hourly.exists():
        d.info("no hourly CSV yet", f"expected {C.HOURLY_CSV} — run fetch_data.py")
        return "no-data"
    if not have_pd:
        d.fail("cannot validate data", "pandas missing")
        return "bad-data"

    import pandas as pd
    from energy_nlt import data as data_mod, features

    if not cap.exists():
        d.fail("capacity CSV missing", f"expected {C.CAPACITY_CSV} alongside the hourly file")
        return "bad-data"

    try:
        df = data_mod.load_hourly(str(hourly), str(cap))
    except Exception as e:  # noqa: BLE001
        d.fail("loader raised", f"{type(e).__name__}: {e}")
        return "bad-data"

    required = ["timestamp", "price", "load_forecast", "wind_forecast", "solar_forecast", "dispatchable_mw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        d.fail("missing columns", ", ".join(missing))
        return "bad-data"
    d.ok("columns present", ", ".join(required))

    n = len(df)
    if n < C.FOLDS * 10:
        d.fail("too few rows", f"{n} rows, need at least {C.FOLDS * 10} for {C.FOLDS} folds")
        return "bad-data"
    (d.ok if n >= 8760 else d.warn)(
        f"{n:,} hourly rows", "" if n >= 8760 else "under ~1 year; backtest will be thin"
    )

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        d.warn("some timestamps unparseable", f"{int(ts.isna().sum())} rows")
    d.ok("date span", f"{ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}")
    dupes = int(ts.duplicated().sum())
    if dupes:
        d.warn("duplicate timestamps", f"{dupes} rows (hourly means should have de-duped)")

    for col in ("price", "load_forecast"):
        na = int(df[col].isna().sum())
        if df[col].isna().all():
            d.fail(f"{col} all NaN")
            return "bad-data"
        (d.ok if na == 0 else d.warn)(f"{col} populated", "" if na == 0 else f"{na} NaN rows dropped downstream")

    disp = df["dispatchable_mw"]
    if disp.isna().any() or (disp <= 0).any():
        d.fail("dispatchable_mw invalid", "some rows are NaN or <= 0; capacity merge failed")
        return "bad-data"
    d.ok("dispatchable_mw valid", f"{disp.min():,.0f} to {disp.max():,.0f} MW")

    # Light NLT sanity (not the backtest — just proves the computed layer runs).
    nlt = features.compute_nlt(df)["nlt"]
    finite = nlt.notna().mean()
    if finite < 0.9:
        d.warn("NLT mostly missing", f"only {finite:.0%} of rows finite")
    else:
        d.ok("NLT computes", f"range {nlt.min():.2f} to {nlt.max():.2f}, median {nlt.median():.2f}")
    neg = (df["price"] <= C.NEG_PRICE_CEILING).mean()
    d.info("negative-price share", f"{neg:.1%} of hours <= {C.NEG_PRICE_CEILING:g} EUR/MWh")

    return "ready"


def main() -> None:
    print(f"energy-nlt-signal doctor — {C.ZONE_LABEL}\n")
    d = Doctor()
    deps_ok = check_env(d)
    check_token(d)
    state = check_data(d, deps_ok)

    print("\n" + "-" * 56)
    if d.fails:
        print(f"NOT READY: {d.fails} failure(s), {d.warns} warning(s). Fix the FAILs above.")
        sys.exit(1)

    tok = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if state == "no-data":
        if tok and deps_ok and _has("entsoe"):
            print("READY TO FETCH: python scripts/fetch_data.py --start 2023-01-01 --end 2026-01-01")
        else:
            print("ALMOST: set ENTSOE_API_TOKEN and `pip install entsoe-py`, then fetch.")
    elif state == "ready":
        print(f"READY: data looks good ({d.warns} warning(s)). Next: python scripts/build_signal.py")
    print("-" * 56)
    sys.exit(0)


if __name__ == "__main__":
    main()
