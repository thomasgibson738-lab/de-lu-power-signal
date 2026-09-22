"""Pull DE-LU history from ENTSO-E to local CSVs. Run this on YOUR laptop with a
free token; the web sandbox can't reach ENTSO-E.

    export ENTSOE_API_TOKEN=<your token>
    python scripts/fetch_data.py --start 2023-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_nlt import config as C  # noqa: E402
from energy_nlt import fetch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch DE-LU inputs from ENTSO-E")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--out-dir", default=C.DATA_DIR)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    hourly, capacity = fetch.fetch(args.start, args.end)

    hourly_path = f"{args.out_dir}/de_lu_hourly.csv"
    cap_path = f"{args.out_dir}/de_lu_capacity.csv"
    hourly.to_csv(hourly_path, index=False)
    capacity.to_csv(cap_path, index=False)
    print(f"wrote {hourly_path} ({len(hourly):,} rows) and {cap_path} ({len(capacity)} years)")


if __name__ == "__main__":
    main()
