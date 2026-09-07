"""
Track B: historical Deribit option chains from the Tardis.dev free archive.

Deribit's public API serves only current state, so the walk-forward backtest
in BUILD_GUIDE.md section 8 has no input from it. Tardis publishes historical
CSV datasets and makes **the first day of each month** free. For
`options_chain` that runs from 2020-03-01, which is ~78 days spread across
six years of regimes -- far better regime coverage than any amount of live
collection can buy in the near term.

    https://datasets.tardis.dev/v1/deribit/options_chain/YYYY/MM/DD/OPTIONS.csv.gz

Two quirks of the endpoint, both found by trial:

- It requires an `Authorization` header to be *present*, but does not check
  it for the free days. Without the header you get a 403.
- It rejects Range requests, so the file has to be taken whole.

Size is the real constraint: one day is ~516 MB gzipped, ~2.1 GB raw, ~10.8M
rows, because it is every tick-level quote update on every instrument. Half
of it is ETH. So this module streams the gzip, filters to BTC, and keeps only
the last quote per instrument per time bucket, never materializing the CSV.
At hourly buckets one day lands at ~20-30k rows and the whole free archive
fits comfortably in a few hundred MB of parquet.

The archive also carries `bid_iv`, `ask_iv` and Deribit's own greeks, which
the live book-summary endpoint does not. Those enable a vol-space spread
filter and a historical check of the Phase 5 greeks.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://datasets.tardis.dev/v1/deribit/options_chain"
HIST_DIR = Path(__file__).resolve().parent / "historical"
FIRST_MONTH = (2020, 3)

# Present but unchecked for the free first-of-month days; a 403 without it.
_HEADERS = {"Authorization": "Bearer FREE"}

_KEEP = [
    "symbol", "timestamp", "type", "strike_price", "expiration",
    "open_interest", "bid_price", "bid_iv", "ask_price", "ask_iv",
    "mark_price", "mark_iv", "underlying_index", "underlying_price",
    "delta", "gamma", "vega", "theta", "rho",
]
_NUMERIC = [
    "strike_price", "open_interest", "bid_price", "bid_iv", "ask_price",
    "ask_iv", "mark_price", "mark_iv", "underlying_price", "delta", "gamma",
    "vega", "theta", "rho",
]


def free_days(until: dt.date | None = None) -> list[dt.date]:
    """Every first-of-month covered by the free tier, oldest first."""
    until = until or dt.date.today()
    days, (year, month) = [], FIRST_MONTH
    while (year, month) <= (until.year, until.month):
        day = dt.date(year, month, 1)
        if day < until:
            days.append(day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return days


def day_url(day: dt.date) -> str:
    return f"{BASE_URL}/{day.year}/{day.month:02d}/{day.day:02d}/OPTIONS.csv.gz"


def load_day(
    day: dt.date,
    bucket_minutes: int = 60,
    currency: str = "BTC",
    chunk_bytes: int = 1 << 20,
) -> pd.DataFrame:
    """
    Stream one archive day and return a downsampled chain snapshot frame.

    Keeps the last quote per (instrument, time bucket), so `bucket_minutes=60`
    yields hourly snapshots comparable to the live collector's cadence. The
    gzip is decompressed on the fly and never written to disk.
    """
    bucket_us = bucket_minutes * 60 * 1_000_000
    prefix = f"{currency}-"
    latest: dict[tuple[str, int], dict] = {}

    with requests.get(
        day_url(day), headers=_HEADERS, stream=True, timeout=600
    ) as response:
        response.raise_for_status()
        stream = gzip.GzipFile(fileobj=response.raw)
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
        for row in reader:
            symbol = row["symbol"]
            if not symbol.startswith(prefix):
                continue
            timestamp = int(row["timestamp"])
            latest[(symbol, timestamp // bucket_us)] = row

    if not latest:
        raise RuntimeError(f"no {currency} rows in the archive for {day}")

    frame = pd.DataFrame(
        [{key: row[key] for key in _KEEP} for row in latest.values()]
    )
    for column in _NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return _to_snapshot_schema(frame, bucket_us)


def _to_snapshot_schema(frame: pd.DataFrame, bucket_us: int) -> pd.DataFrame:
    """
    Rename archive columns onto the live collector's schema so that
    data/clean.py and everything downstream treat both sources identically.

    The snapshot timestamp is snapped to the bucket start rather than kept at
    the last quote's own time, so that all instruments in one bucket share a
    single `as of` -- otherwise time-to-expiry differs by instrument within
    what is meant to be a simultaneous cross-section.
    """
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                (frame["timestamp"] // bucket_us) * bucket_us,
                unit="us", utc=True,
            ),
            "instrument_name": frame["symbol"],
            "strike": frame["strike_price"],
            "expiry_ts": pd.to_datetime(frame["expiration"], unit="us", utc=True),
            "option_type": frame["type"].str.upper().str[0],
            "bid": frame["bid_price"],
            "ask": frame["ask_price"],
            "mark_price": frame["mark_price"],
            "mark_iv": frame["mark_iv"],
            "underlying_price": frame["underlying_price"],
            "underlying_index": frame["underlying_index"],
            "open_interest": frame["open_interest"],
            # Not available from the live endpoint; used for a vol-space
            # spread filter and to validate the Phase 5 greeks historically.
            "bid_iv": frame["bid_iv"],
            "ask_iv": frame["ask_iv"],
            "deribit_delta": frame["delta"],
            "deribit_gamma": frame["gamma"],
            "deribit_vega": frame["vega"],
            "deribit_theta": frame["theta"],
        }
    )
    return out.sort_values(
        ["timestamp", "expiry_ts", "strike", "option_type"]
    ).reset_index(drop=True)


def fetch_free_archive(
    bucket_minutes: int = 60,
    hist_dir: Path = HIST_DIR,
    days: list[dt.date] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Download every free day not already on disk, one parquet per day.

    Each day is a ~516 MB transfer, so the full archive is a long job; it is
    resumable because completed days are skipped.
    """
    hist_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for day in days or free_days():
        path = hist_dir / f"btc_chain_{day:%Y%m%d}.parquet"
        if skip_existing and path.exists():
            written.append(path)
            continue
        try:
            frame = load_day(day, bucket_minutes=bucket_minutes)
        except (requests.RequestException, RuntimeError, OSError) as exc:
            print(f"{day}  SKIPPED: {exc}")
            continue
        frame.to_parquet(path, index=False)
        written.append(path)
        print(
            f"{day}  {len(frame):6d} rows  "
            f"{frame['expiry_ts'].nunique():2d} expiries  -> {path.name}"
        )
    return written


if __name__ == "__main__":
    fetch_free_archive()
