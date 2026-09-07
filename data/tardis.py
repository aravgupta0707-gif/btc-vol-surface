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

import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
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

# Declaring types up front stops Arrow inferring them per block, and the
# microsecond epochs must stay integers for the bucket arithmetic.
_COLUMN_TYPES = {
    "symbol": pa.string(),
    "type": pa.string(),
    "underlying_index": pa.string(),
    "timestamp": pa.int64(),
    "expiration": pa.int64(),
    **{name: pa.float64() for name in _NUMERIC},
}

# Arrow decompresses and parses one block at a time. Bigger blocks mean
# fewer Python-level round trips; 64 MB of decompressed CSV is a few hundred
# thousand rows, which is a comfortable trade against memory.
_BLOCK_BYTES = 64 << 20


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
) -> pd.DataFrame:
    """
    Stream one archive day and return a downsampled chain snapshot frame.

    Keeps the last quote per (instrument, time bucket), so `bucket_minutes=60`
    yields hourly snapshots comparable to the live collector's cadence. The
    gzip is decompressed on the fly and never written to disk.

    Decompression, parsing, column selection and filtering all happen inside
    Arrow; only the surviving few thousand rows per block cross into pandas.
    """
    bucket_us = bucket_minutes * 60 * 1_000_000
    prefix = f"{currency}-"

    # A day's file spills slightly outside its nominal UTC day -- 2020-03-01
    # opens with a 2020-02-29 23:00 bucket. That bucket is partial, so
    # anything outside the requested day is dropped and "one file, one UTC
    # day" stays true. Free days are never adjacent, so nothing is lost.
    day_start_us = int(
        dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
        * 1_000_000
    )
    day_end_us = day_start_us + 24 * 3600 * 1_000_000

    kept: list[pd.DataFrame] = []
    with requests.get(
        day_url(day), headers=_HEADERS, stream=True, timeout=1800
    ) as response:
        response.raise_for_status()
        stream = pa.CompressedInputStream(pa.PythonFile(response.raw), "gzip")
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=_BLOCK_BYTES),
            convert_options=pacsv.ConvertOptions(
                include_columns=_KEEP, column_types=_COLUMN_TYPES
            ),
        )
        for batch in reader:
            # Filter inside Arrow rather than in Python. A recent day is tens
            # of millions of rows, half of them ETH, and almost everything
            # that survives is a superseded quote in a bucket we already
            # have -- so every row that reaches pandas is one we cannot avoid.
            mask = pc.and_(
                pc.starts_with(batch.column("symbol"), pattern=prefix),
                pc.and_(
                    pc.greater_equal(batch.column("timestamp"), day_start_us),
                    pc.less(batch.column("timestamp"), day_end_us),
                ),
            )
            batch = batch.filter(mask)
            if batch.num_rows == 0:
                continue
            chunk = batch.to_pandas()
            chunk["_bucket"] = chunk["timestamp"] // bucket_us
            # Rows arrive in time order, so within a block the last row for a
            # (symbol, bucket) pair is the freshest quote seen so far.
            kept.append(chunk.drop_duplicates(["symbol", "_bucket"], keep="last"))

    if not kept:
        raise RuntimeError(f"no {currency} rows inside the UTC day {day}")

    # One more pass across block boundaries. Each block is already collapsed
    # to a few thousand rows, so this is cheap. Sorting first makes "last"
    # well defined even where exchange timestamps run slightly out of order
    # against the file's own ordering.
    frame = (
        pd.concat(kept, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(["symbol", "_bucket"], keep="last")
        .drop(columns="_bucket")
    )

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
