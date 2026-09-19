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
of it is ETH. So this module streams the gzip, filters to the currencies
asked for, and keeps only the last quote per instrument per time bucket,
never materializing the CSV.

The transfer is the entire cost here, and one transfer carries every
currency. Extracting BTC and ETH together therefore costs no more network
than BTC alone, while fetching ETH in a later pass costs a second full
download of identical bytes -- so ask for everything wanted up front.
At hourly buckets one day lands at ~20-30k rows and the whole free archive
fits comfortably in a few hundred MB of parquet.

The archive also carries `bid_iv`, `ask_iv` and Deribit's own greeks, which
the live book-summary endpoint does not. Those enable a vol-space spread
filter and a historical check of the Phase 5 greeks.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import requests
import urllib3

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


class _RawBody:
    """
    The response body as a file for Arrow, with urllib3's errors translated.

    Arrow reads `response.raw` directly, which bypasses the layer where
    requests wraps urllib3 exceptions. A connection reset mid-transfer then
    surfaces as a bare `urllib3.exceptions.ProtocolError` -- not a
    RequestException, not an OSError -- and escapes every handler written
    against requests. That ended a run on 2026-09-10 and left the archive
    idle for three days. Translating here keeps the whole module speaking
    one exception vocabulary.
    """

    def __init__(self, raw) -> None:
        self._raw = raw
        self.closed = False

    def read(self, nbytes: int = -1) -> bytes:
        try:
            return self._raw.read(nbytes)
        except urllib3.exceptions.HTTPError as exc:
            raise requests.ConnectionError(exc) from exc

    def close(self) -> None:
        self.closed = True


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """
    Write via a temporary file and rename into place.

    Every resumable loop here treats an existing file as a finished day. A
    process killed mid-write -- a laptop lid, a task time limit -- would
    otherwise leave a truncated parquet that is skipped forever.
    """
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def day_url(day: dt.date) -> str:
    return f"{BASE_URL}/{day.year}/{day.month:02d}/{day.day:02d}/OPTIONS.csv.gz"


def load_day(
    day: dt.date,
    bucket_minutes: int = 60,
    currency: str = "BTC",
) -> pd.DataFrame:
    """
    Stream one archive day and return a downsampled chain snapshot frame.

    Single-currency wrapper over `load_day_multi`.
    """
    frames = load_day_multi(day, bucket_minutes, (currency,))
    if currency not in frames:
        raise RuntimeError(f"no {currency} rows inside the UTC day {day}")
    return frames[currency]


def load_day_multi(
    day: dt.date,
    bucket_minutes: int = 60,
    currencies: Sequence[str] = ("BTC",),
) -> dict[str, pd.DataFrame]:
    """
    Stream one archive day once and return a snapshot frame per currency.

    Keeps the last quote per (instrument, time bucket), so `bucket_minutes=60`
    yields hourly snapshots comparable to the live collector's cadence. The
    gzip is decompressed on the fly and never written to disk.

    Decompression, parsing, column selection and filtering all happen inside
    Arrow; only the surviving few thousand rows per block cross into pandas.

    Currencies are split during that single pass, because the download is the
    whole cost and one download carries all of them.

    A currency with no rows in the day is omitted from the result rather than
    failing the day, so one empty book does not cost the others -- the same
    isolation the live collector uses. Only a day with nothing at all raises.
    """
    bucket_us = bucket_minutes * 60 * 1_000_000

    # A day's file spills slightly outside its nominal UTC day -- 2020-03-01
    # opens with a 2020-02-29 23:00 bucket. That bucket is partial, so
    # anything outside the requested day is dropped and "one file, one UTC
    # day" stays true. Free days are never adjacent, so nothing is lost.
    day_start_us = int(
        dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()
        * 1_000_000
    )
    day_end_us = day_start_us + 24 * 3600 * 1_000_000

    kept: dict[str, list[pd.DataFrame]] = {c: [] for c in currencies}
    with requests.get(
        day_url(day), headers=_HEADERS, stream=True, timeout=1800
    ) as response:
        response.raise_for_status()
        stream = pa.CompressedInputStream(
            pa.PythonFile(_RawBody(response.raw), mode="r"), "gzip"
        )
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=_BLOCK_BYTES),
            convert_options=pacsv.ConvertOptions(
                include_columns=_KEEP, column_types=_COLUMN_TYPES
            ),
        )
        for batch in reader:
            # Filter inside Arrow rather than in Python. A recent day is tens
            # of millions of rows, and almost everything that survives is a
            # superseded quote in a bucket we already have -- so every row
            # that reaches pandas is one we cannot avoid.
            in_day = pc.and_(
                pc.greater_equal(batch.column("timestamp"), day_start_us),
                pc.less(batch.column("timestamp"), day_end_us),
            )
            for currency in currencies:
                sub = batch.filter(
                    pc.and_(
                        pc.starts_with(
                            batch.column("symbol"), pattern=f"{currency}-"
                        ),
                        in_day,
                    )
                )
                if sub.num_rows == 0:
                    continue
                chunk = sub.to_pandas()
                chunk["_bucket"] = chunk["timestamp"] // bucket_us
                # Rows arrive in time order, so within a block the last row
                # for a (symbol, bucket) pair is the freshest seen so far.
                kept[currency].append(
                    chunk.drop_duplicates(["symbol", "_bucket"], keep="last")
                )

    if not any(kept.values()):
        raise RuntimeError(
            f"no {'/'.join(currencies)} rows inside the UTC day {day}"
        )

    # One more pass across block boundaries. Each block is already collapsed
    # to a few thousand rows, so this is cheap. Sorting first makes "last"
    # well defined even where exchange timestamps run slightly out of order
    # against the file's own ordering.
    out: dict[str, pd.DataFrame] = {}
    for currency, chunks in kept.items():
        if not chunks:
            continue
        frame = (
            pd.concat(chunks, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates(["symbol", "_bucket"], keep="last")
            .drop(columns="_bucket")
        )
        out[currency] = _to_snapshot_schema(frame, bucket_us)
    return out


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
    currencies: Sequence[str] = ("BTC", "ETH"),
) -> list[Path]:
    """
    Download every free day not already on disk, one parquet per currency-day.

    Each day is a ~516 MB transfer, so the full archive is a long job; it is
    resumable because completed currency-days are skipped.

    A day is re-downloaded when *any* requested currency is missing, and only
    the missing ones are extracted from that pass. So adding a currency later
    re-transfers every day for it -- the reason the default asks for BTC and
    ETH together.
    """
    hist_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for day in days or free_days():
        paths = {
            c: hist_dir / f"{c.lower()}_chain_{day:%Y%m%d}.parquet"
            for c in currencies
        }
        todo = [c for c in currencies if not (skip_existing and paths[c].exists())]
        if not todo:
            written.extend(paths.values())
            continue
        try:
            frames = load_day_multi(day, bucket_minutes, todo)
        except (requests.RequestException, RuntimeError, OSError) as exc:
            print(f"{day}  SKIPPED: {exc}", flush=True)
            continue
        for currency, frame in frames.items():
            path = paths[currency]
            write_parquet_atomic(frame, path)
            written.append(path)
            print(
                f"{day}  {currency:4s} {len(frame):6d} rows  "
                f"{frame['expiry_ts'].nunique():2d} expiries  -> {path.name}",
                flush=True,
            )
    return written


if __name__ == "__main__":
    fetch_free_archive()
