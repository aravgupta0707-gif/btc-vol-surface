"""
Continuous hourly history since 2020: perpetual candles, funding, and the
Deribit index. Cached to data/research, refreshed incrementally.

The archive gives one chain per month; everything that needs a *path*
between those days -- realized vol over an option's life, a delta hedge run
to expiry, settlement at 08:00, the funding actually paid -- comes from here.

Two endpoint behaviours measured before writing this, both silent:

- `get_funding_rate_history` returns at most 744 rows (31 days) whatever
  range is asked for. A 60-day request comes back as 31 days with no error.
  It is paged in 30-day windows here and the result checked for holes.
- Each funding row carries `index_price`, Deribit's spot index for that
  hour -- the settlement reference. It is used as spot in preference to the
  perpetual, which trades at a small, varying basis to it.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2/public"
CACHE = Path(__file__).resolve().parent / "research"
START = dt.datetime(2020, 2, 1, tzinfo=dt.timezone.utc)
FUNDING_PAGE = dt.timedelta(days=30)
CANDLE_PAGE = dt.timedelta(days=90)


def _ms(t: dt.datetime) -> int:
    return int(t.timestamp() * 1000)


def _get(endpoint: str, params: dict) -> object:
    for attempt in range(4):
        try:
            payload = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60).json()
            if "result" in payload:
                return payload["result"]
            raise RuntimeError(payload.get("error"))
        except (requests.RequestException, ValueError, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _pages(start: dt.datetime, end: dt.datetime, step: dt.timedelta):
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def _funding_page(asset: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    rows = _get("get_funding_rate_history",
                dict(instrument_name=f"{asset}-PERPETUAL",
                     start_timestamp=_ms(a), end_timestamp=_ms(b)))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.floor("h")
    return frame[["time", "index_price", "interest_1h", "interest_8h"]]


def _candle_page(asset: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    res = _get("get_tradingview_chart_data",
               dict(instrument_name=f"{asset}-PERPETUAL", start_timestamp=_ms(a),
                    end_timestamp=_ms(b), resolution="60"))
    if not res.get("ticks"):
        return pd.DataFrame()
    return pd.DataFrame({
        "time": pd.to_datetime(res["ticks"], unit="ms", utc=True),
        "open": res["open"], "high": res["high"], "low": res["low"], "close": res["close"],
    })


def _cached(name: str, fetch_page, asset: str, step: dt.timedelta,
            refresh: bool, offline: bool = False) -> pd.DataFrame:
    path = CACHE / f"{asset.lower()}_{name}_hourly.parquet"
    if offline:
        # Cluster compute nodes have no outbound network. Without this the
        # call below would retry four times and then raise a connection
        # error, which reads as a data problem rather than a missing route.
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent and offline=True: copy it from a machine "
                "that can reach Deribit before running here"
            )
        return pd.read_parquet(path)
    have = pd.read_parquet(path) if path.exists() and not refresh else pd.DataFrame()
    start = START if have.empty else have["time"].max().to_pydatetime()
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    pages = [fetch_page(asset, a, b) for a, b in _pages(start, end, step)]
    frame = pd.concat([have, *[p for p in pages if not p.empty]], ignore_index=True)
    frame = frame.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return frame


def hourly_funding(asset: str, refresh: bool = False,
                   offline: bool = False) -> pd.DataFrame:
    """
    time, index_price, interest_1h, interest_8h -- one row per hour.

    `offline=True` returns the cache untouched and never opens a socket.
    """
    return _cached("funding", _funding_page, asset, FUNDING_PAGE, refresh, offline)


def hourly_candles(asset: str, refresh: bool = False,
                   offline: bool = False) -> pd.DataFrame:
    """
    time, open, high, low, close of the perpetual -- one row per hour.

    **`time` is the candle's open.** Its `close` is the price at time + 1h.
    Read at the stamped hour, the close appears to sit 245-315 bps from the
    index at the 99th percentile; aligned to the hour it actually belongs to
    the basis is ~2 bps median, 16-19 bps at the 99th. Using `close` at its
    stamp is a one-hour lookahead. Studies here use the index instead.
    """
    return _cached("perp", _candle_page, asset, CANDLE_PAGE, refresh, offline)


def missing_hours(frame: pd.DataFrame) -> int:
    """Hours absent between the first and last row: holes a page cap would leave."""
    if frame.empty:
        return 0
    expected = int((frame["time"].max() - frame["time"].min()) / pd.Timedelta(hours=1)) + 1
    return expected - frame["time"].nunique()


def index_at(funding: pd.DataFrame, when) -> float:
    """The index price at the hour containing `when` (the settlement reference)."""
    hour = pd.Timestamp(when).floor("h")
    series = funding.set_index("time")["index_price"]
    return float(series.get(hour, float("nan")))
