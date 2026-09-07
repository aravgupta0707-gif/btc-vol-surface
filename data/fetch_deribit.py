"""
Pull live BTC options chain data from Deribit's public API.

Endpoints used:
- public/get_book_summary_by_currency -> the whole chain in ONE call
- public/get_instruments              -> instrument metadata
- public/ticker                       -> per-instrument greeks (validation only)
- public/get_index_price              -> the settlement index

No API key required for public market data.

A note on `snapshot_chain`: the original sketch for this module suggested
looping `public/ticker` over every instrument. That is ~1000 round trips per
snapshot and runs into rate limits. `public/get_book_summary_by_currency`
returns bid, ask, mark_price, mark_iv and underlying_price for the entire
chain in a single request, so that is what the collector uses. `get_ticker`
is kept for the greeks validation in BUILD_GUIDE.md section 6, where it is
called for a handful of instruments rather than all of them.

Prices are stored exactly as Deribit reports them, which for BTC options
means the premium is denominated in BTC, not USD. The conversion to USD
belongs in data/clean.py so that raw snapshots stay faithful to the source.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2"
RAW_DIR = Path(__file__).resolve().parent / "raw"

# The three Deribit currencies that list options. BTC and ETH are the
# inverse, coin-settled books; USDC is the linear book and returns seven
# underlyings (BTC, ETH, SOL, XRP, AVAX, HYPE, TRX) in a single request, so
# it costs one call and yields six more markets.
COLLECT_CURRENCIES = ("BTC", "ETH", "USDC")

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_session = requests.Session()


def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET a public endpoint and return the 'result' payload, with retries."""
    last_error = None
    for attempt in range(retries):
        try:
            response = _session.get(
                f"{BASE_URL}/{endpoint}", params=params or {}, timeout=30
            )
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"Deribit error on {endpoint}: {payload['error']}")
            return payload["result"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"{endpoint} failed after {retries} attempts") from last_error


def parse_instrument_name(name: str) -> dict:
    """
    Split a Deribit option name into its parts.

    'BTC-27DEC26-100000-C' -> currency BTC, expiry 2026-12-27 08:00 UTC,
    strike 100000.0, option_type 'C'. Deribit options expire at 08:00 UTC;
    that time is part of the expiry, not a detail to round away, because a
    few hours matters materially for near-dated time-to-expiry.

    Linear contracts carry the quote asset in the leading field, as in
    'BTC_USDC-25SEP26-88000-C'. That is split out, because the two families
    settle differently and must not be priced the same way -- see
    data/clean.py::to_usd.
    """
    currency, expiry_code, strike, option_type = name.split("-")
    underlying_asset, _, quote_asset = currency.partition("_")
    day = int(expiry_code[:-5])
    month = _MONTHS[expiry_code[-5:-2].upper()]
    year = 2000 + int(expiry_code[-2:])
    expiry = dt.datetime(year, month, day, 8, 0, tzinfo=dt.timezone.utc)
    return {
        "currency": currency,
        "underlying_asset": underlying_asset,
        "quote_asset": quote_asset or underlying_asset,
        "expiry": expiry,
        "strike": float(strike),
        "option_type": option_type,
    }


def get_instruments(currency: str = "BTC", kind: str = "option") -> list[dict]:
    """Return all active (non-expired) option instruments for the currency."""
    return _get(
        "public/get_instruments",
        {"currency": currency, "kind": kind, "expired": "false"},
    )


def get_ticker(instrument_name: str) -> dict:
    """
    Return best_bid_price, best_ask_price, mark_price, mark_iv, greeks and
    underlying_price for a single instrument.

    This is the only endpoint that carries Deribit's own greeks, which is why
    it is kept despite the chain snapshot not using it. Reserve it for the
    delta validation in BUILD_GUIDE.md section 6.
    """
    return _get("public/ticker", {"instrument_name": instrument_name})


def get_index_price(currency: str = "BTC") -> float:
    """
    Return the current Deribit index price.

    This is the settlement reference and the correct underlying for
    delta-hedging -- not raw spot from any single venue. Note that for
    *pricing* each expiry you want that expiry's forward, which the chain
    snapshot carries per-instrument as `underlying_price`.
    """
    result = _get(
        "public/get_index_price", {"index_name": f"{currency.lower()}_usd"}
    )
    return float(result["index_price"])


def snapshot_chain(currency: str = "BTC") -> pd.DataFrame:
    """
    Pull a full snapshot of the current option chain as one timestamped frame.

    Premiums are reported exactly as Deribit quotes them, which differs by
    family: inverse contracts (currency BTC or ETH) quote in the base coin,
    while USDC-quoted contracts quote in dollars. `quote_currency` carries
    the distinction and data/clean.py::to_usd acts on it. `mark_iv` is in
    percent, and `underlying_price` is the forward for that instrument's
    expiry, which removes any need to build a forward curve by hand.

    Note that currency="USDC" returns several underlyings in one response --
    BTC, ETH, SOL, XRP, AVAX, HYPE and TRX -- so `underlying_asset` is the
    column to group by, not the requested currency.
    """
    as_of = dt.datetime.now(dt.timezone.utc)
    summaries = _get(
        "public/get_book_summary_by_currency",
        {"currency": currency, "kind": "option"},
    )
    try:
        index_price = get_index_price(currency)
    except RuntimeError:
        # No usdc_usd index exists; the per-expiry forward on each row is
        # what pricing actually needs, so this is not worth failing over.
        index_price = float("nan")

    rows = []
    for row in summaries:
        name = row["instrument_name"]
        try:
            parsed = parse_instrument_name(name)
        except (ValueError, KeyError):
            continue
        rows.append(
            {
                "timestamp": as_of,
                "instrument_name": name,
                "underlying_asset": parsed["underlying_asset"],
                "quote_currency": row.get("quote_currency") or parsed["quote_asset"],
                "strike": parsed["strike"],
                "expiry_ts": parsed["expiry"],
                "option_type": parsed["option_type"],
                "bid": row.get("bid_price"),
                "ask": row.get("ask_price"),
                "mark_price": row.get("mark_price"),
                "mark_iv": row.get("mark_iv"),
                "underlying_price": row.get("underlying_price"),
                "underlying_index": row.get("underlying_index"),
                "index_price": index_price,
                "open_interest": row.get("open_interest"),
                "volume": row.get("volume"),
                "interest_rate": row.get("interest_rate"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty {currency} option chain snapshot")
    return frame.sort_values(
        ["underlying_asset", "expiry_ts", "strike", "option_type"]
    ).reset_index(drop=True)


def write_snapshot(
    frame: pd.DataFrame, raw_dir: Path = RAW_DIR, currency: str = "BTC"
) -> Path:
    """
    Write one snapshot to a timestamped parquet file and return the path.

    Named `<currency>_chain_<stamp>.parquet`. The prefix matters: globbing
    `*.parquet` across a directory holding several currencies would silently
    blend them into one "chain". Existing BTC files already match this
    pattern, so nothing needs renaming.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = frame["timestamp"].iloc[0].strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{currency.lower()}_chain_{stamp}.parquet"
    frame.to_parquet(path, index=False)
    return path


def main() -> None:
    for currency in COLLECT_CURRENCIES:
        frame = snapshot_chain(currency)
        path = write_snapshot(frame, currency=currency)
        quoted = frame["bid"].notna().sum()
        print(
            f"{frame['timestamp'].iloc[0]:%Y-%m-%d %H:%M:%S}Z  {currency:>5}  "
            f"{len(frame):5d} instruments, "
            f"{frame['underlying_asset'].nunique()} underlyings, "
            f"{quoted} with a bid  ->  {path.name}"
        )


if __name__ == "__main__":
    main()
