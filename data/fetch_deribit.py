"""
Pull live BTC options chain data from Deribit's public API.

Endpoints used:
- public/get_instruments  -> discover the live option chain (all strikes/expiries)
- public/ticker           -> best bid/ask, mark price, mark_iv, Deribit's own greeks
- public/get_index_price  -> the settlement index (correct underlying, not raw spot)

No API key required for public market data.
"""

import requests

BASE_URL = "https://www.deribit.com/api/v2"


def get_instruments(currency: str = "BTC", kind: str = "option") -> list[dict]:
    """
    Return all active option instruments for the given currency.

    TODO: call public/get_instruments, filter to kind='option', expired=False.
    """
    raise NotImplementedError


def get_ticker(instrument_name: str) -> dict:
    """
    Return best_bid_price, best_ask_price, mark_price, mark_iv, greeks,
    underlying_price for a single instrument.

    TODO: call public/ticker?instrument_name=...
    """
    raise NotImplementedError


def get_index_price(currency: str = "BTC") -> float:
    """
    Return the current Deribit index price (correct underlying for
    settlement, delta-hedging, and pricing -- not raw spot from one venue).

    TODO: call public/get_index_price?index_name={currency}_usd
    """
    raise NotImplementedError


def snapshot_chain(currency: str = "BTC") -> "pandas.DataFrame":
    """
    Pull a full snapshot of the current option chain: every live instrument's
    ticker data plus the index price, as one timestamped DataFrame ready to
    write to parquet.

    Columns to include at minimum:
    instrument_name, strike, expiry_ts, option_type, bid, ask, mark_price,
    mark_iv, underlying_price, timestamp

    TODO: implement using get_instruments + get_ticker in a loop
    (or public/get_book_summary_by_currency for fewer round trips).
    """
    raise NotImplementedError
