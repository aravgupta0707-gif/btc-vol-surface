"""
Clean raw option chain snapshots into something safe to invert for
implied vol.

Rules (see BUILD_GUIDE.md section 1.2):
- mid = (bid + ask) / 2
- drop quotes where spread > MAX_SPREAD_PCT of mid
- drop zero-bid quotes entirely
- compute time-to-expiry in years using exact UTC timestamps
  (Deribit expiries are 08:00 UTC)

Two rules the build guide does not mention, both of which matter:

1. Deribit quotes BTC option premiums in BTC, not USD. Strikes are in USD,
   so the premium has to be converted before it meets Black-76 or the
   inversion will not even bracket. The correct multiplier is that expiry's
   forward (`underlying_price`), not the spot index. Checked against
   Deribit's own `mark_iv` over a live chain: converting with the forward
   reproduces their vol to a median of 0.015 vol points, while converting
   with the index is ~30x worse and leaves ~13% of quotes uninvertible.

2. Every strike carries both a call and a put, which in the absence of
   arbitrage imply the same vol. Feeding both into an SVI fit double-counts
   each strike and lets wide ITM quotes drag the fit. `select_otm` keeps the
   out-of-the-money side of each strike, which is also the liquid side.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

MAX_SPREAD_PCT = 0.15
MAX_SPREAD_TICKS = 2
MIN_MID_BTC = 0.0005
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Deribit BTC option tick sizes: 0.0005 BTC above a 0.005 premium, 0.0001
# below it (from public/get_instruments -> tick_size, tick_size_steps).
FINE_TICK = 0.0001
COARSE_TICK = 0.0005
COARSE_TICK_ABOVE = 0.005


def tick_size(mid: "pd.Series | float"):
    """Deribit's minimum price increment at a given premium."""
    if isinstance(mid, pd.Series):
        return np.where(mid > COARSE_TICK_ABOVE, COARSE_TICK, FINE_TICK)
    return COARSE_TICK if mid > COARSE_TICK_ABOVE else FINE_TICK


def time_to_expiry_years(expiry_ts, as_of_ts) -> float:
    """
    Exact annualized time to expiry, UTC-aware.

    Calendar-day counting is not good enough here: Deribit expires at 08:00
    UTC, so a "1 day" option can really be anywhere from a few hours to
    nearly two days, and near-dated vol is very sensitive to that error.
    """
    expiry = pd.Timestamp(expiry_ts)
    as_of = pd.Timestamp(as_of_ts)
    if expiry.tzinfo is None:
        expiry = expiry.tz_localize(dt.timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(dt.timezone.utc)
    return (expiry - as_of).total_seconds() / SECONDS_PER_YEAR


def compute_mid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'mid' column and drop quotes that fail the cleaning rules.

    Dropped: missing or zero bid (a zero bid says nothing about vol), missing
    ask, quotes below MIN_MID_BTC (too close to worthless to carry vol
    information), and quotes that are wide in BOTH relative and absolute
    terms.

    On that last rule -- BUILD_GUIDE.md section 1.2 proposes a bare relative
    spread test, spread > 15% of mid. Measured on a live chain, that filter
    does not measure liquidity, it measures premium size:

        ITM quotes (median premium 0.164 BTC): 96% pass
        OTM quotes (median premium 0.010 BTC): 68% pass

    Cheap options have wide relative spreads because they are cheap. Of the
    132 OTM quotes the relative rule rejected, 73 were within two ticks of
    the exchange minimum -- about as tight as the market is allowed to be.
    Worse, the bias runs opposite to `select_otm`: the relative rule
    preferentially keeps ITM quotes, which the OTM selection then discards,
    and the two together cut a 964-row chain to 278 rows with the front
    expiries down to 2-8 points.

    So a quote passes if it is inside 15% of mid OR within MAX_SPREAD_TICKS
    of the exchange minimum. You cannot ask a market to be tighter than its
    tick.

    This is still a price-space proxy. The principled version measures the
    spread in vol points (invert bid and ask separately, require the gap
    below some threshold), which needs the Phase 2 inverter and belongs
    downstream of it.
    """
    out = df.copy()
    out = out[out["bid"].notna() & out["ask"].notna()]
    out = out[(out["bid"] > 0) & (out["ask"] > 0)]
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out = out[out["mid"] >= MIN_MID_BTC]

    spread = out["ask"] - out["bid"]
    out["spread_pct"] = spread / out["mid"]
    out["spread_ticks"] = spread / tick_size(out["mid"])

    # Both thresholds sit exactly on values the exchange actually quotes, so
    # binary float error decides membership: 0.0008 - 0.0006 evaluates to
    # 2.000000000000001 ticks and a real two-tick market would be dropped.
    tol = 1e-9
    keep = (out["spread_pct"] <= MAX_SPREAD_PCT + tol) | (
        out["spread_ticks"] <= MAX_SPREAD_TICKS + tol
    )
    return out[keep].reset_index(drop=True)


def add_forward_and_expiry(df: pd.DataFrame, as_of_ts=None) -> pd.DataFrame:
    """
    Add `forward`, `T` (years to expiry) and `log_moneyness` k = ln(K/F).

    `forward` is Deribit's own per-expiry forward, carried on every row as
    `underlying_price`. Taking it directly avoids reconstructing a forward
    curve from perpetual funding, which BUILD_GUIDE.md section 1.3 suggests
    but which is unnecessary and an extra source of error.
    """
    out = df.copy()
    as_of = pd.Timestamp(as_of_ts if as_of_ts is not None else out["timestamp"].iloc[0])
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(dt.timezone.utc)

    expiry = pd.to_datetime(out["expiry_ts"], utc=True)
    out["T"] = (expiry - as_of).dt.total_seconds() / SECONDS_PER_YEAR
    out["forward"] = out["underlying_price"]
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    return out[out["T"] > 0].reset_index(drop=True)


LINEAR_QUOTE_CURRENCIES = {"USDC", "USDT", "USD"}


def to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put premiums in USD, which means different arithmetic per contract family.

    Deribit lists two families side by side:

    - **Inverse** (`BTC-...`, `ETH-...`): margined and settled in the base
      coin, premium quoted in that coin. Multiply by the expiry's forward.
    - **Linear** (`BTC_USDC-...`, `SOL_USDC-...`): quoted in dollars already.
      Multiply by nothing.

    Applying the wrong one is not a small error in either direction. Checked
    against Deribit's own mark_iv on live BTC_USDC quotes: inverting the raw
    price gives a median error of 0.28 vol points, while multiplying by the
    forward leaves 0 of 200 quotes invertible at all -- the same failure the
    module docstring describes for inverse options, exactly reversed.

    Frames without a `quote_currency` column are treated as inverse, which is
    right for the Tardis archive (BTC inverse only) and for anything written
    before the linear books were collected.
    """
    out = df.copy()
    if "quote_currency" in out.columns:
        is_linear = (
            out["quote_currency"].astype(str).str.upper().isin(LINEAR_QUOTE_CURRENCIES)
        )
    else:
        is_linear = pd.Series(False, index=out.index)
    multiplier = np.where(is_linear, 1.0, out["forward"])

    for column in ("mid", "bid", "ask", "mark_price"):
        if column in out.columns:
            out[f"{column}_usd"] = out[column] * multiplier
    return out


def select_otm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the out-of-the-money side of each strike: puts below the forward,
    calls above. Halves the row count without losing information, since put
    and call at one strike imply the same vol, and keeps the liquid side.
    """
    otm = ((df["log_moneyness"] < 0) & (df["option_type"] == "P")) | (
        (df["log_moneyness"] >= 0) & (df["option_type"] == "C")
    )
    return df[otm].reset_index(drop=True)


def clean_snapshot(
    df: pd.DataFrame, as_of_ts=None, otm_only: bool = True
) -> pd.DataFrame:
    """
    Full cleaning pipeline: quote filters, time to expiry, forward, USD
    conversion, and (by default) OTM selection.

    Returns a frame ready for implied-vol inversion, carrying at minimum
    strike, T, forward, log_moneyness, mid_usd and option_type.
    """
    out = compute_mid(df)
    out = add_forward_and_expiry(out, as_of_ts)
    out = to_usd(out)
    if otm_only:
        out = select_otm(out)
    return out
