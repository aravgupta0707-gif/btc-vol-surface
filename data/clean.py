"""
Clean raw option chain snapshots into something safe to invert for
implied vol.

Rules (see BUILD_GUIDE.md section 1.2):
- mid = (bid + ask) / 2
- drop quotes where spread > MAX_SPREAD_PCT of mid
- drop zero-bid quotes entirely
- compute time-to-expiry in years using exact UTC timestamps
  (Deribit expiries are 08:00 UTC)
"""

MAX_SPREAD_PCT = 0.15


def compute_mid(df: "pandas.DataFrame") -> "pandas.DataFrame":
    """Add a 'mid' column and drop quotes that fail the cleaning rules."""
    raise NotImplementedError


def time_to_expiry_years(expiry_ts, as_of_ts) -> float:
    """Exact annualized time to expiry, UTC-aware."""
    raise NotImplementedError
