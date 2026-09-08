"""
Does selling BTC volatility actually pay? The variance risk premium, measured.

    python scripts/vrp_study.py            use cached data if present
    python scripts/vrp_study.py --refresh  re-download

This exists to test the premise of the Phase 6 strategy *before* building
it. A delta-hedged short straddle earns roughly
`(implied variance - realized variance) x vega`, so if implied has not
historically exceeded subsequent realized by more than it costs to trade,
the strategy has no edge and a backtest would only discover that slowly.

Method:

- Implied: Deribit's DVOL index, their 30-day forward-looking BTC vol,
  daily since 2021-03-24.
- Realized: close-to-close vol of the BTC perpetual over the 30 days
  *following* each observation, annualized.
- Premium: implied minus subsequent realized, in vol points.

Two methodology points that matter more than they look:

- **Annualize with 365, not 252.** Crypto trades continuously; there are no
  weekends or holidays to exclude. Using the equity convention would inflate
  realized vol by sqrt(365/252) = 1.20, about 8 vol points at these levels,
  which is larger than the entire premium being measured.

- **Overlapping windows are not independent.** Consecutive days share 29 of
  30 days of realized vol, so the effective sample is ~1/30 of the row count
  and naive standard errors are badly overstated. Both overlapping and
  non-overlapping figures are reported; trust the second for significance.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "research"
BASE_URL = "https://www.deribit.com/api/v2/public"

HORIZON_DAYS = 30
ANNUALIZE = np.sqrt(365.0)  # 24/7 market: no trading-day adjustment
DVOL_START = dt.date(2021, 3, 1)


def _ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day,
                           tzinfo=dt.timezone.utc).timestamp() * 1000)


def fetch_dvol(refresh: bool = False) -> pd.Series:
    """Daily DVOL closes, paged back to the index's first day."""
    path = CACHE / "dvol_btc_daily.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["dvol"]

    rows, cursor = [], DVOL_START
    today = dt.date.today()
    while cursor < today:
        window_end = min(cursor + dt.timedelta(days=700), today)
        payload = requests.get(
            f"{BASE_URL}/get_volatility_index_data",
            params=dict(currency="BTC", start_timestamp=_ms(cursor),
                        end_timestamp=_ms(window_end), resolution=86400),
            timeout=60,
        ).json()
        chunk = payload.get("result", {}).get("data", [])
        rows.extend(chunk)
        if not chunk:
            break
        cursor = window_end

    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    frame["date"] = pd.to_datetime(frame["ts"], unit="ms", utc=True).dt.date
    series = (
        frame.drop_duplicates("date").set_index("date")["close"].rename("dvol")
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(path)
    return series


def fetch_btc_closes(refresh: bool = False) -> pd.Series:
    """Daily BTC perpetual closes covering the DVOL window plus the horizon."""
    path = CACHE / "btc_daily_close.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["close"]

    payload = requests.get(
        f"{BASE_URL}/get_tradingview_chart_data",
        params=dict(instrument_name="BTC-PERPETUAL", start_timestamp=_ms(DVOL_START),
                    end_timestamp=_ms(dt.date.today()), resolution="1D"),
        timeout=60,
    ).json()["result"]

    frame = pd.DataFrame({"ts": payload["ticks"], "close": payload["close"]})
    frame["date"] = pd.to_datetime(frame["ts"], unit="ms", utc=True).dt.date
    series = (
        frame.drop_duplicates("date").set_index("date")["close"].rename("close")
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(path)
    return series


def forward_realized_vol(closes: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    """
    Annualized close-to-close vol over the `horizon` days AFTER each date.

    Shifted deliberately: the question is whether today's implied predicts
    the volatility that follows, so the window must contain no information
    available at the observation date.
    """
    log_returns = np.log(closes / closes.shift(1))
    forward = (
        log_returns.rolling(horizon).std().shift(-horizon) * ANNUALIZE * 100
    )
    return forward.rename("realized")


def build_panel(refresh: bool = False) -> pd.DataFrame:
    dvol = fetch_dvol(refresh)
    closes = fetch_btc_closes(refresh)
    panel = pd.concat([dvol, forward_realized_vol(closes)], axis=1).dropna()
    panel.index = pd.to_datetime(panel.index)
    panel["premium"] = panel["dvol"] - panel["realized"]
    # Variance-space premium, which is what a variance swap actually pays and
    # what the straddle P&L decomposition is proportional to.
    panel["var_premium"] = (panel["dvol"] ** 2 - panel["realized"] ** 2) / 100
    return panel.sort_index()


def describe(name: str, premium: pd.Series) -> None:
    hit = (premium > 0).mean()
    print(f"  {name}")
    print(f"    n                {len(premium)}")
    print(f"    mean             {premium.mean():+7.2f} vol points")
    print(f"    median           {premium.median():+7.2f}")
    print(f"    std              {premium.std():7.2f}")
    print(f"    hit rate         {hit:7.1%}  (implied above subsequent realized)")
    print(f"    5th / 95th pct   {premium.quantile(0.05):+7.2f} / "
          f"{premium.quantile(0.95):+.2f}")
    if premium.std() > 0:
        # Not a Sharpe ratio -- just the premium scaled by its own dispersion.
        print(f"    mean / std       {premium.mean() / premium.std():7.2f}")


def main() -> int:
    refresh = "--refresh" in sys.argv
    panel = build_panel(refresh)

    print(f"BTC variance risk premium, {panel.index[0]:%Y-%m-%d} to "
          f"{panel.index[-1]:%Y-%m-%d}")
    print(f"implied = DVOL (30d), realized = following {HORIZON_DAYS}d "
          f"close-to-close, annualized x sqrt(365)\n")

    print("levels")
    print(f"    mean implied     {panel['dvol'].mean():7.2f}")
    print(f"    mean realized    {panel['realized'].mean():7.2f}")

    print("\npremium (implied - realized)")
    describe("all days (overlapping windows)", panel["premium"])
    # Every 30th day, so no two windows share a return.
    print()
    describe(f"every {HORIZON_DAYS}th day (independent windows)",
             panel["premium"].iloc[::HORIZON_DAYS])

    print("\nby year")
    print(f"    {'year':>6} {'n':>5} {'implied':>9} {'realized':>9} "
          f"{'premium':>9} {'hit':>7}")
    for year, group in panel.groupby(panel.index.year):
        print(f"    {year:>6} {len(group):5d} {group['dvol'].mean():9.1f} "
              f"{group['realized'].mean():9.1f} {group['premium'].mean():+9.2f} "
              f"{(group['premium'] > 0).mean():7.1%}")

    print("\nworst episodes for a short-vol position")
    print("  (realized far above implied: this is where the premium is paid back)")
    worst = panel.nsmallest(8, "premium")
    print(f"    {'date':>12} {'implied':>9} {'realized':>9} {'premium':>9}")
    for date, row in worst.iterrows():
        print(f"    {date:%Y-%m-%d} {row['dvol']:9.1f} {row['realized']:9.1f} "
              f"{row['premium']:+9.2f}")

    print("\nhow concentrated are the losses?")
    losses = panel.loc[panel["premium"] < 0, "premium"]
    total_loss = losses.sum()
    worst_5pct = losses.nsmallest(max(1, len(losses) // 20)).sum()
    print(f"    losing days      {len(losses)} of {len(panel)} "
          f"({len(losses) / len(panel):.1%})")
    print(f"    worst 5% of losing days carry "
          f"{worst_5pct / total_loss:.0%} of all losses")

    out = CACHE / "vrp_panel.parquet"
    panel.to_parquet(out)
    print(f"\npanel written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
