"""
Does the options market's cost of carry forecast the funding the perpetual
actually pays?

    python scripts/carry_study.py

Every archive chain carries Deribit's forward for each expiry -- that is the
options-implied forward curve. Measured from the nearest expiry's forward
out to any tenor, it gives an implied carry (see `implied_carry` for why
never against the index). The perpetual's hourly funding is the carry a
long actually paid. A spot check on 9 September found the two agreeing to
~50bp at the long end, with funding far noisier week to week; so the
question is not "where do they disagree" but "which is the better forecast
of funding over the coming weeks: the forward curve, or the funding just
paid?"

For each archive day and each tenor (7, 30, 90 days): implied carry at
00:00, trailing funding over the previous tenor, realised funding over the
next. All annualised, in percent.

Sampling: one day a month. 7- and 30-day windows barely overlap; 90-day
windows overlap about threefold, so their effective sample is a third of n.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIST_DIR = ROOT / "data" / "historical"
OUT_PATH = ROOT / "data" / "research" / "carry_study.parquet"
TENORS = (7, 30, 90)
YEAR_HOURS = 365.25 * 24
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def implied_carry(chain: pd.DataFrame, tenor_days: int) -> float:
    """
    Annualised carry from the nearest expiry's forward out to `tenor_days`.

    Measured forward-to-forward inside one snapshot, never against the index.
    Each archive bucket keeps the last quote in its hour, so a forward can be
    up to 59 minutes away from the 00:00 index; half an hour of BTC moves
    ~0.45% of price, and dividing by T = 7/365 turns that into ~23 points of
    annualised noise. Measured against the index, the 7-day carry had a
    standard deviation of 41 points and an RMSE of 41 against funding;
    measured against the nearest forward (median 8 hours to expiry, so
    effectively spot, quoted with the same mix of timings), 10 and 9.7.
    """
    t0 = pd.Timestamp(chain["timestamp"].iloc[0])
    curve = (chain.groupby("expiry_ts")["underlying_price"].median()
             .rename("F").reset_index())
    curve["T"] = (pd.to_datetime(curve["expiry_ts"], utc=True) - t0).dt.total_seconds() / SECONDS_PER_YEAR
    curve = curve.sort_values("T")
    near = curve.iloc[0]
    far = curve[curve["T"] > 1 / 365.25]
    target = tenor_days / 365.25
    if far.empty or target < far["T"].iloc[0] or target > far["T"].iloc[-1]:
        return np.nan
    y = np.interp(target, far["T"].to_numpy(float), np.log(far["F"].to_numpy(float)))
    return float((y - np.log(near["F"])) / (target - near["T"]) * 100)


def funding_rate(funding: pd.Series, start, end) -> float:
    window = funding.loc[start:end - pd.Timedelta(hours=1)]
    hours = (end - start) / pd.Timedelta(hours=1)
    if len(window) < 0.95 * hours:
        return np.nan
    return float(window.mean() * YEAR_HOURS * 100)


def run() -> pd.DataFrame:
    from data.history import hourly_funding

    rows = []
    for asset in ("BTC", "ETH"):
        funding = hourly_funding(asset).set_index("time")["interest_1h"]
        now = funding.index.max()
        for path in sorted(HIST_DIR.glob(f"{asset.lower()}_chain_*.parquet")):
            raw = pd.read_parquet(path, columns=["timestamp", "expiry_ts", "underlying_price"])
            t0 = raw["timestamp"].min()
            chain = raw[raw["timestamp"] == t0]
            for days in TENORS:
                span = pd.Timedelta(days=days)
                rows.append({
                    "asset": asset, "date": t0.date(), "tenor": days,
                    "implied": implied_carry(chain, days),
                    "trailing": funding_rate(funding, t0 - span, t0),
                    "future": funding_rate(funding, t0, t0 + span) if t0 + span <= now else np.nan,
                })
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT_PATH, index=False)
    return frame


def _fit(y, x):
    """OLS slope, intercept and R^2 of y on x."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta[1], beta[0], 1 - resid.var() / y.var()


def report(frame: pd.DataFrame) -> None:
    frame = frame.assign(year=pd.to_datetime(frame["date"]).dt.year)
    for asset in ("BTC", "ETH"):
        print(f"\n{asset}  (annualised %, means; RMSE against funding realised over the next tenor)")
        print(f"  {'tenor':>5} {'n':>3} {'implied':>8} {'trailing':>9} {'realised':>9}"
              f" {'RMSE impl':>10} {'RMSE trail':>11} {'slope':>6} {'R2':>5}  winner")
        for days in TENORS:
            g = frame[(frame.asset == asset) & (frame.tenor == days)].dropna()
            if len(g) < 5:
                continue
            e_impl = np.sqrt(((g.implied - g.future) ** 2).mean())
            e_trail = np.sqrt(((g.trailing - g.future) ** 2).mean())
            slope, _, r2 = _fit(g.future.to_numpy(), g.implied.to_numpy())
            winner = "implied" if e_impl < e_trail else "trailing"
            print(f"  {days:4d}d {len(g):3d} {g.implied.mean():8.2f} {g.trailing.mean():9.2f}"
                  f" {g.future.mean():9.2f} {e_impl:10.2f} {e_trail:11.2f} {slope:6.2f} {r2:5.2f}  {winner}")
        g = frame[(frame.asset == asset) & (frame.tenor == 30)].dropna()
        by_year = g.groupby("year")[["implied", "trailing", "future"]].mean()
        by_year["implied-future"] = by_year["implied"] - by_year["future"]
        print("  30-day, by year:")
        print(by_year.round(2).to_string().replace("\n", "\n  "))
    print("\nimplied = options-implied carry, forward-to-forward; trailing = funding paid over the previous "
          "tenor;\nrealised = funding paid over the next. 90-day windows overlap ~3x.")


def main() -> int:
    report(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
