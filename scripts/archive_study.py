"""
Six years of one-day delta-hedged short straddles: one independent sample per
Track B archive day.

    python scripts/archive_study.py              run BTC and ETH, then report
    python scripts/archive_study.py --report     re-print from saved results

On each first-of-month since 2020-03, sell an ATM straddle at the day's first
hourly snapshot, rebalance the delta hedge every 4 hours, and buy it back at
the day's last snapshot -- the Phase 6 mechanics, unchanged, fed from the
archive instead of the live collector. Each day is its own position, so the
79 results are independent draws across regimes rather than one path.

**The tenor is fixed, not "nearest".** The engine's default picks the nearest
expiry that survives the window, which across six years means a 7-day option
in one era and a 1-day option in another -- and an era comparison would
silently become a tenor comparison. Two bands exist in every era, so both
are run: `weekly` (first expiry at least 5 days out) and `monthly` (at least
21 days). The tenor actually used is recorded per day.

**Gross and net are reported separately, on purpose.** Gross P&L is the
variance premium the position captured. Net subtracts crossing the bid-ask
twice plus hedge slippage -- and for a one-day hold of a multi-week option,
two spread crossings are paid for one day of decay. Net here answers "what
does a daily round trip cost", not "what does the strategy earn"; a real
book holds to expiry and pays the spread once per option life.

P&L is quoted in basis points of spot so 2020 ($9k) and 2026 ($80k) compare.
With about 13 days per year, per-year means are descriptive; the t-statistic
is printed so that nobody mistakes one for a result.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIST_DIR = ROOT / "data" / "historical"
OUT_PATH = ROOT / "data" / "research" / "archive_study.parquet"
TENORS = {"weekly": 5.0, "monthly": 21.0}   # minimum days to expiry at entry
ASSETS = ("BTC", "ETH")
REBALANCE_HOURS = 4.0


def _day_snapshots(raw: pd.DataFrame, expiry) -> list[pd.DataFrame]:
    """Cleaned, IV-inverted snapshots of one expiry, one per hourly bucket."""
    from data.clean import clean_snapshot
    from pricing.black76 import implied_vol_many

    snaps = []
    for stamp, bucket in raw[raw["expiry_ts"] == expiry].groupby("timestamp", sort=True):
        # The bucket's time is passed explicitly: when the quote filters
        # empty a bucket, clean_snapshot otherwise reads it from row 0 of an
        # empty frame and raises (it did, on one day in 158).
        clean = clean_snapshot(bucket, as_of_ts=pd.Timestamp(stamp), otm_only=False)
        if clean.empty:
            continue
        clean = clean.assign(iv=implied_vol_many(
            clean["forward"], clean["strike"], clean["T"],
            clean["mid_usd"], clean["option_type"].tolist(),
        )).dropna(subset=["iv"])
        if not clean.empty:
            snaps.append(clean)
    return snaps


def run_day(path: Path) -> list[dict]:
    """Both tenors for one archive day; a skipped tenor records why."""
    from strategy.backtest import run_backtest
    from strategy.straddle import Costs, DeltaHedgedStraddle, decompose

    asset = path.name.split("_")[0].upper()
    day = pd.Timestamp(path.stem.split("_")[-1]).date()
    raw = pd.read_parquet(path)
    stamps = sorted(raw["timestamp"].unique())
    first, last = pd.Timestamp(stamps[0]), pd.Timestamp(stamps[-1])
    base = {"date": day, "asset": asset}

    if first.hour > 1 or last.hour < 22:
        return [dict(base, tenor=t, skipped=f"partial day {first:%H}-{last:%H}h")
                for t in TENORS]

    live_at_close = set(raw.loc[raw["timestamp"] == stamps[-1], "expiry_ts"])
    at_open = sorted(e for e in raw.loc[raw["timestamp"] == stamps[0], "expiry_ts"].unique()
                     if e in live_at_close)
    rows = []
    for tenor, min_days in TENORS.items():
        row = dict(base, tenor=tenor)
        candidates = [e for e in at_open
                      if (pd.Timestamp(e) - first).total_seconds() / 86400 >= min_days]
        if not candidates:
            rows.append(dict(row, skipped=f"no expiry >= {min_days:g}d"))
            continue
        expiry = candidates[0]
        snaps = _day_snapshots(raw, expiry)
        if len(snaps) < 4:
            rows.append(dict(row, skipped=f"{len(snaps)} usable snapshots"))
            continue
        strategy = DeltaHedgedStraddle(
            rebalance_rule="fixed_interval", rebalance_interval_hours=REBALANCE_HOURS,
            costs=Costs(hedge_slippage_bps=5.0, funding_rate_8h=0.0001),
        )
        try:
            result = run_backtest(snaps, strategy, expiry_ts=expiry)
        except ValueError as exc:
            rows.append(dict(row, skipped=str(exc)[:80]))
            continue
        if result.empty:
            rows.append(dict(row, skipped="no marked steps"))
            continue

        position = result.attrs["position"]
        realized = float(result["realized_vol"].iloc[-1])
        out = decompose(position, realized)
        gross = out["total_pnl"] + out["slippage_paid"] + position.entry_spread_cost
        spot = position.entry_spot
        bps = 1e4 / spot
        rows.append(dict(
            row,
            skipped="",
            days_to_expiry=(pd.Timestamp(expiry) - first).total_seconds() / 86400,
            hours_held=(pd.Timestamp(position.last_ts) - first).total_seconds() / 3600,
            snapshots=len(snaps),
            spot=spot,
            strike=position.strike,
            entry_iv=position.entry_iv,
            realized_vol=realized,
            premium_bps=position.premium_received * bps,
            gross_bps=gross * bps,
            net_bps=out["total_pnl"] * bps,
            spread_bps=position.entry_spread_cost * bps,
            slippage_bps=out["slippage_paid"] * bps,
            funding_bps=out["funding_paid"] * bps,
            theoretical_bps=out["theoretical_pnl"] * bps,
            rebalances=position.rebalances,
        ))
    return rows


def run_day_safely(path: Path) -> list[dict]:
    """
    One day's failure is recorded against that day, never the whole run.
    The first full run died at day 80 of 158 on a single empty bucket and
    returned nothing -- the archive fetch's lesson, relearned.
    """
    try:
        return run_day(path)
    except Exception as exc:  # noqa: BLE001 - reported per day in the summary
        day = pd.Timestamp(path.stem.split("_")[-1]).date()
        asset = path.name.split("_")[0].upper()
        return [{"date": day, "asset": asset, "tenor": t,
                 "skipped": f"error: {type(exc).__name__}: {str(exc)[:60]}"} for t in TENORS]


def run_all() -> pd.DataFrame:
    paths = sorted(p for a in ASSETS for p in HIST_DIR.glob(f"{a.lower()}_chain_*.parquet"))
    print(f"{len(paths)} archive days across {', '.join(ASSETS)}; "
          f"tenors {', '.join(f'{k} (>={v:g}d)' for k, v in TENORS.items())}", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, day_rows in enumerate(pool.map(run_day_safely, paths), 1):
            rows.extend(day_rows)
            if i % 20 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} days", flush=True)
    frame = pd.DataFrame(rows).sort_values(["asset", "tenor", "date"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    return frame


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def report(frame: pd.DataFrame) -> None:
    done = frame[frame["skipped"] == ""].copy()
    skipped = frame[frame["skipped"] != ""]
    done["year"] = pd.to_datetime(done["date"]).dt.year
    done["gap_pts"] = (done["entry_iv"] - done["realized_vol"]) * 100

    for (asset, tenor), grp in done.groupby(["asset", "tenor"], sort=False):
        print(f"\n{asset} {tenor}: {len(grp)} days, median tenor "
              f"{grp['days_to_expiry'].median():.0f}d, "
              f"{len(skipped[(skipped.asset == asset) & (skipped.tenor == tenor)])} skipped")
        print(f"  {'year':>5} {'n':>3} {'IV':>6} {'RV':>6} {'IV-RV':>6} {'IV>RV':>6} "
              f"{'gross':>7} {'net':>7} {'net>0':>6} {'t(gross)':>8}")
        for label, sub in list(grp.groupby("year")) + [("all", grp)]:
            print(f"  {label!s:>5} {len(sub):3d} {sub.entry_iv.mean()*100:6.1f} "
                  f"{sub.realized_vol.mean()*100:6.1f} {sub.gap_pts.mean():+6.1f} "
                  f"{(sub.gap_pts > 0).mean()*100:5.0f}% "
                  f"{sub.gross_bps.mean():+7.1f} {sub.net_bps.mean():+7.1f} "
                  f"{(sub.net_bps > 0).mean()*100:5.0f}% {_tstat(sub.gross_bps):8.2f}")
        costs = grp["spread_bps"] + grp["slippage_bps"]
        print(f"  costs per day: spread {grp.spread_bps.mean():.1f} + slippage "
              f"{grp.slippage_bps.mean():.1f} = {costs.mean():.1f} bps "
              f"against mean gross {grp.gross_bps.mean():+.1f} bps")
        # One day of 4-hourly hedging is dominated by discretisation noise, so
        # a per-day ratio says little. Across days the question is whether
        # gross P&L moves with the variance gap at all.
        corr = grp["gross_bps"].corr(grp["theoretical_bps"])
        agree = (np.sign(grp["gross_bps"]) == np.sign(grp["theoretical_bps"])).mean()
        print(f"  theory gate: corr(gross, theoretical) {corr:+.2f} across days; "
              f"same sign on {agree*100:.0f}% of days")

    if len(skipped):
        print("\nskipped:")
        for reason, n in skipped["skipped"].value_counts().items():
            print(f"  {n:3d}  {reason}")
    print("\nP&L in bps of spot per 1-unit straddle, one day held. gross = variance premium "
          "captured;\nnet = after crossing the spread twice and hedge slippage. "
          f"Results: {OUT_PATH.relative_to(ROOT)}")


def main() -> int:
    if "--report" in sys.argv:
        if not OUT_PATH.exists():
            print("no saved results; run without --report first")
            return 2
        report(pd.read_parquet(OUT_PATH))
        return 0
    report(run_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
