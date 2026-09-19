"""
Phase 6: backtest the delta-hedged short straddle on collected snapshots.

    python scripts/run_backtest.py                     all Track A history
    python scripts/run_backtest.py --hours 24          the last 24 hours
    python scripts/run_backtest.py --rule delta_band
    python scripts/run_backtest.py --asset ETH

Runs on Track A (the hourly collector). Every snapshot is cleaned with
`otm_only=False`, because a straddle needs a call AND a put at one strike and
the OTM selection keeps only one side of each.

The number to read first is not the Sharpe ratio. It is whether realised P&L
tracks the theoretical (implied - realized variance) x vega estimate. The VRP
study puts the current gross premium at roughly 3-4 vol points, down from
18.6 in 2021; a backtest reporting materially more than that has a bug in the
hedging or the costs, not an edge.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.clean import clean_snapshot  # noqa: E402
from pricing.black76 import implied_vol_many  # noqa: E402
from strategy.backtest import compute_metrics, run_backtest, summarize  # noqa: E402
from strategy.straddle import Costs, DeltaHedgedStraddle  # noqa: E402


def arg(name: str, default):
    if name not in sys.argv:
        return default
    return type(default)(sys.argv[sys.argv.index(name) + 1])


def load_snapshots(asset: str, hours: float | None) -> list:
    currency = asset if asset in ("BTC", "ETH") else "USDC"
    files = sorted(glob.glob(
        str(ROOT / "data" / "raw" / f"{currency.lower()}_chain_*.parquet")
    ))
    if not files:
        print(f"no {currency} snapshots in data/raw/")
        raise SystemExit(2)

    frames = []
    for path in files:
        raw = pd.read_parquet(path)
        if "underlying_asset" in raw.columns:
            raw = raw[raw["underlying_asset"] == asset]
        # Both legs are needed at the straddle strike, so no OTM selection.
        clean = clean_snapshot(raw, otm_only=False)
        if clean.empty:
            continue
        # Inverting the whole chain here costs about 2.3s per snapshot, so a
        # 44-snapshot run spends roughly two minutes in the root finder.
        # Restricting to near-the-money quotes was tried and measured at
        # 2.45s against 2.32s for the full chain -- no saving, because the
        # far wings are not the slow ones. Left inverting everything, which
        # also keeps every strike available to the strike picker.
        clean = clean.assign(iv=implied_vol_many(
            clean["forward"], clean["strike"], clean["T"],
            clean["mid_usd"], clean["option_type"].tolist(),
        )).dropna(subset=["iv"])
        if not clean.empty:
            frames.append(clean)

    frames.sort(key=lambda f: pd.Timestamp(f["timestamp"].iloc[0]))
    if hours:
        cutoff = pd.Timestamp(frames[-1]["timestamp"].iloc[0]) - pd.Timedelta(hours=hours)
        frames = [f for f in frames if pd.Timestamp(f["timestamp"].iloc[0]) >= cutoff]
    # A duplicate timestamp would trip the no-lookahead guard; the collector
    # produced a few during development when run by hand alongside the task.
    seen, unique = set(), []
    for frame in frames:
        stamp = pd.Timestamp(frame["timestamp"].iloc[0]).floor("min")
        if stamp not in seen:
            seen.add(stamp)
            unique.append(frame)
    return unique


def main() -> int:
    asset = arg("--asset", "BTC").upper()
    rule = arg("--rule", "fixed_interval")
    interval = arg("--interval", 4.0)
    hours = arg("--hours", 0.0) or None

    snapshots = load_snapshots(asset, hours)
    if len(snapshots) < 4:
        print(f"only {len(snapshots)} usable snapshots; need at least 4")
        return 2

    start = pd.Timestamp(snapshots[0]["timestamp"].iloc[0])
    end = pd.Timestamp(snapshots[-1]["timestamp"].iloc[0])
    span = (end - start).total_seconds() / 3600.0
    print(f"{asset}: {len(snapshots)} snapshots, {start:%Y-%m-%d %H:%M}Z to "
          f"{end:%Y-%m-%d %H:%M}Z ({span:.1f}h)")
    print(f"rule: {rule}"
          + (f", every {interval:g}h" if rule == "fixed_interval" else ""))

    strategy = DeltaHedgedStraddle(
        rebalance_rule=rule, rebalance_interval_hours=interval,
        costs=Costs(hedge_slippage_bps=5.0, funding_rate_8h=0.0001),
    )
    result = run_backtest(snapshots, strategy)
    if result.empty:
        print("no marked steps")
        return 1

    position = result.attrs["position"]
    out = summarize(result)
    print(f"\nposition: short {position.quantity:g} straddle, "
          f"K={position.strike:,.0f}, expiry {position.expiry_ts:%d%b%y}, "
          f"sold at {position.entry_iv * 100:.2f}% vol")
    print(f"spot {position.entry_spot:,.0f} -> {result['spot'].iloc[-1]:,.0f}")

    print(f"\nP&L decomposition (USD, 1 {asset} notional per leg)")
    print(f"  premium received      {position.premium_received:+12,.2f}")
    print(f"  cost to buy back      {-position.option_value:+12,.2f}")
    print(f"  = option leg          {out['option_pnl']:+12,.2f}")
    print(f"  hedge leg             {out['hedge_pnl']:+12,.2f}")
    print(f"  funding               {-out['funding_paid']:+12,.2f}")
    print(f"  hedge slippage        {-out['slippage_paid']:+12,.2f}")
    print(f"  entry/exit spread     {-position.entry_spread_cost:+12,.2f}  "
          f"(already inside the option leg)")
    print(f"  {'':-<22}{'':->13}")
    print(f"  total                 {out['total_pnl']:+12,.2f}")

    print("\nthe sanity check that gates the Sharpe ratio")
    print(f"  sold vol              {out['entry_iv'] * 100:8.2f}%")
    print(f"  realized vol          {out['realized_vol'] * 100:8.2f}%")
    print(f"  variance gap          {out['variance_gap']:+8.4f}"
          f"   ({'premium earned' if out['variance_gap'] > 0 else 'premium paid back'})")

    # The gate itself. BUILD_GUIDE.md section 8: realised P&L must track
    # 0.5 * vega * (implied^2 - realized^2) / implied, scaled by the fraction
    # of the option's life actually held. If it does not, the hedging or the
    # cost logic is wrong and every metric below is fiction.
    theoretical = out["theoretical_pnl"]
    gross = out["total_pnl"] + out["slippage_paid"] + position.entry_spread_cost
    print(f"  theoretical P&L       {theoretical:+8,.2f}   "
          f"(vega x variance gap, pro-rated for time held)")
    print(f"  realised, gross of costs {gross:+8,.2f}")
    if abs(theoretical) > 1e-9:
        ratio = gross / theoretical
        verdict = "tracks" if 0.5 <= ratio <= 2.0 else "DOES NOT TRACK -- investigate"
        print(f"  ratio                 {ratio:8.2f}   {verdict}")
    # Gross, like the ratio above. Comparing the net total instead flagged a
    # thin edge eaten by costs as a hedging failure (ETH, 13 Sep 2026: gross
    # +8.28 against a positive gap, net -1.13) -- and a gate that cries wolf
    # teaches people to skip the line. Costs get their own verdict below.
    sign_ok = np.sign(out["variance_gap"]) == np.sign(gross)
    print(f"  P&L sign matches gap  {'yes' if sign_ok else 'NO -- investigate'}")
    if sign_ok and np.sign(out["total_pnl"]) != np.sign(gross):
        print("  costs exceed the edge  net P&L has the opposite sign to gross")

    print("\nactivity")
    print(f"  rebalances            {out['rebalances']:8d}")
    print(f"  turnover              {out['turnover_btc']:8.3f} {asset}")
    print(f"  hedge range           {result['hedge_units'].min():+.3f} to "
          f"{result['hedge_units'].max():+.3f} {asset}")

    print("\nstep statistics (descriptive only -- far too few steps to size on)")
    print(f"  steps                 {out['n']:8d}")
    print(f"  hit rate              {out['hit_rate']:8.1%}")
    print(f"  max drawdown          {out['max_drawdown']:+8,.2f}")
    print(f"  best / worst step     {out['best_step']:+,.2f} / {out['worst_step']:+,.2f}")

    print("\n" + "=" * 62)
    horizon_note = (
        "This is a single position over a short window, not a strategy "
        "evaluation.\nIt demonstrates the machinery end to end; the "
        "statistical claim needs\nthe 78-day Track B study or weeks more of "
        "Track A history."
    )
    print(horizon_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
