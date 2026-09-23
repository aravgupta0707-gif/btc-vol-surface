"""
Sell and hold to expiry on the live collection: the cost structure a real
book actually faces.

    python scripts/hold_to_expiry.py            # the Whalley-Wilmott band
    python scripts/hold_to_expiry.py --clock    # the 4h clock, to compare

The archive study round-tripped every position within a day, paying the
bid-ask twice for one day of decay -- and found costs of 34-46 bps a day
against a gross edge of 1-6. A desk selling vol does not do that: it sells
once and lets the option expire, paying the spread once per option life.
The archive cannot test that (it has one day per month), but Track A has a
contiguous fortnight of hourly chains, which holds every daily expiry and a
couple of weeklies.

For each expiry in the collection, sell the ATM straddle at the bid 1 day
(and, where possible, 7 days) before, hedge on a Whalley-Wilmott no-trade
band at whatever snapshots exist, and settle at intrinsic against the
Deribit index at 08:00 UTC. No buy-back, no exit spread.

**The band, not a clock.** This study hedged every 4 hours until the
archive work measured the alternative: a band beats the best clock, and
beats it by more as costs rise (HANDOFF.md section 12.1), and on BTC at 5
bps that survives being chosen without hindsight (section 12.2). Risk
aversion is 1e-4 -- the value the blind walk-forward actually picks, not
the 1e-3 that wins in hindsight. So the live cost measurement and the
archive study now run the same rule, and `strategy.hedging.ww_band` is the
single implementation of it.

**This is a handful of positions over one fortnight of one regime.** It
measures what the cost structure looks like when the spread is paid once,
not whether the strategy has edge. Laptop sleep leaves hedging gaps; the
longest gap per position is reported, and positions left unhedged for more
than 8 hours are marked, because a real book would not be. `hedge_points`
counts the steps the straddle could actually be priced at, and the gaps are
measured between those -- not between the snapshots on disk, which is a
weaker thing to check and was silently passing positions that spent most of
their life unhedged.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HORIZONS = {"1d": pd.Timedelta(days=1), "7d": pd.Timedelta(days=7)}
# The band reads the slippage it is economising on, so these two travel
# together: change the cost and the rule rewidens on its own.
SLIPPAGE_BPS = 5.0
RISK_AVERSION = 1e-4
CLOCK_HOURS = 4.0  # the rule this study used before the band, kept to compare against
ENTRY_TOLERANCE = pd.Timedelta(hours=3)
GAP_LIMIT_HOURS = 8.0


def load(asset: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / f"{asset.lower()}_chain_*.parquet")))
    out = []
    for f in files:
        raw = pd.read_parquet(f)
        if "underlying_asset" in raw.columns:
            raw = raw[raw["underlying_asset"] == asset]
        if not raw.empty:
            out.append((pd.Timestamp(raw["timestamp"].iloc[0]), raw))
    return out


def snapshots_for(chain, expiry, start, end) -> list[pd.DataFrame]:
    from data.clean import clean_snapshot
    from pricing.black76 import implied_vol_many

    snaps, seen = [], set()
    for stamp, raw in chain:
        if not (start <= stamp < end) or stamp.floor("min") in seen:
            continue
        rows = raw[raw["expiry_ts"] == expiry]
        if rows.empty:
            continue
        # `drop_wide=False`: we hold this straddle, so a live two-sided
        # market marks and hedges it however wide the maker is showing.
        # The default rule is calibrated for surface fitting and rejects
        # ordinary one-day ATM quotes, which used to freeze the hedge
        # part-way through most positions here without saying so.
        clean = clean_snapshot(rows, as_of_ts=stamp, otm_only=False,
                               drop_wide=False)
        if clean.empty:
            continue
        clean = clean.assign(iv=implied_vol_many(
            clean["forward"], clean["strike"], clean["T"],
            clean["mid_usd"], clean["option_type"].tolist())).dropna(subset=["iv"])
        if not clean.empty:
            seen.add(stamp.floor("min"))
            snaps.append(clean)
    return snaps


def settle(strategy, position, index_T: float, expiry) -> None:
    """Carry the hedge to the settlement index, flatten it, pay intrinsic."""
    hours = (pd.Timestamp(expiry) - pd.Timestamp(position.last_ts)).total_seconds() / 3600
    position.hedge_pnl += position.hedge_units * (index_T - position.last_spot)
    position.funding_paid += (position.hedge_units * index_T
                              * strategy.config.costs.funding_rate_8h * hours / 8.0)
    strategy._set_hedge(position, 0.0, index_T, charge_slippage=True)
    position.option_value = abs(index_T - position.strike) * position.quantity
    position.last_ts, position.last_spot, position.closed = expiry, index_T, True


def run(asset: str, index: pd.Series, rule: str = "whalley_wilmott") -> list[dict]:
    from strategy.backtest import run_backtest
    from strategy.straddle import Costs, DeltaHedgedStraddle

    chain = load(asset)
    first, last = chain[0][0], chain[-1][0]
    expiries = sorted({pd.Timestamp(e) for _, raw in chain for e in raw["expiry_ts"].unique()
                       if first + pd.Timedelta(hours=20) < pd.Timestamp(e) <= last})
    rows = []
    for expiry in expiries:
        for label, horizon in HORIZONS.items():
            target = expiry - horizon
            near = [s for s, _ in chain if abs(s - target) <= ENTRY_TOLERANCE]
            base = {"asset": asset, "horizon": label, "expiry": expiry}
            if not near:
                if label == "1d" or expiry.dayofweek == 4:
                    rows.append(dict(base, skipped="no snapshot near entry time"))
                continue
            entry = min(near, key=lambda s: abs(s - target))
            snaps = snapshots_for(chain, expiry, entry, expiry)
            if len(snaps) < 2:
                rows.append(dict(base, skipped=f"{len(snaps)} usable snapshots"))
                continue
            # The position must open at the entry snapshot itself. Deribit
            # lists daily expiries only a few days ahead, so a "7-day" hold of
            # a Monday daily silently opened days late in the first version
            # while still being reported as 168 hours.
            opened = pd.Timestamp(snaps[0]["timestamp"].iloc[0])
            if abs(opened - entry) > pd.Timedelta(minutes=1):
                rows.append(dict(base, skipped="not listed (or not quotable) at entry"))
                continue
            strategy = DeltaHedgedStraddle(
                rebalance_rule=rule, risk_aversion=RISK_AVERSION,
                rebalance_interval_hours=CLOCK_HOURS,
                costs=Costs(hedge_slippage_bps=SLIPPAGE_BPS,
                            funding_rate_8h=0.0001))
            try:
                result = run_backtest(snaps, strategy, expiry_ts=expiry, close_at_end=False)
            except ValueError as exc:
                rows.append(dict(base, skipped=str(exc)[:60]))
                continue
            if result.empty:
                rows.append(dict(base, skipped="no marked steps"))
                continue
            position = result.attrs["position"]
            settle_hour = expiry.floor("h")
            if settle_hour not in index.index:
                rows.append(dict(base, skipped="no index at settlement"))
                continue
            S_T = float(index.loc[settle_hour])
            settle(strategy, position, S_T, expiry)
            total = strategy.mark_to_market(position)
            gross = total + position.slippage_paid + position.entry_spread_cost
            # Gaps between the steps the position was actually *marked* at,
            # not between the snapshots that happened to exist. A snapshot
            # the straddle cannot be priced in hedges nothing, so counting
            # it here reported a 3h gap on a position that in truth went 18h
            # unhedged -- the guard was blind to the failure it exists for.
            stamps = ([entry] + [pd.Timestamp(t) for t in result["timestamp"]]
                      + [expiry])
            gaps = np.diff([s.value for s in stamps]) / 3.6e12
            life = index.loc[entry.floor("h"):settle_hour]
            realized = float(np.std(np.diff(np.log(life.to_numpy(float))), ddof=1)
                             * np.sqrt(365.25 * 24)) if len(life) > 3 else np.nan
            spot = position.entry_spot
            rows.append(dict(
                base, skipped="", entry=entry,
                hours=(expiry - entry).total_seconds() / 3600,
                hedge_points=len(result), snapshots=len(snaps),
                max_gap_h=float(gaps.max()),
                rebalances=position.rebalances,
                hedged_properly=bool(gaps.max() <= GAP_LIMIT_HOURS),
                sold_iv=position.entry_iv, realized=realized,
                premium_bps=position.premium_received / spot * 1e4,
                gross_bps=gross / spot * 1e4, net_bps=total / spot * 1e4,
                spread_bps=position.entry_spread_cost / spot * 1e4,
                slippage_bps=position.slippage_paid / spot * 1e4,
                kept=gross / position.premium_received,
            ))
    return rows


def main(argv: list[str] | None = None) -> int:
    from data.history import hourly_funding

    argv = list(sys.argv[1:] if argv is None else argv)
    rule = "fixed_interval" if "--clock" in argv else "whalley_wilmott"

    rows = []
    for asset in ("BTC", "ETH"):
        index = hourly_funding(asset).set_index("time")["index_price"]
        rows += run(asset, index, rule)
    frame = pd.DataFrame(rows)
    # Separate files, because the two are meant to be read side by side and
    # a run under one rule must not quietly overwrite the other's numbers.
    suffix = "_clock" if rule == "fixed_interval" else ""
    out = ROOT / "data" / "research" / f"hold_to_expiry{suffix}.parquet"
    frame.to_parquet(out, index=False)
    print(f"rule: {rule}"
          + (f", every {CLOCK_HOURS:g}h" if rule == "fixed_interval"
             else f", risk aversion {RISK_AVERSION:g}"))

    done = frame[frame["skipped"] == ""]
    pd.set_option("display.width", 200)
    for (asset, horizon), grp in done.groupby(["asset", "horizon"]):
        print(f"\n{asset} {horizon}: {len(grp)} positions held to expiry "
              f"({int(grp.hedged_properly.sum())} hedged without an 8h+ gap)")
        show = grp.assign(expiry=grp.expiry.dt.strftime("%d%b %Hh"),
                          sold=grp.sold_iv * 100, real=grp.realized * 100)
        print(show[["expiry", "hours", "hedge_points", "rebalances", "max_gap_h",
                    "sold", "real", "premium_bps", "gross_bps", "spread_bps",
                    "net_bps"]]
              .round(1).to_string(index=False))
        ok = grp[grp.hedged_properly]
        for label, sub in (("all", grp), ("hedged properly", ok)):
            if len(sub):
                print(f"  {label:16s} mean gross {sub.gross_bps.mean():+6.1f}  spread "
                      f"{sub.spread_bps.mean():5.1f}  slippage {sub.slippage_bps.mean():4.1f}  "
                      f"net {sub.net_bps.mean():+6.1f} bps   seller kept {sub.kept.mean()*100:+.0f}% "
                      f"of premium   net>0 on {(sub.net_bps > 0).mean()*100:.0f}%")
    skipped = frame[frame["skipped"] != ""]
    if len(skipped):
        print("\nskipped:", skipped.groupby(["asset", "horizon"])["skipped"].value_counts().to_dict())
    print("\nOne fortnight, one regime: this shows the cost structure of holding to expiry, "
          "not an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
