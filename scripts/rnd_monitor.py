"""
What does the options market think BTC (or ETH) will be worth?

    python scripts/rnd_monitor.py                    latest snapshot
    python scripts/rnd_monitor.py --history          across Track A history
    python scripts/rnd_monitor.py --days 90          horizon (default 90)
    python scripts/rnd_monitor.py --asset ETH

Reads the risk-neutral density out of the fitted surface (Breeden-
Litzenberger) and turns it into probabilities: the distribution's quantiles,
and the chance of finishing below or above chosen price levels.

Three things to keep in mind when quoting any of these numbers.

**They are risk-neutral, not real-world.** This is the market's *price* of a
digital option, which embeds the risk premium. The risk-neutral probability
of a crash is systematically higher than the physical one, because people
pay to avoid crashes. Reporting "the market thinks there is a 12% chance of
X" overstates the belief; "the market prices a 12% chance" is right.

**Deep tails come from extrapolation.** Each level is flagged when it falls
outside the range where the slice was actually quoted. Beyond that the
surface clamps, vol goes flat and the tail reverts to lognormal -- a
defensible assumption but not a market observation, and it is precisely the
region where the interesting questions live.

**The horizon is interpolated.** A 90-day query almost never lands on a
listed expiry, so it blends the two neighbouring slices in total variance.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from surface.build import build_from_snapshot  # noqa: E402
from surface.density import (  # noqa: E402
    levels_around, nice_step, p_below, quantile, tail_report, total_mass,
)
from surface.surface import from_fits  # noqa: E402


def arg(name: str, default):
    if name not in sys.argv:
        return default
    return type(default)(sys.argv[sys.argv.index(name) + 1])


def snapshots(asset: str) -> list:
    currency = asset if asset in ("BTC", "ETH") else "USDC"
    return sorted(glob.glob(
        str(ROOT / "data" / "raw" / f"{currency.lower()}_chain_*.parquet")
    ))


def surface_for(path: str, asset: str):
    raw = pd.read_parquet(path)
    fits, ranges, refused, quotes = build_from_snapshot(raw, asset)
    if not fits:
        return None, None
    return from_fits(fits), pd.Timestamp(quotes["timestamp"].iloc[0])


def report_one(surface, stamp, T: float, asset: str = "BTC") -> None:
    forward = surface.forward(T)
    report = tail_report(surface, T, levels_around(forward))

    print(f"as of {stamp:%Y-%m-%d %H:%M}Z, horizon {T * 365.25:.0f} days")
    print(f"forward {forward:,.0f}   "
          f"implied vol at the money {surface.iv(forward, T) * 100:.2f}%")
    print(f"density integrates to {report['total_mass']:.4f}"
          + ("" if abs(report["total_mass"] - 1) < 0.02
             else "   <- off by more than 2%, treat with suspicion"))

    print(f"\nwhere the market prices {asset} landing")
    for q, price in report["quantiles"].items():
        move = price / forward - 1
        print(f"  {q:>5.0%}  {price:>10,.0f}   {move:+7.1%}")

    print("\nprobability of finishing beyond a level")
    print(f"  {'level':>18} {'probability':>12}   {'':<12}")
    for label, item in report["levels"].items():
        p = item["p_below"] if label.startswith("below") else item["p_above"]
        flag = "extrapolated" if item["extrapolated"] else ""
        print(f"  {label:>18} {p:>11.1%}   {flag:<12}")

    extrapolated = [n for n, i in report["levels"].items() if i["extrapolated"]]
    if extrapolated:
        print(f"\n  {len(extrapolated)} level(s) sit outside the quoted strike "
              f"range and rely on a\n  clamped wing -- lognormal by assumption, "
              f"not read from the market.")


def history_levels(forward: float) -> tuple[float, float]:
    """
    One low and one high level, two round steps either side of the forward.

    Fixed for the whole window -- a level that moved with spot would hide the
    very instability the history is meant to show. For BTC near 80k this is
    the 60k / 100k pair the monitor always used; for ETH it is a pair that
    means something on ETH.
    """
    step = nice_step(forward)
    base = round(forward / step) * step
    return max(base - 2 * step, step), base + 2 * step


def short(price: float) -> str:
    return f"{price / 1000:g}k" if price >= 1000 else f"{price:g}"


def report_history(paths: list, asset: str, T: float, every: int) -> None:
    rows = []
    low = high = None
    for path in paths[::every]:
        surface, stamp = surface_for(path, asset)
        if surface is None:
            continue
        forward = surface.forward(T)
        if low is None:
            low, high = history_levels(forward)
        below_low, low_x = p_below(surface, low, T)
        below_high, high_x = p_below(surface, high, T)
        rows.append({
            "timestamp": stamp,
            "forward": forward,
            "atm_iv": surface.iv(forward, T) * 100,
            "mass": total_mass(surface, T),
            "q05": quantile(surface, 0.05, T),
            "q50": quantile(surface, 0.50, T),
            "q95": quantile(surface, 0.95, T),
            "p_low": below_low * 100,
            "p_high": (1 - below_high) * 100,
            "extrapolated": low_x or high_x,
        })
    if not rows:
        print("no snapshot produced a fittable surface")
        return

    frame = pd.DataFrame(rows)
    print(f"{asset}: {len(frame)} surfaces, horizon {T * 365.25:.0f} days\n")
    print(f"  {'time':>16} {'forward':>9} {'ATM':>6} {'5%':>9} {'50%':>9} "
          f"{'95%':>9} {'<' + short(low):>7} {'>' + short(high):>7} {'mass':>6}")
    for row in frame.itertuples():
        print(f"  {row.timestamp:%m-%d %H:%M}Z {row.forward:9,.0f} "
              f"{row.atm_iv:5.1f}% {row.q05:9,.0f} {row.q50:9,.0f} "
              f"{row.q95:9,.0f} {row.p_low:6.2f}% {row.p_high:6.2f}% "
              f"{row.mass:6.3f}{'  x' if row.extrapolated else ''}")

    print("\nstability across the window")
    for name, label in (("p_low", f"P(below {short(low)})"),
                        ("p_high", f"P(above {short(high)})"),
                        ("atm_iv", "ATM vol")):
        series = frame[name]
        print(f"  {label:>14}  mean {series.mean():6.2f}  "
              f"range {series.min():6.2f} to {series.max():6.2f}  "
              f"sd {series.std():5.3f}")
    if frame["extrapolated"].any():
        print(f"\n  x = a level outside the quoted strikes on "
              f"{int(frame['extrapolated'].sum())} surface(s): read from the "
              "integrated density,\n  lognormal-tailed by assumption.")
    print("\n  The tail probabilities should move far less than spot does; a "
          "distribution\n  that lurches hour to hour is a fitting problem, "
          "not a change of belief.")


def main() -> int:
    asset = arg("--asset", "BTC").upper()
    T = arg("--days", 90.0) / 365.25
    every = arg("--every", 4)

    paths = snapshots(asset)
    if not paths:
        print(f"no {asset} snapshots in data/raw/")
        return 2

    if "--history" in sys.argv:
        report_history(paths, asset, T, every)
        return 0

    surface, stamp = surface_for(paths[-1], asset)
    if surface is None:
        print("latest snapshot produced no fittable slices")
        return 1
    report_one(surface, stamp, T, asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
