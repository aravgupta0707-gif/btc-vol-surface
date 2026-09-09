"""
What does the options market think BTC will be worth?

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
from surface.density import cdf, quantile, tail_report, total_mass  # noqa: E402
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


def levels_around(forward: float) -> dict:
    """Round levels either side of spot, so the output reads like a market."""
    step = 10_000 if forward > 40_000 else 1_000
    base = round(forward / step) * step
    return {
        f"below {int(base - 3 * step):,}": base - 3 * step,
        f"below {int(base - 2 * step):,}": base - 2 * step,
        f"below {int(base - step):,}": base - step,
        f"above {int(base + step):,}": base + step,
        f"above {int(base + 2 * step):,}": base + 2 * step,
        f"above {int(base + 3 * step):,}": base + 3 * step,
    }


def report_one(surface, stamp, T: float) -> None:
    forward = surface.forward(T)
    report = tail_report(surface, T, levels_around(forward))

    print(f"as of {stamp:%Y-%m-%d %H:%M}Z, horizon {T * 365.25:.0f} days")
    print(f"forward {forward:,.0f}   "
          f"implied vol at the money {surface.iv(forward, T) * 100:.2f}%")
    print(f"density integrates to {report['total_mass']:.4f}"
          + ("" if abs(report["total_mass"] - 1) < 0.02
             else "   <- off by more than 2%, treat with suspicion"))

    print("\nwhere the market prices BTC landing")
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


def report_history(paths: list, asset: str, T: float, every: int) -> None:
    rows = []
    for path in paths[::every]:
        surface, stamp = surface_for(path, asset)
        if surface is None:
            continue
        forward = surface.forward(T)
        rows.append({
            "timestamp": stamp,
            "forward": forward,
            "atm_iv": surface.iv(forward, T) * 100,
            "mass": total_mass(surface, T),
            "q05": quantile(surface, 0.05, T),
            "q50": quantile(surface, 0.50, T),
            "q95": quantile(surface, 0.95, T),
            "p_below_60k": cdf(surface, 60_000.0, T) * 100,
            "p_above_100k": (1 - cdf(surface, 100_000.0, T)) * 100,
        })
    if not rows:
        print("no snapshot produced a fittable surface")
        return

    frame = pd.DataFrame(rows)
    print(f"{asset}: {len(frame)} surfaces, horizon {T * 365.25:.0f} days\n")
    print(f"  {'time':>16} {'forward':>9} {'ATM':>6} {'5%':>9} {'50%':>9} "
          f"{'95%':>9} {'<60k':>7} {'>100k':>7} {'mass':>6}")
    for row in frame.itertuples():
        print(f"  {row.timestamp:%m-%d %H:%M}Z {row.forward:9,.0f} "
              f"{row.atm_iv:5.1f}% {row.q05:9,.0f} {row.q50:9,.0f} "
              f"{row.q95:9,.0f} {row.p_below_60k:6.2f}% {row.p_above_100k:6.2f}% "
              f"{row.mass:6.3f}")

    print("\nstability across the window")
    for name, label in (("p_below_60k", "P(below 60k)"),
                        ("p_above_100k", "P(above 100k)"),
                        ("atm_iv", "ATM vol")):
        series = frame[name]
        print(f"  {label:>14}  mean {series.mean():6.2f}  "
              f"range {series.min():6.2f} to {series.max():6.2f}  "
              f"sd {series.std():5.3f}")
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
    report_one(surface, stamp, T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
