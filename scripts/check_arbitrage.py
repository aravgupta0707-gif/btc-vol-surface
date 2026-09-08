"""
Phase 4 checkpoint: calendar and butterfly checks on a real fitted surface.

    python scripts/check_arbitrage.py               newest stored BTC snapshot
    python scripts/check_arbitrage.py --live
    python scripts/check_arbitrage.py --asset ETH
    python scripts/check_arbitrage.py --log         append findings to VIOLATIONS.md

BUILD_GUIDE.md section 4.3 asks for every violation to be logged along with
how it was resolved, and is right that this log is worth more than a surface
that happened to fit cleanly. `--log` appends a dated entry to VIOLATIONS.md.

Findings are split by whether they fall among strikes that actually trade.
A violation inside the quoted range means the fit or the data is wrong. One
out in the extrapolated wings means the slice should not be trusted that far
out -- real, worth constraining in Phase 5, but not the same problem.
"""

from __future__ import annotations

import datetime as dt
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from surface.arbitrage import (  # noqa: E402
    check_surface, default_k_grid, trust_radius,
)
from surface.build import build_from_snapshot  # noqa: E402

LOG_PATH = ROOT / "VIOLATIONS.md"


def load(live: bool, currency: str) -> pd.DataFrame:
    if live:
        from data.fetch_deribit import snapshot_chain, write_snapshot

        frame = snapshot_chain(currency)
        write_snapshot(frame, currency=currency)
        return frame
    pattern = str(ROOT / "data" / "raw" / f"{currency.lower()}_chain_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no {currency} snapshots in data/raw/; run with --live")
        raise SystemExit(2)
    print(f"using {Path(files[-1]).name}")
    return pd.read_parquet(files[-1])


def main() -> int:
    asset = "BTC"
    if "--asset" in sys.argv:
        asset = sys.argv[sys.argv.index("--asset") + 1].upper()
    currency = asset if asset in ("BTC", "ETH") else "USDC"

    raw = load("--live" in sys.argv, currency)
    params, ranges, refused, quotes = build_from_snapshot(raw, asset)
    if not params:
        print("no slices fitted; nothing to check")
        return 1

    k_grid = default_k_grid(points=1201, width=1.5)
    report = check_surface(params, k_grid, ranges)

    print(f"\n{asset}: {len(params)} fitted slices, "
          f"{len(refused)} refused, checked on k in [-1.5, 1.5]\n")

    print("Gatheral wing bound  b(1+|rho|) <= 4/T")
    print(f"  {'expiry':>9} {'days':>7} {'b(1+|rho|)':>11} {'4/T':>9} {'':>6}")
    for T, bound in report["wing_bound"].items():
        fit = params[T]
        print(f"  {fit['expiry']:%d%b%y} {T * 365.25:7.1f} {bound['lhs']:11.4f} "
              f"{bound['rhs']:9.2f}   {'ok' if bound['ok'] else 'VIOLATED'}")

    for kind in ("butterfly", "calendar"):
        found = report[kind]
        inside = [v for v in found if v["inside_quoted_range"]]
        outside = [v for v in found if not v["inside_quoted_range"]]
        print(f"\n{kind} arbitrage: {len(found)} region(s) "
              f"-- {len(inside)} inside the quoted strikes, "
              f"{len(outside)} in the extrapolated wings")
        for v in found:
            where = "QUOTED" if v["inside_quoted_range"] else "wings "
            if kind == "butterfly":
                expiry = params[v["T"]]["expiry"]
                print(f"  [{where}] {expiry:%d%b%y} "
                      f"k in [{v['k_from']:+.3f}, {v['k_to']:+.3f}]  "
                      f"worst g = {v['g_min']:+.5f} at k={v['k_worst']:+.3f}  "
                      f"density {v['density_min']:+.5f}")
            else:
                print(f"  [{where}] T {v['T1'] * 365.25:.1f}d -> "
                      f"{v['T2'] * 365.25:.1f}d  "
                      f"k in [{v['k_from']:+.3f}, {v['k_to']:+.3f}]  "
                      f"w falls by {v['shortfall']:.6f} at k={v['k_worst']:+.3f}")

    print("\ntrust radius: how far each slice stays arbitrage-free")
    print("  (the practical Phase 5 output -- query inside this, clamp beyond)")
    print(f"  {'expiry':>9} {'quoted k':>16} {'sound to':>16} {'covers':>7} "
          f"{'strike at k_low':>16}")
    for T, fit in sorted(params.items()):
        radius = trust_radius(fit, ranges[T])
        forward = float(quotes.loc[quotes["T"] == T, "forward"].iloc[0])
        quoted = f"[{ranges[T][0]:+.2f},{ranges[T][1]:+.2f}]"
        sound = f"[{radius['k_low']:+.2f},{radius['k_high']:+.2f}]"
        covers = "yes" if radius.get("covers_quoted_range") else "NO"
        print(f"  {fit['expiry']:%d%b%y} {quoted:>16} {sound:>16} {covers:>7} "
              f"{forward * np.exp(radius['k_low']):16,.0f}")

    if refused:
        print(f"\nslices refused before checking ({len(refused)}):")
        for item in refused:
            print(f"  {item['expiry']:%d%b%y} ({item['T'] * 365.25:5.1f}d)  "
                  f"{item['reason']}")

    total_inside = sum(
        1 for kind in ("butterfly", "calendar")
        for v in report[kind] if v["inside_quoted_range"]
    )
    print("\n" + "=" * 62)
    if total_inside:
        print(f"{total_inside} violation(s) among quoted strikes -- these need "
              f"resolving before Phase 5.")
    else:
        print("No violations among quoted strikes.")
        wings = sum(len(report[k]) for k in ("butterfly", "calendar"))
        if wings:
            print(f"{wings} region(s) in the extrapolated wings: the fits are "
                  f"sound where quotes exist,\nbut should be clamped rather "
                  f"than extrapolated freely in Phase 5.")

    if "--log" in sys.argv:
        write_log(asset, params, report, refused, quotes)
        print(f"\nappended to {LOG_PATH.name}")
    return 0


def write_log(asset, params, report, refused, quotes) -> None:
    stamp = dt.datetime.now(dt.timezone.utc)
    as_of = pd.Timestamp(quotes["timestamp"].iloc[0])
    lines = [
        f"\n## {stamp:%Y-%m-%d %H:%M}Z — {asset}, chain as of {as_of:%Y-%m-%d %H:%M}Z\n",
        f"{len(params)} slices fitted, {len(refused)} refused, "
        f"{len(quotes)} inverted quotes.\n",
    ]
    for kind in ("butterfly", "calendar"):
        found = report[kind]
        inside = [v for v in found if v["inside_quoted_range"]]
        lines.append(
            f"\n**{kind.capitalize()}**: {len(found)} region(s), "
            f"{len(inside)} among quoted strikes.\n"
        )
        for v in found:
            where = "quoted" if v["inside_quoted_range"] else "wings"
            if kind == "butterfly":
                lines.append(
                    f"- [{where}] {params[v['T']]['expiry']:%d%b%y}: "
                    f"g minimum {v['g_min']:+.5f} at k={v['k_worst']:+.3f}, "
                    f"implied density {v['density_min']:+.5f}\n"
                )
            else:
                lines.append(
                    f"- [{where}] {v['T1'] * 365.25:.1f}d → "
                    f"{v['T2'] * 365.25:.1f}d: total variance falls by "
                    f"{v['shortfall']:.6f} at k={v['k_worst']:+.3f}\n"
                )
    bad_wings = [T for T, b in report["wing_bound"].items() if not b["ok"]]
    lines.append(
        f"\n**Wing bound** b(1+|rho|) ≤ 4/T: "
        f"{len(bad_wings)} slice(s) violate it.\n"
    )
    for item in refused:
        lines.append(f"\n**Refused** {item['expiry']:%d%b%y}: {item['reason']}\n")
    lines.append("\n**Resolution**: _(fill in what was changed and why)_\n")

    header = (
        "# Arbitrage violations log\n\n"
        "Every arbitrage violation found in a fitted surface, what caused it,\n"
        "and how it was resolved. Per BUILD_GUIDE.md section 4.3 this log is\n"
        "the point of Phase 4 — a surface that fit cleanly on the first try\n"
        "usually means nobody looked hard enough.\n\n"
        "Generated by `python scripts/check_arbitrage.py --log`.\n"
    )
    if not LOG_PATH.exists():
        LOG_PATH.write_text(header, encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)


if __name__ == "__main__":
    raise SystemExit(main())
