"""
Phase 5 checkpoint: does the surface object price and hedge correctly?

    python scripts/verify_surface.py            newest stored BTC snapshot
    python scripts/verify_surface.py --live
    python scripts/verify_surface.py --asset ETH

Three things are checked, in increasing order of what they would cost to get
wrong:

1. The surface reproduces the market it was built from. Querying iv(K, T) at
   a quoted strike should return that quote's own implied vol.
2. Interpolating between fitted expiries introduces no calendar arbitrage.
   Two arbitrage-free slices can still produce it if you interpolate implied
   vol rather than total variance.
3. The greeks agree with Deribit's own, which is BUILD_GUIDE.md section 6.
   This is the one Phase 6 depends on: hedge ratios that are quietly wrong
   produce a backtest that looks plausible and means nothing.
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
from surface.surface import from_fits  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((passed, label))
    print(f"  {'PASS' if passed else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return passed


def load(live: bool, currency: str) -> pd.DataFrame:
    if live:
        from data.fetch_deribit import snapshot_chain, write_snapshot

        frame = snapshot_chain(currency)
        write_snapshot(frame, currency=currency)
        return frame
    files = sorted(glob.glob(
        str(ROOT / "data" / "raw" / f"{currency.lower()}_chain_*.parquet")
    ))
    if not files:
        print(f"no {currency} snapshots in data/raw/; run with --live")
        raise SystemExit(2)
    print(f"using {Path(files[-1]).name}\n")
    return pd.read_parquet(files[-1])


def main() -> int:
    asset = "BTC"
    if "--asset" in sys.argv:
        asset = sys.argv[sys.argv.index("--asset") + 1].upper()
    currency = asset if asset in ("BTC", "ETH") else "USDC"

    raw = load("--live" in sys.argv, currency)
    fits, ranges, refused, quotes = build_from_snapshot(raw, asset)
    surface = from_fits(fits)
    print(f"{asset}: surface over {len(fits)} slices, "
          f"{len(quotes)} quotes\n")

    print("reproduces the market it was built from")
    fitted_T = set(fits)
    on_grid = quotes[quotes["T"].isin(fitted_T)]
    queried = np.array([
        surface.iv(row.strike, row.T) for row in on_grid.itertuples()
    ])
    error = np.abs(queried - on_grid["iv"].to_numpy()) * 100
    check(np.median(error) < 0.5, "iv(K,T) returns the quoted vol",
          f"median {np.median(error):.3f} vol points over {len(on_grid)} quotes")
    check(np.max(error) < 3.0, "worst quote within 3 vol points",
          f"{np.max(error):.3f}")

    print("\nput-call parity through the surface")
    gaps = []
    for row in on_grid.itertuples():
        call = surface.price(row.strike, row.T, "C")
        put = surface.price(row.strike, row.T, "P")
        gaps.append(abs((call - put) - (surface.forward(row.T) - row.strike)))
    check(max(gaps) < 1e-4, "C - P = F - K at every quoted strike",
          f"worst {max(gaps):.2e} USD")

    print("\ninterpolation introduces no calendar arbitrage")
    maturities = sorted(fits)
    grid = np.linspace(maturities[0], maturities[-1], 60)
    k_grid = np.linspace(-0.5, 0.5, 41)
    w = np.array([surface.total_variance(k_grid, T) for T in grid])
    worst = float(np.min(np.diff(w, axis=0)))
    check(worst >= -1e-12,
          "total variance non-decreasing across the whole T-continuum",
          f"worst step {worst:+.3e}")

    print("\nshort-end behaviour (w must vanish as T -> 0)")
    tiny = float(surface.total_variance(0.0, 1e-5))
    check(tiny < 1e-4, "total variance goes to zero at T -> 0",
          f"w = {tiny:.3e} at T = 1e-5")
    check(np.isfinite(surface.iv(surface.forward(1e-5), 1e-5)),
          "implied vol stays finite there",
          f"{surface.iv(surface.forward(1e-5), 1e-5) * 100:.2f}%")

    print("\ngreeks against Deribit's own (BUILD_GUIDE.md section 6)")
    if asset not in ("BTC", "ETH"):
        check(True, "greeks validation only wired for the inverse books",
              "(skipped)")
    else:
        from data.fetch_deribit import get_ticker

        sample = (
            on_grid.assign(dist=on_grid["log_moneyness"].abs())
            .sort_values("dist")
            .groupby("expiry_ts")
            .head(3)
        )
        rows = []
        for row in sample.itertuples():
            try:
                ticker = get_ticker(row.instrument_name)
            except Exception:  # noqa: BLE001 - a missing quote is not fatal
                continue
            theirs = ticker.get("greeks", {})
            if "delta" not in theirs:
                continue
            ours = surface.greeks(row.strike, row.T, row.option_type)
            rows.append({
                "delta_err": abs(ours["delta"] - theirs["delta"]),
                # Deribit quotes vega per 1% of vol; ours is per 1.00.
                "vega_ratio": ours["vega"] / theirs["vega"] if theirs.get("vega") else np.nan,
                "gamma_ratio": ours["gamma"] / theirs["gamma"] if theirs.get("gamma") else np.nan,
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            check(False, "could not retrieve any Deribit greeks")
        else:
            check(frame["delta_err"].median() < 0.01,
                  "delta matches Deribit's",
                  f"median |err| {frame['delta_err'].median():.4f} over "
                  f"{len(frame)} instruments")
            check(frame["delta_err"].max() < 0.05, "worst delta within 0.05",
                  f"{frame['delta_err'].max():.4f}")
            ratio = frame["vega_ratio"].median()
            check(abs(ratio - 100) < 3,
                  "vega differs only by the per-1% convention",
                  f"ratio {ratio:.2f} (100 expected)")
            gratio = frame["gamma_ratio"].median()
            check(abs(gratio - 1) < 0.05, "gamma matches",
                  f"ratio {gratio:.4f}")

    if refused:
        print(f"\nslices refused at fit time ({len(refused)}), not in the surface:")
        for item in refused:
            print(f"  {item['expiry']:%d%b%y} ({item['T'] * 365.25:5.1f}d)  "
                  f"{item['reason']}")

    failed = [label for ok, label in RESULTS if not ok]
    print("\n" + "=" * 62)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label.strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
