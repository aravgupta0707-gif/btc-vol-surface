"""
Phase 3 checkpoint: fit SVI to a real chain and inspect the result.

    python scripts/fit_surface.py            newest stored BTC snapshot
    python scripts/fit_surface.py --live     pull a fresh chain first
    python scripts/fit_surface.py --asset ETH

The gate from PLAN.md is that the fitted parameters vary smoothly across
adjacent expiries. Erratic jumps in `rho` or `b` between neighbouring
maturities are the classic sign of either noisy input or a calibration that
found a different local minimum on each slice, and both need chasing before
Phase 4.

Fit quality is reported in vol points, not in total variance. An RMSE of
0.0004 in w means nothing to the eye; the same number expressed as
0.15 vol points is immediately comparable to the bid-ask.
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
from surface.svi import MIN_POINTS, fit_svi_slice, svi_total_variance  # noqa: E402


def load(live: bool, asset: str) -> pd.DataFrame:
    currency = "BTC" if asset in ("BTC", "ETH") and asset == "BTC" else asset
    if live:
        from data.fetch_deribit import snapshot_chain, write_snapshot

        frame = snapshot_chain(currency)
        write_snapshot(frame, currency=currency)
        return frame
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / f"{currency.lower()}_chain_*.parquet")))
    if not files:
        print(f"no {currency} snapshots in data/raw/; run with --live")
        raise SystemExit(2)
    print(f"using {Path(files[-1]).name}")
    return pd.read_parquet(files[-1])


def main() -> int:
    asset = "BTC"
    if "--asset" in sys.argv:
        asset = sys.argv[sys.argv.index("--asset") + 1].upper()
    raw = load("--live" in sys.argv, asset)
    raw = raw[raw["underlying_asset"] == asset] if "underlying_asset" in raw else raw

    clean = clean_snapshot(raw)
    clean["iv"] = implied_vol_many(
        clean["forward"], clean["strike"], clean["T"],
        clean["mid_usd"], clean["option_type"].tolist(),
    )
    clean = clean.dropna(subset=["iv"])
    clean["w"] = clean["iv"] ** 2 * clean["T"]

    print(f"{asset}: {len(clean)} inverted quotes across "
          f"{clean['expiry_ts'].nunique()} expiries\n")

    fits, refused = {}, []
    for expiry, group in clean.groupby("expiry_ts"):
        T = float(group["T"].iloc[0])
        fit = fit_svi_slice(group["log_moneyness"].to_numpy(), group["w"].to_numpy())
        if fit["ok"]:
            # Error in vol points, computed exactly rather than by
            # linearising w -> sigma. `a` is the vertical shift of total
            # variance, NOT the ATM variance, and it is routinely negative
            # for wide long-dated smiles -- dividing by sqrt(a) to convert
            # produced RMSEs of 22,000 vol points before this was fixed.
            fitted_w = svi_total_variance(
                group["log_moneyness"].to_numpy(), fit["a"], fit["b"],
                fit["rho"], fit["m"], fit["sigma"],
            )
            fitted_iv = np.sqrt(np.maximum(fitted_w, 0.0) / T)
            fit["rmse_vol"] = float(
                np.sqrt(np.mean((fitted_iv - group["iv"].to_numpy()) ** 2)) * 100
            )
            fit["max_vol_err"] = float(
                np.max(np.abs(fitted_iv - group["iv"].to_numpy())) * 100
            )
            w_atm = float(svi_total_variance(0.0, fit["a"], fit["b"],
                                             fit["rho"], fit["m"], fit["sigma"]))
            fit["atm_vol"] = float(np.sqrt(max(w_atm, 0.0) / T)) * 100
            # ATM skew, dw/dk at k=0: the derived quantity the market
            # actually prices, and far better identified than (m, sigma)
            # individually.
            u = -fit["m"]
            fit["atm_skew"] = float(
                fit["b"] * (fit["rho"] + u / np.sqrt(u**2 + fit["sigma"] ** 2))
            )
            fit["T"] = T
            fits[expiry] = fit
        else:
            refused.append((expiry, T, fit))

    print(f"{'expiry':>9} {'days':>7} {'n':>4} "
          f"{'a':>9} {'b':>8} {'rho':>7} {'m':>8} {'sigma':>7} "
          f"{'rmse(vol pts)':>14}")
    for expiry, fit in fits.items():
        print(f"{expiry:%d%b%y} {fit['T'] * 365.25:7.1f} {fit['n']:4d} "
              f"{fit['a']:9.5f} {fit['b']:8.4f} {fit['rho']:+7.3f} "
              f"{fit['m']:+8.4f} {fit['sigma']:7.4f} {fit['rmse_vol']:14.3f}")

    if refused:
        print(f"\nrefused ({len(refused)}), reported rather than forced:")
        for expiry, T, fit in refused:
            print(f"  {expiry:%d%b%y} ({T * 365.25:5.1f}d)  {fit['reason']}")

    print("\nparameter smoothness across adjacent expiries")
    print("  (erratic jumps mean noisy input or a different local minimum "
          "per slice)")
    frame = pd.DataFrame(fits).T
    if len(frame) >= 3:
        for name in ("a", "b", "rho", "m", "sigma"):
            series = frame[name].astype(float)
            steps = series.diff().abs().dropna()
            scale = series.abs().median() or 1.0
            print(f"    {name:>6}  median |step| = {steps.median():9.5f}  "
                  f"({steps.median() / scale:6.1%} of level)   "
                  f"max |step| = {steps.max():9.5f}")

    print("\n  the same check on derived quantities")
    print("  ((m, sigma) are weakly identified for wide smiles, so the raw")
    print("   parameters can drift while the curve is stable. ATM vol and")
    print("   ATM skew are what the market actually prices.)")
    if len(frame) >= 3:
        for name in ("atm_vol", "atm_skew"):
            series = frame[name].astype(float)
            steps = series.diff().abs().dropna()
            scale = series.abs().median() or 1.0
            print(f"    {name:>9}  median |step| = {steps.median():9.5f}  "
                  f"({steps.median() / scale:6.1%} of level)")

    print("\nfit quality (vol points, against the inverted market quotes)")
    vols = frame["rmse_vol"].astype(float)
    print(f"    median RMSE {vols.median():.3f}, worst {vols.max():.3f}")
    print(f"    worst single quote {frame['max_vol_err'].astype(float).max():.3f}")
    print(f"    slices fitted {len(fits)}, refused {len(refused)}, "
          f"MIN_POINTS = {MIN_POINTS}")

    print("\nATM term structure and skew implied by the fits")
    print(f"    {'expiry':>9} {'days':>7} {'ATM vol':>9} {'ATM skew':>10} "
          f"{'rmse':>7}")
    for expiry, fit in fits.items():
        print(f"    {expiry:%d%b%y} {fit['T'] * 365.25:7.1f} "
              f"{fit['atm_vol']:8.2f}% {fit['atm_skew']:+10.4f} "
              f"{fit['rmse_vol']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
