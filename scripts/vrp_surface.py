"""
Where on the surface does the variance premium live -- and where did it go?

    python scripts/vrp_surface.py            build, run, report
    python scripts/vrp_surface.py --report   re-print from saved results

The DVOL study measured one number per day: 30-day ATM-ish implied against
realized, decaying from +18.6 vol points (2021) to +2.9 (2026). This splits
that premium across the smile and across tenor.

For every archive day, the 00:00 surface prices a set of options at fixed
standardised moneyness, K = F * exp(z * sigma_atm * sqrt(T)): OTM puts for
z < 0, an ATM straddle at z = 0, OTM calls for z > 0, at 7 and 30 days. Each
is sold at the surface's mid, delta-hedged every 4 hours at its own implied
vol along the actual hourly Deribit index, and settled at intrinsic against
the index at expiry. The seller's hedged P&L is the premium that strike
carried (Bakshi-Kapadia delta-hedged gains). Reported as a share of the
option's premium, so a 5% put and an ATM straddle compare, and in bps of spot.

What this is not: it is priced at mid, with no spread, so it locates the
premium rather than sizing a trade. Wing spreads are wide on Deribit and
would eat much of a wing premium. And a strike outside where its slice was
quoted is priced from a clamped wing -- a flat extrapolation, not a market
price -- so those rows are kept in the file but excluded from every average.

Samples: one inception per month. 30-day options from consecutive months
barely overlap, 7-day ones not at all, so each row is close to independent;
per-era means come with t-statistics to keep that honest.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIST_DIR = ROOT / "data" / "historical"
OUT_PATH = ROOT / "data" / "research" / "vrp_surface.parquet"
ASSETS = ("BTC", "ETH")
TENOR_DAYS = (7, 30)
Z_GRID = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
HEDGE_HOURS = 4
YEAR_HOURS = 365.25 * 24
ERAS = {"2020-22": (2020, 2022), "2023-26": (2023, 2026)}


def hedged_seller_pnl(path: np.ndarray, hours: np.ndarray, K: float, T0: float,
                      sigma: float, carry: float, option: str) -> float:
    """
    Seller's P&L on one option, delta-hedged every HEDGE_HOURS on the index.

    `path` is the index at `hours` (hours since inception, last = expiry).
    The hedge is sized on the forward, F = S * exp(carry * T_remaining), the
    same convention the option is priced in; at expiry F = S. Delta is
    Black-76 at the option's own inception implied vol.
    """
    from pricing.black76 import intrinsic, price

    F = path * np.exp(carry * np.maximum(T0 - hours / YEAR_HOURS, 0.0))
    T_left = np.maximum(T0 - hours / YEAR_HOURS, 0.0)
    premium = price(F[0], K, T0, sigma, option)
    live = T_left[:-1] > 0
    d1 = np.where(live, (np.log(F[:-1] / K) + 0.5 * sigma**2 * T_left[:-1])
                  / (sigma * np.sqrt(np.where(live, T_left[:-1], 1.0))), 0.0)
    delta = norm.cdf(d1) if option == "C" else norm.cdf(d1) - 1.0
    hedge_gain = np.sum(delta * np.diff(F))
    long_gain = float(intrinsic(path[-1], K, option)) - premium - hedge_gain
    return -long_gain, float(premium)


def run_day(args) -> list[dict]:
    path_file, index = args
    from surface.build import build_from_snapshot
    from surface.surface import from_fits

    asset = path_file.name.split("_")[0].upper()
    raw = pd.read_parquet(path_file)
    t0 = pd.Timestamp(raw["timestamp"].min())
    base = {"date": t0.date(), "asset": asset}
    try:
        fits, _, _, _ = build_from_snapshot(raw[raw["timestamp"] == t0], asset)
        surface = from_fits(fits)
    except Exception as exc:  # noqa: BLE001 - recorded per day
        return [dict(base, skipped=f"no surface: {type(exc).__name__}")]

    series = index.set_index("time")["index_price"]
    rows = []
    for days in TENOR_DAYS:
        tE = t0 + pd.Timedelta(days=days)
        window = series.loc[t0:tE].dropna()
        if window.empty or window.index[-1] < tE or window.index[0] > t0:
            rows.append(dict(base, tenor=days, skipped="expiry not yet reached or index gap"))
            continue
        step = window.iloc[::HEDGE_HOURS]
        if step.index[-1] != window.index[-1]:
            step = pd.concat([step, window.iloc[[-1]]])
        hours = ((step.index - t0) / pd.Timedelta(hours=1)).to_numpy(float)
        spots = step.to_numpy(float)
        T0 = days / 365.25
        F0 = surface.forward(T0)
        carry = np.log(F0 / spots[0]) / T0
        atm = surface.iv(F0, T0)
        realized = float(np.std(np.diff(np.log(window.to_numpy(float))), ddof=1)
                         * np.sqrt(YEAR_HOURS))
        for z in Z_GRID:
            K = F0 * np.exp(z * atm * np.sqrt(T0))
            flags = surface.is_extrapolated(K, T0)
            sigma = surface.iv(K, T0)
            legs = ("P", "C") if z == 0 else (("P",) if z < 0 else ("C",))
            pnl = prem = 0.0
            for leg in legs:
                p, q = hedged_seller_pnl(spots, hours, K, T0, sigma, carry, leg)
                pnl, prem = pnl + p, prem + q
            rows.append(dict(
                base, tenor=days, z=z, skipped="", strike=K, forward=F0, spot=spots[0],
                iv=sigma, atm_iv=atm, realized=realized,
                supported=not (flags["outside_supported_range"] or flags["beyond_fitted_maturities"]),
                premium_bps=prem / spots[0] * 1e4, pnl_bps=pnl / spots[0] * 1e4,
                pnl_share=pnl / prem if prem > 0 else np.nan,
            ))
    return rows


def run_all() -> pd.DataFrame:
    from data.history import hourly_funding

    rows = []
    for asset in ASSETS:
        index = hourly_funding(asset)[["time", "index_price"]]
        files = sorted(HIST_DIR.glob(f"{asset.lower()}_chain_*.parquet"))
        print(f"{asset}: {len(files)} archive days", flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            for day_rows in pool.map(run_day, [(f, index) for f in files]):
                rows.extend(day_rows)
    frame = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_PATH, index=False)
    return frame


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def report(frame: pd.DataFrame) -> None:
    done = frame[(frame["skipped"] == "") & frame["supported"].astype(bool)].copy()
    done["year"] = pd.to_datetime(done["date"]).dt.year
    done["era"] = np.where(done["year"] <= 2022, "2020-22", "2023-26")
    excluded = frame[(frame["skipped"] == "") & ~frame["supported"].astype(bool)]
    skipped = frame[frame["skipped"] != ""]

    label = {-2.0: "put z-2.0", -1.5: "put z-1.5", -1.0: "put z-1.0", -0.5: "put z-0.5",
             0.0: "straddle", 0.5: "call z+0.5", 1.0: "call z+1.0", 1.5: "call z+1.5",
             2.0: "call z+2.0"}
    for asset in ASSETS:
        for days in TENOR_DAYS:
            grp = done[(done.asset == asset) & (done.tenor == days)]
            if grp.empty:
                continue
            ref = grp[grp.z == 0]
            print(f"\n{asset} {days}-day   (ATM implied {ref.atm_iv.mean()*100:.1f} vs realized "
                  f"{ref.realized.mean()*100:.1f} over the option's life, {ref.date.nunique()} inceptions)")
            print(f"  {'':11} {'':>4} " + "".join(f"{e:>24}" for e in ERAS))
            print(f"  {'strike':11} {'n':>4} " + "".join(
                f"{'kept':>8}{'bps':>8}{'t':>8}" for _ in ERAS))
            for z in Z_GRID:
                sub = grp[grp.z == z]
                if sub.empty:
                    continue
                cells = ""
                for era in ERAS:
                    e = sub[sub.era == era]
                    cells += (f"{e.pnl_share.mean()*100:+7.0f}%{e.pnl_bps.mean():+8.1f}{_t(e.pnl_bps):8.2f}"
                              if len(e) > 2 else f"{'-':>24}")
                print(f"  {label[z]:11} {len(sub):4d} {cells}")

    print("\nkept = seller's delta-hedged P&L as a share of the premium received (mean);"
          "\nbps = the same in basis points of spot; t = t-statistic of the bps mean."
          "\nMid prices, no spread: this locates the premium, it does not size a trade.")
    print(f"excluded as extrapolated (strike outside quoted range): {len(excluded)} rows; "
          f"skipped: {skipped['skipped'].value_counts().to_dict()}")


def main() -> int:
    if "--report" in sys.argv:
        report(pd.read_parquet(OUT_PATH))
        return 0
    report(run_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
