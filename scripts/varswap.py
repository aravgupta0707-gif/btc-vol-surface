"""
Replicated 30-day variance swaps on BTC and ETH, and the dispersion book.

    python scripts/varswap.py           (needs hedge_frequency.py's fit stage)

Pre-registered in DISPERSION.md, Part 2, before this file existed.

Per asset and monthly archive inception:

- **Strike**: `K_var = (2/T) * integral OTM(K) / K^2 dK` from the fitted
  surface, undiscounted, over the 30-day slice's quoted range only -- what a
  replicating book can buy. The full strike, with the surface's clamped wings
  out to +/-6 standard deviations, is computed alongside to show what the
  truncation leaves out.
- **Payoff**: realised variance of the hourly index on 4-hour returns over
  the next 30 days, annualised. Long P&L `RV - K_var`.
- **Costs**: the strip's half-spreads, paid once, from the archive's measured
  spread curve (vol units times Black vega); and the log contract's futures
  hedge, `q_t = (2/T)(1/F_t - 1/F_0)` per unit variance notional, rebalanced
  every 4 hours at `SLIP_BPS`.

Everything is reported per unit **vega notional** -- variance P&L divided by
`2 sqrt(K_var)`, times 100 -- so the unit is vol points and the two assets
compare directly. The book is long ETH variance, short BTC variance, equal
vega notional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FITS = ROOT / "data" / "research" / "hedge_fits" / "fits_all.parquet"
SPREADS = ROOT / "data" / "research" / "hedge_fits" / "spreads_all.parquet"
OUT = ROOT / "data" / "research" / "varswap.parquet"

TENOR_DAYS = 30
# Primary specification (DISPERSION.md Part 2); override for robustness only:
#   python scripts/varswap.py --step 24 --slip 2
HEDGE_HOURS = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 4
SLIP_BPS = float(sys.argv[sys.argv.index("--slip") + 1]) if "--slip" in sys.argv else 5.0
GRID = 600
FULL_SD = 6.0
YEAR_HOURS = 365.25 * 24
SVI_KEYS = ("a", "b", "rho", "m", "sigma")
ERAS = {"2020-22": (2020, 2022), "2023-26": (2023, 2026)}


def surface_for(day_fits: pd.DataFrame):
    from surface.surface import from_fits

    fits = {
        float(r["T"]): {**{k: float(r[k]) for k in SVI_KEYS}, "forward": float(r["forward"]),
                        "k_range": (float(r["k_min"]), float(r["k_max"]))}
        for _, r in day_fits.iterrows()
    }
    return from_fits(fits)


def strip_integral(surface, T: float, k_lo: float, k_hi: float, weight=None,
                   points: int = GRID) -> float:
    """
    (2/T) * integral over K in [F e^k_lo, F e^k_hi] of f(K) / K^2 dK, where
    f is the OTM option price (puts below F, calls above) by default, or
    `weight(F, K, T, sigma)` if given (the half-spread, for costs).
    """
    from pricing.black76 import price

    F = surface.forward(T)
    k = np.linspace(k_lo, k_hi, points)
    K = F * np.exp(k)
    sig = np.array([surface.iv(x, T) for x in K])
    if weight is None:
        put = np.asarray(price(F, K, T, sig, "P"))
        call = np.asarray(price(F, K, T, sig, "C"))
        f = np.where(K < F, put, call)
    else:
        f = np.asarray(weight(F, K, T, sig, k))
    return float(2.0 / T * np.trapezoid(f / K**2, K))


def variance_strike(surface, T: float) -> dict:
    """Truncated (quoted-range) and full strikes, as variance and as vol."""
    F = surface.forward(T)
    lo, hi = surface.is_extrapolated(F, T)["supported_k_range"]
    atm = surface.iv(F, T)
    sd = atm * np.sqrt(T)
    k_trunc = strip_integral(surface, T, lo, hi)
    k_full = strip_integral(surface, T, min(lo, -FULL_SD * sd), max(hi, FULL_SD * sd), points=2 * GRID)
    return {"forward": F, "atm_vol": atm, "k_lo": lo, "k_hi": hi,
            "lo_sd": lo / sd, "hi_sd": hi / sd,
            "k_var": k_trunc, "k_var_full": k_full}


def half_spread_curve(spreads: pd.DataFrame, T: float):
    """Half-spread in vol units at k, nearest measured expiry, linear in k, ends flat."""
    if spreads.empty:
        return None
    tenors = spreads["T"].unique()
    nearest = tenors[np.argmin(np.abs(tenors - T))]
    s = spreads[spreads["T"] == nearest].sort_values("k")
    ks, hs = s["k"].to_numpy(), s["half_spread_vol"].to_numpy()
    return lambda k: np.interp(k, ks, hs)


def strip_cost(surface, T: float, k_lo: float, k_hi: float, curve) -> float:
    """Variance units: the strip's half-spreads, weighted like the strip."""
    from pricing.black76 import vega

    def weight(F, K, T_, sig, k):
        return curve(k) * np.asarray(vega(F, K, T_, sig))

    return strip_integral(surface, T, k_lo, k_hi, weight=weight)


def path_measures(index: pd.Series, t0: pd.Timestamp, days: int = TENOR_DAYS,
                  step: int = HEDGE_HOURS, slip_bps: float = SLIP_BPS) -> dict:
    """
    Realised variance on `step`-hour returns, and the log-contract hedge's cost.

    The index is sampled every `step` hours from t0 (nearest earlier hour if a
    stamp is missing). Variance is annualised on the hours actually spanned.
    """
    end = t0 + pd.Timedelta(days=days)
    path = index.loc[t0:end].dropna()
    if len(path) < days * 24 * 0.95:
        return {"rv": np.nan, "hedge_cost": np.nan, "obs": len(path)}
    stamps = pd.date_range(t0, end, freq=f"{step}h")
    S = path.reindex(stamps, method="ffill").dropna().to_numpy(float)
    r = np.diff(np.log(S))
    hours = step * len(r)
    rv = float(np.sum(r**2) * YEAR_HOURS / hours)
    T = days / 365.25
    q = (2.0 / T) * (1.0 / S - 1.0 / S[0])
    turnover = float(np.sum(np.abs(np.diff(q)) * S[1:]))
    return {"rv": rv, "hedge_cost": turnover * slip_bps / 1e4, "obs": len(path),
            "ret": float(np.log(S[-1] / S[0]))}


def to_vol_points(variance: float, k_var: float) -> float:
    """Variance P&L per unit vega notional, in vol points."""
    return variance / (2.0 * np.sqrt(k_var)) * 100.0


def build() -> pd.DataFrame:
    from data.history import hourly_funding

    fits = pd.read_parquet(FITS)
    fits = fits[fits["skipped"] == ""]
    spreads = pd.read_parquet(SPREADS)
    T = TENOR_DAYS / 365.25
    rows = []
    for asset in ("BTC", "ETH"):
        idx = hourly_funding(asset, offline=True).set_index("time")["index_price"]
        f_a, s_a = fits[fits["asset"] == asset], spreads[spreads["asset"] == asset]
        for date, day in f_a.groupby("date"):
            base = {"asset": asset, "date": pd.Timestamp(date)}
            if day["T"].max() < T:
                rows.append({**base, "skipped": "no slice beyond 30d"})
                continue
            try:
                surface = surface_for(day)
                strike = variance_strike(surface, T)
            except Exception as exc:  # noqa: BLE001 -- recorded per day
                rows.append({**base, "skipped": f"surface: {type(exc).__name__}"})
                continue
            curve = half_spread_curve(s_a[s_a["date"] == date], T)
            cost_strip = (strip_cost(surface, T, strike["k_lo"], strike["k_hi"], curve)
                          if curve is not None else np.nan)
            t0 = pd.Timestamp(date).tz_localize("UTC")
            path = path_measures(idx, t0)
            rows.append({**base, "skipped": "" if np.isfinite(path["rv"]) else "index gap",
                         **strike, **path, "strip_cost": cost_strip})
    frame = pd.DataFrame(rows)
    ok = frame["skipped"] == ""
    k = frame["k_var"]
    frame.loc[ok, "premium_vp"] = to_vol_points(k - frame["rv"], k)[ok]
    frame.loc[ok, "strip_cost_vp"] = to_vol_points(frame["strip_cost"], k)[ok]
    frame.loc[ok, "hedge_cost_vp"] = to_vol_points(frame["hedge_cost"], k)[ok]
    frame["costs_vp"] = frame["strip_cost_vp"] + frame["hedge_cost_vp"]
    frame["short_net_vp"] = frame["premium_vp"] - frame["costs_vp"]
    frame["long_net_vp"] = -frame["premium_vp"] - frame["costs_vp"]
    frame["k_vol"] = np.sqrt(frame["k_var"]) * 100
    frame["k_vol_full"] = np.sqrt(frame["k_var_full"]) * 100
    frame["rv_vol"] = np.sqrt(frame["rv"]) * 100
    return frame


def t_stat(x) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def describe(x) -> str:
    x = pd.Series(x, dtype=float).dropna()
    return (f"{x.mean():+6.2f} (t {t_stat(x):+5.2f})  median {x.median():+6.2f}  "
            f"worst {x.min():+7.2f}  hit {(x > 0).mean():4.0%}  n={len(x)}")


def book(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[frame["skipped"] == ""]
    w = ok.pivot_table(index="date", columns="asset",
                       values=["premium_vp", "costs_vp", "short_net_vp", "long_net_vp",
                               "k_vol", "rv_vol", "atm_vol"])
    w.columns = [f"{a.lower()}_{c}" for c, a in w.columns]
    w = w.dropna()
    out = pd.DataFrame(index=w.index)
    out["btc_short_gross"] = w["btc_premium_vp"]
    out["eth_long_gross"] = -w["eth_premium_vp"]
    out["gross"] = out["btc_short_gross"] + out["eth_long_gross"]
    out["costs"] = w["btc_costs_vp"] + w["eth_costs_vp"]
    out["net"] = w["btc_short_net_vp"] + w["eth_long_net_vp"]
    out["btc_short_net"] = w["btc_short_net_vp"]
    out["eth_long_net"] = w["eth_long_net_vp"]
    out["atm_spread"] = (w["eth_atm_vol"] - w["btc_atm_vol"]) * 100
    threshold = out["atm_spread"].expanding(min_periods=6).median().shift(1)
    out["signal"] = np.where(threshold.isna(), np.nan, (out["atm_spread"] < threshold).astype(float))
    return out


def report(frame: pd.DataFrame) -> None:
    ok = frame[frame["skipped"] == ""]
    print(f"Replicated 30d variance swaps, {HEDGE_HOURS}h hedge, {SLIP_BPS:.0f} bps; "
          "vol points per unit vega notional\n")
    skipped = frame[frame["skipped"] != ""].groupby(["asset", "skipped"]).size()
    if len(skipped):
        print("  skipped:", skipped.to_dict())
    for asset, g in ok.groupby("asset"):
        print(f"\n  {asset}: n={len(g)}  strike {g['k_vol'].mean():.1f} vol (full {g['k_vol_full'].mean():.1f}), "
              f"ATM {g['atm_vol'].mean() * 100:.1f}, realised {g['rv_vol'].mean():.1f}")
        print(f"       quoted range {g['lo_sd'].median():+.2f} to {g['hi_sd'].median():+.2f} sd (median); "
              f"truncation drops {(1 - g['k_var'] / g['k_var_full']).median():.1%} of the full strike")
        print(f"       costs: strip {g['strip_cost_vp'].mean():.2f}, hedge {g['hedge_cost_vp'].mean():.2f} vol pts")
        print("       short premium, gross  " + describe(g["premium_vp"]))
        print("       short, net            " + describe(g["short_net_vp"]))
        print("       long, net             " + describe(g["long_net_vp"]))
        for era, (lo, hi) in ERAS.items():
            e = g[(g["date"].dt.year >= lo) & (g["date"].dt.year <= hi)]
            print(f"       {era} short gross   " + describe(e["premium_vp"]))

    b = book(frame)
    on = b[b["signal"] == 1]
    print(f"\n  Book: long ETH variance, short BTC variance, equal vega; {len(b)} paired months")
    print("  V2  gross, every month       " + describe(b["gross"]))
    print("  V4  net, every month         " + describe(b["net"]))
    print("      gross, entry rule ON     " + describe(on["gross"]))
    print("      net, entry rule ON       " + describe(on["net"]))
    print(f"      mean costs, both legs    {b['costs'].mean():.2f} vol pts")
    print(f"  V3  corr of the legs' gross P&L (short BTC vs short ETH): "
          f"{np.corrcoef(b['btc_short_gross'], -b['eth_long_gross'])[0, 1]:+.2f}")
    print(f"      sd: BTC short {b['btc_short_gross'].std():.2f}, ETH long {b['eth_long_gross'].std():.2f}, "
          f"book {b['gross'].std():.2f}")
    for era, (lo, hi) in ERAS.items():
        e = b[(b.index.year >= lo) & (b.index.year <= hi)]
        print(f"      {era} gross " + describe(e["gross"]))
        print(f"      {era} net   " + describe(e["net"]))


def main() -> int:
    if not FITS.exists():
        raise SystemExit(f"no {FITS.relative_to(ROOT)}: run hedge_frequency.py's fit stage")
    frame = build()
    primary = HEDGE_HOURS == 4 and SLIP_BPS == 5.0
    out = OUT if primary else OUT.with_name(f"varswap_{HEDGE_HOURS}h_{SLIP_BPS:g}bps.parquet")
    frame.to_parquet(out)
    report(frame)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
