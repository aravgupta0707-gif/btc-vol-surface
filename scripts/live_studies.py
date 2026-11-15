"""
Three studies on the live (Track A) surfaces, pre-registered in LIVE_STUDIES.md.

    python scripts/live_studies.py

A. sticky strike vs sticky moneyness
B. funding and the call wing, out of sample (H7b)
C. does skew lead returns or realised downside?

Reads the 12h surface grid cached by `surface_pca.py` (run that first to
extend it) and the hourly index from `data/history.hourly_funding`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "data" / "research"

TENORS_A = (7, 14, 30)
STEP = pd.Timedelta(hours=12)
CUTOFF = pd.Timestamp("2026-11-05 20:00", tz="UTC")   # start of the sell-off
LAGS_12H = 4
LAGS_OVERLAP = 14
FWD_DAYS = 7
YEAR_DAYS = 365.25


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ inputs

def load_grid() -> pd.DataFrame:
    return pd.read_parquet(RES / "surface_grid.parquet")


def hourly_index(asset: str) -> pd.Series:
    f = pd.read_parquet(RES / f"{asset.lower()}_funding_hourly.parquet")
    return f.set_index(pd.to_datetime(f["time"], utc=True))["index_price"].sort_index()


def daily_funding(asset: str) -> pd.Series:
    """Daily mean perp funding, annualised %, indexed by UTC date (naive)."""
    f = pd.read_parquet(RES / f"{asset.lower()}_funding_hourly.parquet")
    rate = f.set_index(pd.to_datetime(f["time"], utc=True))["interest_8h"] * 3 * 365 * 100
    daily = rate.resample("D").mean()
    daily.index = daily.index.tz_localize(None)
    return daily


def lagged_trailing(daily: pd.Series, days: int) -> pd.Series:
    """Mean over the `days` UTC days ending the day *before* each date."""
    return daily.rolling(days, min_periods=days).mean().shift(1)


def at_slots(series: pd.Series, slots: pd.DatetimeIndex) -> np.ndarray:
    """A daily series read at each slot's UTC date."""
    dates = slots.tz_convert("UTC").tz_localize(None).normalize()
    return series.reindex(dates).to_numpy()


# ------------------------------------------------------------------ A

def fixed_strike_change(row_t1: pd.Series, atm_t1: float, r: float, T: float,
                        iv_t0_atm: float) -> float:
    """
    sigma_{t+1} at the strike that was ATM at t, minus sigma_t there.

    That strike sits at k' = ln(S_t / S_{t+1}) = -r at t+1 (forward ~ spot),
    so z' = -r / (atm_{t+1} sqrt T). Read the t+1 smile there by linear
    interpolation in z; NaN if z' falls off the grid.
    """
    z = np.asarray(row_t1.index, float)
    iv = np.asarray(row_t1.to_numpy(), float)
    zp = -r / (atm_t1 / 100.0 * np.sqrt(T))
    if zp < z.min() or zp > z.max():
        return float("nan")
    return float(np.interp(zp, z, iv) - iv_t0_atm)


def study_a(grid: pd.DataFrame, asset: str) -> pd.DataFrame:
    sp = _script("surface_pca")
    wide, spot, _ = sp.to_wide(grid, asset)
    rows = []
    for t0 in wide.index:
        t1 = t0 + STEP
        if t1 not in wide.index:
            continue
        r = float(np.log(spot[t1] / spot[t0]))
        for tenor in TENORS_A:
            if (tenor, 0.0) not in wide.columns:
                continue
            row1 = wide.loc[t1, tenor]
            atm0, atm1 = wide.loc[t0, (tenor, 0.0)], wide.loc[t1, (tenor, 0.0)]
            rows.append({"asset": asset, "t1": t1, "tenor": tenor, "r": r,
                         "d_fixed_z": atm1 - atm0,
                         "d_fixed_k": fixed_strike_change(row1, atm1, r, tenor / YEAR_DAYS, atm0)})
    return pd.DataFrame(rows)


def moneyness_share(beta_k: float, beta_z: float) -> float:
    """0 = sticky strike (beta_K = 0), 1 = sticky moneyness (beta_z = 0)."""
    return beta_k / (beta_k - beta_z)


# ------------------------------------------------------------------ B, C

def skew_frame(grid: pd.DataFrame, asset: str, tenor: int = 30) -> pd.DataFrame:
    sp = _script("surface_pca")
    wide, spot, _ = sp.to_wide(grid, asset)
    out = pd.DataFrame({
        "skew": wide[(tenor, -1.0)] - wide[(tenor, 1.0)],
        "atm": wide[(tenor, 0.0)],
        "spot": spot,
    })
    return out


def downside_share(index: pd.Series, t0: pd.Timestamp, days: int = FWD_DAYS) -> float:
    """Down semivariance / total realised variance of hourly returns over (t0, t0+days]."""
    w = index.loc[t0: t0 + pd.Timedelta(days=days)].dropna()
    if len(w) < days * 24 * 0.95:
        return float("nan")
    r = np.diff(np.log(w.to_numpy(float)))
    total = float((r**2).sum())
    return float((r[r < 0] ** 2).sum() / total) if total > 0 else float("nan")


def next_return(spot: pd.Series) -> pd.Series:
    """Log return from each slot to the slot 12h later; NaN across a gap."""
    later = spot.reindex(spot.index + STEP).to_numpy()
    return pd.Series(np.log(later / spot.to_numpy()), index=spot.index)


# ------------------------------------------------------------------ report

def main() -> int:
    vh = _script("vrp_hypotheses")
    nw = vh.nw_ols_multi
    grid = load_grid()
    out = {}

    print("=" * 78 + "\nA. sticky strike vs sticky moneyness (12h changes, ATM)")
    for asset in ("BTC", "ETH"):
        a = study_a(grid, asset).dropna()
        out[f"A_{asset}"] = a
        for tenor in TENORS_A:
            d = a[a["tenor"] == tenor]
            bz, tz = nw(d["d_fixed_z"], d["r"] * 100, LAGS_12H)
            bk, tk = nw(d["d_fixed_k"], d["r"] * 100, LAGS_12H)
            cz = np.corrcoef(d["d_fixed_z"], d["r"])[0, 1]
            ck = np.corrcoef(d["d_fixed_k"], d["r"])[0, 1]
            m = moneyness_share(bk[0], bz[0])
            print(f"  {asset} {tenor:>2}d n={len(d):3}  beta_z {bz[0]:+.3f} (t {tz[0]:+.2f})  "
                  f"beta_K {bk[0]:+.3f} (t {tk[0]:+.2f})  m {m:+.2f}   corr z {cz:+.2f}  K {ck:+.2f}")
        d = a[(a["tenor"] == 30) & (a["t1"] > CUTOFF)]
        if len(d) > 3:
            bk, tk = nw(d["d_fixed_k"], d["r"] * 100, 1)
            print(f"  {asset} 30d sell-off only n={len(d)}: beta_K {bk[0]:+.3f} (t {tk[0]:+.2f}), "
                  f"mean r {d['r'].mean():+.2%}, mean d_fixed_k {d['d_fixed_k'].mean():+.2f}")
    print("  (beta in vol points per 1% spot move)")

    print("\n" + "=" * 78 + "\nB. skew on lagged trailing funding (30d skew, vol pts per funding pt)")
    for asset in ("BTC", "ETH"):
        s = skew_frame(grid, asset)
        daily = daily_funding(asset)
        s["f7"] = at_slots(lagged_trailing(daily, 7), s.index)
        s["f30"] = at_slots(lagged_trailing(daily, 30), s.index)
        s = s.dropna()
        for col in ("f7", "f30"):
            b, t = nw(s["skew"], s[col], LAGS_OVERLAP)
            b2, t2 = nw(s["skew"], s[[col, "atm"]], LAGS_OVERLAP)
            print(f"  {asset} {col:>3} n={len(s):3}  slope {b[0]:+.3f} (t {t[0]:+.2f});  "
                  f"with ATM {b2[0]:+.3f} (t {t2[0]:+.2f});  corr {s['skew'].corr(s[col]):+.2f}  "
                  f"sd({col}) {s[col].std():.2f}")
        out[f"B_{asset}"] = s

    print("\n" + "=" * 78 + "\nC. does skew lead?")
    for asset in ("BTC", "ETH"):
        s = skew_frame(grid, asset)
        idx = hourly_index(asset)
        s["r_next"] = next_return(s["spot"]) * 100
        s["d_skew"] = s["skew"] - s["skew"].reindex(s.index - STEP).to_numpy()
        s["down_share"] = [downside_share(idx, t) for t in s.index]
        c1 = s.dropna(subset=["skew", "r_next"])
        b, t = nw(c1["r_next"], c1["skew"], LAGS_12H)
        print(f"  C1 {asset} r(next 12h, %) on skew        n={len(c1):3}  slope {b[0]:+.3f}  t {t[0]:+.2f}")
        c2 = s.dropna(subset=["skew", "down_share"])
        b, t = nw(c2["down_share"], c2["skew"], LAGS_OVERLAP)
        print(f"  C2 {asset} next-7d downside share on skew n={len(c2):3}  slope {b[0]:+.4f}  t {t[0]:+.2f}"
              f"   (mean share {c2['down_share'].mean():.2f})")
        c3 = s.dropna(subset=["d_skew", "r_next"])
        b, t = nw(c3["r_next"], c3["d_skew"], LAGS_12H)
        print(f"  C3 {asset} r(next 12h, %) on d_skew      n={len(c3):3}  slope {b[0]:+.3f}  t {t[0]:+.2f}")
        cc = s.dropna(subset=["d_skew"])
        r_now = np.log(cc["spot"] / s["spot"].reindex(cc.index - STEP).to_numpy()) * 100
        ok = r_now.notna()
        print(f"     {asset} contemporaneous corr(d_skew, r) {np.corrcoef(cc['d_skew'][ok], r_now[ok])[0, 1]:+.2f}")
        out[f"C_{asset}"] = s

    pd.concat({k: v.reset_index() for k, v in out.items()}, names=["part"]).to_parquet(
        RES / "live_studies.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
