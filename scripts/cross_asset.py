"""
BTC against ETH implied vol: does the spread mean-revert, and is it a trade?

    python scripts/cross_asset.py        (needs scripts/vrp_surface.py first)

Reads the ATM implied vol and the realised vol over each option's life from
the surface study, so every number comes from the same 79 paired fitted
surfaces rather than a second, slightly different fit.

Three questions:

1. **Mean reversion.** Month to month, how persistent is the ETH-BTC implied
   vol spread? An AR(1) on the monthly series gives a half-life.
2. **Relative forecasting.** Does the implied ETH/BTC vol ratio forecast the
   realised ratio over the next 30 days better than the trailing realised
   ratio does?
3. **Dispersion.** Short ETH vol against long BTC vol at equal vega earns,
   roughly, (ETH implied - realised) - (BTC implied - realised). Is it
   positive on average -- and larger when the spread is wide? "Wide" is
   judged against the spread's own expanding median, using only months
   already seen, so the split carries no lookahead.

Implied correlation is not attempted: it needs a quanto or cross option
Deribit does not list, and a number inferred without one would be assumed,
not measured. Realised correlation is reported for context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SURFACE = ROOT / "data" / "research" / "vrp_surface.parquet"
YEAR_HOURS = 365.25 * 24


def trailing_realized(index: pd.Series, t0, days: int) -> float:
    window = index.loc[t0 - pd.Timedelta(days=days):t0].dropna()
    if len(window) < days * 24 * 0.95:
        return np.nan
    return float(np.std(np.diff(np.log(window.to_numpy(float))), ddof=1) * np.sqrt(YEAR_HOURS))


def realized_corr(a: pd.Series, b: pd.Series, t0, days: int) -> float:
    span = slice(t0, t0 + pd.Timedelta(days=days))
    ra = np.log(a.loc[span]).diff()
    rb = np.log(b.loc[span]).diff()
    joined = pd.concat([ra, rb], axis=1).dropna()
    return float(joined.corr().iloc[0, 1]) if len(joined) > days * 20 else np.nan


def build(tenor: int = 30) -> pd.DataFrame:
    from data.history import hourly_funding

    s = pd.read_parquet(SURFACE)
    s = s[(s["skipped"] == "") & (s["z"] == 0.0) & (s["tenor"] == tenor)]
    wide = s.pivot_table(index="date", columns="asset", values=["atm_iv", "realized"])
    wide.columns = [f"{a.lower()}_{v}" for v, a in wide.columns]
    wide = wide.dropna().sort_index()

    idx = {a: hourly_funding(a).set_index("time")["index_price"] for a in ("BTC", "ETH")}
    t0s = [pd.Timestamp(d, tz="UTC") for d in wide.index]
    for a in ("BTC", "ETH"):
        wide[f"{a.lower()}_trailing"] = [trailing_realized(idx[a], t, tenor) for t in t0s]
    wide["corr"] = [realized_corr(idx["BTC"], idx["ETH"], t, tenor) for t in t0s]
    wide["spread"] = (wide.eth_atm_iv - wide.btc_atm_iv) * 100
    wide["iv_ratio"] = wide.eth_atm_iv / wide.btc_atm_iv
    wide["rv_ratio"] = wide.eth_realized / wide.btc_realized
    wide["trail_ratio"] = wide.eth_trailing / wide.btc_trailing
    wide["dispersion"] = ((wide.eth_atm_iv - wide.eth_realized)
                          - (wide.btc_atm_iv - wide.btc_realized)) * 100
    wide["spread_median_so_far"] = wide["spread"].expanding().median().shift(1)
    return wide


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def report(w: pd.DataFrame) -> None:
    w = w.assign(year=pd.to_datetime(w.index).year)
    print(f"{len(w)} paired months, 30-day ATM implied vol, "
          f"{w.index.min()} .. {w.index.max()}")

    print("\nby year (vol points; ratio = ETH/BTC):")
    t = w.groupby("year").agg(btc_iv=("btc_atm_iv", "mean"), eth_iv=("eth_atm_iv", "mean"),
                              spread=("spread", "mean"), iv_ratio=("iv_ratio", "mean"),
                              rv_ratio=("rv_ratio", "mean"), corr=("corr", "mean"),
                              dispersion=("dispersion", "mean"))
    t[["btc_iv", "eth_iv"]] *= 100
    print(t.round(2).to_string())

    s = w["spread"].to_numpy()
    x, y = s[:-1], s[1:]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    phi = beta[1]
    half = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
    print(f"\n1. mean reversion: AR(1) on the monthly spread, phi = {phi:.2f} (se {se:.2f}); "
          f"half-life {half:.1f} months; long-run level {beta[0] / (1 - phi):+.1f} vol pts")

    g = w.dropna(subset=["rv_ratio", "trail_ratio", "iv_ratio"])
    e_iv = np.sqrt(((g.iv_ratio - g.rv_ratio) ** 2).mean())
    e_tr = np.sqrt(((g.trail_ratio - g.rv_ratio) ** 2).mean())
    print(f"2. forecasting the realised ETH/BTC vol ratio over the next 30 days (n={len(g)}):"
          f"\n   RMSE implied ratio {e_iv:.3f}, trailing realised ratio {e_tr:.3f}; "
          f"corr(implied, realised) {g.iv_ratio.corr(g.rv_ratio):+.2f}, "
          f"corr(trailing, realised) {g.trail_ratio.corr(g.rv_ratio):+.2f}"
          f"\n   mean implied ratio {g.iv_ratio.mean():.3f} vs realised {g.rv_ratio.mean():.3f}")

    d = w.dropna(subset=["dispersion", "spread_median_so_far"])
    wide_ = d[d.spread > d.spread_median_so_far]
    narrow = d[d.spread <= d.spread_median_so_far]
    print(f"3. dispersion, short ETH vol / long BTC vol at equal vega (vol pts earned, mean):"
          f"\n   all months      {w.dispersion.mean():+5.2f}  (t {_t(w.dispersion):+.2f}, n={w.dispersion.notna().sum()})"
          f"\n   spread wide     {wide_.dispersion.mean():+5.2f}  (t {_t(wide_.dispersion):+.2f}, n={len(wide_)})"
          f"\n   spread narrow   {narrow.dispersion.mean():+5.2f}  (t {_t(narrow.dispersion):+.2f}, n={len(narrow)})"
          f"\n   (wide/narrow judged against the spread's median over months already seen)")
    print("\nVol points, mid, before costs. Equal-vega is approximate; this sizes the "
          "premium, not a trade.")


def main() -> int:
    if not SURFACE.exists():
        print("run scripts/vrp_surface.py first")
        return 2
    report(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
