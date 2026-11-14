"""
Long ETH vol, short BTC vol, with real spreads and a real hedge.

    python scripts/dispersion.py        (needs hedge_frequency.py's cells)

Pre-registered in DISPERSION.md before this file existed. Each leg is a cell
of the hedging sweep -- same surface, same index path, same measured entry
spread, same hedge simulation -- so this is a re-aggregation and cannot
drift from the §12 numbers:

- short BTC straddle: the cell's own `net_bps` (the sweep is written from the
  seller's side);
- long ETH straddle: the mirror, `-gross - spread - slippage`. A buyer earns
  minus the seller's hedged P&L and pays the same spread and hedge costs.

Equal spot notional per leg. For an ATM straddle that is vega-neutral to
within the ratio `phi(sigma_E sqrt(T)/2) / phi(sigma_B sqrt(T)/2)`, which is
reported rather than assumed.

The entry rule trades a month only when the ETH-BTC ATM implied spread is
below the median of the spread on earlier inceptions (expanding, at least
`MIN_HISTORY` of them). Earlier inceptions' *implied* vols are known at entry
-- nothing here waits on a position to settle -- so unlike §12.2 there is no
settlement lag to respect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CELL_DIR = ROOT / "data" / "research" / "hedge_cells"
OUT = ROOT / "data" / "research" / "dispersion.parquet"

PRIMARY = {"tenor": 30, "z": 0.0, "policy": "time 4h", "slip_bps": 5.0}
MIN_HISTORY = 6
ERAS = {"2020-22": (2020, 2022), "2023-26": (2023, 2026)}


def load_cells() -> pd.DataFrame:
    files = sorted(CELL_DIR.glob("cells_*.parquet"))
    if not files:
        raise SystemExit(f"no cells in {CELL_DIR}: run scripts/hedge_frequency.py first")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    frame = frame[(frame["skipped"] == "") & frame["supported"].astype(bool)]
    frame = frame.assign(date=pd.to_datetime(frame["date"]))
    return frame


def legs(cells: pd.DataFrame, tenor: int, z: float, policy: str, slip_bps: float) -> pd.DataFrame:
    """One row per inception with both assets present: each leg's P&L and the book's."""
    sub = cells[(cells["tenor"] == tenor) & (cells["z"] == z)
                & (cells["policy"] == policy) & (cells["slip_bps"] == slip_bps)]
    cols = ["gross_bps", "net_bps", "spread_bps", "slippage_bps", "atm_iv", "premium_bps"]
    wide = sub.pivot_table(index="date", columns="asset", values=cols)
    wide.columns = [f"{a.lower()}_{c}" for c, a in wide.columns]
    need = [f"{a}_{c}" for a in ("btc", "eth") for c in cols]
    wide = wide.dropna(subset=[c for c in need if c in wide.columns]).sort_index()
    if wide.empty or not set(need) <= set(wide.columns):
        return pd.DataFrame()

    out = pd.DataFrame(index=wide.index)
    out["btc_iv"] = wide["btc_atm_iv"] * 100
    out["eth_iv"] = wide["eth_atm_iv"] * 100
    out["spread"] = out["eth_iv"] - out["btc_iv"]
    # Seller of BTC: the cell as written.
    out["btc_short_gross"] = wide["btc_gross_bps"]
    out["btc_short_net"] = wide["btc_net_bps"]
    # Buyer of ETH: the mirror of the seller's hedged P&L, paying the same costs.
    out["eth_long_gross"] = -wide["eth_gross_bps"]
    out["eth_long_net"] = -wide["eth_gross_bps"] - wide["eth_spread_bps"] - wide["eth_slippage_bps"]
    out["costs"] = (wide["btc_spread_bps"] + wide["btc_slippage_bps"]
                    + wide["eth_spread_bps"] + wide["eth_slippage_bps"])
    out["book_gross"] = out["btc_short_gross"] + out["eth_long_gross"]
    out["book_net"] = out["btc_short_net"] + out["eth_long_net"]
    T = tenor / 365.25
    out["vega_ratio"] = (norm.pdf(wide["eth_atm_iv"] * np.sqrt(T) / 2)
                         / norm.pdf(wide["btc_atm_iv"] * np.sqrt(T) / 2))
    out["signal"] = entry_signal(out["spread"])
    return out


def entry_signal(spread: pd.Series, min_history: int = MIN_HISTORY) -> pd.Series:
    """
    True where the spread is below the median of all *earlier* observations.

    NaN (no decision) until `min_history` earlier observations exist. The
    shift is what keeps today's spread out of today's threshold.
    """
    threshold = spread.expanding(min_periods=min_history).median().shift(1)
    signal = (spread < threshold).astype(object)
    signal[threshold.isna()] = np.nan
    return signal


def t_stat(x) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def describe(x: pd.Series) -> str:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return f"n={len(x)}"
    return (f"{x.mean():+7.1f} (t {t_stat(x):+5.2f})  median {x.median():+7.1f}  "
            f"worst {x.min():+7.1f}  hit {(x > 0).mean():4.0%}  n={len(x)}")


def era_of(index: pd.DatetimeIndex) -> pd.Series:
    years = pd.Series(index.year, index=index)
    return years.map(lambda y: next((k for k, (lo, hi) in ERAS.items() if lo <= y <= hi), None))


def realized_corr(dates, days: int = 30) -> pd.Series:
    """Hourly return correlation of BTC and ETH over each position's life."""
    from data.history import hourly_funding

    idx = {a: hourly_funding(a).set_index("time")["index_price"] for a in ("BTC", "ETH")}
    out = {}
    for d in dates:
        t0 = pd.Timestamp(d, tz="UTC")
        span = slice(t0, t0 + pd.Timedelta(days=days))
        r = pd.concat([np.log(idx[a].loc[span]).diff() for a in ("BTC", "ETH")], axis=1).dropna()
        out[d] = float(r.corr().iloc[0, 1]) if len(r) > days * 20 else np.nan
    return pd.Series(out)


def report(book: pd.DataFrame) -> None:
    on = book[book["signal"] == True]  # noqa: E712 -- object column with NaN
    decided = book[book["signal"].notna()]
    print(f"Primary: {PRIMARY}")
    print(f"  {len(book)} paired inceptions {book.index[0]:%Y-%m} to {book.index[-1]:%Y-%m}; "
          f"entry rule decided on {len(decided)}, trades {len(on)}")
    print(f"  ETH-BTC implied spread: mean {book['spread'].mean():+.1f} vol pts, "
          f"vega ratio ETH/BTC {book['vega_ratio'].mean():.4f} "
          f"(range {book['vega_ratio'].min():.4f}-{book['vega_ratio'].max():.4f})")
    if "corr" in book:
        print(f"  realised BTC/ETH correlation over each position: mean {book['corr'].mean():.2f}")

    print("\n  D1  gross, every month           " + describe(book["book_gross"]))
    print("  D2  net, every month             " + describe(book["book_net"]))
    print("      net, months the rule decided " + describe(decided["book_net"]))
    print("      net, entry rule ON           " + describe(on["book_net"]))
    print("      net, entry rule OFF          " + describe(decided[decided["signal"] == False]["book_net"]))  # noqa: E712
    print("      gross, entry rule ON         " + describe(on["book_gross"]))
    print(f"      mean costs, both legs        {book['costs'].mean():+.1f} bps")

    print("\n  D3  by era, entry rule ON")
    for era, g in on.groupby(era_of(on.index)):
        print(f"      {era:<8} net   " + describe(g["book_net"]))
        print(f"      {'':<8} gross " + describe(g["book_gross"]))
    print("      by era, every month")
    for era, g in book.groupby(era_of(book.index)):
        print(f"      {era:<8} net   " + describe(g["book_net"]))

    print("\n  D4  legs, entry rule ON")
    print("      long ETH alone  net   " + describe(on["eth_long_net"]))
    print("      short BTC alone net   " + describe(on["btc_short_net"]))
    print("      book            net   " + describe(on["book_net"]))
    print("      legs, every month")
    print("      long ETH alone  net   " + describe(book["eth_long_net"]))
    print("      short BTC alone net   " + describe(book["btc_short_net"]))


def robustness(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tenor in sorted(cells["tenor"].unique()):
        for policy in sorted(cells["policy"].unique()):
            for slip in sorted(cells["slip_bps"].unique()):
                b = legs(cells, tenor, 0.0, policy, slip)
                if b.empty:
                    continue
                on = b[b["signal"] == True]["book_net"]  # noqa: E712
                rows.append({"tenor": tenor, "policy": policy, "slip": slip,
                             "all_mean": b["book_net"].mean(), "all_t": t_stat(b["book_net"]),
                             "on_mean": on.mean(), "on_t": t_stat(on), "n_on": len(on),
                             "gross": b["book_gross"].mean()})
    return pd.DataFrame(rows)


def main() -> int:
    cells = load_cells()
    book = legs(cells, **PRIMARY)
    if book.empty:
        raise SystemExit("no paired inceptions for the primary specification")
    try:
        book["corr"] = realized_corr(book.index)
    except Exception as exc:  # the index history is optional context
        print(f"  (realised correlation unavailable: {exc})")
    report(book)

    grid = robustness(cells)
    print("\nRobustness (not the test): every rule x slippage, ATM, mean net bps (t)")
    for tenor, g in grid.groupby("tenor"):
        print(f"\n  {tenor}d   {'every month':>18} {'rule ON':>18}  gross")
        for _, r in g.sort_values(["slip", "policy"]).iterrows():
            print(f"  {r.policy:<10} {r.slip:>4.0f}bps  {r.all_mean:+7.1f} ({r.all_t:+5.2f})"
                  f"  {r.on_mean:+7.1f} ({r.on_t:+5.2f}) n={r.n_on:<3d} {r.gross:+7.1f}")
        sign = (g["on_mean"] > 0).mean()
        print(f"  rule-ON net positive in {sign:.0%} of {len(g)} cells; "
              f"t > 2 in {(g['on_t'] > 2).mean():.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    book.assign(signal=book["signal"].astype(float)).to_parquet(OUT)
    grid.to_parquet(OUT.with_name("dispersion_grid.parquet"))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
