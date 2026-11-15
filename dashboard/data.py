"""
Everything the dashboard shows, computed without Streamlit.

`app.py` is layout only. Every number on screen comes from a function here,
so it can be tested, and so the dashboard cannot quietly compute something
the studies do not: the surface is the same `VolSurface` the strategy queries,
the density is `surface.density`, the metrics are `strategy.backtest.
compute_metrics`, and the backtest rows are the hedging sweep's own cells.

Data lives under `data/` (gitignored). Set `VOL_DATA_ROOT` to point at a
different `data/` directory -- a worktree, say, reading the main checkout's.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standardised moneyness grid, the same coordinate the figures and the
# hedging sweep use: z = k / (atm vol * sqrt(T)).
Z_GRID = np.linspace(-2.0, 2.0, 81)
ERAS = {"all": (2020, 2026), "2020-22": (2020, 2022), "2023-26": (2023, 2026)}
# Monthly inceptions: twelve a year. Used to annualise the Sharpe, which on
# overlapping monthly positions is descriptive, not something to size on.
INCEPTIONS_PER_YEAR = 12.0


def data_root() -> Path:
    return Path(os.environ.get("VOL_DATA_ROOT", ROOT / "data"))


def research(name: str) -> Path:
    return data_root() / "research" / name


# --------------------------------------------------------------------------
# Live chains and the surface
# --------------------------------------------------------------------------

def snapshot_files(asset: str) -> list[str]:
    """Track A chain files for an asset, oldest first."""
    currency = asset if asset in ("BTC", "ETH") else "USDC"
    return sorted(glob.glob(str(data_root() / "raw" / f"{currency.lower()}_chain_*.parquet")))


def stamp_of(path: str) -> pd.Timestamp:
    """The UTC time in a chain file's name, e.g. btc_chain_20261114T094725Z."""
    ts = pd.Timestamp(Path(path).stem.split("_")[-1])
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def load_snapshot(path: str, asset: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if "underlying_asset" in raw.columns:
        raw = raw[raw["underlying_asset"] == asset]
    if raw.empty:
        return raw
    t0 = raw["timestamp"].min()
    return raw[raw["timestamp"] == t0]


def build_surface(path: str, asset: str) -> dict | None:
    """
    Fit a chain and return what the surface tab needs.

    A plain dict of plain values, so Streamlit can cache it. The `VolSurface`
    itself is included; it pickles, being SVI parameters and forwards.
    """
    from surface.build import build_from_snapshot
    from surface.surface import from_fits

    raw = load_snapshot(path, asset)
    if raw.empty:
        return None
    fits, ranges, refused, quotes = build_from_snapshot(raw, asset)
    if not fits:
        return None
    surface = from_fits(fits, quoted_ranges=ranges)
    return {
        "surface": surface,
        "fits": fits,
        "ranges": ranges,
        "refused": refused,
        "quotes": quotes[["T", "expiry_ts", "strike", "option_type", "log_moneyness",
                          "iv", "mark_iv", "forward", "bid_usd", "ask_usd"]].copy(),
        "stamp": pd.Timestamp(quotes["timestamp"].iloc[0]),
        "spot": float(raw["index_price"].iloc[0]) if "index_price" in raw else np.nan,
    }


def surface_grid(surface, Ts, zs=Z_GRID) -> np.ndarray:
    """Implied vol (%) on a tenor x standardised-moneyness grid."""
    rows = []
    for T in Ts:
        F = surface.forward(T)
        atm = surface.iv(F, T)
        rows.append([surface.iv(F * np.exp(z * atm * np.sqrt(T)), T) * 100 for z in zs])
    return np.array(rows)


def smile(built: dict, T: float, pad: float = 0.35, points: int = 240) -> dict:
    """The fitted slice over its quoted range plus a margin, and the quotes it saw."""
    surface, ranges = built["surface"], built["ranges"]
    lo, hi = ranges[T]
    F = surface.forward(T)
    grid = np.linspace(lo - pad * (hi - lo), hi + pad * (hi - lo), points)
    fitted = np.array([surface.iv(F * np.exp(k), T) * 100 for k in grid])
    quotes = built["quotes"]
    obs = quotes[np.isclose(quotes["T"], T)]
    return {"k": grid, "iv": fitted, "lo": lo, "hi": hi, "forward": F, "quotes": obs}


def fit_table(built: dict) -> pd.DataFrame:
    """One row per expiry: fitted, or refused and why."""
    rows = []
    for T, f in sorted(built["fits"].items()):
        lo, hi = built["ranges"][T]
        rows.append({
            "expiry": pd.Timestamp(f["expiry"]).strftime("%Y-%m-%d"),
            "days": T * 365.25, "status": "fitted", "quotes": int(f["n"]),
            "forward": f["forward"], "atm_vol": f["atm_vol"],
            "rmse_vol": f["rmse_vol"], "max_err_vol": f["max_vol_err"],
            "rho": f["rho"], "k_lo": lo, "k_hi": hi,
        })
    for r in built["refused"]:
        rows.append({
            "expiry": pd.Timestamp(r["expiry"]).strftime("%Y-%m-%d"),
            "days": r["T"] * 365.25, "status": f"refused: {r.get('reason', '')}".strip(": "),
            "quotes": int(r.get("n", 0) or 0),
        })
    return pd.DataFrame(rows).sort_values("days").reset_index(drop=True)


# --------------------------------------------------------------------------
# Risk-neutral density
# --------------------------------------------------------------------------

def density_curve(surface, T: float, width: float = 4.0, points: int = 801) -> pd.DataFrame:
    """Density over price and its CDF, on the same grid `surface.density` integrates."""
    from surface.density import _grid

    k, p = _grid(surface, T, width, points)
    F = surface.forward(T)
    price = F * np.exp(k)
    pdf_price = p / price  # change of variable from k to price
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(k))])
    return pd.DataFrame({"k": k, "price": price, "pdf": pdf_price, "cdf": cdf})


from surface.density import levels_around, nice_step  # noqa: E402,F401  (shared with rnd_monitor)


def tail_table(surface, T: float) -> tuple[pd.DataFrame, dict]:
    from surface.density import tail_report

    rep = tail_report(surface, T, levels_around(surface.forward(T)))
    rows = []
    for label, item in rep["levels"].items():
        p = item["p_below"] if label.startswith("below") else item["p_above"]
        rows.append({"level": label, "probability": p,
                     "move": item["price"] / rep["forward"] - 1,
                     "extrapolated": item["extrapolated"]})
    return pd.DataFrame(rows), rep


# --------------------------------------------------------------------------
# Surface history (the PCA grid cache)
# --------------------------------------------------------------------------

def atm_history(asset: str) -> pd.DataFrame:
    """ATM vol per tenor through Track A, from `surface_pca.py`'s cached grid."""
    path = research("surface_grid.parquet")
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_parquet(path)
    g = g[(g["asset"] == asset) & (g["z"] == 0.0) & g["iv"].notna()]
    return g.pivot_table(index="stamp", columns="tenor", values="iv").sort_index()


def skew_history(asset: str, tenor: int = 30) -> pd.DataFrame:
    """Put wing minus call wing (z=-1 minus z=+1), vol points, at one tenor."""
    path = research("surface_grid.parquet")
    if not path.exists():
        return pd.DataFrame()
    g = pd.read_parquet(path)
    g = g[(g["asset"] == asset) & (g["tenor"] == tenor) & g["z"].isin([-1.0, 1.0])]
    w = g.pivot_table(index="stamp", columns="z", values="iv").sort_index().dropna()
    spot = g.groupby("stamp")["spot"].first()
    return pd.DataFrame({"skew": w[-1.0] - w[1.0], "spot": spot.reindex(w.index)})


# --------------------------------------------------------------------------
# Backtest: the hedging sweep's cells
# --------------------------------------------------------------------------

def load_cells() -> pd.DataFrame:
    """Usable cells of `hedge_frequency.py`'s sweep (one row per inception)."""
    files = sorted(research("hedge_cells").glob("cells_*.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    frame = frame[(frame["skipped"] == "") & frame["supported"].astype(bool)]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    return frame


def in_era(frame: pd.DataFrame, era: str) -> pd.DataFrame:
    lo, hi = ERAS[era]
    return frame[(frame["year"] >= lo) & (frame["year"] <= hi)]


def policy_order(labels) -> list:
    """Clocks by hours, then fixed bands, then Whalley-Wilmott, each by parameter."""
    kind = {"time": 0, "band": 1, "ww": 2}

    def key(label):
        name, _, param = label.partition(" ")
        return kind.get(name, 9), float(param.rstrip("h") or 0)

    return sorted(set(labels), key=key)


def t_stat(x) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def equity_curve(series: pd.Series) -> pd.DataFrame:
    """Cumulative P&L and drawdown from its running peak, in the series' units."""
    s = pd.Series(series).dropna().sort_index()
    equity = s.cumsum()
    return pd.DataFrame({"pnl": s, "equity": equity, "drawdown": equity - equity.cummax()})


def cell_metrics(series: pd.Series) -> dict:
    """`compute_metrics` on one inception per row, plus the t-stat it leaves out."""
    from strategy.backtest import compute_metrics

    hours = 365.25 * 24 / INCEPTIONS_PER_YEAR
    m = compute_metrics(pd.Series(series).dropna(), hours_per_step=hours)
    m["t"] = t_stat(series)
    return m


def policy_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Mean net bps and t per rule x slippage -- the shape of HANDOFF §12."""
    g = cells.groupby(["policy", "slip_bps"])["net_bps"]
    out = pd.DataFrame({"mean": g.mean(), "t": g.apply(t_stat), "n": g.size()}).reset_index()
    out["policy"] = pd.Categorical(out["policy"], policy_order(out["policy"]), ordered=True)
    return out.sort_values(["policy", "slip_bps"]).reset_index(drop=True)


def cost_breakdown(cells: pd.DataFrame) -> pd.Series:
    """Where the money goes for one cell: premium, hedging P&L, spread, slippage."""
    gross, net = cells["gross_bps"].mean(), cells["net_bps"].mean()
    premium = cells["premium_bps"].mean()
    spread, slip = cells["spread_bps"].mean(), cells["slippage_bps"].mean()
    return pd.Series({
        "premium sold": premium,
        "realised hedging P&L": gross - premium,
        "option spread": -spread,
        "hedge slippage": -slip,
        "net": net,
    })


# --------------------------------------------------------------------------
# Live book and the premium
# --------------------------------------------------------------------------

def load_live_book(rule: str = "whalley_wilmott") -> pd.DataFrame:
    """`hold_to_expiry.py` output: band-hedged by default, or the clock run."""
    name = "hold_to_expiry.parquet" if rule == "whalley_wilmott" else "hold_to_expiry_clock.parquet"
    path = research(name)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame[frame["skipped"] == ""].copy()
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    return frame


def load_premium() -> pd.DataFrame:
    """Daily 30-day DVOL against forward realised, `vrp_hypotheses.py`'s panel."""
    path = research("vrp_hypotheses_btc.parquet")
    if not path.exists():
        path = research("vrp_panel.parquet")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index)
    return frame[["dvol", "realized", "premium"]].dropna()
