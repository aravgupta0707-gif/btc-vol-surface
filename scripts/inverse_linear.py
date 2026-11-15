"""
Inverse vs linear: do Deribit's two BTC option books agree under stress?

    python scripts/inverse_linear.py            build (cached) and report
    python scripts/inverse_linear.py --rebuild  re-invert every snapshot pair

Pre-registered in INVERSE_LINEAR.md, whose method section this follows. In
short: pair each inverse (`BTC-`) snapshot with the linear (`BTC_USDC-`)
snapshot taken seconds later, match quotes on (expiry, strike, type), invert
each book with its own forward, and ask how far apart the vols are, in which
direction, and whether that changes when the market is moving.

Outputs to data/research/:
    inverse_linear_quotes.parquet   one row per matched quote per hour
    inverse_linear_hours.parquet    one row per hour, with the stress measures
"""

from __future__ import annotations

import glob
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.clean import add_forward_and_expiry, compute_mid, select_otm, to_usd  # noqa: E402
from pricing.black76 import implied_vol_many  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "research"

MAX_PAIR_SECONDS = 120
MIN_T_YEARS = 2 / 365.25
FLOOR_BTC = 0.0005          # the cleaner's MIN_MID_BTC, applied in dollars to both books
CLEAN_WIDTH = 0.03          # vol-space bid-ask width allowed in *both* books
RV_WINDOW_HOURS = 24
RV_MIN_OBS = 12
LAGS = 24
STRESS_Q = 0.90
CALM_Q = 0.50


def _nw():
    spec = importlib.util.spec_from_file_location("vrp_hypotheses", ROOT / "scripts" / "vrp_hypotheses.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nw_ols_multi


# --------------------------------------------------------------------------
# Pairing and per-book preparation

def stamp(path: str) -> pd.Timestamp:
    return pd.Timestamp(re.search(r"(\d{8}T\d{6})Z", path).group(1), tz="UTC")


def pair_snapshots(inverse: list, linear: list, max_seconds: float = MAX_PAIR_SECONDS) -> list:
    """(inverse_path, linear_path) for each inverse snapshot with a linear one close enough."""
    if not inverse or not linear:
        return []
    lin_times = pd.DatetimeIndex([stamp(p) for p in linear])
    pairs = []
    for path in inverse:
        t = stamp(path)
        i = int(np.searchsorted(lin_times.asi8, t.value))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(linear):
                gap = abs((lin_times[j] - t).total_seconds())
                if gap <= max_seconds and (best is None or gap < best[0]):
                    best = (gap, linear[j])
        if best is not None:
            pairs.append((path, best[1]))
    return pairs


def prepare(raw: pd.DataFrame, as_of) -> pd.DataFrame:
    """
    One book's BTC quotes, OTM, two-sided, in dollars, T >= 2 days, with the
    same dollar floor applied to both books.

    The width rule is switched off: the vol-space width below replaces it.
    """
    if "underlying_asset" in raw.columns:
        raw = raw[raw["underlying_asset"] == "BTC"]
    out = compute_mid(raw, drop_wide=False)
    if out.empty:
        return out
    out = add_forward_and_expiry(out, as_of)
    out = select_otm(to_usd(out))
    out = out[(out["T"] >= MIN_T_YEARS) & (out["mid_usd"] >= FLOOR_BTC * out["forward"])]
    return out[["expiry_ts", "strike", "option_type", "T", "forward", "log_moneyness",
                "bid_usd", "ask_usd", "mid_usd"]].reset_index(drop=True)


def match_books(inv: pd.DataFrame, lin: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (expiry, strike, type); columns suffixed _inv / _lin."""
    keys = ["expiry_ts", "strike", "option_type"]
    m = inv.merge(lin, on=keys, suffixes=("_inv", "_lin"))
    return m.reset_index(drop=True)


def add_vols(m: pd.DataFrame) -> pd.DataFrame:
    """Mid, bid and ask implied vol in each book, each from its own forward."""
    out = m.copy()
    types = out["option_type"].tolist()
    for book in ("inv", "lin"):
        F, K, T = out[f"forward_{book}"], out["strike"], out[f"T_{book}"]
        for side in ("mid", "bid", "ask"):
            out[f"iv_{side}_{book}"] = implied_vol_many(F, K, T, out[f"{side}_usd_{book}"], types)
        out[f"width_{book}"] = out[f"iv_ask_{book}"] - out[f"iv_bid_{book}"]
    out = out.dropna(subset=["iv_mid_inv", "iv_mid_lin"])
    out["diff"] = out["iv_mid_lin"] - out["iv_mid_inv"]
    out["clean"] = (out["width_inv"] <= CLEAN_WIDTH) & (out["width_lin"] <= CLEAN_WIDTH)
    # A crossing: one book's bid above the other's ask, both in dollars.
    out["crossed"] = (out["bid_usd_lin"] > out["ask_usd_inv"]) | (out["bid_usd_inv"] > out["ask_usd_lin"])
    return out.reset_index(drop=True)


def atm_iv_30d(m: pd.DataFrame) -> float:
    """Inverse-book mid iv at the expiry nearest 30 days, strike nearest the forward."""
    if m.empty:
        return np.nan
    days = m["T_inv"] * 365.25
    expiry = m.loc[(days - 30).abs().idxmin(), "expiry_ts"]
    s = m[m["expiry_ts"] == expiry]
    return float(s.loc[s["log_moneyness_inv"].abs().idxmin(), "iv_mid_inv"])


def build(pairs: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    quotes, hours = [], []
    for n, (inv_path, lin_path) in enumerate(pairs):
        t = stamp(inv_path)
        inv_raw, lin_raw = pd.read_parquet(inv_path), pd.read_parquet(lin_path)
        index_price = float(inv_raw["index_price"].median()) if "index_price" in inv_raw else np.nan
        inv, lin = prepare(inv_raw, t), prepare(lin_raw, t)
        hour = {"timestamp": t, "index_price": index_price, "n_matched": 0}
        if not inv.empty and not lin.empty:
            m = add_vols(match_books(inv, lin))
            if not m.empty:
                m.insert(0, "timestamp", t)
                quotes.append(m)
                hour["n_matched"] = len(m)
                hour["atm_iv_30d"] = atm_iv_30d(m)
                # Forward gap per shared expiry, then the hour's median.
                f = m.groupby("expiry_ts")[["forward_inv", "forward_lin"]].first()
                hour["fwd_gap_bps"] = float(((f["forward_lin"] / f["forward_inv"] - 1) * 1e4).median())
        hours.append(hour)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(pairs)} pairs", flush=True)
    q = pd.concat(quotes, ignore_index=True) if quotes else pd.DataFrame()
    return q, pd.DataFrame(hours)


# --------------------------------------------------------------------------
# Hourly summaries and the stress measure

def trailing_rv(times: pd.Series, price: pd.Series, window_hours: float = RV_WINDOW_HOURS,
                min_obs: int = RV_MIN_OBS) -> pd.Series:
    """
    Annualised realised vol over the trailing window, from whatever snapshots
    fall in it. Track A has sleep gaps, so returns are not all one hour apart:
    the variance is summed squared log returns over the elapsed time, and a
    return that spans a gap longer than the window is not used at all.
    """
    t = pd.to_datetime(times).reset_index(drop=True)
    p = pd.Series(np.asarray(price, float))
    r = np.log(p).diff()
    dt_h = t.diff().dt.total_seconds() / 3600
    r[dt_h > window_hours] = np.nan
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        inside = (t > t[i] - pd.Timedelta(hours=window_hours)) & (t <= t[i]) & r.notna()
        if inside.sum() >= min_obs:
            hours = dt_h[inside].sum()
            out[i] = np.sqrt((r[inside] ** 2).sum() / hours * 24 * 365.25)
    return pd.Series(out, index=times.index)


def hourly(quotes: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    h = hours.sort_values("timestamp").reset_index(drop=True)
    h["rv24"] = trailing_rv(h["timestamp"], h["index_price"]).to_numpy()
    q = quotes
    clean = q[q["clean"]]
    g_all = q.groupby("timestamp")
    g = clean.groupby("timestamp")
    stats = pd.DataFrame({
        "n_clean": g.size(),
        "abs_diff": g["diff"].apply(lambda d: d.abs().median()),
        "signed_diff": g["diff"].median(),
        "put_diff": clean[clean["option_type"] == "P"].groupby("timestamp")["diff"].median(),
        "call_diff": clean[clean["option_type"] == "C"].groupby("timestamp")["diff"].median(),
        "width_lin": g["width_lin"].median(),
        "width_inv": g["width_inv"].median(),
        "crossed_rate": g["crossed"].mean(),
        "abs_diff_all": g_all["diff"].apply(lambda d: d.abs().median()),
        "signed_diff_all": g_all["diff"].median(),
    })
    h = h.merge(stats, left_on="timestamp", right_index=True, how="left")
    lo, hi = h["rv24"].quantile(CALM_Q), h["rv24"].quantile(STRESS_Q)
    h["regime"] = np.where(h["rv24"] >= hi, "stress", np.where(h["rv24"] <= lo, "calm", "mid"))
    h.loc[h["rv24"].isna(), "regime"] = "unknown"
    return h


# --------------------------------------------------------------------------
# Report

def report(quotes: pd.DataFrame, h: pd.DataFrame) -> None:
    nw = _nw()
    vp = 100  # vol points
    print(f"\npairs {len(h)}, hours with matched quotes {int((h['n_matched'] > 0).sum())}, "
          f"matched quotes {len(quotes):,} ({quotes['clean'].mean():.0%} clean)")
    print(f"window {h['timestamp'].min():%Y-%m-%d} to {h['timestamp'].max():%Y-%m-%d}; "
          f"rv24 median {h['rv24'].median():.1%}, p90 {h['rv24'].quantile(.9):.1%}, "
          f"max {h['rv24'].max():.1%}")
    print(f"median clean matches per hour {h['n_clean'].median():.0f}")

    by = h[h["regime"].isin(["calm", "stress"])].groupby("regime")
    cols = ["abs_diff", "signed_diff", "put_diff", "call_diff", "width_lin", "width_inv",
            "abs_diff_all", "crossed_rate", "fwd_gap_bps"]
    table = by[cols].median()
    table.loc[:, [c for c in cols if c not in ("crossed_rate", "fwd_gap_bps")]] *= vp
    table["hours"] = by.size()
    print("\nmedians by regime (vol points except crossed rate and forward gap in bps)")
    print(table.T.round(3).to_string())

    d = h.dropna(subset=["rv24", "abs_diff", "width_lin", "put_diff", "call_diff"])
    print(f"\nregressions on {len(d)} hours, NW {LAGS} lags (rv24 in vol points)")
    rv = d["rv24"] * vp
    b, t = nw(d["abs_diff"] * vp, rv, LAGS)
    print(f"  P2 abs diff on rv24:              slope {b[0]:+.4f}  t {t[0]:+.2f}")
    b2, t2 = nw(d["abs_diff"] * vp, np.column_stack([rv, d["width_lin"] * vp]), LAGS)
    print(f"  P3 ... adding linear width:       rv slope {b2[0]:+.4f} t {t2[0]:+.2f}; "
          f"width slope {b2[1]:+.4f} t {t2[1]:+.2f}; rv slope kept {b2[0] / b[0]:.0%}")
    bw, tw = nw(d["width_lin"] * vp, rv, LAGS)
    print(f"      linear width on rv24:         slope {bw[0]:+.4f}  t {tw[0]:+.2f}")
    for name in ("put_diff", "call_diff"):
        b, t = nw(d[name] * vp, rv, LAGS)
        print(f"  P4 {name:9} on rv24:           slope {b[0]:+.4f}  t {t[0]:+.2f}")

    clean = quotes[quotes["clean"]]
    print(f"\nP5 crossings: {clean['crossed'].mean():.3%} of clean matched quotes, "
          f"{quotes['crossed'].mean():.3%} of all")
    reg = clean.merge(h[["timestamp", "regime"]], on="timestamp")
    print(reg.groupby("regime")["crossed"].agg(["mean", "sum", "size"]).to_string())
    print(f"\nP6 forward gap bps: median {h['fwd_gap_bps'].median():+.2f}, "
          f"max |gap| {h['fwd_gap_bps'].abs().max():.2f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qpath, hpath = OUT / "inverse_linear_quotes.parquet", OUT / "inverse_linear_hours.parquet"
    if "--rebuild" in sys.argv or not qpath.exists():
        pairs = pair_snapshots(sorted(glob.glob(str(RAW / "btc_chain_*.parquet"))),
                               sorted(glob.glob(str(RAW / "usdc_chain_*.parquet"))))
        print(f"{len(pairs)} snapshot pairs")
        quotes, hours = build(pairs)
        quotes.to_parquet(qpath)
        hours.to_parquet(hpath)
    quotes, hours = pd.read_parquet(qpath), pd.read_parquet(hpath)
    h = hourly(quotes, hours)
    h.to_parquet(OUT / "inverse_linear_hourly.parquet")
    report(quotes, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
