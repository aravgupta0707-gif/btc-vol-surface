"""
Why is the premium going away? Tests of the hypotheses in HYPOTHESES.md.

    python scripts/vrp_hypotheses.py            run everything, offline
    python scripts/vrp_hypotheses.py --fetch    fetch ETH DVOL first if missing

The predictions were written to HYPOTHESES.md before this was run. Each
section below names the hypothesis it tests and prints the number that would
count for or against it.

Inference, because it is the easiest thing here to get wrong:

- Premium observations are 30-day windows sampled daily, so neighbours share
  29 of 30 days. Every t-statistic is Newey-West with a 30-day lag, and every
  p-value on a break or a mean comes from a circular block bootstrap, with the
  block length varied because the premium is persistent over months and a short
  block understates that.
- A break date estimated from the series cannot be tested with an ordinary
  t-test (the date was chosen to make the gap large). The p-value is for the
  supremum of the shift statistic over all candidate dates, computed on a
  bootstrap of the *demeaned* series, i.e. under "no break".
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "research"
HIST = ROOT / "data" / "historical"
sys.path.insert(0, str(ROOT))

HORIZON = 30
HOURS_Y = 365.0 * 24  # 24/7 market: no trading-day adjustment
LAGS = 30
ERA_SPLIT = pd.Timestamp("2023-01-01")  # the README's 2021-22 / 2023-26 cut
SEED = 20260930

# Dates fixed in advance from the hypotheses, not read off the data.
EVENTS = {
    "Terra collapse": "2022-05-09",
    "FTX bankruptcy": "2022-11-11",
    "US spot BTC ETFs trade": "2024-01-11",
    "IBIT options list": "2024-11-19",
}

# Costs from README / hedge_frequency.py: option spread plus hedge slippage.
SPREAD_BPS = 11.0
HEDGE_BPS = (20.0, 75.0)


# --------------------------------------------------------------------- data

def fetch_dvol(currency: str, fetch: bool) -> pd.Series:
    """Daily DVOL closes for `currency`, cached; BTC reuses vrp_study's file."""
    path = CACHE / f"dvol_{currency.lower()}_daily.parquet"
    if path.exists():
        series = pd.read_parquet(path)["dvol"]
    else:
        if not fetch:
            raise SystemExit(f"{path.name} missing; rerun with --fetch")
        import requests

        base = "https://www.deribit.com/api/v2/public"
        rows, cursor, today = [], dt.date(2021, 3, 1), dt.date.today()
        while cursor < today:
            end = min(cursor + dt.timedelta(days=700), today)
            ms = lambda d: int(dt.datetime(d.year, d.month, d.day,
                                           tzinfo=dt.timezone.utc).timestamp() * 1000)
            chunk = requests.get(
                f"{base}/get_volatility_index_data",
                params=dict(currency=currency, start_timestamp=ms(cursor),
                            end_timestamp=ms(end), resolution=86400),
                timeout=60,
            ).json().get("result", {}).get("data", [])
            rows.extend(chunk)
            if not chunk:
                break
            cursor = end
        frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
        frame["date"] = pd.to_datetime(frame["ts"], unit="ms", utc=True).dt.date
        series = frame.drop_duplicates("date").set_index("date")["close"].rename("dvol")
        series.to_frame().to_parquet(path)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def hourly_returns(asset: str, step: int = 1) -> pd.Series:
    """Log returns over `step` hours (every step-th hourly close, anchored 00:00)."""
    candles = pd.read_parquet(CACHE / f"{asset.lower()}_perp_hourly.parquet")
    close = candles.set_index(pd.to_datetime(candles["time"]))["close"]
    return np.log(close.iloc[::step]).diff().dropna()


def daily_variance(returns: pd.Series, step: int = 1) -> pd.DataFrame:
    """
    Per UTC day: realised variance, bipower variation, and hours observed.

    Bipower variation (pi/2) * sum |r_i||r_{i-1}| estimates the *continuous*
    part of variance and is robust to a single jump; RV - BV is the jump part.
    """
    bv_terms = (np.pi / 2) * returns.abs() * returns.abs().shift(1)
    day = returns.index.tz_convert("UTC").tz_localize(None).floor("D")
    frame = pd.DataFrame({"rv": returns**2, "bv": bv_terms, "n": float(step)})
    frame.index = day
    return frame.groupby(level=0).sum()


def build_panel(asset: str, dvol: pd.Series, step: int = 1) -> pd.DataFrame:
    """
    Implied and forward-realised quantities, one row per observation date.
    Forward windows cover the HORIZON days after the date, as in vrp_study.py.
    `step` is the sampling interval of the realised variance in hours.
    """
    daily = daily_variance(hourly_returns(asset, step), step)
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(days).fillna(0.0)

    def forward(col: str) -> pd.Series:
        return daily[col].rolling(HORIZON).sum().shift(-HORIZON)

    hours = forward("n")
    ok = hours >= 0.95 * HORIZON * 24
    scale = HOURS_Y / hours.where(ok)
    rv_var = forward("rv") * scale
    bv_var = forward("bv") * scale

    back_hours = daily["n"].rolling(HORIZON).sum()
    back_rv = daily["rv"].rolling(HORIZON).sum() * HOURS_Y / back_hours.where(
        back_hours >= 0.95 * HORIZON * 24)

    panel = pd.DataFrame({
        "dvol": dvol,
        "rv_var": rv_var, "bv_var": bv_var,
        "trailing_rv": np.sqrt(back_rv) * 100,
    }).dropna()
    panel["realized"] = np.sqrt(panel["rv_var"]) * 100
    panel["continuous"] = np.sqrt(panel["bv_var"]) * 100
    panel["premium"] = panel["dvol"] - panel["realized"]
    panel["jump_share"] = (1 - panel["bv_var"] / panel["rv_var"]).clip(lower=0)
    return panel


def trailing_funding(asset: str) -> pd.Series:
    """30-day mean perp funding, annualised %, using data up to each day's end."""
    funding = pd.read_parquet(CACHE / f"{asset.lower()}_funding_hourly.parquet")
    rate = funding.set_index(pd.to_datetime(funding["time"]))["interest_8h"]
    rate.index = rate.index.tz_convert("UTC").tz_localize(None)
    daily = rate.resample("D").mean() * 3 * 365 * 100
    return daily.rolling(HORIZON).mean().rename("funding30")


# ---------------------------------------------------------------- inference

def long_run_var(e: np.ndarray, lags: int = LAGS) -> float:
    n = len(e)
    total = e @ e / n
    for lag in range(1, lags + 1):
        total += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return total


def nw_mean(x, lags: int = LAGS) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    mean = x.mean()
    se = np.sqrt(long_run_var(x - mean, lags) / len(x))
    return mean, se, mean / se


def nw_ols_multi(y, X, lags: int = LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Slopes and Newey-West t of y on [1, X] (intercept dropped from output)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    X = X[:, None] if X.ndim == 1 else X
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.solve(design.T @ design, design.T @ y)
    resid = y - design @ beta
    score = design * resid[:, None]
    n = len(y)
    s = score.T @ score / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = score[lag:].T @ score[:-lag] / n
        s += w * (g + g.T)
    bread = np.linalg.inv(design.T @ design / n)
    cov = bread @ s @ bread / n
    return beta[1:], beta[1:] / np.sqrt(np.diag(cov)[1:])


def nw_ols(y, x, lags: int = LAGS) -> tuple[float, float]:
    """Slope and Newey-West t of y on [1, x]."""
    beta, t = nw_ols_multi(y, x, lags)
    return float(beta[0]), float(t[0])


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)).ravel()[:n]) % n


def boot_mean_ci(x, block: int, draws: int = 2000) -> tuple[float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    means = np.array([x[block_indices(len(x), block, rng)].mean() for _ in range(draws)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def shift_stat(p: np.ndarray, tau: int) -> float:
    a, b = p[:tau], p[tau:]
    resid = np.concatenate([a - a.mean(), b - b.mean()])
    var = long_run_var(resid) * (1 / len(a) + 1 / len(b))
    return (a.mean() - b.mean()) / np.sqrt(var)


def sup_shift(p: np.ndarray, step: int = 10, trim: float = 0.15) -> tuple[int, float]:
    n = len(p)
    grid = np.arange(int(n * trim), int(n * (1 - trim)), step)
    stats = np.array([shift_stat(p, t) for t in grid])
    best = int(np.argmax(np.abs(stats)))
    return int(grid[best]), float(stats[best])


def sup_shift_pvalue(p: np.ndarray, block: int, draws: int = 300) -> float:
    """P(sup |shift t| this large | constant mean), by block bootstrap."""
    _, observed = sup_shift(p)
    null = p - p.mean()
    rng = np.random.default_rng(SEED)
    sims = [abs(sup_shift(null[block_indices(len(p), block, rng)])[1])
            for _ in range(draws)]
    return float((np.array(sims) >= abs(observed)).mean())


# ----------------------------------------------------------------- reporting

def era(index: pd.DatetimeIndex) -> np.ndarray:
    return np.where(index < ERA_SPLIT, "2021-22", "2023-26")


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def baseline(panel: pd.DataFrame) -> None:
    header("BASELINE  BTC premium = DVOL - following 30d realised (hourly RV)")
    p = panel["premium"]
    print(f"  {len(p)} daily observations, {p.index[0]:%Y-%m-%d} to {p.index[-1]:%Y-%m-%d}")
    print(f"\n  {'':>10} {'n':>5} {'implied':>8} {'realised':>9} {'premium':>8} "
          f"{'NW t':>6} {'hit':>6}")
    groups = [("all", panel)] + [(str(y), g) for y, g in panel.groupby(panel.index.year)]
    groups += [(e, panel[era(panel.index) == e]) for e in ("2021-22", "2023-26")]
    for name, g in groups:
        m, _, t = nw_mean(g["premium"])
        print(f"  {name:>10} {len(g):5d} {g['dvol'].mean():8.1f} "
              f"{g['realized'].mean():9.1f} {m:+8.2f} {t:6.2f} "
              f"{(g['premium'] > 0).mean():6.0%}")


def test_h0_h1(panel: pd.DataFrame) -> None:
    header("H0 / H1  Is the premium stable, and did it break?")
    p = panel["premium"].to_numpy()
    print("\n  Mean premium, 95% block-bootstrap interval (vol points)")
    print(f"  {'':>10} " + "".join(f"{'block ' + str(b):>22}" for b in (30, 90)))
    for name, mask in [("all", np.ones(len(p), bool)),
                       ("2021-22", era(panel.index) == "2021-22"),
                       ("2023-26", era(panel.index) == "2023-26")]:
        cells = []
        for block in (30, 90):
            lo, hi = boot_mean_ci(p[mask], block)
            cells.append(f"{p[mask].mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}]")
        print(f"  {name:>10} " + "".join(f"{c:>22}" for c in cells))

    pre, post = p[era(panel.index) == "2021-22"], p[era(panel.index) == "2023-26"]
    diff = pre.mean() - post.mean()
    print(f"\n  2021-22 minus 2023-26: {diff:+.2f} vol points")

    print("\n  Estimated break (supremum over candidate dates, trimmed 15%)")
    tau, stat = sup_shift(p)
    print(f"    break at {panel.index[tau]:%Y-%m-%d}   mean before "
          f"{p[:tau].mean():+.2f}  after {p[tau:].mean():+.2f}   sup|t| {abs(stat):.2f}")
    for block in (30, 90, 180):
        print(f"    bootstrap p-value, block {block:>3}d: "
              f"{sup_shift_pvalue(p, block):.3f}")

    print("\n  Pre-specified dates (ordinary NW test of the difference)")
    print(f"    {'event':<26} {'date':>11} {'before':>8} {'after':>8} {'diff':>8} {'t':>6}")
    for name, date in EVENTS.items():
        k = int(np.searchsorted(panel.index, pd.Timestamp(date)))
        if k < 60 or k > len(p) - 60:
            continue
        print(f"    {name:<26} {date:>11} {p[:k].mean():+8.2f} {p[k:].mean():+8.2f} "
              f"{p[:k].mean() - p[k:].mean():+8.2f} {shift_stat(p, k):6.2f}")

    print("\n  Rolling 180-day mean (vol points), every 90 days")
    roll = panel["premium"].rolling(180).mean().iloc[179::90]
    for date, value in roll.items():
        print(f"    {date:%Y-%m-%d}  premium {value:+6.2f}   "
              f"implied {panel['dvol'].loc[date - pd.Timedelta(days=179):date].mean():5.1f}  "
              f"realised {panel['realized'].loc[date - pd.Timedelta(days=179):date].mean():5.1f}")

    print("\n  Which side moved? Implied and realised, by era (vol points)")
    for e in ("2021-22", "2023-26"):
        g = panel[era(panel.index) == e]
        print(f"    {e}  implied {g['dvol'].mean():5.1f}   realised {g['realized'].mean():5.1f}")


def test_h2(panel: pd.DataFrame, funding: pd.Series) -> None:
    header("H2  Does leverage demand (perp funding) predict the premium?")
    frame = panel.join(funding, how="inner").dropna()
    frame["richness"] = frame["dvol"] - frame["trailing_rv"]
    year = frame.index.year

    def within(s: pd.Series) -> pd.Series:
        return s - s.groupby(year).transform("mean")

    print(f"  {len(frame)} observations. Predictors are known at the observation date.")
    print(f"  Regressing premium (vol pts) on each; slope per unit of the predictor.\n")
    print(f"  {'predictor':<26} {'slope':>8} {'NW t':>7}   {'within-year slope':>18} {'NW t':>7}")
    for name, label in [("funding30", "30d funding (ann. %)"),
                        ("dvol", "DVOL level"),
                        ("trailing_rv", "trailing 30d realised"),
                        ("richness", "DVOL - trailing realised")]:
        s, t = nw_ols(frame["premium"], frame[name])
        ws, wt = nw_ols(within(frame["premium"]), within(frame[name]))
        print(f"  {label:<26} {s:+8.3f} {t:7.2f}   {ws:+18.3f} {wt:7.2f}")

    sparse = frame.iloc[::HORIZON]
    s, t = nw_ols(sparse["premium"], sparse["funding30"], lags=1)
    print(f"\n  funding30, non-overlapping sample (n={len(sparse)}): slope {s:+.3f}, t {t:.2f}")
    beta, ts = nw_ols_multi(frame["premium"], frame[["funding30", "dvol"]])
    print(f"  joint [funding30, DVOL level]: funding {beta[0]:+.3f} (t {ts[0]:.2f}), "
          f"DVOL {beta[1]:+.3f} (t {ts[1]:.2f})   <- does funding survive the level control?")
    print("\n  Mean funding by year (ann. %) -- is it an era proxy?")
    for y, g in frame.groupby(frame.index.year):
        print(f"    {y}  funding {g['funding30'].mean():+7.2f}   premium {g['premium'].mean():+6.2f}")


def test_h3(panel: pd.DataFrame) -> None:
    header("H3  Is the premium jump compensation?")
    print("  Variances in annualised vol^2 points; 'vol' columns are sqrt of the mean.\n")
    print(f"  {'':>10} {'implied':>8} {'total RV':>9} {'continuous':>11} "
          f"{'jump var':>9} {'jump %':>7} | {'imp-RV':>7} {'imp-cont':>9}")
    groups = [("all", panel)] + [(e, panel[era(panel.index) == e])
                                 for e in ("2021-22", "2023-26")]
    for name, g in groups:
        imp, rv, bv = g["dvol"] ** 2, g["rv_var"] * 1e4, g["bv_var"] * 1e4
        print(f"  {name:>10} {np.sqrt(imp.mean()):8.1f} {np.sqrt(rv.mean()):9.1f} "
              f"{np.sqrt(bv.mean()):11.1f} {(rv - bv).mean():9.0f} "
              f"{(rv.sum() - bv.sum()) / rv.sum():7.1%} | "
              f"{np.sqrt(imp.mean()) - np.sqrt(rv.mean()):+7.2f} "
              f"{np.sqrt(imp.mean()) - np.sqrt(bv.mean()):+9.2f}")

    print("\n  Premium by the window's own jump share (terciles, ex post)")
    print(f"    {'tercile':<10} {'jump share':>11} {'premium':>9} {'hit':>6}")
    terciles = pd.qcut(panel["jump_share"], 3, labels=["low", "mid", "high"])
    for label, g in panel.groupby(terciles, observed=True):
        print(f"    {label:<10} {g['jump_share'].mean():11.1%} "
              f"{g['premium'].mean():+9.2f} {(g['premium'] > 0).mean():6.0%}")

    beta, ts = nw_ols_multi(panel["premium"], panel[["jump_share", "dvol"]])
    print(f"\n  joint [jump share, DVOL level]: jump share {beta[0]:+.1f} (t {ts[0]:.2f}), "
          f"DVOL {beta[1]:+.3f} (t {ts[1]:.2f})")
    print("    (jump share and vol level move together; this asks whether the tercile")
    print("     pattern above is just the vol level showing through)")

    print("\n  Does implied forecast realised? realised = a + b * DVOL (NW t for b=1)")
    slope, t0 = nw_ols(panel["realized"], panel["dvol"])
    se = abs(slope / t0)
    print(f"    slope b = {slope:.2f}  (t vs 1: {(slope - 1) / se:+.2f})")
    both = panel.dropna()
    slope2, t2 = nw_ols(both["realized"] - both["trailing_rv"],
                        both["dvol"] - both["trailing_rv"])
    print(f"    realised - trailing on DVOL - trailing: slope {slope2:.2f} "
          f"(1 = implied is the whole forecast, 0 = trailing realised is)")


def verdict_vs_cost(bps: float, low: float, high: float) -> str:
    if bps < low:
        return "below the cheapest cost case"
    if bps <= high:
        return "inside the cost range"
    return "above the dearest cost case"


def test_h4(panel: pd.DataFrame, dvol: pd.Series) -> None:
    header("H4  Is the premium about the size of the cost of harvesting it?")
    per_pt = 0.8 * np.sqrt(HORIZON / 365.0) * 1e4 * 0.01
    low = SPREAD_BPS + HEDGE_BPS[0]
    high = SPREAD_BPS + HEDGE_BPS[1]
    print(f"  Cost range: {SPREAD_BPS:.0f} bps option spread + {HEDGE_BPS[0]:.0f}-{HEDGE_BPS[1]:.0f} "
          f"bps hedge slippage = {low:.0f}-{high:.0f} bps of spot.")

    print(f"\n  (a) DVOL premium converted to bps. A 30d ATM straddle is worth")
    print(f"  ~0.8*sigma*sqrt(T) of spot, so one vol point is ~{per_pt:.0f} bps (first order).")
    print("  UPPER BOUND: DVOL is a variance-swap-style index that prices the whole")
    print("  smile, so it includes convexity an ATM straddle cannot collect.\n")
    print(f"  {'':>10} {'premium':>9} {'= bps':>7}   {'vs cost range':>28}")
    groups = [("all", panel)] + [(e, panel[era(panel.index) == e])
                                 for e in ("2021-22", "2023-26")]
    for name, g in groups:
        m = g["premium"].mean()
        print(f"  {name:>10} {m:+9.2f} {m * per_pt:+7.0f}   "
              f"{verdict_vs_cost(m * per_pt, low, high):>28}")

    surface = pd.read_parquet(CACHE / "vrp_surface.parquet")
    surface["date"] = pd.to_datetime(surface["date"])
    atm = surface[(surface["skipped"] == "") & surface["supported"].astype(bool)
                  & (surface["z"] == 0) & (surface["asset"] == "BTC")].dropna(subset=["pnl_bps"])

    print("\n  (b) What the ATM straddle itself earned: delta-hedged every 4h, sold at")
    print("  mid, settled at expiry (vrp_surface.py; monthly inceptions, close to")
    print("  independent). Before any spread or slippage, so compare with the range.\n")
    print(f"  {'tenor':>6} {'era':>9} {'n':>4} {'gross bps':>10} {'t':>6}   {'vs cost range':>28}")
    for tenor, g in atm.groupby("tenor"):
        for name, h in [("all", g)] + [(e, g[era(g["date"]) == e])
                                       for e in ("2021-22", "2023-26")]:
            m, se = h["pnl_bps"].mean(), h["pnl_bps"].std() / np.sqrt(len(h))
            print(f"  {tenor:5.0f}d {name:>9} {len(h):4d} {m:+10.1f} {m / se:6.2f}   "
                  f"{verdict_vs_cost(m, low, high):>28}")
    print("  (the cost range was measured on 30-day holds; for 7 days it is only indicative)")

    a30 = atm[atm["tenor"] == 30].set_index("date")
    same = a30[["atm_iv"]].join(dvol, how="inner").join(
        a30["realized"].rename("real"), how="inner")
    wedge = same["dvol"] - same["atm_iv"] * 100
    print(f"\n  (c) DVOL minus 30d ATM implied on the same {len(same)} days: "
          f"{wedge.mean():+.2f} vol points")
    print(f"      premium against that realised: DVOL {(same['dvol'] - same['real'] * 100).mean():+.2f}"
          f"   ATM IV {(same['atm_iv'] * 100 - same['real'] * 100).mean():+.2f}")


def test_posthoc(panel: pd.DataFrame, dvol: pd.Series) -> None:
    header("POST-HOC  Added after the first run, in response to what it showed")
    print("  These were not in HYPOTHESES.md's predictions. They are here because the")
    print("  first results (premium falling steadily, tracking vol level, differing")
    print("  from the earlier close-to-close study) raised these questions. Weigh them")
    print("  as exploratory.\n")

    print("  (1) Does the premium depend on how realised variance is sampled?")
    print("      The hedge samples price moves at its own rebalancing interval.\n")
    print(f"  {'sampling':>9} " + "".join(f"{y:>8}" for y in range(2021, 2027))
          + f" {'2021-22':>8} {'2023-26':>8}")
    for step in (1, 4, 24):
        p = build_panel("BTC", dvol, step)["premium"]
        cells = "".join(f"{p[p.index.year == y].mean():+8.2f}" for y in range(2021, 2027))
        print(f"  {str(step) + 'h':>9} {cells} "
              f"{p[p.index < ERA_SPLIT].mean():+8.2f} {p[p.index >= ERA_SPLIT].mean():+8.2f}")

    print("\n  (2) Is the decline a trend, and is it just the vol level falling?")
    years = ((panel.index - panel.index[0]).days / 365.25).to_numpy()
    s, t = nw_ols(panel["premium"], years)
    print(f"      premium on time:               {s:+.2f} vol pts per year  (NW t {t:.2f})")
    beta, ts = nw_ols_multi(panel["premium"], np.column_stack([years, panel["dvol"]]))
    print(f"      premium on time + DVOL level:  time {beta[0]:+.2f}/yr (t {ts[0]:.2f}), "
          f"DVOL {beta[1]:+.3f} (t {ts[1]:.2f})")
    ratio = np.log(panel["dvol"] / panel["realized"])
    s2, t2 = nw_ols(ratio, years)
    print(f"      log(implied/realised) on time: {s2:+.3f} per year  (NW t {t2:.2f})")
    print(f"      {'':>4} " + "".join(f"{y:>8}" for y in range(2021, 2027)))
    print(f"      mean log(imp/real) " + "".join(
        f"{ratio[ratio.index.year == y].mean():+8.3f}" for y in range(2021, 2027)))
    print(f"      premium as % of implied " + "".join(
        f"{(panel['premium'] / panel['dvol'])[panel.index.year == y].mean():8.1%}"
        for y in range(2021, 2027)))


def test_h5(btc: pd.DataFrame, eth: pd.DataFrame) -> None:
    header("H5  Is the premium larger in BTC than ETH, and why?")
    both = btc[["dvol", "realized", "premium"]].join(
        eth[["dvol", "realized", "premium"]], lsuffix="_b", rsuffix="_e", how="inner")
    both["gap"] = both["premium_b"] - both["premium_e"]
    both["ratio_gap"] = np.log(both["dvol_b"] / both["realized_b"]) \
        - np.log(both["dvol_e"] / both["realized_e"])
    print(f"  {len(both)} common days. gap = BTC premium - ETH premium (vol points);")
    print("  positive gap is the long-ETH / short-BTC trade earning.\n")
    print(f"  {'':>8} {'BTC prem':>9} {'ETH prem':>9} {'gap':>7} {'NW t':>6} "
          f"{'ratio gap':>10} {'NW t':>6}")
    groups = [("all", both)] + [(str(y), g) for y, g in both.groupby(both.index.year)]
    groups += [(e, both[era(both.index) == e]) for e in ("2021-22", "2023-26")]
    for name, g in groups:
        m, _, t = nw_mean(g["gap"])
        rm, _, rt = nw_mean(g["ratio_gap"])
        print(f"  {name:>8} {g['premium_b'].mean():+9.2f} {g['premium_e'].mean():+9.2f} "
              f"{m:+7.2f} {t:6.2f} {rm:+10.3f} {rt:6.2f}")
    print("  (ratio gap: log(implied/realised) for BTC minus ETH -- the level-free version)")

    spread = both["dvol_e"] - both["dvol_b"]
    median = spread.expanding(min_periods=90).median()
    narrow = (spread < median) & median.notna()
    wide = (spread >= median) & median.notna()
    print("\n  Conditioning on the ETH-BTC implied spread vs its expanding median")
    print("  (uses only days already seen, so no lookahead)")
    for name, mask in [("narrow", narrow), ("wide", wide)]:
        g = both.loc[mask, "gap"]
        m, _, t = nw_mean(g)
        print(f"    {name:<7} n={len(g):4d}  gap {m:+6.2f}  NW t {t:5.2f}")


def test_h5_openinterest(btc: pd.DataFrame, eth: pd.DataFrame) -> None:
    header("H5 mechanism  Does the gap track BTC's share of open interest?")
    rows = []
    for path in sorted(HIST.glob("btc_chain_*.parquet")):
        stamp = path.stem.split("_")[-1]
        eth_path = HIST / f"eth_chain_{stamp}.parquet"
        if not eth_path.exists():
            continue
        sizes = []
        for p in (path, eth_path):
            c = pd.read_parquet(p, columns=["timestamp", "open_interest", "underlying_price"])
            c["usd"] = c["open_interest"] * c["underlying_price"]
            sizes.append(c.groupby("timestamp")["usd"].sum().median())
        if sizes[0] + sizes[1] > 0:
            rows.append((pd.Timestamp(stamp), sizes[0] / (sizes[0] + sizes[1])))
    oi = pd.Series(dict(rows), name="btc_share")

    gap = (btc["premium"] - eth["premium"]).dropna()
    frame = pd.concat([oi, gap.rename("gap")], axis=1, join="inner").dropna()
    print(f"  {len(frame)} archive days with open interest and a paired premium "
          f"({frame.index[0]:%Y-%m} to {frame.index[-1]:%Y-%m})")
    if len(frame) < 20:
        print("  too few points to say anything.")
        return
    print(f"  BTC share of combined option open interest: mean {frame['btc_share'].mean():.0%}, "
          f"range {frame['btc_share'].min():.0%}-{frame['btc_share'].max():.0%}")
    s, t = nw_ols(frame["gap"], frame["btc_share"], lags=1)
    rho = frame["gap"].corr(frame["btc_share"], method="spearman")
    print(f"  gap on BTC share: slope {s:+.2f} per unit share, t {t:.2f}, Spearman {rho:+.2f}")
    year = frame.index.year
    wy = frame["gap"] - frame["gap"].groupby(year).transform("mean")
    wx = frame["btc_share"] - frame["btc_share"].groupby(year).transform("mean")
    ws, wt = nw_ols(wy, wx, lags=1)
    print(f"  within-year (removes the trend): slope {ws:+.2f}, t {wt:.2f}")
    trend = frame["btc_share"].corr(pd.Series(np.arange(len(frame)), index=frame.index))
    print(f"  share vs time correlation {trend:+.2f}  (a trend here confounds the first line)")

    print("\n  Robustness of the within-year relation")
    print(f"    {'year':>6} {'n':>3} {'share':>6} {'gap':>7} {'within-yr Spearman':>19}")
    for y, g in frame.groupby(year):
        rho_y = g["gap"].corr(g["btc_share"], method="spearman") if len(g) >= 8 else np.nan
        print(f"    {y:>6} {len(g):3d} {g['btc_share'].mean():6.0%} {g['gap'].mean():+7.2f} "
              f"{rho_y:19.2f}")
    print(f"    pooled within-year Spearman {wy.corr(wx, method='spearman'):+.2f}")
    drops = []
    for y in sorted(set(year)):
        keep = year != y
        drops.append(np.polyfit(wx[keep], wy[keep], 1)[0])
    print(f"    slope dropping each year in turn: {min(drops):+.0f} to {max(drops):+.0f}")


def main() -> int:
    fetch = "--fetch" in sys.argv
    btc_dvol = fetch_dvol("BTC", fetch)
    btc = build_panel("BTC", btc_dvol)
    eth = build_panel("ETH", fetch_dvol("ETH", fetch))

    baseline(btc)
    test_h0_h1(btc)
    test_h2(btc, trailing_funding("BTC"))
    test_h3(btc)
    test_h4(btc, btc_dvol)
    test_h5(btc, eth)
    test_h5_openinterest(btc, eth)
    test_posthoc(btc, btc_dvol)

    btc.to_parquet(CACHE / "vrp_hypotheses_btc.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
