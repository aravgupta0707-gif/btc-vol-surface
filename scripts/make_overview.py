"""
Overview figures: what the project built and what it found, in eight charts.

    python scripts/make_overview.py        writes figures/overview/*.png

Every number is read from the studies' saved outputs in data/research/ (or a
live fit of the latest snapshot), except the evidence scoreboard. That chart
collects t-statistics quoted in the write-ups, each with its source file,
because several come from regressions that are expensive to rerun here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "overview"
RES = ROOT / "data" / "research"

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 36, "legend.frameon": False, "lines.linewidth": 2,
})


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.012, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom", wrap=True)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return path


def latest_surface():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rnd_monitor", ROOT / "scripts" / "rnd_monitor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.surface_for(m.snapshots("BTC")[-1], "BTC")


# --------------------------------------------------------------------------

def fig_smiles(surface, stamp):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for days, color in ((7, ORANGE), (30, BLUE), (90, AQUA)):
        T = days / 365.25
        F = surface.forward(T)
        sd = surface.iv(F, T) * np.sqrt(T)
        k = np.linspace(-2.2 * sd, 2.2 * sd, 200)
        iv = [surface.iv(F * np.exp(x), T) * 100 for x in k]
        ax.plot(np.exp(k) * 100, iv, color=color)
        ax.text(np.exp(k[-1]) * 100 + 1, iv[-1], f"{days}d", color=INK2, va="center", fontsize=9.5)
    ax.axvline(100, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlabel("strike, % of forward")
    ax.set_ylabel("implied vol, %")
    ax.set_title("1 · The fitted BTC vol surface: three slices of it")
    subtitle(ax, f"SVI fit of the live Deribit chain, {stamp:%Y-%m-%d %H:%M} UTC. "
                 "Puts (left) dearer than calls; short tenors more curved.")
    return save(fig, "1_surface_smiles")


def fig_density(surface, stamp):
    from surface.density import density_price, smile_quantile
    T = 90 / 365.25
    F = surface.forward(T)
    prices = np.linspace(F * 0.35, F * 2.0, 900)
    q = density_price(surface, prices, T) * 1000
    # Finite differences across the kink where a slice's fit meets its clamped
    # wing give one-point spikes (HANDOFF §5). Drop points far off their
    # neighbours' median rather than draw them.
    med = pd.Series(q).rolling(15, center=True, min_periods=1).median().to_numpy()
    q = np.where(np.abs(q - med) > 0.15 * np.nanmax(med), np.nan, q)
    inside = np.array([not surface.is_extrapolated(x, T)["outside_supported_range"] for x in prices])
    lo, _ = smile_quantile(surface, 0.05, T)
    hi, _ = smile_quantile(surface, 0.95, T)
    x = prices / 1000
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(x, np.nan_to_num(q), where=(prices >= lo) & (prices <= hi), color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, np.where(inside, q, np.nan), color=BLUE)
    ax.plot(x, np.where(inside, np.nan, q), color=BLUE, ls=(0, (3, 2)), lw=1.4)
    top = np.nanmax(q) * 1.18
    ax.set_ylim(0, top)
    for xv, label in ((lo, "5%"), (F, "forward"), (hi, "95%")):
        ax.axvline(xv / 1000, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(xv / 1000, top * 0.98, f" {label}\n {xv / 1000:,.0f}k", color=INK2,
                fontsize=9, va="top")
    ax.set_xlabel("BTC price in 90 days, $k")
    ax.set_ylabel("probability density")
    ax.set_yticklabels([])
    ax.set_title("2 · What the options market prices for BTC in 90 days")
    subtitle(ax, "Risk-neutral density read off the surface. Shaded: middle 90%. "
                 "Dashed: beyond the quoted strikes (assumed, not observed).")
    return save(fig, "2_implied_density")


def fig_premium():
    import dashboard.data as D
    p = D.load_premium().copy()
    p.index = pd.to_datetime(p.index)
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.18})
    a.plot(p.index, p["dvol"], color=BLUE, lw=1.4, label="implied (DVOL, 30d)")
    a.plot(p.index, p["realized"], color=ORANGE, lw=1.4, label="realised over the next 30d")
    a.set_ylabel("vol, %")
    a.legend(loc="upper right", ncol=2, fontsize=9.5)
    a.set_title("3 · The variance premium: real, and shrinking")
    subtitle(a, "BTC. Options usually price more vol than then happens; "
                "the gap was wide in 2021-22 and has mostly closed.")
    roll = p["premium"].rolling(180, min_periods=180).mean()
    b.axhline(0, color=INK2, lw=1)
    b.fill_between(p.index, roll, 0, where=roll >= 0, color=BLUE, alpha=0.18, lw=0)
    b.plot(p.index, roll, color=BLUE, lw=1.6)
    b.set_ylabel("premium, vol pts\n(180d mean)")
    return save(fig, "3_premium_history")


def fig_backtest():
    import dashboard.data as D
    cells = D.load_cells()
    cell = D.in_era(cells[(cells["asset"] == "BTC") & (cells["tenor"] == 30) & (cells["z"] == 0.0)
                          & (cells["policy"] == "ww 0.001") & (cells["slip_bps"] == 5.0)], "2023-26")
    br = D.cost_breakdown(cell)
    series = cell.set_index("date")["net_bps"].sort_index()
    m = D.cell_metrics(series)
    eq = D.equity_curve(series)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.25})
    labels = ["premium\nsold", "realised\nhedging P&L", "option\nspread", "hedge\nslippage", "net"]
    vals = br.to_numpy()
    running = 0.0
    for i, v in enumerate(vals):
        if i == len(vals) - 1:
            bottom, height = 0, v
        else:
            bottom, height = running, v
            running += v
        color = BLUE if height >= 0 else ORANGE
        a.bar(i, height, bottom=bottom, color=color, width=0.62)
        a.text(i, bottom + height + (8 if height >= 0 else -8), f"{v:+.0f}", ha="center",
               va="bottom" if height >= 0 else "top", color=INK, fontsize=9.5)
    a.axhline(0, color=INK2, lw=1)
    lowest = min(0.0, np.cumsum(vals[:-1]).min(), vals[-1])
    a.set_ylim(lowest - 90, vals[0] * 1.12)
    a.set_xticks(range(len(vals)), labels, fontsize=9)
    a.set_ylabel("bps of spot per position")
    a.set_title("4 · Where the premium goes")
    subtitle(a, "Short 30d ATM BTC straddle, Whalley-Wilmott hedge band,\n"
                "5 bps slippage, 2023-26. Mean per position.")

    b.fill_between(eq.index, eq["drawdown"], 0, color=ORANGE, alpha=0.18, lw=0, label="drawdown")
    b.plot(eq.index, eq["equity"], color=BLUE, label="cumulative net")
    b.axhline(0, color=INK2, lw=1)
    b.set_ylabel("bps of spot, cumulative")
    b.legend(loc="upper left", fontsize=9.5)
    import matplotlib.dates as mdates
    b.xaxis.set_major_locator(mdates.YearLocator())
    b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    b.set_title("5 · The same cell, one position a month")
    subtitle(b, f"Mean {m['mean_step_pnl']:+.0f} bps, t {m['t']:.2f}, Sharpe {m['sharpe']:.2f},\n"
                f"max drawdown {m['max_drawdown']:,.0f} bps, n = {m['n']}.")
    return save(fig, "4_costs_and_backtest")


def _script(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fig_scoreboard():
    vsm = _script("varswap")
    d = pd.read_parquet(RES / "dispersion.parquet")
    vb = vsm.book(pd.read_parquet(RES / "varswap.parquet"))
    nw_t = vsm.t_stat  # plain t on monthly positions, as the write-ups quote
    book_gross, book_net = vb["gross"], vb["net"]

    rows = [  # (label, t, source)
        ("Premium clears the option spread (BTC 30d ATM, 2023-26)", 2.4, "HANDOFF §12"),
        ("… and survives the hedge (best band, walked forward)", 1.75, "HANDOFF §12.2, t 1.4-2.1"),
        ("High funding → call wing bid (skew slope)", -5.2, "HYPOTHESES H7b"),
        ("High funding → downside realised vol", -3.3, "HYPOTHESES H7a"),
        ("Sell puts only when funding low: on − off", 2.69, "HYPOTHESES H8"),
        ("   … since 2023 only", 0.71, "HYPOTHESES H8"),
        ("ETH/BTC dispersion, straddles, before costs", 1.90, "DISPERSION part 1"),
        ("ETH/BTC dispersion, straddles, after costs", nw_t(d["book_net"]), "dispersion.parquet"),
        ("ETH/BTC dispersion, var swaps, before costs", nw_t(book_gross), "varswap.parquet"),
        ("ETH/BTC dispersion, var swaps, after costs", nw_t(book_net), "varswap.parquet"),
        ("Inverse vs linear books widen under stress", -0.05, "INVERSE_LINEAR P2"),
        ("Crash: vol level moves against spot (BTC, n=13)", -0.44 * np.sqrt(11) / np.sqrt(1 - 0.44**2), "SURFACE_PCA R3"),
    ]
    labels = [r[0] for r in rows][::-1]
    ts = np.array([r[1] for r in rows])[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    strong = np.abs(ts) >= 2
    ax.axvspan(-2, 2, color=GRID, alpha=0.6, lw=0)
    ax.barh(range(len(ts)), ts, color=np.where(strong, BLUE, MUTED), height=0.62)
    for i, t in enumerate(ts):
        ax.text(t + (0.12 if t >= 0 else -0.12), i, f"{t:+.1f}", va="center",
                ha="left" if t >= 0 else "right", color=INK, fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(ts)), labels, fontsize=9.5)
    ax.set_xlabel("t-statistic, as quoted in each write-up")
    ax.set_xlim(-6.5, 4.5)
    ax.set_title("6 · How strong is each finding?")
    subtitle(ax, "Blue: |t| ≥ 2. Grey band: what chance alone produces 95% of the time. "
                 "Pre-registered tests, failures included.")
    return save(fig, "6_evidence_scoreboard")


def fig_inverse_linear():
    h = pd.read_parquet(RES / "inverse_linear_hourly.parquet").sort_values("timestamp")
    t = pd.to_datetime(h["timestamp"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2})
    a.plot(t, h["index_price"] / 1000, color=INK2, lw=1.4)
    a.set_ylabel("BTC index, $k")
    a.set_title("7 · Two BTC option books, one price, even through the crash")
    subtitle(a, "Inverse (BTC-settled) vs linear (USDC-settled) Deribit books, hourly, "
                "same strikes and expiries.")
    roll = h.set_index(t)["abs_diff"].rolling("24h", min_periods=6).median() * 100
    b.plot(roll.index, roll, color=BLUE, lw=1.6)
    b.set_ylim(0, 0.5)
    b.set_ylabel("|vol gap|, vol pts\n(24h median)")
    b.text(0.99, 0.92, "median 0.13 calm, 0.125 in stress", transform=b.transAxes,
           ha="right", va="top", color=INK2, fontsize=9.5)
    return save(fig, "7_inverse_vs_linear")


def fig_pca():
    import importlib.util
    spec = importlib.util.spec_from_file_location("surface_pca", ROOT / "scripts" / "surface_pca.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    grid = pd.read_parquet(sp.CACHE)
    wide, spot, _ = sp.to_wide(grid, "BTC")
    dv, ret, ends = sp.changes_over(wide, spot, pd.Timedelta(hours=12))
    old = pd.DatetimeIndex(ends) <= sp.CUTOFF
    share, vec = sp.pca(dv[old])
    cells = list(wide.columns)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.28})
    load = pd.Series(vec[:, 0], index=pd.MultiIndex.from_tuples(cells)).unstack()
    for tenor, color in zip([7, 30, 90], [ORANGE, BLUE, AQUA]):
        if tenor in load.index:
            row = load.loc[tenor].dropna()
            a.plot(row.index, row.values, color=color, marker="o", ms=5)
            a.text(row.index[-1] + 0.08, row.values[-1], f"{tenor}d", color=INK2, va="center", fontsize=9.5)
    a.set_xlabel("strike, standardised moneyness z")
    a.set_ylabel("loading on factor 1")
    a.set_title("8 · The surface moves mostly as one level factor")
    subtitle(a, f"BTC, 12h changes, fitted pre-crash. Factor 1 = {share[0]:.0%};\n"
                "the 7d slice moves ~6x the 90d.")
    groups = ["BTC", "ETH"]
    res = {}
    for asset in groups:
        w, s, _ = sp.to_wide(grid, asset)
        d2, _, e2 = sp.changes_over(w, s, pd.Timedelta(hours=12))
        o = pd.DatetimeIndex(e2) <= sp.CUTOFF
        sh, v2 = sp.pca(d2[o])
        res[asset] = (sh[:3].sum(), sp.oos_share(d2[~o], v2[:, :3]))
    x = np.arange(2)
    a2 = b
    a2.bar(x - 0.17, [res[g][0] * 100 for g in groups], width=0.32, color=BLUE, label="quiet months (fit)")
    a2.bar(x + 0.17, [res[g][1] * 100 for g in groups], width=0.32, color=ORANGE, label="Nov sell-off (unseen)")
    for i, g in enumerate(groups):
        for dx, v in ((-0.17, res[g][0]), (0.17, res[g][1])):
            a2.text(i + dx, v * 100 + 1.5, f"{v:.1%}", ha="center", fontsize=9, color=INK)
    a2.set_xticks(x, groups)
    a2.set_ylim(0, 112)
    a2.set_yticks(range(0, 101, 25))
    a2.set_ylabel("variance explained by 3 factors, %")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9.5)
    a2.set_title("9 · …and the same factors describe the crash")
    subtitle(a2, "Loadings fitted before 5 Nov,\njudged on the sell-off they never saw.")
    return save(fig, "8_surface_factors")


def main() -> int:
    surface, stamp = latest_surface()
    paths = [fig_smiles(surface, stamp), fig_density(surface, stamp), fig_premium(),
             fig_backtest(), fig_scoreboard(), fig_inverse_linear(), fig_pca()]
    print(f"{len(paths)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
