"""
Render the surface snapshots and the premium figures the write-ups need.

    python scripts/make_figures.py            surface snapshots + premium figures
    python scripts/make_figures.py premium    premium figures only (no live chain needed)

BUILD_GUIDE.md section 9 asks the README to carry "a couple of surface
snapshots". These are those, generated from the most recent Track A chain so
they can be regenerated rather than being screenshots nobody can reproduce.

Two figures, chosen because they are the two the surface has to get right:

- **`figures/smile.png`** — the fitted SVI slice against the market's own
  quotes, per expiry. This is the honest one: it shows the fit next to what
  it is fitting, so a bad slice is visible rather than averaged into an RMSE.
  The quoted range is shaded, because outside it the surface clamps and the
  curve there is a boundary value rather than a fit.
- **`figures/surface.png`** — implied vol over log-moneyness and tenor, the
  conventional view. Drawn from the same `VolSurface` the strategy queries,
  not from a separate plotting path, so it cannot flatter a surface the
  backtest would not see.

Matplotlib rather than plotly: these are static images for a README, and
plotly's static export needs `kaleido`, a second dependency for no gain. The
dashboard is where plotly earns its place.

The four premium figures illustrate HYPOTHESES.md and are drawn from the same
panel `vrp_hypotheses.py` tests on (its functions are imported, not copied):

- **`premium_time.png`** -- implied against forward realised, and the premium.
- **`premium_level.png`** -- the premium against the DVOL level, by era.
- **`premium_funding.png`** -- funding against the premium, raw and with the
  vol level partialled out.
- **`premium_drawdown.png`** -- the premium by 90-day drawdown tercile.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
# Enough expiries to show the term structure, few enough to stay readable.
MAX_SLICES = 6


def latest_chain(asset: str = "BTC"):
    """The most recent Track A snapshot, as a single-timestamp frame."""
    import pandas as pd

    files = sorted(glob.glob(str(ROOT / "data" / "raw" / f"{asset.lower()}_chain_*.parquet")))
    if not files:
        raise SystemExit(f"no {asset} chains in data/raw -- run the collector first")
    raw = pd.read_parquet(files[-1])
    if "underlying_asset" in raw.columns:
        raw = raw[raw["underlying_asset"] == asset]
    t0 = pd.Timestamp(raw["timestamp"].min())
    return raw[raw["timestamp"] == t0], t0


def build(snapshot, asset: str):
    from surface.build import build_from_snapshot
    from surface.surface import from_fits

    fits, ranges, refused, quotes = build_from_snapshot(snapshot, asset)
    if not fits:
        raise SystemExit("no slice fitted on the latest chain")
    return from_fits(fits, quoted_ranges=ranges), fits, ranges, quotes, refused


def smile_figure(surface, fits, ranges, quotes, asset, stamp, path):
    import matplotlib.pyplot as plt

    Ts = sorted(fits)[:MAX_SLICES]
    cols = 3
    rows = int(np.ceil(len(Ts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.3 * rows), squeeze=False)

    for ax, T in zip(axes.flat, Ts):
        F = surface.forward(T)
        obs = quotes[np.isclose(quotes["T"], T)]
        lo, hi = ranges[T]
        grid = np.linspace(lo - 0.35, hi + 0.35, 300)
        fitted = [surface.iv(F * np.exp(k), T) * 100 for k in grid]

        ax.axvspan(lo, hi, color="0.92", zorder=0)
        ax.plot(grid, fitted, color="#1f4e79", lw=1.8, zorder=2, label="SVI")
        if len(obs):
            ax.scatter(obs["log_moneyness"], obs["iv"] * 100, s=14, zorder=3,
                       color="#c1121f", alpha=0.8, label="market mid")
        ax.set_title(f"{T * 365.25:.0f}d to expiry   F={F:,.0f}", fontsize=9)
        ax.set_xlabel("log-moneyness  k = ln(K/F)", fontsize=8)
        ax.set_ylabel("implied vol (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes.flat[len(Ts):]:
        ax.axis("off")

    axes.flat[0].legend(fontsize=7, loc="upper center")
    fig.suptitle(
        f"{asset} SVI slices against the market, {stamp:%Y-%m-%d %H:%M}Z\n"
        "shaded band = quoted strike range; outside it the surface clamps",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def surface_figure(surface, fits, ranges, asset, stamp, path):
    import matplotlib.pyplot as plt

    Ts = np.array(sorted(fits))
    # Standardised moneyness z = k / (atm_vol * sqrt(T)), the same coordinate
    # `hedge_frequency`'s Z_GRID uses. Plotting in raw k instead would set the
    # axis by the *narrowest* slice -- the 1-day one is quoted over about
    # +/-0.03 -- and collapse the picture to a sliver of the real surface.
    # In z every slice spans a comparable range, which is the point.
    zs = np.linspace(-2.0, 2.0, 90)
    rows = []
    for T in Ts:
        F = surface.forward(T)
        atm = surface.iv(F, T)
        rows.append([surface.iv(F * np.exp(z * atm * np.sqrt(T)), T) * 100 for z in zs])
    Z = np.array(rows)

    fig = plt.figure(figsize=(9.5, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    K, TT = np.meshgrid(zs, Ts * 365.25)
    ax.plot_surface(K, TT, Z, cmap="viridis", edgecolor="none", alpha=0.92,
                    rstride=1, cstride=2)
    ax.set_xlabel("standardised moneyness  z", fontsize=8, labelpad=2)
    ax.set_ylabel("days to expiry", fontsize=8, labelpad=2)
    ax.set_zlabel("implied vol (%)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-128)
    ax.set_title(f"{asset} implied volatility surface, {stamp:%Y-%m-%d %H:%M}Z\n"
                 f"{len(Ts)} fitted slices, in standardised moneyness "
                 "z = k / (atm vol * sqrt(T))",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Light-mode reference palette from the dataviz skill; slots 1-2 validate all-pairs.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ERA_SPLIT = "2023-01-01"


def _vh():
    """vrp_hypotheses.py as a module, so the figures use its panel, not a copy."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vrp_hypotheses", ROOT / "scripts" / "vrp_hypotheses.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrp_hypotheses"] = module
    spec.loader.exec_module(module)
    return module


def _figure(*args, **kw):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
    return fig, axes


def _save(fig, path):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _binned(x, y, bins: int):
    """Mean of x and of y within equal-count bins of x."""
    import pandas as pd

    q = pd.qcut(x, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}, index=x.index).groupby(q)
    return g["x"].mean().to_numpy(), g["y"].mean().to_numpy()


def _mean_line(ax, x, y, label=None):
    bx, by = _binned(x, y, 10)
    ax.plot(bx, by, color=INK, lw=2, marker="o", ms=8, mfc=INK, mec=SURFACE, mew=2,
            label=label)


def _legend(ax, **kw):
    ax.legend(fontsize=8, frameon=False, labelcolor=INK, **kw)


def premium_time_figure(panel, path):
    import pandas as pd

    split = pd.Timestamp(ERA_SPLIT)
    fig, (top, bot) = _figure(2, 1, figsize=(9, 5.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 2]))
    top.plot(panel.index, panel["dvol"], color=BLUE, lw=1.4, label="implied (DVOL)")
    top.plot(panel.index, panel["realized"], color=ORANGE, lw=1.4,
             label="realised over the next 30 days")
    top.set_ylabel("annualised vol (%)")
    _legend(top, loc="upper right")

    roll = panel["premium"].rolling(90, min_periods=60).mean()
    bot.fill_between(panel.index, 0, panel["premium"], color=BLUE, alpha=0.18, lw=0,
                     label="daily")
    bot.plot(roll.index, roll, color=BLUE, lw=2, label="90-day mean")
    bot.axhline(0, color=INK_2, lw=0.8)
    bot.set_ylabel("premium (vol pts)")
    _legend(bot, loc="lower left", ncol=2)
    for ax in (top, bot):
        ax.axvline(split, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    for label, sub in (("2021-22", panel[panel.index < split]),
                       ("2023-26", panel[panel.index >= split])):
        bot.text(sub.index[len(sub) // 2], bot.get_ylim()[1] * 0.92,
                 f"{label}  mean {sub['premium'].mean():+.1f}",
                 ha="center", va="top", fontsize=8, color=INK)
    top.set_title("BTC 30-day implied vs realised vol, and the premium between them",
                  fontsize=10, color=INK, loc="left")
    _save(fig, path)


def premium_level_figure(panel, path, slope, t):
    import pandas as pd

    early = panel.index < pd.Timestamp(ERA_SPLIT)
    fig, ax = _figure(figsize=(7.5, 5))
    for mask, color, label in ((early, BLUE, "2021-22"), (~early, ORANGE, "2023-26")):
        sub = panel[mask]
        ax.scatter(sub["dvol"], sub["premium"], s=9, color=color, alpha=0.35, lw=0,
                   label=f"{label}, daily")
    _mean_line(ax, panel["dvol"], panel["premium"], "all days, mean per DVOL decile")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("DVOL on the observation date (%)")
    ax.set_ylabel("premium over the next 30 days (vol pts)")
    _legend(ax, loc="upper left")
    ax.set_title(f"The premium scales with the vol level: {slope:+.2f} per DVOL point "
                 f"(NW t {t:.1f})", fontsize=10, color=INK, loc="left")
    _save(fig, path)


def premium_funding_figure(frame, path, slope, t):
    def resid(col):
        b = np.polyfit(frame["dvol"], frame[col], 1)
        return frame[col] - np.polyval(b, frame["dvol"])

    fig, axes = _figure(1, 2, figsize=(10, 4.4))
    cases = ((frame["funding30"], frame["premium"], "raw",
              "30-day mean funding (annualised %)", "premium (vol pts)"),
             (resid("funding30"), resid("premium"), "DVOL level partialled out",
              "funding, residual on DVOL", "premium, residual on DVOL"))
    for ax, (x, y, title, xl, yl) in zip(axes, cases):
        ax.scatter(x, y, s=9, color=BLUE, alpha=0.25, lw=0, label="daily")
        _mean_line(ax, x, y, "mean per funding decile")
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    _legend(axes[0], loc="upper right")
    fig.suptitle(f"Higher funding, lower premium: {slope:+.2f} per funding point with the "
                 f"level controlled (NW t {t:.1f})", fontsize=10, color=INK, x=0.01, ha="left")
    _save(fig, path)


def premium_drawdown_figure(frame, path):
    import pandas as pd

    bins = pd.qcut(frame["drawdown90"], 3, labels=["shallow", "middle", "deep"])
    g = frame.groupby(bins, observed=True).agg(
        dd=("drawdown90", "mean"), prem=("premium", "mean"),
        dvol=("dvol", "mean"), fund=("funding30", "mean"))
    fig, ax = _figure(figsize=(7, 4.6))
    x = np.arange(len(g))
    ax.bar(x, g["prem"], width=0.6, color=BLUE)
    for i, prem in enumerate(g["prem"]):
        ax.text(i, prem + 0.25, f"{prem:+.1f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, [f"{name}: {r.dd:.0f}% off 90d high\nDVOL {r.dvol:.0f}, "
                      f"funding {int(round(r.fund)):+d}%" for name, r in zip(g.index, g.itertuples())])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean premium over next 30 days (vol pts)")
    ax.set_ylim(0, g["prem"].max() * 1.2)
    ax.set_title("Premium by drawdown tercile: U-shaped, not rising with fear",
                 fontsize=10, color=INK, loc="left")
    _save(fig, path)


def premium_figures() -> list[Path]:
    vh = _vh()
    panel = vh.build_panel("BTC", vh.fetch_dvol("BTC", fetch=False))
    frame = panel.join([vh.trailing_funding("BTC"), vh.trailing_drawdown("BTC")],
                       how="inner").dropna()
    level, level_t = vh.nw_ols(panel["premium"], panel["dvol"])
    beta, ts = vh.nw_ols_multi(frame["premium"], frame[["funding30", "dvol"]])

    paths = [FIG_DIR / f"premium_{n}.png" for n in ("time", "level", "funding", "drawdown")]
    premium_time_figure(panel, paths[0])
    premium_level_figure(panel, paths[1], level, level_t)
    premium_funding_figure(frame, paths[2], beta[0], ts[0])
    premium_drawdown_figure(frame, paths[3])
    return paths


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    arg = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    FIG_DIR.mkdir(exist_ok=True)
    premium = premium_figures()
    if arg == "PREMIUM":
        for p in premium:
            print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
        return 0
    asset = arg

    snapshot, stamp = latest_chain(asset)
    surface, fits, ranges, quotes, refused = build(snapshot, asset)
    print(f"{asset} chain {stamp:%Y-%m-%d %H:%M}Z: {len(fits)} slices fitted, "
          f"{len(refused)} refused, {len(quotes)} inverted quotes")

    smile = FIG_DIR / f"{asset.lower()}_smile.png"
    surf = FIG_DIR / f"{asset.lower()}_surface.png"
    smile_figure(surface, fits, ranges, quotes, asset, stamp, smile)
    surface_figure(surface, fits, ranges, asset, stamp, surf)
    for p in (smile, surf, *premium):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
