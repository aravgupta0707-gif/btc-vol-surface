"""
Render the surface snapshots the write-up needs.

    python scripts/make_figures.py

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


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    asset = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    FIG_DIR.mkdir(exist_ok=True)

    snapshot, stamp = latest_chain(asset)
    surface, fits, ranges, quotes, refused = build(snapshot, asset)
    print(f"{asset} chain {stamp:%Y-%m-%d %H:%M}Z: {len(fits)} slices fitted, "
          f"{len(refused)} refused, {len(quotes)} inverted quotes")

    smile = FIG_DIR / f"{asset.lower()}_smile.png"
    surf = FIG_DIR / f"{asset.lower()}_surface.png"
    smile_figure(surface, fits, ranges, quotes, asset, stamp, smile)
    surface_figure(surface, fits, ranges, asset, stamp, surf)
    for p in (smile, surf):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
