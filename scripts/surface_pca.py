"""
How does the surface move? PCA on the live surfaces (SURFACE_PCA.md).

    python scripts/surface_pca.py            fit what is missing, then analyse
    python scripts/surface_pca.py --refit    refit every snapshot from scratch

The setup was fixed in SURFACE_PCA.md before this was written: two snapshots
a day per asset (nearest 08:00 and 20:00 UTC), each fitted with the same
`build_from_snapshot` -> `VolSurface` path the strategy uses, and read off on
a grid of tenor x standardised moneyness. The PCA is on changes between
consecutive 12h slots, never on levels: levels are dominated by whatever
regime the two months happened to be, and a factor model is about moves.

Fits are cached in `data/research/surface_grid.parquet`, one row per
(asset, slot, cell), so a rerun only fits snapshots it has not seen.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "research" / "surface_grid.parquet"
TENORS = (7, 14, 30, 60, 90)                       # days
ZS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)        # k / (atm vol * sqrt(T))
SLOT_HOURS = (8, 20)
SLOT_TOLERANCE = pd.Timedelta(hours=2)
MAX_EXTRAPOLATED = 0.10
WORKERS = 6
STAMP = re.compile(r"_(\d{8}T\d{6})Z\.parquet$")


# ------------------------------------------------------------------ selection

def file_stamp(path: str) -> pd.Timestamp:
    return pd.Timestamp(STAMP.search(path).group(1), tz="UTC")


def select_slots(stamps: list[pd.Timestamp], hours=SLOT_HOURS,
                 tolerance=SLOT_TOLERANCE) -> dict[pd.Timestamp, int]:
    """{slot time: index into `stamps` of the snapshot nearest it, within tolerance}."""
    if not stamps:
        return {}
    arr = pd.DatetimeIndex(stamps)
    days = pd.date_range(arr.min().floor("D"), arr.max().ceil("D"), freq="D")
    out = {}
    for day in days:
        for h in hours:
            slot = day + pd.Timedelta(hours=h)
            gaps = np.abs((arr - slot).total_seconds())
            i = int(np.argmin(gaps))
            if gaps[i] <= tolerance.total_seconds():
                out[slot] = i
    return out


# ---------------------------------------------------------------------- fit

def grid_from_file(args: tuple[str, str, pd.Timestamp]) -> list[dict]:
    """Fit one snapshot and read the grid off it. Top-level so a pool can pickle it."""
    path, asset, slot = args
    from surface.build import build_from_snapshot
    from surface.surface import from_fits

    raw = pd.read_parquet(path)
    if "underlying_asset" in raw.columns:
        raw = raw[raw["underlying_asset"] == asset]
    raw = raw[raw["timestamp"] == raw["timestamp"].min()]
    base = {"asset": asset, "slot": slot, "stamp": pd.Timestamp(raw["timestamp"].iloc[0])}
    try:
        fits, ranges, _, _ = build_from_snapshot(raw, asset)
        surface = from_fits(fits, quoted_ranges=ranges)
    except Exception as exc:  # a failed fit is a missing slot, not a crash
        return [dict(base, tenor=np.nan, z=np.nan, iv=np.nan, extrapolated=True,
                     spot=np.nan, error=str(exc)[:80])]
    spot = float(raw["index_price"].median()) if "index_price" in raw else np.nan
    rows = []
    for days in TENORS:
        T = days / 365.25
        F = surface.forward(T)
        atm = surface.iv(F, T)
        for z in ZS:
            K = F * np.exp(z * atm * np.sqrt(T))
            flags = surface.is_extrapolated(K, T)
            rows.append(dict(base, tenor=days, z=z, iv=surface.iv(K, T) * 100,
                             extrapolated=bool(flags["beyond_fitted_maturities"]
                                               or flags["outside_supported_range"]),
                             spot=spot, error=""))
    return rows


def fit_all(refit: bool) -> pd.DataFrame:
    from multiprocessing import Pool

    cached = pd.DataFrame() if refit or not CACHE.exists() else pd.read_parquet(CACHE)
    done = set() if cached.empty else set(zip(cached["asset"], cached["slot"]))
    jobs = []
    for asset in ("BTC", "ETH"):
        files = sorted(glob.glob(str(ROOT / "data" / "raw" / f"{asset.lower()}_chain_*.parquet")))
        stamps = [file_stamp(f) for f in files]
        for slot, i in select_slots(stamps).items():
            if (asset, slot) not in done:
                jobs.append((files[i], asset, slot))
    print(f"{len(done)} slots cached, {len(jobs)} to fit on {WORKERS} workers")
    if jobs:
        with Pool(WORKERS) as pool:
            rows = [r for chunk in pool.imap_unordered(grid_from_file, jobs, chunksize=2)
                    for r in chunk]
        fresh = pd.DataFrame(rows)
        cached = fresh if cached.empty else pd.concat([cached, fresh], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(CACHE, index=False)
    return cached


# ------------------------------------------------------------------ analysis

def pca(changes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Explained-variance shares and loadings (columns), largest first.

    Each loading's sign is fixed so its entries sum to >= 0; eigenvectors
    have no intrinsic sign, and without this two halves of the sample can
    disagree by a flip that is not a real difference.
    """
    X = changes - changes.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(X, rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    vectors = vectors * np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    return values / values.sum(), vectors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def to_wide(grid: pd.DataFrame, asset: str) -> tuple[pd.DataFrame, pd.Series, list]:
    """Slot x cell matrix of IV, cells that are rarely extrapolated, and spot by slot."""
    g = grid[(grid["asset"] == asset) & (grid["error"] == "")]
    share = g.groupby(["tenor", "z"])["extrapolated"].mean()
    keep = sorted(share[share <= MAX_EXTRAPOLATED].index)
    wide = g.pivot_table(index="slot", columns=["tenor", "z"], values="iv")[keep].sort_index()
    spot = g.groupby("slot")["spot"].first().reindex(wide.index)
    dropped = sorted(set(share.index) - set(keep))
    return wide.dropna(), spot, dropped


def changes_over(wide: pd.DataFrame, spot: pd.Series, step: pd.Timedelta):
    """Changes between slots exactly `step` apart; anything spanning a gap is dropped."""
    later = wide.index + step
    pairs = [(t, u) for t, u in zip(wide.index, later) if u in wide.index]
    if not pairs:
        return np.empty((0, wide.shape[1])), np.empty(0), []
    a = [t for t, _ in pairs]
    b = [u for _, u in pairs]
    dv = wide.loc[b].to_numpy() - wide.loc[a].to_numpy()
    ret = np.log(spot.loc[b].to_numpy() / spot.loc[a].to_numpy())
    return dv, ret, b


def print_loadings(vec: np.ndarray, cells: list, title: str) -> None:
    print(f"\n  {title}  (rows tenor in days, columns z)")
    frame = pd.Series(vec, index=pd.MultiIndex.from_tuples(cells)).unstack()
    print("    " + frame.round(2).to_string().replace("\n", "\n    "))


def analyse(grid: pd.DataFrame) -> dict:
    out = {}
    for asset in ("BTC", "ETH"):
        wide, spot, dropped = to_wide(grid, asset)
        cells = list(wide.columns)
        dv, ret, ends = changes_over(wide, spot, pd.Timedelta(hours=12))
        print(f"\n{'=' * 78}\n{asset}: {len(wide)} slots "
              f"({wide.index[0]:%Y-%m-%d} to {wide.index[-1]:%Y-%m-%d}), "
              f"{len(dv)} 12h changes, {len(cells)} of {len(TENORS) * len(ZS)} cells")
        if dropped:
            print(f"  dropped as extrapolated on >{MAX_EXTRAPOLATED:.0%} of slots: {dropped}")
        share, vec = pca(dv)
        out[asset] = dict(vec=vec, cells=cells)

        print(f"\n  explained: " + "  ".join(f"PC{i + 1} {s:.1%}" for i, s in enumerate(share[:5]))
              + f"   PC1-3 {share[:3].sum():.1%}")
        print(f"  E1 PC1 > 70%: {share[0]:.1%}; PC1 one sign: "
              f"{bool((vec[:, 0] > 0).all() or (vec[:, 0] < 0).all())}")
        for k in range(3):
            print_loadings(vec[:, k], cells, f"PC{k + 1} loadings ({share[k]:.1%})")

        idx = {c: i for i, c in enumerate(cells)}
        if (7, 0.0) in idx and (90, 0.0) in idx:
            r = vec[idx[(7, 0.0)], 0] / vec[idx[(90, 0.0)], 0]
            print(f"\n  E2 PC1 7d ATM / 90d ATM loading: {r:.2f}  (expected >= 1.5)")

        sd = dv.std(axis=0)
        print("  std of 12h changes by tenor at z=0 (vol pts): " + "  ".join(
            f"{t}d {sd[idx[(t, 0.0)]]:.2f}" for t in TENORS if (t, 0.0) in idx))

        half = len(dv) // 2
        _, v1 = pca(dv[:half])
        _, v2 = pca(dv[half:])
        print(f"  E5 halves (n={half}, {len(dv) - half}): |cos| PC1 {cosine(v1[:, 0], v2[:, 0]):.2f}"
              f"  PC2 {cosine(v1[:, 1], v2[:, 1]):.2f}  PC3 {cosine(v1[:, 2], v2[:, 2]):.2f}")

        scores = (dv - dv.mean(axis=0)) @ vec[:, :3]
        ok = np.isfinite(ret)
        for k in range(3):
            rho = np.corrcoef(scores[ok, k], ret[ok])[0, 1]
            print(f"  E6 corr(PC{k + 1} score, 12h spot log return) {rho:+.2f}"
                  f"   (n={ok.sum()}, ~±{2 / np.sqrt(ok.sum()):.2f} is 2 s.e.)")
        mean_atm = wide[(30, 0.0)].mean() if (30, 0.0) in wide else np.nan
        print(f"  ATM 30d level over the sample: mean {mean_atm:.1f}, "
              f"range {wide[(30, 0.0)].min():.1f}-{wide[(30, 0.0)].max():.1f}")

        daily_wide = wide[wide.index.hour == 8]
        dd, _, _ = changes_over(daily_wide, spot.reindex(daily_wide.index), pd.Timedelta(days=1))
        if len(dd) > 10:
            ds, dvec = pca(dd)
            print(f"  robustness, daily 08:00 changes (n={len(dd)}): PC1 {ds[0]:.1%}, "
                  f"PC1-3 {ds[:3].sum():.1%}, |cos| PC1 vs 12h {cosine(dvec[:, 0], vec[:, 0]):.2f}"
                  f"  PC2 {cosine(dvec[:, 1], vec[:, 1]):.2f}")

    common = [c for c in out["BTC"]["cells"] if c in out["ETH"]["cells"]]
    if common:
        def sub(a, k):
            i = [out[a]["cells"].index(c) for c in common]
            return out[a]["vec"][i, k]
        print(f"\nE7 BTC vs ETH on {len(common)} shared cells: |cos| PC1 {cosine(sub('BTC', 0), sub('ETH', 0)):.2f}"
              f"  PC2 {cosine(sub('BTC', 1), sub('ETH', 1)):.2f}  PC3 {cosine(sub('BTC', 2), sub('ETH', 2)):.2f}")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    grid = fit_all(refit="--refit" in argv)
    failed = grid[grid["error"] != ""]
    if len(failed):
        print(f"{failed['slot'].nunique()} slots failed to fit: "
              f"{failed['error'].value_counts().head(3).to_dict()}")
    analyse(grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
