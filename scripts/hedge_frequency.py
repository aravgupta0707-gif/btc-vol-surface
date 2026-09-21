"""
Can the surviving premium actually be collected, at any hedging frequency?

    python scripts/hedge_frequency.py run          fit, sweep, report (local)
    python scripts/hedge_frequency.py fit          stage 1 only  (shardable)
    python scripts/hedge_frequency.py sweep        stage 2 only  (shardable)
    python scripts/hedge_frequency.py report       re-print from saved cells

HANDOFF.md section 11.2 found +83 bps of spot a month on the BTC 30-day ATM
straddle and the call wing in 2023-26 -- priced at mid, with no spread and no
hedge cost. Section 5 calls that "about the size of realistic costs", which
makes the frequency of rebalancing, not the premium, the thing that decides
the outcome. Slippage scales *with* rebalancing and discrete-hedging error
scales *against* it, so there is a minimum somewhere and the question is
whether it sits above or below zero.

This sweeps that: hedge every 1/2/4/6/12/24h, charge 0/2/5/10 bps of traded
notional on each rebalance, both assets, both tenors, the same nine
standardised strikes.

**The option spread is measured, not assumed.** The archive carries bid and
ask per instrument, so each expiry gets a half-spread curve in vol points,
(ask_usd - bid_usd) / 2 / vega, interpolated in log-moneyness onto the
synthetic strike. The seller is filled at that bid, once, at inception --
there is no exit spread because the position is held to expiry and settled
at intrinsic. Only the hedge leg's cost is a parameter, because the hourly
index carries no book to measure one from.

Two stages, because the expensive part is not the part being swept. Fitting a
day's surface is ~98% of the work and does not depend on the hedge interval
or the slippage assumption at all, so stage 1 fits each archive day once and
banks the SVI parameters and the spread curve; stage 2 sweeps the 48 grid
cells over that cache. Re-fitting per cell would burn ~1.9 CPU-hours to
recompute identical surfaces 48 times.

Both stages shard with `--shard i/n` for a Slurm array; every task writes its
own file and nothing is shared, so there is no locking and no combine step.
See slurm/README.md.

What this is not: the gross column is still a mid-priced number and carries
every caveat section 11.2 does -- one inception a month, adjacent strikes not
independent, 72 cells reported. The net column adds costs to that, it does
not rescue it. A frequency that turns net positive on a strike that was not
significant gross is noise with a cost model bolted on.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIST_DIR = ROOT / "data" / "historical"
RESEARCH = ROOT / "data" / "research"
FIT_DIR = RESEARCH / "hedge_fits"
CELL_DIR = RESEARCH / "hedge_cells"

def _grid(name: str, default: tuple, cast=float) -> tuple:
    """
    Grid axis, overridable from the environment.

    A cluster sweep wants to widen the grid without editing the source the
    job is running from -- `HEDGE_HOURS="1 2 3 4 6 8 12 24" sbatch ...` --
    and every task in an array must agree on the axis, which an environment
    variable exported once in the job script guarantees and an edited file
    does not.
    """
    raw = os.environ.get(name)
    return default if not raw else tuple(cast(x) for x in raw.replace(",", " ").split())


ASSETS = ("BTC", "ETH")
TENOR_DAYS = _grid("TENOR_DAYS", (7, 30), int)
Z_GRID = _grid("Z_GRID", (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0))
HEDGE_HOURS = _grid("HEDGE_HOURS", (1, 2, 4, 6, 12, 24), int)
SLIPPAGE_BPS = _grid("SLIPPAGE_BPS", (0.0, 2.0, 5.0, 10.0))
CALL_WING = tuple(z for z in Z_GRID if z > 0)

YEAR_HOURS = 365.25 * 24
ERAS = {"2020-22": (2020, 2022), "2023-26": (2023, 2026)}
# Below this the option has no vega to divide a price spread by, and the
# vol-space half-spread explodes. Those strikes are dropped from the curve,
# not from the study -- the curve is interpolated, so they stay covered.
MIN_VEGA_USD = 1.0

SVI_KEYS = ("a", "b", "rho", "m", "sigma")


# ---- grid ---------------------------------------------------------------

def cells() -> list:
    """Every (asset, hedge interval, slippage) the sweep covers."""
    return [(a, h, s) for a in ASSETS for h in HEDGE_HOURS for s in SLIPPAGE_BPS]


def shard(items: list, spec) -> list:
    """
    `items[i::n]` for spec "i/n".

    Strided, not contiguous: archive days grow from 7k quotes in 2020 to 22k
    in 2026, so contiguous chunks would hand one task all the expensive days.
    """
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {i} outside 0..{n - 1}")
    return items[i::n]


# ---- stage 1: fit each archive day once ---------------------------------

def fit_day(path_file):
    """One archive day -> its SVI slices and its measured half-spread curve."""
    from pricing.black76 import vega
    from surface.build import build_from_snapshot

    asset = path_file.name.split("_")[0].upper()
    raw = pd.read_parquet(path_file)
    t0 = pd.Timestamp(raw["timestamp"].min())
    base = {"asset": asset, "date": t0.date()}
    try:
        fits, _, _, quotes = build_from_snapshot(raw[raw["timestamp"] == t0], asset)
    except Exception as exc:  # noqa: BLE001 - recorded per day, not raised
        return [dict(base, T=np.nan, skipped=f"no surface: {type(exc).__name__}")], []

    fit_rows = [
        dict(base, T=float(T), skipped="",
             **{key: float(fit[key]) for key in SVI_KEYS},
             forward=float(fit["forward"]),
             k_min=float(fit["k_range"][0]), k_max=float(fit["k_range"][1]))
        for T, fit in fits.items()
    ]

    need = {"bid_usd", "ask_usd", "forward", "strike", "T", "iv", "log_moneyness"}
    if not need <= set(quotes.columns):
        raise KeyError(f"cleaned quotes lack {sorted(need - set(quotes.columns))}")

    v = np.asarray(vega(quotes["forward"].to_numpy(float), quotes["strike"].to_numpy(float),
                        quotes["T"].to_numpy(float), quotes["iv"].to_numpy(float)), dtype=float)
    half_usd = (quotes["ask_usd"].to_numpy(float) - quotes["bid_usd"].to_numpy(float)) / 2.0
    keep = (v > MIN_VEGA_USD) & np.isfinite(half_usd) & (half_usd >= 0)
    spread_rows = [
        dict(base, T=float(t), k=float(k), half_spread_vol=float(h / vv))
        for t, k, h, vv in zip(quotes["T"].to_numpy(float)[keep],
                               quotes["log_moneyness"].to_numpy(float)[keep],
                               half_usd[keep], v[keep])
    ]
    return fit_rows, spread_rows


def run_fit(spec, only, workers: int) -> None:
    files = [f for a in ASSETS if only in (None, a)
             for f in sorted(HIST_DIR.glob(f"{a.lower()}_chain_*.parquet"))]
    mine = shard(files, spec)
    tag = (spec or "all").replace("/", "of")
    print(f"fit: {len(mine)} of {len(files)} archive days  (shard {spec or 'all'})", flush=True)

    fit_rows, spread_rows = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (fits, spreads) in enumerate(pool.map(fit_day, mine), 1):
            fit_rows.extend(fits)
            spread_rows.extend(spreads)
            if i % 10 == 0:
                print(f"  {i}/{len(mine)}", flush=True)

    FIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_rows).to_parquet(FIT_DIR / f"fits_{tag}.parquet", index=False)
    pd.DataFrame(spread_rows).to_parquet(FIT_DIR / f"spreads_{tag}.parquet", index=False)
    print(f"fit: wrote {len(fit_rows)} slices, {len(spread_rows)} spread points", flush=True)


def _no_duplicates(frame, keys: list, where: Path) -> None:
    """
    Refuse a cache that holds two generations of shards.

    Both loaders glob, so a run at `--shard i/4` followed by one at `i/8`, or
    by an unsharded `run`, leaves both sets on disk and every day is counted
    twice. The means barely move and the t-statistics inflate by ~sqrt(2),
    which is the worst kind of wrong: plausible. Cheaper to refuse than to
    explain later.
    """
    dupes = frame.duplicated(subset=keys).sum()
    if dupes:
        raise ValueError(
            f"{where} holds {dupes} duplicate rows on {keys} -- it has shards "
            f"from more than one run. Delete {where}/*.parquet and redo the "
            f"stage (slurm/submit.sh does this for you)."
        )


def load_fits():
    fit_files = sorted(FIT_DIR.glob("fits_*.parquet"))
    if not fit_files:
        raise FileNotFoundError(f"no fits in {FIT_DIR}: run the fit stage first")
    fits = pd.concat([pd.read_parquet(p) for p in fit_files], ignore_index=True)
    spreads = pd.concat([pd.read_parquet(p) for p in sorted(FIT_DIR.glob("spreads_*.parquet"))],
                        ignore_index=True)
    # Only the usable rows: a skipped day carries NaN for T, and comparing
    # NaN keys would read two different failures as one repeated day.
    _no_duplicates(fits[fits["skipped"] == ""], ["asset", "date", "T"], FIT_DIR)
    return fits, spreads


# ---- the position -------------------------------------------------------

def hedged_seller_pnl(spots, hours, K: float, T0: float, sigma: float,
                      carry: float, option: str,
                      half_spread_vol: float = 0.0, slip_bps: float = 0.0) -> dict:
    """
    Seller's P&L on one option, hedged at every point of `hours`, with costs.

    `spots` is the index at `hours` (hours since inception, last = expiry).
    The hedge is sized on the forward F = S exp(carry * T_remaining), the same
    convention the option is priced in; at expiry F = S. Delta is Black-76 at
    the option's own inception implied vol.

    `gross` reproduces scripts/vrp_surface.py exactly when both costs are
    zero -- that equality is asserted in the tests, so this stays a cost model
    bolted onto the published result rather than a second, different
    calculation of it.

    Costs, for a seller who is short the option and long delta to hedge it:
      - the option is sold at the bid, `sigma - half_spread_vol`, once;
      - every rebalance pays `slip_bps` on the notional it trades, including
        putting the initial hedge on and lifting it at expiry.
    """
    from pricing.black76 import intrinsic, price

    T_left = np.maximum(T0 - hours / YEAR_HOURS, 0.0)
    F = spots * np.exp(carry * T_left)
    mid_premium = float(price(F[0], K, T0, sigma, option))

    sell_vol = sigma - half_spread_vol
    if sell_vol <= 0:
        return {"unsellable": True}
    received = float(price(F[0], K, T0, sell_vol, option))

    live = T_left[:-1] > 0
    d1 = np.where(live, (np.log(F[:-1] / K) + 0.5 * sigma**2 * T_left[:-1])
                  / (sigma * np.sqrt(np.where(live, T_left[:-1], 1.0))), 0.0)
    delta = norm.cdf(d1) if option == "C" else norm.cdf(d1) - 1.0

    hedge_gain = float(np.sum(delta * np.diff(F)))
    payoff = float(intrinsic(spots[-1], K, option))

    # |d0 - 0| to put the hedge on, |di - d(i-1)| at each rebalance, |d_last|
    # to lift it at expiry. Each trade pays at the forward prevailing then.
    traded = np.abs(np.diff(delta, prepend=0.0))
    notional = float(np.sum(traded * F[:-1])) + abs(float(delta[-1])) * float(F[-1])
    slippage = notional * slip_bps / 1e4

    return {
        "unsellable": False,
        "premium": mid_premium,
        "gross": mid_premium + hedge_gain - payoff,
        "net": received + hedge_gain - payoff - slippage,
        "spread_cost": mid_premium - received,
        "slippage": slippage,
        "turnover": notional,
        "hedges": int(len(delta)),
    }


# ---- stage 2: sweep the grid over the cached fits -----------------------

def _surface_for(day_fits):
    from surface.surface import from_fits
    fits = {
        float(row["T"]): {**{key: float(row[key]) for key in SVI_KEYS},
                          "forward": float(row["forward"]),
                          "k_range": (float(row["k_min"]), float(row["k_max"]))}
        for _, row in day_fits.iterrows()
    }
    return from_fits(fits)


def _half_spread_at(curve, T: float, k: float) -> float:
    """
    Half-spread in vol points at (T, k), from the measured curve.

    Nearest fitted expiry in T, then linear in k with the ends held flat --
    the same clamp the surface itself uses outside its quoted range, so the
    spread does not extrapolate where the vol does not.
    """
    if curve.empty:
        return float("nan")
    tenors = curve["T"].unique()
    nearest = tenors[np.argmin(np.abs(tenors - T))]
    slice_ = curve[curve["T"] == nearest].sort_values("k")
    return float(np.interp(k, slice_["k"].to_numpy(), slice_["half_spread_vol"].to_numpy()))


def run_cell(spec) -> list:
    """One (asset, hedge interval, slippage) over every archive day."""
    from data.history import hourly_funding

    asset, hedge_h, slip = spec
    fits_all, spreads_all = load_fits()
    fits_all = fits_all[(fits_all["asset"] == asset) & (fits_all["skipped"] == "")]
    spreads_all = spreads_all[spreads_all["asset"] == asset]
    series = hourly_funding(asset, offline=True).set_index("time")["index_price"]

    rows = []
    for date, day_fits in fits_all.groupby("date"):
        base = {"asset": asset, "date": date, "hedge_hours": hedge_h, "slip_bps": slip}
        try:
            surface = _surface_for(day_fits)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(base, skipped=f"no surface: {type(exc).__name__}"))
            continue
        curve = spreads_all[spreads_all["date"] == date]
        t0 = pd.Timestamp(date, tz="UTC")

        for days in TENOR_DAYS:
            tE = t0 + pd.Timedelta(days=days)
            window = series.loc[t0:tE].dropna()
            if window.empty or window.index[-1] < tE or window.index[0] > t0:
                rows.append(dict(base, tenor=days, skipped="index gap"))
                continue
            step = window.iloc[::hedge_h]
            if step.index[-1] != window.index[-1]:
                step = pd.concat([step, window.iloc[[-1]]])
            hours = ((step.index - t0) / pd.Timedelta(hours=1)).to_numpy(float)
            spots = step.to_numpy(float)

            T0 = days / 365.25
            F0 = surface.forward(T0)
            carry = float(np.log(F0 / spots[0]) / T0)
            atm = surface.iv(F0, T0)

            for z in Z_GRID:
                K = F0 * np.exp(z * atm * np.sqrt(T0))
                flags = surface.is_extrapolated(K, T0)
                sigma = surface.iv(K, T0)
                hs = _half_spread_at(curve, T0, float(np.log(K / F0)))
                legs = ("P", "C") if z == 0 else (("P",) if z < 0 else ("C",))

                out = [hedged_seller_pnl(spots, hours, K, T0, sigma, carry, leg,
                                         half_spread_vol=hs, slip_bps=slip)
                       for leg in legs]
                if not np.isfinite(hs) or any(o["unsellable"] for o in out):
                    rows.append(dict(base, tenor=days, z=z, skipped="spread exceeds vol"))
                    continue
                agg = {key: sum(o[key] for o in out) for key in
                       ("premium", "gross", "net", "spread_cost", "slippage", "turnover")}
                rows.append(dict(
                    base, tenor=days, z=z, skipped="", strike=K, spot=spots[0],
                    iv=sigma, atm_iv=atm, half_spread_vol=hs, hedges=out[0]["hedges"],
                    supported=not (flags["outside_supported_range"]
                                   or flags["beyond_fitted_maturities"]),
                    premium_bps=agg["premium"] / spots[0] * 1e4,
                    gross_bps=agg["gross"] / spots[0] * 1e4,
                    net_bps=agg["net"] / spots[0] * 1e4,
                    spread_bps=agg["spread_cost"] / spots[0] * 1e4,
                    slippage_bps=agg["slippage"] / spots[0] * 1e4,
                    turnover_x=agg["turnover"] / spots[0],
                ))
    return rows


def run_sweep(spec, workers: int) -> None:
    mine = shard(cells(), spec)
    tag = (spec or "all").replace("/", "of")
    print(f"sweep: {len(mine)} of {len(cells())} cells  (shard {spec or 'all'})", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell, out in zip(mine, pool.map(run_cell, mine)):
            rows.extend(out)
            print(f"  {cell[0]} {cell[1]:>2}h {cell[2]:>4.1f}bps -> {len(out)} rows", flush=True)

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(CELL_DIR / f"cells_{tag}.parquet", index=False)
    print(f"sweep: wrote {len(rows)} rows", flush=True)


# ---- report -------------------------------------------------------------

def _t(x) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def load_cells():
    files = sorted(CELL_DIR.glob("cells_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cells in {CELL_DIR}: run the sweep stage first")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    _no_duplicates(frame[frame["skipped"] == ""],
                   ["asset", "date", "tenor", "z", "hedge_hours", "slip_bps"], CELL_DIR)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    return frame


def _panel(sub, label: str) -> None:
    """mean net bps (t) over hedge interval x slippage, per era."""
    print(f"\n  {label}")
    # Axes come from the swept data, not from the module constants, so a
    # report still reads correctly against cells swept under a wider grid.
    intervals = sorted(sub["hedge_hours"].unique())
    slips = sorted(sub["slip_bps"].unique())
    header = "".join(f"{s:>6.0f} bps" for s in slips)
    for era, (lo, hi) in ERAS.items():
        era_rows = sub[(sub["year"] >= lo) & (sub["year"] <= hi)]
        if era_rows.empty:
            continue
        print(f"    {era}  hedge |{header}       n")
        for hours in intervals:
            line, count = "", 0
            for slip in slips:
                cell = era_rows[(era_rows["hedge_hours"] == hours)
                                & (era_rows["slip_bps"] == slip)]
                by_day = cell.groupby("date")["net_bps"].mean()
                count = len(by_day)
                if by_day.empty:
                    line += f"{'--':>11}"
                else:
                    line += f"{by_day.mean():>+6.0f}({_t(by_day):>+4.1f})"
            print(f"      {hours:>3}h      |{line}  {count:>5}")


def report(frame) -> None:
    ok = frame[(frame["skipped"] == "") & frame["supported"].fillna(False)]
    print(f"\nHedging frequency vs cost -- {len(ok)} priced positions, "
          f"{ok['date'].nunique()} inceptions")
    print("mean net bps of spot (t), seller, held to expiry, spread measured from the book")

    for asset in ASSETS:
        for days in TENOR_DAYS:
            sub = ok[(ok["asset"] == asset) & (ok["tenor"] == days)]
            if sub.empty:
                continue
            print(f"\n{'=' * 78}")
            print(f"{asset} {days}-day")
            _panel(sub[sub["z"] == 0.0], "ATM straddle (z = 0)")
            wing = sorted(z for z in sub["z"].unique() if z > 0)
            if wing:
                _panel(sub[sub["z"].isin(wing)],
                       f"call wing (z = +{min(wing)}..+{max(wing)}, per-day mean)")

    print(f"\n{'=' * 78}")
    print("cost structure at 5 bps slippage, BTC 30-day ATM, 2023-26:")
    look = ok[(ok["asset"] == "BTC") & (ok["tenor"] == 30) & (ok["z"] == 0.0)
              & (ok["slip_bps"] == 5.0) & (ok["year"] >= 2023)]
    if look.empty:
        print("  (no rows)")
    else:
        print(f"  {'hedge':>6}{'premium':>10}{'gross':>9}{'spread':>9}"
              f"{'slippage':>10}{'net':>9}{'turnover':>10}")
        for hours in sorted(look["hedge_hours"].unique()):
            row = look[look["hedge_hours"] == hours]
            if row.empty:
                continue
            print(f"  {hours:>5}h{row['premium_bps'].mean():>10.0f}"
                  f"{row['gross_bps'].mean():>9.0f}{row['spread_bps'].mean():>9.0f}"
                  f"{row['slippage_bps'].mean():>10.0f}{row['net_bps'].mean():>9.0f}"
                  f"{row['turnover_x'].mean():>9.1f}x")

    print("\nRead the gross column first: where it is not significant, a positive net is")
    print("noise with a cost model on it. Section 11.2's caveats all still apply.")


# ---- cli ----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="hedging frequency vs cost")
    parser.add_argument("stage", choices=("fit", "sweep", "report", "run"))
    parser.add_argument("--shard", help="i/n -- this task's slice of the work")
    parser.add_argument("--asset", choices=ASSETS, help="fit stage: one asset only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.stage == "run":
        # `run` means the whole study from scratch, so it owns these
        # directories. A sharded run left behind by an earlier array would
        # otherwise be globbed in alongside this one -- see _no_duplicates.
        # A bare `fit`/`sweep` must NOT do this: shards are each other's peers.
        for directory in (FIT_DIR, CELL_DIR):
            for stale in directory.glob("*.parquet"):
                stale.unlink()

    if args.stage in ("fit", "run"):
        run_fit(args.shard, args.asset, args.workers)
    if args.stage in ("sweep", "run"):
        run_sweep(args.shard if args.stage == "sweep" else None, args.workers)
    if args.stage in ("report", "run"):
        report(load_cells())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
