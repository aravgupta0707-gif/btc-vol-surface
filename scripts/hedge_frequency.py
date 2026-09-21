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

This sweeps that: sixteen rebalancing rules against 0/2/5/10 bps of traded
notional, both assets, both tenors, the same nine standardised strikes.

The rules are three families, and the comparison between them is the point:

- **clocks** -- rebalance every 1/2/4/6/12/24h, the only rule the project had.
- **fixed bands** -- hold until the delta drifts more than H from target.
- **Whalley-Wilmott** -- the asymptotic band H = (3/2 lambda F Gamma^2 /
  gamma)^(1/3), which widens where gamma is low and where cost is high.

A clock trades when the calendar says so, not when the position has moved,
which is why it can pay to re-hedge something that has barely drifted and sit
through something that has run. A band answers that directly, and it is the
right thing to try before anything learned: a policy that cannot beat a
closed form is not worth fitting. `strategy/hedging.py` holds all three and
reduces each to the same walk, so a band's number and a clock's number come
out of identical code over identical paths.

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

from strategy.hedging import walk  # noqa: E402 - needs the path above

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
FIXED_BANDS = _grid("FIXED_BANDS", (0.01, 0.02, 0.05, 0.10, 0.20))
WW_GAMMAS = _grid("WW_GAMMAS", (1e-4, 3e-4, 1e-3, 3e-3, 1e-2))
SLIPPAGE_BPS = _grid("SLIPPAGE_BPS", (0.0, 2.0, 5.0, 10.0))
CALL_WING = tuple(z for z in Z_GRID if z > 0)


def policies() -> list:
    """
    Every rebalancing rule the sweep compares, as (kind, param, label).

    The clock rules come first and are the baseline: they reproduce the
    fixed-interval result exactly, so a band's number is read against a
    number computed by the same walk over the same path.
    """
    return ([("time", float(h), f"time {h:g}h") for h in HEDGE_HOURS]
            + [("fixed", b, f"band {b:g}") for b in FIXED_BANDS]
            + [("whalley_wilmott", g, f"ww {g:g}") for g in WW_GAMMAS])

YEAR_HOURS = 365.25 * 24
ERAS = {"2020-22": (2020, 2022), "2023-26": (2023, 2026)}
# Below this the option has no vega to divide a price spread by, and the
# vol-space half-spread explodes. Those strikes are dropped from the curve,
# not from the study -- the curve is interpolated, so they stay covered.
MIN_VEGA_USD = 1.0

SVI_KEYS = ("a", "b", "rho", "m", "sigma")


# ---- grid ---------------------------------------------------------------

def cells() -> list:
    """Every (asset, policy, slippage) the sweep covers."""
    return [(a, p, s) for a in ASSETS for p in policies() for s in SLIPPAGE_BPS]


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

    **Reference implementation.** The sweep itself now runs through
    `strategy.hedging.walk`, which generalises this to rules that decide on
    drift rather than the clock. This one is kept because it is the tested
    link back to the published result: `gross` reproduces
    scripts/vrp_surface.py exactly at zero cost, and `walk` under a clock
    rule reproduces *this* exactly, so the chain from §11.2 to a band's
    number is asserted end to end rather than assumed.

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
    """One (asset, policy, slippage) over every archive day."""
    from data.history import hourly_funding

    asset, (kind, param, label), slip = spec
    fits_all, spreads_all = load_fits()
    fits_all = fits_all[(fits_all["asset"] == asset) & (fits_all["skipped"] == "")]
    spreads_all = spreads_all[spreads_all["asset"] == asset]
    series = hourly_funding(asset, offline=True).set_index("time")["index_price"]

    rows = []
    for date, day_fits in fits_all.groupby("date"):
        base = {"asset": asset, "date": date, "policy": label, "policy_kind": kind,
                "policy_param": param, "slip_bps": slip,
                "hedge_hours": param if kind == "time" else np.nan}
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
            # The full hourly path, every rule. A clock rule subsamples it
            # through its own `forced` mask, so a band is free to trade at
            # any hour and both are judged on identical data.
            hours = ((window.index - t0) / pd.Timedelta(hours=1)).to_numpy(float)
            spots = window.to_numpy(float)

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

                out = [walk(spots, hours, K, T0, sigma, carry, leg, kind, param,
                            slip_bps=slip, half_spread_vol=hs)
                       for leg in legs]
                if not np.isfinite(hs) or any(o["unsellable"] for o in out):
                    rows.append(dict(base, tenor=days, z=z, skipped="spread exceeds vol"))
                    continue
                agg = {key: sum(o[key] for o in out) for key in
                       ("premium", "gross", "net", "spread_cost", "slippage", "turnover")}
                rows.append(dict(
                    base, tenor=days, z=z, skipped="", strike=K, spot=spots[0],
                    iv=sigma, atm_iv=atm, half_spread_vol=hs,
                    trades=out[0]["trades"], mean_band=out[0]["mean_band"],
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
            print(f"  {cell[0]:4s} {cell[1][2]:<10s} {cell[2]:>4.1f}bps -> {len(out)} rows",
                  flush=True)

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
                   ["asset", "date", "tenor", "z", "policy", "slip_bps"], CELL_DIR)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    return frame


def _ordered_policies(sub) -> list:
    """
    Policies in grid order -- clocks, then fixed bands, then Whalley-Wilmott.

    Not alphabetical: the comparison only reads if the baseline the bands are
    meant to beat sits above them.
    """
    present = set(sub["policy"].unique())
    ordered = [label for _, _, label in policies() if label in present]
    return ordered + sorted(present - set(ordered))


def _panel(sub, label: str) -> None:
    """mean net bps (t) over rebalancing rule x slippage, per era."""
    print(f"\n  {label}")
    # Axes come from the swept data, not from the module constants, so a
    # report still reads correctly against cells swept under a wider grid.
    rules = _ordered_policies(sub)
    slips = sorted(sub["slip_bps"].unique())
    header = "".join(f"{s:>6.0f} bps" for s in slips)
    for era, (lo, hi) in ERAS.items():
        era_rows = sub[(sub["year"] >= lo) & (sub["year"] <= hi)]
        if era_rows.empty:
            continue
        print(f"    {era}  rule     |{header}       n")
        for rule in rules:
            line, count = "", 0
            for slip in slips:
                cell = era_rows[(era_rows["policy"] == rule)
                                & (era_rows["slip_bps"] == slip)]
                by_day = cell.groupby("date")["net_bps"].mean()
                count = len(by_day)
                if by_day.empty:
                    line += f"{'--':>11}"
                else:
                    line += f"{by_day.mean():>+6.0f}({_t(by_day):>+4.1f})"
            print(f"      {rule:<10s}|{line}  {count:>5}")


def report(frame) -> None:
    ok = frame[(frame["skipped"] == "") & frame["supported"].fillna(False)]
    print(f"\nRebalancing rule vs cost -- {len(ok)} priced positions, "
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
        print(f"  {'rule':<11}{'premium':>9}{'gross':>8}{'spread':>8}"
              f"{'slippage':>10}{'net':>8}{'t':>7}{'trades':>8}{'turnover':>10}{'band':>8}")
        for rule in _ordered_policies(look):
            row = look[look["policy"] == rule]
            if row.empty:
                continue
            by_day = row.groupby("date")["net_bps"].mean()
            band = row["mean_band"].mean()
            print(f"  {rule:<11}{row['premium_bps'].mean():>9.0f}"
                  f"{row['gross_bps'].mean():>8.0f}{row['spread_bps'].mean():>8.0f}"
                  f"{row['slippage_bps'].mean():>10.0f}{row['net_bps'].mean():>8.0f}"
                  f"{_t(by_day):>7.1f}{row['trades'].mean():>8.0f}"
                  f"{row['turnover_x'].mean():>9.1f}x"
                  f"{'' if not np.isfinite(band) else f'{band:>8.3f}'}")

    _best_by_cost(ok)

    print("\nRead the gross column first: where it is not significant, a positive net is")
    print("noise with a cost model on it. Section 11.2's caveats all still apply.")
    print("A band only earns its place by beating the best clock in the same column,")
    print("not by beating the naive hourly one -- and only if its gross holds up.")
    print("A band wide enough to lift net by hedging *less* has stopped being a hedge.")


def _best_by_cost(ok) -> None:
    """
    Best clock against best band, per cost level -- the study's actual question.

    Both sides are picked in sample, so both carry the same selection penalty;
    that is the point of comparing best-to-best rather than best-band against
    one arbitrary clock. `gross` is carried alongside because a band can lift
    net two different ways: by trading less for the same replication, which is
    the result, or by replicating worse, which is just a smaller hedge wearing
    the same name.
    """
    print(f"\n{'=' * 78}")
    print("best clock vs best band, ATM straddle, 2023-26 (net bps, gross in brackets)")
    for asset in ASSETS:
        for days in TENOR_DAYS:
            sub = ok[(ok["asset"] == asset) & (ok["tenor"] == days) & (ok["z"] == 0.0)
                     & (ok["year"] >= 2023)]
            if sub.empty:
                continue
            print(f"\n  {asset} {days}-day")
            print(f"    {'slip':>5}  {'best clock':>26}  {'best band':>26}  {'gain':>6}")
            for slip in sorted(sub["slip_bps"].unique()):
                at = sub[sub["slip_bps"] == slip]
                clocks = at[at["policy_kind"] == "time"]
                bands = at[at["policy_kind"] != "time"]
                if clocks.empty or bands.empty:
                    continue
                best = {}
                for name, group in (("clock", clocks), ("band", bands)):
                    means = group.groupby("policy")[["net_bps", "gross_bps"]].mean()
                    top = means["net_bps"].idxmax()
                    best[name] = (top, means.loc[top, "net_bps"], means.loc[top, "gross_bps"])
                gain = best["band"][1] - best["clock"][1]
                print(f"    {slip:>5.0f}  {best['clock'][0]:>10s} "
                      f"{best['clock'][1]:>+6.0f} ({best['clock'][2]:>+4.0f})  "
                      f"{best['band'][0]:>10s} {best['band'][1]:>+6.0f} "
                      f"({best['band'][2]:>+4.0f})  {gain:>+6.0f}")


# ---- walk-forward: choose the rule without hindsight --------------------

def settled_history(dates, as_of, tenor_days: int):
    """
    Inceptions whose outcome was *known* by `as_of`.

    Not "earlier inceptions" -- a 30-day position opened one month before
    `as_of` settles within hours of it, so its P&L was not available to choose
    with. Requiring inception + tenor < as_of costs one observation and is the
    difference between a walk-forward and a very slightly clairvoyant one.
    """
    return dates[dates + pd.Timedelta(days=tenor_days) < as_of]


def walk_forward(sub, rules: list, tenor_days: int, warmup: int = 6):
    """
    Pick the rule that did best on settled history, then use it once, blind.

    Returns one row per inception that had enough history: the rule chosen, and
    what that rule actually went on to earn. The choice at each date sees only
    outcomes that had settled, so nothing here is fitted to the P&L it reports.
    """
    table = sub.pivot_table(index="date", columns="policy", values="net_bps")
    keep = [r for r in rules if r in table.columns]
    if not keep:
        return pd.DataFrame(columns=["date", "chosen", "net_bps"])
    table = table[keep].sort_index()

    rows = []
    for as_of in table.index:
        history = table.loc[settled_history(table.index, as_of, tenor_days)]
        history = history.dropna(how="all")
        if len(history) < warmup:
            continue
        chosen = history.mean().idxmax()
        realised = table.loc[as_of, chosen]
        if pd.notna(realised):
            rows.append({"date": as_of, "chosen": chosen, "net_bps": float(realised)})
    return pd.DataFrame(rows)


def _walk_forward_block(sub, tenor_days: int, label: str, warmup: int) -> None:
    """One (asset, tenor, strike-set) block: every selection scheme, same dates."""
    families = {
        "all 16 rules": [lab for _, _, lab in policies()],
        "clocks only": [lab for k, _, lab in policies() if k == "time"],
        "bands only": [lab for k, _, lab in policies() if k != "time"],
        "ww only": [lab for k, _, lab in policies() if k == "whalley_wilmott"],
    }
    chosen_sets = {name: walk_forward(sub, rules, tenor_days, warmup)
                   for name, rules in families.items()}
    dated = [frame for frame in chosen_sets.values() if not frame.empty]
    if not dated:
        return
    # Every scheme, and every fixed comparator, is judged on exactly the dates
    # the walk-forward could reach. Scoring a hindsight rule over a longer
    # sample than the one that had to warm up would flatter it for free.
    common = sorted(set.intersection(*[set(f["date"]) for f in dated]))
    if not common:
        return
    table = sub.pivot_table(index="date", columns="policy", values="net_bps")
    table = table.loc[[d for d in table.index if d in set(common)]]

    print(f"\n  {label}  ({len(common)} inceptions out of sample, warmup {warmup})")
    print(f"    {'scheme':<16}{'net bps':>9}{'t':>7}   {'chosen most often':<28}")
    for name, frame in chosen_sets.items():
        if frame.empty:
            continue
        on_common = frame[frame["date"].isin(common)]
        counts = on_common["chosen"].value_counts()
        top = f"{counts.index[0]} ({counts.iloc[0]}/{len(on_common)})" if len(counts) else "-"
        print(f"    {name:<16}{on_common['net_bps'].mean():>+9.0f}"
              f"{_t(on_common['net_bps']):>7.1f}   {top:<28}")

    hindsight = table.mean().idxmax()
    print(f"    {'-- hindsight':<16}{table[hindsight].mean():>+9.0f}"
          f"{_t(table[hindsight]):>7.1f}   {hindsight + ' (upper bound)':<28}")
    for fixed in ("time 4h", "time 6h"):
        if fixed in table.columns:
            print(f"    {'-- fixed ' + fixed:<16}{table[fixed].mean():>+9.0f}"
                  f"{_t(table[fixed]):>7.1f}   {'chosen a priori':<28}")


def walk_forward_report(frame, warmup: int = 6) -> None:
    ok = frame[(frame["skipped"] == "") & frame["supported"].fillna(False)]
    print(f"\n{'=' * 78}")
    print("CHOOSING THE RULE WITHOUT HINDSIGHT")
    print(f"{'=' * 78}")
    print("At each inception the rule is whichever did best on history that had already")
    print("settled; it is then used once, blind. Compared against the rule that wins in")
    print("hindsight (§12.1's number, an upper bound) and against a clock fixed a priori.")

    for slip in sorted(ok["slip_bps"].unique()):
        if slip == 0:
            continue                      # nothing to economise on; rules coincide
        print(f"\n{'-' * 78}\n{slip:.0f} bps hedge slippage")
        for asset in ASSETS:
            for days in TENOR_DAYS:
                base = ok[(ok["asset"] == asset) & (ok["tenor"] == days)
                          & (ok["slip_bps"] == slip)]
                if base.empty:
                    continue
                atm = base[base["z"] == 0.0]
                if not atm.empty:
                    _walk_forward_block(atm, days, f"{asset} {days}-day ATM", warmup)

    print(f"\n{'=' * 78}")
    print("A scheme beats the a-priori clock only if selection paid for itself. The")
    print("hindsight row is not achievable -- it is there to show how much of §12.1's")
    print("level was the rule and how much was knowing which rule in advance.")
    _warmup_sensitivity(ok)


def _warmup_sensitivity(ok, warmups=(3, 6, 12, 18)) -> None:
    """
    Does selection pay, and does that answer survive the warmup?

    The warmup is an arbitrary choice, so a single value is a single draw. This
    prints the gap between within-family selection and the best clock fixed a
    priori, across warmups: a gap that changes sign is selection noise wearing
    a result's clothes, and one that holds is worth something.
    """
    print(f"\n{'=' * 78}")
    print("ROBUSTNESS: ww-only walk-forward minus the better a-priori clock (bps)")
    print(f"{'=' * 78}")
    clocks = [lab for k, _, lab in policies() if k == "time"]
    ww = [lab for k, _, lab in policies() if k == "whalley_wilmott"]
    print(f"  {'block':<18}{'slip':>5}" + "".join(f"{'w=' + str(w):>8}" for w in warmups)
          + f"{'verdict':>12}")
    for asset in ASSETS:
        for days in TENOR_DAYS:
            for slip in sorted(ok["slip_bps"].unique()):
                if slip == 0:
                    continue
                sub = ok[(ok["asset"] == asset) & (ok["tenor"] == days)
                         & (ok["slip_bps"] == slip) & (ok["z"] == 0.0)]
                if sub.empty:
                    continue
                table = sub.pivot_table(index="date", columns="policy", values="net_bps")
                gaps = []
                for warmup in warmups:
                    picked = walk_forward(sub, ww, days, warmup)
                    if picked.empty:
                        gaps.append(float("nan"))
                        continue
                    on = table.loc[[d for d in table.index if d in set(picked["date"])]]
                    best_clock = max(on[c].mean() for c in clocks if c in on.columns)
                    gaps.append(picked["net_bps"].mean() - best_clock)
                finite = [g for g in gaps if g == g]
                verdict = ("--" if not finite
                           else "holds" if all(g > 0 for g in finite)
                           else "never" if all(g <= 0 for g in finite) else "flips")
                print(f"  {asset + ' ' + str(days) + 'd ATM':<18}{slip:>5.0f}"
                      + "".join(f"{g:>+8.0f}" for g in gaps) + f"{verdict:>12}")
    print("\n'flips' means the sign depends on the warmup, which is not a result.")


# ---- cli ----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="hedging frequency vs cost")
    parser.add_argument("stage", choices=("fit", "sweep", "report", "run", "walkforward"))
    parser.add_argument("--shard", help="i/n -- this task's slice of the work")
    parser.add_argument("--asset", choices=ASSETS, help="fit stage: one asset only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=6,
                        help="walkforward: settled inceptions required before choosing")
    args = parser.parse_args()

    if args.stage == "walkforward":
        walk_forward_report(load_cells(), args.warmup)
        return 0

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
