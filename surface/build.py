"""
Snapshot -> fitted SVI slices, in one place.

Both the Phase 3 fit report and the Phase 4 arbitrage checks need the same
pipeline: clean the chain, invert to implied vol, convert to total variance,
and fit a slice per expiry. Phase 5's `VolSurface` will consume the same
output, so it lives here rather than being duplicated in each script.

The quoted k-range of each slice is carried alongside the parameters. It is
what lets the arbitrage checks distinguish a violation among strikes that
actually trade from one out in the extrapolated wings, which are very
different findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.clean import clean_snapshot
from pricing.black76 import implied_vol_many
from surface.svi import fit_svi_slice, svi_total_variance


def invert_chain(raw: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
    """Clean a raw snapshot and add `iv` and `w` (total variance) columns."""
    if asset is not None and "underlying_asset" in raw.columns:
        raw = raw[raw["underlying_asset"] == asset]
    clean = clean_snapshot(raw)
    clean = clean.assign(
        iv=implied_vol_many(
            clean["forward"], clean["strike"], clean["T"],
            clean["mid_usd"], clean["option_type"].tolist(),
        )
    ).dropna(subset=["iv"])
    clean["w"] = clean["iv"] ** 2 * clean["T"]
    return clean


def build_slices(quotes: pd.DataFrame) -> tuple[dict, dict, list]:
    """
    Fit one SVI slice per expiry.

    Returns (params_by_T, quoted_ranges_by_T, refusals) where params_by_T is
    keyed on time to expiry in years -- the natural key for the calendar
    check and for interpolation, both of which work in T rather than in
    calendar dates.
    """
    params, ranges, refused = {}, {}, []
    for expiry, group in quotes.groupby("expiry_ts"):
        T = float(group["T"].iloc[0])
        k = group["log_moneyness"].to_numpy()
        fit = fit_svi_slice(k, group["w"].to_numpy())
        if not fit["ok"]:
            refused.append({"expiry": expiry, "T": T, **fit})
            continue

        fitted_w = svi_total_variance(
            k, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"]
        )
        fitted_iv = np.sqrt(np.maximum(fitted_w, 0.0) / T)
        observed_iv = group["iv"].to_numpy()
        fit["rmse_vol"] = float(
            np.sqrt(np.mean((fitted_iv - observed_iv) ** 2)) * 100
        )
        fit["max_vol_err"] = float(np.max(np.abs(fitted_iv - observed_iv)) * 100)
        w_atm = float(svi_total_variance(
            0.0, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"]
        ))
        fit["atm_vol"] = float(np.sqrt(max(w_atm, 0.0) / T)) * 100
        fit["expiry"] = expiry
        fit["T"] = T

        params[T] = fit
        ranges[T] = (float(k.min()), float(k.max()))
    return params, ranges, refused


def build_from_snapshot(
    raw: pd.DataFrame, asset: str | None = None
) -> tuple[dict, dict, list, pd.DataFrame]:
    """Full pipeline: raw snapshot in, fitted slices out."""
    quotes = invert_chain(raw, asset)
    params, ranges, refused = build_slices(quotes)
    return params, ranges, refused, quotes
