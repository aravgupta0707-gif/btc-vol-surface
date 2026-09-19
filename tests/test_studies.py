"""
Tests for the pure pieces the research studies rest on.

Each study's conclusion depends on one or two small calculations -- a carry
measured from a forward curve, an annualised funding rate, a delta-hedged
P&L -- and each of those had, or could easily have had, a quiet error that
would change the conclusion rather than crash.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.history import missing_hours

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


carry = _script("carry_study")
vrp = _script("vrp_surface")

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")


def chain(rate: float, spot: float = 60_000.0) -> pd.DataFrame:
    """A snapshot whose forwards compound at `rate` from spot."""
    days = [0.33, 1.33, 7.33, 14.33, 28.33, 91.33, 182.33]
    return pd.DataFrame({
        "timestamp": T0,
        "expiry_ts": [T0 + pd.Timedelta(days=d) for d in days],
        "underlying_price": [spot * np.exp(rate * d / 365.25) for d in days],
    })


class TestImpliedCarry:
    @pytest.mark.parametrize("tenor", [7, 30, 90])
    def test_recovers_a_known_carry(self, tenor):
        assert carry.implied_carry(chain(0.08), tenor) == pytest.approx(8.0, abs=1e-6)

    def test_does_not_depend_on_where_spot_was_at_another_moment(self):
        # The regression: measuring against the 00:00 index turned an hour of
        # quote staleness into ~23 points of annualised 7-day carry. Scaling
        # every forward by a common move -- all quoted a little later -- must
        # leave the carry unchanged.
        moved = chain(0.08).assign(underlying_price=lambda f: f.underlying_price * 1.005)
        assert carry.implied_carry(moved, 7) == pytest.approx(8.0, abs=1e-6)

    def test_refuses_to_extrapolate_beyond_the_curve(self):
        assert np.isnan(carry.implied_carry(chain(0.08), 400))


class TestFundingRate:
    def test_annualises_an_hourly_rate(self):
        hours = pd.date_range(T0, periods=24 * 30, freq="h")
        funding = pd.Series(1e-5, index=hours)
        rate = carry.funding_rate(funding, T0, T0 + pd.Timedelta(days=30))
        assert rate == pytest.approx(1e-5 * 365.25 * 24 * 100)

    def test_a_window_with_holes_is_refused(self):
        hours = pd.date_range(T0, periods=24 * 30, freq="h")
        funding = pd.Series(1e-5, index=hours).iloc[::2]
        assert np.isnan(carry.funding_rate(funding, T0, T0 + pd.Timedelta(days=30)))


class TestHistory:
    def test_missing_hours_counts_holes(self):
        hours = pd.date_range(T0, periods=100, freq="h")
        frame = pd.DataFrame({"time": hours.delete([10, 11, 50])})
        assert missing_hours(frame) == 3


class TestHedgedSellerPnl:
    def test_a_path_that_never_moves_keeps_the_whole_premium(self):
        hours = np.arange(0, 30 * 24 + 1, 4, dtype=float)
        path = np.full(len(hours), 60_000.0)
        pnl, premium = vrp.hedged_seller_pnl(path, hours, 60_000.0, 30 / 365.25, 0.5, 0.0, "C")
        assert pnl == pytest.approx(premium, rel=1e-9)

    def test_an_otm_put_that_expires_worthless_after_a_rally(self):
        hours = np.arange(0, 7 * 24 + 1, 4, dtype=float)
        path = np.linspace(60_000.0, 66_000.0, len(hours))
        pnl, premium = vrp.hedged_seller_pnl(path, hours, 50_000.0, 7 / 365.25, 0.5, 0.0, "P")
        # Expired worthless: the seller keeps the premium less the hedge's
        # losses on a short-delta hedge through a rally -- positive and below it.
        assert 0 < pnl <= premium
