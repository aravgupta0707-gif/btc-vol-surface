"""
Unit tests for the study code that had none: replicated variance swaps
(`scripts/varswap.py`, DISPERSION.md Part 2), the realised-vol helpers in
`cross_asset.py` and `vrp_study.py`, and H7's forward semivariance.

Each test pins a number to a closed form. A flat-vol surface has a known
variance strike (sigma^2) and strip vega (2 sigma). A path with constant
log returns has a known realised vol. A symmetric path splits its variance
evenly between up and down.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surface.surface import VolSurface

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vs = _script("varswap")
ca = _script("cross_asset")
vst = _script("vrp_study")
vh = _script("vrp_hypotheses")

F, T, SIGMA = 80_000.0, 30 / 365.25, 0.5


def flat(sigma: float = SIGMA, k_range=(-3.0, 3.0)) -> VolSurface:
    params = {"a": sigma**2 * T, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    return VolSurface({T: params}, {T: F}, {T: k_range})


def hourly(n_hours: int, step_return: float = 0.0, start="2024-01-01", alternate=False) -> pd.Series:
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    r = np.full(n_hours, step_return)
    if alternate:
        r = r * np.where(np.arange(n_hours) % 2 == 0, 1.0, -1.0)
    r[0] = 0.0
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)


class TestVarianceStrike:
    def test_flat_vol_strike_is_sigma_squared(self):
        out = vs.variance_strike(flat(), T)
        assert out["k_var_full"] == pytest.approx(SIGMA**2, rel=2e-3)
        assert out["atm_vol"] == pytest.approx(SIGMA, rel=1e-9)

    def test_truncating_the_strip_loses_variance(self):
        narrow = vs.variance_strike(flat(k_range=(-0.2, 0.2)), T)
        assert narrow["k_var"] < narrow["k_var_full"]
        assert narrow["lo_sd"] == pytest.approx(-0.2 / (SIGMA * np.sqrt(T)))

    def test_strip_cost_of_a_constant_half_spread_is_its_strip_vega(self):
        # d(K_var)/d(sigma) = 2 sigma on a flat surface, and the cost
        # integrand is half-spread x vega -- so cost = 2 sigma h.
        h = 0.01
        cost = vs.strip_cost(flat(), T, -3.0, 3.0, lambda k: np.full_like(k, h))
        assert cost == pytest.approx(2 * SIGMA * h, rel=5e-3)

    def test_half_spread_curve_uses_the_nearest_tenor_and_holds_its_ends(self):
        spreads = pd.DataFrame({"T": [0.05, 0.05, 0.2, 0.2], "k": [-0.1, 0.1, -0.1, 0.1],
                                "half_spread_vol": [0.01, 0.03, 0.5, 0.5]})
        curve = vs.half_spread_curve(spreads, 0.06)
        assert curve(0.0) == pytest.approx(0.02)
        assert curve(-5.0) == pytest.approx(0.01) and curve(5.0) == pytest.approx(0.03)
        assert vs.half_spread_curve(pd.DataFrame(), 0.06) is None


class TestPathMeasures:
    def test_constant_moves_give_their_realised_variance(self):
        r = 0.002
        path = hourly(31 * 24, r, alternate=True)
        out = vs.path_measures(path, path.index[0], days=30, step=4, slip_bps=5.0)
        # Alternating +/- r every hour nets to 0 over an even 4h step, so
        # sample hourly to see the moves.
        hourly_out = vs.path_measures(path, path.index[0], days=30, step=1, slip_bps=5.0)
        assert hourly_out["rv"] == pytest.approx(r**2 * vs.YEAR_HOURS, rel=1e-9)
        assert out["rv"] == pytest.approx(0.0, abs=1e-12)

    def test_flat_path_costs_nothing_to_hedge(self):
        path = hourly(31 * 24, 0.0)
        out = vs.path_measures(path, path.index[0], days=30, step=4, slip_bps=5.0)
        assert out["rv"] == 0.0 and out["hedge_cost"] == 0.0 and out["ret"] == 0.0

    def test_hedge_cost_scales_with_slippage(self):
        path = hourly(31 * 24, 0.001)
        a = vs.path_measures(path, path.index[0], days=30, step=4, slip_bps=5.0)["hedge_cost"]
        b = vs.path_measures(path, path.index[0], days=30, step=4, slip_bps=10.0)["hedge_cost"]
        assert a > 0 and b == pytest.approx(2 * a)

    def test_a_path_with_too_few_hours_is_not_measured(self):
        path = hourly(20 * 24, 0.001)
        assert np.isnan(vs.path_measures(path, path.index[0], days=30)["rv"])


class TestBook:
    def frame(self):
        rows = []
        for d, (pb, pe) in enumerate([(3.0, 1.0), (-1.0, -2.0)]):
            for asset, p, c in (("BTC", pb, 0.5), ("ETH", pe, 0.7)):
                rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * d), "asset": asset,
                             "skipped": "", "premium_vp": p, "costs_vp": c,
                             "short_net_vp": p - c, "long_net_vp": -p - c,
                             "k_vol": 0.5, "rv_vol": 0.45, "atm_vol": 0.5 if asset == "BTC" else 0.6})
        return pd.DataFrame(rows)

    def test_book_is_short_btc_long_eth_and_net_is_gross_less_both_costs(self):
        b = vs.book(self.frame())
        assert b["gross"].tolist() == pytest.approx([3.0 - 1.0, -1.0 + 2.0])
        assert (b["gross"] - b["costs"]).to_numpy() == pytest.approx(b["net"].to_numpy())
        assert b["atm_spread"].iloc[0] == pytest.approx(10.0)

    def test_vol_points_conversion(self):
        # A 1-point vol premium on a 50-vol strike is ~ variance 0.0100.
        assert vs.to_vol_points(0.0100, 0.25) == pytest.approx(1.0)
        assert vs.to_vol_points(0.51**2 - 0.50**2, 0.25) == pytest.approx(1.0, abs=0.02)


class TestRealised:
    def test_trailing_realised_matches_constant_moves_and_needs_coverage(self):
        r = 0.003
        path = hourly(40 * 24, r, alternate=True)
        t0 = path.index[-1]
        # Returns alternate +r / -r, so their sd is r (to within ddof=1).
        assert ca.trailing_realized(path, t0, 30) == pytest.approx(r * np.sqrt(ca.YEAR_HOURS), rel=1e-2)
        assert np.isnan(ca.trailing_realized(path.iloc[: 10 * 24], path.index[10 * 24 - 1], 30))

    def test_realised_corr_of_a_series_with_itself_and_its_mirror(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=40 * 24, freq="h")
        a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        mirror = pd.Series(100 * np.exp(-np.log(a / 100)), index=idx)
        assert ca.realized_corr(a, a, idx[0], 30) == pytest.approx(1.0)
        assert ca.realized_corr(a, mirror, idx[0], 30) == pytest.approx(-1.0)

    def test_forward_realised_uses_only_the_following_days(self):
        closes = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=80, freq="D"))
        closes.iloc[50:] = 100 * np.exp(0.02 * np.arange(1, 31))
        fwd = vst.forward_realized_vol(closes, horizon=30)
        # Day 10's window is days 11-40, all before the move at day 50, so
        # zero vol.
        assert fwd.iloc[10] == pytest.approx(0.0)
        # The last 30 days have no full forward window.
        assert fwd.iloc[-30:].isna().all()
        # Changing a close after day 40 cannot move day 10's number.
        bumped = closes.copy()
        bumped.iloc[60] *= 2
        assert vst.forward_realized_vol(bumped, 30).iloc[10] == fwd.iloc[10]


class TestForwardSemivariance:
    def test_symmetric_moves_split_evenly_and_each_side_equals_total_vol(self, tmp_path, monkeypatch):
        n = 70 * 24
        r = 0.004
        times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = 100 * np.exp(np.cumsum(np.where(np.arange(n) % 2, r, -r)))
        pd.DataFrame({"time": times, "close": close}).to_parquet(tmp_path / "btc_perp_hourly.parquet")
        monkeypatch.setattr(vh, "CACHE", tmp_path)
        out = vh.forward_semivariance("BTC")
        assert len(out) > 0
        assert out["up_share"].to_numpy() == pytest.approx(0.5, abs=0.01)
        total = r * np.sqrt(vh.HOURS_Y) * 100
        assert out["up_vol"].to_numpy() == pytest.approx(total, rel=0.02)
        assert out["down_vol"].to_numpy() == pytest.approx(total, rel=0.02)

    def test_a_rising_path_puts_all_variance_on_the_upside(self, tmp_path, monkeypatch):
        n = 70 * 24
        times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
        pd.DataFrame({"time": times, "close": close}).to_parquet(tmp_path / "btc_perp_hourly.parquet")
        monkeypatch.setattr(vh, "CACHE", tmp_path)
        out = vh.forward_semivariance("BTC")
        assert out["up_share"].to_numpy() == pytest.approx(1.0)
        assert (out["down_vol"] == 0).all()
        assert out["fwd_return"].to_numpy() == pytest.approx(0.001 * 30 * 24 * 100, rel=1e-6)


class TestSettlement:
    """`hold_to_expiry.settle`: carry the hedge to the index, flatten, pay intrinsic."""

    hte = _script("hold_to_expiry")

    def position(self, hedge_units: float):
        from strategy.straddle import Position
        expiry = pd.Timestamp("2026-11-13 08:00", tz="UTC")
        return Position(strike=80_000.0, expiry_ts=expiry, entry_ts=expiry - pd.Timedelta(days=1),
                        quantity=1.0, premium_received=2_000.0, entry_iv=0.5, entry_spot=80_000.0,
                        hedge_units=hedge_units, last_ts=expiry - pd.Timedelta(hours=8),
                        last_spot=80_000.0), expiry

    def strategy(self, slip_bps: float = 5.0, funding: float = 0.0):
        from strategy.straddle import Costs, DeltaHedgedStraddle
        return DeltaHedgedStraddle(costs=Costs(hedge_slippage_bps=slip_bps, funding_rate_8h=funding))

    def test_pays_intrinsic_and_carries_the_hedge_to_the_index(self):
        s = self.strategy()
        p, expiry = self.position(hedge_units=0.3)
        self.hte.settle(s, p, 82_000.0, expiry)
        assert p.option_value == pytest.approx(2_000.0)        # |82k - 80k| x 1
        assert p.hedge_pnl == pytest.approx(0.3 * 2_000.0)     # long 0.3 BTC, +2k
        assert p.hedge_units == 0.0 and p.closed
        assert p.slippage_paid == pytest.approx(0.3 * 82_000.0 * 5 / 1e4)
        # Premium 2000 - intrinsic 2000 + hedge 600 - slippage 12.3
        assert s.mark_to_market(p) == pytest.approx(600.0 - 12.3)

    def test_funding_accrues_on_the_hedge_for_the_hours_held(self):
        s = self.strategy(funding=0.0001)
        p, expiry = self.position(hedge_units=-0.5)
        self.hte.settle(s, p, 80_000.0, expiry)
        # Short 0.5 BTC for 8h at 1bp per 8h: receives funding (negative paid).
        assert p.funding_paid == pytest.approx(-0.5 * 80_000.0 * 0.0001)
        assert p.option_value == 0.0
