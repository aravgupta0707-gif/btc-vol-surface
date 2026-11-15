"""
Tests for the inverse-vs-linear comparison (`scripts/inverse_linear.py`).

These check the things that would fake a result. Two books priced off the
same vol must show zero difference, so neither conversion can leak in. The
dollar floor must treat both books alike. Snapshots too far apart in time
must not be paired. And the stress measure must not read across a sleep gap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing.black76 import price

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("inverse_linear", ROOT / "scripts" / "inverse_linear.py")
il = importlib.util.module_from_spec(spec)
sys.modules["inverse_linear"] = il
spec.loader.exec_module(il)

NOW = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-12-01 08:00:00", tz="UTC")
F = 80_000.0


def raw_book(linear: bool, sigma: float = 0.5, half_spread_vol: float = 0.005,
             strikes=(60_000, 70_000, 90_000, 100_000)) -> pd.DataFrame:
    """A raw snapshot priced off a flat vol, quoted the way that book quotes."""
    T = (EXPIRY - NOW).total_seconds() / (365.25 * 86400)
    rows = []
    for K in strikes:
        cp = "P" if K < F else "C"
        bid, mid, ask = (float(price(F, K, T, s, cp)) for s in
                         (sigma - half_spread_vol, sigma, sigma + half_spread_vol))
        scale = 1.0 if linear else 1.0 / F
        rows.append({
            "timestamp": NOW, "instrument_name": f"{'BTC_USDC' if linear else 'BTC'}-1DEC26-{K}-{cp}",
            "underlying_asset": "BTC", "quote_currency": "USDC" if linear else "BTC",
            "strike": float(K), "expiry_ts": EXPIRY, "option_type": cp,
            "bid": bid * scale, "ask": ask * scale, "mark_price": mid * scale,
            "underlying_price": F, "index_price": F,
        })
    return pd.DataFrame(rows)


def matched(lin_sigma: float = 0.5, inv_sigma: float = 0.5, **kw) -> pd.DataFrame:
    inv = il.prepare(raw_book(False, inv_sigma, **kw), NOW)
    lin = il.prepare(raw_book(True, lin_sigma, **kw), NOW)
    return il.add_vols(il.match_books(inv, lin))


class TestConversion:
    def test_same_vol_in_both_books_is_zero_difference(self):
        m = matched()
        assert len(m) == 4
        assert m["diff"].abs().max() < 1e-5
        # The inverter stops at a 1e-6 price tolerance, which is up to ~0.01
        # vol points on an absolute vol; the difference between books cancels.
        assert m["iv_mid_inv"].to_numpy() == pytest.approx(0.5, abs=2e-4)

    def test_a_known_vol_gap_is_recovered_with_its_sign(self):
        m = matched(lin_sigma=0.52)
        assert m["diff"].to_numpy() == pytest.approx(0.02, abs=5e-5)

    def test_vol_width_is_measured_and_gates_the_clean_set(self):
        m = matched(half_spread_vol=0.005)
        assert m["width_lin"].to_numpy() == pytest.approx(0.01, abs=1e-5)
        assert m["clean"].all()
        assert not matched(half_spread_vol=0.02)["clean"].any()

    def test_books_crossing_is_detected_only_when_bid_beats_ask(self):
        assert not matched()["crossed"].any()
        # Linear 3 vol points over inverse with half-spreads of 0.5: its bid
        # sits 2 points above the inverse ask.
        assert matched(lin_sigma=0.53)["crossed"].all()

    def test_dollar_floor_applies_to_both_books_alike(self):
        # A far-OTM put worth well under 0.0005 BTC: the cleaner's own floor
        # would drop it from the inverse book but keep it in the linear one.
        inv = il.prepare(raw_book(False, strikes=(20_000,)), NOW)
        lin = il.prepare(raw_book(True, strikes=(20_000,)), NOW)
        assert inv.empty and lin.empty


class TestPairing:
    def test_pairs_only_within_the_window_and_to_the_nearest(self):
        inv = ["btc_chain_20261101T000000Z.parquet", "btc_chain_20261101T010000Z.parquet"]
        lin = ["usdc_chain_20261101T000003Z.parquet", "usdc_chain_20261101T000100Z.parquet",
               "usdc_chain_20261101T013000Z.parquet"]
        pairs = il.pair_snapshots(inv, lin, max_seconds=120)
        assert pairs == [(inv[0], lin[0])]


class TestTrailingRv:
    def test_constant_hourly_moves_give_the_matching_annual_vol(self):
        n = 48
        times = pd.Series(pd.date_range(NOW, periods=n, freq="h"))
        r = 0.01
        prices = pd.Series(100 * np.exp(r * np.arange(n) * np.where(np.arange(n) % 2, 1, 1)))
        rv = il.trailing_rv(times, prices, window_hours=24, min_obs=12)
        assert rv.iloc[-1] == pytest.approx(r * np.sqrt(24 * 365.25), rel=1e-9)
        assert rv.iloc[:12].isna().all()

    def test_a_return_across_a_sleep_gap_is_not_counted(self):
        times = pd.Series(list(pd.date_range(NOW, periods=20, freq="h"))
                          + list(pd.date_range(NOW + pd.Timedelta(days=3), periods=20, freq="h")))
        prices = pd.Series(np.r_[np.full(20, 100.0), np.full(20, 150.0)])
        rv = il.trailing_rv(times, prices, window_hours=24, min_obs=12)
        # Flat on both sides of the gap; the 40% jump across it is excluded.
        assert rv.dropna().max() == pytest.approx(0.0)
