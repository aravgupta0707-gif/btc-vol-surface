"""
Tests for the dispersion book (`scripts/dispersion.py`).

Two things could quietly change its conclusion: the long leg built with the
wrong sign or without paying its costs, and an entry rule that sees the
month it is deciding. Both are checked here on synthetic cells.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ds = _script("dispersion")


def cells(n: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    rows = []
    for i, d in enumerate(dates):
        for asset, iv, gross in (("BTC", 0.50, 100.0 + i), ("ETH", 0.60 + 0.01 * (i % 3), 30.0 - i)):
            rows.append({"asset": asset, "date": d, "tenor": 30, "z": 0.0, "policy": "time 4h",
                         "slip_bps": 5.0, "gross_bps": gross, "spread_bps": 10.0 if asset == "BTC" else 20.0,
                         "slippage_bps": 30.0, "net_bps": gross - (10.0 if asset == "BTC" else 20.0) - 30.0,
                         "atm_iv": iv, "premium_bps": 1000.0})
    return pd.DataFrame(rows)


class TestLegs:
    def test_short_btc_is_the_cell_and_long_eth_is_its_mirror_paying_costs(self):
        b = ds.legs(cells(), 30, 0.0, "time 4h", 5.0)
        first = b.iloc[0]
        assert first["btc_short_net"] == pytest.approx(100.0 - 10 - 30)
        assert first["eth_long_gross"] == pytest.approx(-30.0)
        assert first["eth_long_net"] == pytest.approx(-30.0 - 20 - 30)
        assert first["book_net"] == pytest.approx(first["btc_short_net"] + first["eth_long_net"])
        # Book gross less every cost on both legs is book net.
        assert (b["book_gross"] - b["costs"]).to_numpy() == pytest.approx(b["book_net"].to_numpy())

    def test_spread_is_eth_minus_btc_in_vol_points(self):
        b = ds.legs(cells(), 30, 0.0, "time 4h", 5.0)
        assert b["spread"].iloc[0] == pytest.approx(10.0)

    def test_equal_notional_is_vega_neutral_to_under_one_percent(self):
        b = ds.legs(cells(), 30, 0.0, "time 4h", 5.0)
        assert (abs(b["vega_ratio"] - 1) < 0.01).all()

    def test_months_missing_an_asset_are_dropped(self):
        c = cells()
        c = c[~((c["asset"] == "ETH") & (c["date"] == c["date"].min()))]
        assert len(ds.legs(c, 30, 0.0, "time 4h", 5.0)) == 11


class TestEntrySignal:
    def test_undecided_until_enough_history(self):
        s = ds.entry_signal(pd.Series(np.arange(10.0)), min_history=6)
        assert s.iloc[:6].isna().all()
        assert s.iloc[6:].notna().all()

    def test_today_is_not_in_todays_threshold(self):
        # A rising series is always above the median of what came before, so
        # the rule never fires. If today leaked into its own median, the last
        # point could never be strictly below it either -- so also check a
        # point that is below only the *earlier* median.
        rising = ds.entry_signal(pd.Series(np.arange(10.0)), min_history=3)
        assert not rising.dropna().astype(bool).any()
        s = pd.Series([5, 5, 5, 5, 4.9])
        assert ds.entry_signal(s, min_history=3).iloc[-1] == True  # noqa: E712

    def test_changing_the_future_does_not_change_the_past(self):
        base = pd.Series([3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        bumped = base.copy()
        bumped.iloc[-1] = 100.0
        a = ds.entry_signal(base, min_history=3).iloc[:-1]
        b = ds.entry_signal(bumped, min_history=3).iloc[:-1]
        pd.testing.assert_series_equal(a, b)
