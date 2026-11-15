"""
Tests for `scripts/live_studies.py` (LIVE_STUDIES.md).

The things that would fake a result: a fixed-strike change that reads the
wrong point of the smile, a funding series that sees the day it is matched
to, and a forward window that leaks into the past.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("live_studies", ROOT / "scripts" / "live_studies.py")
ls = importlib.util.module_from_spec(spec)
sys.modules["live_studies"] = ls
spec.loader.exec_module(ls)

Z = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


class TestFixedStrike:
    def test_no_spot_move_means_fixed_strike_equals_fixed_moneyness(self):
        row = pd.Series([50, 48, 46, 45, 46, 47, 48.0], index=Z)
        assert ls.fixed_strike_change(row, 45.0, 0.0, 30 / 365.25, 44.0) == pytest.approx(1.0)

    def test_sticky_strike_smile_gives_zero_fixed_strike_change(self):
        # A smile fixed in strike: sigma(K) = 45 - 20 * ln(K / K0). After spot
        # rises by r, the old ATM strike K0 sits at k' = -r, and the t+1 grid
        # is that same strike curve re-read around the new spot.
        T, r, atm = 30 / 365.25, 0.02, 45.0
        sd = atm / 100 * np.sqrt(T)
        z = np.array(Z)
        k_new = z * sd                       # moneyness relative to the new spot
        lnK_over_K0 = k_new + r              # K / K0 = (S1 / S0) e^k
        row = pd.Series(45 - 20 * lnK_over_K0, index=Z)
        change = ls.fixed_strike_change(row, atm, r, T, 45.0)
        # Linear in k, so linear interpolation in z is exact.
        assert change == pytest.approx(0.0, abs=1e-9)

    def test_off_the_grid_is_nan(self):
        row = pd.Series([50, 48, 46, 45, 46, 47, 48.0], index=Z)
        assert np.isnan(ls.fixed_strike_change(row, 45.0, 0.5, 7 / 365.25, 45.0))

    def test_moneyness_share_ends(self):
        assert ls.moneyness_share(0.0, -0.3) == 0.0     # sticky strike
        assert ls.moneyness_share(0.3, 0.0) == 1.0      # sticky moneyness


class TestFunding:
    def test_trailing_funding_excludes_the_day_itself(self):
        daily = pd.Series(np.arange(10.0), index=pd.date_range("2026-11-01", periods=10))
        f = ls.lagged_trailing(daily, 3)
        # On 11-05 (value 4): mean of 11-02..11-04 = (1+2+3)/3.
        assert f.loc["2026-11-05"] == pytest.approx(2.0)
        assert f.iloc[:3].isna().all()

    def test_slots_read_their_own_utc_date(self):
        daily = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-11-01", "2026-11-02"]))
        slots = pd.DatetimeIndex(["2026-11-01 20:00", "2026-11-02 08:00"], tz="UTC")
        assert ls.at_slots(daily, slots).tolist() == [1.0, 2.0]


class TestForward:
    def idx(self, returns):
        t = pd.date_range("2026-10-01", periods=len(returns) + 1, freq="h", tz="UTC")
        return pd.Series(100 * np.exp(np.r_[0, np.cumsum(returns)]), index=t)

    def test_downside_share_of_symmetric_and_one_sided_paths(self):
        n = 8 * 24
        sym = self.idx(np.where(np.arange(n) % 2, 0.01, -0.01))
        assert ls.downside_share(sym, sym.index[0]) == pytest.approx(0.5, abs=0.01)
        down = self.idx(np.full(n, -0.001))
        assert ls.downside_share(down, down.index[0]) == pytest.approx(1.0)

    def test_downside_share_ignores_the_past(self):
        n = 16 * 24
        r = np.r_[np.full(8 * 24, -0.01), np.where(np.arange(8 * 24) % 2, 0.01, -0.01)]
        path = self.idx(r)
        t0 = path.index[8 * 24]
        assert ls.downside_share(path, t0) == pytest.approx(0.5, abs=0.01)

    def test_next_return_is_twelve_hours_on_and_nan_across_gaps(self):
        t = pd.DatetimeIndex(["2026-11-01 08:00", "2026-11-01 20:00", "2026-11-03 08:00"], tz="UTC")
        spot = pd.Series([100.0, 110.0, 120.0], index=t)
        r = ls.next_return(spot)
        assert r.iloc[0] == pytest.approx(np.log(1.1))
        assert r.iloc[1:].isna().all()
