"""
Tests for the data layer.

The cases that matter here are the ones that fail silently in production:
the BTC/USD conversion, the exact-UTC time to expiry, and the spread filter's
interaction with OTM selection.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from data.clean import (
    MIN_MID_BTC,
    clean_snapshot,
    compute_mid,
    select_otm,
    tick_size,
    time_to_expiry_years,
)
from data.fetch_deribit import parse_instrument_name

UTC = dt.timezone.utc


def make_row(**overrides) -> dict:
    row = {
        "timestamp": dt.datetime(2026, 9, 7, 8, 0, tzinfo=UTC),
        "instrument_name": "BTC-8SEP26-80000-C",
        "strike": 80000.0,
        "expiry_ts": dt.datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
        "option_type": "C",
        "bid": 0.0100,
        "ask": 0.0105,
        "mark_price": 0.01025,
        "mark_iv": 40.0,
        "underlying_price": 80000.0,
        "underlying_index": "BTC-8SEP26",
        "index_price": 79900.0,
        "open_interest": 10.0,
        "volume": 1.0,
        "interest_rate": 0.0,
    }
    row.update(overrides)
    return row


def make_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestParseInstrumentName:
    def test_parses_strike_expiry_and_type(self):
        parsed = parse_instrument_name("BTC-27DEC26-100000-C")
        assert parsed["strike"] == 100000.0
        assert parsed["option_type"] == "C"
        assert parsed["expiry"] == dt.datetime(2026, 12, 27, 8, 0, tzinfo=UTC)

    def test_single_digit_day_has_no_leading_zero(self):
        parsed = parse_instrument_name("BTC-6SEP26-80000-P")
        assert parsed["expiry"] == dt.datetime(2026, 9, 6, 8, 0, tzinfo=UTC)

    def test_expiry_is_0800_utc_not_midnight(self):
        # A few hours matters a great deal for near-dated time to expiry.
        assert parse_instrument_name("BTC-6SEP26-80000-P")["expiry"].hour == 8


class TestTimeToExpiry:
    def test_exact_hours_not_calendar_days(self):
        # 08:00 to 08:00 next day is exactly one day, not two calendar dates.
        years = time_to_expiry_years(
            dt.datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
            dt.datetime(2026, 9, 7, 8, 0, tzinfo=UTC),
        )
        assert years == pytest.approx(1 / 365.25, rel=1e-9)

    def test_partial_day_is_not_rounded_up(self):
        # Six hours before expiry is a quarter day, not one day.
        years = time_to_expiry_years(
            dt.datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
            dt.datetime(2026, 9, 8, 2, 0, tzinfo=UTC),
        )
        assert years == pytest.approx(0.25 / 365.25, rel=1e-9)

    def test_naive_timestamps_are_treated_as_utc(self):
        aware = time_to_expiry_years(
            dt.datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
            dt.datetime(2026, 9, 7, 8, 0, tzinfo=UTC),
        )
        naive = time_to_expiry_years(
            dt.datetime(2026, 9, 8, 8, 0), dt.datetime(2026, 9, 7, 8, 0)
        )
        assert aware == pytest.approx(naive)


class TestQuoteFilters:
    def test_zero_bid_is_dropped(self):
        out = compute_mid(make_frame([make_row(bid=0.0, ask=0.005)]))
        assert out.empty

    def test_missing_side_is_dropped(self):
        out = compute_mid(make_frame([make_row(bid=np.nan)]))
        assert out.empty

    def test_near_worthless_quote_is_dropped(self):
        out = compute_mid(
            make_frame([make_row(bid=MIN_MID_BTC / 4, ask=MIN_MID_BTC / 2)])
        )
        assert out.empty

    def test_wide_relative_spread_is_dropped(self):
        # 0.02 / 0.03 is a 40% spread and many ticks wide: genuinely illiquid.
        out = compute_mid(make_frame([make_row(bid=0.020, ask=0.030)]))
        assert out.empty

    def test_tight_market_survives_a_wide_relative_spread(self):
        # 0.0006 / 0.0008 is a 29% relative spread but only two ticks wide
        # (the tick is 0.0001 below a 0.005 premium). Rejecting it measures
        # premium size rather than liquidity -- the defect in the guide's
        # bare relative-spread rule.
        out = compute_mid(make_frame([make_row(bid=0.0006, ask=0.0008)]))
        assert len(out) == 1
        assert out["spread_pct"].iloc[0] > 0.15
        assert out["spread_ticks"].iloc[0] == pytest.approx(2.0)

    def test_many_ticks_wide_is_still_dropped(self):
        # Same cheap premium, but five ticks wide: the tick allowance is a
        # floor, not a licence to keep everything cheap.
        out = compute_mid(make_frame([make_row(bid=0.0006, ask=0.0011)]))
        assert out.empty

    def test_tick_size_steps_at_the_documented_boundary(self):
        assert tick_size(0.010) == 0.0005
        assert tick_size(0.001) == 0.0001


class TestUsdConversion:
    def test_premium_is_converted_with_the_expiry_forward(self):
        # The single most consequential line in the data layer: strikes are
        # USD, premiums are BTC, and the correct multiplier is the forward.
        out = clean_snapshot(
            make_frame([make_row(bid=0.0100, ask=0.0105)]), otm_only=False
        )
        assert out["mid_usd"].iloc[0] == pytest.approx(0.01025 * 80000.0)

    def test_deep_itm_put_premium_recovers_intrinsic_value(self):
        # A put struck far above the forward is almost all intrinsic, so its
        # USD premium must land near K - F. This is the check that catches a
        # conversion done with the index instead of the forward.
        forward, strike = 80000.0, 125000.0
        intrinsic_btc = (strike - forward) / forward
        row = make_row(
            instrument_name="BTC-25SEP26-125000-P",
            strike=strike,
            option_type="P",
            bid=intrinsic_btc - 0.0005,
            ask=intrinsic_btc + 0.0005,
            underlying_price=forward,
            index_price=70000.0,  # deliberately wrong multiplier
        )
        out = clean_snapshot(make_frame([row]), otm_only=False)
        assert out["mid_usd"].iloc[0] == pytest.approx(strike - forward, rel=1e-6)


class TestMoneynessAndSelection:
    def test_log_moneyness_is_zero_at_the_forward(self):
        out = clean_snapshot(make_frame([make_row()]), otm_only=False)
        assert out["log_moneyness"].iloc[0] == pytest.approx(0.0)

    def test_select_otm_keeps_one_side_per_strike(self):
        rows = make_frame(
            [
                make_row(strike=70000.0, option_type="P", log_moneyness=-0.1),
                make_row(strike=70000.0, option_type="C", log_moneyness=-0.1),
                make_row(strike=90000.0, option_type="C", log_moneyness=0.1),
                make_row(strike=90000.0, option_type="P", log_moneyness=0.1),
            ]
        )
        rows["log_moneyness"] = np.log(rows["strike"] / rows["underlying_price"])
        kept = select_otm(rows)
        assert len(kept) == 2
        assert set(zip(kept["strike"], kept["option_type"])) == {
            (70000.0, "P"),
            (90000.0, "C"),
        }

    def test_expired_contracts_are_dropped(self):
        row = make_row(expiry_ts=dt.datetime(2026, 9, 6, 8, 0, tzinfo=UTC))
        assert clean_snapshot(make_frame([row]), otm_only=False).empty
