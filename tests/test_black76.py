"""
Tests for Black-76 pricing and the implied vol inversion.

The inversion is the part of the pipeline most likely to silently produce
garbage, so the cases that matter are the awkward ones: near-zero vega,
one-day expiries, deep wings, and prices that no volatility can produce.
A wrong number here is far worse than a None.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricing.black76 import (
    VOL_BRACKET,
    greeks,
    implied_vol,
    implied_vol_many,
    intrinsic,
    price,
    vega,
)

F0, T0 = 80_000.0, 30 / 365.25


class TestPrice:
    def test_call_and_put_are_positive(self):
        assert price(F0, F0, T0, 0.5, "C") > 0
        assert price(F0, F0, T0, 0.5, "P") > 0

    def test_atm_call_equals_atm_put(self):
        # At F == K the forward terms cancel, so call and put must agree.
        assert price(F0, F0, T0, 0.5, "C") == pytest.approx(
            price(F0, F0, T0, 0.5, "P")
        )

    def test_put_call_parity(self):
        # Undiscounted Black-76: C - P = F - K, for any strike and vol.
        for strike in (60_000.0, 80_000.0, 110_000.0):
            call = price(F0, strike, T0, 0.45, "C")
            put = price(F0, strike, T0, 0.45, "P")
            assert call - put == pytest.approx(F0 - strike, abs=1e-6)

    def test_price_rises_with_volatility(self):
        prices = [price(F0, F0, T0, s, "C") for s in (0.2, 0.4, 0.8, 1.6)]
        assert all(b > a for a, b in zip(prices, prices[1:]))

    def test_price_falls_with_strike_for_calls(self):
        prices = [price(F0, k, T0, 0.5, "C") for k in (60_000, 80_000, 100_000)]
        assert all(b < a for a, b in zip(prices, prices[1:]))

    def test_zero_time_gives_intrinsic(self):
        assert price(F0, 70_000.0, 0.0, 0.5, "C") == pytest.approx(10_000.0)
        assert price(F0, 90_000.0, 0.0, 0.5, "C") == pytest.approx(0.0)

    def test_zero_vol_gives_intrinsic(self):
        assert price(F0, 70_000.0, T0, 0.0, "C") == pytest.approx(10_000.0)

    def test_respects_no_arbitrage_bounds(self):
        call = price(F0, 90_000.0, T0, 0.6, "C")
        assert intrinsic(F0, 90_000.0, "C") <= call <= F0

    def test_vectorises_over_strikes(self):
        strikes = np.array([60_000.0, 80_000.0, 100_000.0])
        out = price(F0, strikes, T0, 0.5, "C")
        assert out.shape == (3,)
        assert out[0] > out[1] > out[2]

    def test_rejects_bad_option_type(self):
        with pytest.raises(ValueError):
            price(F0, F0, T0, 0.5, "X")


class TestVega:
    def test_positive_at_the_money(self):
        assert vega(F0, F0, T0, 0.5) > 0

    def test_collapses_in_the_far_wing(self):
        # The reason Newton needs a fallback: vega goes to nearly nothing and
        # the objective flattens out.
        assert vega(F0, 300_000.0, 1 / 365.25, 0.5) < 1e-6

    def test_zero_at_expiry(self):
        assert vega(F0, F0, 0.0, 0.5) == 0.0


class TestImpliedVolRoundTrip:
    @pytest.mark.parametrize("sigma", [0.15, 0.40, 0.75, 1.5, 3.0])
    @pytest.mark.parametrize("sd", [-2.5, -1.0, 0.0, 1.0, 2.5])
    @pytest.mark.parametrize("option_type", ["C", "P"])
    def test_price_then_invert_recovers_sigma(self, sigma, sd, option_type):
        # Strikes are placed in standard deviations of the move implied by
        # (sigma, T), not at fixed dollar levels. A strike 50% away is a
        # routine quote at 150% vol and worth 1e-19 dollars at 15% -- fixing
        # the strike in dollars would be testing the inverter on options that
        # do not exist rather than on the range it will actually meet.
        strike = F0 * np.exp(sd * sigma * np.sqrt(T0))
        target = price(F0, strike, T0, sigma, option_type)
        recovered = implied_vol(F0, strike, T0, target, option_type)
        assert recovered is not None
        assert recovered == pytest.approx(sigma, abs=1e-5)

    @pytest.mark.parametrize("days", [0.5, 1.0, 7.0, 90.0, 365.0])
    def test_round_trip_across_maturities(self, days):
        T = days / 365.25
        target = price(F0, 85_000.0, T, 0.6, "C")
        recovered = implied_vol(F0, 85_000.0, T, target, "C")
        assert recovered is not None
        assert recovered == pytest.approx(0.6, abs=1e-4)

    def test_deep_itm_inverts_through_parity(self):
        # A call struck far below the forward is ~all intrinsic. Inverting it
        # directly asks the solver to find vol in the last fraction of a
        # percent of the price; the parity switch to the OTM put avoids that.
        sigma = 0.55
        target = price(F0, 40_000.0, T0, sigma, "C")
        assert target > 39_000  # overwhelmingly intrinsic
        recovered = implied_vol(F0, 40_000.0, T0, target, "C")
        assert recovered is not None
        assert recovered == pytest.approx(sigma, abs=1e-4)

    def test_one_day_deep_otm_still_inverts(self):
        # Vega here is ~1e-3 of its ATM value, so Newton's step blows up and
        # this passes only through the Brent fallback.
        T = 1 / 365.25
        sigma = 0.9
        target = price(F0, 95_000.0, T, sigma, "C")
        assert 0.01 < target < 10.0  # small but a real, quotable price
        recovered = implied_vol(F0, 95_000.0, T, target, "C")
        assert recovered is not None
        assert recovered == pytest.approx(sigma, rel=1e-3)


class TestImpliedVolFailsHonestly:
    def test_price_above_upper_bound_returns_none(self):
        # No vol produces a call worth more than the forward.
        assert implied_vol(F0, 80_000.0, T0, F0 * 1.1, "C") is None

    def test_price_below_intrinsic_returns_none(self):
        assert implied_vol(F0, 40_000.0, T0, 10_000.0, "C") is None

    def test_price_exactly_intrinsic_returns_none(self):
        assert implied_vol(F0, 60_000.0, T0, 20_000.0, "C") is None

    def test_negative_price_returns_none(self):
        assert implied_vol(F0, 80_000.0, T0, -5.0, "C") is None

    def test_expired_option_returns_none(self):
        assert implied_vol(F0, 80_000.0, 0.0, 100.0, "C") is None

    def test_nan_input_returns_none(self):
        assert implied_vol(F0, 80_000.0, T0, np.nan, "C") is None

    def test_vol_above_the_bracket_returns_none(self):
        # 800% vol is outside [0.001, 5.0]; better to drop than to clamp
        # silently to the boundary and report 500%.
        target = price(F0, 80_000.0, T0, 8.0, "C")
        assert implied_vol(F0, 80_000.0, T0, target, "C") is None

    def test_worthless_option_returns_none_not_the_bracket_floor(self):
        # Regression guard. A 110k-strike one-day call at 90% vol is worth
        # about 1e-9 dollars. The objective is then flat across the whole
        # bracket and Brent returns its left endpoint, so the inverter used
        # to report 0.001 -- 0.1% vol for an option priced at 90%. Wrong in a
        # way that looks like data rather than an error, which is worse than
        # failing, because it would flow into an SVI fit unnoticed.
        T = 1 / 365.25
        target = price(F0, 110_000.0, T, 0.9, "C")
        assert target < 1e-6
        assert implied_vol(F0, 110_000.0, T, target, "C") is None

    def test_never_returns_the_bracket_endpoints(self):
        rng = np.random.default_rng(7)
        for _ in range(400):
            strike = float(rng.uniform(20_000, 250_000))
            T = float(rng.uniform(0.25, 400) / 365.25)
            sigma = float(rng.uniform(0.05, 6.0))
            target = price(F0, strike, T, sigma, "C")
            result = implied_vol(F0, strike, T, float(target), "C")
            if result is not None:
                assert result > VOL_BRACKET[0] + 1e-6
                assert result < VOL_BRACKET[1] - 1e-6

    def test_never_returns_a_value_outside_the_bracket(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            strike = float(rng.uniform(20_000, 200_000))
            T = float(rng.uniform(0.5, 400) / 365.25)
            premium = float(rng.uniform(0, F0))
            result = implied_vol(F0, strike, T, premium, "C")
            assert result is None or VOL_BRACKET[0] <= result <= VOL_BRACKET[1]


class TestImpliedVolMany:
    def test_inverts_a_column_and_marks_failures_nan(self):
        strikes = np.array([70_000.0, 80_000.0, 90_000.0, 80_000.0])
        prices = np.array(
            [
                price(F0, 70_000.0, T0, 0.5, "C"),
                price(F0, 80_000.0, T0, 0.5, "C"),
                price(F0, 90_000.0, T0, 0.5, "C"),
                F0 * 2,  # impossible
            ]
        )
        out = implied_vol_many(F0, strikes, T0, prices, "C")
        assert out[:3] == pytest.approx([0.5, 0.5, 0.5], abs=1e-5)
        assert np.isnan(out[3])

    def test_accepts_per_quote_option_types(self):
        out = implied_vol_many(
            [F0, F0],
            [70_000.0, 90_000.0],
            [T0, T0],
            [price(F0, 70_000.0, T0, 0.4, "P"), price(F0, 90_000.0, T0, 0.6, "C")],
            ["P", "C"],
        )
        assert out == pytest.approx([0.4, 0.6], abs=1e-5)


class TestGreeks:
    def test_call_delta_between_zero_and_one(self):
        assert 0.0 < greeks(F0, F0, T0, 0.5, "C")["delta"] < 1.0

    def test_put_delta_between_minus_one_and_zero(self):
        assert -1.0 < greeks(F0, F0, T0, 0.5, "P")["delta"] < 0.0

    def test_atm_delta_is_near_a_half(self):
        assert greeks(F0, F0, T0, 0.5, "C")["delta"] == pytest.approx(0.5, abs=0.06)

    def test_call_and_put_delta_differ_by_one(self):
        call = greeks(F0, 90_000.0, T0, 0.5, "C")["delta"]
        put = greeks(F0, 90_000.0, T0, 0.5, "P")["delta"]
        assert call - put == pytest.approx(1.0)

    def test_gamma_and_vega_match_calls_and_puts(self):
        call = greeks(F0, 90_000.0, T0, 0.5, "C")
        put = greeks(F0, 90_000.0, T0, 0.5, "P")
        assert call["gamma"] == pytest.approx(put["gamma"])
        assert call["vega"] == pytest.approx(put["vega"])

    def test_theta_is_negative_for_a_long_option(self):
        assert greeks(F0, F0, T0, 0.5, "C")["theta"] < 0

    def test_delta_matches_a_numerical_derivative(self):
        bump = 1.0
        up = price(F0 + bump, 85_000.0, T0, 0.5, "C")
        down = price(F0 - bump, 85_000.0, T0, 0.5, "C")
        assert greeks(F0, 85_000.0, T0, 0.5, "C")["delta"] == pytest.approx(
            (up - down) / (2 * bump), abs=1e-6
        )

    def test_vega_matches_a_numerical_derivative(self):
        bump = 1e-4
        up = price(F0, 85_000.0, T0, 0.5 + bump, "C")
        down = price(F0, 85_000.0, T0, 0.5 - bump, "C")
        assert greeks(F0, 85_000.0, T0, 0.5, "C")["vega"] == pytest.approx(
            (up - down) / (2 * bump), rel=1e-5
        )

    def test_gamma_matches_a_numerical_second_derivative(self):
        bump = 10.0
        up = price(F0 + bump, 85_000.0, T0, 0.5, "C")
        mid = price(F0, 85_000.0, T0, 0.5, "C")
        down = price(F0 - bump, 85_000.0, T0, 0.5, "C")
        assert greeks(F0, 85_000.0, T0, 0.5, "C")["gamma"] == pytest.approx(
            (up - 2 * mid + down) / bump**2, rel=1e-3
        )
