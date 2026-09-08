"""
Tests for the queryable surface object.

The properties that matter are the ones that make the surface safe to price
from: exact reproduction at the fitted maturities, interpolation that stays
between its neighbours, extrapolation that does not blow up as T -> 0, and
no calendar arbitrage manufactured by the interpolation itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface.arbitrage import check_calendar_arbitrage
from surface.surface import VolSurface, from_fits, inverse_greeks
from surface.svi import svi_total_variance

NEAR = {"a": 0.004, "b": 0.030, "rho": -0.25, "m": 0.010, "sigma": 0.060}
FAR = {"a": 0.020, "b": 0.090, "rho": -0.32, "m": 0.020, "sigma": 0.140}
FORWARD = 80_000.0


def make_surface(**kwargs) -> VolSurface:
    return VolSurface(
        {0.10: NEAR, 0.50: FAR},
        {0.10: FORWARD, 0.50: FORWARD * 1.01},
        **kwargs,
    )


class TestConstruction:
    def test_rejects_an_empty_surface(self):
        with pytest.raises(ValueError):
            VolSurface({}, {})

    def test_rejects_a_slice_with_no_forward(self):
        with pytest.raises(ValueError):
            VolSurface({0.1: NEAR, 0.5: FAR}, {0.1: FORWARD})

    def test_from_fits_takes_the_forward_each_slice_was_calibrated_against(self):
        fits = {
            0.25: {**NEAR, "forward": 79_000.0},
            0.75: {**FAR, "forward": 81_000.0},
        }
        surface = from_fits(fits)
        assert surface.forward(0.25) == pytest.approx(79_000.0)
        assert surface.forward(0.75) == pytest.approx(81_000.0)


class TestForwardCurve:
    def test_exact_at_the_fitted_maturities(self):
        surface = make_surface()
        assert surface.forward(0.10) == pytest.approx(FORWARD)
        assert surface.forward(0.50) == pytest.approx(FORWARD * 1.01)

    def test_interpolates_between_them(self):
        f = make_surface().forward(0.30)
        assert FORWARD < f < FORWARD * 1.01

    def test_is_log_linear_not_linear(self):
        # A forward curve is a level times accumulated carry, so ln F is what
        # moves linearly in T. The midpoint must sit at the geometric mean.
        surface = make_surface()
        assert surface.forward(0.30) == pytest.approx(
            np.sqrt(FORWARD * FORWARD * 1.01)
        )

    def test_held_flat_beyond_the_fitted_range(self):
        surface = make_surface()
        assert surface.forward(5.0) == pytest.approx(surface.forward(0.50))
        assert surface.forward(0.001) == pytest.approx(surface.forward(0.10))


class TestTotalVariance:
    def test_reproduces_a_fitted_slice_exactly(self):
        surface = make_surface(clamp_to_trust_radius=False)
        k = np.linspace(-0.3, 0.3, 21)
        assert surface.total_variance(k, 0.10) == pytest.approx(
            svi_total_variance(k, **NEAR)
        )

    def test_interpolation_lies_between_its_neighbours(self):
        surface = make_surface(clamp_to_trust_radius=False)
        k = np.linspace(-0.3, 0.3, 21)
        mid = surface.total_variance(k, 0.30)
        low = svi_total_variance(k, **NEAR)
        high = svi_total_variance(k, **FAR)
        assert (mid >= np.minimum(low, high) - 1e-12).all()
        assert (mid <= np.maximum(low, high) + 1e-12).all()

    def test_interpolation_is_linear_in_T(self):
        surface = make_surface(clamp_to_trust_radius=False)
        k = 0.05
        quarter = surface.total_variance(k, 0.20)
        expected = 0.75 * svi_total_variance(k, **NEAR) + 0.25 * svi_total_variance(
            k, **FAR
        )
        assert quarter == pytest.approx(expected)

    def test_goes_to_zero_as_T_goes_to_zero(self):
        # Holding total variance flat below the shortest slice would send
        # IV = sqrt(w/T) to infinity. Holding the variance RATE flat sends w
        # to zero instead, which is what an option with no time left implies.
        surface = make_surface()
        assert surface.total_variance(0.0, 1e-6) < 1e-6

    def test_is_non_decreasing_in_T_everywhere(self):
        surface = make_surface(clamp_to_trust_radius=False)
        k = np.linspace(-0.5, 0.5, 41)
        maturities = np.linspace(0.01, 3.0, 120)
        w = np.array([surface.total_variance(k, T) for T in maturities])
        assert (np.diff(w, axis=0) >= -1e-12).all()

    def test_zero_maturity_gives_zero_variance(self):
        assert float(make_surface().total_variance(0.0, 0.0)) == 0.0


class TestImpliedVol:
    def test_flat_in_T_beyond_the_longest_expiry(self):
        # The consequence of holding the variance rate constant: vol stays at
        # the level of the last fitted slice rather than decaying to zero.
        surface = make_surface()
        far = surface.iv(FORWARD, 0.50)
        assert surface.iv(FORWARD, 2.0) == pytest.approx(far, rel=0.02)

    def test_rejects_non_positive_maturity(self):
        with pytest.raises(ValueError):
            make_surface().iv(FORWARD, 0.0)

    def test_smile_is_higher_in_the_wings(self):
        surface = make_surface()
        atm = surface.iv(FORWARD, 0.30)
        assert surface.iv(FORWARD * 0.75, 0.30) > atm
        assert surface.iv(FORWARD * 1.35, 0.30) > atm

    def test_accepts_an_array_of_strikes(self):
        out = make_surface().iv(np.array([70_000.0, 80_000.0, 95_000.0]), 0.3)
        assert np.shape(out) == (3,)

    def test_interpolation_introduces_no_calendar_arbitrage(self):
        # Two arbitrage-free slices can still produce arbitrage if you
        # interpolate implied vol instead of total variance. This surface
        # interpolates w, so the whole T-continuum must stay clean.
        surface = make_surface(clamp_to_trust_radius=False)
        k = np.linspace(-0.4, 0.4, 41)
        grid = np.linspace(0.12, 0.48, 10)
        w = {T: surface.total_variance(k, T) for T in grid}
        for T1, T2 in zip(grid, grid[1:]):
            assert (w[T2] - w[T1] >= -1e-12).all()


class TestClamping:
    def test_clamped_and_unclamped_agree_near_the_money(self):
        clamped = make_surface(clamp_to_trust_radius=True)
        free = make_surface(clamp_to_trust_radius=False)
        assert clamped.iv(FORWARD, 0.3) == pytest.approx(free.iv(FORWARD, 0.3))

    def test_reports_when_a_query_leaves_the_fitted_region(self):
        surface = make_surface()
        assert not surface.is_extrapolated(FORWARD, 0.3)["beyond_fitted_maturities"]
        assert surface.is_extrapolated(FORWARD, 9.0)["beyond_fitted_maturities"]

    def test_log_moneyness_is_reported_against_the_forward_at_T(self):
        surface = make_surface()
        info = surface.is_extrapolated(surface.forward(0.3), 0.3)
        assert info["k"] == pytest.approx(0.0, abs=1e-12)


class TestPriceAndGreeks:
    def test_price_is_positive_and_below_the_forward(self):
        surface = make_surface()
        price = surface.price(FORWARD, 0.3, "C")
        assert 0 < price < surface.forward(0.3)

    def test_put_call_parity_holds_through_the_surface(self):
        surface = make_surface()
        K, T = 85_000.0, 0.3
        call = surface.price(K, T, "C")
        put = surface.price(K, T, "P")
        assert call - put == pytest.approx(surface.forward(T) - K, abs=1e-6)

    def test_greeks_carry_the_vol_and_forward_used(self):
        out = make_surface().greeks(FORWARD, 0.3, "C")
        assert out["iv"] == pytest.approx(make_surface().iv(FORWARD, 0.3))
        assert out["forward"] == pytest.approx(make_surface().forward(0.3))

    def test_call_delta_is_between_zero_and_one(self):
        assert 0.0 < make_surface().greeks(FORWARD, 0.3, "C")["delta"] < 1.0

    def test_delta_matches_a_numerical_derivative_of_price(self):
        surface = make_surface()
        T, bump = 0.3, 1.0
        forward = surface.forward(T)
        # Differentiate the Black-76 price in F at fixed vol, which is what
        # delta means -- not the total derivative through the smile.
        from pricing.black76 import price as bs_price

        sigma = surface.iv(forward, T)
        up = bs_price(forward + bump, forward, T, sigma, "C")
        down = bs_price(forward - bump, forward, T, sigma, "C")
        assert surface.greeks(forward, T, "C")["delta"] == pytest.approx(
            (up - down) / (2 * bump), abs=1e-6
        )


class TestInverseGreeks:
    def test_coin_price_is_the_usd_price_over_the_forward(self):
        out = inverse_greeks(4_000.0, 0.5, 1e-5, 80_000.0)
        assert out["price_coin"] == pytest.approx(0.05)

    def test_coin_delta_subtracts_the_settlement_term(self):
        # dV_coin/dF = (delta_usd - V_coin) / F. The subtraction is the whole
        # point: an inverse payoff depends on the price twice.
        out = inverse_greeks(4_000.0, 0.5, 1e-5, 80_000.0)
        assert out["delta_coin"] == pytest.approx((0.5 - 0.05) / 80_000.0)

    def test_coin_delta_matches_a_numerical_derivative(self):
        from pricing.black76 import greeks as bs_greeks
        from pricing.black76 import price as bs_price

        F, K, T, sigma = 80_000.0, 85_000.0, 0.3, 0.45
        bump = 0.5

        def coin_price(forward):
            return bs_price(forward, K, T, sigma, "C") / forward

        numeric = (coin_price(F + bump) - coin_price(F - bump)) / (2 * bump)
        g = bs_greeks(F, K, T, sigma, "C")
        analytic = inverse_greeks(
            bs_price(F, K, T, sigma, "C"), g["delta"], g["gamma"], F
        )["delta_coin"]
        assert analytic == pytest.approx(numeric, rel=1e-6)

    def test_coin_and_usd_delta_differ_materially(self):
        # If these agreed there would be nothing to say about inverse
        # settlement. Scaled by F they differ by the coin premium.
        F = 80_000.0
        out = inverse_greeks(4_000.0, 0.5, 1e-5, F)
        assert abs(out["delta_coin"] * F - 0.5) == pytest.approx(0.05)


class TestCalendarMonotonicityEnforcement:
    def test_a_narrow_short_slice_cannot_overtake_a_longer_one(self):
        # The live failure this exists for: a slice quoted over a narrow
        # k-range, extrapolated far into the wing, exceeded the next
        # maturity's total variance and manufactured calendar arbitrage out
        # of two individually sound fits.
        narrow = {"a": 0.004, "b": 0.30, "rho": -0.6, "m": 0.0, "sigma": 0.05}
        wide = {"a": 0.020, "b": 0.05, "rho": -0.2, "m": 0.0, "sigma": 0.20}
        surface = VolSurface(
            {0.10: narrow, 0.50: wide},
            {0.10: FORWARD, 0.50: FORWARD},
            {0.10: (-0.10, 0.10), 0.50: (-0.60, 0.60)},
        )
        k = np.linspace(-1.0, 1.0, 81)
        grid = np.linspace(0.10, 0.50, 20)
        w = np.array([surface.total_variance(k, T) for T in grid])
        assert (np.diff(w, axis=0) >= -1e-12).all()

    def test_without_the_running_max_the_crossing_reappears(self):
        # Confirms the guard is what fixes it, not the clamping alone.
        narrow = {"a": 0.004, "b": 0.30, "rho": -0.6, "m": 0.0, "sigma": 0.05}
        wide = {"a": 0.020, "b": 0.05, "rho": -0.2, "m": 0.0, "sigma": 0.20}
        args = ({0.10: narrow, 0.50: wide}, {0.10: FORWARD, 0.50: FORWARD},
                {0.10: (-0.10, 0.10), 0.50: (-0.60, 0.60)})
        free = VolSurface(*args, enforce_calendar_monotonicity=False)
        k = np.linspace(-1.0, 1.0, 81)
        w = np.array([free.total_variance(k, T)
                      for T in np.linspace(0.10, 0.50, 20)])
        assert np.min(np.diff(w, axis=0)) < 0

    def test_the_correction_does_not_bind_where_slices_are_ordered(self):
        # Two properly ordered slices must be returned untouched.
        surface = make_surface()
        k = np.linspace(-0.2, 0.2, 21)
        raw_near = surface._slice_variance(k, 0.10)
        assert surface._slice_variances(k)[0] == pytest.approx(raw_near)

    def test_clamping_holds_the_wing_flat_beyond_the_quoted_range(self):
        surface = VolSurface(
            {0.25: NEAR}, {0.25: FORWARD}, {0.25: (-0.10, 0.10)}
        )
        far = surface.total_variance(-0.9, 0.25)
        edge = surface.total_variance(-0.10, 0.25)
        assert float(far) == pytest.approx(float(edge))
