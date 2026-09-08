"""
Tests for the arbitrage checks.

A checker that never fires is indistinguishable from one that works, so the
important tests here construct surfaces with arbitrage deliberately built in
and confirm it is found. The Durrleman function is also cross-checked against
a numerical density, since an algebra slip in g(k) would otherwise pass every
test that only ever looks at clean slices.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface.arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_surface,
    durrleman_g,
    implied_density,
    svi_derivatives,
    trust_radius,
    wing_bound,
)
from surface.svi import svi_total_variance

CLEAN = {"a": 0.012, "b": 0.085, "rho": -0.28, "m": 0.018, "sigma": 0.115}


def scaled(params: dict, factor: float) -> dict:
    """Same shape, more total variance -- what a longer maturity looks like."""
    return {**params, "a": params["a"] * factor, "b": params["b"] * factor}


class TestDerivatives:
    def test_w_matches_the_svi_formula(self):
        k = np.linspace(-0.5, 0.5, 11)
        w, _, _ = svi_derivatives(k, CLEAN)
        assert w == pytest.approx(svi_total_variance(k, **CLEAN))

    def test_first_derivative_matches_finite_differences(self):
        k = np.linspace(-0.5, 0.5, 11)
        h = 1e-6
        up, _, _ = svi_derivatives(k + h, CLEAN)
        down, _, _ = svi_derivatives(k - h, CLEAN)
        _, w1, _ = svi_derivatives(k, CLEAN)
        assert w1 == pytest.approx((up - down) / (2 * h), rel=1e-5)

    def test_second_derivative_matches_finite_differences(self):
        k = np.linspace(-0.5, 0.5, 11)
        h = 1e-4
        up, _, _ = svi_derivatives(k + h, CLEAN)
        mid, _, _ = svi_derivatives(k, CLEAN)
        down, _, _ = svi_derivatives(k - h, CLEAN)
        _, _, w2 = svi_derivatives(k, CLEAN)
        assert w2 == pytest.approx((up - 2 * mid + down) / h**2, rel=1e-4)

    def test_second_derivative_is_positive(self):
        # w'' = b*sigma^2 / root^3 > 0 for b > 0: an SVI slice is convex in k.
        # That is not the same as convex in strike, which is why g(k) still
        # has to be checked.
        _, _, w2 = svi_derivatives(np.linspace(-2, 2, 101), CLEAN)
        assert (w2 > 0).all()


class TestDurrleman:
    def test_clean_slice_has_positive_g(self):
        g = durrleman_g(np.linspace(-1.5, 1.5, 601), CLEAN)
        assert (g > 0).all()

    def test_density_integrates_to_about_one(self):
        # The strongest check on the g(k) algebra: if g were wrong, the
        # implied density would not be a probability density.
        k = np.linspace(-6, 6, 24001)
        density = implied_density(k, CLEAN)
        assert np.trapezoid(density, k) == pytest.approx(1.0, abs=5e-3)

    def test_density_is_positive_for_a_clean_slice(self):
        assert (implied_density(np.linspace(-3, 3, 1201), CLEAN) >= 0).all()

    def test_density_sign_follows_g(self):
        k = np.linspace(-2, 2, 401)
        bad = {**CLEAN, "b": 1.6, "rho": -0.95, "sigma": 0.02}
        g = durrleman_g(k, bad)
        density = implied_density(k, bad)
        assert np.array_equal(np.sign(g), np.sign(density))


class TestButterfly:
    def test_clean_slice_reports_nothing(self):
        assert check_butterfly_arbitrage(CLEAN) == []

    def test_steep_wing_slice_is_caught(self):
        # Large b with rho near -1 and tight sigma is the classic butterfly
        # violation: the left wing is too steep for the density to stay
        # non-negative.
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        found = check_butterfly_arbitrage(bad)
        assert found
        assert found[0]["g_min"] < 0
        assert found[0]["density_min"] < 0

    def test_violations_are_grouped_into_runs(self):
        # One bad region should read as one finding, not one per grid point.
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        found = check_butterfly_arbitrage(bad, np.linspace(-1.5, 1.5, 2001))
        assert 1 <= len(found) <= 3
        for item in found:
            assert item["k_from"] <= item["k_worst"] <= item["k_to"]

    def test_quoted_range_flag_distinguishes_the_two_cases(self):
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        wide = check_butterfly_arbitrage(bad, quoted_range=(-2.0, 2.0))
        narrow = check_butterfly_arbitrage(bad, quoted_range=(-0.001, 0.001))
        assert any(v["inside_quoted_range"] for v in wide)
        assert not any(v["inside_quoted_range"] for v in narrow)


class TestCalendar:
    def test_increasing_total_variance_reports_nothing(self):
        surface = {0.25: CLEAN, 0.50: scaled(CLEAN, 2.0), 1.00: scaled(CLEAN, 4.0)}
        assert check_calendar_arbitrage(surface) == []

    def test_decreasing_total_variance_is_caught(self):
        # The long-dated slice implies LESS accumulated variance than the
        # short-dated one: a guaranteed calendar spread arbitrage.
        surface = {0.25: scaled(CLEAN, 2.0), 1.00: CLEAN}
        found = check_calendar_arbitrage(surface)
        assert found
        assert found[0]["shortfall"] > 0
        assert found[0]["w2"] < found[0]["w1"]

    def test_only_adjacent_maturities_are_compared(self):
        # One bad slice in the middle should not be reported against every
        # other maturity as well.
        surface = {0.25: CLEAN, 0.50: scaled(CLEAN, 0.5), 1.00: scaled(CLEAN, 4.0)}
        found = check_calendar_arbitrage(surface)
        pairs = {(v["T1"], v["T2"]) for v in found}
        assert (0.25, 1.00) not in pairs

    def test_a_crossing_only_in_the_wings_is_found_and_flagged(self):
        near = {**CLEAN, "b": 0.30}
        far = {**scaled(CLEAN, 1.5), "b": 0.05}
        found = check_calendar_arbitrage(
            {0.25: near, 1.00: far},
            quoted_ranges={0.25: (-0.1, 0.1), 1.00: (-0.1, 0.1)},
        )
        assert found
        assert all(abs(v["k_worst"]) > 0.1 for v in found)
        assert not any(v["inside_quoted_range"] for v in found)


class TestWingBound:
    def test_clean_slice_satisfies_the_bound(self):
        assert wing_bound(CLEAN, 0.25)["ok"]

    def test_steep_slice_violates_it(self):
        assert not wing_bound({**CLEAN, "b": 20.0}, 1.0)["ok"]

    def test_bound_tightens_with_maturity(self):
        # 4/T shrinks as T grows, so the same slice can pass short and fail long.
        steep = {**CLEAN, "b": 3.0}
        assert wing_bound(steep, 0.1)["ok"]
        assert not wing_bound(steep, 5.0)["ok"]


class TestCheckSurface:
    def test_clean_surface_is_reported_clean(self):
        surface = {0.25: CLEAN, 0.50: scaled(CLEAN, 2.0), 1.00: scaled(CLEAN, 4.0)}
        report = check_surface(surface)
        assert report["butterfly"] == []
        assert report["calendar"] == []
        assert all(v["ok"] for v in report["wing_bound"].values())

    def test_both_kinds_of_violation_are_reported_together(self):
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        report = check_surface({0.25: scaled(CLEAN, 3.0), 1.00: bad})
        assert report["butterfly"]
        assert report["calendar"]

    def test_butterfly_findings_carry_their_maturity(self):
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        report = check_surface({0.75: bad})
        assert all(v["T"] == 0.75 for v in report["butterfly"])


class TestTrustRadius:
    def test_clean_slice_is_sound_across_the_whole_test_width(self):
        radius = trust_radius(CLEAN, width=3.0)
        assert radius["sound_at_the_money"]
        assert radius["k_low"] <= -2.9 and radius["k_high"] >= 2.9

    def test_steep_slice_is_bounded_on_the_side_that_breaks(self):
        # Negative rho steepens the left wing, so that is where convexity
        # fails first -- matching the live surface, where every violation
        # found was on the downside.
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        radius = trust_radius(bad)
        assert radius["sound_at_the_money"]
        assert radius["k_low"] > -3.0

    def test_reports_whether_the_quoted_range_is_covered(self):
        # This slice is deliberately broken and its left wing fails almost
        # immediately -- sound only to k = -0.016 -- so the covered range has
        # to be narrower than that to pass.
        bad = {"a": 0.02, "b": 1.6, "rho": -0.95, "m": 0.0, "sigma": 0.02}
        radius = trust_radius(bad)
        assert -0.05 < radius["k_low"] < 0.0
        assert trust_radius(bad, quoted_range=(-0.01, 0.01))["covers_quoted_range"]
        assert not trust_radius(bad, quoted_range=(-2.5, 2.5))["covers_quoted_range"]

    def test_a_slice_broken_at_the_money_reports_zero_radius(self):
        broken = {"a": 0.02, "b": 8.0, "rho": -0.99, "m": 0.0, "sigma": 0.005}
        radius = trust_radius(broken)
        if not radius["sound_at_the_money"]:
            assert radius["k_low"] == 0.0 and radius["k_high"] == 0.0

    def test_radius_is_a_contiguous_interval_around_the_money(self):
        radius = trust_radius(CLEAN)
        assert radius["k_low"] <= 0.0 <= radius["k_high"]
