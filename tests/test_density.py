"""
Tests for the risk-neutral density.

The decisive test is the flat-vol case. With w constant the density must
collapse to the lognormal one -- k Normal(-w/2, w) -- which is forced by
risk-neutrality, since E[S_T] = F means E[e^k] = 1. That pins the algebra,
the Jacobian and the normalisation all at once; a density that only ever
gets checked for "looks plausible" would pass with a sign error in d2.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from surface.density import (
    cdf,
    density_log_moneyness,
    density_price,
    probability_between,
    quantile,
    tail_report,
    total_mass,
)
from surface.surface import VolSurface

FORWARD = 80_000.0
NEAR = {"a": 0.004, "b": 0.030, "rho": -0.25, "m": 0.010, "sigma": 0.060}
FAR = {"a": 0.020, "b": 0.090, "rho": -0.32, "m": 0.020, "sigma": 0.140}


def smiled() -> VolSurface:
    return VolSurface({0.10: NEAR, 0.50: FAR},
                      {0.10: FORWARD, 0.50: FORWARD})


def flat(sigma: float = 0.5, T: float = 0.25) -> VolSurface:
    """A surface with no smile at all: w constant in k, so b = 0."""
    params = {"a": sigma**2 * T, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    return VolSurface({T: params}, {T: FORWARD}, clamp_to_trust_radius=False)


class TestFlatVolReducesToLognormal:
    def test_density_matches_the_normal_density_of_k(self):
        sigma, T = 0.5, 0.25
        w = sigma**2 * T
        surface = flat(sigma, T)
        k = np.linspace(-1.0, 1.0, 41)
        expected = norm.pdf(k, loc=-w / 2, scale=np.sqrt(w))
        assert density_log_moneyness(surface, k, T) == pytest.approx(
            expected, rel=1e-3
        )

    def test_cdf_matches_the_lognormal_cdf(self):
        sigma, T = 0.5, 0.25
        w = sigma**2 * T
        surface = flat(sigma, T)
        for price in (55_000.0, 70_000.0, 80_000.0, 95_000.0, 120_000.0):
            k = np.log(price / FORWARD)
            expected = norm.cdf((k + w / 2) / np.sqrt(w))
            assert cdf(surface, price, T) == pytest.approx(expected, abs=2e-3)

    def test_the_forward_is_the_risk_neutral_mean(self):
        # E[S_T] = F is what makes the measure risk-neutral. If the Jacobian
        # or the drift term were wrong this would miss.
        sigma, T = 0.5, 0.25
        surface = flat(sigma, T)
        prices = np.linspace(1_000.0, 600_000.0, 60_001)
        q = density_price(surface, prices, T)
        assert np.trapezoid(prices * q, prices) == pytest.approx(FORWARD, rel=5e-3)


class TestNormalisation:
    def test_density_integrates_to_one(self):
        assert total_mass(smiled(), 0.25) == pytest.approx(1.0, abs=5e-3)

    @pytest.mark.parametrize("T", [0.12, 0.25, 0.45])
    def test_holds_across_maturities(self, T):
        assert total_mass(smiled(), T) == pytest.approx(1.0, abs=1e-2)

    def test_price_density_also_integrates_to_one(self):
        surface = smiled()
        prices = np.linspace(2_000.0, 900_000.0, 80_001)
        q = density_price(surface, prices, 0.25)
        assert np.trapezoid(q, prices) == pytest.approx(1.0, abs=1e-2)


class TestCdf:
    def test_is_monotonic_in_price(self):
        surface = smiled()
        values = [cdf(surface, p, 0.25)
                  for p in (40_000, 60_000, 80_000, 100_000, 140_000)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_bounded_by_zero_and_one(self):
        surface = smiled()
        assert 0.0 <= cdf(surface, 1_000.0, 0.25) <= 1.0
        assert 0.0 <= cdf(surface, 5_000_000.0, 0.25) <= 1.0

    def test_far_below_is_almost_zero_and_far_above_almost_one(self):
        surface = smiled()
        assert cdf(surface, 500.0, 0.25) < 0.02
        assert cdf(surface, 8_000_000.0, 0.25) > 0.98

    def test_probability_between_is_the_difference(self):
        surface = smiled()
        assert probability_between(surface, 70_000, 90_000, 0.25) == pytest.approx(
            cdf(surface, 90_000, 0.25) - cdf(surface, 70_000, 0.25)
        )

    def test_uncertainty_widens_with_maturity(self):
        # A longer horizon must put more mass far from the forward.
        surface = smiled()
        near = probability_between(surface, 70_000, 90_000, 0.12)
        far = probability_between(surface, 70_000, 90_000, 0.45)
        assert far < near


class TestQuantiles:
    def test_are_ordered(self):
        surface = smiled()
        values = [quantile(surface, p, 0.25) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_invert_the_cdf(self):
        surface = smiled()
        for p in (0.1, 0.5, 0.9):
            assert cdf(surface, quantile(surface, p, 0.25), 0.25) == pytest.approx(
                p, abs=0.01
            )

    def test_median_sits_above_the_forward_for_a_downside_skew(self):
        # Not what it looks like at first. The lognormal median is
        # F*exp(-w/2) < F, so the instinct is that the median is below the
        # forward. But risk-neutrality pins the MEAN at F (verified to five
        # decimals), and a downside skew is a long left tail -- which with
        # the mean fixed pushes the median the other way, above F.
        surface = smiled()
        assert quantile(surface, 0.5, 0.25) > surface.forward(0.25)

    def test_a_flat_surface_does_put_the_median_below_the_forward(self):
        # With no skew the lognormal intuition is correct again, which is
        # what makes the previous test a statement about skew rather than
        # about the code.
        sigma, T = 0.5, 0.25
        surface = flat(sigma, T)
        assert quantile(surface, 0.5, T) == pytest.approx(
            FORWARD * np.exp(-(sigma**2 * T) / 2), rel=1e-3
        )

    def test_rejects_probabilities_outside_the_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                quantile(smiled(), bad, 0.25)


class TestSkew:
    def test_downside_skew_puts_more_mass_in_the_left_tail(self):
        # rho < 0 steepens the left wing, so the crash probability must
        # exceed the mirror-image rally probability.
        T = 0.25
        skewed = {"a": 0.02, "b": 0.10, "rho": -0.6, "m": 0.0, "sigma": 0.15}
        surface = VolSurface({T: skewed}, {T: FORWARD},
                             clamp_to_trust_radius=False)
        down = cdf(surface, FORWARD * 0.7, T)
        up = 1.0 - cdf(surface, FORWARD / 0.7, T)
        assert down > up

    def test_a_symmetric_smile_is_nearly_balanced(self):
        T = 0.25
        symmetric = {"a": 0.02, "b": 0.10, "rho": 0.0, "m": 0.0, "sigma": 0.15}
        surface = VolSurface({T: symmetric}, {T: FORWARD},
                             clamp_to_trust_radius=False)
        down = cdf(surface, FORWARD * 0.7, T)
        up = 1.0 - cdf(surface, FORWARD / 0.7, T)
        assert down == pytest.approx(up, abs=0.03)


class TestTailReport:
    def test_reports_levels_quantiles_and_mass(self):
        out = tail_report(smiled(), 0.25, levels={"below 50k": 50_000.0})
        assert out["total_mass"] == pytest.approx(1.0, abs=1e-2)
        assert set(out["quantiles"]) == {0.05, 0.25, 0.5, 0.75, 0.95}
        level = out["levels"]["below 50k"]
        assert level["p_below"] + level["p_above"] == pytest.approx(1.0)

    def test_flags_a_level_that_depends_on_a_clamped_wing(self):
        # The honest part: a deep-tail probability read off a clamped wing is
        # an assumption about extrapolation, not a market observation.
        surface = VolSurface(
            {0.25: NEAR}, {0.25: FORWARD}, {0.25: (-0.10, 0.10)}
        )
        out = tail_report(surface, 0.25,
                          levels={"near": 79_000.0, "deep": 20_000.0})
        assert not out["levels"]["near"]["extrapolated"]
        assert out["levels"]["deep"]["extrapolated"]
