"""
Tests for SVI calibration.

The decisive test is parameter recovery: generate total variance from known
(a, b, rho, m, sigma), fit it back, and check the fit returns what it was
given. A calibrator that cannot recover parameters from its own model has no
business being pointed at market data.

The rest guard the failure modes that matter -- refusing thin slices instead
of over-fitting them, keeping the parameters inside their admissible ranges,
and not silently returning a local minimum on a skewed smile.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface.svi import (
    MIN_POINTS,
    fit_svi_slice,
    fit_svi_surface,
    svi_implied_vol,
    svi_total_variance,
    vega_weights,
)

# A realistic BTC slice: ~40% vol at three months, skewed to the downside.
TRUE = {"a": 0.012, "b": 0.085, "rho": -0.28, "m": 0.018, "sigma": 0.115}
K_GRID = np.linspace(-0.45, 0.45, 25)


def synth(params=None, k=None):
    params = params or TRUE
    k = K_GRID if k is None else k
    return k, svi_total_variance(k, **params)


class TestTotalVariance:
    def test_matches_the_closed_form(self):
        a, b, rho, m, s = 0.02, 0.1, -0.3, 0.05, 0.2
        k = 0.1
        expected = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
        assert svi_total_variance(k, a, b, rho, m, s) == pytest.approx(expected)

    def test_is_positive_across_a_wide_grid(self):
        _, w = synth()
        assert (w > 0).all()

    def test_minimum_sits_near_m(self):
        # `m` is the horizontal shift, so the smile's floor should be close
        # to it for a mildly skewed slice.
        k = np.linspace(-1.0, 1.0, 4001)
        w = svi_total_variance(k, **TRUE)
        assert k[int(np.argmin(w))] == pytest.approx(TRUE["m"], abs=0.12)

    def test_wings_rise_on_both_sides(self):
        w = svi_total_variance(np.array([-1.0, 0.0, 1.0]), **TRUE)
        assert w[0] > w[1] and w[2] > w[1]

    def test_implied_vol_conversion(self):
        T = 0.25
        k = np.array([-0.1, 0.0, 0.1])
        w = svi_total_variance(k, **TRUE)
        assert svi_implied_vol(k, T, TRUE) == pytest.approx(np.sqrt(w / T))


class TestParameterRecovery:
    def test_recovers_known_parameters_exactly(self):
        k, w = synth()
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        for name, value in TRUE.items():
            assert fit[name] == pytest.approx(value, abs=2e-3), name
        assert fit["rmse"] < 1e-5

    @pytest.mark.parametrize("rho", [-0.75, -0.4, 0.0, 0.4, 0.75])
    def test_recovers_skew_of_either_sign(self, rho):
        # A fit started only from m=0 tends to collapse toward symmetric
        # smiles. BTC's is persistently skewed, so this must hold both ways.
        params = {**TRUE, "rho": rho}
        k, w = synth(params)
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        assert fit["rho"] == pytest.approx(rho, abs=0.05)

    @pytest.mark.parametrize("sigma", [0.03, 0.1])
    def test_recovers_curvature_across_scales(self, sigma):
        params = {**TRUE, "sigma": sigma}
        k, w = synth(params)
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        assert fit["sigma"] == pytest.approx(sigma, rel=0.1)

    def test_wide_curvature_is_identified_only_up_to_the_curve(self):
        # With sigma large against the observed k-range, sqrt(y^2+1) is
        # nearly flat over the data and the slice is close to parabolic.
        # Many (a, b, rho, m, sigma) combinations then fit equally well, so
        # the parameters are only weakly identified while the curve is not.
        # The surface serves the curve, so that is what must be right.
        k, w = synth({**TRUE, "sigma": 0.4})
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        curve = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                                   fit["m"], fit["sigma"])
        assert np.max(np.abs(curve - w)) < 1e-4

    def test_fit_reproduces_the_curve_even_where_parameters_drift(self):
        # (m, sigma) trade off against each other, so the parameters can be
        # slightly off while the curve is right. The curve is what the
        # surface actually serves, so check that too.
        k, w = synth()
        fit = fit_svi_slice(k, w)
        refit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                                   fit["m"], fit["sigma"])
        assert np.max(np.abs(refit - w)) < 1e-5

    def test_survives_realistic_quote_noise(self):
        k, w = synth()
        rng = np.random.default_rng(11)
        noisy = w * (1 + rng.normal(0, 0.01, size=len(w)))
        fit = fit_svi_slice(k, noisy)
        assert fit["ok"]
        clean = svi_total_variance(k, **TRUE)
        refit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                                   fit["m"], fit["sigma"])
        # Fit should track the underlying curve, not chase the noise.
        assert np.sqrt(np.mean((refit - clean) ** 2)) < 0.002


class TestConstraints:
    @pytest.mark.parametrize("rho", [-0.9, 0.0, 0.9])
    def test_parameters_stay_admissible(self, rho):
        k, w = synth({**TRUE, "rho": rho})
        fit = fit_svi_slice(k, w)
        assert fit["b"] >= 0.0
        assert -1.0 <= fit["rho"] <= 1.0
        assert fit["sigma"] > 0.0

    def test_total_variance_stays_non_negative_far_outside_the_data(self):
        # a + sqrt(c^2 - d^2) >= 0 is imposed on the inner solve, so the
        # fitted slice cannot imply negative variance anywhere.
        k, w = synth()
        fit = fit_svi_slice(k, w)
        far = np.linspace(-5.0, 5.0, 2001)
        assert (svi_total_variance(far, fit["a"], fit["b"], fit["rho"],
                                   fit["m"], fit["sigma"]) >= 0).all()

    def test_noisy_data_still_yields_admissible_parameters(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            k, w = synth()
            noisy = np.maximum(w + rng.normal(0, 0.004, size=len(w)), 1e-6)
            fit = fit_svi_slice(k, noisy)
            if fit["ok"]:
                assert fit["b"] >= 0 and abs(fit["rho"]) <= 1 and fit["sigma"] > 0


class TestThinSlices:
    def test_refuses_fewer_points_than_parameters(self):
        k = np.array([-0.05, 0.0, 0.05])
        fit = fit_svi_slice(k, svi_total_variance(k, **TRUE))
        assert not fit["ok"]
        assert "need" in fit["reason"]

    def test_refuses_just_below_the_threshold(self):
        k = np.linspace(-0.2, 0.2, MIN_POINTS - 1)
        fit = fit_svi_slice(k, svi_total_variance(k, **TRUE))
        assert not fit["ok"]
        assert fit["n"] == MIN_POINTS - 1

    def test_accepts_exactly_the_threshold(self):
        k = np.linspace(-0.2, 0.2, MIN_POINTS)
        fit = fit_svi_slice(k, svi_total_variance(k, **TRUE))
        assert fit["ok"]

    def test_non_finite_and_non_positive_points_are_excluded_from_the_count(self):
        k = np.linspace(-0.2, 0.2, MIN_POINTS + 3)
        w = svi_total_variance(k, **TRUE)
        w[:4] = [np.nan, np.inf, -0.01, 0.0]
        fit = fit_svi_slice(k, w)
        assert not fit["ok"]
        assert fit["n"] == MIN_POINTS - 1


class TestWeights:
    def test_vega_weights_peak_near_the_money(self):
        k = np.linspace(-0.6, 0.6, 41)
        w = svi_total_variance(k, **TRUE)
        weights = vega_weights(k, w)
        assert abs(k[int(np.argmax(weights))]) < 0.12

    def test_vega_weights_decay_into_the_wings(self):
        k = np.array([-0.8, -0.1, 0.0, 0.1, 0.8])
        weights = vega_weights(k, svi_total_variance(k, **TRUE))
        assert weights[0] < weights[1] and weights[4] < weights[3]

    def test_weighting_reduces_the_pull_of_a_bad_wing_quote(self):
        k, w = synth()
        corrupted = w.copy()
        # A 12% error on one deep-OTM quote: small enough to survive the
        # cleaning filters, large enough to move an unweighted fit.
        corrupted[0] *= 1.12
        weighted = fit_svi_slice(k, corrupted)
        flat = fit_svi_slice(k, corrupted, weights=np.ones_like(k))
        clean = svi_total_variance(k, **TRUE)

        def error(fit):
            curve = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                                       fit["m"], fit["sigma"])
            return np.sqrt(np.mean((curve[1:] - clean[1:]) ** 2))

        assert error(weighted) < error(flat)


class TestDegenerateFits:
    def test_kink_degeneracy_is_refused(self):
        # Raw SVI can drive sigma to its floor and b to thousands, matching
        # one bad point with a V-shaped kink. That scores well in least
        # squares and is useless downstream, since Phase 4 and Phase 5 read
        # b and rho directly.
        k, w = synth()
        corrupted = w.copy()
        corrupted[0] *= 1.6
        fit = fit_svi_slice(k, corrupted)
        assert not fit["ok"]
        assert "sigma pinned" in fit["reason"]

    def test_a_wide_smile_is_kept_not_refused(self):
        # sigma large with b large is a wide parabola, not a kink, and fits
        # the curve perfectly well. The real 27Nov26 BTC slice lands here.
        # Rejecting it for having steep asymptotes would discard a slice
        # that matches the market to under 0.2 vol points.
        k, w = synth({**TRUE, "sigma": 0.6, "b": 0.72, "rho": -0.81})
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        curve = svi_total_variance(k, fit["a"], fit["b"], fit["rho"],
                                   fit["m"], fit["sigma"])
        assert np.max(np.abs(curve - w)) < 1e-3

    def test_a_clean_slice_is_never_called_degenerate(self):
        k, w = synth()
        fit = fit_svi_slice(k, w)
        assert fit["ok"]
        assert fit["sigma"] > 0.02

    def test_wing_slope_ratio_is_reported_on_every_fit(self):
        k, w = synth()
        fit = fit_svi_slice(k, w)
        assert fit["wing_slope_ratio"] > 0


class TestSurfaceFit:
    def test_fits_every_expiry_and_keeps_refusals_visible(self):
        slices = {
            0.05: synth(k=np.linspace(-0.1, 0.1, 4)),   # too thin
            0.25: synth(),
            1.00: synth(),
        }
        fits = fit_svi_surface(slices)
        assert set(fits) == {0.05, 0.25, 1.00}
        assert not fits[0.05]["ok"]
        assert fits[0.25]["ok"] and fits[1.00]["ok"]

    def test_results_are_ordered_by_maturity(self):
        slices = {1.0: synth(), 0.1: synth(), 0.5: synth()}
        assert list(fit_svi_surface(slices)) == [0.1, 0.5, 1.0]


class TestFitBudget:
    # The real 11Dec20 slice from the 2020-12-01 archive chain: calls only,
    # vol rising 78% -> 118%, with a 10-point jump between the last put and
    # the first call. Unbudgeted, the fit ran for half an hour and stalled
    # the surface study; the quotes do not determine a smile.
    K = [-0.077537, -0.06359, -0.050335, -0.037089, -0.02385, -0.011039, 0.0019,
         0.014478, 0.026901, 0.039171, 0.051292, 0.063161, 0.075103, 0.086799,
         0.098253, 0.109788, 0.120621, 0.131813, 0.142859, 0.153966, 0.164777]
    W = [0.0171505, 0.0178169, 0.0178155, 0.0181462, 0.018593, 0.0191184,
         0.0237247, 0.0242559, 0.0254726, 0.027253, 0.0279577, 0.0290271,
         0.0303949, 0.0315309, 0.0326834, 0.0339716, 0.035197, 0.0361226,
         0.0374528, 0.0386931, 0.0397388]

    def test_replay_an_undeterminable_slice_is_refused_promptly(self):
        import time

        start = time.time()
        fit = fit_svi_slice(np.array(self.K), np.array(self.W))
        assert not fit["ok"]
        assert "did not settle" in fit["reason"]
        assert time.time() - start < 120
