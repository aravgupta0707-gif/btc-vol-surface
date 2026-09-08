"""
The queryable surface object: given (K, T), return IV, price, or greeks --
independent of how the surface was built. See BUILD_GUIDE.md section 5.

Interpolation across expiries: linear in TOTAL VARIANCE space (not IV
directly -- variance is the additive quantity over time). Flat-extrapolate
beyond the longest fitted expiry.

Three decisions the guide leaves open, made explicitly here.

**What "flat-extrapolate" means.** Holding total variance constant outside
the fitted range is arbitrage-free but wrong at the short end: as T -> 0 a
constant w implies IV = sqrt(w/T) -> infinity, so a one-hour option would be
quoted at thousands of percent. Holding the variance *rate* w/T constant
instead -- so w scales linearly in T -- keeps implied vol flat at the level
of the nearest fitted slice, tends to w -> 0 as T -> 0 as it must, and is
still non-decreasing in T, so it introduces no calendar arbitrage. That is
what this does at both ends.

**Interpolation happens at fixed log-moneyness, not fixed strike.** Each
slice was fitted against its own forward, so k = ln(K/F) is the coordinate in
which the slices are comparable. A query for (K, T) resolves the forward at T
first, converts to k, and interpolates the two neighbouring slices at that k.

**Queries outside what a slice supports are clamped, not extrapolated.**
Each slice is evaluated only over the narrower of where it was quoted and
where Phase 4 says it stays arbitrage-free, then held flat. Beyond the
trust radius the fitted wing implies a negative risk-neutral density, so a
price from it is not merely uncertain but incoherent; and beyond the quoted
range the wing is pure extrapolation. Extrapolating a narrow slice far
enough will overtake a longer-dated one and manufacture calendar arbitrage
out of two perfectly sound fits -- on a live chain, 18Sep26 quoted over
k in [-0.15, +0.14] reached w = 0.0404 at k = -0.5, above 25Sep26's 0.0308,
while at the money the two were correctly ordered.

Clamping alone does not finish the job, because the quoted ranges differ in
both directions; a running maximum across maturities closes the remainder.
See `_slice_variances`.
"""

from __future__ import annotations

import numpy as np

from pricing.black76 import greeks as black76_greeks
from pricing.black76 import price as black76_price
from surface.arbitrage import trust_radius
from surface.svi import svi_total_variance


class VolSurface:
    def __init__(self, svi_params_by_expiry: dict, forward_by_expiry: dict,
                 quoted_range_by_expiry: dict | None = None,
                 clamp_to_trust_radius: bool = True,
                 enforce_calendar_monotonicity: bool = True):
        """
        svi_params_by_expiry: {T: {'a':, 'b':, 'rho':, 'm':, 'sigma':}}
        forward_by_expiry: {T: forward_price}
        quoted_range_by_expiry: {T: (k_min, k_max)} -- where quotes existed
        """
        if not svi_params_by_expiry:
            raise ValueError("a surface needs at least one fitted slice")
        missing = set(svi_params_by_expiry) - set(forward_by_expiry)
        if missing:
            raise ValueError(f"no forward supplied for T in {sorted(missing)}")

        self.svi_params_by_expiry = svi_params_by_expiry
        self.forward_by_expiry = forward_by_expiry
        self.quoted_range_by_expiry = quoted_range_by_expiry or {}
        self.clamp_to_trust_radius = clamp_to_trust_radius
        self.enforce_calendar_monotonicity = enforce_calendar_monotonicity

        self.maturities = np.array(sorted(svi_params_by_expiry))
        self._log_forwards = np.array(
            [np.log(forward_by_expiry[T]) for T in self.maturities]
        )

        # Each slice is evaluated only over the narrower of (where it was
        # quoted) and (where it stays arbitrage-free), then held flat.
        self._bounds = {}
        for T in self.maturities:
            low, high = -np.inf, np.inf
            radius = trust_radius(svi_params_by_expiry[T])
            if radius["sound_at_the_money"]:
                low, high = radius["k_low"], radius["k_high"]
            quoted = self.quoted_range_by_expiry.get(T)
            if quoted is not None:
                low, high = max(low, quoted[0]), min(high, quoted[1])
            self._bounds[T] = (low, high)

    # ---- forwards -------------------------------------------------------

    def forward(self, T: float) -> float:
        """
        Forward at T, interpolated log-linearly.

        A forward curve is a level times an accumulated carry, so it is the
        logarithm that behaves linearly in T, not the price. Held flat beyond
        the fitted range rather than extrapolating a carry rate off the end.
        """
        T = float(T)
        if len(self.maturities) == 1:
            return float(np.exp(self._log_forwards[0]))
        clamped = np.clip(T, self.maturities[0], self.maturities[-1])
        return float(np.exp(np.interp(clamped, self.maturities, self._log_forwards)))

    # ---- variance -------------------------------------------------------

    def _slice_variance(self, k, T_slice: float) -> "np.ndarray":
        params = self.svi_params_by_expiry[T_slice]
        k = np.asarray(k, dtype=float)
        if self.clamp_to_trust_radius:
            low, high = self._bounds[T_slice]
            k = np.clip(k, low, high)
        return svi_total_variance(
            k, params["a"], params["b"], params["rho"],
            params["m"], params["sigma"],
        )

    def total_variance(self, k, T: float) -> "np.ndarray":
        """
        Total implied variance at log-moneyness k and maturity T.

        Inside the fitted range this is a linear interpolation in T between
        the two neighbouring slices, which is the only place interpolation is
        theoretically safe: total variance is what accumulates additively
        over time, so a linear blend of two arbitrage-free slices stays
        non-decreasing in T. Interpolating implied vol instead can manufacture
        calendar arbitrage from two perfectly sound slices.
        """
        T = float(T)
        if T <= 0:
            return np.zeros_like(np.asarray(k, dtype=float))

        maturities = self.maturities
        slices = self._slice_variances(k)

        if len(maturities) == 1 or T <= maturities[0] or T >= maturities[-1]:
            edge = 0 if T <= maturities[0] else len(maturities) - 1
            # Hold the variance rate, not total variance: keeps IV flat and
            # sends w -> 0 as T -> 0. See the module docstring.
            return slices[edge] * (T / float(maturities[edge]))

        index = int(np.searchsorted(maturities, T))
        T1, T2 = float(maturities[index - 1]), float(maturities[index])
        weight = (T - T1) / (T2 - T1)
        return slices[index - 1] + weight * (slices[index] - slices[index - 1])

    def _slice_variances(self, k) -> list:
        """
        Every slice evaluated at k, made non-decreasing in T if required.

        Clamping each slice to its own quoted range is not by itself enough
        to keep total variance non-decreasing, because the ranges differ: on
        a live chain 30Oct26 was quoted out to k = -0.56 while 27Nov26
        stopped at -0.31, so at k = -0.5 the shorter expiry's genuine value
        was compared against the longer one's clamped value and came out
        higher by 0.0067 -- a calendar arbitrage produced entirely by the
        shorter slice having wider quotes.

        Taking a running maximum across maturities lifts the longer slice to
        at least the shorter one's level, which is the smallest change that
        restores the no-arbitrage condition. It is safe to do because it
        never binds where the market actually quotes: measured over the
        overlap of every adjacent pair's quoted ranges, the worst crossing is
        exactly zero. The correction only ever touches extrapolated wings.
        """
        values = [self._slice_variance(k, float(T)) for T in self.maturities]
        if not self.enforce_calendar_monotonicity:
            return values

        adjusted, running = [], None
        for w in values:
            running = w if running is None else np.maximum(w, running)
            adjusted.append(running)
        return adjusted

    # ---- the public interface ------------------------------------------

    def iv(self, K: float, T: float) -> float:
        """Implied vol at (K, T), interpolating in total variance space
        if T doesn't match a fitted expiry exactly."""
        T = float(T)
        if T <= 0:
            raise ValueError("T must be positive")
        k = np.log(np.asarray(K, dtype=float) / self.forward(T))
        w = np.maximum(self.total_variance(k, T), 0.0)
        result = np.sqrt(w / T)
        return float(result) if result.ndim == 0 else result

    def price(self, K: float, T: float, option_type: str) -> float:
        """Black-76 price in USD, using the surface's own vol and forward."""
        return black76_price(
            self.forward(T), K, T, self.iv(K, T), option_type
        )

    def greeks(self, K: float, T: float, option_type: str) -> dict:
        """
        Black-76 greeks in USD terms, plus the inverse-settled equivalents.

        Conventions, measured against Deribit's own quoted greeks rather than
        assumed -- and the result is the opposite of what the inverse
        settlement suggests:

        - **delta** is the plain USD Black-76 delta, dV_usd/dF. It matches
          Deribit's to a median of 0.0027 across liquid strikes. Recomputing
          with their `mark_iv` instead of ours does not shrink that residual,
          so it is not an implied-vol difference and the convention is
          confirmed. The coin-adjusted delta is 26x further away.
        - **vega** is per 1.00 of vol; Deribit quotes per 1%. The measured
          ratio is 100.01, so divide by 100 to compare.
        - **gamma** agrees within about 1%.

        `delta_coin` is still returned and still matters, but not for
        comparison with Deribit -- for hedging. See `inverse_greeks`.
        """
        forward = self.forward(T)
        sigma = self.iv(K, T)
        out = black76_greeks(forward, K, T, sigma, option_type)
        out["iv"] = sigma
        out["forward"] = forward
        out["price"] = black76_price(forward, K, T, sigma, option_type)
        out.update(inverse_greeks(out["price"], out["delta"], out["gamma"], forward))
        return out

    def is_extrapolated(self, K: float, T: float) -> dict:
        """
        Whether a query falls outside what the fit actually supports, and how.

        A price is still returned in every case, but a caller marking
        positions to market should know when it came from a clamped wing or
        from beyond the longest fitted expiry.
        """
        T = float(T)
        k = float(np.log(K / self.forward(T)))
        beyond_T = T < self.maturities[0] or T > self.maturities[-1]
        nearest = float(self.maturities[int(np.argmin(np.abs(self.maturities - T)))])
        low, high = self._bounds[nearest]
        clamped_k = not (low <= k <= high)
        return {
            "k": k,
            "beyond_fitted_maturities": bool(beyond_T),
            "outside_supported_range": bool(clamped_k),
            "supported_k_range": (float(low), float(high)),
            "nearest_T": nearest,
        }


def inverse_greeks(price_usd: float, delta_usd: float, gamma_usd: float,
                   forward: float) -> dict:
    """
    Convert USD-denominated greeks to the coin-settled convention.

    A Deribit BTC option's premium is quoted, margined and settled in BTC:
    V_coin = V_usd / F. So the sensitivity of what is actually paid is

        dV_coin/dF = (dV_usd/dF) / F  -  V_usd / F**2
                   = (delta_usd - V_coin) / F

    The second term is what makes an inverse option different: the payoff is
    converted to coin at the settlement price, so the position's coin value
    depends on the price twice, and a hedge sized on delta_usd alone leaves
    that part unhedged.

    Note what this is NOT for. BUILD_GUIDE.md section 6 implies your delta
    will disagree with Deribit's until the inverse settlement is handled;
    measured, that is backwards. Deribit publishes the ordinary dimensionless
    delta and our unadjusted number matches it to 0.0027, while this
    coin-adjusted one is 26x further off. So compare with `delta`, and hedge
    a coin-margined book with `delta_coin`. They answer different questions.
    """
    price_coin = price_usd / forward
    return {
        "price_coin": float(price_coin),
        "delta_coin": float((delta_usd - price_coin) / forward),
        "gamma_coin": float(
            (gamma_usd * forward - 2 * (delta_usd - price_coin)) / forward**2
        ),
    }


def from_fits(fits: dict, forwards: dict | None = None,
              quoted_ranges: dict | None = None, **kwargs) -> VolSurface:
    """
    Build a surface from `surface.build.build_slices` output.

    Each fit already carries the forward it was calibrated against and the
    span of log-moneyness it was quoted over, so both maps are optional;
    supplying them overrides what the fits carry.
    """
    params = {
        T: {key: fit[key] for key in ("a", "b", "rho", "m", "sigma")}
        for T, fit in fits.items()
    }
    if forwards is None:
        forwards = {T: fit["forward"] for T, fit in fits.items()}
    if quoted_ranges is None:
        quoted_ranges = {
            T: fit["k_range"] for T, fit in fits.items() if "k_range" in fit
        }
    return VolSurface(params, forwards, quoted_ranges, **kwargs)
