"""
The risk-neutral density implied by a fitted surface, and probabilities read
off it.

Breeden-Litzenberger (1978): the second derivative of call price in strike
*is* the risk-neutral density of the terminal price. So an option surface is
not only a price quoter -- it is the market's full probability distribution
for where the underlying lands, and this module reads it out.

In log-moneyness k = ln(K/F), with w(k) the total implied variance,

    p(k) = g(k) / sqrt(2*pi*w) * exp(-d2**2 / 2),  d2 = -k/sqrt(w) - sqrt(w)/2

where g is Durrleman's function from `surface/arbitrage.py`. The two modules
share this formula for a reason: g >= 0 is exactly the condition that p is a
density rather than a signed measure, so the butterfly check and the density
are the same statement seen twice.

Sanity check on the algebra: with w constant, g = 1 and p reduces to the
lognormal case -- k is Normal(-w/2, w), which is what risk-neutrality
requires since E[S_T] = F means E[e^k] = 1. The tests assert both that and
that p integrates to one.

**Derivatives are taken numerically here, not analytically.** `arbitrage.py`
differentiates a single SVI slice in closed form, which is exact and fast.
This module reads the *surface*, whose value at a given (k, T) may be an
interpolation between two slices, a clamped wing, or the running maximum
that keeps total variance non-decreasing. None of those has a clean closed
form, and finite differences over the surface capture what the surface
actually serves rather than what one slice would say.

**Where the density can be trusted.** Only inside each slice's trust radius.
Beyond it the surface clamps, so w goes flat, g -> 1, and the tail reverts to
lognormal. That is a deliberate and defensible answer -- it is what a flat
extrapolated vol implies -- but it is an assumption, not information from the
market, and `tail_report` flags when a query depends on it. Deep-tail
probabilities are the least reliable numbers this module produces, which is
unfortunate because they are the most interesting.
"""

from __future__ import annotations

import numpy as np

# Step for the finite differences in k. Large enough to step over the small
# kinks the surface's running-maximum correction can leave, small enough that
# the second derivative of a realistic smile is still resolved.
_H = 1e-3


def _w_derivatives(surface, k, T: float):
    """(w, w', w'') of the surface at log-moneyness k, by central difference."""
    k = np.asarray(k, dtype=float)
    w = surface.total_variance(k, T)
    up = surface.total_variance(k + _H, T)
    down = surface.total_variance(k - _H, T)
    return w, (up - down) / (2 * _H), (up - 2 * w + down) / _H**2


def density_log_moneyness(surface, k, T: float) -> "np.ndarray":
    """
    Risk-neutral density of k = ln(S_T / F), from the surface.

    Integrates to one over k. Negative wherever the surface would imply a
    butterfly arbitrage, which cannot happen inside a slice's trust radius
    but is left visible rather than clipped -- a negative density is a real
    signal that the surface is being asked something it cannot answer.
    """
    w, w1, w2 = _w_derivatives(surface, k, T)
    safe_w = np.maximum(w, 1e-12)
    k = np.asarray(k, dtype=float)
    g = (
        (1.0 - k * w1 / (2.0 * safe_w)) ** 2
        - (w1**2 / 4.0) * (1.0 / safe_w + 0.25)
        + w2 / 2.0
    )
    d2 = -k / np.sqrt(safe_w) - np.sqrt(safe_w) / 2.0
    return g / np.sqrt(2 * np.pi * safe_w) * np.exp(-(d2**2) / 2)


def density_price(surface, prices, T: float) -> "np.ndarray":
    """
    Density in price space. q(S) = p(ln(S/F)) / S -- the Jacobian of the
    change of variable, without which the result integrates to something
    other than one and every probability is wrong by a factor of S.
    """
    prices = np.asarray(prices, dtype=float)
    forward = surface.forward(T)
    k = np.log(prices / forward)
    return density_log_moneyness(surface, k, T) / prices


def _grid(surface, T: float, width: float, points: int):
    k = np.linspace(-width, width, points)
    return k, density_log_moneyness(surface, k, T)


def cdf(surface, price: float, T: float, width: float = 4.0,
        points: int = 4001) -> float:
    """
    P(S_T <= price) under the risk-neutral measure.

    This is not a real-world probability. It is the market's price of a
    digital option, which embeds the risk premium -- the risk-neutral
    probability of a crash is systematically higher than the physical one,
    because people pay to avoid crashes. Worth saying out loud whenever one
    of these numbers is quoted.
    """
    k_target = float(np.log(price / surface.forward(T)))
    k, p = _grid(surface, T, width, points)

    # Integrate to the exact target, not to the nearest grid point below it.
    # Truncating at a grid point drops up to half a cell of mass -- with the
    # density near its peak that was a 0.0016 error on a probability, which
    # is the wrong order for a number meant to be read as a percentage.
    cumulative = np.concatenate(
        ([0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(k)))
    )
    total = cumulative[-1]
    if total <= 0:
        return float("nan")
    # np.interp clamps outside the grid, which is the behaviour wanted here:
    # a target below the grid gives 0 and above it gives the full mass.
    return float(np.interp(k_target, k, cumulative) / total)


def digital_cdf(surface, price: float, T: float) -> float:
    """
    P(S_T <= price) from the smile *at* that strike -- no integration.

    Breeden-Litzenberger's first derivative: the undiscounted put's slope in
    strike is the probability of finishing below it. With total variance
    w(k) the put is Black's, and differentiating through the smile gives

        P(S_T <= K) = N(-d2) + phi(d2) * w'(k) / (2 sqrt(w))

    i.e. the flat-vol digital plus a skew correction. This is the price of a
    tight put spread, which is what the market actually quotes.

    Why it exists alongside `cdf`: `cdf` integrates the density -- the
    *second* derivative -- across the whole k grid and normalises by its
    total. Any error anywhere on that grid leaks into every probability, and
    the clamped wings and the kinks where they meet the fit are exactly where
    the second derivative is worst. Across Track A that made P(below 60k)
    correlate +0.85 with the density's total mass (HANDOFF §5). This one
    reads only w and w' at k: inside a slice's quoted range, nothing outside
    it can move the answer.

    Uses a one-sided difference when a central one would straddle the edge of
    the quoted range, so a strike just inside the edge does not pick up the
    kink where the clamp begins.
    """
    from scipy.stats import norm

    forward = surface.forward(T)
    k = float(np.log(price / forward))
    w = float(surface.total_variance(k, T))
    if w <= 0:
        return float("nan")
    lo, hi = surface.is_extrapolated(price, T)["supported_k_range"]
    if k + _H > hi and k - _H >= lo:
        w1 = (w - float(surface.total_variance(k - _H, T))) / _H
    elif k - _H < lo and k + _H <= hi:
        w1 = (float(surface.total_variance(k + _H, T)) - w) / _H
    else:
        w1 = (float(surface.total_variance(k + _H, T))
              - float(surface.total_variance(k - _H, T))) / (2 * _H)
    sqrt_w = np.sqrt(w)
    d2 = -k / sqrt_w - sqrt_w / 2
    return float(np.clip(norm.cdf(-d2) + norm.pdf(d2) * w1 / (2 * sqrt_w), 0.0, 1.0))


def probability_between(surface, low: float, high: float, T: float,
                        **kwargs) -> float:
    """P(low < S_T <= high)."""
    return cdf(surface, high, T, **kwargs) - cdf(surface, low, T, **kwargs)


def quantile(surface, probability: float, T: float, width: float = 4.0,
             points: int = 4001) -> float:
    """The price below which the risk-neutral measure puts `probability`."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    k, p = _grid(surface, T, width, points)
    cumulative = np.concatenate(([0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(k))))
    total = cumulative[-1]
    if total <= 0:
        return float("nan")
    return float(surface.forward(T) * np.exp(
        np.interp(probability * total, cumulative, k)
    ))


def total_mass(surface, T: float, width: float = 4.0, points: int = 4001) -> float:
    """
    How close the density integrates to one.

    A useful diagnostic rather than a formality: mass far from one means the
    surface is doing something odd -- a clamped wing dominating, or a slice
    whose density goes negative inside the integration range.
    """
    k, p = _grid(surface, T, width, points)
    return float(np.trapezoid(p, k))


def tail_report(surface, T: float, levels: dict | None = None,
                quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """
    Everything worth quoting about the distribution at one maturity.

    `levels` maps a label to a price, e.g. {"below 50k": 50_000}. Each is
    reported with whether it falls inside the trust radius of the slice the
    surface would use, because a tail probability that depends on a clamped
    wing is an assumption rather than a reading.
    """
    forward = surface.forward(T)
    out = {
        "T": float(T),
        "forward": float(forward),
        "total_mass": total_mass(surface, T),
        "quantiles": {q: quantile(surface, q, T) for q in quantiles},
        "levels": {},
    }
    for label, price in (levels or {}).items():
        info = surface.is_extrapolated(price, T)
        below = cdf(surface, price, T)
        out["levels"][label] = {
            "price": float(price),
            "p_below": below,
            "p_above": 1.0 - below,
            "k": info["k"],
            "extrapolated": bool(
                info["outside_supported_range"] or info["beyond_fitted_maturities"]
            ),
        }
    return out
