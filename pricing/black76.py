"""
Black-76 (forward-based) option pricing and implied vol inversion.

Why Black-76 instead of spot Black-Scholes: BTC options are economically
closer to forward-priced instruments once you account for the
cost-of-carry embedded in perpetual futures funding rates. See
BUILD_GUIDE.md section 1.3.

Inversion strategy (BUILD_GUIDE.md section 2):
1. Initial guess via Brenner-Subrahmanyam approximation
2. Newton-Raphson using vega as the derivative
3. Fall back to Brent's method with bracket [0.01, 5.0] on Newton failure
4. Drop the quote if neither converges

Three conventions this module fixes, all of which have to match what
data/clean.py produces or the numbers are meaningless:

- **Prices are USD.** data/clean.py has already multiplied the BTC-quoted
  premium by that expiry's forward. Passing a BTC premium here against a
  USD strike will not bracket.

- **No discounting.** Black-76 is normally exp(-rT) * [F N(d1) - K N(d2)],
  but the forward already carries the financing, and Deribit reports
  interest_rate 0.0 on every instrument. Checked empirically: pricing
  undiscounted reproduces Deribit's own mark_iv to a median of 0.015 vol
  points across a live chain. A `discount_rate` argument is available for
  completeness but defaults to zero.

- **Greeks are with respect to the forward, per unit vol, per year.** So
  vega is the change for a 1.00 (i.e. 100 vol point) move, not 1%. These
  are textbook Black-76 greeks; converting them to Deribit's own
  conventions -- delta in BTC terms, because these are inverse,
  BTC-settled options -- is Phase 5's problem, not this module's.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# The guide suggests bracketing Brent on [0.01, 5.0]. The upper bound is
# right -- 500% annualized is far above anything BTC has printed. The lower
# bound is loosened to 0.1%, since a quote implying under 1% vol is one we
# want to see and reject on its own merits rather than have the bracket
# silently exclude.
VOL_BRACKET = (1e-3, 5.0)
MAX_VOL_STEP = 0.5

# An option's vol lives entirely in its time value -- the part of the price
# above intrinsic. When that is vanishingly small the objective is flat
# across the whole bracket and the root finder will happily return the
# bracket endpoint: a 130k-strike one-day call worth 1e-9 dollars inverted to
# 0.001, reporting 0.1% vol for an option actually priced at 90%. That is the
# silent-garbage outcome this module exists to avoid, so anything whose time
# value is below this fraction of the forward is refused outright.
MIN_TIME_VALUE_FRAC = 1e-8

# A root sitting on a bracket endpoint is an artefact of the objective being
# flat there, not a real solution.
BRACKET_EDGE_TOL = 1e-6


def _d1_d2(F, K, T, sigma):
    variance = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * variance**2) / variance
    return d1, d1 - variance


def intrinsic(F, K, option_type: str):
    """Undiscounted intrinsic value, which is also the T -> 0 price limit."""
    return np.maximum(F - K, 0.0) if option_type == "C" else np.maximum(K - F, 0.0)


def price(
    F: float, K: float, T: float, sigma: float, option_type: str,
    discount_rate: float = 0.0,
) -> float:
    """
    Black-76 price in the same currency as F and K. option_type is 'C' or 'P'.

    Accepts scalars or numpy arrays for F, K, T and sigma; option_type stays
    a single character, so price calls and puts in separate groups.
    """
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    degenerate = (T <= 0) | (sigma <= 0)
    # Evaluate the closed form on safe inputs, then substitute the intrinsic
    # value wherever the inputs were degenerate. Computing it this way keeps
    # the function vectorised without tripping log/divide warnings.
    safe_T = np.where(degenerate, 1.0, T)
    safe_sigma = np.where(degenerate, 1.0, sigma)
    d1, d2 = _d1_d2(F, K, safe_T, safe_sigma)

    if option_type == "C":
        value = F * norm.cdf(d1) - K * norm.cdf(d2)
    elif option_type == "P":
        value = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'C' or 'P', got {option_type!r}")

    value = np.where(degenerate, intrinsic(F, K, option_type), value)
    if discount_rate:
        value = value * np.exp(-discount_rate * np.maximum(T, 0.0))
    return value if value.ndim else float(value)


def vega(F: float, K: float, T: float, sigma: float) -> float:
    """
    Black-76 vega, identical for calls and puts.

    Per unit of vol: multiply by 0.01 for the change per vol point. This is
    the derivative Newton-Raphson uses, and its collapse toward zero for
    short-dated deep-OTM options is exactly why the inversion needs a
    fallback -- the objective goes flat and the Newton step overshoots.
    """
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    degenerate = (T <= 0) | (sigma <= 0)
    safe_T = np.where(degenerate, 1.0, T)
    safe_sigma = np.where(degenerate, 1.0, sigma)
    d1, _ = _d1_d2(F, K, safe_T, safe_sigma)
    value = np.where(degenerate, 0.0, F * norm.pdf(d1) * np.sqrt(safe_T))
    return value if value.ndim else float(value)


def _initial_guess(F: float, K: float, T: float, market_price: float) -> float:
    """
    Brenner-Subrahmanyam approximation: sigma_0 ~= sqrt(2*pi/T) * (price/F).

    Exact at the money and good enough there that Newton converges in two or
    three steps. It degrades in the wings, which is what the fallback covers.
    """
    if T <= 0 or F <= 0:
        return 0.5
    guess = np.sqrt(2 * np.pi / T) * (market_price / F)
    return float(np.clip(guess, *VOL_BRACKET))


def _no_arbitrage_bounds(F: float, K: float, option_type: str) -> tuple[float, float]:
    """
    Undiscounted price bounds. Outside these no implied vol exists at all,
    so the quote is bad and should be dropped rather than forced.
    """
    if option_type == "C":
        return max(F - K, 0.0), F
    return max(K - F, 0.0), K


def implied_vol(
    F: float, K: float, T: float, market_price: float, option_type: str,
    max_iter: int = 50, tol: float = 1e-6,
) -> float | None:
    """
    Invert Black-76 for implied vol. Returns None if inversion fails
    (caller should drop the quote rather than use a garbage value).

    Deep in-the-money quotes are converted to their out-of-the-money
    counterpart through put-call parity first (with no discounting,
    C - P = F - K). The OTM side carries all the vol information and almost
    none of the intrinsic value, so the objective is far better conditioned
    there -- inverting a price that is 99% intrinsic asks the solver to
    resolve vol from the last 1% of the number.
    """
    if not np.isfinite([F, K, T, market_price]).all() or T <= 0 or F <= 0 or K <= 0:
        return None

    if option_type == "C" and K < F:
        market_price, option_type = market_price - (F - K), "P"
    elif option_type == "P" and K > F:
        market_price, option_type = market_price + (F - K), "C"

    low, high = _no_arbitrage_bounds(F, K, option_type)
    if market_price < low - 1e-9 or market_price > high + 1e-9:
        return None
    if market_price - low <= MIN_TIME_VALUE_FRAC * F:
        return None  # no resolvable time value, so no recoverable vol

    def objective(sigma: float) -> float:
        return float(price(F, K, T, sigma, option_type)) - market_price

    sigma = _initial_guess(F, K, T, market_price)
    for _ in range(max_iter):
        difference = objective(sigma)
        if abs(difference) < tol:
            return float(sigma)
        derivative = float(vega(F, K, T, sigma))
        if derivative < 1e-12:
            break  # flat objective; Newton cannot make progress
        step = difference / derivative
        # Cap the step. Without this a tiny vega throws sigma far negative or
        # to absurd values on the first iteration and the loop never recovers.
        sigma -= float(np.clip(step, -MAX_VOL_STEP, MAX_VOL_STEP))
        if not np.isfinite(sigma) or sigma <= VOL_BRACKET[0] or sigma >= VOL_BRACKET[1]:
            break

    try:
        low_value, high_value = objective(VOL_BRACKET[0]), objective(VOL_BRACKET[1])
        if low_value * high_value > 0:
            return None  # no root inside the bracket
        root = float(brentq(objective, *VOL_BRACKET, xtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return None

    if (
        root <= VOL_BRACKET[0] + BRACKET_EDGE_TOL
        or root >= VOL_BRACKET[1] - BRACKET_EDGE_TOL
    ):
        return None
    return root


def implied_vol_many(F, K, T, market_price, option_type) -> np.ndarray:
    """
    Invert a whole column of quotes, returning NaN where inversion failed.

    `option_type` may be a single character or a per-quote sequence.
    """
    # Broadcast so a scalar forward or a single maturity can be paired with a
    # column of strikes without the caller having to tile them by hand.
    F, K, T, market_price = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, dtype=float))
          for x in (F, K, T, market_price))
    )
    types = (
        [option_type] * len(F) if isinstance(option_type, str)
        else list(option_type)
    )
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        result = implied_vol(F[i], K[i], T[i], market_price[i], types[i])
        if result is not None:
            out[i] = result
    return out


def greeks(F: float, K: float, T: float, sigma: float, option_type: str) -> dict:
    """Return delta, gamma, vega, theta. Validate against Deribit's own
    quoted greeks before trusting these for hedging (BUILD_GUIDE.md section 6).

    Forward-based and undiscounted, matching `price`. Delta is d(price)/dF,
    vega is per unit vol, theta is per year (divide by 365 for per day).
    """
    if T <= 0 or sigma <= 0:
        step = 0.0 if F == K else (1.0 if F > K else 0.0)
        delta = step if option_type == "C" else step - 1.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1, d2 = _d1_d2(F, K, T, sigma)
    delta = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1.0
    return {
        "delta": float(delta),
        "gamma": float(norm.pdf(d1) / (F * sigma * np.sqrt(T))),
        "vega": float(F * norm.pdf(d1) * np.sqrt(T)),
        # With no discounting the carry terms drop out and theta is the same
        # for calls and puts: pure time decay of the option's time value.
        "theta": float(-F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))),
    }
