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
"""

import numpy as np
from scipy.stats import norm


def price(F: float, K: float, T: float, sigma: float, option_type: str) -> float:
    """Black-76 price. option_type is 'C' or 'P'."""
    raise NotImplementedError


def vega(F: float, K: float, T: float, sigma: float) -> float:
    """Black-76 vega (same for calls and puts)."""
    raise NotImplementedError


def _initial_guess(F: float, K: float, T: float, market_price: float) -> float:
    """Brenner-Subrahmanyam approximation: sigma_0 ~= sqrt(2*pi/T) * (price/F)."""
    raise NotImplementedError


def implied_vol(
    F: float, K: float, T: float, market_price: float, option_type: str,
    max_iter: int = 50, tol: float = 1e-6,
) -> float | None:
    """
    Invert Black-76 for implied vol. Returns None if inversion fails
    (caller should drop the quote rather than use a garbage value).
    """
    raise NotImplementedError


def greeks(F: float, K: float, T: float, sigma: float, option_type: str) -> dict:
    """Return delta, gamma, vega, theta. Validate against Deribit's own
    quoted greeks before trusting these for hedging (BUILD_GUIDE.md section 6)."""
    raise NotImplementedError
