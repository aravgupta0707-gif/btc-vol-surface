"""
Raw SVI (Gatheral) calibration, fit independently per expiry.

w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))

where k = ln(K/F) is log-moneyness and w is total implied variance
(sigma_BS**2 * T).

Parameter meanings (BUILD_GUIDE.md section 3.1):
- a: overall variance level
- b: steepness of both wings (b >= 0)
- rho: skew / rotation (-1 < rho < 1)
- m: horizontal shift
- sigma: ATM curvature (sigma > 0)

Fit via vega-weighted least squares, L-BFGS-B, bounds as above.
"""

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k: "np.ndarray", a: float, b: float, rho: float, m: float, sigma: float) -> "np.ndarray":
    raise NotImplementedError


def fit_svi_slice(
    log_moneyness: "np.ndarray",
    total_variance: "np.ndarray",
    weights: "np.ndarray | None" = None,
) -> dict:
    """
    Fit one expiry slice. Returns {'a':, 'b':, 'rho':, 'm':, 'sigma':, 'rmse':}.

    TODO:
    - initial guess: a ~= min(total_variance), b ~= 0.1, rho ~= 0, m ~= 0, sigma ~= 0.1
    - bounds: b >= 0, -1 < rho < 1, sigma > 0
    - weight ATM points more heavily than deep OTM
    """
    raise NotImplementedError
