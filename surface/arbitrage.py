"""
Calendar and butterfly arbitrage checks for a fitted SVI surface.

See BUILD_GUIDE.md section 4. Log every violation found and how it was
resolved -- this is the most valuable part of the project to be able
to talk through in an interview.
"""


def check_calendar_arbitrage(svi_params_by_expiry: dict, k_grid: "np.ndarray") -> list[dict]:
    """
    At each k in k_grid, total variance w(k, T) must be non-decreasing in T
    across expiries. Return a list of violations (k, T1, T2, w1, w2).
    """
    raise NotImplementedError


def check_butterfly_arbitrage(svi_params: dict, k_grid: "np.ndarray") -> list[dict]:
    """
    Durrleman's condition: g(k) must be >= 0 everywhere for the fitted
    slice's implied density to be non-negative (no butterfly arbitrage).
    Return a list of k values where g(k) < 0.

    See Gatheral's SVI paper for the g(k) formula in terms of w, w', w''.
    """
    raise NotImplementedError
