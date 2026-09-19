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

**How the fit is actually run.** Minimising all five parameters at once is a
non-convex problem with well-known local minima: the wings trade off against
`m` and `sigma`, so a single descent from one starting point lands somewhere
arbitrary, and you get parameters that jump between adjacent expiries for no
market reason. The guide's advice to plot the fitted parameters across
expiries is really a way of noticing this after the fact.

Instead this uses the quasi-explicit reduction of De Marco and Martini. Set
y = (k - m) / sigma; then

    w = a + (b*rho*sigma) * y + (b*sigma) * sqrt(y**2 + 1)
      = a + d*y + c*sqrt(y**2 + 1)

which is **linear** in (a, d, c) once (m, sigma) are fixed. So the inner
three parameters are solved exactly by constrained least squares, and only
(m, sigma) are searched numerically. A five-dimensional non-convex problem
becomes a two-dimensional one with an exact inner solution, which is both
far more robust and much faster. Recovering the raw parameters is
b = c/sigma and rho = d/c.

Constraints imposed on the inner solve:
- c >= 0                       (b >= 0)
- |d| <= c                     (|rho| <= 1)
- a + sqrt(c**2 - d**2) >= 0   (total variance non-negative everywhere;
                                the minimum of w sits at that value)

Deliberately *not* imposed: the no-butterfly bound b(1 + |rho|) <= 4/T.
Phase 4 exists to find those violations and record them, and pre-constraining
the fit would hide the thing the project is meant to demonstrate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Five parameters. Fewer than about eight points and the fit is interpolation
# wearing a parameterisation, so it is refused rather than reported -- see
# `fit_svi_slice`. Front-week BTC expiries routinely fall below this.
MIN_POINTS = 8

# Raw SVI has a well-known degeneracy: as sigma -> 0 with b*sigma held
# finite, sqrt(y^2 + 1) -> |y| and the smile collapses to a V-shaped kink.
# The inner solve is happy there -- it is still linear -- but b = c/sigma
# explodes. Left at a 1e-3 floor, one corrupted wing quote produced
# b = 267,905 with rho = -0.99999997: a fit that scores marginally better in
# least squares and is useless for anything downstream, since Phase 4's
# arbitrage checks and Phase 5's interpolation both read b and rho directly.
# A floor of 0.01 is far below any real smile's curvature and rules it out.
_SIGMA_BOUNDS = (1e-2, 5.0)
_M_PAD = 0.5

# Below this the fit is exact to numerical precision and further starts
# cannot improve it. Skipping them makes clean slices roughly ten times
# faster without changing any result.
_CONVERGED = 1e-14

# A fit that keeps hitting the constrained fallback is not converging on a
# smile; it is being pushed along a constraint boundary because the quotes
# do not pin one down. Measured over 208 slices on 24 archive days, a normal
# fit needs at most 167 fallback solves (median 2). The 11Dec20 slice of
# 2020-12-01 -- calls only, vol rising 78% -> 118% with a 10-point jump
# between the last put and the first call -- sent 84% of evaluations down
# the fallback at 83 ms each and would have run for half an hour, stalling a
# whole study. Three times the worst normal fit, then refuse.
_MAX_FALLBACKS = 500


class _FitBudgetExceeded(Exception):
    pass

# Reported on every fit as `wing_slope_ratio`: how much steeper the fitted
# asymptotic wings are than the steepest slope the quotes actually show.
# A diagnostic for how far a slice can be trusted outside its quoted range,
# not a rejection criterion -- see the note in `fit_svi_slice`.
WIDE_WING_RATIO = 10.0


def svi_total_variance(
    k: "np.ndarray", a: float, b: float, rho: float, m: float, sigma: float
) -> "np.ndarray":
    """Raw SVI total variance at log-moneyness k."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_implied_vol(k, T: float, params: dict) -> "np.ndarray":
    """Implied vol from a fitted slice: sigma_BS = sqrt(w / T)."""
    w = svi_total_variance(
        k, params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    )
    return np.sqrt(np.maximum(w, 0.0) / T)


def _design(y: "np.ndarray") -> "np.ndarray":
    """Columns of the inner linear problem: [1, y, sqrt(y^2+1)]."""
    return np.column_stack([np.ones_like(y), y, np.sqrt(y**2 + 1.0)])


def _feasible(a: float, d: float, c: float, tol: float = 1e-9) -> bool:
    return c >= -tol and abs(d) <= c + tol and a + np.sqrt(
        max(c**2 - d**2, 0.0)
    ) >= -tol


def _solve_inner(
    y: "np.ndarray", w: "np.ndarray", weights: "np.ndarray",
    fallbacks: list | None = None,
) -> tuple[np.ndarray, float]:
    """
    Exact least-squares solution for (a, d, c) at fixed (m, sigma).

    Tries the unconstrained normal equations first, which is usually feasible
    and costs one small solve; only falls back to a constrained optimiser
    when that solution violates a bound.
    """
    design = _design(y)
    root_w = np.sqrt(weights)
    scaled_design = design * root_w[:, None]
    scaled_target = w * root_w

    solution, *_ = np.linalg.lstsq(scaled_design, scaled_target, rcond=None)
    if _feasible(*solution):
        residual = scaled_design @ solution - scaled_target
        return solution, float(residual @ residual)

    if fallbacks is not None:
        fallbacks.append(1)
        if len(fallbacks) > _MAX_FALLBACKS:
            raise _FitBudgetExceeded

    def cost(p):
        residual = scaled_design @ p - scaled_target
        return float(residual @ residual)

    constraints = [
        {"type": "ineq", "fun": lambda p: p[2]},                       # c >= 0
        {"type": "ineq", "fun": lambda p: p[2] - abs(p[1])},           # |d| <= c
        {"type": "ineq",
         "fun": lambda p: p[0] + np.sqrt(max(p[2] ** 2 - p[1] ** 2, 0.0))},
    ]
    start = np.array([max(float(np.min(w)), 0.0), 0.0, max(solution[2], 1e-6)])
    result = minimize(cost, start, method="SLSQP", constraints=constraints,
                      options={"maxiter": 200, "ftol": 1e-12})
    return result.x, float(result.fun)


def vega_weights(log_moneyness, total_variance) -> "np.ndarray":
    """
    Weights proportional to Black-76 vega, normalised to mean one.

    Vega peaks at the money and decays exponentially into the wings, which is
    exactly the weighting the fit wants: ATM quotes are the liquid, reliable
    ones, and deep-OTM quotes are noisy and should not drag the fit. Computed
    from total variance directly, so it needs no forward or strike --
    d1 = -k/sqrt(w) + sqrt(w)/2, and vega is proportional to the standard
    normal density at d1.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.maximum(np.asarray(total_variance, dtype=float), 1e-12)
    root = np.sqrt(w)
    d1 = -k / root + root / 2.0
    weights = np.exp(-0.5 * d1**2)
    total = weights.sum()
    return weights * (len(weights) / total) if total > 0 else np.ones_like(k)


def fit_svi_slice(
    log_moneyness: "np.ndarray",
    total_variance: "np.ndarray",
    weights: "np.ndarray | None" = None,
) -> dict:
    """
    Fit one expiry slice. Returns {'a':, 'b':, 'rho':, 'm':, 'sigma':, 'rmse':}.

    `weights` defaults to vega weighting. `rmse` is the weighted root mean
    squared error in total variance; `rmse_vol` converts it to vol points at
    the slice's own maturity when `T` is supplied via `fit_svi_slice_for`.

    Returns `{'ok': False, 'reason': ...}` rather than a fit when there are
    too few points. Five parameters through six observations is not a
    calibration and reporting it as one would be the dishonest option.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    finite = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[finite], w[finite]

    if len(k) < MIN_POINTS:
        return {"ok": False, "n": int(len(k)),
                "reason": f"{len(k)} usable points, need {MIN_POINTS}"}

    weights = (
        vega_weights(k, w) if weights is None
        else np.asarray(weights, dtype=float)[finite]
    )

    fallbacks: list = []

    def outer_cost(params) -> float:
        m, sigma = params
        if not (_SIGMA_BOUNDS[0] <= sigma <= _SIGMA_BOUNDS[1]):
            return 1e12
        _, cost = _solve_inner((k - m) / sigma, w, weights, fallbacks)
        return cost

    # Multi-start over (m, sigma). The outer surface is smooth but not
    # convex, and starting only from m=0 biases the fit toward symmetric
    # smiles -- which BTC's, with its persistent skew, are not.
    spread = float(k.max() - k.min()) or 0.1
    starts = [
        (m0, s0)
        for m0 in (-spread / 3, 0.0, spread / 3)
        for s0 in (0.05, 0.2, 0.5)
    ]
    bounds = [
        (float(k.min()) - _M_PAD, float(k.max()) + _M_PAD),
        _SIGMA_BOUNDS,
    ]

    try:
        best, _ = _multistart(outer_cost, starts, bounds)
    except _FitBudgetExceeded:
        return {
            "ok": False, "n": int(len(k)),
            "reason": (f"did not settle: over {_MAX_FALLBACKS} constrained solves "
                       f"(a normal fit needs under 170) -- the quotes do not "
                       f"determine a smile"),
        }
    return _finish(k, w, weights, best)


def _multistart(outer_cost, starts, bounds) -> tuple:
    best, best_cost = None, np.inf
    for start in starts:
        # Nelder-Mead rather than L-BFGS-B. The inner solve switches between
        # the unconstrained normal equations and an SLSQP fallback when a
        # bound binds (about 4% of the (m, sigma) plane), which puts small
        # discontinuities in the outer objective. A gradient method
        # differences straight across them and gets meaningless steps; a
        # simplex method does not care.
        result = minimize(outer_cost, np.array(start), method="Nelder-Mead",
                          bounds=bounds,
                          options={"xatol": 1e-10, "fatol": 1e-16,
                                   "maxiter": 800})
        if result.fun < best_cost:
            best, best_cost = result.x, float(result.fun)
        if best_cost < _CONVERGED:
            break
    return best, best_cost


def _finish(k, w, weights, best) -> dict:
    """The exact inner solve at the best (m, sigma), and the refusal checks."""
    m, sigma = float(best[0]), float(best[1])
    (a, d, c), cost = _solve_inner((k - m) / sigma, w, weights)
    b = c / sigma
    rho = float(np.clip(d / c, -1.0, 1.0)) if c > 1e-12 else 0.0

    fitted = svi_total_variance(k, a, b, rho, m, sigma)
    residual = fitted - w

    # Refuse the kink degeneracy, and only that. A solution sitting on the
    # sigma bound is an artefact of where the search was stopped, not a fit
    # -- the same reasoning that rejects a Brent root landing on a bracket
    # endpoint in pricing/black76.py.
    #
    # Large b alone is NOT the signal. There are two distinct regimes and
    # they need telling apart. Measured on live data:
    #
    #   kink        sigma = 0.010000 (the floor), b = 2680, rho = -0.99995
    #   wide smile  sigma = 0.604,               b = 0.72,  rho = -0.81
    #
    # The first is a V-shape chasing one corrupted quote. The second is the
    # genuine 27Nov26 slice, which fits the market to 0.169 vol points --
    # entirely usable. An earlier version of this guard compared the fitted
    # wing slope against the steepest observed slope and threw the good
    # slice away for being 11x steeper.
    # Relative, not absolute: Nelder-Mead's bound handling returns values a
    # hair inside the bound (0.010000001432 against a floor of 0.01), so an
    # absolute 1e-9 tolerance misses the very case it exists to catch. The
    # margin against a genuine fit is enormous either way -- real slices come
    # back with sigma from 0.014 to 0.60.
    if sigma < _SIGMA_BOUNDS[0] * 1.001:
        return {
            "ok": False, "n": int(len(k)),
            "reason": (f"degenerate: sigma pinned at its floor "
                       f"{_SIGMA_BOUNDS[0]} with b={b:.4g}, rho={rho:+.4f} "
                       f"-- a kink, not a smile"),
        }

    # Reported, never fatal. A wide smile extrapolates far more steeply than
    # the quoted range shows, which matters for Phase 5's extrapolation and
    # for the k-grid Phase 4 checks over, but says nothing about fit quality
    # inside the data.
    order = np.argsort(k)
    observed_slope = float(np.max(np.abs(np.diff(w[order]) / np.diff(k[order]))))
    wing_slope = b * (1 + abs(rho))

    return {
        "ok": True,
        "a": float(a), "b": float(b), "rho": rho,
        "m": m, "sigma": sigma,
        "n": int(len(k)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rmse_weighted": float(np.sqrt(cost / weights.sum())),
        "max_abs_error": float(np.max(np.abs(residual))),
        "wing_slope": float(wing_slope),
        "wing_slope_ratio": (
            float(wing_slope / observed_slope) if observed_slope > 0 else np.inf
        ),
    }


def fit_svi_surface(
    slices: dict[float, tuple["np.ndarray", "np.ndarray"]]
) -> dict[float, dict]:
    """
    Fit every expiry independently. `slices` maps T (years) to
    (log_moneyness, total_variance). Slices with too few points are returned
    with ok=False rather than dropped, so the caller can see what was refused.
    """
    return {
        T: fit_svi_slice(k, w) for T, (k, w) in sorted(slices.items())
    }
