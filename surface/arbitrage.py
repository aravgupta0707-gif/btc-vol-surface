"""
Calendar and butterfly arbitrage checks for a fitted SVI surface.

See BUILD_GUIDE.md section 4. Log every violation found and how it was
resolved -- this is the most valuable part of the project to be able
to talk through in an interview.

**Calendar.** At fixed log-moneyness, total variance must be non-decreasing
in T. A longer-dated option cannot imply less accumulated uncertainty than a
shorter-dated one; if it does, buy the long and sell the short and the
calendar spread is free money.

**Butterfly (Durrleman).** The fitted price surface must be convex in strike.
By Breeden-Litzenberger the second derivative of price in strike *is* the
risk-neutral density, so convexity failing means the surface implies negative
probability over some range of terminal prices -- not an odd distribution, an
incoherent one. Every price quoted from it is meaningless.

In terms of total variance, the condition is g(k) >= 0 where

    g(k) = (1 - k*w'/(2w))**2 - (w'**2 / 4) * (1/w + 1/4) + w''/2

and the density itself is

    p(k) = g(k) / sqrt(2*pi*w) * exp(-d2**2 / 2),   d2 = -k/sqrt(w) - sqrt(w)/2

so g is exactly the factor that can turn the density negative. For raw SVI
the derivatives are closed form, so no numerical differentiation is needed:

    w'(k)  = b * (rho + (k-m)/sqrt((k-m)**2 + sigma**2))
    w''(k) = b * sigma**2 / ((k-m)**2 + sigma**2)**1.5

Note that w'' > 0 always when b > 0, so an SVI slice is convex in k. Convex
in k is not the same as convex in strike, which is why g(k) still has to be
checked rather than assumed.

**Inside versus outside the quoted range.** Violations are reported with a
flag for whether they occur where quotes actually existed. A violation among
the traded strikes is a real problem with the fit or the data. One far out in
the extrapolated wings says the slice should not be trusted that far out --
worth knowing, and worth constraining in Phase 5, but a different and much
less serious finding. Reporting them undifferentiated would overstate the
problem.
"""

from __future__ import annotations

import numpy as np

# Gatheral's necessary condition for no butterfly arbitrage in the wings.
# Reported alongside the pointwise check as a second opinion.
def wing_bound(params: dict, T: float) -> dict:
    """b*(1 + |rho|) <= 4/T is necessary for the wings to be arbitrage-free."""
    lhs = params["b"] * (1 + abs(params["rho"]))
    return {"lhs": float(lhs), "rhs": 4.0 / T, "ok": bool(lhs <= 4.0 / T)}


def svi_derivatives(k, params: dict) -> tuple:
    """Return (w, w', w'') for a raw SVI slice, all in closed form."""
    k = np.asarray(k, dtype=float)
    a, b, rho = params["a"], params["b"], params["rho"]
    m, sigma = params["m"], params["sigma"]
    u = k - m
    root = np.sqrt(u**2 + sigma**2)
    w = a + b * (rho * u + root)
    w1 = b * (rho + u / root)
    w2 = b * sigma**2 / root**3
    return w, w1, w2


def durrleman_g(k, params: dict) -> "np.ndarray":
    """
    Durrleman's function. g(k) >= 0 everywhere is equivalent to the slice
    implying a non-negative risk-neutral density.
    """
    w, w1, w2 = svi_derivatives(k, params)
    safe_w = np.maximum(w, 1e-12)
    return (
        (1.0 - k * w1 / (2.0 * safe_w)) ** 2
        - (w1**2 / 4.0) * (1.0 / safe_w + 0.25)
        + w2 / 2.0
    )


def implied_density(k, params: dict) -> "np.ndarray":
    """
    Risk-neutral density in log-moneyness. Negative wherever g(k) < 0, which
    is the concrete statement of what a butterfly violation means.
    """
    w, _, _ = svi_derivatives(k, params)
    safe_w = np.maximum(w, 1e-12)
    d2 = -k / np.sqrt(safe_w) - np.sqrt(safe_w) / 2.0
    return durrleman_g(k, params) / np.sqrt(2 * np.pi * safe_w) * np.exp(-(d2**2) / 2)


def default_k_grid(points: int = 601, width: float = 1.5) -> "np.ndarray":
    return np.linspace(-width, width, points)


def check_butterfly_arbitrage(
    svi_params: dict, k_grid: "np.ndarray | None" = None,
    quoted_range: tuple[float, float] | None = None,
) -> list[dict]:
    """
    Durrleman's condition: g(k) must be >= 0 for the fitted slice's implied
    density to be non-negative (no butterfly arbitrage). Return a list of k
    values where g(k) < 0.

    Violations are grouped into contiguous runs rather than returned per grid
    point, so a single bad region reads as one finding instead of two hundred.
    """
    k_grid = default_k_grid() if k_grid is None else np.asarray(k_grid, float)
    g = durrleman_g(k_grid, svi_params)
    bad = g < 0

    violations = []
    if not bad.any():
        return violations

    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        segment = slice(start, stop)
        worst = start + int(np.argmin(g[segment]))
        inside = (
            quoted_range is not None
            and k_grid[worst] >= quoted_range[0]
            and k_grid[worst] <= quoted_range[1]
        )
        violations.append({
            "type": "butterfly",
            "k_from": float(k_grid[start]),
            "k_to": float(k_grid[stop - 1]),
            "k_worst": float(k_grid[worst]),
            "g_min": float(g[worst]),
            "density_min": float(implied_density(k_grid[worst], svi_params)),
            "inside_quoted_range": bool(inside),
        })
    return violations


def trust_radius(
    svi_params: dict, quoted_range: tuple[float, float] | None = None,
    width: float = 3.0, points: int = 3001,
) -> dict:
    """
    How far either side of the money the slice stays arbitrage-free.

    Returns the largest interval around k=0 on which g(k) >= 0 holds
    continuously. This is the practical output of the butterfly check: not
    "the surface has arbitrage" but "this slice may be queried out to here,
    and must be clamped beyond it". Phase 5 extrapolation should respect it.
    """
    k = np.linspace(-width, width, points)
    g = durrleman_g(k, svi_params)
    centre = int(np.argmin(np.abs(k)))
    if g[centre] < 0:
        return {"k_low": 0.0, "k_high": 0.0, "sound_at_the_money": False}

    low = centre
    while low > 0 and g[low - 1] >= 0:
        low -= 1
    high = centre
    while high < len(k) - 1 and g[high + 1] >= 0:
        high += 1

    result = {
        "k_low": float(k[low]), "k_high": float(k[high]),
        "sound_at_the_money": True,
    }
    if quoted_range is not None:
        result["covers_quoted_range"] = bool(
            k[low] <= quoted_range[0] and k[high] >= quoted_range[1]
        )
    return result


def check_calendar_arbitrage(
    svi_params_by_expiry: dict, k_grid: "np.ndarray | None" = None,
    quoted_ranges: dict | None = None,
) -> list[dict]:
    """
    At each k in k_grid, total variance w(k, T) must be non-decreasing in T
    across expiries. Return a list of violations (k, T1, T2, w1, w2).

    Only adjacent maturities are compared. If w is non-decreasing between
    every neighbouring pair it is non-decreasing overall, and reporting every
    pair would multiply one bad slice into a violation against all of them.
    """
    k_grid = default_k_grid() if k_grid is None else np.asarray(k_grid, float)
    maturities = sorted(svi_params_by_expiry)

    violations = []
    for T1, T2 in zip(maturities, maturities[1:]):
        w1, _, _ = svi_derivatives(k_grid, svi_params_by_expiry[T1])
        w2, _, _ = svi_derivatives(k_grid, svi_params_by_expiry[T2])
        gap = w2 - w1
        bad = gap < 0
        if not bad.any():
            continue

        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
        )
        for start, stop in zip(edges[::2], edges[1::2]):
            worst = start + int(np.argmin(gap[start:stop]))
            inside = False
            if quoted_ranges and T1 in quoted_ranges and T2 in quoted_ranges:
                low = max(quoted_ranges[T1][0], quoted_ranges[T2][0])
                high = min(quoted_ranges[T1][1], quoted_ranges[T2][1])
                inside = low <= k_grid[worst] <= high
            violations.append({
                "type": "calendar",
                "T1": float(T1), "T2": float(T2),
                "k_from": float(k_grid[start]),
                "k_to": float(k_grid[stop - 1]),
                "k_worst": float(k_grid[worst]),
                "w1": float(w1[worst]), "w2": float(w2[worst]),
                "shortfall": float(-gap[worst]),
                "inside_quoted_range": bool(inside),
            })
    return violations


def check_surface(
    svi_params_by_expiry: dict,
    k_grid: "np.ndarray | None" = None,
    quoted_ranges: dict | None = None,
) -> dict:
    """Run both checks over a fitted surface and return everything found."""
    k_grid = default_k_grid() if k_grid is None else np.asarray(k_grid, float)
    butterfly = []
    for T, params in sorted(svi_params_by_expiry.items()):
        found = check_butterfly_arbitrage(
            params, k_grid,
            quoted_range=(quoted_ranges or {}).get(T),
        )
        for item in found:
            item["T"] = float(T)
        butterfly.extend(found)

    calendar = check_calendar_arbitrage(svi_params_by_expiry, k_grid, quoted_ranges)
    wings = {
        T: wing_bound(params, T)
        for T, params in sorted(svi_params_by_expiry.items())
    }
    return {"butterfly": butterfly, "calendar": calendar, "wing_bound": wings}
