"""
When to rebalance a delta hedge, and what each rule costs.

`scripts/hedge_frequency.py` measured the only rule the project had: rebalance
on a fixed clock. It found the premium dies on hedge slippage -- 38-75 bps
against a gross of 83 -- and that the best fixed interval is 4-6h. But a clock
is the crudest possible rule. It trades when the calendar says so, not when
the position has actually drifted, which means it pays to re-hedge a position
that has barely moved and sits through one that has moved a lot.

The alternative is a no-trade band: hold the hedge until the delta drifts more
than H away from target, then trade back. Two ways to set H here:

- **fixed** -- a constant band, the simplest move-based rule and the right
  first baseline.
- **whalley_wilmott** -- the asymptotic band from Whalley & Wilmott (1997),
  H = (3/2 * lambda * F * Gamma^2 / gamma)^(1/3), for proportional cost
  `lambda` and risk aversion `gamma`. It widens where gamma is low (the
  hedge barely drifts, so leave it) and where cost is high, which is the
  behaviour a clock cannot express at any interval.

All three rules are the same walk over the same path; they differ only in when
they say "trade". So `policy_arrays` reduces each to two per-step arrays --
`forced` (rebalance regardless) and `H` (band half-width, `inf` when the rule
is not band-driven) -- and one loop consumes them. That matters for more than
tidiness: the fixed-clock rule expressed this way reproduces the old
vectorised implementation *exactly*, which is asserted in the tests, so a
band's numbers and a clock's numbers remain comparable.

The loop is sequential because the held hedge depends on its own history --
that is what a band means. Everything that does not depend on the hedge state
(forwards, target deltas, gammas, band widths) is computed vectorised first,
so the loop body is a handful of scalar operations over Python floats.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

YEAR_HOURS = 365.25 * 24

KINDS = ("time", "fixed", "whalley_wilmott")


def _d1(F, K: float, T_left, sigma: float):
    """Black-76 d1, with the expired steps parked at a safe value."""
    live = T_left > 0
    safe = np.where(live, T_left, 1.0)
    return np.where(live,
                    (np.log(F / K) + 0.5 * sigma**2 * safe) / (sigma * np.sqrt(safe)),
                    0.0), live


def target_delta_and_gamma(F, K: float, T_left, sigma: float, option: str):
    """
    Delta and gamma with respect to the forward, at every step of the path.

    Neither depends on the hedge that is actually held, so both come out of
    one vectorised pass and the sequential loop never touches scipy.
    """
    d1, live = _d1(F, K, T_left, sigma)
    delta = norm.cdf(d1) if option == "C" else norm.cdf(d1) - 1.0
    delta = np.where(live, delta, 0.0)
    safe = np.where(live, T_left, 1.0)
    gamma = np.where(live, norm.pdf(d1) / (F * sigma * np.sqrt(safe)), 0.0)
    return delta, gamma


def ww_band(F, gamma, risk_aversion: float, slip_bps: float):
    """
    Whalley-Wilmott no-trade half-width, in units of the underlying.

        H = (3/2 * lambda * F * Gamma^2 / gamma)^(1/3)

    for proportional cost `lambda` and risk aversion `gamma`. Discounting is
    omitted because the whole project prices undiscounted off the forward, so
    `F` is the forward here and `Gamma` is d(delta)/dF of the whole position.

    Scalars or arrays, because two callers need it: `policy_arrays` reduces a
    swept path to an array of widths, and `strategy.straddle` asks for one
    width at one snapshot of the live book. One formula, so the rule the
    archive study measured and the rule the live book runs cannot drift apart.

    The two degenerate cases both invert the rule if they are got wrong:

    - **No cost.** There is nothing to economise on, so tolerating drift buys
      nothing: the band collapses to zero, meaning *trade at every
      opportunity*, not never.
    - **No gamma** -- expired, or so far out of the money the hedge cannot
      drift. The band is infinite, which is the correct reading: there is
      nothing to re-hedge. A finite band here would churn on rounding.
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk aversion must be positive, got {risk_aversion}")
    gamma = np.asarray(gamma, dtype=float)
    lam = slip_bps / 1e4
    if lam <= 0:
        return np.zeros(gamma.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.cbrt(1.5 * lam * np.asarray(F, dtype=float) * gamma**2
                    / float(risk_aversion))
    return np.where(gamma > 0, H, np.inf)


def policy_arrays(kind: str, param: float, F, T_left, gamma, slip_bps: float):
    """
    Reduce a rule to (forced, H): when it must trade, and how far it tolerates.

    `time` ignores drift entirely, so its band is infinite and `forced` carries
    the whole rule. The band rules force only the opening trade -- a desk puts
    the hedge on when it sells the option rather than waiting for the position
    to drift into tolerance, and starting flat would otherwise leave a deep
    out-of-the-money option unhedged for its whole life on a technicality.
    """
    n = len(F)
    steps = np.arange(n)
    if kind == "time":
        interval = int(param)
        if interval < 1:
            raise ValueError(f"time interval must be >= 1 hour, got {param}")
        return steps % interval == 0, np.full(n, np.inf)

    forced = steps == 0
    if kind == "fixed":
        if param <= 0:
            raise ValueError(f"a fixed band must be positive, got {param}")
        return forced, np.full(n, float(param))

    if kind == "whalley_wilmott":
        return forced, ww_band(F, gamma, param, slip_bps)

    raise ValueError(f"unknown policy {kind!r}, expected one of {KINDS}")


def walk(spots, hours, K: float, T0: float, sigma: float, carry: float,
         option: str, kind: str, param: float,
         slip_bps: float = 0.0, half_spread_vol: float = 0.0) -> dict:
    """
    Seller's P&L on one option, hedged under `kind`/`param`, with costs.

    `spots` is the index at `hours` (hours since inception, last = expiry).
    Pass the *full* hourly path: a clock rule subsamples it through `forced`,
    so every rule is judged on the same underlying data and a band is free to
    trade at any hour it likes.

    The seller is short the option and long delta against it, sells at the bid
    (`sigma - half_spread_vol`) once at inception, and pays `slip_bps` on the
    notional of every trade including opening the hedge and lifting it at
    expiry.
    """
    from pricing.black76 import intrinsic, price

    T_left = np.maximum(T0 - np.asarray(hours, dtype=float) / YEAR_HOURS, 0.0)
    F = np.asarray(spots, dtype=float) * np.exp(carry * T_left)

    mid_premium = float(price(F[0], K, T0, sigma, option))
    sell_vol = sigma - half_spread_vol
    if sell_vol <= 0:
        return {"unsellable": True}
    received = float(price(F[0], K, T0, sell_vol, option))

    # Everything up to here is hedge-independent, so it stays vectorised.
    delta, gamma = target_delta_and_gamma(F[:-1], K, T_left[:-1], sigma, option)
    forced, bands = policy_arrays(kind, param, F[:-1], T_left[:-1], gamma, slip_bps)

    # Python floats in the loop: indexing numpy scalars costs more than the
    # arithmetic does, and this runs ~1M times per swept cell.
    f_list = F.tolist()
    target = delta.tolist()
    band = bands.tolist()
    force = forced.tolist()

    held = 0.0
    hedge_gain = 0.0
    turnover = 0.0
    trades = 0
    for i in range(len(target)):
        want = target[i]
        if force[i] or abs(want - held) > band[i]:
            turnover += abs(want - held) * f_list[i]
            held = want
            trades += 1
        hedge_gain += held * (f_list[i + 1] - f_list[i])

    turnover += abs(held) * f_list[-1]          # lifted at expiry
    slippage = turnover * slip_bps / 1e4
    payoff = float(intrinsic(float(spots[-1]), K, option))

    return {
        "unsellable": False,
        "premium": mid_premium,
        "gross": mid_premium + hedge_gain - payoff,
        "net": received + hedge_gain - payoff - slippage,
        "spread_cost": mid_premium - received,
        "slippage": slippage,
        "turnover": turnover,
        "trades": trades,
        "mean_band": float(np.mean([b for b in band if np.isfinite(b)]))
        if np.isfinite(bands).any() else float("nan"),
    }
