"""
Tests for the rebalancing rules.

The load-bearing one is the first: a clock rule walked over the full hourly
path must reproduce the vectorised implementation walked over the subsampled
path, exactly. That is what makes a band's number comparable to §11.2's --
`hedged_seller_pnl` is already asserted equal to `vrp_surface` at zero cost,
so the chain runs published result -> vectorised -> walker -> band, with every
link checked rather than assumed.

The rest guard the two ways a band goes quietly wrong: a width that does not
respond to what it is supposed to respond to, and a degenerate case (no gamma,
no cost, expired) that trades when it should not or vice versa.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from strategy import hedging

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hf = _script("hedge_frequency")


def hourly_path(days: int, seed: int = 0, vol: float = 0.004):
    rng = np.random.default_rng(seed)
    n = days * 24 + 1
    return (60_000.0 * np.exp(np.cumsum(rng.normal(0, vol, n))),
            np.arange(n, dtype=float))


def walk_atm(kind, param, days=30, seed=0, slip=5.0, sigma=0.6, **kw):
    spots, hours = hourly_path(days, seed)
    T0 = float(hours[-1] / hedging.YEAR_HOURS)
    return hedging.walk(spots, hours, 60_000.0, T0, sigma, 0.05, "C",
                        kind, param, slip_bps=slip, **kw)


# ---- the link back to the published result ------------------------------

@pytest.mark.parametrize("interval", [1, 2, 4, 6, 12, 24])
def test_clock_rule_reproduces_the_vectorised_implementation(interval):
    """
    Walked hourly under a clock, the held hedge is constant between trades, so
    the P&L must telescope to the subsampled calculation. If it does not, a
    band's number cannot be read against §11.2's.
    """
    rng = np.random.default_rng(interval)
    for _ in range(12):
        days = int(rng.integers(3, 31))
        spots, hours = hourly_path(days, seed=int(rng.integers(0, 9999)))
        T0 = float(hours[-1] / hedging.YEAR_HOURS)
        K = 60_000.0 * float(rng.uniform(0.75, 1.35))
        sigma, carry = float(rng.uniform(0.25, 1.4)), float(rng.uniform(-0.1, 0.2))
        option = str(rng.choice(["C", "P"]))
        slip, spread = float(rng.choice([0.0, 2.0, 5.0, 10.0])), float(rng.choice([0.0, 0.005]))

        idx = list(range(0, len(hours), interval))
        if idx[-1] != len(hours) - 1:
            idx.append(len(hours) - 1)
        want = hf.hedged_seller_pnl(spots[idx], hours[idx], K, T0, sigma, carry, option,
                                    half_spread_vol=spread, slip_bps=slip)
        got = hedging.walk(spots, hours, K, T0, sigma, carry, option, "time", interval,
                           slip_bps=slip, half_spread_vol=spread)
        for key in ("gross", "net", "premium", "spread_cost", "slippage", "turnover"):
            assert got[key] == pytest.approx(want[key], abs=1e-7), key


# ---- bands respond to what they are supposed to respond to --------------

def test_a_wider_band_trades_less():
    trades = [walk_atm("fixed", h)["trades"] for h in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)]
    assert trades == sorted(trades, reverse=True)
    assert trades[0] > 5 * trades[-1]


def test_a_band_trades_less_than_the_hourly_clock():
    """The entire point: same path, same hours available, fewer trades."""
    clock = walk_atm("time", 1)
    band = walk_atm("fixed", 0.05)
    assert band["trades"] < clock["trades"]
    assert band["turnover"] < clock["turnover"]
    assert band["slippage"] < clock["slippage"]


def test_whalley_wilmott_widens_with_cost_and_narrows_with_risk_aversion():
    """
    H = (3/2 lambda F Gamma^2 / gamma)^(1/3): cube-root in both. A band that
    ignored the cost it is meant to economise on would be the bug that makes
    the whole rule pointless, and it would still look plausible in the table.
    """
    cheap = walk_atm("whalley_wilmott", 1e-3, slip=2.0)["mean_band"]
    dear = walk_atm("whalley_wilmott", 1e-3, slip=10.0)["mean_band"]
    assert dear > cheap
    assert dear / cheap == pytest.approx((10.0 / 2.0) ** (1 / 3), rel=1e-9)

    timid = walk_atm("whalley_wilmott", 1e-2)["mean_band"]
    bold = walk_atm("whalley_wilmott", 1e-4)["mean_band"]
    assert bold > timid
    assert bold / timid == pytest.approx((1e-2 / 1e-4) ** (1 / 3), rel=1e-9)


def test_a_costless_hedge_tolerates_no_drift():
    """With lambda = 0 the band collapses, which must mean 'trade every hour',
    not 'never trade' -- a zero that divides the wrong way inverts the rule."""
    free = walk_atm("whalley_wilmott", 1e-3, slip=0.0)
    clock = walk_atm("time", 1, slip=0.0)
    assert free["trades"] == clock["trades"]
    assert free["turnover"] == pytest.approx(clock["turnover"], rel=1e-9)


def test_no_gamma_means_no_band_and_no_trading():
    """Where the option cannot drift there is nothing to re-hedge; an
    infinite band says so, and a finite one would churn on rounding."""
    F = np.array([60_000.0, 60_100.0, 60_200.0])
    gamma = np.zeros(3)
    forced, H = hedging.policy_arrays("whalley_wilmott", 1e-3, F,
                                      np.array([0.1, 0.05, 0.0]), gamma, 5.0)
    assert np.isinf(H).all()
    assert forced.tolist() == [True, False, False]


# ---- shape of the rules -------------------------------------------------

def test_a_clock_ignores_drift_entirely():
    forced, H = hedging.policy_arrays("time", 4, np.ones(9), np.ones(9), np.ones(9), 5.0)
    assert np.isinf(H).all()
    assert forced.tolist() == [True, False, False, False, True,
                               False, False, False, True]


def test_a_band_forces_only_the_opening_trade():
    forced, H = hedging.policy_arrays("fixed", 0.05, np.ones(5), np.ones(5), np.ones(5), 5.0)
    assert forced.tolist() == [True, False, False, False, False]
    assert (H == 0.05).all()


def test_clocks_report_no_band():
    assert np.isnan(walk_atm("time", 4)["mean_band"])
    assert np.isfinite(walk_atm("fixed", 0.05)["mean_band"])


@pytest.mark.parametrize("kind,param", [
    ("time", 0), ("fixed", 0.0), ("fixed", -0.1), ("whalley_wilmott", 0.0),
])
def test_degenerate_parameters_are_refused(kind, param):
    with pytest.raises(ValueError):
        hedging.policy_arrays(kind, param, np.ones(3), np.ones(3), np.ones(3), 5.0)


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="unknown policy"):
        hedging.policy_arrays("magic", 1.0, np.ones(3), np.ones(3), np.ones(3), 5.0)


def test_an_unsellable_quote_is_still_refused():
    out = walk_atm("fixed", 0.05, sigma=0.20, half_spread_vol=0.25)
    assert out["unsellable"] is True
