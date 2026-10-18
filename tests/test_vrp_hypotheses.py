"""
Tests for the statistical helpers behind HYPOTHESES.md.

The conclusions there rest on four small calculations -- a Newey-West standard
error, a bipower-variation jump estimator, a block bootstrap and a break
statistic. Each has a failure that leaves plausible-looking numbers, so each is
checked here against a case where the answer is known in advance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vh = _script("vrp_hypotheses")


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    return x


def test_newey_west_matches_plain_se_on_independent_data():
    x = np.random.default_rng(0).standard_normal(5000)
    _, se, _ = vh.nw_mean(x, lags=10)
    assert abs(se / (x.std() / np.sqrt(len(x))) - 1) < 0.1


def test_newey_west_widens_the_error_for_autocorrelated_data():
    """The reason it exists: overlapping windows make the plain SE far too small."""
    x = _ar1(5000, 0.9, seed=1)
    _, se, _ = vh.nw_mean(x, lags=30)
    plain = x.std() / np.sqrt(len(x))
    assert se > 2.5 * plain


def test_ols_recovers_slope_and_flags_it_on_noise():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3000)
    y = 2.0 * x + rng.standard_normal(3000)
    slope, t = vh.nw_ols(y, x, lags=5)
    assert abs(slope - 2.0) < 0.1 and t > 20
    slope, t = vh.nw_ols(rng.standard_normal(3000), x, lags=5)
    assert abs(t) < 3


def test_multi_ols_separates_two_regressors():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(4000), rng.standard_normal(4000)
    y = 1.5 * a - 0.5 * b + 0.2 * rng.standard_normal(4000)
    beta, _ = vh.nw_ols_multi(y, np.column_stack([a, b]), lags=5)
    assert np.allclose(beta, [1.5, -0.5], atol=0.05)


def _hourly(returns: np.ndarray) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(returns), freq="h", tz="UTC")
    return pd.Series(returns, index=index)


def test_bipower_ignores_a_jump_that_realised_variance_counts():
    rng = np.random.default_rng(4)
    clean = rng.standard_normal(24 * 200) * 0.005
    jumped = clean.copy()
    jumped[1000] += 0.15  # one 15% gap

    clean_day = vh.daily_variance(_hourly(clean))
    jumped_day = vh.daily_variance(_hourly(jumped))

    # No jump: continuous and total variance agree to estimator noise.
    assert abs(clean_day["bv"].sum() / clean_day["rv"].sum() - 1) < 0.1
    # One jump of size J adds J^2 to realised variance; bipower should miss almost all of it.
    jump_variance = 0.15 ** 2
    rv_added = jumped_day["rv"].sum() - clean_day["rv"].sum()
    bv_added = jumped_day["bv"].sum() - clean_day["bv"].sum()
    assert abs(rv_added / jump_variance - 1) < 0.15
    assert bv_added < 0.25 * jump_variance


def test_daily_variance_weights_hours_by_sampling_step():
    """48 four-hour returns cover 192 hours, and that is what annualisation divides by."""
    day = vh.daily_variance(_hourly(np.full(48, 0.001)), step=4)
    assert day["n"].sum() == 4.0 * 48


def test_block_indices_are_valid_and_blocked():
    rng = np.random.default_rng(5)
    idx = vh.block_indices(100, 10, rng)
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    # Inside a block consecutive draws step by one (mod n): serial structure survives.
    steps = np.diff(idx[:10]) % 100
    assert (steps == 1).all()


def test_sup_shift_finds_a_planted_break():
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.standard_normal(600) + 3.0, rng.standard_normal(600)])
    tau, stat = vh.sup_shift(p)
    assert abs(tau - 600) <= 40
    assert stat > 8


def test_sup_shift_pvalue_is_large_without_a_break():
    p = np.random.default_rng(7).standard_normal(800)
    assert vh.sup_shift_pvalue(p, block=30, draws=60) > 0.1


def test_sup_shift_pvalue_is_small_with_a_break():
    rng = np.random.default_rng(8)
    p = np.concatenate([rng.standard_normal(400) + 2.0, rng.standard_normal(400)])
    assert vh.sup_shift_pvalue(p, block=30, draws=60) < 0.05


def test_drawdown_is_zero_at_new_highs_and_uses_only_the_past():
    close = pd.Series([100.0, 110, 99, 120, 60], index=pd.date_range("2024-01-01", periods=5))
    dd = vh.drawdown(close, window=3)
    assert np.isnan(dd.iloc[1])                    # not enough history yet
    assert abs(dd.iloc[2] - 10.0) < 1e-9           # 99 vs high 110
    assert dd.iloc[3] == 0.0                       # a new high
    assert abs(dd.iloc[4] - 50.0) < 1e-9           # 60 vs 120
    # changing a later close must not change an earlier drawdown
    later = close.copy(); later.iloc[4] = 1.0
    assert vh.drawdown(later, window=3).iloc[:4].equals(dd.iloc[:4])
