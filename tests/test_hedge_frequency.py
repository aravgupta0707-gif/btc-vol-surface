"""
Tests for the hedging-frequency sweep.

The study's whole claim is that it adds *costs* to the section 11.2 result
rather than recomputing it differently, so the first and most important test
is that its P&L reduces exactly to `vrp_surface`'s when both costs are zero.
Everything else guards the two things a cost model gets quietly wrong: a sign
(a cost that helps the seller) and a direction (a cost that does not respond
to the parameter being swept).

The sharding is tested too. A Slurm array that drops or double-counts a day
produces a plausible number from the wrong sample, which no downstream check
would catch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hf = _script("hedge_frequency")
vrp = _script("vrp_surface")


def path_at(hedge_hours: int, days: int = 30, seed: int = 3):
    """An hourly index path, subsampled the way the sweep subsamples it."""
    rng = np.random.default_rng(seed)
    hourly = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, days * 24 + 1)))
    hours = np.arange(len(hourly), dtype=float)
    step = slice(None, None, hedge_hours)
    spots, taken = hourly[step], hours[step]
    if taken[-1] != hours[-1]:
        spots = np.append(spots, hourly[-1])
        taken = np.append(taken, hours[-1])
    return spots, taken


# ---- the invariant that keeps this tied to the published result ---------

@pytest.mark.parametrize("option", ["C", "P"])
def test_zero_cost_reduces_to_vrp_surface(option):
    """
    With no spread and no slippage this must equal scripts/vrp_surface.py to
    the last bit -- otherwise the sweep is a second, different calculation of
    section 11.2 and any disagreement between them is unattributable.
    """
    rng = np.random.default_rng(11)
    for _ in range(50):
        spots, hours = path_at(int(rng.integers(1, 25)), seed=int(rng.integers(0, 9999)))
        T0 = float(hours[-1] / hf.YEAR_HOURS)
        K = 60_000.0 * float(rng.uniform(0.75, 1.35))
        sigma, carry = float(rng.uniform(0.25, 1.4)), float(rng.uniform(-0.1, 0.2))

        want_pnl, want_premium = vrp.hedged_seller_pnl(
            spots, hours, K, T0, sigma, carry, option)
        got = hf.hedged_seller_pnl(spots, hours, K, T0, sigma, carry, option,
                                   half_spread_vol=0.0, slip_bps=0.0)

        assert got["gross"] == pytest.approx(want_pnl, abs=1e-9)
        assert got["net"] == pytest.approx(want_pnl, abs=1e-9)
        assert got["premium"] == pytest.approx(want_premium, abs=1e-9)


# ---- costs have the right sign and the right direction ------------------

def test_costs_only_ever_hurt_the_seller():
    spots, hours = path_at(4)
    free = hf.hedged_seller_pnl(spots, hours, 60_000.0, 30 / 365.25, 0.6, 0.05, "C")
    costly = hf.hedged_seller_pnl(spots, hours, 60_000.0, 30 / 365.25, 0.6, 0.05, "C",
                                  half_spread_vol=0.01, slip_bps=5.0)
    assert costly["spread_cost"] > 0
    assert costly["slippage"] > 0
    assert costly["net"] < costly["gross"]
    assert costly["net"] < free["net"]
    # The spread does not touch the mid-priced number.
    assert costly["gross"] == pytest.approx(free["gross"], rel=1e-12)


def test_slippage_is_linear_in_the_rate():
    spots, hours = path_at(4)
    args = (spots, hours, 60_000.0, 30 / 365.25, 0.6, 0.05, "C")
    five = hf.hedged_seller_pnl(*args, half_spread_vol=0.0, slip_bps=5.0)
    ten = hf.hedged_seller_pnl(*args, half_spread_vol=0.0, slip_bps=10.0)
    assert ten["slippage"] == pytest.approx(2 * five["slippage"], rel=1e-12)
    assert five["slippage"] == pytest.approx(five["turnover"] * 5 / 1e4, rel=1e-12)


def test_hedging_more_often_trades_more():
    """
    The premise of the whole sweep: turnover rises as the interval shortens,
    which is what pushes against the falling hedging error. If this inverted,
    every conclusion in the study would flip.
    """
    turnovers = []
    for hedge_hours in (1, 2, 4, 6, 12, 24):
        spots, hours = path_at(hedge_hours)
        out = hf.hedged_seller_pnl(spots, hours, 60_000.0, 30 / 365.25, 0.6, 0.05, "C",
                                   slip_bps=5.0)
        turnovers.append(out["turnover"])
    assert turnovers == sorted(turnovers, reverse=True)
    assert turnovers[0] > 2 * turnovers[-1]


def test_an_unsellable_quote_is_refused_not_clamped():
    """A half-spread wider than the vol is a broken quote; pricing it at some
    floor would invent a fill that the book never offered."""
    spots, hours = path_at(4)
    out = hf.hedged_seller_pnl(spots, hours, 60_000.0, 30 / 365.25, 0.20, 0.05, "C",
                               half_spread_vol=0.25)
    assert out["unsellable"] is True
    assert "net" not in out


# ---- the measured spread curve ------------------------------------------

def test_half_spread_interpolates_and_clamps():
    curve = pd.DataFrame({
        "T": [0.08] * 3 + [0.5] * 3,
        "k": [-0.2, 0.0, 0.2] * 2,
        "half_spread_vol": [0.02, 0.01, 0.02, 0.006, 0.003, 0.006],
    })
    # picks the nearest fitted expiry, interpolates linearly in k
    assert hf._half_spread_at(curve, 0.09, -0.1) == pytest.approx(0.015)
    assert hf._half_spread_at(curve, 0.45, 0.0) == pytest.approx(0.003)
    # held flat outside the quoted range rather than extrapolated
    assert hf._half_spread_at(curve, 0.08, -5.0) == pytest.approx(0.02)
    assert hf._half_spread_at(curve, 0.08, +5.0) == pytest.approx(0.02)
    assert np.isnan(hf._half_spread_at(curve.iloc[:0], 0.08, 0.0))


# ---- sharding: a Slurm array must cover every item exactly once ---------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16])
def test_shards_partition_the_work(n):
    items = list(range(158))
    pieces = [hf.shard(items, f"{i}/{n}") for i in range(n)]
    flat = [x for piece in pieces for x in piece]
    assert sorted(flat) == items, "a shard dropped or duplicated work"
    assert max(len(p) for p in pieces) - min(len(p) for p in pieces) <= 1


def test_shard_is_strided_not_contiguous():
    """Contiguous chunks would hand one task every late-era day, which are
    the expensive ones."""
    assert hf.shard(list(range(10)), "0/2") == [0, 2, 4, 6, 8]


def test_shard_rejects_an_index_outside_the_array():
    with pytest.raises(ValueError):
        hf.shard([1, 2, 3], "3/3")
    assert hf.shard([1, 2, 3], None) == [1, 2, 3]


def test_mixed_shard_generations_are_refused():
    """
    Both loaders glob, so shards from two different runs would be read
    together and every day counted twice -- means barely move, t inflates by
    ~sqrt(2). Plausible and wrong, so it has to fail rather than average.
    """
    rows = pd.DataFrame({"asset": ["BTC", "BTC"], "date": ["2024-01-01"] * 2,
                         "T": [0.08, 0.08], "skipped": ["", ""]})
    with pytest.raises(ValueError, match="more than one run"):
        hf._no_duplicates(rows, ["asset", "date", "T"], Path("somewhere"))

    fine = rows.assign(T=[0.08, 0.25])
    hf._no_duplicates(fine, ["asset", "date", "T"], Path("somewhere"))


def test_distinct_failures_on_one_day_are_not_duplicates():
    """Skipped rows carry NaN keys; two different failures must not read as
    one repeated day."""
    rows = pd.DataFrame({
        "asset": ["BTC", "BTC"], "date": ["2024-01-01"] * 2,
        "tenor": [7, 30], "z": [np.nan, np.nan], "skipped": ["index gap"] * 2,
    })
    hf._no_duplicates(rows[rows["skipped"] == ""], ["asset", "date", "tenor", "z"],
                      Path("somewhere"))


def test_every_cell_is_covered_once():
    every = hf.cells()
    assert len(every) == len(hf.ASSETS) * len(hf.policies()) * len(hf.SLIPPAGE_BPS)
    assert len(set(every)) == len(every)


def test_the_clock_baseline_comes_first_in_the_grid():
    """A band is only interesting against the best clock, so the clocks have
    to be swept in the same run and printed above the bands."""
    labels = [label for _, _, label in hf.policies()]
    assert labels[:len(hf.HEDGE_HOURS)] == [f"time {h:g}h" for h in hf.HEDGE_HOURS]
    assert any(l.startswith("band") for l in labels)
    assert any(l.startswith("ww") for l in labels)
    assert len(set(labels)) == len(labels)


# ---- the grid is overridable from the job script ------------------------

def test_grid_axes_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("HEDGE_HOURS", "1 3 9")
    assert hf._grid("HEDGE_HOURS", (1, 2), int) == (1, 3, 9)
    monkeypatch.setenv("SLIPPAGE_BPS", "0,2.5,7")
    assert hf._grid("SLIPPAGE_BPS", (0.0,)) == (0.0, 2.5, 7.0)
    monkeypatch.delenv("HEDGE_HOURS")
    assert hf._grid("HEDGE_HOURS", (4, 6), int) == (4, 6)
    monkeypatch.setenv("Z_GRID", "")
    assert hf._grid("Z_GRID", (0.0, 1.0)) == (0.0, 1.0)


# ---- walk-forward selection must not see forward ------------------------

def _cells(dates, per_rule: dict, tenor: int = 30):
    """A minimal cells frame: one row per (date, rule)."""
    return pd.DataFrame([
        {"date": pd.Timestamp(d), "policy": rule, "net_bps": value, "tenor": tenor}
        for rule, values in per_rule.items()
        for d, value in zip(dates, values)
    ])


def test_settled_history_excludes_what_had_not_settled_yet():
    """
    A 30-day position opened one month before `as_of` settles within hours of
    it, so its P&L was not available to choose with. Counting it would be a
    one-month lookahead dressed as an expanding window.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    got = hf.settled_history(dates, pd.Timestamp("2024-04-01"), 30)
    assert list(got) == list(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    # At Mar 1, Feb 1 + 30d = Mar 2 has not settled, so only Jan 1 is known.
    got = hf.settled_history(dates, pd.Timestamp("2024-03-01"), 30)
    assert list(got) == [pd.Timestamp("2024-01-01")]


def test_walk_forward_cannot_pick_a_rule_for_its_own_future():
    """
    `late` is the worst rule for a year and the best thereafter. A selector
    that peeked would ride it from the start; an honest one cannot choose it
    until its good results are settled history, so its early picks are `early`.
    """
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    frame = _cells(dates, {
        "early": [10.0] * 12 + [0.0] * 12,
        "late": [-10.0] * 12 + [100.0] * 12,
    })
    picked = hf.walk_forward(frame, ["early", "late"], tenor_days=30, warmup=3)
    assert not picked.empty
    first_year = picked[picked["date"] < pd.Timestamp("2021-01-01")]
    assert (first_year["chosen"] == "early").all(), "the selector saw the future"
    assert (picked["chosen"] == "late").any(), "it never learned"


def test_walk_forward_respects_the_warmup():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    frame = _cells(dates, {"a": [1.0] * 12, "b": [0.0] * 12})
    few, many = (hf.walk_forward(frame, ["a", "b"], 30, warmup=w) for w in (2, 9))
    assert len(few) > len(many)
    assert hf.walk_forward(frame, ["a", "b"], 30, warmup=99).empty


def test_walk_forward_reports_what_the_chosen_rule_actually_earned():
    """The realised number must be the chosen rule's own outcome on that date,
    not the best outcome available on it."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    frame = _cells(dates, {"a": [5.0] * 12, "b": [-1.0] * 11 + [999.0]})
    picked = hf.walk_forward(frame, ["a", "b"], 30, warmup=3)
    assert (picked["chosen"] == "a").all()
    assert picked["net_bps"].unique().tolist() == [5.0]


def test_walk_forward_ignores_rules_absent_from_the_data():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    frame = _cells(dates, {"a": [1.0] * 12})
    picked = hf.walk_forward(frame, ["a", "nonexistent"], 30, warmup=2)
    assert (picked["chosen"] == "a").all()
    assert hf.walk_forward(frame, ["nonexistent"], 30, warmup=2).empty


# ---- the offline cache read the compute nodes depend on -----------------

def test_offline_history_never_opens_a_socket(tmp_path, monkeypatch):
    from data import history

    monkeypatch.setattr(history, "CACHE", tmp_path)
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "index_price": [1.0, 2.0, 3.0],
        "interest_1h": [0.0] * 3, "interest_8h": [0.0] * 3,
    })
    frame.to_parquet(tmp_path / "btc_funding_hourly.parquet", index=False)

    def explode(*a, **k):
        raise AssertionError("offline=True must not reach the network")

    monkeypatch.setattr(history, "_get", explode)
    got = history.hourly_funding("BTC", offline=True)
    assert len(got) == 3

    with pytest.raises(FileNotFoundError):
        history.hourly_funding("ETH", offline=True)
