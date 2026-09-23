"""
Tests for the delta-hedged straddle and the backtest loop.

Two things matter more than the rest. First, that the accounting is right --
premium banked, options marked, hedge P&L and every cost accumulated, with no
sign errors, because a sign error here produces a plausible-looking equity
curve that is simply wrong. Second, that the loop cannot see the future,
which is the failure mode BUILD_GUIDE.md section 8 calls the most common way
backtests silently lie.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pricing.black76 import greeks as black76_greeks
from pricing.black76 import price as black76_price
from strategy.backtest import (
    ANNUALIZE,
    assert_no_lookahead,
    compute_metrics,
    realized_vol,
    run_backtest,
    summarize,
)
from strategy.straddle import Costs, DeltaHedgedStraddle, decompose

UTC = dt.timezone.utc
EXPIRY = pd.Timestamp("2026-10-30 08:00", tz="UTC")
STRIKES = [70_000.0, 75_000.0, 80_000.0, 85_000.0, 90_000.0]


def snapshot(stamp, forward: float, sigma: float = 0.50,
             half_spread: float = 0.02, strikes=None) -> pd.DataFrame:
    """A minimal cleaned-chain frame: both legs at every strike."""
    stamp = pd.Timestamp(stamp, tz="UTC") if pd.Timestamp(stamp).tz is None else pd.Timestamp(stamp)
    T = (EXPIRY - stamp).total_seconds() / (365.25 * 24 * 3600)
    rows = []
    for strike in (strikes or STRIKES):
        for option_type in ("C", "P"):
            mid = black76_price(forward, strike, T, sigma, option_type)
            rows.append({
                "timestamp": stamp, "expiry_ts": EXPIRY, "strike": strike,
                "option_type": option_type, "forward": forward, "T": T,
                "iv": sigma, "mid_usd": mid,
                "bid_usd": mid * (1 - half_spread),
                "ask_usd": mid * (1 + half_spread),
                "log_moneyness": float(np.log(strike / forward)),
            })
    return pd.DataFrame(rows)


def path(spots, start="2026-09-10 00:00", hours=4.0, **kwargs) -> list:
    return [
        snapshot(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=hours * i), s, **kwargs)
        for i, s in enumerate(spots)
    ]


class TestStrikeSelection:
    def test_picks_the_strike_nearest_the_forward(self):
        quotes = snapshot("2026-09-10", 81_000.0)
        assert DeltaHedgedStraddle.pick_atm_strike(quotes, EXPIRY) == 80_000.0

    def test_requires_both_legs_at_the_strike(self):
        # The nearest strike has only a call, so the next one out must win.
        quotes = snapshot("2026-09-10", 80_000.0)
        quotes = quotes.drop(
            quotes[(quotes.strike == 80_000.0) & (quotes.option_type == "P")].index
        )
        assert DeltaHedgedStraddle.pick_atm_strike(quotes, EXPIRY) != 80_000.0

    def test_returns_none_when_no_strike_has_two_sides(self):
        quotes = snapshot("2026-09-10", 80_000.0)
        assert DeltaHedgedStraddle.pick_atm_strike(
            quotes[quotes.option_type == "C"], EXPIRY
        ) is None


class TestEntry:
    def test_banks_the_bid_not_the_mid(self):
        quotes = snapshot("2026-09-10", 80_000.0, half_spread=0.05)
        strategy = DeltaHedgedStraddle()
        position = strategy.enter(quotes, EXPIRY, quotes["timestamp"].iloc[0])
        legs = quotes[quotes.strike == position.strike]
        assert position.premium_received == pytest.approx(legs["bid_usd"].sum())
        assert position.premium_received < legs["mid_usd"].sum()

    def test_entry_spread_cost_is_recorded(self):
        quotes = snapshot("2026-09-10", 80_000.0, half_spread=0.05)
        position = DeltaHedgedStraddle().enter(quotes, EXPIRY, quotes["timestamp"].iloc[0])
        assert position.entry_spread_cost > 0

    def test_no_spread_crossed_when_disabled(self):
        quotes = snapshot("2026-09-10", 80_000.0, half_spread=0.05)
        strategy = DeltaHedgedStraddle(costs=Costs(cross_spread=False))
        position = strategy.enter(quotes, EXPIRY, quotes["timestamp"].iloc[0])
        legs = quotes[quotes.strike == position.strike]
        assert position.premium_received == pytest.approx(legs["mid_usd"].sum())

    def test_atm_straddle_opens_close_to_delta_neutral(self):
        quotes = snapshot("2026-09-10", 80_000.0)
        position = DeltaHedgedStraddle().enter(quotes, EXPIRY, quotes["timestamp"].iloc[0])
        assert abs(position.hedge_units) < 0.2

    def test_pnl_starts_negative_by_the_spread_paid(self):
        # Selling at the bid and marking at mid means the position opens at a
        # loss equal to the half-spread. Anything else would manufacture
        # profit at t=0.
        quotes = snapshot("2026-09-10", 80_000.0, half_spread=0.05)
        strategy = DeltaHedgedStraddle()
        position = strategy.enter(quotes, EXPIRY, quotes["timestamp"].iloc[0])
        assert strategy.mark_to_market(position) < 0


class TestHedging:
    def test_short_gamma_buys_as_spot_rises(self):
        # A short straddle must buy the underlying into a rally and sell into
        # a fall -- buying high, selling low. That bleed is the gamma leg.
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        frames = path([80_000.0, 84_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        before = position.hedge_units
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.hedge_units > before

    def test_short_gamma_sells_as_spot_falls(self):
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        frames = path([80_000.0, 76_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        before = position.hedge_units
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.hedge_units < before

    def test_delta_band_rule_does_not_trade_on_a_small_move(self):
        strategy = DeltaHedgedStraddle(rebalance_rule="delta_band", delta_band=0.5)
        frames = path([80_000.0, 80_100.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.rebalances == 0

    def test_delta_band_rule_trades_on_a_large_move(self):
        strategy = DeltaHedgedStraddle(rebalance_rule="delta_band", delta_band=0.05)
        frames = path([80_000.0, 92_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.rebalances == 1

    def test_whalley_wilmott_holds_through_a_small_move(self):
        strategy = DeltaHedgedStraddle(rebalance_rule="whalley_wilmott")
        frames = path([80_000.0, 80_100.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.rebalances == 0

    def test_whalley_wilmott_trades_on_a_large_move(self):
        strategy = DeltaHedgedStraddle(rebalance_rule="whalley_wilmott")
        frames = path([80_000.0, 92_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.rebalances == 1

    def test_whalley_wilmott_trades_less_as_slippage_rises(self):
        """
        The whole reason to prefer it to a clock: the rule reads the cost it
        is economising on. A clock cannot, at any interval, and a band that
        ignored cost would look identical on every other test here.
        """
        spots = [80_000.0, 82_000.0, 79_000.0, 84_000.0, 81_000.0, 86_000.0]
        counts = []
        for slip in (0.5, 5.0, 50.0):
            strategy = DeltaHedgedStraddle(
                rebalance_rule="whalley_wilmott",
                costs=Costs(hedge_slippage_bps=slip))
            frames = path(spots)
            position = strategy.enter(frames[0], EXPIRY,
                                      frames[0]["timestamp"].iloc[0])
            for frame in frames[1:]:
                strategy.step(position, frame, frame["timestamp"].iloc[0])
            counts.append(position.rebalances)
        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1]

    def test_an_unknown_rebalance_rule_is_refused(self):
        """It used to fall through to the clock, so a typo in a study script
        ran a different rule than the one its report claimed."""
        strategy = DeltaHedgedStraddle(rebalance_rule="whalley_wilmot")
        frames = path([80_000.0, 92_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        with pytest.raises(ValueError, match="unknown rebalance rule"):
            strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])

    def test_fixed_interval_rule_waits_for_the_interval(self):
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=8.0)
        frames = path([80_000.0, 90_000.0], hours=4.0)
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.rebalances == 0

    def test_hedge_pnl_has_the_right_sign(self):
        # Long the hedge into a rally must make money on the hedge leg.
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        frames = path([80_000.0, 88_000.0, 96_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.hedge_units > 0
        strategy.step(position, frames[2], frames[2]["timestamp"].iloc[0])
        assert position.hedge_pnl > 0


class TestCosts:
    def test_slippage_scales_with_turnover(self):
        cheap = DeltaHedgedStraddle(rebalance_interval_hours=0.0,
                                    costs=Costs(hedge_slippage_bps=1.0))
        dear = DeltaHedgedStraddle(rebalance_interval_hours=0.0,
                                   costs=Costs(hedge_slippage_bps=50.0))
        frames = path([80_000.0, 88_000.0, 74_000.0])
        results = []
        for strategy in (cheap, dear):
            position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
            for frame in frames[1:]:
                strategy.step(position, frame, frame["timestamp"].iloc[0])
            results.append(position.slippage_paid)
        assert results[1] > results[0] * 10

    def test_a_long_hedge_pays_positive_funding(self):
        strategy = DeltaHedgedStraddle(
            rebalance_interval_hours=0.0, costs=Costs(funding_rate_8h=0.01)
        )
        frames = path([80_000.0, 95_000.0, 96_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        for frame in frames[1:]:
            strategy.step(position, frame, frame["timestamp"].iloc[0])
        assert position.hedge_units > 0
        assert position.funding_paid > 0

    def test_funding_is_zero_with_no_hedge_and_no_rate(self):
        strategy = DeltaHedgedStraddle(costs=Costs(funding_rate_8h=0.0))
        frames = path([80_000.0, 81_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.funding_paid == 0.0

    def test_costs_reduce_pnl(self):
        frames = path([80_000.0, 86_000.0, 78_000.0])
        outcomes = []
        for costs in (Costs(hedge_slippage_bps=0.0, funding_rate_8h=0.0,
                            cross_spread=False),
                      Costs(hedge_slippage_bps=50.0, funding_rate_8h=0.01)):
            strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0, costs=costs)
            position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
            for frame in frames[1:]:
                strategy.step(position, frame, frame["timestamp"].iloc[0])
            outcomes.append(strategy.mark_to_market(position))
        assert outcomes[1] < outcomes[0]


class TestAccounting:
    def test_flat_market_earns_the_premium_less_decay_of_nothing(self):
        # Spot unchanged and vol unchanged: P&L is the spread paid plus
        # whatever theta has accrued between the two timestamps.
        strategy = DeltaHedgedStraddle(costs=Costs(hedge_slippage_bps=0.0,
                                                   funding_rate_8h=0.0,
                                                   cross_spread=False))
        frames = path([80_000.0, 80_000.0], hours=24.0)
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        # Short an option through a day of no movement: theta is earned.
        assert strategy.mark_to_market(position) > 0

    def test_mark_to_market_is_the_sum_of_its_parts(self):
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        frames = path([80_000.0, 85_000.0, 82_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        for frame in frames[1:]:
            strategy.step(position, frame, frame["timestamp"].iloc[0])
        expected = (
            position.premium_received - position.option_value
            + position.hedge_pnl - position.funding_paid - position.slippage_paid
        )
        assert strategy.mark_to_market(position) == pytest.approx(expected)

    def test_closing_flattens_the_hedge(self):
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        frames = path([80_000.0, 88_000.0])
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        strategy.step(position, frames[1], frames[1]["timestamp"].iloc[0])
        strategy.close(position, frames[1], frames[1]["timestamp"].iloc[0])
        assert position.hedge_units == 0.0
        assert position.closed


class TestNoLookahead:
    def test_rejects_out_of_order_snapshots(self):
        frames = path([80_000.0, 81_000.0, 82_000.0])
        with pytest.raises(ValueError, match="increasing"):
            assert_no_lookahead([frames[0], frames[2], frames[1]])

    def test_rejects_duplicate_timestamps(self):
        frames = path([80_000.0, 81_000.0])
        with pytest.raises(ValueError, match="increasing"):
            assert_no_lookahead([frames[0], frames[0]])

    def test_rejects_a_snapshot_holding_two_timestamps(self):
        frames = path([80_000.0, 81_000.0])
        mixed = pd.concat(frames, ignore_index=True)
        with pytest.raises(ValueError, match="more than one timestamp"):
            assert_no_lookahead([mixed])

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError):
            assert_no_lookahead([])

    def test_realized_vol_uses_only_the_path_so_far(self):
        # The reported realized vol at step i must not change when later
        # steps are appended. If it did, the strategy would know at entry
        # how volatile the period it is about to trade turns out to be.
        calm = [80_000.0] * 6
        violent = calm + [60_000.0, 100_000.0]
        assert realized_vol(calm, 4.0) == pytest.approx(
            realized_vol(violent[: len(calm)], 4.0)
        )

    def test_backtest_reports_vol_that_only_ever_looks_backward(self):
        frames = path([80_000.0, 80_100.0, 79_900.0, 80_050.0, 95_000.0, 70_000.0])
        result = run_backtest(frames, DeltaHedgedStraddle(rebalance_interval_hours=0.0))
        vols = result["realized_vol"].dropna()
        # The crash is at the end, so early estimates must be far calmer.
        assert vols.iloc[0] < vols.iloc[-1]


class TestRealizedVol:
    def test_annualizes_with_365_not_252(self):
        rng = np.random.default_rng(0)
        # One year of daily steps with a known daily sigma.
        daily = 0.02
        spots = 80_000 * np.exp(np.cumsum(rng.normal(0, daily, 4000)))
        estimate = realized_vol(list(spots), hours_per_step=24.0)
        assert estimate == pytest.approx(daily * np.sqrt(365.0), rel=0.1)
        assert ANNUALIZE == pytest.approx(np.sqrt(365.0))

    def test_needs_at_least_three_points(self):
        assert np.isnan(realized_vol([80_000.0, 81_000.0], 4.0))

    def test_a_flat_path_has_zero_vol(self):
        assert realized_vol([80_000.0] * 8, 4.0) == pytest.approx(0.0)


class TestBacktestLoop:
    def test_produces_a_row_per_marked_snapshot(self):
        frames = path([80_000.0, 81_000.0, 82_000.0, 81_500.0])
        result = run_backtest(frames, DeltaHedgedStraddle())
        assert len(result) == len(frames) - 1
        assert {"timestamp", "pnl", "cumulative_pnl", "turnover"} <= set(result.columns)

    def test_pnl_is_a_running_level_not_a_per_step_increment(self):
        # `pnl` is mark-to-market, so in a dead-flat market it rises with
        # theta at every step. The final row is the exception: closing buys
        # the straddle back at the ask, so it drops by the exit half-spread.
        frames = path([80_000.0] * 6, hours=24.0)
        result = run_backtest(frames, DeltaHedgedStraddle())
        assert result["pnl"].iloc[:-1].is_monotonic_increasing

    def test_the_closing_step_pays_the_exit_spread(self):
        frames = path([80_000.0] * 6, hours=24.0, half_spread=0.02)
        held = run_backtest(frames, DeltaHedgedStraddle(), close_at_end=False)
        closed = run_backtest(frames, DeltaHedgedStraddle(), close_at_end=True)
        assert closed["pnl"].iloc[-1] < held["pnl"].iloc[-1]

    def test_theta_accrues_in_a_flat_market(self):
        frames = path([80_000.0] * 6, hours=24.0)
        result = run_backtest(frames, DeltaHedgedStraddle(), close_at_end=False)
        assert result["pnl"].iloc[-1] > result["pnl"].iloc[0]

    def test_closes_the_position_at_the_end(self):
        frames = path([80_000.0, 82_000.0, 81_000.0])
        result = run_backtest(frames, DeltaHedgedStraddle())
        assert result.attrs["position"].closed

    def test_a_calm_path_is_profitable_and_a_violent_one_is_not(self):
        # The whole thesis in one test: short vol wins when realized comes in
        # below implied and loses when it does not.
        calm = path([80_000.0 + 20 * np.sin(i) for i in range(12)], hours=4.0)
        violent = path(
            [80_000.0 * (1 + 0.06 * (-1) ** i) for i in range(12)], hours=4.0
        )
        strategy = lambda: DeltaHedgedStraddle(  # noqa: E731
            rebalance_interval_hours=4.0,
            costs=Costs(hedge_slippage_bps=0.0, funding_rate_8h=0.0),
        )
        calm_pnl = run_backtest(calm, strategy())["pnl"].iloc[-1]
        violent_pnl = run_backtest(violent, strategy())["pnl"].iloc[-1]
        assert calm_pnl > violent_pnl


class TestMetrics:
    def test_reports_the_headline_statistics(self):
        steps = pd.Series([10.0, -5.0, 20.0, -2.0, 8.0])
        out = compute_metrics(steps, hours_per_step=4.0)
        assert out["n"] == 5
        assert out["total_pnl"] == pytest.approx(31.0)
        assert out["hit_rate"] == pytest.approx(0.6)
        assert out["max_drawdown"] <= 0

    def test_drawdown_is_measured_from_the_running_peak(self):
        steps = pd.Series([100.0, -30.0, -20.0, 10.0])
        assert compute_metrics(steps)["max_drawdown"] == pytest.approx(-50.0)

    def test_handles_a_degenerate_series(self):
        assert compute_metrics(pd.Series([1.0]))["n"] == 1

    def test_summarize_carries_the_decomposition(self):
        frames = path([80_000.0, 81_000.0, 80_500.0, 80_800.0])
        result = run_backtest(frames, DeltaHedgedStraddle())
        out = summarize(result)
        assert {"option_pnl", "hedge_pnl", "funding_paid", "variance_gap"} <= set(out)


class TestDecomposition:
    def test_variance_gap_is_positive_when_realized_undershoots(self):
        frames = path([80_000.0, 80_100.0, 80_050.0])
        strategy = DeltaHedgedStraddle()
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        for frame in frames[1:]:
            strategy.step(position, frame, frame["timestamp"].iloc[0])
        assert decompose(position, realized_vol=0.20)["variance_gap"] > 0

    def test_variance_gap_is_negative_when_realized_overshoots(self):
        frames = path([80_000.0, 80_100.0, 80_050.0])
        strategy = DeltaHedgedStraddle()
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        for frame in frames[1:]:
            strategy.step(position, frame, frame["timestamp"].iloc[0])
        assert decompose(position, realized_vol=1.50)["variance_gap"] < 0

    def test_parts_add_up_to_the_total(self):
        frames = path([80_000.0, 83_000.0, 79_000.0])
        strategy = DeltaHedgedStraddle(rebalance_interval_hours=0.0)
        position = strategy.enter(frames[0], EXPIRY, frames[0]["timestamp"].iloc[0])
        for frame in frames[1:]:
            strategy.step(position, frame, frame["timestamp"].iloc[0])
        parts = decompose(position, realized_vol=0.5)
        assert parts["total_pnl"] == pytest.approx(
            parts["option_pnl"] + parts["hedge_pnl"]
            - parts["funding_paid"] - parts["slippage_paid"]
        )
