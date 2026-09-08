"""
Walk-forward backtest loop. See BUILD_GUIDE.md section 8.

Critical rule: at each timestamp, only use data that would have been
available at that timestamp. This is the most common way backtests
silently lie -- worth a dedicated test that deliberately tries to feed
future data in and confirms it gets rejected.

Sanity check before trusting any Sharpe number: realized P&L should
roughly track the theoretical (implied - realized variance) * vega
estimate. If it doesn't, the hedging or cost logic is wrong.

The no-lookahead rule is enforced structurally rather than by convention.
`run_backtest` sorts snapshots by timestamp and hands the strategy exactly
one at a time; the strategy's `step` never receives the sequence, so it
cannot see forward even by accident. `assert_no_lookahead` checks the input
is well formed before any of it is used.

There is a second, subtler leak this guards against: realized volatility is
computed only from spot observed strictly before the current timestamp.
Computing it over the whole path -- the natural thing to write -- would tell
the strategy at entry how volatile the period it is about to trade turned
out to be, which is the entire quantity the trade is betting on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZE = np.sqrt(365.0)  # 24/7 market, as in the VRP study


def assert_no_lookahead(snapshots: list) -> None:
    """
    Check the snapshot sequence is usable for a walk-forward run.

    Raises rather than warns: a backtest on out-of-order or duplicated
    timestamps produces a number that looks fine and means nothing.
    """
    if not snapshots:
        raise ValueError("no snapshots supplied")
    stamps = [pd.Timestamp(frame["timestamp"].iloc[0]) for frame in snapshots]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("snapshots must be strictly increasing in time")
    for frame, stamp in zip(snapshots, stamps):
        if frame["timestamp"].nunique() != 1:
            raise ValueError(f"snapshot at {stamp} holds more than one timestamp")


def realized_vol(spots: list, hours_per_step: float) -> float:
    """
    Annualized close-to-close volatility of the spot path so far.

    Uses sqrt(365), not sqrt(252): crypto trades continuously, so there are
    no non-trading days to exclude. The equity convention would overstate
    realized vol by 20%, which is larger than the premium being measured.
    """
    if len(spots) < 3:
        return float("nan")
    returns = np.diff(np.log(np.asarray(spots, dtype=float)))
    steps_per_year = 365.0 * 24.0 / max(hours_per_step, 1e-9)
    return float(np.std(returns, ddof=1) * np.sqrt(steps_per_year))


def run_backtest(snapshots: list, strategy, expiry_ts=None,
                 close_at_end: bool = True) -> pd.DataFrame:
    """
    Walk through snapshots in time order, entering/rebalancing/marking
    positions per the strategy, applying transaction costs at each step.

    Returns a DataFrame with at least: timestamp, pnl, cumulative_pnl,
    turnover.
    """
    assert_no_lookahead(snapshots)

    first = snapshots[0]
    if expiry_ts is None:
        # The nearest expiry that still has quotes in the LAST snapshot, so
        # the position does not expire mid-run. Chosen from the first and
        # last snapshots only, both of which a trader would know at entry
        # (the expiry calendar is public and fixed in advance).
        live = set(snapshots[-1]["expiry_ts"].unique())
        candidates = sorted(e for e in first["expiry_ts"].unique() if e in live)
        if not candidates:
            raise ValueError("no expiry survives the whole window")
        expiry_ts = candidates[0]

    position = strategy.enter(first, expiry_ts, pd.Timestamp(first["timestamp"].iloc[0]))
    if position is None:
        raise ValueError("could not open a straddle in the first snapshot")

    hours_per_step = _median_step_hours(snapshots)
    rows, spots = [], [position.entry_spot]
    for index, frame in enumerate(snapshots[1:], start=1):
        stamp = pd.Timestamp(frame["timestamp"].iloc[0])
        last = close_at_end and index == len(snapshots) - 1
        state = (
            {"marked": True, "pnl": strategy.close(position, frame, stamp)}
            if last
            else strategy.step(position, frame, stamp)
        )
        if not state.get("marked"):
            continue
        if "spot" in state:
            spots.append(state["spot"])

        rows.append({
            "timestamp": stamp,
            "spot": position.last_spot,
            "pnl": state["pnl"],
            "option_pnl": position.premium_received - position.option_value,
            "hedge_pnl": position.hedge_pnl,
            "funding_paid": position.funding_paid,
            "slippage_paid": position.slippage_paid,
            "hedge_units": position.hedge_units,
            "turnover": position.turnover,
            "rebalances": position.rebalances,
            "net_delta": state.get("net_delta", np.nan),
            # Strictly backward-looking: only spot seen up to and including
            # this step. Using the full path would leak the answer.
            "realized_vol": realized_vol(spots, hours_per_step),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["cumulative_pnl"] = frame["pnl"]
    frame["step_pnl"] = frame["pnl"].diff().fillna(frame["pnl"].iloc[0])
    frame.attrs["position"] = position
    frame.attrs["hours_per_step"] = hours_per_step
    return frame


def _median_step_hours(snapshots: list) -> float:
    stamps = pd.Series([pd.Timestamp(f["timestamp"].iloc[0]) for f in snapshots])
    steps = stamps.diff().dropna().dt.total_seconds() / 3600.0
    return float(steps.median()) if len(steps) else 1.0


def compute_metrics(pnl_series: pd.Series, hours_per_step: float = 1.0,
                    notional: float | None = None) -> dict:
    """Sharpe ratio, max drawdown, hit rate, turnover."""
    steps = pd.Series(pnl_series).dropna()
    if len(steps) < 2:
        return {"n": int(len(steps))}

    periods_per_year = 365.0 * 24.0 / max(hours_per_step, 1e-9)
    mean, sigma = steps.mean(), steps.std(ddof=1)
    equity = steps.cumsum()
    peak = equity.cummax()

    out = {
        "n": int(len(steps)),
        "total_pnl": float(equity.iloc[-1]),
        "mean_step_pnl": float(mean),
        "std_step_pnl": float(sigma),
        # Annualised from the step frequency. On a few dozen steps this is a
        # descriptive statistic, not an estimate anyone should size on.
        "sharpe": float(mean / sigma * np.sqrt(periods_per_year)) if sigma > 0 else np.nan,
        "max_drawdown": float((equity - peak).min()),
        "hit_rate": float((steps > 0).mean()),
        "worst_step": float(steps.min()),
        "best_step": float(steps.max()),
    }
    if notional:
        out["return_on_notional"] = float(equity.iloc[-1] / notional)
    return out


def summarize(frame: pd.DataFrame) -> dict:
    """Metrics plus the P&L decomposition for a completed run."""
    from strategy.straddle import decompose

    if frame.empty:
        return {}
    position = frame.attrs.get("position")
    hours = frame.attrs.get("hours_per_step", 1.0)
    metrics = compute_metrics(frame["step_pnl"], hours,
                              notional=position.entry_spot if position else None)
    if position is not None:
        metrics.update(decompose(position, float(frame["realized_vol"].iloc[-1])))
        metrics["rebalances"] = int(position.rebalances)
        metrics["turnover_btc"] = float(position.turnover)
    return metrics
