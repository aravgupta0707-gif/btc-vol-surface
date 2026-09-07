"""
Walk-forward backtest loop. See BUILD_GUIDE.md section 8.

Critical rule: at each timestamp, only use data that would have been
available at that timestamp. This is the most common way backtests
silently lie -- worth a dedicated test that deliberately tries to feed
future data in and confirms it gets rejected.

Sanity check before trusting any Sharpe number: realized P&L should
roughly track the theoretical (implied - realized variance) * vega
estimate. If it doesn't, the hedging or cost logic is wrong.
"""


def run_backtest(snapshots: list, strategy) -> "pandas.DataFrame":
    """
    Walk through snapshots in time order, entering/rebalancing/marking
    positions per the strategy, applying transaction costs at each step.

    Returns a DataFrame with at least: timestamp, pnl, cumulative_pnl,
    turnover.
    """
    raise NotImplementedError


def compute_metrics(pnl_series: "pandas.Series") -> dict:
    """Sharpe ratio, max drawdown, hit rate, turnover."""
    raise NotImplementedError
