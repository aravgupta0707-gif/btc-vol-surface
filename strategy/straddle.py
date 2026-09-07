"""
Delta-hedged ATM straddle: sell an ATM straddle, hedge delta with BTC
perpetual futures, rebalance periodically. See BUILD_GUIDE.md section 7.

P&L is approximately (implied variance - realized variance) * vega --
being able to decompose backtest P&L this way is the core interview point.

Decisions to make explicit (don't leave implicit in code):
- rebalance rule: fixed time interval vs. delta-band threshold
- transaction costs: bid-ask on entry, hedging slippage per rebalance,
  funding rate paid/received on the perpetual hedge
"""


class DeltaHedgedStraddle:
    def __init__(self, surface, rebalance_rule: str = "fixed_interval", rebalance_interval_hours: float = 4.0):
        self.surface = surface
        self.rebalance_rule = rebalance_rule
        self.rebalance_interval_hours = rebalance_interval_hours

    def enter(self, K: float, T: float, as_of_ts) -> dict:
        """Sell ATM straddle at (K, T), record initial delta hedge."""
        raise NotImplementedError

    def rebalance(self, position: dict, as_of_ts) -> dict:
        """Check rebalance rule, adjust perpetual hedge if triggered,
        apply hedging slippage and funding cost."""
        raise NotImplementedError

    def mark_to_market(self, position: dict, as_of_ts) -> float:
        """Current P&L of the position given current surface + hedge."""
        raise NotImplementedError
