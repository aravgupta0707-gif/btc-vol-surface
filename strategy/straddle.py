"""
Delta-hedged ATM straddle: sell an ATM straddle, hedge delta with BTC
perpetual futures, rebalance periodically. See BUILD_GUIDE.md section 7.

P&L is approximately (implied variance - realized variance) * vega --
being able to decompose backtest P&L this way is the core interview point.

Decisions made explicit rather than left implicit in code:

**Rebalance rule.** Three, selectable: a fixed clock, a fixed delta band, and
the Whalley-Wilmott band. A band trades less in calm markets and more in
violent ones, which is the right shape for a short-gamma position, at the
cost of an extra parameter to justify. `whalley_wilmott` is the one the
archive study found actually beats a clock out of sample on BTC at realistic
costs (HANDOFF.md sections 12.1 and 12.2); its width comes from
`strategy.hedging.ww_band`, shared with the study so the two cannot diverge.

**Marking.** Positions are marked against the actual quotes in each snapshot,
not against the fitted surface. The surface is the strategy's view of fair
value and is used to choose the strike and to fill gaps; the market's own mid
is what you could actually close at, and a backtest that marks to its own
model will report the model's opinion rather than a P&L.

**Entry crosses the spread, marking does not.** We sell at the bid and mark
at mid thereafter, charging the other half-spread once on exit. Marking at
mid while entering at mid would manufacture a profit at t=0 out of nothing.

**Costs.** Three of them, all real in crypto and all separately reported:
bid-ask on entry and exit, slippage on every rehedge, and funding on the
perpetual hedge. The last is the one that gets forgotten -- a hedge held for
weeks pays or receives funding the whole time, and at typical crypto rates
that is not a rounding error against a 3-4 vol point edge.

**Sign convention.** `hedge_units` is the perpetual position in BTC, positive
long. A short straddle is short gamma, so as spot rises the hedge must be
bought and as it falls it must be sold -- buying high and selling low, every
time. That bleed is the gamma leg of the decomposition and it is what the
premium has to cover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600.0

REBALANCE_RULES = ("fixed_interval", "delta_band", "whalley_wilmott")


@dataclass
class Costs:
    """All in the units the venue actually charges."""

    # Slippage per rehedge, in basis points of the notional traded.
    hedge_slippage_bps: float = 5.0
    # Perpetual funding, per 8 hours, as a decimal. Longs pay when positive.
    # Deribit's public/get_funding_rate_history reports `interest_8h`.
    funding_rate_8h: float = 0.0001
    # Whether entry and exit cross the bid-ask.
    cross_spread: bool = True


@dataclass
class StraddleConfig:
    rebalance_rule: str = "fixed_interval"  # or "delta_band", "whalley_wilmott"
    rebalance_interval_hours: float = 4.0
    delta_band: float = 0.05
    # Risk aversion for the Whalley-Wilmott band. 1e-4 is what the blind
    # walk-forward picks on 58-74 of 70-74 inceptions (HANDOFF.md section
    # 12.2), not the 1e-3 that wins in hindsight -- taking the hindsight
    # value would reintroduce exactly the selection the walk-forward removed.
    risk_aversion: float = 1e-4
    quantity: float = 1.0  # straddles, each one call plus one put on 1 BTC
    costs: Costs = field(default_factory=Costs)


@dataclass
class Position:
    strike: float
    expiry_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    quantity: float

    premium_received: float          # USD, banked at entry
    entry_iv: float                  # the vol we sold
    entry_spot: float

    hedge_units: float = 0.0         # BTC, positive long
    hedge_pnl: float = 0.0           # realised mark-to-market on the hedge
    funding_paid: float = 0.0
    slippage_paid: float = 0.0
    entry_spread_cost: float = 0.0

    last_ts: pd.Timestamp | None = None
    last_spot: float | None = None
    last_rebalance_ts: pd.Timestamp | None = None
    rebalances: int = 0
    turnover: float = 0.0            # BTC traded in hedging, cumulative

    closed: bool = False
    option_value: float = 0.0        # USD, current cost to buy it back


class DeltaHedgedStraddle:
    def __init__(self, surface=None, rebalance_rule: str = "fixed_interval",
                 rebalance_interval_hours: float = 4.0, **kwargs):
        self.surface = surface
        self.config = StraddleConfig(
            rebalance_rule=rebalance_rule,
            rebalance_interval_hours=rebalance_interval_hours,
            **kwargs,
        )

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _legs(quotes: pd.DataFrame, expiry_ts, strike: float) -> pd.DataFrame:
        return quotes[
            (quotes["expiry_ts"] == expiry_ts) & (quotes["strike"] == strike)
        ]

    @staticmethod
    def pick_atm_strike(quotes: pd.DataFrame, expiry_ts) -> float | None:
        """
        The strike nearest the forward that has BOTH a call and a put quoted.

        Requiring both matters: a straddle needs two legs, and the strike
        closest to the forward frequently has only one side quoted on a thin
        expiry. Silently substituting a synthetic leg would understate cost.
        """
        chain = quotes[quotes["expiry_ts"] == expiry_ts]
        if chain.empty:
            return None
        both = (
            chain.groupby("strike")["option_type"].nunique().pipe(lambda s: s[s >= 2])
        )
        if both.empty:
            return None
        forward = float(chain["forward"].iloc[0])
        strikes = both.index.to_numpy(dtype=float)
        return float(strikes[int(np.argmin(np.abs(strikes - forward)))])

    def _straddle_value(self, quotes: pd.DataFrame, expiry_ts, strike: float,
                        side: str) -> tuple[float, float, float, float]:
        """
        (value, delta, gamma, iv) of one straddle from the snapshot's own
        quotes.

        `side` is 'bid' when we are selling into the market, 'mid' for
        marking, 'ask' when buying back.

        Gamma is here for the Whalley-Wilmott band, which needs it at every
        snapshot. Note it comes from each leg's *current* market vol rather
        than the vol the straddle was sold at, so the band responds to the
        smile the book actually faces.
        """
        legs = self._legs(quotes, expiry_ts, strike)
        if len(legs) < 2:
            return np.nan, np.nan, np.nan, np.nan
        column = {"bid": "bid_usd", "mid": "mid_usd", "ask": "ask_usd"}[side]
        value = float(legs[column].sum())
        # Delta of the straddle we are short is -(delta_call + delta_put).
        # Reported here as the long-straddle delta; the caller applies sign.
        from pricing.black76 import greeks as black76_greeks

        delta = gamma = 0.0
        for leg in legs.itertuples():
            leg_greeks = black76_greeks(
                leg.forward, leg.strike, leg.T, leg.iv, leg.option_type
            )
            delta += leg_greeks["delta"]
            gamma += leg_greeks["gamma"]
        return value, float(delta), float(gamma), float(legs["iv"].mean())

    # ---- lifecycle ------------------------------------------------------

    def enter(self, quotes: pd.DataFrame, expiry_ts, as_of_ts,
              strike: float | None = None) -> Position | None:
        """Sell an ATM straddle at (K, expiry), record the initial delta hedge."""
        strike = strike if strike is not None else self.pick_atm_strike(quotes, expiry_ts)
        if strike is None:
            return None

        side = "bid" if self.config.costs.cross_spread else "mid"
        received, long_delta, _, iv = self._straddle_value(
            quotes, expiry_ts, strike, side
        )
        mid_value, _, _, _ = self._straddle_value(
            quotes, expiry_ts, strike, "mid"
        )
        if not np.isfinite(received) or not np.isfinite(long_delta):
            return None

        quantity = self.config.quantity
        spot = float(self._legs(quotes, expiry_ts, strike)["forward"].iloc[0])
        position = Position(
            strike=strike,
            expiry_ts=expiry_ts,
            entry_ts=as_of_ts,
            quantity=quantity,
            premium_received=received * quantity,
            entry_iv=iv,
            entry_spot=spot,
            entry_spread_cost=max(mid_value - received, 0.0) * quantity,
            last_ts=as_of_ts,
            last_spot=spot,
            last_rebalance_ts=as_of_ts,
            option_value=mid_value * quantity,
        )
        # Short straddle delta is -(long delta); hedge is its negative.
        self._set_hedge(position, target_units=long_delta * quantity, spot=spot,
                        charge_slippage=True)
        return position

    def _set_hedge(self, position: Position, target_units: float, spot: float,
                   charge_slippage: bool) -> None:
        traded = abs(target_units - position.hedge_units)
        if charge_slippage and traded > 0:
            position.slippage_paid += (
                traded * spot * self.config.costs.hedge_slippage_bps / 10_000.0
            )
        position.turnover += traded
        position.hedge_units = target_units

    def should_rebalance(self, position: Position, as_of_ts,
                         current_delta: float, gamma: float | None = None,
                         forward: float | None = None) -> bool:
        """
        Has the hedge drifted far enough -- or waited long enough -- to trade?

        `current_delta` is the drift: the target delta less the hedge held.
        The band rules read it directly; the clock ignores it entirely.
        """
        rule = self.config.rebalance_rule
        if rule == "fixed_interval":
            elapsed = (
                pd.Timestamp(as_of_ts) - pd.Timestamp(position.last_rebalance_ts)
            ).total_seconds() / SECONDS_PER_HOUR
            return elapsed >= self.config.rebalance_interval_hours - 1e-9
        if rule == "delta_band":
            return abs(current_delta) > self.config.delta_band
        if rule == "whalley_wilmott":
            if gamma is None or forward is None:
                raise ValueError(
                    "the Whalley-Wilmott band needs gamma and the forward; "
                    "call should_rebalance from step(), which supplies both"
                )
            return abs(current_delta) > self.band(position, gamma, forward)
        raise ValueError(
            f"unknown rebalance rule {rule!r}, expected one of {REBALANCE_RULES}"
        )

    def band(self, position: Position, gamma: float, forward: float) -> float:
        """
        The Whalley-Wilmott half-width for this position, right now.

        Shared with `strategy.hedging`, so the rule the archive study
        measured and the rule the live book runs cannot drift apart.

        `gamma` arrives per straddle and is scaled to the whole position
        before the formula sees it, because the band is a tolerance on the
        position's own delta drift -- which is what `step` compares it
        against. H then grows as q^(2/3) while the drift grows as q, so a
        larger book tolerates proportionally less drift. That is the
        formula's shape rather than an approximation of it.
        """
        from strategy.hedging import ww_band

        return float(ww_band(forward, gamma * position.quantity,
                             self.config.risk_aversion,
                             self.config.costs.hedge_slippage_bps))

    def step(self, position: Position, quotes: pd.DataFrame, as_of_ts) -> dict:
        """
        Advance one snapshot: accrue hedge P&L and funding, then rebalance if
        the rule fires. Uses only information in this snapshot -- no lookahead.
        """
        legs = self._legs(quotes, position.expiry_ts, position.strike)
        if legs.empty:
            return {"marked": False}

        spot = float(legs["forward"].iloc[0])
        hours = (
            pd.Timestamp(as_of_ts) - pd.Timestamp(position.last_ts)
        ).total_seconds() / SECONDS_PER_HOUR

        # Hedge P&L on the position carried into this step, then funding on
        # the notional held over the interval. A long perp pays when funding
        # is positive.
        position.hedge_pnl += position.hedge_units * (spot - position.last_spot)
        position.funding_paid += (
            position.hedge_units * spot
            * self.config.costs.funding_rate_8h * (hours / 8.0)
        )

        mid_value, long_delta, gamma, iv = self._straddle_value(
            quotes, position.expiry_ts, position.strike, "mid"
        )
        if np.isfinite(mid_value):
            position.option_value = mid_value * position.quantity

        net_delta = np.nan
        rebalanced = False
        if np.isfinite(long_delta):
            target = long_delta * position.quantity
            net_delta = target - position.hedge_units
            if self.should_rebalance(position, as_of_ts, net_delta,
                                     gamma=gamma, forward=spot):
                self._set_hedge(position, target, spot, charge_slippage=True)
                position.last_rebalance_ts = as_of_ts
                position.rebalances += 1
                rebalanced = True

        position.last_ts = as_of_ts
        position.last_spot = spot
        return {
            "marked": True, "spot": spot, "iv": iv,
            "net_delta": net_delta, "rebalanced": rebalanced,
            "pnl": self.mark_to_market(position),
        }

    def close(self, position: Position, quotes: pd.DataFrame, as_of_ts) -> float:
        """Buy the straddle back at the ask and flatten the hedge."""
        side = "ask" if self.config.costs.cross_spread else "mid"
        value, _, _, _ = self._straddle_value(
            quotes, position.expiry_ts, position.strike, side
        )
        mid_value, _, _, _ = self._straddle_value(
            quotes, position.expiry_ts, position.strike, "mid"
        )
        if np.isfinite(value):
            position.option_value = value * position.quantity
            position.entry_spread_cost += max(value - mid_value, 0.0) * position.quantity
        legs = self._legs(quotes, position.expiry_ts, position.strike)
        if not legs.empty:
            spot = float(legs["forward"].iloc[0])
            self._set_hedge(position, 0.0, spot, charge_slippage=True)
        position.closed = True
        return self.mark_to_market(position)

    def mark_to_market(self, position: Position, as_of_ts=None) -> float:
        """
        Current P&L of the position in USD.

        Premium banked, less what it would cost to buy the straddle back,
        plus the hedge's accumulated P&L, less every cost charged so far.
        """
        return (
            position.premium_received
            - position.option_value
            + position.hedge_pnl
            - position.funding_paid
            - position.slippage_paid
        )


def decompose(position: Position, realized_vol: float) -> dict:
    """
    Split the P&L into the pieces BUILD_GUIDE.md section 7 asks about.

    The theoretical result for a delta-hedged short option is that P&L is
    approximately

        0.5 * vega_notional * (implied_variance - realized_variance)

    so a straddle sold at `entry_iv` and hedged through realized volatility
    `realized_vol` should earn roughly the variance gap times the position's
    sensitivity. This is the number the backtest has to be checked against:
    if realised P&L does not track it, the hedging or the cost logic is
    wrong and the Sharpe ratio is meaningless.
    """
    variance_gap = position.entry_iv**2 - realized_vol**2
    years = max(
        (pd.Timestamp(position.expiry_ts) - pd.Timestamp(position.entry_ts))
        .total_seconds() / (365.25 * 24 * 3600),
        1e-12,
    )
    held = max(
        (pd.Timestamp(position.last_ts) - pd.Timestamp(position.entry_ts))
        .total_seconds() / (365.25 * 24 * 3600),
        0.0,
    )
    # Vega of an ATM straddle is about 0.8 * S * sqrt(T) per unit of vol.
    vega_notional = 0.8 * position.entry_spot * np.sqrt(years) * position.quantity
    return {
        "entry_iv": position.entry_iv,
        "realized_vol": realized_vol,
        "variance_gap": variance_gap,
        "theoretical_pnl": 0.5 * vega_notional * variance_gap
        * (held / years if years else 0.0) / max(position.entry_iv, 1e-12),
        "option_pnl": position.premium_received - position.option_value,
        "hedge_pnl": position.hedge_pnl,
        "funding_paid": position.funding_paid,
        "slippage_paid": position.slippage_paid,
        "entry_spread_cost": position.entry_spread_cost,
        "total_pnl": (
            position.premium_received - position.option_value
            + position.hedge_pnl - position.funding_paid - position.slippage_paid
        ),
    }
