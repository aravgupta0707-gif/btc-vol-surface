# Long ETH vol, short BTC vol: the dispersion trade with real costs

HANDOFF §10.3 found the largest effect in the project with a rough
calculation: relative to what each then realises, ETH 30-day implied vol is
cheap against BTC's. Selling the spread (short ETH vol / long BTC vol, equal
vega) lost **-3.65 vol points a month (t -2.75)**, and **-6.14 (t -4.56)**
when entered only while the ETH-BTC implied spread sat below its own
expanding median. That was implied-minus-realised arithmetic: no hedging
path, no spreads, no slippage.

This file states, **before running anything**, how the trade is built
properly and what the result has to look like to count. Results are appended
after the run.

## The book

At each monthly archive inception (Tardis free tier, 2020-26, about 74
months with both assets), open:

- **long** one 30-day ATM straddle on ETH, delta-hedged;
- **short** one 30-day ATM straddle on BTC, delta-hedged;
- equal spot notional. For an ATM straddle, vega in bps of spot per vol point
  is `2 * phi(sigma sqrt(T)/2) * sqrt(T)`, which depends on sigma only through
  a term that differs by under 1% between the two assets at 30 days, so equal
  notional *is* vega-neutral to that accuracy. The ratio is reported, not
  assumed.

Each leg is exactly a cell of the hedging sweep (`hedge_frequency.py`,
`data/research/hedge_cells/`): the same fitted surface, the same hourly index
path, the same per-strike entry spread measured from the archive's own book,
the same hedge simulation. The short leg's net is the cell's `net_bps`. The
long leg is the mirror: `-gross - spread - slippage`, since a buyer pays the
spread and the hedge costs too. Nothing is re-simulated; this is a
re-aggregation, so the legs cannot drift from the published §12 numbers.

**Primary specification, fixed now:** 30-day ATM, both legs hedged on a
**4-hour clock** (the best fixed clock in README's table, and a rule chosen
without reference to this trade), **5 bps** hedge slippage. Book P&L in bps
of one leg's notional, one observation per inception. 30-day positions opened
monthly barely overlap, so plain t-statistics are used.

**Entry rule, fixed now:** trade a month only if the ETH-BTC 30-day ATM
implied spread on that day is **below the median of the spread on all
earlier inception days** (expanding, at least 6 prior months). Uses nothing
not known at entry.

## Predictions

**D1. Gross of costs, the trade earns, unconditionally.** Mean gross book
P&L positive with t > 2. Scale: 30-day ATM straddle vega is about 23 bps of
spot per vol point, so §10.3's 3.65 vol points maps to roughly +80 bps; a
hedged path is not the same as implied-minus-realised, so anything from +40
to +130 is consistent.

**D2. After costs it survives only on the entry rule.** Costs are two
spreads plus two hedging bills, roughly 30-80 bps a month in total. So:
- unconditional net: positive, t < 2;
- **conditional net (entry rule): positive with t > 2.** This is the claim.

**D3. It is not one era.** The conditional net has the same sign in 2020-22
and 2023-26.

**D4. It is relative value, not just "ETH vol is cheap".** If the long-ETH
leg alone earns as much as the book, the BTC leg is only adding cost and the
trade is an outright long-ETH-vol bet. Prediction: the conditional book earns
more than the ETH leg alone, i.e. the short BTC leg contributes positively
(BTC 30-day ATM carries a premium, §10.2).

**What counts against:**
- D1 fails: gross not distinguishable from zero. Then §10.3 was an artefact of
  implied-minus-realised arithmetic that a hedged path does not reproduce.
- D2 fails: conditional net t < 2. Then it is a gross effect that costs eat,
  like the straddle.
- D3 fails: sign differs by era. Then it is a regime, not a structure.

## Known issues, stated up front

- About 74 months, and roughly half of them pass the entry rule. Power is low
  and a single bad month can move the mean a lot; the median and the worst
  month are reported alongside.
- The sweep grid is 16 rules x 4 slippages x 2 tenors. Only the primary
  specification above is the test; the grid is reported as robustness, and a
  better-looking cell in it is not evidence.
- The expanding median needs warm-up months, which drops the earliest
  inceptions (2020, the highest-vol period) from the conditional sample.
- Straddle P&L is not pure vega: a 30-day hedged straddle is a gamma bet
  along one path. The two assets' paths are correlated (~0.8), which is what
  makes the relative trade lower-risk than either leg; the realised
  correlation is reported.
- This is a test on cells that were themselves already looked at in §12, for
  the BTC leg. The ETH-long leg and the combination have not been.
