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

## Results

Run: `python scripts/dispersion.py` (seconds; reads the sweep's cells). The
predictions above were committed (`751e699`) before the script existed.

**Verdict: not a trade with straddles.** The relative-value effect is real
in vol points and reproduces in gross P&L at the predicted size, but two
spreads and two hedging bills cost more than it earns, the entry rule adds
nothing once P&L runs along a hedged path, and the book's profit comes from
the BTC short, not from ETH being cheap.

Primary specification (30d ATM, 4-hour clock, 5 bps slippage), 74 paired
months 2020-04 to 2026-08, bps of one leg's spot notional:

| | Prediction | Result |
|---|---|---|
| D1 | gross positive, t > 2, +40 to +130 | **+92.5, t 1.90.** Size as predicted, t just short |
| D2 | net positive unconditionally (t < 2), **t > 2 on the entry rule** | **Fails.** Net -29.5 (t -0.62) every month; **-28.3 (t -0.66)** on the rule (n=31); rule-off months -58.8 |
| D3 | same sign in both eras | Net negative in both on the rule (-27, -29); moot |
| D4 | book beats the ETH leg alone (BTC leg adds) | **The other way round from the premise.** Rule-on: long ETH alone -41.5, short BTC alone +13.2. Every month: long ETH **-92.8**, short BTC **+63.3** |

Costs, both legs: **122 bps a month** — spreads 36 (BTC 15.6, ETH 20.4) and
hedge slippage 87 at a 4-hour clock. Even at zero slippage, spreads alone
take the book from +92 gross to +57 net (t 1.19). Vega ratio ETH/BTC 0.998
(0.991-1.000): equal notional is vega-neutral, as assumed. Realised BTC/ETH
return correlation over each position: 0.85.

**Robustness (not the test).** Across 64 rule x slippage cells at 30 days,
rule-on net is positive in 52% and has t > 2 in **none**. The best cells
(e.g. `ww 0.0001` at 10 bps, +48) are single noisy cells in a grid of 64.
The 7-day book has no gross to speak of (-2 to +49 bps).

### Why §10.3 looked so much stronger

§10.3 measured `(ETH implied - realised) - (BTC implied - realised)` in vol
points. On these same 74 months that measure is **-4.01 vol pts (t -3.11)**,
and **-5.96 (t -4.73)** on the entry rule's 31 months — the rule flags the
same months as §10.3's (99% agreement). So the vol-point effect is solid. It
maps to book gross at **24 bps per vol point** (regression slope; predicted
23), correlation 0.65. Three things then go wrong between vol points and money:

1. **A straddle is a poor instrument for a variance spread.** Its hedged P&L
   is gamma along one path: once spot drifts from the strike, realised
   variance stops paying. On the rule's months the vol points imply +145 bps
   and the straddles delivered +88. The entry rule's extra 2 vol points
   disappear into that noise (gross +88 on the rule vs +92 every month).
2. **The legs do not hedge each other well.** Hedged P&L correlates only
   about 0.65 across the two legs (from the variances: BTC sd 299, ETH 546,
   book 418), far below the 0.85 return correlation. Adding the long ETH leg
   *raises* the book's dispersion above the BTC short's alone. A minimum-
   variance hedge would hold about a third of the ETH notional, not one-for-one.
3. **Costs double.** Two spreads and two hedge bills, against one premium.

### What it means

The finding in §10.3 is better read as **"BTC 30-day vol carries a premium
and ETH's carries much less"** than as "ETH vol is cheap". In seller gross,
BTC ATM earns +121 bps a month (t 3.5) and ETH +28 (t 0.4). Being long ETH vol
on its own loses after costs (-93 net); as a hedge for the BTC short it costs
more than it removes in risk.

The instrument that would capture the vol-point effect is a **variance swap
spread**, not a straddle spread: it pays realised variance regardless of
where spot goes. Deribit lists no variance swaps, but a 30-day variance swap
is replicable from the strip of OTM options the surface already prices, and
the archive already holds those strikes and their spreads. That is the next
test, and the obvious risk is the same one in sharper form: replication needs
the wings, and wing spreads on Deribit are wide.

---

## Part 2: the same trade with variance swaps. Written before testing

The straddle test found the vol-point effect real (t -3.1) but the instrument
wrong: a straddle's hedged P&L is gamma along one path, so its noise swamps
the effect, and its two legs hedge each other poorly. A variance swap pays
realised variance whatever path spot takes. Deribit lists none, so this
replicates one from the options the surface prices.

**Construction, per asset, per monthly archive inception, 30 days:**

- **Strike.** The model-free variance strike from the fitted surface:
  `K_var = (2/T) * sum_i (dK_i / K_i^2) * OTM_i`, undiscounted, puts below
  the forward and calls above, on a fine strike grid across the 30-day
  slice's **quoted range only**. Truncating there under-states the full
  strike; the truncated strike is the one a replicating book can actually
  buy, so it is the one used. The share of the full (clamped-wing) strike
  that the truncation drops is reported.
- **Payoff.** Realised variance from the hourly index, on **4-hour** returns
  (the hedge interval, so the discretely monitored variance the dynamic leg
  actually earns), annualised, over the same 30 days. Long P&L in variance
  units is `RV - K_var`; reported per unit **vega notional**, i.e. divided by
  `2 sqrt(K_var)`, which is vol points.
- **Costs, both paid by either side.** (a) The strip's spread, paid once at
  entry: `(2/T) * sum_i (dK_i / K_i^2) * half_spread_i`, each half-spread
  from the archive's measured curve (`hedge_fits/spreads_all`, vol units
  times Black vega). (b) The futures hedge of the log contract, rebalanced
  every 4 hours: holding `(2/T)(1/F_t - 1/F_0)` per unit variance notional,
  so turnover is `(2/T) * sum |1/F_t - 1/F_{t-1}| * F_t`, charged at 5 bps.

**The book:** long ETH variance, short BTC variance, equal vega notional.

**Predictions.**

- **V1.** The single-leg premium: short BTC variance earns more than short
  ETH variance, gross, by a margin in the same direction as the straddle
  legs (BTC +121 bps, ETH +28).
- **V2.** Gross book P&L positive with **t > 2** — higher than the straddle
  book's 1.90, because the path noise is gone. Expected size near the
  ATM vol-point gap (-4.0 vol pts), adjusted for skew: the variance strike
  includes the wings, which are steeper on BTC in 2023-26.
- **V3.** The legs hedge each other better than the straddles did: the
  correlation of the two legs' P&L above the straddles' 0.65.
- **V4.** Net of costs, positive. Costs are expected to be **lower** than the
  straddle book's 122 bps-equivalent, because OTM wing options have small
  absolute spreads and the hedge trades on 4-hour returns. This is the
  least certain prediction: wing spreads in vol terms are wide.

**What counts against:** V2 fails (gross t < 2) — the noise was not the
problem, the effect is just small in money; or V4 fails — the variance
instrument is cleaner but replicating it costs as much as the effect.

**Known issues, stated up front:**
- Truncation: the quoted range on a 30-day slice spans roughly ±1.5-2
  standard deviations, so the replicated strike misses some wing variance,
  and a big move beyond the range realises variance the strip never bought.
  That makes long variance look better and short variance worse in crash
  months; it is reported, not corrected.
- The same 74 months, already looked at in Part 1. This is a different
  instrument on the same sample, not new data.
- Costs are modelled from measured half-spreads, not from executed trades.
