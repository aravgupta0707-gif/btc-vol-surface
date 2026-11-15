# Inverse vs linear: do Deribit's two BTC option books stay tied under stress?

Spin-off D (HANDOFF §8). Deribit lists BTC options twice. The **inverse** book
(`BTC-...`) is margined, quoted and settled in BTC. The **linear** book
(`BTC_USDC-...`) does all three in USDC. Converted to dollars, the two pay the
same thing at expiry: an inverse put pays (K - S)+ / S coins, which is worth
(K - S)+ dollars. So the same strike and expiry should imply the same vol in
both books. HANDOFF finding 8 found that on one calm snapshot: 130 shared
strikes, agreeing to 0.124 vol points.

What differs is the collateral. A seller of inverse puts holds BTC margin
against a short put, so a crash hits them twice: the put loses money and so
does the margin. A seller of linear puts holds USDC and is hit once. If the
books come apart anywhere, it should be the downside in a sell-off.

Track A has 939 pairs of hourly snapshots, one from each book, taken seconds
apart (2026-09-07 to 2026-11-15). That window includes the November sell-off.

**Written 2026-11-15, before any comparison was run.** I had seen only
metadata: file counts, timestamps, column names and expiries listed.

## Method, fixed in advance

- **Pairing.** Each BTC snapshot is matched to the USDC snapshot nearest in
  time, provided it is within 120 s.
- **Quotes.** Both books go through `data.clean.clean_snapshot`, so each is
  inverted with its own forward. Matching is on (expiry, strike, type). Only
  OTM quotes count, both books must be two-sided, and T must be at least
  2 days.
- **Price floor.** The cleaner's MIN_MID_BTC floor is 0.0005 in the quote
  currency. That bites on the inverse book (BTC) but is meaningless on the
  linear one (USDC). Both books get the same dollar floor instead: 0.0005 ×
  forward.
- **Vol spread.** Each side is inverted separately and the vol-space width
  is ask iv minus bid iv. The **clean set** is matched quotes whose width is
  at most 3 vol points in *both* books. Results are reported on the clean set
  and on all two-sided matches.
- **Per hour.** The median signed difference, linear iv minus inverse iv, and
  the median absolute difference. Also the forward gap: F_linear / F_inverse
  - 1, in bps.
- **Stress.** Trailing 24h realised vol from the inverse book's front
  forward, sampled hourly. A stress hour is in the top decile of that over
  the window, and a calm hour is in the bottom half. Second measure: ATM
  30-day iv from the same snapshot.
- **Inference.** Newey-West, 24 lags, on hourly series. Hours are not
  independent, and the window has perhaps a handful of independent stress
  episodes. Any t-stat here overstates what we know.

## Predictions

1. **Calm replicates the snapshot.** In calm hours the median absolute
   difference on the clean set is at most 0.25 vol points. The median signed
   difference is within ±0.10.
2. **Stress widens it.** The median absolute difference in stress hours is
   at least 1.5× the calm level. Regressing it hourly on trailing realised
   vol gives a slope with NW t > 2.
3. **Most of the widening is microstructure.** Adding the linear book's
   median vol-space width to that regression shrinks the realised-vol slope
   by more than half. The thin book (about $109k a day, against $20.8M
   inverse) quotes wider when vol is high, and wider quotes give noisier
   mids.
4. **Direction: inverse downside gets richer in stress.** On OTM puts the
   signed difference (linear - inverse) is more negative in stress hours than
   in calm ones, with NW t < -2 on the realised-vol slope. On OTM calls there
   is no such effect: |t| < 2. This is the margin effect described above.
5. **The books almost never cross.** A linear bid above an inverse ask (in
   dollars), or the reverse, occurs on fewer than 1% of matched clean quotes.
   Crossings are at least twice as frequent in stress hours as in calm ones.
6. **Forwards agree.** The forward gap has a median within ±5 bps, and the
   largest hourly |gap| is under 25 bps.

If prediction 2 fails, the books held together through the sell-off, and that
is the finding. If 2 holds but 3 also holds, the widening is the thin book's
quoting, not a pricing disagreement. Only 4 would show that settlement
currency moves price.

## Results (2026-11-15)

**The books held together.** BTC fell from 87k to 62k inside the window, and
trailing 24h realised vol peaked at 104%. Through all of it the two books
agreed to the same 0.13 vol points as on a calm day. Settlement currency did
not move price.

Data: 939 pairs, all matched, with 171,723 matched quotes (89% clean) and a
median of 161 clean matches an hour. Trailing realised vol: median 27.9%,
p90 53.1%, maximum 104.0%. That gives 372 calm hours and 75 stress hours.

| clean set, median (vol pts) | calm | stress |
|---|---|---|
| \|linear - inverse\| | 0.130 | 0.125 |
| signed, all | -0.009 | -0.012 |
| signed, OTM puts | -0.018 | -0.035 |
| signed, OTM calls | -0.002 | -0.004 |
| linear vol-space width | 0.78 | 0.77 |
| inverse vol-space width | 1.06 | 1.08 |
| \|diff\|, all two-sided | 0.143 | 0.146 |

Per quote, |diff| at the 50th / 90th / 99th percentile is 0.13 / 0.42 / 0.96
in calm hours and 0.13 / 0.43 / 0.97 in stress. The whole distribution is
the same, not just its middle.

| | prediction | result | verdict |
|---|---|---|---|
| 1 | calm \|diff\| <= 0.25, signed within ±0.10 | 0.130 and -0.009 | **pass**; replicates the one-snapshot 0.124 over 372 hours |
| 2 | stress >= 1.5x calm, rv slope t > 2 | 0.96x; slope -0.0000, t -0.05 | **fail**: no widening at all |
| 3 | adding linear width halves the rv slope | nothing to halve; width itself t +3.63 | **moot**; the hour-to-hour variation that exists is quote width |
| 4 | puts get richer on the inverse book in stress, t < -2; calls \|t\| < 2 | puts t -0.84, calls t +1.28 | **fail** on puts, pass on calls |
| 5 | crossings < 1%, at least 2x as frequent in stress | 0.335%; stress 0.28% vs calm 0.35% | first half **pass**, second **fail** |
| 6 | forward gap median within ±5 bps, max < 25 | median +0.00, max 7.55 | **pass** |

The direction prediction 4 expected does show up in the stress medians.
Inverse puts are 0.035 points richer in stress against 0.018 in calm, and in
the -0.3 < k < -0.1 bucket the figures are 0.046 against 0.029. But that is
about two hundredths of a vol point, far inside the books' own one-point
quote widths, and the regression does not distinguish it from zero. If the
margin effect exists, it is too small to measure here and too small to
trade.

### Checks run after the predictions (not pre-registered)

- **Crossings are not arbitrage.** There are 562 crossed quotes in total,
  with a median depth of $3.32 and a p90 of $10.64. After paying taker fees
  on both legs (0.03% of the index each, capped at 12.5% of premium), *none
  of them* survives. 46% sit in the far wings (|k| > 0.3), where both books
  quote cheap options a tick or two wide.
- **The wings behave like the middle.** Median |diff| by moneyness bucket,
  calm then stress: k < -0.3: 0.18 / 0.18; -0.3 to -0.1: 0.12 / 0.12; ±0.1:
  0.14 / 0.14; 0.1 to 0.3: 0.11 / 0.10; k > 0.3: 0.18 / 0.13.
- **The worst hours look like any others.** The eight highest-rv hours, from
  the 5-6 Nov drop and the 15 Nov low near 64.7k, have |diff| between 0.11
  and 0.16.

### What this says

1. **One price, two collateral systems.** On Deribit's BTC options, the
   currency of margin and settlement is not priced, even in a 28% drawdown
   with realised vol above 100%. Prediction 4's mechanism, doubled exposure
   for inverse put sellers, is real in principle. But nothing in these
   quotes shows anyone charging for it. The likeliest reason: the same
   market makers quote both books off one surface and hedge across them,
   and the forwards agree to 0.00 bps at the median.
2. **The "thin" book is not the noisy one.** The linear book turns over
   about 1/200th of the inverse volume, yet its quotes are *tighter* in vol
   terms (0.78 against 1.06 points). Daily volume measures who trades, not
   who quotes. HANDOFF §8 named the thin book's microstructure as the main
   risk to this study, and it did not materialise.
3. **For this project, this is validation, not a trade.** The 0.124 cross
   check in HANDOFF finding 8 now holds for 939 hours and through a
   stress episode. The pipeline's conversion arithmetic is right on both
   contract families, under all conditions seen so far.

**Caveats.** The window has two or three stress episodes, not 75
independent hours. Per-hour medians hide individual quotes, though the per-quote
99th percentile does not move either. Mid against mid says nothing about
size: the linear book may show the same price for far less depth. And
the stress measure uses the index price, not the inverse front forward as the method
section said. It switched because the front forward jumps at each expiry
roll. The two differ by basis points, so the regimes are the same.
