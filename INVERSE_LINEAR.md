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

## Results

*(to be appended after running `scripts/inverse_linear.py`)*
