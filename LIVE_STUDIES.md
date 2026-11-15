# Three studies on the live surfaces, written before testing

Track A: hourly Deribit chains from 2026-09-07 to 2026-11-15, for BTC and
ETH. The surface is fitted twice a day (08:00 and 20:00 UTC) and read onto
the grid that `SURFACE_PCA.md` uses: tenors 7, 14, 30, 60 and 90 days by
standardised moneyness z = k / (ATM vol × sqrt(T)). That gives about 99
slots and about 65 12h changes per asset, including the November sell-off.
The hourly index price comes from `data/history.hourly_funding`, refreshed
on 2026-11-15.

**Written 2026-11-15, before any of the three was run.** The one thing I
looked at first was a predictor, not an outcome: how much funding varies in
the window (study B).

Code: `scripts/live_studies.py`. Inference is Newey-West. 12h slots overlap
in nothing, but their changes are autocorrelated, so 4 lags; overlapping
7-day windows get 14.

---

## A. Do vols stick to strikes or to moneyness?

Both factor write-ups end on the same caveat. The grid is in z, so after
spot moves each cell is a different strike. On a skewed smile, part of any
"vol moves with spot" result is therefore re-indexing along the smile rather
than the market repricing. This study separates the two.

**Method.** For each 12h pair (t, t+1) and tenor τ in {7, 14, 30} days,
take the strike K at z = 0 at time t (the ATM strike). Two changes are
measured:

- fixed-moneyness: Δσ_z = σ_{t+1}(z = 0) - σ_t(z = 0)
- fixed-strike: Δσ_K = σ_{t+1}(z') - σ_t(z = 0), where z' is where K sits at
  t+1. It is read off the t+1 grid by linear interpolation in z, with the
  forward approximated by spot at both ends.

Regress each on the 12h spot log return r. Under **sticky strike** β_K = 0.
Under **sticky moneyness** β_z = 0. The **moneyness share** is
m = β_K / (β_K - β_z): 0 is pure sticky strike, 1 is pure sticky moneyness.
The denominator is the smile's slide term, which is positive on a put-skewed
smile.

**A1. Neither extreme.** At 30d ATM, 0.25 ≤ m ≤ 0.75 on both assets.

**A2. Part of the PCA's spot correlation is re-indexing.** At 30d ATM, for
ETH, |corr(Δσ_K, r)| < |corr(Δσ_z, r)| over the full sample. The ETH
fixed-z level/spot correlation was -0.30, and some of that should be the
grid sliding along the smile.

**A3. In the sell-off, fixed-strike vol rose as spot fell.** On changes
after 2026-11-05 20:00, β_K < 0 for BTC 30d ATM: the market repriced, it did
not just slide. (n ≈ 13, so this is weak whichever way it goes.)

## B. Does funding still bid the call wing? (H7b, out of sample)

H7b (`HYPOTHESES.md`): on 74 monthly archive days, 2021-26, the 30d skew
(σ at z = -1 minus σ at z = +1) fell 0.29 vol points per point of trailing
30-day funding (t -3.9), or 0.34 (t -5.2) with ATM iv controlled. Track A
is data that test never saw.

**Power problem, found before testing.** Over Track A, 30-day trailing
funding has an sd of 0.52 points for BTC and 0.96 for ETH, against 11-14
over 2021-26. At the original slope that is a skew effect of 0.15-0.3 vol
points, which cannot be detected on 99 slots. **The 30-day replication is
reported but not judged.** The tested version uses 7-day trailing funding
(sd about 2 points in the window). Funding is taken up to the *previous*
UTC day's end, which is stricter than the original's same-day alignment.

**B1. The sign replicates on BTC.** Regressing the 08:00 and 20:00 30d skew
on 7-day funding gives a slope < 0 with NW t < -2.

**B2. The size is in range.** The BTC 7-day slope is between -0.15 and
-0.60 vol points per funding point, within a factor of two of the original.

**B3. ETH, never tested before, has the same sign** (slope < 0; any t).

## C. Does skew predict anything?

Skew moving with spot (factor 2, +0.43 BTC and +0.52 ETH over the full
sample) is the most stable spot-vol link found. But that is
contemporaneous. Here the question is whether skew *leads*. Skew is the 30d
skew as in B, at each slot.

**C1. Skew level does not predict the next 12h return.** Regressing r_{t+1}
on skew_t gives |NW t| < 2 on both assets.

**C2. Skew level does predict the downside share of the next 7 days'
realised variance.** The share is down semivariance over total, from hourly
index returns. On BTC the slope is > 0 with t > 2: a richer put wing comes
before more downside realised variance. This is the one prediction here
that says the surface carries forward information. Given H7a (funding
predicts downside realised), it is plausible, not assured.

**C3. Skew changes do not lead returns.** Regressing r_{t+1} on Δskew_t
gives |t| < 2 on both assets.

## Known weaknesses

- About 65 12h changes per asset from one two-month regime, one sell-off
  included. A significant result here is a lead, not a finding.
- Approximating the forward by spot shifts every k by the basis, about
  0.5% at 30 days. The shift is nearly the same at t and t+1, so it largely
  cancels in a change, but not exactly.
- Grid cells outside the quoted strikes on more than 10% of slots are
  dropped, as in `SURFACE_PCA.md`. The z = ±1 wings at 30d survive on BTC
  and ETH.
- Each study has three predictions, which makes nine tests. One spurious
  pass is likely.

---

# Results

Run: `python scripts/live_studies.py` (2026-11-15). There are 65 / 64 12h
changes (BTC / ETH) and 99 / 98 slots. β is in vol points per 1% spot move.

## A. Sticky strike vs sticky moneyness

| ATM | β_z (t) | β_K (t) | m | corr z / K |
|---|---|---|---|---|
| BTC 7d | -0.02 (-0.07) | +0.40 (+1.59) | 0.95 | -0.01 / +0.25 |
| BTC 14d | +0.10 (+0.48) | +0.37 (+2.20) | 1.39 | +0.10 / +0.36 |
| BTC 30d | +0.11 (+0.85) | **+0.29 (+2.85)** | **1.58** | +0.17 / +0.46 |
| ETH 7d | -0.35 (-1.03) | -0.10 (-0.41) | -0.38 | -0.24 / -0.08 |
| ETH 14d | -0.22 (-1.16) | -0.06 (-0.43) | -0.38 | -0.24 / -0.07 |
| ETH 30d | -0.13 (-1.26) | -0.04 (-0.50) | **-0.48** | -0.24 / -0.08 |

| | prediction | result | verdict |
|---|---|---|---|
| A1 | 0.25 ≤ m ≤ 0.75 at 30d | BTC 1.58, ETH -0.48 | **fails on both, in opposite directions** |
| A2 | ETH \|corr K\| < \|corr z\| | 0.08 against 0.24 | **holds** |
| A3 | BTC sell-off β_K < 0 | -0.10, t -0.37 (n = 13) | right sign, nothing more |

**What it says.** The two assets sit on opposite sides of both textbook
rules.

- **BTC: fixed-strike vol rose *with* spot** (+0.29 vol points per 1%, t
  2.85). That goes beyond sticky moneyness. When BTC rallied, the strike
  that had been ATM got *more* expensive. That is the "spot up, vol up"
  pattern of a market whose buyers chase upside, the same leverage-on-calls
  story as H7b. In the sell-off the sign flipped (-0.10), but on 13 changes.
- **ETH: almost sticky strike**, slightly past it. Fixed-strike vol barely
  moves (t -0.5). Almost all of ETH's fixed-moneyness level/spot correlation
  (-0.24 here, -0.30 in `SURFACE_PCA.md`) is the grid sliding along a
  put-skewed smile, not the market repricing. That is exactly the caveat
  the factor write-up raised, and it is now measured: **for ETH, the
  caveat is most of the effect.**

## B. Funding and the call wing, out of sample

| 30d skew on | BTC slope (t) | with ATM | ETH slope (t) | with ATM |
|---|---|---|---|---|
| 7d funding (tested) | +0.18 (+0.60) | +0.20 (+0.81) | **-0.81 (-4.06)** | -0.78 (-4.76) |
| 30d funding (not judged) | +2.45 (+3.75) | +1.33 (+1.28) | -2.75 (-9.86) | -2.57 (-8.31) |

| | prediction | result | verdict |
|---|---|---|---|
| B1 | BTC 7d slope < 0, t < -2 | +0.18, t +0.60 | **fails** |
| B2 | BTC slope in -0.15..-0.60 | +0.18 | **fails** |
| B3 | ETH same sign | -0.81, t -4.06 | **holds**, and strongly |

**H7b did not replicate on BTC in these two months; it did on ETH.** On
ETH, a 1-point rise in 7-day funding came with 0.8 vol points less put
skew, more than twice the archive's BTC slope. The 30-day rows were
declared untestable in advance and stay that way. A predictor with an sd of
0.5-1.0 points over two months is mostly a trend, and their t-stats (+3.75,
-9.86) show how easily a trending regressor manufactures one.

## C. Does skew lead?

| | prediction | BTC | ETH | verdict |
|---|---|---|---|---|
| C1 | skew level → next 12h return, \|t\| < 2 | -0.14, t -1.83 | -0.15, t -1.98 | **holds, barely** on both |
| C2 | skew → next-7d downside share, BTC t > 2 | +0.026, t +2.45 | +0.024, t +3.29 | **holds** |
| C3 | Δskew → next 12h return, \|t\| < 2 | **-0.45, t -4.11** | -0.12, t -0.34 | **fails on BTC** |

### Checks run after seeing the results (not pre-registered)

| | BTC | ETH |
|---|---|---|
| C2, windows ending before the sell-off | +0.017, t 1.38 (n 75) | +0.019, **t 2.44** (n 74) |
| C2, ATM iv controlled | t 2.45 | **t 4.11** |
| C3, last 12h return controlled | **-0.43, t -3.96** | t -0.43 |
| C3, before the sell-off only | **-0.51, t -2.06** (n 39) | +0.34, t 0.85 |
| plain momentum, r_next on r_now | +0.22, t 1.25 | t 0.15 |

- **Skew carries information about the downside to come.** A put wing 1
  vol point richer is followed by a 2.4-2.6 point higher downside share of
  the next week's realised variance. Skew's sd is 2.6-3.1 points, so one
  sd moves the share by about 7 points around a mean of 53-57%. On ETH it
  survives dropping the sell-off and controlling for ATM vol. On BTC it
  leans on the sell-off (t 1.38 without it). The options market's skew is
  partly a forecast, not only a premium.
- **On BTC, a 12h rise in put skew is followed by a lower 12h return**:
  -0.43% per vol point. It is not momentum in disguise (controlling for the
  last return leaves t -3.96, and momentum alone is t 1.25), and it is there
  before the sell-off (t -2.06). It was not predicted, it is absent on ETH,
  it rests on 48 changes, and it is one of nine tests here plus five
  post-hoc checks. **Treat it as a lead to re-test as Track A grows, not a
  finding.** If it holds, it means skew moves before spot on BTC.
- **The skew level does not forecast returns** (C1). Both t's sit just
  inside 2 with a negative sign, so a richer put wing comes before slightly
  lower returns. That sign is consistent with C3, and worth watching.

## What it adds up to

1. **"Vol moves with spot" means different things on the two coins.** BTC
   reprices with spot. ETH's apparent level/spot link is mostly
   re-indexing. Any hedging model that assumes one rule for both is wrong
   on one of them.
2. **The funding → call-wing link (H7b) is real on ETH and not on BTC in
   the live window.** That is the reverse of where it was found. It is not
   a stable BTC property.
3. **Skew forecasts the shape of realised moves (C2), and on BTC its
   changes may lead price (C3).** These are the first results in the
   project where the surface tells you something about the future beyond
   its own level.

Nine pre-registered tests: four hold (A2, B3, C1, C2), four fail (A1, B1,
B2, C3), and one is only the right sign (A3). Expect about one spurious
pass at this count.
