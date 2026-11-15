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

*(appended after running `scripts/live_studies.py`)*
