# How does the surface move? PCA on the live surfaces, written before testing

Track A now holds two months of hourly chains (2026-09-07 to 11-04). This
asks what the fitted surface's day-to-day changes are made of: how many
independent ways it moves, what they look like, and whether they are stable
enough to build on. HANDOFF.md calls this the precursor to anything
model-shaped. Expectations are written here **before the code runs**; the
results are appended below them.

Code: `scripts/surface_pca.py`.

## Setup, fixed in advance

- **Snapshots.** Two a day per asset, the ones nearest 08:00 and 20:00 UTC
  (a fit takes ~10 s, so all ~1,550 would take hours). Skip a slot with no
  snapshot within 2 hours of it.
- **Grid.** Tenors 7, 14, 30, 60, 90 days; standardised moneyness
  z = k / (ATM vol × sqrt(T)) at -1.5, -1, -0.5, 0, +0.5, +1, +1.5, where ATM
  vol is the surface's own at that tenor. 35 cells. Implied vol in vol
  points, read off the `VolSurface` the strategy uses.
- **Extrapolation.** A cell is dropped from the PCA if it falls outside the
  supported strike range on more than 10% of snapshots.
- **Changes.** PCA on the covariance of changes between consecutive slots
  (12h apart; a change spanning a missing slot is dropped). Main results on
  the 12h changes; daily (08:00 to 08:00) changes as a robustness check.
- **Stability.** Loadings estimated separately on the first and second half
  of the sample, compared by absolute cosine similarity.

## Expectations

These come from what equity-index and FX surfaces are known to do, plus
what this project has already seen of BTC.

**E1. One factor dominates.** PC1 explains more than 70% of the variance of
changes. It is a level shift: every loading has the same sign.

**E2. PC1 is largest at the short end.** Short-dated vol moves more than
long-dated vol, so PC1's loadings fall with tenor; the 7-day ATM loading is
at least 1.5x the 90-day ATM loading.

**E3. PC2 is the term-structure slope**: short and long tenors load with
opposite signs. **PC3 is skew**: the loadings at z < 0 and z > 0 have
opposite signs within a tenor.

**E4. Three factors suffice.** PC1-3 together explain more than 90%.

**E5. The factors are stable.** First-half and second-half loadings have
|cosine| above 0.9 for PC1 and above 0.7 for PC2.

**E6. Spot and vol move against each other.** PC1 scores correlate
negatively with the spot log return over the same 12h. (The archive shows
a put-over-call skew on average, +4.2 vol points at z = ±1, which is the
equity-like pattern.)

**E7. BTC and ETH share the structure.** Same shape on both: PC1 a level
shift, PC1 cosine between the two assets above 0.9.

## Known issues, stated up front

- About 115 changes per asset on 35 correlated cells. The top one or two
  components are well determined at that size; past PC3 the loadings are
  mostly noise, and they are not interpreted.
- The surface is a fit, so some movement is fitting noise rather than
  market. That shows up as a long tail of small components, and it could
  inflate the short end, where few quotes survive cleaning (README, *What
  broke* #2).
- Two months is one regime. Stability within it says nothing about the next.
