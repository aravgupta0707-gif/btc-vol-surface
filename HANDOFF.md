# Handoff

Everything needed to pick this project up cold. Written 2026-09-09, updated
2026-09-23 after the archive completed, the six-year studies ran (§10), the
hedging-rule work landed (§12, §12.1, §12.2), and putting that rule on the
live book turned up a marking bug that had been freezing its hedge (§10.5).
Updated 2026-10-25 for the study of why the premium shrank (§10.6,
`HYPOTHESES.md`), and 2026-11-14 for the two-month live-book rerun (§10.5)
and the surface PCA (`SURFACE_PCA.md`).

**The repository has a remote now**: `origin` is
`github.com/aravgupta0707-gif/btc-vol-surface`, private, branch `master`. A
clone gets the code and nothing else — every data directory is gitignored, so
see §2 for what has to be moved separately.

`BUILD_GUIDE.md` is the original brief. `PLAN.md` is the working plan with
per-phase checkpoints. `VIOLATIONS.md` is the arbitrage log. `HYPOTHESES.md` is the pre-registered
study of why the premium shrank, with its figures; `SURFACE_PCA.md` the same
for how the live surface moves. This file is the
state of play, what was learned, and what to do next.

---

## 1. State

Every build-guide stage but the dashboard is complete. **430 tests, 51
commits, 12,554 lines of Python, zero remaining stubs.**

| Stage | State | Checkpoint |
|---|---|---|
| Data pipeline | done | `verify_data.py` — 34/34 |
| Black-76 inversion | done | `verify_pricing.py` — 8/8, matches Deribit `mark_iv` to 0.048 vol pts |
| SVI calibration | done | `fit_surface.py` — 10/10 slices, 0.165 vol pts median |
| Arbitrage validation | done | `check_arbitrage.py` — no violations among quoted strikes |
| Surface object | done | `verify_surface.py` — 10/10 |
| Greeks + validation | done | delta matches Deribit to 0.0007 median |
| Strategy + backtest | **done** | 79-day study, 314 positions — §10 |
| Costs + hedging rule | **done** | 128 cells, clocks vs bands, walked forward — §12 |
| Write-up | **done** | `README.md`; detail in this file, §10 and §12 |
| Why the premium shrank | **done** | `HYPOTHESES.md` — pre-registered, H0-H6 |
| Surface factors | **done** | `SURFACE_PCA.md` — 3 factors, 96-97% of 12h changes |
| Dashboard | **not started** | — |

The statistical claim that §5 used to call open is now answered, and the
answer is mostly negative: see §10. The remaining build-guide gap is the
dashboard.

### Running things

```bash
# checkpoints -- PASS/FAIL per check, non-zero exit on failure
python scripts/verify_data.py        # Phase 1, 34 checks
python scripts/verify_pricing.py     # Phase 2
python scripts/fit_surface.py        # Phase 3: fit SVI, report smoothness
python scripts/check_arbitrage.py    # Phase 4 (--log appends to VIOLATIONS.md)
python scripts/verify_surface.py     # Phase 5

# studies (--report re-prints saved results without recomputing)
python scripts/run_backtest.py       # one position on Track A
python scripts/archive_study.py      # 79 one-day positions, six years
python scripts/vrp_surface.py        # premium by strike and tenor  [8m02s]
python scripts/cross_asset.py        # BTC vs ETH (needs vrp_surface first)
python scripts/carry_study.py        # forward curve vs funding
python scripts/hold_to_expiry.py     # Track A, settled at expiry  [~4 min]
python scripts/hold_to_expiry.py --clock       # the same, on the old 4h clock
python scripts/hedge_frequency.py run          # rebalancing rule vs cost [~5 min]
python scripts/hedge_frequency.py walkforward  # the same, chosen blind (§12.2)
python scripts/vrp_study.py          # DVOL-based VRP, 2021-2026
python scripts/vrp_hypotheses.py     # why the premium shrank (§10.6)  [~25 s]
python scripts/make_figures.py premium  # its four figures, into figures/
python scripts/surface_pca.py        # surface factors on Track A  [~5 min first run]
python scripts/rnd_monitor.py        # risk-neutral density (--history)

# operations
python scripts/collector.py status   # hourly snapshots
python scripts/fetch_archive.py status
python scripts/health.py             # watchdog: everything, now
python -m pytest tests/ -q
```

Each `verify_*` script prints PASS/FAIL per check and exits non-zero on
failure, so they work as a regression suite over live data as well as
documentation of what "correct" means.

---

## 2. Data

Three independent sources. They are easy to confuse — see §4 finding 8.

**Track A — the live collector.** Hourly Windows scheduled task
(`BTC-Vol-Surface-Collector`), managed entirely through
`scripts/collector.py` (`status`, `watch`, `logs`, `stop`, `start`,
`uninstall`, `install --visible`). Collects three currencies each hour: BTC
and ETH (inverse books) and USDC (linear book, six underlyings in one call —
BTC, ETH, SOL, XRP, AVAX, HYPE, TRX). Writes
`data/raw/<currency>_chain_<stamp>.parquet`, ~5 MB/day for all three.

At 2026-11-14: **BTC 943 snapshots, ETH and USDC 919 each**, from
2026-09-07, 159 MB — ~68 days at ~14 snapshots a day rather than 24. Every
long gap is the laptop sleeping, and each ends in a
catch-up run at an odd minute — `StartWhenAvailable` firing on wake.
`-WakeToRun` would close them at the cost of waking the machine hourly;
deliberately not enabled. So Track A is hourly but **not contiguous**, which
is what limits anything needing an unbroken path.

**Track B — the Tardis.dev free archive. 81 days, BTC and ETH** (the 79
of 2020-03 to 2026-09 that every study here uses, plus 2026-10-01 and
11-01 fetched since by the archive task). First day of each month, `options_chain` from 2020-03-01. No
API key, no payment, but the endpoint needs *some* `Authorization` header
present (a 403 without one) and rejects Range requests. `data/tardis.py`
streams and downsamples on the fly — one day is ~516 MB gzipped and ~10.8M
rows, half of it ETH, so both currencies come out of one transfer. Archive
rows also carry `bid_iv`, `ask_iv` and Deribit's own greeks, which the live
book-summary endpoint does not serve.

A new free day appears on the 1st of each month and the hourly task picks it
up on its own (2026-10-01 around 2 October). **Contiguous** history — the
thing needed to hold an option to expiry in a past era — exists only behind
the paid tier.

**Hourly continuous history — `data/history.py`.** Perpetual candles,
funding, and the Deribit **index**, hourly for BTC and ETH from 2020-02-01:
58,203 hours each at 2026-09-22, 2 missing. Cached in `data/research`,
refreshed incrementally — and **only** when called without `offline=True`, so
the cache lags whenever recent work has been offline (§12's sweep reads it
that way); any one online call tops it up.

This is what makes the archive usable: the chains give one
cross-section a month, and this gives the price path between them, so an
option can be hedged and settled. Use the index, not the perp — see §4
finding 13.

**Deribit index history.** `DVOL` (their 30-day vol index, daily since
2021-03-24) and perpetual OHLC. One number per day, not chain data — you
cannot build a surface from it. Used only by `vrp_study.py`.

---

## 3. Architecture

```
data/fetch_deribit.py   one API call per currency for the whole chain
data/clean.py           filters, USD conversion, exact-UTC expiry, OTM select
data/tardis.py          Track B streaming loader (Arrow), multi-currency
data/history.py         hourly index, funding and perp candles since 2020
pricing/black76.py      price, vega, greeks, implied vol inversion
surface/svi.py          SVI calibration (quasi-explicit reduction)
surface/arbitrage.py    Durrleman g(k), calendar check, trust radius
surface/build.py        snapshot -> fitted slices (shared pipeline)
surface/surface.py      VolSurface: iv / price / greeks for any (K, T)
surface/density.py      risk-neutral density, quantiles, tail probabilities
strategy/straddle.py    delta-hedged short straddle, position + costs
strategy/hedging.py     when to rebalance: clocks, fixed bands, Whalley-Wilmott
strategy/backtest.py    walk-forward loop, no-lookahead guards, metrics
scripts/health.py       watchdog over everything that runs unattended
slurm/                  array jobs for the §12 sweep, unused by default
```

Dependency direction is strictly downward — `data` knows nothing about
`pricing`, `pricing` nothing about `surface`. `surface/build.py` is the only
module that spans layers, and exists so the Phase 3/4/5 scripts share one
pipeline rather than three copies.

---

## 4. What was learned

The most valuable output of the project. Every item below came from
measuring something that could have been assumed — and in most cases the
assumption was wrong.

### Five places the build guide is wrong or silent

1. **Premiums are quoted in BTC, not USD** — the guide never mentions it.
   The correct multiplier is the per-expiry forward, not the spot index:
   forward reproduces Deribit's `mark_iv` to a median of 0.015 vol points,
   index is ~30x worse and leaves 13% of quotes uninvertible. Fails inside
   the root finder, looking exactly like a solver bug.

2. **The forward is published, not derived.** §1.3 says reconstruct it from
   index plus funding curve. Deribit sends `underlying_price` per expiry on
   every row. Extra work, extra error source, no benefit.

3. **The 15% spread filter measures premium size, not liquidity.** ITM
   quotes pass 96% of the time, OTM only 68% — because ITM options are
   expensive, not liquid. The bias runs opposite to `select_otm`, and
   together they cut a 964-row chain to 278. Fix: pass if inside 15% of mid
   **or** within two exchange ticks (0.0001 BTC below a 0.005 premium,
   0.0005 above). You cannot ask a market to be tighter than its tick.

4. **The greeks convention is the reverse of what §6 implies.** The guide
   says delta will disagree with Deribit's until inverse settlement is
   handled. Measured: the plain USD Black-76 delta matches to **0.0007**
   median; the coin-adjusted one is 26x further off. Deribit publishes the
   ordinary dimensionless delta. Vega differs only by their per-1%
   convention (measured ratio 99.96); gamma agrees to 1%. `delta_coin` is
   still computed — for hedging a coin-margined book, not for comparison.

5. **"Flat-extrapolate the longest slice" is ambiguous and the obvious
   reading is wrong.** Holding total variance constant is arbitrage-free but
   sends `IV = sqrt(w/T) -> infinity` as `T -> 0`, quoting a one-hour option
   at thousands of percent. Hold the variance *rate* instead.

Also: the guide's `snapshot_chain` sketch loops `public/ticker` per
instrument — ~1000 round trips into a rate limit, where
`get_book_summary_by_currency` returns the whole chain in one call. And §8
assumes a snapshot history exists; the public API serves current state only,
which is the entire reason for the two-track data strategy.

### The bug class that matters

Six separate times something failed by **reporting success**. The first
three are pipeline bugs, the next three operational — the class does not
care which:

- `implied_vol` returned **0.001** for an option genuinely priced at 90% vol.
  A 110k-strike one-day call is worth ~1e-9 dollars, the objective is flat
  across the whole bracket, and Brent returned its left endpoint. Wrong in a
  way that looks like data, and it would have flowed into an SVI fit unseen.
- Task Scheduler reported `Last Result: 0` and a fresh `Last Run Time` while
  collecting **nothing** — `schtasks` defaults `DisallowStartIfOnBatteries`
  to true, parking the task in Queued on battery.
- SVI returned `b = 267,905, rho = -0.99999997` — a V-shaped kink chasing one
  corrupted quote, which scored *better* in least squares than the truth.
- The archive fetch printed `--- done ---` having **skipped 53 of 79 days**.
  One 429 was logged as a per-day skip, and since the next request is
  refused too, a single rate limit burned the whole queue in seconds and
  left it looking attempted. The end line now always carries what is still
  missing, and "done" was renamed "end".
- A one-time scheduled task sat in `Ready` for **three days** after a
  dropped connection ended one run. The task reported nothing wrong; there
  was simply nothing left to run it. Now hourly, with `IgnoreNew`.
- An SVI slice ran for **half an hour** instead of being refused (§4
  finding 14), stalling a whole study behind it with no output at all.

The unifying idea, and the single most useful lesson: **a result sitting on
a boundary is an artefact of where the search stopped, not an answer.** That
one observation fixed the Brent root, the SVI degeneracy, and the surface
clamping. `status` now warns explicitly on a Queued task.

The operational half has its own lesson: **an unattended job needs a
watchdog that checks outcomes, not exit codes.** `scripts/health.py` checks
what exists — file freshness, row counts, whether runs make progress — and
notifies on change. Building it also showed the cost of getting alarms
wrong: in its first four days it sent eight notifications and **every one
was a false alarm** (a scheduler refusing an overlapping trigger, a query
that timed out on wake, a long-but-alive run, a single failed catch-up).
Each is now tested against the real log line that caused it. An alarm that
cries wolf is worse than no alarm.

A related distinction worth keeping: **"the solver failed" and "the data
does not determine the answer" need different responses.** For a deep-OTM
option worth 1e-19 dollars, no algorithm recovers the vol; the only correct
output is a refusal.

### Genuine market findings

6. **The variance risk premium is real and decaying fast.** Over 1,963 days
   since 2021-03, implied averages 60.4% against 51.3% realized — a mean
   premium of **+9.2 vol points at a 73% hit rate**, and +10.8 at 76% on
   non-overlapping windows. But annually: **+18.6 (2021), +12.7, +6.8, +8.8,
   +4.4, +3.4 (2026)**, hit rate 86% -> 62%. A backtest on 2021-22 data
   describes a strategy that no longer exists. At 3-4 points, transaction
   costs are the whole question, not a refinement.

7. **The arbitrage was never in the data.** Zero calendar crossings where two
   slices both have quotes — worst is exactly `+0.000e+00`. Every violation
   came from *extrapolating* a narrow slice past where it was quoted until it
   overtook a longer one. `18Sep26`, quoted over `k in [-0.15, +0.14]`,
   reached `w = 0.0404` at `k = -0.5` against `25Sep26`'s `0.0308`, while
   being correctly ordered at the money.

8. **Deribit runs a linear USDC book beside the inverse one**, and it is a
   free independent validation: 130 strikes quoted in both, agreeing on
   implied vol to **0.124 vol points** through entirely different conversion
   arithmetic, with forwards matching to $2.36.

9. **Front-week crypto expiries are structurally unfittable** — 1 to 13
   usable quotes against five SVI parameters. A market fact, not a pipeline
   defect. `MIN_POINTS = 8`; thinner slices are refused, not forced. This
   also costs the 7-day studies about half their inceptions: with no fitted
   slice inside a week, a 7-day query is extrapolation and is excluded.

10. **The premium did not shrink evenly — it moved to the call wing.**
    Finding 6 measured one number per day. Decomposed across the smile
    (§10), the BTC 30-day downside-put premium is **gone** after 2022
    (z-1.0: +72 bps, t 3.4 in 2020-22; +4 bps, t 0.3 in 2023-26) while the
    ATM and the **upside calls** still pay (+83 bps t 2.4; +53 t 3.2; +46
    t 3.0; +28 t 2.8). That is the reverse of the equity-index pattern,
    where the put wing carries the premium. At 7 days nothing survives for
    either asset, and for ETH almost nothing survives anywhere.

11. **ETH volatility is systematically cheap against BTC.** Over 77 paired
    months the implied ETH/BTC ATM ratio averages **1.244 against a
    realised 1.327**. Long ETH vol / short BTC vol at equal vega earns
    **+3.65 vol points a month (t 2.75)**, rising to +6.1 when the implied
    spread is narrow (t 4.6), and it holds separately in both eras and in
    6 of 7 years. The spread itself mean-reverts with a **1.6-month
    half-life** around +12.9 points. Mid prices, before costs.

12. **The options forward curve forecasts funding better than recent
    funding does** — BTC at 7, 30 and 90 days, ETH at 30 and 90 — and sits
    about a point above realised funding on average, which is the
    borrow/collateral wedge rather than a mispricing.

13. **Deribit's hourly candle `close` belongs to the next hour mark.** Read
    at its own timestamp it sits 245-315 bps from the index at the 99th
    percentile; aligned to the hour it actually belongs to, the perp-index
    basis is ~2 bps median and 16-19 bps at the 99th. Using `close` at its
    stamp is a one-hour lookahead — silent, and it would have flattered
    every path-dependent result. Use the index from `history.py`.

14. **A one-sided smile cannot be fitted, and says so slowly.** The 11Dec20
    slice of 2020-12-01 — calls only, vol rising 78% -> 118%, with a
    10-point jump between the last put and the first call — sent 84% of the
    fit's evaluations into the constrained fallback at 83 ms each: bounded,
    but about half an hour. Normal slices need at most 167 such solves
    (median 2, measured over 208). Past 500 it is refused, in 15 seconds,
    with a reason.

### Where I was wrong, caught by measurement

Worth keeping because it is the honest half of the story.

- Assumed inverse settlement would shift Deribit's delta. It does not.
- A wing-slope guard threw away a **good** slice (`27Nov26`, a textbook
  smile fitting to 0.169 vol points) for being "11x too steep". Large `b`
  does not distinguish a kink from a wide smile; `sigma` does — kink has
  sigma pinned at the 0.01 floor, wide smile sits at 0.60.
- Claimed a near-the-money filter cut the backtest "several-fold". Measured
  2.45s against 2.32s. No saving; the far wings are not the slow ones.
- Estimated the 79-day archive at half an hour. It is 3-4.
- Converted RMSE to vol points by dividing by `sqrt(a)` — but `a` is the
  vertical shift of total variance, not the ATM variance, and is routinely
  negative for wide smiles. Produced RMSEs of 22,000 vol points.
- Assumed a flat 0.0005 tick and reported "109 one-tick quotes". The tick is
  0.0001 below a 0.005 premium; the real figure was 12.
- Reported the 2026-07-01 archive file as **12.8 GB** from the server's own
  `Content-Length`, and reasoned about whether the last three days were
  obtainable at all. One malformed response. They downloaded normally.
- Diagnosed the collector's failures as a wake-time problem, and they were —
  until one failed with the machine awake and the network up. The wait-for-
  network fix was right but incomplete; the retry that followed is the part
  that mattered. A correct diagnosis of *some* of the evidence.
- Measured 7-day implied carry against the 00:00 index and concluded
  trailing funding forecasts better. Hourly bucketing puts a forward up to
  59 minutes out of step with the index, which at T = 7/365 becomes ~23
  points of annualised noise. Measured forward-to-forward inside one
  snapshot, sd falls 41 -> 10 and the conclusion reverses (finding 12).
- Let a test write a one-row synthetic "snapshot" into the real `data/raw`,
  where every study would have read it as a market observation.
  `tests/conftest.py` now fails the session if any test touches real data.
- Reported "7-day" holds of Monday daily expiries that Deribit had not
  listed 7 days earlier; the position had quietly opened days late while the
  table still said 168 hours.

---

## 5. Known limitations

**The strategy has no demonstrated edge at a one-day horizon, and the rest
is marginal against costs.** 314 independent positions over six years say
the premium was only capturable in 2021-22, and pooled over the whole
period there is none even before costs (t 0.1-0.7). What survives in
2023-26 is BTC 30-day ATM and the call wing at roughly **+83 bps of spot a
month**, against an entry spread of ~18 bps and hedge slippage anywhere
from 10 to 60 bps depending on rebalancing frequency. So the surviving
premium is about the size of realistic costs, and **hedging frequency, not
the premium, decides the outcome.** §12 measured it: the premium clears the
option spread (+73 bps, t 2.4) and dies on the hedge — +33 bps at t 0.9 on the
best clock, +43 at t 1.2 once the rule is a Whalley-Wilmott band (§12.1), and
the band's advantage grows with the cost it economises on.

**Every study price is a mid.** No study charges the bid-ask on the option
except `hold_to_expiry` (entry only), `archive_study` (both sides) and
`hedge_frequency` (entry, measured per strike from the archive's own book). Wing
spreads on Deribit are wide, so the call-wing premium in §10 is the most
cost-exposed result in the file.

**The statistics are thin and multiple.** About 12 inceptions a year; BTC
and ETH on the same day are not independent, nor are adjacent strikes; §10
reports 72 cells, so a couple of |t| > 2 are expected by chance. §12.1 adds
128 more cells and picks its band parameter in hindsight, so its headline
level is an upper bound. §12.2 walked that choice forward: on BTC at 5 bps and
above the band still beats an a-priori clock at every warmup, on ETH it never
does, and 70-90% of the in-sample level is reachable. The two
results that survive being split by era are the BTC call wing and the
ETH/BTC dispersion.

**Track A is not contiguous.** ~68 days of hourly snapshots with sleep
gaps. Long holds and anything needing an unbroken path are limited by it:
every 7- and 30-day live position had an 11-54h hedging gap (§10.5), and the
surface PCA lost about half its possible 12h pairs (`SURFACE_PCA.md`).
Positions left unhedged over a gap are marked in `hold_to_expiry`.

**The theory-gate ratio is approximate.** It uses the textbook
`0.8 * S * sqrt(T)` ATM vega proxy and assumes P&L accrues linearly in time,
which it does not for a short-dated option. A ratio in 0.5-2.0 confirms the
hedging and cost logic; it is not a precision measurement.

**The surface clamps rather than extrapolates.** Beyond a slice's quoted
range it returns the boundary value. Honest, but it means far-wing queries
are flat rather than smooth. A desk would fit a wing model instead.

**The RND monitor's tail probabilities move with its own integration
error.** Across 32 Track A surfaces, P(below 60k) correlates **+0.85** with
the density's total mass, and removing that effect halves its hour-to-hour
spread (sd 1.78 -> 0.93). The largest 4-hour jump, 5.3% -> 10.3%, happened
with the forward moving 0.5% and ATM vol 0.2 points. A level inside the
quoted strikes is still marked "not extrapolated" even though its
probability integrates over the unquoted left wing. Also, the ETH report
lists a "below 0" level and says "BTC" in a heading.

**The cleaner is blind to the short end, and it has bitten twice.** The
quote filters key on *relative* spread, which scales with premium size, so
cheap short-dated options are rejected for being ordinary. It froze the live
hedge (§10.5) and it independently stops the archive seeing its own front
expiries: on 2023-07-01 the chain's 0.33d expiry keeps 0 of its quotes, 1.33d
keeps 1, 2.33d keeps 5, 6.33d keeps 6, and SVI refuses anything under 8
points. The shortest *fitted* slice is 6.33d at the median across the archive
and exceeds 1.0d on 100% of asset-days, so a 1-day tenor comes back
unsupported everywhere — measured, by running `TENOR_DAYS="1 7 30"`, which
returned 90,432 rows and zero usable ones.

`clean_snapshot(drop_wide=False)` largely reverses it where it has been
applied (1.33d goes from 1 point to 10 on that day, 5 to 27 on 2021-01-01),
but 0.33d still reaches only 2-5, so the shortest anchorable tenor is ~1.33d
rather than 1.0d. **Deliberately not fixed in the archive fit.** Doing so
adds points at 7d and 30d too (7.33d: 24 → 33), which moves the fits and so
re-derives the published numbers in §12, §12.1 and §12.2 — a large blast
radius to refine a question this project does not need answered. Recorded as
a limitation, with the live-book fix (§10.5) kept because there it was
breaking a measurement rather than refining one.

**`dashboard/app.py` is still the original stub.**

---

## 6. What runs unattended

Three hourly Windows scheduled tasks, all with battery starts allowed,
`StartWhenAvailable`, and `IgnoreNew` so a long run is never doubled:

| Task | Script | Does |
|---|---|---|
| `BTC-Vol-Surface-Collector` | `collector.py` | one snapshot per currency, :46 |
| `BTC-Vol-Archive-Fetch` | `fetch_archive.py` | any archive day not on disk, :39 |
| `BTC-Vol-Health-Check` | `health.py` | watchdog over all of it, :16 |

The archive task stays registered now that it is complete: it is silent
when there is nothing to fetch, and it collects each new free month on its
own. Each task is staggered so they do not race for the same files.

**The watchdog** checks outcomes rather than exit codes — task settings,
snapshot freshness and row counts, collector failures, archive progress,
file integrity, disk — and raises a Windows notification when a check
changes state, again after 12 hours if a failure persists, and once on
recovery. It tells sleep from failure using its own run history: if it has
itself run hourly for two hours and the collector has written nothing, the
scheduler was demonstrably able to run tasks. Windows power events cannot
be used for this — on this machine tasks ran through some modern-standby
periods and not others. `python scripts/health.py` runs every check now;
`logs` shows one line per hourly verdict.

---

## 7. Next steps

**Settled, do not redo.** The costs question is answered across §12, §12.1 and
§12.2: the premium clears the option spread (+73 bps, t 2.4) and dies on the
hedge; a Whalley-Wilmott no-trade band beats the best clock and beats it by
more as costs rise; on BTC at 5 bps and above that survives being chosen
without hindsight, on ETH it never does. It is still not a trade — the best
honest cell is ~+59 to +75 bps at t 1.4-2.1 over 65-70 overlapping monthly
inceptions.

**Done, 2026-09-23: the band is on the live book.** `strategy.hedging.walk`
could not be swapped in literally — it walks one option at a constant vol
over a spot path, while the live book is snapshot-driven with two legs and
market vols that move. What transfers is the rule, so `ww_band` is now one
shared function called by both the sweep and a `whalley_wilmott` rule in
`DeltaHedgedStraddle`, with tests asserting the two call sites return the
same width. It was not mechanical: it exposed the marking freeze in §10.5,
which had to be fixed before any live cost number meant anything. Read
§10.5 before trusting the daily rows — at one day the band barely trades,
and that is outside where §12.2 validated it.

In priority order from here:

**Recalibrated 2026-09-30 against BUILD_GUIDE §0.** This is a
demonstration piece — "that log is your interview material, it's more valuable
than the final Sharpe number" — and the brief warns in §1.2 against
over-engineering the data layer at the expense of the modelling stages. The
research stages are all complete. **Phase 7, the dashboard and write-up, is
the only unchecked box in both README and PLAN**, so it is step 1, and the
former step 1 is closed as out of scope.

*What was dropped, and why:* walking gamma forward at a 1-day tenor needed the
archive's short end, which needed the cleaner relaxed in the fit, which
re-derives §12's published numbers. That is a large change to three sections
marked "settled, do not redo", in service of a trading question the brief
never asks. The finding is recorded in §5 instead. For the write-up, "the band
is not validated at one day, and here is the measurement showing why" is a
better artifact than a 1-day table would have been.

1. **Phase 7 — the write-up.** BUILD_GUIDE §9 is specific about what the
   README owes: the pipeline diagram, the arbitrage violations found and
   fixed, final metrics with honest caveats about crypto costs, and a couple
   of surface snapshots. **README done 2026-10-05** (pipeline, results,
   snapshots, what broke; premium figures added 2026-10-25). The four interview
   talking points §9 lists should have concrete answers ready; three of them
   are unusually well served by §10 and §12, and the fourth — what would
   break this in a real market — now has an empirical answer rather than a
   hypothetical one, in §10.5. `VIOLATIONS.md`'s "Resolution" field, the
   question §9 says to prepare, was filled 2026-11-14.

2. **Phase 7 — the dashboard.** `dashboard/app.py` is still the 14-line stub
   and is the last unstarted build-guide stage. The `VolSurface` object
   already serves everything it needs, and there is more worth showing than
   when it was scoped: the RND monitor, the premium-across-the-smile surface,
   and the rule comparison.

3. **The dispersion trade properly** — long ETH vol / short BTC vol,
   vega-neutral sizing rather than the approximation in §10, real spreads, and
   an entry rule on the spread's own history. Still the largest effect found
   (-6.14 vol points a month, t 4.6 conditional) and the least examined. Better
   equipped than when it was scoped: §12's measured spread curve gives it real
   entry costs, and `strategy/hedging.py` gives it a hedging rule worth using.
   Note the asymmetry §12.2 found — a rule validated on BTC did not transfer to
   ETH — before assuming anything crosses between the two.

4. **Fix the RND tail** (§5): normalise the density over the quoted range, or
   mark a level as extrapolated when its *integration domain* leaves the quoted
   strikes rather than only when the level itself does. Also fix the ETH
   report's "below 0" level and its "BTC" heading.

5. **Why high funding means a lower premium** (§10.6, H6). Two cheap tests on
   data already on disk, both listed at the end of `HYPOTHESES.md`: split
   realised into up and down semivariance and see whether high-funding
   windows realise a lot of upside; and regress the z=-1 minus z=+1 skew on
   funding over the 64 monthly days to see whether leverage bids the call
   wing. Pre-register them the same way.

6. **Surface PCA done 2026-11-06** (`SURFACE_PCA.md`): three factors explain
   96-97% of 12h changes on both assets — level (5-7x larger at 7d than 90d),
   then skew, then a short-end term factor — identical across BTC and ETH.
   Rerun as Track A grows; half the possible pairs were lost to laptop
   sleep. **Still runnable**: the inverse-vs-linear stress test (spin-off D)
   on the USDC chains Track A has been banking.

**On models.** Not yet, and the constraint is observations rather than
ambition: 44 inceptions in the modern era, one a month, and the free archive
tier adds twelve a year. The one honest ML-adjacent test that has run came back
negative — `cross_asset` found implied carries no information about realised
beyond trailing realised (RMSE 0.208 vs 0.215). Where data genuinely exists is
the hourly path (58k hours), and the question living there is hedging policy,
which §12.1 and §12.2 now answer with a closed form. So a fitted policy has to
beat Whalley-Wilmott chosen out of sample, not a clock — and §12.2 is the
harness to judge it with. The factor structure is now established
(`SURFACE_PCA.md`: level, skew, short-end term, 96-97% of 12h changes), so
the remaining constraint is Track A's length and its gaps, not the absence
of a structure to model.

---

## 8. Scoped spin-off projects

Ranked by value per unit of work given what already exists. All four reuse
the existing pipeline; none needs new infrastructure.

### A. Implied risk-neutral density monitor — **BUILT**

`surface/density.py` + `scripts/rnd_monitor.py`, 22 tests. See §9.

Original scoping follows for reference.

#### (original scope) ~1 day

**Question:** what probability does the market assign to BTC below $50k in
three months, and how does that move over time?

**Already built:** `surface/arbitrage.py::implied_density` computes the
risk-neutral density from a fitted slice — it is how the butterfly check
works, and it integrates to 1.0 to within 5e-3.

**To add:** integrate the density over price ranges to get tail
probabilities; run across the Track A time series; a chart of a few tail
probabilities over time.

**Why it is good:** highest value per unit of work of the four — the hard
part is done. Visually compelling, directly extends Phases 4-5, and states
the surface's content in terms anyone can read. Breeden-Litzenberger is also
a strong thing to be able to explain.

**Risk:** the density is only trustworthy inside each slice's trust radius,
so deep-tail probabilities are extrapolation. Report the radius alongside.

### B. BTC vs ETH implied vol spread — **BUILT**

`scripts/cross_asset.py`, on 77 paired archive months. See §10. The
wall-clock risk below never materialised because the archive carried ETH.
Implied correlation was dropped on purpose: it needs a cross option Deribit
does not list, so any number would be assumed rather than measured.

#### (original scope) ~2 days

**Question:** how do BTC and ETH vol surfaces move relative to each other,
and is the spread mean-reverting?

**Already built:** both books collect hourly; the whole surface pipeline is
asset-agnostic and already runs on ETH (9/11 slices, 0.129 vol pts median).

**To add:** paired ATM term structures, the spread as a time series, implied
correlation from the two.

**Why it is good:** vol dispersion is a real desk trade and almost no
portfolio project does cross-asset vol. Needs no new collection.

**Risk:** wall-clock gated — a spread study needs weeks of paired history,
and ETH collection only started 2026-09-07.

### C. Basis term structure vs perpetual funding — **BUILT**

`scripts/carry_study.py`, on all 79 archive days. See §10 and finding 12.
The risk below was the right one to name — the two do not disagree much —
so the question became "which forecasts funding better", and the forward
curve wins.

#### (original scope) ~1 day

**Question:** do the options market and the perpetual market agree on the
cost of carry, and what does a dislocation mean?

**Already built:** every snapshot stores `underlying_price` for all 12
expiries — that *is* the options-implied forward curve, already collected
and sitting in the parquet files. Deribit's
`public/get_funding_rate_history` gives hourly `interest_1h` with full
history (verified).

**To add:** extract the forward curve per snapshot, convert to an implied
carry rate per expiry, compare against realised funding.

**Why it is good:** genuine relative-value content, zero new collection, and
it uses data already banked. Cheapest real project on the list.

**Risk:** the two need not agree — options carry embeds borrow and
collateral costs that funding does not — so the interesting output is the
*shape* of the disagreement, not a mispricing claim.

### D. Inverse vs linear vol spread — ~1-2 days

**Question:** the two BTC books agree to 0.124 vol points in calm
conditions. Does that hold under stress?

**Already built:** both books collect hourly; the quote-currency-aware
conversion in `clean.py` handles both families correctly.

**Why it is good:** the most novel of the four — almost nobody looks at this
— and it is a direct measure of how well two settlement conventions on the
same underlying stay tied together.

**Risk:** the linear book is thin (BTC_USDC turns over ~$109k/day against
$20.8M inverse), so much of any spread will be microstructure rather than
signal. Needs the vol-space spread filter to separate them.

### Also considered, not recommended yet

**Altcoin surfaces** (SOL, XRP, AVAX, HYPE, TRX) come free with the USDC
call and are already being banked, but daily volumes of $7k-159k make them
too thin to build on today. In six months there may be something there.

---

## 9. The RND monitor (spin-off A, built)

```bash
python scripts/rnd_monitor.py              # latest snapshot
python scripts/rnd_monitor.py --history    # across Track A
python scripts/rnd_monitor.py --days 30
```

`surface/density.py` reads the risk-neutral density out of the surface and
turns it into quantiles and tail probabilities. It shares Durrleman's g(k)
with `arbitrage.py` deliberately: g >= 0 is exactly the condition that the
density is a density, so the butterfly check and the RND are one statement
seen twice.

Derivatives are numerical here rather than closed form, because the surface
value at a point may be an interpolation, a clamped wing, or the running
maximum -- none of which has a clean analytic derivative, and finite
differences capture what the surface actually serves.

Verification, in order of what it would cost to get wrong:

- With flat vol the density must collapse to the lognormal one, k Normal
  (-w/2, w). Tested against `scipy.stats.norm` directly.
- **E[S_T] = F**, the risk-neutrality condition, holds to five decimals.
  This is the invariant that pins the Jacobian and the drift together.
- The density integrates to one across maturities.

Two things found while building it:

- The CDF originally truncated at the nearest grid point below the target
  rather than interpolating to it, losing up to half a cell of mass -- a
  0.0016 error on a number meant to be read as a percentage.
- A test asserting the median sits below the forward was wrong, not the
  code. Risk-neutrality pins the *mean* at F, and a downside skew is a long
  left tail, which with the mean fixed pushes the median the other way.

Measured on 45 hours of Track A at a 90-day horizon: ATM vol 38.6-39.8%,
5% quantile ~53k, 95% ~108k, density mass 0.974-1.019. **P(below 60k) has
standard deviation 1.74 against P(above 100k)'s 0.73** -- the downside tail
is 2.4x noisier, because the left wing is where quotes thin out and where
clamping starts to bite. Worth knowing before quoting a crash probability.

---

## 10. The six-year studies (Phase 7)

Four studies on the completed archive plus the live collection. All are
re-printable from saved results in `data/research/` without recomputing.

### 10.1 One-day positions, 79 days, both assets — `archive_study.py`

Sell the ATM straddle at 00:00, hedge every 4h, buy it back at 23:00; one
independent position per archive day. 79 days x 2 assets x 2 tenors is 316
attempted and **314 measured**: ETH weekly in March and April 2020 could not
be opened at all, because no strike had both a call and a put quoted on an
ETH chain that thin. Every statistic below runs on the 314. The tenor is **fixed** to a weekly
(>=5d) and a monthly (>=21d) band, because Deribit listed no dailies in 2020
and several a week by 2026 — "nearest expiry" would have made an era
comparison a tenor comparison.

Mean gross P&L, bps of spot per day (the premium captured, before costs):

| Year | BTC mo | BTC wk | ETH mo | ETH wk |
|---|---|---|---|---|
| 2020 | -64 | -54 | -38 | -15 |
| **2021** | **+78** | **+60** | **+52** | +27 |
| **2022** | **+31** | **+32** | **+25** | **+39** |
| 2023 | -9 | +2 | -17 | -24 |
| 2024 | -16 | -16 | -19 | -13 |
| 2025 | -7 | -8 | -2 | +2 |
| 2026 | +9 | +21 | -2 | -1 |
| **All** | **+5** | **+6** | **+1** | **+3** |

Only 2021-22 is consistently positive (t 1.8-2.5 in 7 of 8 cells). Pooled
over six years there is no significant edge **before** costs (t 0.1-0.7).
Costs are 34-46 bps a day against that, so net is negative everywhere
(-27 to -46). Across the 314 positions, gross P&L correlates +0.37 to +0.46
with the theoretical variance-gap estimate and shares its sign on 65-70% of
days: the machinery does what theory says, on a sample large enough to tell.

**A daily round trip cannot work** — it pays the spread twice for one day of
decay. That is a statement about the cost structure, not about the premium.

### 10.2 The premium across the smile — `vrp_surface.py`

Price options at fixed standardised moneyness `K = F exp(z sigma_atm sqrt(T))`
off each day's 00:00 surface, sell at mid, delta-hedge every 4h along the
hourly index, settle at intrinsic. Seller's hedged P&L = the premium that
strike carried. Strikes outside their slice's quoted range are excluded
(1,041 rows). Validated on simulated paths first: realised = implied keeps
0.2% of premium, half keeps 49.8%, 1.5x keeps -50.3%, against theory's
0 / +50 / -50.

BTC 30-day, mean seller P&L in bps of spot (t in brackets), 77 inceptions:

| Strike | 2020-22 | 2023-26 |
|---|---|---|
| put z-1.5 | +42 (1.98) | +5 (0.64) |
| put z-1.0 | +72 (3.36) | +4 (0.28) |
| put z-0.5 | +79 (2.60) | +26 (1.43) |
| straddle | +180 (2.66) | **+83 (2.42)** |
| call z+0.5 | +78 (1.47) | **+53 (3.17)** |
| call z+1.0 | +84 (2.36) | **+46 (3.04)** |
| call z+1.5 | +113 (2.72) | **+28 (2.84)** |
| call z+2.0 | +5 (0.34) | **+27 (2.70)** |

The put-wing premium is gone; ATM and the call wing survive. See finding 10.
BTC 7-day: nothing significant after 2022. ETH 30-day: the 2020-22 premium
sat in the *put* wing (z-2.0 +48, t 2.93) and after 2022 only the far call
wing survives (z+2.0 +19, t 2.97). ETH 7-day: nothing.

### 10.3 BTC against ETH — `cross_asset.py`

77 paired months, 30-day ATM, from the same fitted surfaces.

- **Spread** ETH - BTC implied: long-run **+12.9 vol points**, AR(1)
  phi = 0.65 (se 0.09), **half-life 1.6 months**.
- **Forecasting**: the implied ETH/BTC ratio predicts the realised ratio no
  better than the trailing realised ratio does (RMSE 0.208 vs 0.215,
  corr +0.66 vs +0.65). No information advantage.
- **Level**: implied ratio 1.244 against realised 1.327 — ETH vol is
  relatively cheap.
- **Dispersion**: short ETH vol / long BTC vol at equal vega loses **-3.65
  vol points a month (t -2.75)**; i.e. the reverse trade earns it. Split on
  the spread's own expanding median (no lookahead): -6.14 (t -4.56) when the
  spread is narrow, -1.57 (t -0.70) when wide. Holds separately in both eras
  (-8.13 t -3.18; -4.44 t -3.86) and in 6 of 7 years; 64% of months.

### 10.4 Carry against funding — `carry_study.py`

Implied carry measured **forward-to-forward inside one snapshot** (never
against the index — finding 12 and the §4 correction), against the funding
actually paid over the following tenor, 79 days per asset.

| | BTC 7d | BTC 30d | BTC 90d | ETH 7d | ETH 30d | ETH 90d |
|---|---|---|---|---|---|---|
| RMSE implied | **9.7** | **11.7** | **9.0** | 13.3 | **13.7** | **11.7** |
| RMSE trailing | 9.9 | 13.5 | 10.7 | **9.8** | 15.7 | 14.5 |
| slope | 0.72 | 0.63 | 0.68 | 0.83 | 0.86 | 0.70 |

The forward curve is the better forecast everywhere except ETH at 7 days,
and runs about a point rich to realised funding — the borrow and collateral
wedge, not a mispricing. Slopes below 1 with positive intercepts: the curve
is a shrunk, mean-reverting forecast of a much noisier series.

### 10.5 Held to expiry on live data — `hold_to_expiry.py`

The realistic cost structure: sell at the bid 1 or 7 days out, hedge on the
Whalley-Wilmott band, settle at intrinsic against the 08:00 index — **spread
paid once**. 23 positions over 16 days (11 BTC daily, 10 ETH daily, 1
weekly each). `--clock` re-runs the same inceptions on the old 4h clock.

The cost number: **9.4 bps of spread on a 147 bps premium, about 6%**,
against 34-46 bps a day for the round trip in §10.1. Outcomes over 16 days
of one regime are not a result — the dailies mean -14 bps gross with a
standard deviation of 119.

#### The hedge was frozen, and the guard did not see it

Putting a band on the book meant looking at *when* it traded, which is how
this surfaced. `compute_mid`'s wide-spread filter rejects ordinary one-day
ATM quotes — a real one: BTC 79,500 straddle at 0.0024/0.0028, which is
15.4% wide and four ticks, so both the relative and the tick rule reject it.
That filter asks whether a quote is good enough to *fit a surface through*.
For an option already held it is the wrong question, and dropping the quote
does not exclude a bad price: it removes the position's only mark, and an
unmarked position is never stepped, so the hedge froze at its last value.
On a typical BTC daily, 6 of 23 snapshots marked and the hedge sat unchanged
for 17 of the position's 24 hours.

**`max_gap_h` was measuring the wrong thing**, which is why nothing caught
it. It took gaps between the snapshots on disk rather than between the steps
the position was actually marked at, so it reported 3.0h on a position that
went 18h unhedged, and then set `hedged_properly`. The guard was blind to
precisely the failure it exists for. Both are fixed: `clean_snapshot` takes
`drop_wide=False` for marking a held position, and the gaps are measured
between marked steps. `hedge_points` now counts marks, `snapshots` counts
what was on disk; the two differing is the diagnostic.

What it cost the old numbers, BTC dailies, before and after the fix:

| | rebalances | slippage | gross |
|---|---|---|---|
| clock, frozen | 0.6 | 1.3 bps | -8.5 |
| clock, repaired | 2.4 | 6.8 bps | -4.0 |

So this section used to quote the cost of a hedge that had largely stopped
running. **§10.1 and §12 are unaffected** — checked, not assumed: the
archive study holds weeklies and monthlies, whose ATM premium is large
enough that the relative spread is never near 15%, and 313 of its 314
positions run the full 22 hours at 4.8 rebalances. The freeze needs a small
premium, so it only ever bit the short-dated live book.

#### What the band does here

Same 23 inceptions, repaired pipeline, clock → band:

| block | n | rebalances | gross | net |
|---|---|---|---|---|
| BTC 1d | 11 | 2.4 → 0.6 | -4.0 → **+3.1** | -19.2 → -10.4 |
| ETH 1d | 10 | 2.5 → 0.1 | -21.0 → **-33.6** | -38.9 → -45.5 |
| BTC 7d | 1 | 23 → 4 | +138.9 → +204.8 | +112.6 → +181.4 |
| ETH 7d | 1 | 23 → 0 | +110.9 → +415.6 | +83.4 → +407.0 |

**At one day the band degenerates into not hedging.** H goes as Gamma^(2/3)
and Gamma as 1/sqrt(T), so the band widens into expiry: about 0.43-0.56
delta at one day, against a straddle delta that spans ±1. It is the formula
behaving correctly, not a bug, but gamma = 1e-4 was walked forward on 7- and
30-day inceptions (§12.2) and the dailies are extrapolation outside it. The
live book is mostly dailies, so most of what the band does here is untested
by §12.2 — treat the daily rows above as a description, not a rule.

**§12.2's BTC/ETH asymmetry reproduces independently.** Apply its own gross
diagnostic: on BTC the band lifts gross (-4.0 → +3.1), so it is replicating
better; on ETH it gives up gross (-21.0 → -33.6) and net with it, which is
the same "lifted net by hedging less" pathology that failed ETH out of
sample on the archive. Different data, same verdict. That diagnostic keeps
earning its place as the thing to check first. The ETH 7-day +415.6 is one
unhedged position over a kind 16 days, n = 1, and is not evidence of
anything.

#### Rerun on 58 days, 2026-11-05

Track A now runs 2026-09-07 to 11-04, so the study gained a 30-day horizon
(the 10-30 monthly, sold 09-30). Same rule, same costs; band against clock on
matched inceptions:

| block | n | rebalances clock → band | gross clock → band | net clock → band |
|---|---|---|---|---|
| BTC 1d | 41 | 2.3 → 0.7 | +1.6 → +3.5 | -12.8 → **-9.4** |
| ETH 1d | 40 | 2.5 → 0.1 | +4.2 → +5.2 | -12.6 → **-6.4** |
| BTC 7d | 5 | 22.6 → 3.6 | -69.9 → +46.0 | -97.0 → +24.9 |
| ETH 7d | 5 | 21.6 → 0.6 | -75.2 → +16.1 | -99.7 → +2.5 |
| BTC 30d | 1 | 95 → 12 | -1.3 → +3.6 | -45.6 → -25.9 |
| ETH 30d | 1 | 86 → 2 | -16.7 → -58.8 | -51.3 → -78.6 |

- **The cost finding holds and is now on 81 dailies, not 21.** Spread paid
  once is 8-10 bps against a 125-164 bps premium, 6-7%. Net is slightly
  negative and indistinguishable from zero (t -0.8 BTC, -0.3 ETH) with a
  standard deviation of 77-131 bps a position.
- **The BTC/ETH asymmetry above does not survive more data.** At one day the
  band and the clock now differ by +1.9 gross on BTC and +1.0 on ETH, both
  t < 0.3. With 21 positions the ETH gap looked like §12.2's pathology; with
  81 it is noise. The paragraph above is left as written, as what 16 days
  showed.
- **7d and 30d are not hedged, so they say nothing about the rule.** Not one
  of the 12 longer positions ran without an 8h+ hedging gap: 7d gaps are
  11-25h, the 30d position's worst is 54h. That is the laptop sleeping, not
  the market. The 7d band-minus-clock gap (+116 gross BTC, t 2.1; +91 ETH,
  t 1.3; n = 5) is in the direction §12.1 predicts but cannot be separated
  from gaps landing on different moves under each rule.
- **The live 30-day trade is still untested.** One position per asset, both
  with 54h gaps. Testing it needs collection that does not sleep for a month
  (a server, or `health.py` keeping the laptop awake), then a few months of
  monthlies. Until then the archive's 44 inceptions (§12) remain the only
  30-day evidence.

### 10.6 Why the premium shrank — `vrp_hypotheses.py`

Full write-up, predictions and figures in `HYPOTHESES.md`. Each hypothesis's
predictions were written before its test ran; H6's were committed separately
(`d5fc7b9`) ahead of the code. 30-day DVOL premium against hourly realised,
1,977 days, Newey-West t with a 30-day lag throughout.

| | Hypothesis | Verdict |
|---|---|---|
| H0 | No stable premium | Premium ≠ 0 (t 4.8; 2023-26 alone [+2.1, +6.5]); a constant mean is not rejected |
| H1 | Competition caused a break | Not supported: break p 0.11-0.29, decline is a drift that the vol level absorbs |
| H2 | Leverage demand raises it | Contradicted: funding predicts it with the *opposite* sign |
| H3 | Jump compensation | Contradicted: high-jump windows earn *more* |
| H4 | Premium ≈ cost of harvesting it | Supported at 30 days in 2023-26 (+83 bps vs 31-86 bps cost) |
| H5 | BTC carries more than ETH | Supported as an association; reverses in 2026 |
| H6 | Fear, not leverage | Contradicted: drawdown and skew add nothing past the level; funding *strengthens* (-0.54 → -0.79) |

The surviving summary: **the premium scales with the vol level and is lower
when perp funding is high**, with no mechanism yet for the second part. Two
other findings worth knowing before reusing any number here: the premium
moves 3-6 vol points with the realised-variance sampling interval, and about
two thirds of the DVOL premium is smile convexity an ATM straddle does not
earn.

---

## 11. Operational notes

- `data/raw/`, `data/historical/` and `data/research/` are gitignored; no
  data is tracked. 62 tracked files, zero parquet or logs among them. So a
  fresh clone runs the tests but reproduces no study: the archive (105 MB,
  158 files), the hourly caches and every `data/research` output have to be
  copied from this machine, not fetched from GitHub.
- **Compute stays local unless it genuinely does not fit.** Measured at
  `--workers 4`: ~1.05 archive-day/s to fit and ~0.5 sweep-cell/s, so
  projected minutes are about `(days + cells x 2) / 60`. The §12 grid is 5
  minutes; `slurm/` exists for when a grid outgrows that and is deliberately
  unused otherwise, because queue wait and a 105 MB rsync cost more than the
  job saves at this size. `slurm/README.md` carries the arithmetic.
- **`core.autocrlf` is true here, so checkouts rewrite text files to CRLF.**
  A shell script that goes to a Linux box dies on that with
  `bad interpreter: /bin/bash^M`. `.gitattributes` pins `*.sh` and `*.slurm`
  to LF; extend it before adding anything else destined for a cluster.
- **Study caches are globbed, so two generations silently double-count.**
  `hedge_frequency.py` writes one file per shard and reads them all back, so
  a re-run at a different shard count would read both sets — means barely
  move, t inflates by ~sqrt(2). Both loaders refuse duplicates rather than
  averaging them, `run` clears the directories it owns, and a bare
  `fit`/`sweep` does not (shards are peers). Any future sharded study needs
  the same guard.
- **More than one Claude session works in this tree at once.** That produced
  three real failures: two fetch processes writing the same parquet paths, a
  `taskkill` from one session written into another's docstring as a crash it
  had diagnosed, and an uncommitted change swept into someone else's
  `git add -A`. Before killing a process or registering a task, check what
  else is running (`Get-CimInstance Win32_Process` — the archive job runs
  under `pythonw.exe`, so filtering on `python.exe` misses it). Commit your
  own files promptly, by explicit pathspec.
- **Tests must never touch `data/raw` or `data/historical`.**
  `tests/conftest.py` enforces it. `write_snapshot`'s directory is a default
  argument bound at definition, so patching the module's `RAW_DIR` does not
  redirect it — patch the function.
- `multiprocessing` cannot spawn workers from a script piped through stdin
  on Windows: the children re-import `<stdin>`, fail, and respawn forever.
  Write the script to a file.
- The shell here has `noclobber` set (`>` refuses to overwrite; use `>|`)
  and `rm` is interactive (`rm -f`, or it silently does nothing).
- The collector isolates currencies: one failing book does not cost the
  others. Observed once — a transient BTC fetch error where ETH and USDC
  both succeeded a minute later in the same run. 1 failure in 63 runs.
- Snapshots are named `<currency>_chain_<stamp>.parquet`. Globbing
  `*.parquet` across `data/raw/` would blend inverse and linear books into
  one "chain"; every script globs a currency prefix explicitly.
- Windows line endings: multi-line search-and-replace against these files
  fails silently because the files are CRLF. Use the editor, not `sed`.
- Long-running jobs: `| tail` buffers everything until EOF, and Python
  buffers stdout when redirected. Use `python -u` and no pipe, or the output
  file stays empty and the job looks hung.
- **Timing a study by sampling it in-process understates it.** Sampling
  `vrp_surface`'s per-day function in a warm loop suggested ~5 minutes and the
  file claimed ~25; timed end to end it is 8m02s. The gap is per-task parquet
  reads and the index frame being pickled to each worker, neither of which a
  warm in-process sample pays. Time the whole script before quoting a number.
  Its per-asset progress line also prints when a leg *starts*, so polling it
  measures your own poll interval rather than the leg.

---

## 12. Hedging frequency vs cost

`scripts/hedge_frequency.py`, 18 tests. Answers the question §5 left open:
the surviving premium is about the size of realistic costs, so does *any*
rebalancing frequency leave a positive number?

Two stages, because the expensive part is not the part being swept. Fitting
a day's surface is ~98% of the compute and depends on no swept parameter, so
stage 1 calibrates each of the 158 archive days once and banks the SVI
parameters plus a measured half-spread curve; stage 2 sweeps the grid over
that cache. Local cost: **2m31s to fit, 46s for all 48 cells.** Re-fitting
per cell would have burned ~1.9 CPU-hours recomputing identical surfaces.

**The option spread is measured, not assumed.** The archive carries bid and
ask per instrument, so each expiry gets a half-spread curve in vol points,
`(ask_usd - bid_usd) / 2 / vega`, interpolated in log-moneyness onto the
synthetic strike and clamped at the ends like the surface itself. Measured
half-spreads run 1.3–1.5 vol points at 8 hours to expiry down to ~0.36 at
two months — which on a 30-day ATM straddle is ~11 bps of spot, against the
9.4 bps §10.5 measured independently on the live book. Only the hedge leg's
cost is a parameter, because the hourly index carries no book to measure.

At zero costs the P&L reduces to `vrp_surface`'s to the last bit — asserted
over 50 random paths per option type in the tests, and confirmed on real
data below, where gross at 4h comes back as **+83 bps, exactly §10.2's
published number.** This is a cost model bolted onto that result, not a
second calculation of it.

### BTC 30-day ATM straddle, 2023-26, mean net bps of spot (t), 44 inceptions

| hedge | 0 bps | 2 bps | 5 bps | 10 bps |
|---|---|---|---|---|
| 1h | +73 (2.4) | +43 (1.4) | -2 (-0.1) | -78 (-2.5) |
| 2h | +74 (2.3) | +52 (1.6) | +18 (0.5) | -39 (-1.2) |
| 4h | +72 (2.1) | +55 (1.6) | +28 (0.8) | -15 (-0.4) |
| **6h** | +71 (1.8) | +55 (1.4) | **+33 (0.9)** | -5 (-0.1) |
| 12h | +60 (1.4) | +48 (1.1) | +29 (0.7) | -1 (-0.0) |
| 24h | +42 (0.9) | +32 (0.7) | +16 (0.3) | -9 (-0.2) |

### Where the money goes, at 5 bps, same cell

| hedge | premium | gross | spread | slippage | net | turnover |
|---|---|---|---|---|---|---|
| 1h | 1068 | +83 | 11 | 75 | -2 | 15.0x |
| 4h | 1068 | +83 | 11 | 44 | +28 | 8.7x |
| 6h | 1068 | +81 | 11 | 38 | +33 | 7.6x |
| 24h | 1068 | +53 | 11 | 26 | +16 | 5.1x |

**The premium survives the option spread and dies on the hedge.** Charging
the measured spread alone leaves +73 bps at t 2.4 — still significant. Five
bps of hedge slippage takes 38–75 bps of that depending on frequency, and
the t never gets back above 1. The optimum is real and sits where theory
says (4–6h: turnover has fallen 2x from hourly while gross has only given
up 2 bps), but a best case of **+33 bps at t 0.9** is not a trade.

Ten bps is negative at every frequency. Hourly hedging is the worst cell in
the table at 15x turnover — it spends 75 bps to collect 83.

The rest, briefly: **BTC 7-day** 2023-26 is flat-to-negative everywhere.
**The BTC call wing** — the other 2023-26 survivor in §10.2 — does not clear
costs either, best case +3 bps (t 0.2) at 6–12h. **ETH** is negative nearly
everywhere in 2023-26, down to -128 bps at 1h / 10 bps. **2020-22 is the
exception that holds**: BTC 7-day ATM keeps +143 bps (t 2.4) at 5 bps, and
the call wing +53 (t 3.1). The trade worked then and does not now, which is
what every other study in this file also says.

So §5's phrasing was right: hedging frequency decides the outcome. It just
decides it negatively at 2023-26 premium levels, unless hedge slippage is
genuinely under ~2 bps.

### 12.1 A no-trade band beats the clock, and by more as costs rise

`strategy/hedging.py`, 20 tests. The sweep above measured only one rule:
rebalance on a fixed clock. A clock trades when the calendar says so, not when
the position has drifted, so it pays to re-hedge something that has barely
moved and sits through something that has run. Sixteen rules now compete:
six clocks, five fixed no-trade bands, and five Whalley-Wilmott bands,
H = (3/2 lambda F Gamma^2 / gamma)^(1/3), which widens where gamma is low and
where cost is high. 128 cells, 5m18s locally.

All three families are the same walk over the same hourly path, differing only
in when they say "trade" — so a clock rule expressed this way reproduces the
old vectorised implementation *exactly* (asserted over 200 random combinations
plus a parametrised sweep), which in turn reproduces `vrp_surface` exactly.
The chain from §10.2 to a band's number is checked link by link rather than
assumed.

BTC 30-day ATM, 2023-26, 5 bps slippage:

| rule | gross | spread | slippage | net | t | trades | turnover |
|---|---|---|---|---|---|---|---|
| time 6h (best clock) | +81 | 11 | 38 | +33 | 0.9 | 120 | 7.6x |
| band 0.05 | +81 | 11 | 34 | +36 | 1.1 | 36 | 6.9x |
| **ww gamma=1e-3** | **+87** | 11 | **33** | **+43** | **1.2** | **38** | 6.6x |
| band 0.2 | +24 | 11 | 20 | -7 | -0.1 | 5 | 3.9x |

**The band's advantage scales with the cost it exists to economise on**, which
is the signature that matters more than any single cell:

| slippage | best clock | best band | gain |
|---|---|---|---|
| 0 bps | +74 | +76 | +2 |
| 2 bps | +55 | +61 | +5 |
| 5 bps | +33 | +43 | +10 |
| 10 bps | **-1** | **+13** | **+14** |

At zero cost there is nothing to economise and the gain is ~0, as it must be.
At 10 bps every clock is negative and the band is not. The same monotone
pattern holds in BTC 7-day (+2/+6/+10/+8) and ETH 30-day (+0/+1/+6/+24).
Selection across 16 rules would not track the cost axis this cleanly.

**Read the gross column to tell the two mechanisms apart.** A band can lift
net by trading less for the same replication, which is the result, or by
replicating worse, which is a smaller hedge wearing the same name. The WW
winner does the former — gross *rises* 81 to 87 while slippage falls 38 to 33,
so it re-allocates trades toward where gamma actually is rather than simply
doing fewer of them. Wide fixed bands do the latter: gross collapses 81 to 50
to 24 across band 0.05 / 0.1 / 0.2, and by band 0.2 the position is carrying
directional risk, not hedging it. ETH's winners at 5-10 bps also give up gross
(25 vs 32, 18 vs 24), so ETH's gains are partly under-hedging and should not
be read as the same finding.

**gamma is chosen in hindsight.** The +43 bps cell picks gamma = 1e-3 after
seeing all 44 inceptions, which is the selection this project refuses
everywhere else (`cross_asset` splits on an expanding median for exactly this
reason). The cost-axis monotonicity is what carries the finding, because it is
a predicted shape rather than a selected level; the *level* is optimistic and
should be read as an upper bound until it is walked forward. §7 step 1.

**The ceiling moved; the conclusion did not.** +43 bps at t 1.2 is a third
better than the best clock and still not a trade. What changed is that the
honest ceiling is now measured against a rule worth using rather than against
a deliberately naive one — and that at realistic costs the *rule* is worth
about as much as a third of the premium, which is not a small thing to know.

One consequence for anything learned later: a fitted hedging policy now has to
beat Whalley-Wilmott, not the clock. That is a much higher bar, and it is the
right one.

### 12.2 Choosing the band without hindsight — it survives on BTC, not on ETH

`walkforward` stage, 6 tests. §12.1's +43 bps picked gamma after seeing every
inception. This picks it at each inception from history that had already
**settled** — not merely from earlier inceptions, because a 30-day position
opened one month prior settles within hours of the next one, so counting it
would be a one-month lookahead wearing an expanding window's clothes. The rule
is then used once, blind. It is a re-aggregation of `hedge_cells`; no re-run.

Against the better clock fixed a priori, gap in bps, across warmups:

| block | slip | w=3 | w=6 | w=12 | w=18 | verdict |
|---|---|---|---|---|---|---|
| BTC 7d ATM | 2 | +4 | -1 | +2 | +7 | flips |
| **BTC 7d ATM** | **5** | **+16** | **+11** | **+15** | **+32** | **holds** |
| BTC 7d ATM | 10 | +6 | +1 | -5 | +11 | flips |
| BTC 30d ATM | 2 | -8 | -7 | +8 | +0 | flips |
| **BTC 30d ATM** | **5** | **+6** | **+2** | **+14** | **+10** | **holds** |
| **BTC 30d ATM** | **10** | **+6** | **+4** | **+24** | **+23** | **holds** |
| ETH 7d ATM | 2 | -21 | -36 | -2 | +12 | flips |
| ETH 7d ATM | 5 | -38 | -66 | -85 | -78 | **never** |
| ETH 7d ATM | 10 | -5 | -37 | -29 | -6 | **never** |
| ETH 30d ATM | 2 | -23 | -19 | -11 | -16 | **never** |
| ETH 30d ATM | 5 | -42 | -40 | -32 | -42 | **never** |
| ETH 30d ATM | 10 | -52 | -51 | -10 | -18 | **never** |

**On BTC at realistic costs the band survives.** Every cell at 5 bps and the
30-day at 10 bps beats the best a-priori clock at all four warmups. At 2 bps
the sign flips, which is what it should do — there is little cost to economise
on, so there is little for a band to win. The selection is also *stable*: the
walk-forward picks gamma = 1e-4 on 58-74 of 70-74 inceptions, so it is not
flip-flopping between rules, which is why it works.

**On ETH it fails everywhere, and §12.1 already said why.** ETH's in-sample
band winners gave up gross (25 against 32 at 5 bps) — they lifted net by
hedging *less*, carrying directional risk rather than replicating better. That
does not generalise, and the walk-forward is where it stops paying. BTC's
winner raised gross 81 to 87. **The gross diagnostic predicted which asset
would survive**, which is the most useful thing in either section: it is a
test that can be run before the out-of-sample check rather than after.

Two methodological findings worth keeping:

- **A narrower candidate set beats a wider one.** Selecting within the WW
  family beats selecting across all sixteen rules in nearly every block (BTC
  30-day at 5 bps: +59 against +49; at 2 bps: +72 against +37). More
  candidates is more noise to chase, at ~12 observations a year.
- **warmup = 6 was the least favourable value I could have picked.** On BTC
  30-day at 5 bps the gap is +2 at w=6 and +14 at w=12. A single warmup is a
  single draw; the table is the result, not any row of it.

**Levels, so the ceiling is not overstated.** BTC 30-day ATM at 5 bps,
ww-only walk-forward: +59 bps (t 1.4) at warmup 6, +75 (t 2.1) at warmup 12,
against a hindsight ceiling of +68 to +85. So roughly 70-90% of §12.1's
in-sample level is actually reachable — the rest was knowing which gamma in
advance. It is still not a trade: the one t above 2 is itself a warmup-selected
draw, and n is 65-70 overlapping monthly inceptions.

### Running it

```bash
python scripts/hedge_frequency.py run          # fit + sweep + report, 5m18s
python scripts/hedge_frequency.py report       # re-print from saved cells
python scripts/hedge_frequency.py walkforward  # choose the rule blind (§12.2)
python scripts/hedge_frequency.py walkforward --warmup 12
```

`run` owns `data/research/hedge_fits` and `hedge_cells` and clears them
first. Both loaders refuse a cache holding shards from two different runs
rather than averaging it — globbed together, every day is counted twice, the
means barely move and the t-statistics inflate by ~sqrt(2).

**Run it locally.** Measured throughput is ~1.05 archive-day/s for the fit
and ~0.5 sweep-cell/s at `--workers 4`, so projected minutes are about
`(days + cells x 2) / 60`, the 2 being the cost of walking the full hourly
path now that a band may trade at any hour. The 128-cell policy grid is
5m18s measured; it takes roughly 800 cells — a new strike or tenor axis, or
a much finer band grid — before a cluster beats sitting through it.

For when it does: `bash slurm/submit.sh` runs two Slurm array jobs chained
with `afterok`, grid axes overridable from the environment
(`HEDGE_HOURS="1 2 3 4 6 8 12 24" bash slurm/submit.sh`). Verified by
simulating a 4-way fit array and an 8-way sweep array locally — identical
slice counts, row counts and report. See `slurm/README.md` for what must be
rsynced across (the data directories are all gitignored) and why
`data/history.py` grew an `offline=True` path: compute nodes have no
outbound network, and the cache normally tops itself up from Deribit on
every call.
