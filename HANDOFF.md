# Handoff

Everything needed to pick this project up cold. Written 2026-09-09, updated
2026-09-20 after the archive completed and the six-year studies ran (§10).

`BUILD_GUIDE.md` is the original brief. `PLAN.md` is the working plan with
per-phase checkpoints. `VIOLATIONS.md` is the arbitrage log. This file is the
state of play, what was learned, and what to do next.

---

## 1. State

Every build-guide stage but the dashboard is complete. **348 tests, 28
commits, 9,294 lines of Python, zero remaining stubs.**

| Stage | State | Checkpoint |
|---|---|---|
| Data pipeline | done | `verify_data.py` — 34/34 |
| Black-76 inversion | done | `verify_pricing.py` — 8/8, matches Deribit `mark_iv` to 0.048 vol pts |
| SVI calibration | done | `fit_surface.py` — 10/10 slices, 0.165 vol pts median |
| Arbitrage validation | done | `check_arbitrage.py` — no violations among quoted strikes |
| Surface object | done | `verify_surface.py` — 10/10 |
| Greeks + validation | done | delta matches Deribit to 0.0007 median |
| Strategy + backtest | **done** | 79-day study, 316 positions — §10 |
| Write-up | **this file, §10** | — |
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
python scripts/vrp_surface.py        # premium by strike and tenor  [~25 min]
python scripts/cross_asset.py        # BTC vs ETH (needs vrp_surface first)
python scripts/carry_study.py        # forward curve vs funding
python scripts/hold_to_expiry.py     # Track A, settled at expiry  [~4 min]
python scripts/vrp_study.py          # DVOL-based VRP, 2021-2026
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

At 2026-09-20: **BTC 209 snapshots, ETH and USDC 185 each**, from
2026-09-07, 32 MB. Every long gap is the laptop sleeping, and each ends in a
catch-up run at an odd minute — `StartWhenAvailable` firing on wake.
`-WakeToRun` would close them at the cost of waking the machine hourly;
deliberately not enabled. So Track A is hourly but **not contiguous**, which
is what limits anything needing an unbroken path.

**Track B — the Tardis.dev free archive. COMPLETE: 79/79 days, BTC and
ETH**, 105 MB. First day of each month, `options_chain` from 2020-03-01. No
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
58,154 hours each, 2 missing. Cached in `data/research`, refreshed
incrementally. This is what makes the archive usable: the chains give one
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
strategy/backtest.py    walk-forward loop, no-lookahead guards, metrics
scripts/health.py       watchdog over everything that runs unattended
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
is marginal against costs.** 316 independent positions over six years say
the premium was only capturable in 2021-22, and pooled over the whole
period there is none even before costs (t 0.1-0.7). What survives in
2023-26 is BTC 30-day ATM and the call wing at roughly **+83 bps of spot a
month**, against an entry spread of ~18 bps and hedge slippage anywhere
from 10 to 60 bps depending on rebalancing frequency. So the surviving
premium is about the size of realistic costs, and **hedging frequency, not
the premium, decides the outcome.** That is the next thing to measure.

**Every study price is a mid.** No study charges the bid-ask on the option
except `hold_to_expiry` (entry only) and `archive_study` (both sides). Wing
spreads on Deribit are wide, so the call-wing premium in §10 is the most
cost-exposed result in the file.

**The statistics are thin and multiple.** About 12 inceptions a year; BTC
and ETH on the same day are not independent, nor are adjacent strikes; §10
reports 72 cells, so a couple of |t| > 2 are expected by chance. The two
results that survive being split by era are the BTC call wing and the
ETH/BTC dispersion.

**Track A is not contiguous.** ~13 days of hourly snapshots with sleep
gaps. Long holds and anything needing an unbroken path are limited by it;
positions left unhedged over a gap are marked in `hold_to_expiry`.

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

In priority order:

1. **Hedging-frequency study.** The one surviving result (BTC 30-day ATM
   and call wing, ~83 bps a month) is the same size as its costs, and
   slippage scales with rebalancing while hedging error scales against it.
   Run `vrp_surface` at 1h / 4h / 12h / daily hedging with the spread and
   5 bps slippage charged, and see whether any frequency leaves a positive
   number. This is the difference between "a premium exists" and "it can be
   collected", and everything needed is already on disk.
2. **The dispersion trade properly** — long ETH vol / short BTC vol,
   vega-neutral sizing rather than the approximation in §10, real spreads,
   and an entry rule on the spread's own history. It is the largest effect
   found (t 4.6 conditional) and the least examined.
3. **Dashboard** — plotly 3D surface, per-expiry smile, RND, backtest P&L.
   The `VolSurface` object already serves everything it needs. This is the
   last unstarted build-guide stage.
4. **Fix the RND tail** (§5): normalise the density over the quoted range,
   or mark a level as extrapolated when its *integration domain* leaves the
   quoted strikes rather than only when the level itself does.
5. **Waiting on data**: surface PCA and the inverse-vs-linear stress test
   (spin-off D) both need weeks more Track A.

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

**Surface PCA** (level/slope/curvature factors from the fitted time series)
is the natural sequel once weeks of Track A exist, but it is wall-clock
gated and needs more history than B does.

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

### 11.1 One-day positions, 79 days, both assets — `archive_study.py`

Sell the ATM straddle at 00:00, hedge every 4h, buy it back at 23:00; one
independent position per archive day. The tenor is **fixed** to a weekly
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
(-27 to -46). Across the 316 positions, gross P&L correlates +0.37 to +0.46
with the theoretical variance-gap estimate and shares its sign on 65-70% of
days: the machinery does what theory says, on a sample large enough to tell.

**A daily round trip cannot work** — it pays the spread twice for one day of
decay. That is a statement about the cost structure, not about the premium.

### 11.2 The premium across the smile — `vrp_surface.py`

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

### 11.3 BTC against ETH — `cross_asset.py`

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

### 11.4 Carry against funding — `carry_study.py`

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

### 11.5 Held to expiry on live data — `hold_to_expiry.py`

The realistic cost structure: sell at the bid 1 or 7 days out, hedge every
4h, settle at intrinsic against the 08:00 index — **spread paid once**.
19 positions over one fortnight (9 BTC daily, 8 ETH daily, 1 weekly each).

The cost number is the point: **~8.6 bps of spread on a ~125 bps premium,
about 7%**, against 34-46 bps a day for the round trip in §10.1. Outcomes
over a fortnight of one regime are not a result — the BTC dailies mean -8.6
bps gross with one -271 bps day on 19 September — and positions spanning a
sleep gap (max 13h) are marked, because a real book would not have one.

---

## 11. Operational notes

- `data/raw/`, `data/historical/` and `data/research/` are gitignored; no
  data is tracked. 53 tracked files, zero parquet or logs among them.
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
