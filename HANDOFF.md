# Handoff

Everything needed to pick this project up cold. Written 2026-09-09.

`BUILD_GUIDE.md` is the original brief. `PLAN.md` is the working plan with
per-phase checkpoints. `VIOLATIONS.md` is the arbitrage log. This file is the
state of play, what was learned, and what to do next.

---

## 1. State

Seven of eight build-guide stages complete. **253 tests, 15 commits, 5,643
lines of Python, zero remaining stubs.**

| Stage | State | Checkpoint |
|---|---|---|
| Data pipeline | done | `verify_data.py` — 32/32 |
| Black-76 inversion | done | `verify_pricing.py` — 8/8, matches Deribit `mark_iv` to 0.048 vol pts |
| SVI calibration | done | `fit_surface.py` — 10/10 slices, 0.165 vol pts median |
| Arbitrage validation | done | `check_arbitrage.py` — no violations among quoted strikes |
| Surface object | done | `verify_surface.py` — 10/10 |
| Greeks + validation | done | delta matches Deribit to 0.0007 median |
| Strategy + backtest | machinery done | theory gate tracks at ratio 1.51 |
| Dashboard / write-up | **not started** | — |

The one genuinely open item is the **statistical claim**: the backtest has
run on a single 41-hour position. See §5.

### Running things

```bash
python scripts/verify_data.py        # Phase 1 checkpoint, 32 checks
python scripts/verify_pricing.py     # Phase 2 checkpoint
python scripts/fit_surface.py        # Phase 3: fit SVI, report smoothness
python scripts/check_arbitrage.py    # Phase 4 (--log appends to VIOLATIONS.md)
python scripts/verify_surface.py     # Phase 5 checkpoint
python scripts/run_backtest.py       # Phase 6 on Track A data
python scripts/vrp_study.py          # variance risk premium, 2021-2026
python scripts/collector.py status   # the hourly collector
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

Measured reliability over 45.8 hours: 44 runs against 46 possible, 20 gaps
of exactly 60 minutes, 6 hours missed entirely. Every long gap is the laptop
sleeping, and each ends in a catch-up run at an odd minute — that is
`StartWhenAvailable` firing on wake. `-WakeToRun` would close those 6 hours
at the cost of waking the machine hourly; deliberately not enabled.

**Track B — the Tardis.dev free archive.** First day of each month,
`options_chain` from 2020-03-01, 79 days available. No API key, no payment,
but the endpoint needs *some* `Authorization` header present (a 403 without
one) and rejects Range requests. `data/tardis.py` streams and downsamples on
the fly — one day is ~516 MB gzipped and ~10.8M rows, half of it ETH.

**Deribit index history.** `DVOL` (their 30-day vol index, daily since
2021-03-24) and perpetual OHLC. One number per day, not chain data — you
cannot build a surface from it. Used only by `vrp_study.py`.

---

## 3. Architecture

```
data/fetch_deribit.py   one API call per currency for the whole chain
data/clean.py           filters, USD conversion, exact-UTC expiry, OTM select
data/tardis.py          Track B streaming loader (Arrow)
pricing/black76.py      price, vega, greeks, implied vol inversion
surface/svi.py          SVI calibration (quasi-explicit reduction)
surface/arbitrage.py    Durrleman g(k), calendar check, trust radius
surface/build.py        snapshot -> fitted slices (shared pipeline)
surface/surface.py      VolSurface: iv / price / greeks for any (K, T)
strategy/straddle.py    delta-hedged short straddle, position + costs
strategy/backtest.py    walk-forward loop, no-lookahead guards, metrics
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

Three separate times something failed by **reporting success**:

- `implied_vol` returned **0.001** for an option genuinely priced at 90% vol.
  A 110k-strike one-day call is worth ~1e-9 dollars, the objective is flat
  across the whole bracket, and Brent returned its left endpoint. Wrong in a
  way that looks like data, and it would have flowed into an SVI fit unseen.
- Task Scheduler reported `Last Result: 0` and a fresh `Last Run Time` while
  collecting **nothing** — `schtasks` defaults `DisallowStartIfOnBatteries`
  to true, parking the task in Queued on battery.
- SVI returned `b = 267,905, rho = -0.99999997` — a V-shaped kink chasing one
  corrupted quote, which scored *better* in least squares than the truth.

The unifying idea, and the single most useful lesson: **a result sitting on
a boundary is an artefact of where the search stopped, not an answer.** That
one observation fixed the Brent root, the SVI degeneracy, and the surface
clamping. `status` now warns explicitly on a Queued task.

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
   defect. `MIN_POINTS = 8`; thinner slices are refused, not forced.

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

---

## 5. Known limitations

**The strategy has not been shown to have edge.** One 41-hour position that
caught realized at half of implied (sold 34.26%, realized 16.79%, +$460).
The machinery is proven — the P&L decomposition tracks the theoretical
`(implied - realized variance) x vega` estimate at a ratio of 1.51 — but a
single favourable sample is not a result. The VRP study says to expect ~3-4
vol points gross in the current regime, not the 17.5 this sample caught.

**Track A has ~46 hours** of the ~336 that a two-week contiguous backtest
needs.

**The theory-gate ratio is approximate.** It uses the textbook
`0.8 * S * sqrt(T)` ATM vega proxy and assumes P&L accrues linearly in time,
which it does not for a short-dated option. A ratio in 0.5-2.0 confirms the
hedging and cost logic; it is not a precision measurement.

**The surface clamps rather than extrapolates.** Beyond a slice's quoted
range it returns the boundary value. Honest, but it means far-wing queries
are flat rather than smooth. A desk would fit a wing model instead.

**`dashboard/app.py` is still the original stub.**

---

## 6. In flight

**Track B archive download**, managed by `scripts/fetch_archive.py`
(`start` / `status` / `logs` / `stop`), log at
`data/research/archive_fetch.log`.

It runs as a scheduled task rather than a background process, for a reason
worth keeping: the first attempt used `Start-Process`, reached 26 of 79 days
and stopped with an empty stderr and no traceback -- killed with its parent
shell, not crashed. A scheduled task is the mechanism the collector already
proves survives.

It fetches **BTC and ETH together**, because the transfer is the entire cost
and one download carries every currency. Fetching ETH in a later pass would
re-transfer identical bytes for all 79 days. That also moves spin-off
project B off the wall-clock critical path -- ETH history arrives with BTC
history rather than needing weeks of live collection.
Fetching all 79 free days to `data/historical/btc_chain_YYYYMMDD.parquet`.
Resumable — `fetch_free_archive` skips days already on disk, so re-running
after an interruption costs nothing. Expect 2-4 hours; early days take ~25s,
2026 days ~258s.

When it finishes, the study to run is 79 **independent intraday** backtests:
enter a delta-hedged straddle at 00:00 UTC, rebalance every 4h, close at
23:00, one per archive day. That gives independent samples across six years
of regimes — statistically cleaner than one contiguous path, and the thing
that turns Phase 6 from machinery into a claim. `run_backtest.py` already
does the per-day mechanics; it needs a loader pointed at
`data/historical/` and an aggregation over days.

---

## 7. Next steps

In priority order:

1. **Run the 79-day Track B study** once the download lands. Report by era,
   not as a single pooled number — the VRP decay means a 2021-2026 average
   overstates what is available now.
2. **Phase 7 write-up.** The material in §4 is the write-up. The build guide
   is right that the violations log is worth more than a clean surface.
3. **Dashboard** — plotly 3D surface, per-expiry smile, backtest P&L. The
   `VolSurface` object already serves everything it needs.

---

## 8. Scoped spin-off projects

Ranked by value per unit of work given what already exists. All four reuse
the existing pipeline; none needs new infrastructure.

### A. Implied risk-neutral density monitor — **BUILT**

`surface/density.py` + `scripts/rnd_monitor.py`, 22 tests. See §10.

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

### B. BTC vs ETH implied vol spread — ~2 days

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

### C. Basis term structure vs perpetual funding — ~1 day

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

## 10. The RND monitor (spin-off A, built)

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

## 9. Operational notes

- `data/raw/`, `data/historical/` and `data/research/` are gitignored; no
  data is tracked. 36 tracked files, zero parquet or logs among them.
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
