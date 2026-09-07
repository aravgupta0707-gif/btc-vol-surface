# Build Plan

Working plan for the project laid out in `BUILD_GUIDE.md`. The guide covers
the *what* and *why* of each stage; this file covers the order of attack,
the checkpoints that gate each stage, and the API-specific gotchas that the
guide doesn't mention.

---

## Findings that shaped this plan

These come from probing the live Deribit API and the Tardis.dev archive
before writing any code. Three of them contradict or extend the build guide.

### 1. Premiums are quoted in BTC, not USD

`quote_currency: BTC`. A `BTC-25SEP26-125000-P` with the forward at 80,064
shows `mark_price: 0.5614` — that is 44,936 USD of intrinsic value divided
by 80,064.

```
price_usd = price_btc * underlying_price
```

Every quote needs this conversion before it reaches `pricing/black76.py`.
Feeding a premium of 0.56 into an inversion against a strike of 125,000
doesn't produce a slightly-wrong IV — it fails to bracket, and the quote gets
dropped. Skip this and most of the chain silently disappears.

These are **inverse options**, settled in BTC. That has a second consequence,
see §6 below.

### 2. Deribit hands you the forward, per expiry

Every row carries `underlying_index: "BTC-25SEP26"` and
`underlying_price: 80064.01`. That *is* F for that slice.

`BUILD_GUIDE.md` §1.3 suggests constructing the forward from the index plus a
funding curve. Don't — it's extra work and an extra error source. Use
Deribit's quoted per-expiry forward directly. The reasoning in §1.3 about
*why* Black-76 is the right model still stands; only the mechanics change.

### 3. One API call replaces 978

`public/get_book_summary_by_currency?currency=BTC&kind=option` returns the
entire chain — bid, ask, mark_price, mark_iv, underlying_price, open_interest
— in a single request.

`fetch_deribit.py::snapshot_chain` currently suggests looping `get_ticker`
per instrument. That's 978 round trips per snapshot and a rate limit. Keep
`get_ticker` only for the §6 Greeks validation, on a handful of instruments,
since the book summary doesn't include greeks.

### 4. The backtest has no data source on the public API

Deribit's public API returns *current* state only. There is no historical
options chain endpoint. `BUILD_GUIDE.md` §8 says "walk forward through your
snapshot history" — that history doesn't exist yet.

Everything upstream (data → IV → SVI → arbitrage → Greeks) works fine on a
single live snapshot. Only §7–8 are blocked, and on wall-clock time rather
than on code. Hence the two-track data strategy below.

### Live chain, as measured

```
978 option instruments, 12 expiries, ~735 with usable two-sided quotes
6SEP26 -> 25JUN27  (1 day to ~10 months)

front expiries (6-11 SEP):  23-37 usable quotes per slice
back expiries (OCT+):       88-97 usable quotes per slice
```

The front slices have barely enough points for a stable 5-parameter SVI fit.
Fit the back expiries first — see Phase 3.

---

## Data strategy: two tracks

### Track A — live collector (contiguous, starts empty)

Hourly snapshots to `data/raw/*.parquet` from the book-summary endpoint.
Accrues while Phases 2–5 are built. After 2–3 weeks it supports a genuine
contiguous walk-forward backtest on data collected first-hand.

**This has to start running on day one.** It is the long pole in the project.

### Track B — Tardis.dev free archive (historical, available now)

Verified working, no API key, no payment:

```
https://datasets.tardis.dev/v1/deribit/options_chain/YYYY/MM/DD/OPTIONS.csv.gz
```

- Free tier is **the first day of each month**. `options_chain` starts
  2020-03-01, so ~78 days are available through Aug 2026.
- Requires *any* `Authorization` header to be present (`Bearer x` works).
  Without one you get a 403. Range requests are also rejected — download whole.
- Columns include `bid_iv`, `ask_iv`, `mark_iv`, `underlying_index`,
  `underlying_price`, and **Deribit's own `delta`, `gamma`, `vega`, `theta`,
  `rho`** — exactly what §6 validation needs, historically.
- The Q4-2021/Q1-2022 promotional free key mentioned in Deribit's blog
  expired in March 2022. Don't count on it.

**Size warning:** one day is 516 MB gzipped, 2.1 GB raw, 10.8M rows — it's
every tick-level quote update. Roughly 50% is ETH; filter to `BTC-`.

The ingest must **stream-decompress and downsample on the fly**, never
materializing the CSV. Taking the last quote per instrument per hour gives
~23k rows/day, so all 78 days land at ~1.8M rows — a few hundred MB of
parquet, which is entirely manageable.

### Why both tracks are worth having

They're complementary, not redundant:

- **Track B gives regime coverage.** 78 days spread across six years —
  the COVID crash, the 2021 bull run, the 2022 bear market, and since. Far
  better than one live snapshot for testing whether SVI fitting and the
  arbitrage checks hold up when the market isn't calm.
- **Track A gives contiguity.** Consecutive hours, which Track B can't
  provide.

They also support two different backtests, see Phase 6.

---

## Phases

### Phase 0 — repo setup

- [ ] `git init`, first commit of the scaffold as-is
- [ ] add `pytest` to `requirements.txt`, create `tests/`
- [ ] delete `btc-vol-surface.zip` once the tree is committed

### Phase 1 — collector and cleaning

- [ ] `fetch_deribit.py` on the book-summary endpoint, timestamped parquet
      to `data/raw/` (already gitignored)
- [ ] **get it on a scheduler immediately** — Track A starts accruing
- [ ] `clean.py`: BTC→USD conversion, mid, spread filter (>15% of mid),
      zero-bid drop, exact UTC time-to-expiry against 08:00 expiries
- [ ] Track B loader: streaming download + hourly downsample + BTC filter

**Checkpoint:** a clean smile for one expiry that looks like a smile.

### Phase 2 — Black-76 and inversion

- [ ] `price`, `vega`
- [ ] Brenner-Subrahmanyam seed → Newton → Brent `[0.01, 5.0]` fallback →
      `None` on failure

The most test-worthy module in the project. Worth real tests:

- round trip: price → invert → recover σ
- put-call parity
- deep-OTM and 1-day-to-expiry, where Newton diverges and the Brent
  fallback is what saves you

**Checkpoint:** computed IV closely tracks Deribit's `mark_iv` across the
liquid strikes. If it doesn't, it is almost certainly the BTC/USD units
(finding #1). Do not proceed until this holds.

### Phase 3 — SVI calibration

- [ ] per-expiry fit, vega-weighted, L-BFGS-B, bounds per `BUILD_GUIDE.md` §3.1
- [ ] plot fitted params across expiries to spot instability

Start with the back expiries (88–97 usable points each), then work forward.
The 1–5 day slices have 23–37 points for 5 parameters and will be unstable;
that's a data property, not a bug in the fit.

**Checkpoint:** fitted params vary smoothly across adjacent expiries.

### Phase 4 — arbitrage checks

- [ ] calendar: `w(k,T)` non-decreasing in T at fixed k
- [ ] butterfly: Durrleman's `g(k) >= 0`
- [ ] **a violations log** — file, timestamp, what broke, how it was resolved

Expect violations at the wings and in the thin front slices. Per
`BUILD_GUIDE.md` §4.3 the log is the deliverable, not just the clean surface.
Running the checks across Track B's 78 days will surface far more interesting
failures than one calm live snapshot ever would.

### Phase 5 — surface object and Greeks

- [ ] `VolSurface`: total-variance interpolation, flat extrapolation past
      the long end
- [ ] Black-76 greeks
- [ ] validate delta against Deribit's own quoted delta

**Inverse-option wrinkle:** Deribit's quoted delta is in BTC terms, and the
payoff being BTC-settled means position value depends on the BTC price twice
— a linear USD hedge doesn't capture that. Your delta won't match theirs
until this is handled, and Phase 6's hedge ratios depend on it being right.

**Checkpoint:** delta matches Deribit's within a small tolerance across
liquid strikes.

### Phase 6 — strategy and backtest

Two backtests, one per data track:

**B: 78 independent intraday runs.** Each free Tardis day is a full day of
ticks. Enter a delta-hedged straddle at 00:00 UTC, rebalance every 4h, close
at 23:59. One day of implied vs. realized at a daily horizon, repeated across
78 days spanning six years. Independent samples across regimes — statistically
cleaner than a single contiguous run, and available now.

**A: one contiguous multi-week run** on collector data, once enough has
accumulated. Tests multi-day position management, which B by construction
cannot.

Either way: explicit rebalance rule, honest costs including perp funding, and
P&L decomposed into theta carry vs. gamma/hedging. Sanity-check realized P&L
against the theoretical `(implied − realized variance) × vega` before
trusting any Sharpe number.

### Phase 7 — dashboard and write-up

- [ ] Streamlit + plotly 3D surface, per-expiry smile, backtest P&L
- [ ] README: pipeline diagram, the violations log from Phase 4, final
      metrics with honest caveats, surface snapshots

---

## Dependency note

Phases 2–5 are the bulk of the work and none of them need accumulated
history — they run on a single live snapshot. Track A collects in the
background throughout; Track B is available from Phase 1 onward. Nothing
blocks on waiting except the contiguous backtest.

Per `BUILD_GUIDE.md` §10: don't advance a phase until its checkpoint holds.
Debugging a bad surface *through* a bad backtest is much harder than
debugging it in isolation.
