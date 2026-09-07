# BTC Implied Volatility Surface + Vol Strategy — Technical Build Guide

## 0. What this project actually demonstrates

Every stage below maps to a specific thing quant interviewers probe for:

| Stage | What it proves |
|---|---|
| Data pipeline | You understand market microstructure noise, not just clean textbook data |
| BS inversion | You understand the model well enough to invert it robustly, not just call a library |
| SVI calibration | You know the industry-standard parameterization and can fit it |
| Arbitrage checks | You understand *why* naive interpolation is wrong — this is the part that separates a tutorial project from a real one |
| Strategy + backtest | You can turn a model into P&L with honest transaction costs |

Keep a running log of what broke and how you fixed it at each stage. That log is your interview material — it's more valuable than the final Sharpe number.

---

## 1. Data Pipeline

### 1.1 Deribit specifics
- Instrument naming: `BTC-{expiry}-{strike}-{C/P}`, e.g. `BTC-27DEC26-100000-C`.
- Key endpoints: `public/get_instruments` (discover the live option chain), `public/ticker` (best bid/ask, mark price, mark IV, and Deribit's own greeks — useful later as a validation reference), `public/get_order_book` (deeper book if you want more than top-of-book).
- Deribit publishes its own `mark_iv` per instrument. Don't use this as your IV — compute your own from mid price — but keep it alongside your computed value as a sanity check.
- BTC options settle against Deribit's **index price**, not raw spot from one exchange. Pull the index price series too; that's your correct underlying for both pricing and delta-hedging.

### 1.2 Cleaning rules (decide these explicitly, don't wing it)
- Compute mid = (bid + ask) / 2. Discard quotes where the spread exceeds some threshold (e.g. > 15% of mid) — these are typically illiquid deep-OTM strikes with unreliable prices.
- Discard zero-bid options entirely; a zero bid tells you nothing about IV.
- Time-to-expiry: Deribit options expire at 08:00 UTC. Compute time-to-expiry in years using exact UTC timestamps, not calendar-day counting — being off by a few hours matters a lot for near-dated options.
- Snapshot cadence: even hourly snapshots stored as timestamped parquet files are plenty. You don't need tick-level data for this project — resist the urge to over-engineer the data layer at the expense of the modeling stages, which is where the real signal is.

### 1.3 One nuance worth getting right
BTC options are economically closer to **Black-76** (forward-based) than spot Black-Scholes, because the relevant forward price already embeds the cost-of-carry implied by perpetual futures funding rates. Compute the forward price from the index + funding curve (or just use Deribit's quoted forward/future price for the matching expiry) and price off that. This is a small change mechanically but shows you understand *why* crypto derivatives pricing isn't a copy-paste of equity option pricing.

---

## 2. Black-Scholes (Black-76) Inversion — Price to IV

You have a market price; you need the σ that makes Black-76 output that price. This has no closed form, so you solve it numerically.

**Newton-Raphson step:**
```
σ_{n+1} = σ_n - (BS(σ_n) - market_price) / vega(σ_n)
```
where `vega` is the Black-76 vega (∂price/∂σ). This converges fast when it converges, but can diverge or oscillate for very short-dated, deep-OTM options where vega is tiny (the objective function is nearly flat, so the update step overshoots wildly).

**Practical implementation:**
1. Use the Brenner-Subrahmanyam approximation as your initial guess: `σ_0 ≈ sqrt(2π/T) * (price/F)`. This is close enough near-the-money that Newton converges in 2-3 iterations.
2. Cap iterations (e.g. 50) and check for divergence (σ going negative or exploding).
3. On Newton failure, fall back to **Brent's method** with a bracket like [0.01, 5.0] (1% to 500% annualized vol) — slower but guaranteed to converge if a root exists in the bracket.
4. If neither converges, the quote is bad — drop it rather than forcing a number.

This fallback logic is worth writing as its own tested function; it's the part of the pipeline most likely to silently produce garbage if you don't handle it explicitly.

---

## 3. SVI Calibration

### 3.1 The parameterization
Gatheral's raw SVI models **total implied variance** `w(k) = σ_BS(k)² · T` as a function of log-moneyness `k = ln(K/F)`:

```
w(k) = a + b·(ρ·(k - m) + √((k - m)² + σ²))
```

Each parameter has a concrete geometric meaning, which is why this beats a generic spline — you can reason about what's moving:
- `a` — overall variance level (vertical shift)
- `b` — controls the angle/steepness of both wings (must be ≥ 0)
- `ρ` — controls skew, i.e. rotation of the smile (must satisfy |ρ| < 1)
- `m` — horizontal shift, roughly where the smile's minimum sits
- `σ` — curvature at the money (must be > 0; larger = smoother ATM curve)

### 3.2 Fitting it
Fit **one SVI slice per expiry independently**. For each:
1. Build (k, w_market) pairs from your cleaned IVs.
2. Minimize a **vega-weighted** sum of squared errors — weight ATM points more heavily since they're the most liquid and most reliable; deep OTM points are noisy and shouldn't dominate the fit.
3. Optimize with `scipy.optimize.minimize` (L-BFGS-B), bounds: `b ≥ 0`, `-1 < ρ < 1`, `σ > 0`.
4. Initial guess: `a` ≈ min(w_market), `b` ≈ 0.1, `ρ` ≈ 0, `m` ≈ 0, `σ` ≈ 0.1 — refine from there.

After fitting each expiry independently, plot the fitted parameters *across* expiries. If `ρ` or `b` jump around erratically between adjacent expiries, that's usually a sign of noisy input data rather than a modeling problem — worth checking before you move on.

---

## 4. Arbitrage-Free Validation

This is the section that actually differentiates the project, so don't rush it.

### 4.1 Calendar arbitrage
At fixed log-moneyness `k`, total variance must be **non-decreasing in T**. Intuition: a longer-dated option can't imply *less* cumulative variance than a shorter-dated one — otherwise you could construct a calendar spread that's a guaranteed arbitrage. Check this numerically across your fitted expiry curves at a grid of `k` values.

### 4.2 Butterfly arbitrage (Durrleman's condition)
The fitted price surface must be **convex in strike** — otherwise a butterfly spread (long one low strike, short two mid strikes, long one high strike) has negative cost but a guaranteed non-negative payoff, which is free money and therefore can't exist in a no-arbitrage market. For SVI specifically, this reduces to checking that a function `g(k)` (built from `w(k)` and its first/second derivatives — this is in Gatheral's original paper, worth reading directly) stays non-negative across your strike range.

### 4.3 When you find violations
This will happen, especially at the wings. Options, in order of preference:
1. Tighten your input data filters (a lot of violations trace back to one bad quote).
2. Adjust optimizer bounds to gently restrict the fit near the wings.
3. As a last resort, blend adjacent expiries' parameters to smooth a bad slice.

**Log every violation you find and how you resolved it.** This is your best interview material — "here's what broke and why" is a far stronger story than a surface that fit cleanly on the first try.

---

## 5. Building the Queryable Surface Object

Wrap all of this behind a clean interface: `surface.iv(K, T)`, `surface.price(K, T, option_type)`, `surface.greeks(K, T)`. Internally:
- For `T` matching a fitted expiry, use that slice directly.
- For `T` between two fitted expiries, **interpolate linearly in total variance space** (not in IV space directly — variance is the quantity that's additive over time, so this is the theoretically correct place to interpolate).
- For `T` beyond your longest fitted expiry, flat-extrapolate the longest slice rather than guessing.

Designing this as an object with a clean interface — rather than a script that computes one plot — is what lets you say in an interview "the surface is queried independently of how it was built," which is a real software-design point, not just a modeling one.

---

## 6. Greeks

Using Black-76 formulas (forward-based, matching section 1.3):
- Delta, gamma, vega, theta all have closed forms once you have `σ` for that (K, T) from your surface.
- **Validation step**: compare your computed delta against Deribit's own quoted delta (available via the ticker endpoint) for the same instrument. They should match closely. Discrepancies here usually mean either a forward-price mismatch or a units error (annualized vs per-period) — worth tracking down before moving to the strategy layer, since your hedge ratios depend on this being right.

---

## 7. Strategy: Delta-Hedged Straddle

**Mechanics**: sell an ATM straddle (short call + short put at the same strike, typically ATM), then delta-hedge the resulting position using BTC perpetual futures, rebalancing periodically.

**Decisions to make explicit:**
- Rebalancing rule: fixed time interval (e.g. every 4 hours) vs. a delta-band threshold (rehedge when |delta| exceeds some bound). The band approach reduces unnecessary trading in calm periods; trade-off is complexity.
- **P&L decomposition** — this is the conceptual core of the strategy: your P&L is approximately `(implied variance − realized variance) × vega`, i.e. you're harvesting the gap between what the market paid for volatility and what actually happened. Being able to decompose your backtest P&L into theta carry vs. gamma/hedging P&L is exactly what a vol-desk interviewer will ask you to explain.
- Transaction costs: bid-ask cost on the initial straddle, hedging slippage on every rebalance, and the **funding rate** you pay/receive on the perpetual hedge (this is real cost in crypto and often gets forgotten — including it is a good sign of rigor).

---

## 8. Backtest Loop & Metrics

- **No lookahead**: at each timestamp, only use data that would have been available at that timestamp. This sounds obvious but is the most common way backtests silently lie.
- Walk forward through your snapshot history, marking positions to market and applying costs at each rebalance.
- Report: Sharpe ratio, max drawdown, turnover (how much hedging activity you generated), and win rate.
- **Sanity check**: your realized P&L should roughly track the theoretical `(implied − realized variance) × vega` estimate from section 7. If it doesn't, something in the hedging or cost logic is off — chase this down before trusting the Sharpe number.

---

## 9. Presentation Layer

**Repo structure** (roughly):
```
/data          — snapshot ingestion + cleaning
/pricing       — BS/Black-76 inversion, Greeks
/surface       — SVI calibration + arbitrage checks + surface object
/strategy      — position logic + backtest loop
/dashboard     — optional Streamlit app
README.md      — methodology + what broke + results
```

**README should include**: the pipeline diagram, the arbitrage violations you found and fixed, final Sharpe/drawdown with honest caveats about crypto-specific costs (funding, slippage), and a couple of surface snapshots.

**Interview talking points to prep** (have concrete answers ready, not just "it worked"):
- What arbitrage violation did you find, and how did you fix it?
- Why total-variance interpolation instead of IV interpolation?
- What's your strategy's actual source of P&L — decompose it.
- What would break this in a real market (regime change, liquidity drying up, etc.)?

---

## 10. Suggested Order of Attack

1. Data pipeline + IV inversion — get one clean smile for one expiry working end to end before touching SVI.
2. SVI fit + arbitrage checks — the technical core; budget the most time here.
3. Greeks + validation against Deribit's own numbers.
4. Strategy + backtest.
5. Polish, write-up, dashboard.

Don't move to step 2 until step 1 produces a smile you'd trust; don't move to step 4 until step 3's Greeks are validated. Each stage's output is the next stage's input, and debugging a bad surface *through* a bad strategy backtest is much harder than debugging it in isolation.
