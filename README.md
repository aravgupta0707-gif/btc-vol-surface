# BTC Implied Volatility Surface + Vol Strategy

Build an arbitrage-free implied volatility surface from live BTC options
data (Deribit), then backtest a delta-hedged volatility strategy on top
of it.

See `BUILD_GUIDE.md` for the full technical write-up of every stage —
data pipeline, Black-76 inversion, SVI calibration, arbitrage checks,
Greeks, strategy, and backtest.

## Suggested build order
1. `data/` — pull and clean Deribit options chains
2. `pricing/` — Black-76 pricing + implied vol inversion
3. `surface/` — SVI calibration, arbitrage-free checks, queryable surface object
4. `strategy/` + backtest loop
5. `dashboard/` — optional Streamlit exploration app

Don't move to the next stage until the current one's output is
something you'd trust — debugging a bad surface through a bad
backtest is much harder than debugging it in isolation.

## Setup
```
pip install -r requirements.txt
```

## Status
- [x] Data pipeline — live collector, cleaning, historical archive loader.
      `python scripts/verify_data.py` runs the checkpoint (33 checks);
      `python scripts/collector.py status` shows the hourly collector.
- [x] BS/Black-76 inversion — matches Deribit's own `mark_iv` to a median
      of 0.048 vol points. `python scripts/verify_pricing.py`
- [x] SVI calibration — 10/10 live slices, median 0.165 vol points against
      the market. `python scripts/fit_surface.py`
- [x] Arbitrage-free validation — no violations among quoted strikes;
      6 regions in the extrapolated wings, logged in `VIOLATIONS.md`.
      `python scripts/check_arbitrage.py`
- [x] Surface object — arbitrage-free by construction across the whole
      T-continuum. `python scripts/verify_surface.py`
- [x] Greeks + validation — delta matches Deribit's to 0.0007 median.
- [x] Strategy + backtest — delta-hedged short straddle with honest costs
      (spread, slippage, perp funding) and a walk-forward loop that cannot
      see forward. `python scripts/run_backtest.py`
- [x] Six-year studies — the premium across the smile, BTC vs ETH, carry vs
      funding, held to expiry. `HANDOFF.md` §10.
- [x] Rebalancing rule vs cost — 128-cell sweep of clocks against no-trade
      bands, spreads measured from the archive's own book. A Whalley-Wilmott
      band beats the best clock by more as costs rise, and survives being
      chosen without hindsight on BTC but not on ETH.
      `python scripts/hedge_frequency.py run` (~5 min locally) and
      `... walkforward`, `HANDOFF.md` §12. Shards to Slurm if the grid ever outgrows
      the laptop — `slurm/README.md` has the threshold.
- [ ] Dashboard / write-up
