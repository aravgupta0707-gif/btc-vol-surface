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
- [ ] SVI calibration
- [ ] Arbitrage-free validation
- [ ] Surface object
- [ ] Greeks + validation
- [ ] Strategy + backtest
- [ ] Dashboard / write-up
