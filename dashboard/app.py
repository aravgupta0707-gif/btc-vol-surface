"""
Streamlit dashboard for exploring the live vol surface.
Run with: streamlit run dashboard/app.py

See BUILD_GUIDE.md section 9 for what to include:
- interactive 3D surface (plotly)
- current smile per expiry
- backtest P&L chart with drawdown
"""

import streamlit as st

st.title("BTC Implied Volatility Surface")
st.write("TODO: load latest surface snapshot and render interactive plot.")
