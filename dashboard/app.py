"""
Streamlit dashboard for the live vol surface and what was measured on it.

    streamlit run dashboard/app.py

BUILD_GUIDE §9 asks for an interactive 3D surface, the smile per expiry, and
the backtest P&L with drawdown. Those are the first and fourth tabs. The rest
show what the studies found, from their own cached outputs: the implied
density, how the surface has moved through Track A, the live book held to
expiry, and the premium over six years.

Layout only. Every number comes from `dashboard/data.py`, which is tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import data as D  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#eb6834", "#2e9e5b", "#c1121f", "#8a8984"
TENOR_SHADES = ["#9ec5f0", "#6aa6e6", "#2a78d6", "#1d5aa6", "#123b70"]

st.set_page_config(page_title="BTC vol surface", layout="wide")


def layout(fig: go.Figure, height: int = 420, title: str = "", **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=78, b=10),
                      title=dict(text=title, y=0.98, yanchor="top", x=0, xanchor="left",
                                 pad=dict(l=4)),
                      legend=dict(orientation="h", y=1.0, x=0, yanchor="bottom"), **kw)
    return fig


# ---------------------------------------------------------------- cached loads

@st.cache_data(show_spinner=False)
def files(asset: str) -> list[str]:
    return D.snapshot_files(asset)


@st.cache_data(show_spinner="Fitting SVI to the chain...", max_entries=24)
def built(path: str, asset: str):
    return D.build_surface(path, asset)


@st.cache_data(show_spinner="Loading the hedging sweep...")
def cells():
    return D.load_cells()


@st.cache_data
def atm_history(asset):
    return D.atm_history(asset)


@st.cache_data
def skew_history(asset, tenor):
    return D.skew_history(asset, tenor)


@st.cache_data
def live_book(rule):
    return D.load_live_book(rule)


@st.cache_data
def premium():
    return D.load_premium()


# ---------------------------------------------------------------- sidebar

st.sidebar.title("BTC vol surface")
asset = st.sidebar.radio("Asset", ["BTC", "ETH"], horizontal=True)
paths = files(asset)
if paths:
    stamps = [D.stamp_of(p) for p in paths]
    pick = st.sidebar.select_slider(
        "Chain snapshot (UTC)", options=list(range(len(paths))), value=len(paths) - 1,
        format_func=lambda i: f"{stamps[i]:%Y-%m-%d %H:%M}")
    path = paths[pick]
    st.sidebar.caption(f"{len(paths):,} hourly snapshots on disk, "
                       f"{stamps[0]:%Y-%m-%d} to {stamps[-1]:%Y-%m-%d}. "
                       "Gaps are laptop sleep.")
else:
    path = None
    st.sidebar.warning(f"No {asset} chains under {D.data_root() / 'raw'}. "
                       "Run the collector, or set VOL_DATA_ROOT.")

st.sidebar.markdown(
    "---\nThe headline: **the premium clears the option bid-ask and dies on the "
    "hedge.** How often you rebalance decides the outcome, not the premium. "
    "See `README.md` and `HANDOFF.md` §10, §12.")

tabs = st.tabs(["Surface", "Density", "History", "Backtest", "Live book", "Premium"])
b = built(path, asset) if path else None

# ---------------------------------------------------------------- surface

with tabs[0]:
    if b is None:
        st.info("No fitted surface for this snapshot.")
    else:
        surface, Ts = b["surface"], sorted(b["fits"])
        c1, c2, c3, c4 = st.columns(4)
        T_front = Ts[0]
        c1.metric("Index", f"{b['spot']:,.0f}")
        c2.metric(f"ATM vol, {T_front * 365.25:.0f}d", f"{surface.iv(surface.forward(T_front), T_front) * 100:.1f}%")
        c3.metric("Slices fitted", f"{len(Ts)} of {len(Ts) + len(b['refused'])}",
                  help="SVI refuses a slice with fewer than 8 usable quotes or a degenerate fit.")
        c4.metric("Quotes inverted", f"{len(b['quotes']):,}")

        left, right = st.columns([3, 2])
        with left:
            Z = D.surface_grid(surface, Ts)
            fig = go.Figure(go.Surface(
                x=D.Z_GRID, y=np.array(Ts) * 365.25, z=Z, colorscale="Viridis",
                colorbar=dict(title="IV %", len=0.7),
                hovertemplate="z %{x:.2f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
            fig.update_scenes(xaxis_title="standardised moneyness z", yaxis_title="days to expiry",
                              zaxis_title="implied vol (%)", camera=dict(eye=dict(x=-1.6, y=-1.5, z=0.8)))
            st.plotly_chart(layout(fig, 520, title=f"{asset} surface, {b['stamp']:%Y-%m-%d %H:%M}Z"),
                            use_container_width=True)
            st.caption("z = ln(K/F) / (ATM vol × √T): the coordinate that makes slices of "
                       "different maturity comparable. Beyond each slice's quoted range the "
                       "surface clamps, so edges can be flat by construction.")
        with right:
            labels = {T: f"{pd.Timestamp(b['fits'][T]['expiry']):%d %b %y}  ({T * 365.25:.0f}d)" for T in Ts}
            T = st.selectbox("Expiry", Ts, format_func=labels.get, index=min(2, len(Ts) - 1))
            s = D.smile(b, T)
            fig = go.Figure()
            fig.add_vrect(x0=s["lo"], x1=s["hi"], fillcolor=GREY, opacity=0.12, line_width=0,
                          annotation_text="quoted range", annotation_position="top left")
            fig.add_scatter(x=s["k"], y=s["iv"], name="SVI", line=dict(color=BLUE, width=2.5))
            q = s["quotes"]
            fig.add_scatter(x=q["log_moneyness"], y=q["iv"] * 100, mode="markers", name="market mid",
                            marker=dict(color=RED, size=7), customdata=q[["strike"]],
                            hovertemplate="K %{customdata[0]:,.0f}<br>IV %{y:.2f}%<extra></extra>")
            fig.add_scatter(x=q["log_moneyness"], y=q["mark_iv"], mode="markers", name="Deribit mark",
                            marker=dict(color=GREY, size=5, symbol="x"))
            fig.update_xaxes(title="log-moneyness k = ln(K/F)")
            fig.update_yaxes(title="implied vol (%)")
            st.plotly_chart(layout(fig, 440, title=f"Smile, F = {s['forward']:,.0f}"),
                            use_container_width=True)

        st.subheader("Slices")
        ft = D.fit_table(b)
        st.dataframe(ft, hide_index=True, use_container_width=True, column_config={
            "days": st.column_config.NumberColumn(format="%.1f"),
            "forward": st.column_config.NumberColumn(format="%.0f"),
            "atm_vol": st.column_config.NumberColumn("ATM vol %", format="%.2f"),
            "rmse_vol": st.column_config.NumberColumn("RMSE (vol pts)", format="%.3f"),
            "max_err_vol": st.column_config.NumberColumn("max err", format="%.3f"),
            "rho": st.column_config.NumberColumn(format="%.2f"),
            "k_lo": st.column_config.NumberColumn(format="%.3f"),
            "k_hi": st.column_config.NumberColumn(format="%.3f"),
        })
        st.caption("Short expiries are often refused: the quote filter keys on relative spread, "
                   "which rejects cheap short-dated options for being ordinary (HANDOFF §5). "
                   "SVI refuses fewer than 8 points or a degenerate kink rather than fit it.")

# ---------------------------------------------------------------- density

with tabs[1]:
    if b is None:
        st.info("No fitted surface for this snapshot.")
    else:
        surface, Ts = b["surface"], sorted(b["fits"])
        T = st.selectbox("Horizon", Ts, key="rnd_T",
                         format_func=lambda T: f"{T * 365.25:.0f} days",
                         index=min(3, len(Ts) - 1))
        curve = D.density_curve(surface, T)
        table, rep = D.tail_table(surface, T)
        F = rep["forward"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Forward", f"{F:,.0f}")
        c2.metric("5% / 95% quantile", f"{rep['quantiles'][0.05]:,.0f} / {rep['quantiles'][0.95]:,.0f}")
        mass_ok = abs(rep["total_mass"] - 1) <= 0.02
        c3.metric("Density mass", f"{rep['total_mass']:.4f}" + ("" if mass_ok else "  ⚠"),
                  help="Should be 1. More than 2% off means a clamped wing or a negative "
                       "density is distorting the tails.")

        view = curve[(curve["price"] > F * 0.4) & (curve["price"] < F * 1.8)]
        lo, hi = (F * np.exp(r) for r in b["ranges"].get(T, (np.nan, np.nan)))
        fig = go.Figure()
        if np.isfinite(lo):
            fig.add_vrect(x0=lo, x1=hi, fillcolor=GREY, opacity=0.12, line_width=0,
                          annotation_text="quoted strikes", annotation_position="top left")
        fig.add_scatter(x=view["price"], y=view["pdf"], fill="tozeroy", name="implied density",
                        line=dict(color=BLUE, width=2))
        fig.add_vline(x=F, line=dict(color=GREY, dash="dot"), annotation_text="forward")
        fig.update_xaxes(title=f"{asset} price at expiry")
        fig.update_yaxes(title="density", showticklabels=False)
        st.plotly_chart(layout(fig, 380, title="Risk-neutral density (Breeden-Litzenberger on SVI)"),
                        use_container_width=True)

        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "probability": st.column_config.NumberColumn(format="percent"),
            "move": st.column_config.NumberColumn("move from forward", format="percent"),
        })
        st.warning("Treat the tails with care. Across Track A, P(below 60k) moved with the "
                   "density's own total mass (corr +0.85), so hour-to-hour changes in a tail "
                   "probability are partly integration error. A level marked not extrapolated "
                   "can still integrate over the unquoted wing (HANDOFF §5).")

# ---------------------------------------------------------------- history

with tabs[2]:
    hist = atm_history(asset)
    if hist.empty:
        st.info("No surface grid cache. Run `python scripts/surface_pca.py` first.")
    else:
        fig = go.Figure()
        for tenor, color in zip(hist.columns, TENOR_SHADES):
            fig.add_scatter(x=hist.index, y=hist[tenor], name=f"{tenor}d", mode="lines",
                            line=dict(color=color, width=1.8), connectgaps=False)
        fig.update_yaxes(title="ATM implied vol (%)")
        st.plotly_chart(layout(fig, 380, title=f"{asset} ATM vol by tenor through Track A"),
                        use_container_width=True)
        st.caption("Level moves are 5-7x larger at 7 days than at 90 (SURFACE_PCA.md): the "
                   "first factor, 70-80% of 12-hour changes, is a level shift concentrated "
                   "at the short end.")
        tenor = st.select_slider("Skew tenor (days)", [7, 14, 30, 60, 90], value=30)
        sk = skew_history(asset, tenor)
        if not sk.empty:
            fig = go.Figure()
            fig.add_scatter(x=sk.index, y=sk["skew"], name="put minus call wing (z=-1 vs +1)",
                            line=dict(color=ORANGE, width=1.8))
            fig.add_scatter(x=sk.index, y=sk["spot"], name="index", yaxis="y2",
                            line=dict(color=GREY, width=1.2))
            fig.update_layout(yaxis=dict(title="skew (vol pts)"),
                              yaxis2=dict(title="index", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(layout(fig, 340, title=f"{tenor}-day skew against spot"),
                            use_container_width=True)
            st.caption("The second factor: when spot rises the call wing richens and this "
                       "falls (corr +0.36 BTC, +0.64 ETH between spot moves and call-wing moves).")

# ---------------------------------------------------------------- backtest

with tabs[3]:
    cf = cells()
    if cf.empty:
        st.info("No hedging-sweep cells. Run `python scripts/hedge_frequency.py`.")
    else:
        st.markdown("Delta-hedged **short straddle**, one inception a month on the archive "
                    "(Tardis free tier, 2020-26), held to expiry, entry spread measured per "
                    "strike from the archive's own book. Units are **bps of spot** per position.")
        c = st.columns(5)
        b_asset = c[0].selectbox("Asset", sorted(cf["asset"].unique()), key="bt_asset",
                                 index=sorted(cf["asset"].unique()).index(asset))
        sub = cf[cf["asset"] == b_asset]
        tenor = c[1].selectbox("Tenor (days)", sorted(sub["tenor"].unique()),
                               index=len(sub["tenor"].unique()) - 1)
        sub = sub[sub["tenor"] == tenor]
        zs = sorted(sub["z"].unique())
        z = c[2].selectbox("Strike, z", zs, index=zs.index(0.0) if 0.0 in zs else 0)
        sub = sub[sub["z"] == z]
        era = c[3].selectbox("Era", list(D.ERAS), index=2)
        sub = D.in_era(sub, era)
        slips = sorted(sub["slip_bps"].unique())
        slip = c[4].selectbox("Hedge slippage (bps)", slips, index=slips.index(5.0) if 5.0 in slips else 0)
        rules = D.policy_order(sub["policy"].unique())
        rule = st.radio("Rebalancing rule", rules, horizontal=True,
                        index=rules.index("ww 0.001") if "ww 0.001" in rules else 0)
        cell = sub[(sub["policy"] == rule) & (sub["slip_bps"] == slip)]

        if len(cell) < 2:
            st.info("Too few inceptions in this cell.")
        else:
            series = cell.set_index("date")["net_bps"].sort_index()
            m = D.cell_metrics(series)
            k = st.columns(6)
            k[0].metric("Mean net", f"{m['mean_step_pnl']:+.0f} bps")
            k[1].metric("t", f"{m['t']:.2f}")
            k[2].metric("Sharpe (ann.)", f"{m['sharpe']:.2f}")
            k[3].metric("Max drawdown", f"{m['max_drawdown']:,.0f} bps")
            k[4].metric("Hit rate", f"{m['hit_rate']:.0%}")
            k[5].metric("Inceptions", f"{m['n']}")

            eq = D.equity_curve(series)
            left, right = st.columns([3, 2])
            with left:
                fig = go.Figure()
                fig.add_bar(x=eq.index, y=eq["pnl"], name="per inception",
                            marker_color=np.where(eq["pnl"] >= 0, BLUE, ORANGE), opacity=0.45)
                fig.add_scatter(x=eq.index, y=eq["equity"], name="cumulative",
                                line=dict(color=BLUE, width=2.5))
                fig.add_scatter(x=eq.index, y=eq["drawdown"], name="drawdown", fill="tozeroy",
                                line=dict(color=RED, width=1), fillcolor="rgba(193,18,31,0.15)")
                fig.update_yaxes(title="bps of spot")
                st.plotly_chart(layout(fig, 400, title=f"{rule}, {slip:.0f} bps slippage"),
                                use_container_width=True)
            with right:
                br = D.cost_breakdown(cell)
                fig = go.Figure(go.Waterfall(
                    x=br.index, y=list(br.values[:-1]) + [br.values[-1]],
                    measure=["relative"] * 4 + ["total"],
                    increasing=dict(marker_color=BLUE), decreasing=dict(marker_color=ORANGE),
                    totals=dict(marker_color=GREEN if br["net"] >= 0 else RED),
                    text=[f"{v:+.0f}" for v in br.values], textposition="outside"))
                fig.update_yaxes(title="mean bps of spot")
                st.plotly_chart(layout(fig, 400, title="Where the money goes", showlegend=False),
                                use_container_width=True)

            st.caption("Sharpe is annualised at 12 inceptions a year. The inceptions overlap "
                       "(30-day holds, monthly starts) so they are not independent draws, and "
                       "the standard deviation dwarfs the mean. Descriptive, not something to "
                       "size on.")

        st.subheader("Every rule against every cost")
        pt = D.policy_table(sub)
        heat = pt.pivot(index="policy", columns="slip_bps", values="mean")
        tt = pt.pivot(index="policy", columns="slip_bps", values="t")
        text = [[f"{v:+.0f}<br>t {t:.1f}" for v, t in zip(rv, rt)] for rv, rt in zip(heat.values, tt.values)]
        lim = float(np.nanmax(np.abs(heat.values))) if heat.size else 1.0
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=[f"{s:.0f} bps" for s in heat.columns], y=[str(p) for p in heat.index],
            text=text, texttemplate="%{text}", colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
            colorbar=dict(title="net bps")))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(fig, 40 + 34 * len(heat), title="Mean net bps of spot (t)"),
                        use_container_width=True)
        st.caption("Clocks rebalance every N hours; fixed bands when |Δ| drifts past the band; "
                   "Whalley-Wilmott ('ww', γ) when it leaves a band whose width scales with cost "
                   "and gamma. The band's advantage grows with cost. Choosing γ here is "
                   "hindsight: HANDOFF §12.2 walks it forward (BTC survives, ETH does not).")

# ---------------------------------------------------------------- live book

with tabs[4]:
    rule = st.radio("Hedging rule", ["whalley_wilmott", "clock"], horizontal=True,
                    format_func={"whalley_wilmott": "Whalley-Wilmott band", "clock": "4-hour clock"}.get)
    lb = live_book(rule)
    lb = lb[lb["asset"] == asset] if not lb.empty else lb
    if lb.empty:
        st.info("No live-book results. Run `python scripts/hold_to_expiry.py`.")
    else:
        st.markdown("Straddles sold on the **live hourly chain** (Track A) and held to expiry, "
                    "paying the spread once. Positions with a hedging gap from laptop sleep "
                    "are flagged; every 7- and 30-day position has one.")
        summary = lb.groupby("horizon").agg(
            n=("net_bps", "size"),
            properly_hedged=("hedged_properly", lambda s: int(s.astype(bool).sum())),
            premium=("premium_bps", "mean"), gross=("gross_bps", "mean"),
            spread=("spread_bps", "mean"), slippage=("slippage_bps", "mean"),
            net=("net_bps", "mean"), net_t=("net_bps", D.t_stat),
            max_gap_h=("max_gap_h", "max"))
        st.dataframe(summary, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(format="%.1f")
            for c in ["premium", "gross", "spread", "slippage", "net", "max_gap_h"]} | {
            "net_t": st.column_config.NumberColumn("t", format="%.2f")})
        horizon = st.radio("Horizon", sorted(lb["horizon"].unique()), horizontal=True)
        h = lb[lb["horizon"] == horizon].sort_values("expiry")
        eq = D.equity_curve(h.set_index("expiry")["net_bps"])
        ok = h.set_index("expiry")["hedged_properly"].astype(bool).reindex(eq.index)
        fig = go.Figure()
        fig.add_bar(x=eq.index, y=eq["pnl"], name="per position",
                    marker_color=np.where(ok, BLUE, GREY), opacity=0.55)
        fig.add_scatter(x=eq.index, y=eq["equity"], name="cumulative", line=dict(color=BLUE, width=2.5))
        fig.add_scatter(x=eq.index, y=eq["drawdown"], name="drawdown", fill="tozeroy",
                        line=dict(color=RED, width=1), fillcolor="rgba(193,18,31,0.15)")
        fig.update_yaxes(title="net bps of spot")
        st.plotly_chart(layout(fig, 380, title=f"{asset} {horizon}: grey bars had a hedging gap"),
                        use_container_width=True)

# ---------------------------------------------------------------- premium

with tabs[5]:
    pr = premium()
    if pr.empty:
        st.info("No premium panel. Run `python scripts/vrp_hypotheses.py`.")
    else:
        roll = st.slider("Rolling mean (days)", 30, 365, 90, step=15)
        split = pd.Timestamp("2023-01-01")
        fig = go.Figure()
        fig.add_scatter(x=pr.index, y=pr["dvol"], name="implied (DVOL)", line=dict(color=BLUE, width=1.4))
        fig.add_scatter(x=pr.index, y=pr["realized"], name="realised, next 30 days",
                        line=dict(color=ORANGE, width=1.4))
        fig.add_vline(x=split, line=dict(color=GREY, dash="dot"))
        fig.update_yaxes(title="annualised vol (%)")
        st.plotly_chart(layout(fig, 340, title="BTC 30-day implied against what was then realised"),
                        use_container_width=True)
        fig = go.Figure()
        fig.add_scatter(x=pr.index, y=pr["premium"], name="daily", fill="tozeroy",
                        line=dict(color=BLUE, width=0.5), fillcolor="rgba(42,120,214,0.18)")
        fig.add_scatter(x=pr.index, y=pr["premium"].rolling(roll, min_periods=roll // 2).mean(),
                        name=f"{roll}-day mean", line=dict(color=BLUE, width=2.5))
        fig.add_hline(y=0, line=dict(color=GREY))
        fig.update_yaxes(title="premium (vol pts)")
        st.plotly_chart(layout(fig, 300, title="The premium"), use_container_width=True)
        early, late = pr[pr.index < split]["premium"], pr[pr.index >= split]["premium"]
        c = st.columns(3)
        c[0].metric("Mean premium 2021-22", f"{early.mean():+.1f} vol pts")
        c[1].metric("Mean premium 2023-26", f"{late.mean():+.1f} vol pts")
        c[2].metric("Mean DVOL, era to era", f"{pr[pr.index < split]['dvol'].mean():.1f} → "
                    f"{pr[pr.index >= split]['dvol'].mean():.1f}")
        st.caption("Mostly the vol level: the premium scales with DVOL (+0.34 per point, t 4.0) "
                   "and implied fell from 81.5 to 49.9. High funding goes with a lower premium "
                   "once the level is controlled. See HYPOTHESES.md.")
