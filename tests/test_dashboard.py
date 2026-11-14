"""
Tests for the dashboard's data layer, and a smoke test of the app itself.

The dashboard is only worth having if it shows the numbers the studies
report. So the checks here are that its arithmetic is the studies'
arithmetic: the equity curve and drawdown agree with `compute_metrics`, the
cost breakdown adds up to net, the surface grid sits on the surface, and the
density's CDF ends where the density's mass says it should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard import data as D
from surface.surface import VolSurface

NEAR = {"a": 0.004, "b": 0.030, "rho": -0.25, "m": 0.010, "sigma": 0.060}
FAR = {"a": 0.020, "b": 0.090, "rho": -0.32, "m": 0.020, "sigma": 0.140}


def surface() -> VolSurface:
    return VolSurface({0.10: NEAR, 0.50: FAR}, {0.10: 80_000.0, 0.50: 80_800.0},
                      {0.10: (-0.2, 0.15), 0.50: (-0.5, 0.4)})


def cells(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    premium = rng.normal(900, 100, n)
    gross = premium - rng.normal(850, 200, n)
    spread, slip = rng.uniform(5, 15, n), rng.uniform(10, 40, n)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="MS"),
        "premium_bps": premium, "gross_bps": gross,
        "spread_bps": spread, "slippage_bps": slip,
        "net_bps": gross - spread - slip,
    })


class TestStamps:
    def test_reads_the_utc_time_in_a_chain_filename(self):
        ts = D.stamp_of("data/raw/btc_chain_20261114T094725Z.parquet")
        assert ts == pd.Timestamp("2026-11-14 09:47:25", tz="UTC")

    def test_honours_the_data_root_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOL_DATA_ROOT", str(tmp_path))
        assert D.research("x.parquet") == tmp_path / "research" / "x.parquet"
        assert D.snapshot_files("BTC") == []
        assert D.load_cells().empty
        assert D.load_live_book().empty
        assert D.load_premium().empty
        assert D.atm_history("BTC").empty


class TestSurfaceViews:
    def test_grid_is_the_surface_at_each_node(self):
        s = surface()
        Ts = [0.10, 0.30, 0.50]
        grid = D.surface_grid(s, Ts, zs=np.array([-1.0, 0.0, 1.0]))
        assert grid.shape == (3, 3)
        for i, T in enumerate(Ts):
            F = s.forward(T)
            assert grid[i, 1] == pytest.approx(s.iv(F, T) * 100)
            atm = s.iv(F, T)
            assert grid[i, 2] == pytest.approx(s.iv(F * np.exp(atm * np.sqrt(T)), T) * 100)

    def test_density_cdf_ends_at_the_density_mass(self):
        from surface.density import total_mass

        s = surface()
        curve = D.density_curve(s, 0.5, points=4001)
        # Not asserted monotone: a clamped wing can make the density dip below
        # zero, which is the surface's property and the reason mass is shown.
        assert curve["cdf"].iloc[-1] == pytest.approx(total_mass(s, 0.5), abs=1e-6)
        # The price density integrates to the same mass as the k density.
        mass = np.trapezoid(curve["pdf"], curve["price"])
        assert mass == pytest.approx(curve["cdf"].iloc[-1], rel=1e-3)

    def test_levels_bracket_the_forward_and_stay_positive(self):
        for F in (95_000, 3_100, 140.0):
            levels = D.levels_around(F)
            below = [v for k, v in levels.items() if k.startswith("below")]
            above = [v for k, v in levels.items() if k.startswith("above")]
            assert below and above
            assert all(0 < v < F * 1.05 for v in below)
            assert all(v > F * 0.95 for v in above)

    def test_tail_table_reports_complementary_sides(self):
        s = surface()
        table, rep = D.tail_table(s, 0.5)
        assert len(table) == len(D.levels_around(rep["forward"]))
        assert table["probability"].between(0, 1).all()


class TestBacktestViews:
    def test_equity_and_drawdown_agree_with_compute_metrics(self):
        c = cells()
        series = c.set_index("date")["net_bps"]
        eq = D.equity_curve(series)
        m = D.cell_metrics(series)
        assert eq["equity"].iloc[-1] == pytest.approx(m["total_pnl"])
        assert eq["drawdown"].min() == pytest.approx(m["max_drawdown"])
        assert (eq["drawdown"] <= 0).all()
        assert m["n"] == len(c)

    def test_sharpe_is_annualised_at_twelve_inceptions(self):
        s = pd.Series([10.0, -5.0, 20.0, 0.0, 15.0])
        m = D.cell_metrics(s)
        assert m["sharpe"] == pytest.approx(s.mean() / s.std(ddof=1) * np.sqrt(12), rel=1e-3)
        assert m["t"] == pytest.approx(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))

    def test_cost_breakdown_adds_up_to_net(self):
        br = D.cost_breakdown(cells())
        assert br.drop("net").sum() == pytest.approx(br["net"])
        assert br["option spread"] < 0 and br["hedge slippage"] < 0

    def test_policies_order_clocks_then_bands_then_whalley_wilmott(self):
        got = D.policy_order(["ww 0.01", "band 0.1", "time 12h", "time 2h", "ww 0.0001",
                              "band 0.02"])
        assert got == ["time 2h", "time 12h", "band 0.02", "band 0.1", "ww 0.0001", "ww 0.01"]

    def test_policy_table_has_one_row_per_rule_and_cost(self):
        rows = []
        for policy in ("time 4h", "ww 0.001"):
            for slip in (0.0, 5.0):
                c = cells(seed=int(slip) + len(policy))
                rows.append(c.assign(policy=policy, slip_bps=slip))
        table = D.policy_table(pd.concat(rows))
        assert len(table) == 4
        assert list(table["policy"].astype(str)) == ["time 4h"] * 2 + ["ww 0.001"] * 2
        assert (table["n"] == 30).all()

    def test_eras_cut_on_year(self):
        frame = pd.DataFrame({"year": [2020, 2022, 2023, 2026]})
        assert list(D.in_era(frame, "2023-26")["year"]) == [2023, 2026]
        assert list(D.in_era(frame, "2020-22")["year"]) == [2020, 2022]


@pytest.mark.skipif(not D.snapshot_files("BTC"), reason="no Track A chains on disk")
def test_the_app_runs_end_to_end_on_real_data():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(D.ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value("ETH").run()
    assert not at.exception, at.exception
