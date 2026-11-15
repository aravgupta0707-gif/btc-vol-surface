"""
Smoke tests for `scripts/make_overview.py`: each chart builds from the saved
study outputs and writes a non-empty PNG. Skipped where a study has not been
run on this machine -- the charts read its results, they do not recompute them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "research"

spec = importlib.util.spec_from_file_location("make_overview", ROOT / "scripts" / "make_overview.py")
mo = importlib.util.module_from_spec(spec)
sys.modules["make_overview"] = mo
spec.loader.exec_module(mo)


def needs(*names):
    missing = [n for n in names if not (RES / n).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"study output missing: {missing}")


@pytest.fixture(autouse=True)
def to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "OUT", tmp_path)
    return tmp_path


def wrote(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10_000


@needs("vrp_hypotheses_btc.parquet")
def test_premium_history():
    assert wrote(mo.fig_premium())


@needs("hedge_cells")
def test_costs_and_backtest():
    assert wrote(mo.fig_backtest())


@needs("dispersion.parquet", "varswap.parquet")
def test_scoreboard():
    assert wrote(mo.fig_scoreboard())


@needs("inverse_linear_hourly.parquet")
def test_inverse_vs_linear():
    assert wrote(mo.fig_inverse_linear())


@needs("surface_grid.parquet")
def test_surface_factors():
    assert wrote(mo.fig_pca())
