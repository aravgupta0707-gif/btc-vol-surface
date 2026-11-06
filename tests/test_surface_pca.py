"""
Tests for the PCA helpers behind SURFACE_PCA.md: the slot picker, the
decomposition and its sign convention, and the change builder that must
never difference across a gap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("surface_pca", ROOT / "scripts" / "surface_pca.py")
sp = importlib.util.module_from_spec(spec)
sys.modules["surface_pca"] = sp
spec.loader.exec_module(sp)


def test_slots_pick_the_nearest_snapshot_and_skip_far_ones():
    t = lambda s: pd.Timestamp(s, tz="UTC")
    stamps = [t("2026-10-01 07:46"), t("2026-10-01 08:46"), t("2026-10-01 23:30")]
    slots = sp.select_slots(stamps)
    assert slots[t("2026-10-01 08:00")] == 0          # 14 min beats 46 min
    assert t("2026-10-01 20:00") not in slots         # nearest is 3.5h away


def test_pca_recovers_a_planted_level_factor_with_a_fixed_sign():
    rng = np.random.default_rng(0)
    load = np.linspace(2.0, 1.0, 10)                   # all one sign, falling
    X = np.outer(rng.standard_normal(500), load) + 0.1 * rng.standard_normal((500, 10))
    share, vec = sp.pca(X)
    assert share[0] > 0.95
    assert sp.cosine(vec[:, 0], load) > 0.99
    assert (vec[:, 0] > 0).all()                       # sign convention holds
    _, vec_neg = sp.pca(-X)
    assert (vec_neg[:, 0] > 0).all()                   # and is not data-dependent


def test_changes_never_span_a_missing_slot():
    idx = pd.DatetimeIndex(["2026-10-01 08:00", "2026-10-01 20:00", "2026-10-02 20:00"], tz="UTC")
    wide = pd.DataFrame({"a": [1.0, 3.0, 10.0]}, index=idx)
    spot = pd.Series([100.0, 110.0, 121.0], index=idx)
    dv, ret, ends = sp.changes_over(wide, spot, pd.Timedelta(hours=12))
    assert dv.tolist() == [[2.0]]                      # 08:00 -> 20:00 only
    assert abs(ret[0] - np.log(1.1)) < 1e-12
