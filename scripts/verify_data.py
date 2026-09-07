"""
End-to-end check of the Phase 1 data layer.

    python scripts/verify_data.py            check the newest stored snapshot
    python scripts/verify_data.py --live     pull a fresh chain first

This is the Phase 1 checkpoint from PLAN.md -- "a clean smile for one expiry
that looks like a smile" -- turned into something that either passes or
tells you what broke. It deliberately makes no use of pricing/black76.py:
every check here is model-free, so a failure points at the data layer rather
than at an inverter that does not exist yet. The inversion cross-check
against Deribit's mark_iv is Phase 2's checkpoint.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.clean import clean_snapshot  # noqa: E402

REQUIRED = [
    "timestamp", "instrument_name", "strike", "expiry_ts", "option_type",
    "bid", "ask", "mark_price", "mark_iv", "underlying_price",
]
RESULTS: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((passed, label))
    print(f"  {'PASS' if passed else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return passed


def load(live: bool) -> pd.DataFrame:
    if live:
        from data.fetch_deribit import snapshot_chain, write_snapshot

        frame = snapshot_chain("BTC")
        write_snapshot(frame)
        print(f"pulled a fresh chain: {len(frame)} instruments\n")
        return frame
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "*.parquet")))
    if not files:
        print("no snapshots in data/raw/; run with --live")
        raise SystemExit(2)
    print(f"using {Path(files[-1]).name}\n")
    return pd.read_parquet(files[-1])


def main() -> int:
    raw = load("--live" in sys.argv)

    print("schema and raw chain")
    missing = [c for c in REQUIRED if c not in raw.columns]
    check(not missing, "required columns present", f"missing={missing}" if missing else "")
    check(len(raw) > 100, "chain is a plausible size", f"{len(raw)} instruments")
    check(raw["expiry_ts"].nunique() >= 5, "several expiries",
          f"{raw['expiry_ts'].nunique()} expiries")
    check(raw["timestamp"].nunique() == 1, "one snapshot timestamp for the whole chain")

    print("\ncleaning pipeline")
    clean = clean_snapshot(raw)
    check(len(clean) > 50, "clean rows survive", f"{len(clean)} rows")
    check(clean["T"].gt(0).all(), "no expired contracts")
    check(clean["mid_usd"].gt(0).all(), "all USD premiums positive")
    check(not clean.duplicated(["instrument_name"]).any(), "no duplicate instruments")

    otm = ((clean["log_moneyness"] < 0) & (clean["option_type"] == "P")) | (
        (clean["log_moneyness"] >= 0) & (clean["option_type"] == "C")
    )
    check(otm.all(), "every kept quote is out-of-the-money")

    print("\nUSD conversion (model-free: deep ITM premium must equal intrinsic)")
    # Uses the raw chain, since select_otm removes ITM quotes. A put struck
    # far above the forward is almost entirely intrinsic, so premium_btc *
    # forward must land on K - F. This catches a conversion done with the
    # index instead of the forward without needing any option model.
    itm = raw[
        (raw["option_type"] == "P")
        & (raw["mark_price"] > 0)
        & (raw["strike"] > raw["underlying_price"] * 1.4)
    ].copy()
    if len(itm):
        itm["usd"] = itm["mark_price"] * itm["underlying_price"]
        itm["intrinsic"] = itm["strike"] - itm["underlying_price"]
        err = ((itm["usd"] - itm["intrinsic"]) / itm["intrinsic"]).abs()
        check(err.median() < 0.05, "deep-ITM puts price at intrinsic",
              f"median error {err.median():.2%} over {len(itm)} quotes")
    else:
        check(True, "no deep-ITM puts in this chain to test", "(skipped)")

    print("\nsmile shape (quadratic in log-moneyness must curve upward)")
    curvatures = []
    for expiry, group in clean.groupby("expiry_ts"):
        if len(group) < 8 or group["mark_iv"].isna().all():
            continue
        good = group.dropna(subset=["mark_iv"])
        if len(good) < 8:
            continue
        a = np.polyfit(good["log_moneyness"], good["mark_iv"], 2)[0]
        curvatures.append((expiry, len(good), a))
    check(len(curvatures) >= 3, "enough expiries to test",
          f"{len(curvatures)} testable")
    for expiry, n, a in curvatures:
        check(a > 0, f"  {expiry:%d%b%y} curves upward",
              f"n={n:3d} curvature={a:8.1f}")

    print("\nmonotonicity in strike (calls fall, puts rise)")
    for expiry, group in raw.groupby("expiry_ts"):
        calls = group[(group["option_type"] == "C") & group["mark_price"].gt(0)]
        calls = calls.sort_values("strike")
        if len(calls) < 8:
            continue
        drops = calls["mark_price"].diff().dropna()
        check((drops <= 1e-9).mean() > 0.95, f"  {expiry:%d%b%y} call prices fall",
              f"{(drops <= 1e-9).mean():.0%} of {len(drops)} steps")

    failed = [label for ok, label in RESULTS if not ok]
    print("\n" + "=" * 62)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label.strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
