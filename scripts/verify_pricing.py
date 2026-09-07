"""
Phase 2 checkpoint: does our Black-76 inversion reproduce Deribit's own vol?

    python scripts/verify_pricing.py            newest stored snapshot
    python scripts/verify_pricing.py --live     pull a fresh chain first

Deribit publishes `mark_iv` per instrument. We never use it as our vol, but
it is a free independent implementation to check against, and agreeing with
it closely is strong evidence that the whole chain -- BTC to USD conversion,
forward selection, time to expiry, and the root finder -- is right.

The second check needs no reference at all: a call and a put on the same
strike must imply the same vol, or put-call parity is violated. That one is
pure market data against our own model.
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
from pricing.black76 import implied_vol_many  # noqa: E402

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
    clean = clean_snapshot(raw)

    clean["iv"] = implied_vol_many(
        clean["forward"], clean["strike"], clean["T"],
        clean["mid_usd"], clean["option_type"].tolist(),
    )
    inverted = clean.dropna(subset=["iv"]).copy()

    print("inversion coverage")
    rate = len(inverted) / len(clean)
    check(rate > 0.95, "almost every clean quote inverts",
          f"{len(inverted)}/{len(clean)} = {rate:.1%}")
    check(inverted["iv"].between(0.05, 3.0).all(), "all vols are plausible",
          f"range {inverted['iv'].min():.1%} to {inverted['iv'].max():.1%}")

    print("\nagreement with Deribit's mark_iv (vol points)")
    ref = inverted.dropna(subset=["mark_iv"]).copy()
    ref["err"] = ref["iv"] * 100 - ref["mark_iv"]
    median_abs = ref["err"].abs().median()
    p95 = ref["err"].abs().quantile(0.95)
    check(median_abs < 0.5, "median absolute error under 0.5 vol points",
          f"{median_abs:.4f} over {len(ref)} quotes")
    check(p95 < 2.0, "95th percentile under 2 vol points", f"{p95:.4f}")
    check(abs(ref["err"].median()) < 0.25, "no systematic bias",
          f"median signed error {ref['err'].median():+.4f}")

    print("\n  by expiry:")
    print(f"    {'expiry':>9} {'n':>4} {'med|err|':>9} {'max|err|':>9} {'our ATM':>8}")
    for expiry, group in ref.groupby("expiry_ts"):
        atm = group.iloc[(group["log_moneyness"].abs()).argmin()]
        print(f"    {expiry:%d%b%y} {len(group):4d} "
              f"{group['err'].abs().median():9.4f} {group['err'].abs().max():9.4f} "
              f"{atm['iv'] * 100:7.2f}%")

    print("\nput-call parity on live quotes (no reference needed)")
    # Both sides of a strike must imply the same vol. Uses the raw chain,
    # since select_otm keeps only one side of each strike.
    pairs = clean_snapshot(raw, otm_only=False)
    pairs["iv"] = implied_vol_many(
        pairs["forward"], pairs["strike"], pairs["T"],
        pairs["mid_usd"], pairs["option_type"].tolist(),
    )
    pairs = pairs.dropna(subset=["iv"])
    wide = pairs.pivot_table(
        index=["expiry_ts", "strike"], columns="option_type", values="iv"
    ).dropna()
    if len(wide) >= 10:
        gap = (wide["C"] - wide["P"]).abs() * 100
        check(gap.median() < 1.0, "call and put imply the same vol",
              f"median gap {gap.median():.4f} vol points over {len(wide)} strikes")
        check(gap.quantile(0.9) < 3.0, "parity holds in the tail",
              f"90th percentile {gap.quantile(0.9):.4f}")
    else:
        check(True, "too few two-sided strikes to test parity", "(skipped)")

    print("\nround trip through our own model")
    from pricing.black76 import price as bs_price

    repriced = np.array([
        bs_price(row.forward, row.strike, row.T, row.iv, row.option_type)
        for row in inverted.itertuples()
    ])
    rel = np.abs(repriced - inverted["mid_usd"].to_numpy()) / inverted["mid_usd"]
    check(rel.max() < 1e-6, "repricing at the recovered vol returns the input",
          f"max relative error {rel.max():.2e}")

    failed = [label for ok, label in RESULTS if not ok]
    print("\n" + "=" * 62)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label.strip()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
