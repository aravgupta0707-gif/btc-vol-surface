"""
Guard the real data directories for the whole test session.

A test once wrote a one-row synthetic "snapshot" into data/raw, where every
analysis would have read it as a market observation. Tests must work in
tmp_path; this fails the session loudly if any of them touches real data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARDED = [ROOT / "data" / "raw", ROOT / "data" / "historical"]


def _listing() -> dict[str, tuple[int, int]]:
    files = {}
    for directory in GUARDED:
        if directory.exists():
            for p in directory.glob("*.parquet"):
                stat = p.stat()
                files[str(p)] = (stat.st_size, stat.st_mtime_ns)
    return files


@pytest.fixture(autouse=True, scope="session")
def real_data_is_untouched():
    # The live collector may add a snapshot mid-session; only files that
    # existed before, or new files the tests created, are judged.
    before = _listing()
    yield
    after = _listing()
    changed = [p for p in before if p in after and after[p] != before[p]]
    removed = [p for p in before if p not in after]
    added = [p for p in after if p not in before
             and "_chain_2" in p and Path(p).stat().st_size < 10_000]
    problems = changed + removed + added
    assert not problems, f"tests modified real data: {problems[:5]}"
