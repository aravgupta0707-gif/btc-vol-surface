"""
Tests for the collector's two defences against waking from sleep.

The catch-up run fires the moment the laptop wakes -- before the network is
back, and at the same instant as the watchdog. The first lost a snapshot on
each of 18 and 19 September; the second let the watchdog read a snapshot
half-written and call it corrupt.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path

import pandas as pd

from data.fetch_deribit import write_snapshot

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("collector", ROOT / "scripts" / "collector.py")
collector = importlib.util.module_from_spec(spec)
sys.modules["collector"] = collector
spec.loader.exec_module(collector)


class _Connection:
    def close(self):
        pass


def _network_after(failures: int, monkeypatch):
    attempts = []

    def connect(address, timeout):
        attempts.append(address)
        if len(attempts) <= failures:
            raise OSError("getaddrinfo failed")
        return _Connection()

    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return attempts


class TestWaitForNetwork:
    def test_a_connected_machine_does_not_wait(self, monkeypatch):
        attempts = _network_after(0, monkeypatch)
        assert collector.wait_for_network() is not None
        assert len(attempts) == 1

    def test_waits_through_a_reconnect_after_waking(self, monkeypatch):
        attempts = _network_after(6, monkeypatch)
        assert collector.wait_for_network(limit=1e9) is not None
        assert len(attempts) == 7

    def test_gives_up_rather_than_hanging_the_run(self, monkeypatch):
        clock = iter(range(0, 10_000, 5))
        monkeypatch.setattr("time.monotonic", lambda: next(clock))
        _network_after(10_000, monkeypatch)
        assert collector.wait_for_network(limit=60, poll=5) is None


class TestDescribe:
    def test_includes_the_root_cause(self):
        try:
            try:
                raise ConnectionError("getaddrinfo failed")
            except ConnectionError as inner:
                raise RuntimeError("failed after 3 attempts") from inner
        except RuntimeError as exc:
            text = collector._describe(exc)
        assert "failed after 3 attempts" in text and "getaddrinfo failed" in text


class TestAtomicSnapshot:
    def test_writes_a_readable_file_and_leaves_no_temporary(self, tmp_path):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-09-19 18:46:02"], utc=True),
            "strike": [80_000.0],
        })
        path = write_snapshot(frame, raw_dir=tmp_path, currency="BTC")
        assert path.name == "btc_chain_20260919T184602Z.parquet"
        assert len(pd.read_parquet(path)) == 1
        assert list(tmp_path.iterdir()) == [path]


class TestRunOnceRetries:
    def test_a_transient_failure_is_retried_and_the_snapshot_kept(self, monkeypatch, tmp_path):
        import data.fetch_deribit as fd

        calls = {"n": 0}

        def flaky(currency):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("get_book_summary_by_currency failed after 3 attempts")
            return pd.DataFrame({
                "timestamp": pd.to_datetime(["2026-09-19 18:46:02"], utc=True),
                "underlying_asset": [currency], "strike": [80_000.0],
            })

        monkeypatch.setattr(fd, "snapshot_chain", flaky)
        # write_snapshot's default directory is bound at definition time, so
        # patching RAW_DIR does not redirect it -- the first version of this
        # test wrote a one-row "snapshot" into the real data/raw.
        real_write = fd.write_snapshot
        monkeypatch.setattr(fd, "write_snapshot",
                            lambda frame, currency: real_write(frame, tmp_path, currency))
        monkeypatch.setattr(collector, "LOG_PATH", tmp_path / "collector.log")
        monkeypatch.setattr(collector, "COLLECT_CURRENCIES", ("BTC",))
        monkeypatch.setattr(collector, "wait_for_network", lambda: 0.0)
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert collector.run_once() == 0
        log = (tmp_path / "collector.log").read_text()
        assert "ok" in log and "after 1 retry" in log and "FAIL" not in log

    def test_a_persistent_failure_is_logged_once_with_its_root_cause(self, monkeypatch, tmp_path):
        import data.fetch_deribit as fd

        def down(currency):
            try:
                raise ConnectionError("Failed to resolve 'www.deribit.com'")
            except ConnectionError as inner:
                raise RuntimeError("failed after 3 attempts") from inner

        monkeypatch.setattr(fd, "snapshot_chain", down)
        monkeypatch.setattr(collector, "LOG_PATH", tmp_path / "collector.log")
        monkeypatch.setattr(collector, "COLLECT_CURRENCIES", ("BTC",))
        monkeypatch.setattr(collector, "wait_for_network", lambda: 0.0)
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert collector.run_once() == 1
        log = (tmp_path / "collector.log").read_text()
        assert log.count("FAIL") == 1 and "Failed to resolve" in log
