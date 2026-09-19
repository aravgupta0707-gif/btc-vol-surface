"""
Tests for the archive download's failure handling.

Every case here is one that actually happened. One 429 was logged as a skip
and burned the remaining 53 days in seconds. One connection reset surfaced as
a bare urllib3 exception, escaped every handler, and left the archive idle
for three days. Both runs reported themselves as finished. The tests pin the
behaviour that replaced them: temporary failures wait and then stop, day
failures move on, and nothing half-written can pass for a finished day.
"""

from __future__ import annotations

import datetime as dt
import gzip
import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest
import requests
import urllib3

import data.tardis as tardis

ROOT = Path(__file__).resolve().parent.parent
DAY = dt.date(2022, 6, 1)


def _load_fetch_archive():
    spec = importlib.util.spec_from_file_location(
        "fetch_archive", ROOT / "scripts" / "fetch_archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_archive = _load_fetch_archive()


# --------------------------------------------------------------------------
# The stream: what a dropped or truncated transfer turns into
# --------------------------------------------------------------------------

def _archive_body() -> bytes:
    """A gzip archive file with a few hours of BTC and ETH quotes on DAY."""
    start = int(dt.datetime(2022, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1e6)
    values = {
        "type": "call", "strike_price": 30000, "open_interest": 1,
        "bid_price": 0.01, "bid_iv": 50, "ask_price": 0.011, "ask_iv": 52,
        "mark_price": 0.0105, "mark_iv": 51, "underlying_index": "BTC-1JUL22",
        "underlying_price": 31000, "delta": 0.5, "gamma": 0.001, "vega": 10,
        "theta": -5, "rho": 1, "expiration": start + 30 * 86_400_000_000,
    }
    lines = [",".join(tardis._KEEP)]
    for i in range(20_000):
        for currency in ("BTC", "ETH"):
            row = dict(values, symbol=f"{currency}-1JUL22-30000-C",
                       timestamp=start + i * 1_000_000)
            lines.append(",".join(str(row[c]) for c in tardis._KEEP))
    return gzip.compress(("\n".join(lines) + "\n").encode())


class _Raw(io.RawIOBase):
    """A response body that can drop the connection part way through."""

    def __init__(self, data: bytes, reset_at: int | None = None) -> None:
        self._buffer = io.BytesIO(data)
        self._reset_at = reset_at

    def read(self, size=-1):
        if self._reset_at is not None and self._buffer.tell() >= self._reset_at:
            raise urllib3.exceptions.ProtocolError(
                "Connection broken", ConnectionResetError(10054, "reset")
            )
        return self._buffer.read(size if size and size > 0 else -1)


@pytest.fixture
def serve(monkeypatch):
    def install(raw):
        class Response:
            def __init__(self):
                self.raw = raw

            def raise_for_status(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(tardis.requests, "get", lambda *a, **k: Response())
    return install


class TestStream:
    def test_a_clean_transfer_yields_both_currencies(self, serve):
        serve(_Raw(_archive_body()))
        frames = tardis.load_day_multi(DAY, 60, ("BTC", "ETH"))
        assert set(frames) == {"BTC", "ETH"}
        assert all(len(f) > 0 for f in frames.values())

    def test_a_reset_mid_transfer_is_a_requests_error(self, serve):
        # The regression: this used to escape as urllib3.ProtocolError, which
        # no handler in the project was written to catch.
        body = _archive_body()
        serve(_Raw(body, reset_at=len(body) // 2))
        with pytest.raises(requests.ConnectionError):
            tardis.load_day_multi(DAY, 60, ("BTC", "ETH"))

    def test_a_silently_truncated_transfer_is_an_error_not_a_short_day(self, serve):
        # A clean close part way through must never produce a partial frame
        # that gets written as if the day were complete.
        body = _archive_body()
        serve(_Raw(body[: len(body) // 2]))
        with pytest.raises(OSError):
            tardis.load_day_multi(DAY, 60, ("BTC", "ETH"))


class TestAtomicWrite:
    def test_writes_the_frame_and_leaves_no_temporary_file(self, tmp_path):
        path = tmp_path / "btc_chain_20220601.parquet"
        tardis.write_parquet_atomic(pd.DataFrame({"a": [1, 2]}), path)
        assert pd.read_parquet(path)["a"].tolist() == [1, 2]
        assert list(tmp_path.iterdir()) == [path]

    def test_a_failed_write_leaves_no_file_that_looks_finished(self, tmp_path):
        path = tmp_path / "btc_chain_20220601.parquet"
        unwritable = pd.DataFrame({"a": [object()]})
        with pytest.raises(Exception):
            tardis.write_parquet_atomic(unwritable, path)
        assert not path.exists()


# --------------------------------------------------------------------------
# Classification: does waiting help?
# --------------------------------------------------------------------------

def _http_error(code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = code
    return requests.HTTPError(f"{code}", response=response)


class TestClassify:
    @pytest.mark.parametrize("exc, expected", [
        (_http_error(429), "rate_limited"),
        (_http_error(503), "transient"),
        (_http_error(404), "permanent"),
        (requests.ConnectionError("reset"), "transient"),
        (requests.Timeout("slow"), "transient"),
        (OSError("Truncated compressed stream"), "transient"),
        (ConnectionResetError(10054, "reset"), "transient"),
        (RuntimeError("no BTC/ETH rows inside the UTC day"), "permanent"),
    ])
    def test_classification(self, exc, expected):
        assert fetch_archive.classify(exc) == expected


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------

DAYS = [dt.date(2022, 6, 1), dt.date(2022, 7, 1), dt.date(2022, 8, 1)]


def _frames(todo):
    return {c: pd.DataFrame({"expiry_ts": [1, 2], "x": [0.1, 0.2]}) for c in todo}


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Point the run loop at a temporary archive with a scripted loader."""
    monkeypatch.setattr(fetch_archive, "HIST_DIR", tmp_path / "historical")
    monkeypatch.setattr(fetch_archive, "LOG_PATH", tmp_path / "fetch.log")
    monkeypatch.setattr(tardis, "free_days", lambda: list(DAYS))
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    calls: list[dt.date] = []

    def script(outcomes):
        """outcomes: day -> list of exceptions to raise before succeeding."""
        pending = {day: list(errors) for day, errors in outcomes.items()}

        def load(day, bucket, todo):
            calls.append(day)
            if pending.get(day):
                raise pending[day].pop(0)
            return _frames(todo)

        monkeypatch.setattr(tardis, "load_day_multi", load)

    def files():
        return sorted(p.name for p in (tmp_path / "historical").glob("*.parquet"))

    def log():
        path = tmp_path / "fetch.log"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return type("Loop", (), {"script": staticmethod(script), "calls": calls,
                             "files": staticmethod(files), "log": staticmethod(log)})


class TestRunLoop:
    def test_a_brief_reset_is_retried_and_the_day_is_kept(self, loop):
        loop.script({DAYS[0]: [requests.ConnectionError("reset")] * 2})
        assert fetch_archive.run() == 0
        assert len(loop.files()) == 6
        assert "0 of 3 still missing" in loop.log()

    def test_a_persistent_reset_stops_the_run_and_leaves_later_days_untried(self, loop):
        retries = len(fetch_archive.BACKOFF_SECONDS)
        loop.script({DAYS[0]: [requests.ConnectionError("reset")] * (retries + 1)})
        assert fetch_archive.run() == 0
        assert loop.calls == [DAYS[0]] * (retries + 1)
        assert loop.files() == []
        assert "stopping" in loop.log()
        assert "3 of 3 still missing" in loop.log()

    def test_a_persistent_rate_limit_stops_the_run(self, loop):
        retries = len(fetch_archive.BACKOFF_SECONDS)
        loop.script({DAYS[0]: [_http_error(429)] * (retries + 1)})
        fetch_archive.run()
        assert set(loop.calls) == {DAYS[0]}
        assert "rate limited" in loop.log()

    def test_a_broken_day_is_recorded_and_the_run_moves_on(self, loop):
        loop.script({DAYS[0]: [_http_error(404)]})
        fetch_archive.run()
        assert loop.calls == DAYS
        assert "FAILED" in loop.log()
        assert "1 of 3 still missing" in loop.log()

    def test_days_already_on_disk_cost_no_request(self, loop):
        loop.script({})
        fetch_archive.run()
        loop.calls.clear()
        loop.script({})
        fetch_archive.run()
        assert loop.calls == []

    def test_a_complete_archive_runs_silently(self, loop):
        # The task stays registered hourly after completion; it must not
        # write a pair of log lines every hour forever.
        loop.script({})
        fetch_archive.run()
        before = loop.log()
        assert fetch_archive.run() == 0
        assert loop.log() == before
