"""
Tests for the watchdog's decisions.

The replay tests reconstruct failures that actually happened here and assert
the watchdog would have raised them. The quiet tests matter as much: a laptop
asleep overnight, an archive waiting out a rate limit, and a problem already
reported must all stay silent, or the notifications get ignored.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("health", ROOT / "scripts" / "health.py")
health = importlib.util.module_from_spec(spec)
sys.modules["health"] = health   # dataclasses resolve annotations through it
spec.loader.exec_module(health)

NOW = dt.datetime(2026, 9, 13, 18, 16, tzinfo=dt.timezone.utc)


def hours_ago(h: float) -> dt.datetime:
    return NOW - dt.timedelta(hours=h)


def hourly_runs(first_hours_ago: float, last_hours_ago: float = 1.0) -> list[dt.datetime]:
    runs, h = [], first_hours_ago
    while h >= last_hours_ago:
        runs.append(hours_ago(h))
        h -= 1
    return runs


# --------------------------------------------------------------------------
# Asleep versus broken
# --------------------------------------------------------------------------

class TestAwakeSince:
    def test_an_unbroken_hourly_streak_reaches_back_to_its_first_run(self):
        assert health.awake_since(hourly_runs(5), NOW) == hours_ago(5)

    def test_a_gap_longer_than_the_streak_limit_restarts_the_streak(self):
        runs = hourly_runs(30, 20) + hourly_runs(3)   # asleep for 17 hours between
        assert health.awake_since(runs, NOW) == hours_ago(3)

    def test_no_runs_means_awake_only_now(self):
        assert health.awake_since([], NOW) == NOW


class TestFreshness:
    def test_a_recent_snapshot_is_fine(self):
        check = health.freshness("BTC", hours_ago(0.5), NOW, hours_ago(10))
        assert check.level == "OK"

    def test_replay_dead_collector_while_machine_is_awake_fails(self):
        # Scheduled tasks demonstrably ran for five hours; nothing collected.
        check = health.freshness("BTC", hours_ago(4), NOW, hours_ago(5))
        assert check.level == "FAIL"

    def test_a_stale_snapshot_just_after_waking_is_not_an_alarm(self):
        # The 24-hour hole of 11-12 September: asleep, then woken 40 min ago.
        check = health.freshness("BTC", hours_ago(24), NOW, hours_ago(0.66))
        assert check.level == "OK"
        assert "asleep" in check.detail

    def test_the_same_staleness_fails_once_awake_long_enough(self):
        check = health.freshness("BTC", hours_ago(24), NOW, hours_ago(2.5))
        assert check.level == "FAIL"

    def test_without_watchdog_history_staleness_can_only_warn(self):
        check = health.freshness("BTC", hours_ago(10), NOW, None)
        assert check.level == "WARN"

    def test_no_snapshots_at_all_fails(self):
        assert health.freshness("BTC", None, NOW, hours_ago(3)).level == "FAIL"


# --------------------------------------------------------------------------
# The scheduler: settings that report success while doing nothing
# --------------------------------------------------------------------------

def task(**overrides) -> dict:
    info = {"state": "Ready", "lastResult": 0, "lastRun": "", "batteryBlocked": False,
            "instances": "IgnoreNew", "repeat": "PT1H"}
    info.update(overrides)
    return info


class TestTaskCheck:
    def test_a_well_configured_task_is_fine(self):
        assert health.task_check("x", task(), "collector", required=True).level == "OK"

    def test_replay_battery_blocked_task_fails(self):
        # First install: Last Result 0, collecting nothing, parked on battery.
        check = health.task_check("x", task(batteryBlocked=True), "collector", required=True)
        assert check.level == "FAIL"

    def test_replay_one_time_archive_task_fails(self):
        # The registration that sat idle for three days after one reset.
        check = health.task_check("x", task(repeat=""), "archive", required=True)
        assert check.level == "FAIL"
        assert "one-time" in check.detail

    def test_overlapping_runs_fail(self):
        # Two fetches once wrote the same parquet paths at the same time.
        check = health.task_check("x", task(instances="Parallel"), "archive", required=True)
        assert check.level == "FAIL"

    def test_a_missing_task_fails_only_when_it_is_needed(self):
        assert health.task_check("x", None, "archive", required=True).level == "FAIL"
        assert health.task_check("x", None, "archive", required=False).level == "OK"

    def test_a_running_task_is_not_a_failure(self):
        check = health.task_check("x", task(state="Running", lastResult=267009), "a", True)
        assert check.level == "OK"

    def test_replay_scheduler_refusing_an_overlapping_trigger_is_not_a_warning(self):
        # 0x800710E0 raised three false warnings between 14 and 17 September:
        # IgnoreNew declining a trigger while the previous run was still going.
        check = health.task_check("x", task(state="Running", lastResult=2147946720), "a", True)
        assert check.level == "OK"


# --------------------------------------------------------------------------
# The logs
# --------------------------------------------------------------------------

class TestCollectorLog:
    def test_replay_one_failed_run_on_wake_only_warns(self):
        # 18 and 19 September: the catch-up run fired before Wi-Fi was back,
        # every book failed once, and the next hour was fine.
        lines = ["2026-09-19 06:46:02Z    BTC  ok      978 instruments",
                 "2026-09-19 17:58:20Z    BTC  FAIL  RuntimeError: failed after 3 attempts",
                 "2026-09-19 17:58:20Z    ETH  ok      892 instruments"]
        checks = {c.name: c.level for c in health.collector_failures(lines, NOW)}
        assert checks["BTC fetches"] == "WARN"
        assert checks["ETH fetches"] == "OK"

    def test_consecutive_failed_runs_fail(self):
        lines = ["2026-09-13 16:46:02Z    BTC  FAIL  RuntimeError: book summary failed",
                 "2026-09-13 17:46:02Z    BTC  FAIL  RuntimeError: book summary failed"]
        checks = {c.name: c.level for c in health.collector_failures(lines, NOW)}
        assert checks["BTC fetches"] == "FAIL"

    def test_note_lines_are_not_mistaken_for_attempts(self):
        lines = ["2026-09-13 17:46:02Z   note  waited 40s for the network (woke from sleep?)",
                 "2026-09-13 17:46:42Z    BTC  ok      938 instruments"]
        checks = {c.name: c.level for c in health.collector_failures(lines, NOW)}
        assert checks["BTC fetches"] == "OK"

    def test_a_failure_that_has_recovered_only_warns(self):
        lines = ["2026-09-13 15:46:02Z    BTC  FAIL  RuntimeError: transient",
                 "2026-09-13 16:46:02Z    BTC  ok      938 instruments"]
        checks = {c.name: c.level for c in health.collector_failures(lines, NOW)}
        assert checks["BTC fetches"] == "WARN"


ARCHIVE_CRASH = """\
--- start 2026-09-10 02:43:15Z, currencies BTC, ETH ---
2022-05-01  BTC   10442 rows  12 expiries
2022-05-01  ETH   10192 rows  12 expiries
--- FAILED ---
Traceback (most recent call last):
  File "fetch_archive.py", line 105, in run
urllib3.exceptions.ProtocolError: ("Connection broken: ConnectionResetError(10054)")
"""


def ended(start_hours_ago: float, fetched: int, missing: int = 26) -> str:
    start = hours_ago(start_hours_ago)
    end = start + dt.timedelta(minutes=11)
    return (f"--- start {start:%Y-%m-%d %H:%M:%SZ}, currencies BTC, ETH ---\n"
            f"--- end {end:%Y-%m-%d %H:%M:%SZ}, {fetched} day(s) fetched, 0 failed, "
            f"{missing} of 79 still missing ---\n")


class TestArchiveProgress:
    def test_replay_the_crash_that_idled_the_archive_for_three_days(self):
        runs = health.archive_runs(ARCHIVE_CRASH)
        checks = {c.name: c for c in health.archive_progress(runs, 52, NOW)}
        assert checks["last run"].level == "FAIL"
        assert "ProtocolError" in checks["last run"].detail

    def test_waiting_out_a_rate_limit_is_not_an_alarm(self):
        text = ended(20, fetched=25) + "".join(ended(h, 0) for h in range(19, 0, -1))
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 26, NOW)}
        assert checks == {"last run": "OK", "progress": "OK"}

    def test_a_day_and_more_of_runs_with_nothing_to_show_fails(self):
        text = ended(40, fetched=25) + "".join(
            ended(h, 0) for h in range(health.ARCHIVE_STUCK_RUNS + 5, 0, -1))
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 26, NOW)}
        assert checks["progress"] == "FAIL"

    def test_a_run_that_never_logged_an_end_warns_after_the_time_limit(self):
        text = f"--- start {hours_ago(9):%Y-%m-%d %H:%M:%SZ}, currencies BTC, ETH ---\n"
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 26, NOW)}
        assert checks["last run"] == "WARN"

    def test_replay_a_long_run_the_scheduler_says_is_alive_is_not_suspect(self):
        # 16-17 September: a healthy run stretched to 9.8 h by sleep was
        # reported as killed at the 9-hour mark.
        text = f"--- start {hours_ago(9):%Y-%m-%d %H:%M:%SZ}, currencies BTC, ETH ---\n"
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 3, NOW, task_running=True)}
        assert checks["last run"] == "OK"

    def test_a_long_open_run_with_nothing_running_still_warns(self):
        text = f"--- start {hours_ago(9):%Y-%m-%d %H:%M:%SZ}, currencies BTC, ETH ---\n"
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 3, NOW, task_running=False)}
        assert checks["last run"] == "WARN"

    def test_a_run_in_progress_is_fine(self):
        text = f"--- start {hours_ago(0.1):%Y-%m-%d %H:%M:%SZ}, currencies BTC, ETH ---\n"
        checks = {c.name: c.level for c in
                  health.archive_progress(health.archive_runs(text), 26, NOW)}
        assert checks["last run"] == "OK"

    def test_a_complete_archive_needs_nothing(self):
        checks = health.archive_progress(health.archive_runs(ARCHIVE_CRASH), 0, NOW)
        assert [c.level for c in checks] == ["OK"]


# --------------------------------------------------------------------------
# When to interrupt a person
# --------------------------------------------------------------------------

def C(level: str, name: str = "progress") -> "health.Check":
    return health.Check(level, "archive", name, "detail")


class TestDecideAlert:
    def test_a_new_failure_notifies(self):
        message, _ = health.decide_alert({}, [C("FAIL")], NOW)
        assert message and "FAIL" in message

    def test_a_known_failure_stays_quiet_until_the_realert_interval(self):
        state = {"problems": {"archive:progress": "FAIL"},
                 "last_alert": hours_ago(2).isoformat()}
        assert health.decide_alert(state, [C("FAIL")], NOW)[0] is None

    def test_a_persistent_failure_notifies_again_after_the_interval(self):
        state = {"problems": {"archive:progress": "FAIL"},
                 "last_alert": hours_ago(13).isoformat()}
        message, _ = health.decide_alert(state, [C("FAIL")], NOW)
        assert message and message.startswith("still failing")

    def test_a_warning_escalating_to_failure_notifies(self):
        state = {"problems": {"archive:progress": "WARN"}, "last_alert": hours_ago(1).isoformat()}
        assert health.decide_alert(state, [C("FAIL")], NOW)[0]

    def test_a_failure_easing_to_a_warning_is_quiet(self):
        state = {"problems": {"archive:progress": "FAIL"}, "last_alert": hours_ago(1).isoformat()}
        assert health.decide_alert(state, [C("WARN")], NOW)[0] is None

    def test_recovery_notifies_once(self):
        state = {"problems": {"archive:progress": "FAIL"}, "last_alert": hours_ago(1).isoformat()}
        message, record = health.decide_alert(state, [C("OK")], NOW)
        assert message == "all checks passing again"
        assert health.decide_alert(record, [C("OK")], NOW)[0] is None

    def test_all_quiet_stays_quiet(self):
        assert health.decide_alert({}, [C("OK")], NOW)[0] is None


# --------------------------------------------------------------------------
# A failed scheduler query
# --------------------------------------------------------------------------

class TestSchedulerQueryFailure:
    def test_replay_a_failed_query_does_not_report_tasks_missing(self, monkeypatch, tmp_path):
        # 15 September 20:46: the query failed just after a wake, and both
        # tasks were reported "not registered" -- a false FAIL notification.
        monkeypatch.setattr(health, "scheduled_tasks", lambda: {"_error": {}})
        monkeypatch.setattr(health, "collector_snapshots", lambda now, awake: [])
        monkeypatch.setattr(health, "archive_files", lambda now, verified: ([], 3, {}))
        monkeypatch.setattr(health, "COLLECTOR_LOG", tmp_path / "none.log")
        monkeypatch.setattr(health, "ARCHIVE_LOG", tmp_path / "none.log")
        checks, _ = health.gather({"runs": [hours_ago(1).isoformat()]}, NOW, record_run=True)
        assert not [c for c in checks if c.level == "FAIL"]
        assert not [c for c in checks if c.name == "task"]
        assert [c.name for c in checks if c.area == "system" and c.level == "WARN"] == ["scheduler"]
