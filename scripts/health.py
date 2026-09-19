"""
Watchdog for everything that runs unattended: the collector and the archive.

    python scripts/health.py              check now; exit 1 if anything FAILs
    python scripts/health.py install      register the hourly watchdog task
    python scripts/health.py uninstall    remove it
    python scripts/health.py logs         recent verdicts, newest last
    python scripts/health.py notify-test  show a test notification
    python scripts/health.py run          what the task executes

Why this exists. Every serious failure in this project reported itself as a
success. Task Scheduler said `Last Result: 0` while collecting nothing. The
archive printed `--- done ---` with 53 of 79 days skipped. A connection reset
ended a run, the task sat in `Ready`, and the archive was idle for three days
before anyone looked. Each fix closed one hole; none would have caught the
next one. This checks outcomes rather than exit codes -- files on disk, how
fresh they are, whether they read back, whether runs make progress -- and
raises a Windows notification when a check changes state, so a stall costs
hours rather than days.

**Asleep versus broken.** A laptop that sleeps produces exactly the same
stale files as a dead collector. Windows' power events cannot separate them
here: on this machine scheduled tasks kept running through some modern
standby periods and not others. So the watchdog uses its own run history
instead. It runs hourly under the same scheduler conditions as the collector;
if it has itself run continuously for two hours and the collector has written
nothing in that time, the scheduler was demonstrably able to run tasks and
the collector is broken. When the machine sleeps, the watchdog's streak
breaks too, and no alarm is raised for a gap nobody could have filled.

Checks are PASS/WARN/FAIL like the `verify_*` scripts. A FAIL notifies when
it first appears and again every 12 hours while it persists; a new WARN
notifies once; recovery notifies once.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
HIST_DIR = ROOT / "data" / "historical"
RESEARCH_DIR = ROOT / "data" / "research"
COLLECTOR_LOG = RAW_DIR / "collector.log"
ARCHIVE_LOG = RESEARCH_DIR / "archive_fetch.log"
STATE_PATH = RESEARCH_DIR / "health_state.json"
HEALTH_LOG = RESEARCH_DIR / "health.log"

TASK_NAME = "BTC-Vol-Health-Check"
COLLECTOR_TASK = "BTC-Vol-Surface-Collector"
ARCHIVE_TASK = "BTC-Vol-Archive-Fetch"
PYTHONW = Path(sys.executable).with_name("pythonw.exe")
COLLECT = ("BTC", "ETH", "USDC")
ARCHIVE_CURRENCIES = ("BTC", "ETH")

# Freshness. The collector runs at :46 and this at :16, so a healthy
# snapshot is at most ~75 minutes old whenever this looks.
STREAK_GAP = dt.timedelta(minutes=75)     # longer between runs = machine was down
STALE_AFTER = dt.timedelta(minutes=90)
AWAKE_NEEDED = dt.timedelta(hours=2)
UNWATCHED_STALE = dt.timedelta(hours=3)   # no history: can only warn

# The archive's rate limit clears within ~22 hours, and a limited run stops
# cleanly and logs zero days. Thirty consecutive empty runs is more than a
# full window of hourly attempts with nothing to show.
ARCHIVE_STUCK_RUNS = 30
ARCHIVE_RUN_LIMIT = dt.timedelta(hours=8)  # the task's own execution limit

# Calibrated on the data itself: archive days hold 23-24 hourly buckets,
# and snapshot row counts sit within a few percent of their median.
MIN_BUCKETS = 20
MIN_ARCHIVE_ROWS = 1000
ROW_FRACTION = 0.5
TMP_AGE = dt.timedelta(hours=8)
DISK_WARN_GB, DISK_FAIL_GB = 2.0, 0.5

REALERT = dt.timedelta(hours=12)
HISTORY_KEEP = dt.timedelta(hours=48)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# ok, currently running, never run -- and 0x800710E0, "the operator or
# administrator has refused the request", which is the scheduler declining an
# hourly trigger because the previous run is still going. That is IgnoreNew
# working as intended, not a failure: it raised three false warnings in the
# first four days. A run that genuinely never happens still shows up in the
# freshness and progress checks, which look at outcomes rather than codes.
_REFUSED_OVERLAP = 0x800710E0
_RUNNING_OR_NEW = {0, 267009, 267011, _REFUSED_OVERLAP}


@dataclass
class Check:
    level: str   # OK / WARN / FAIL
    area: str
    name: str
    detail: str

    @property
    def key(self) -> str:
        return f"{self.area}:{self.name}"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _ago(delta: dt.timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours * 60:.0f} min" if hours < 2 else f"{hours:.1f} h"


# --------------------------------------------------------------------------
# Pure decisions -- no I/O, so every rule is tested directly
# --------------------------------------------------------------------------

def awake_since(runs: list[dt.datetime], now: dt.datetime) -> dt.datetime:
    """
    Start of the unbroken hourly streak of watchdog runs that reaches `now`.

    A gap longer than STREAK_GAP means the scheduler was not running tasks --
    the machine was asleep or off -- so the streak restarts after it.
    """
    since = now
    for t in sorted((r for r in runs if r <= now), reverse=True):
        if since - t > STREAK_GAP:
            break
        since = t
    return since


def freshness(currency: str, last: dt.datetime | None, now: dt.datetime,
              awake_from: dt.datetime | None) -> Check:
    """
    `awake_from` is None when there is no watchdog history to reason from;
    then staleness can only ever be a warning.
    """
    if last is None:
        return Check("FAIL", "collector", f"{currency} freshness", "no snapshots on disk")
    age = now - last
    detail = f"last snapshot {_ago(age)} ago"
    if age <= STALE_AFTER:
        return Check("OK", "collector", f"{currency} freshness", detail)
    if awake_from is None:
        level = "WARN" if age > UNWATCHED_STALE else "OK"
        return Check(level, "collector", f"{currency} freshness",
                     detail + " (no watchdog history: cannot tell sleep from failure)")
    if now - awake_from >= AWAKE_NEEDED and last < now - STALE_AFTER:
        return Check("FAIL", "collector", f"{currency} freshness",
                     detail + f", although scheduled tasks have run for {_ago(now - awake_from)}")
    return Check("OK", "collector", f"{currency} freshness",
                 detail + " (machine was asleep; catching up)")


_COLLECTOR_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ)\s+(\w+)\s+(ok|FAIL)\b\s*(.*)$")


def collector_failures(lines: list[str], now: dt.datetime,
                       window: dt.timedelta = dt.timedelta(hours=24)) -> list[Check]:
    """
    One check per currency. Two or more consecutive failed runs is a FAIL:
    that book is not being collected. A single failed latest run is a WARN
    -- it is most often the catch-up run on wake, before the network is
    back, and it clears by itself at the next hour; calling it a FAIL sent a
    failure notification for something nobody could act on. A dead collector
    that stops running altogether is caught by the freshness check instead.
    Failures earlier in the window that have since recovered are a WARN.
    """
    latest: dict[str, tuple[str, str]] = {}
    streak: dict[str, int] = {}
    recent_fails: dict[str, int] = {}
    for line in lines:
        m = _COLLECTOR_LINE.match(line.strip())
        if not m:
            continue
        stamp = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
        currency, outcome, rest = m.group(2), m.group(3), m.group(4)
        latest[currency] = (outcome, rest)
        streak[currency] = streak.get(currency, 0) + 1 if outcome == "FAIL" else 0
        if outcome == "FAIL" and now - stamp <= window:
            recent_fails[currency] = recent_fails.get(currency, 0) + 1
    checks = []
    for currency in COLLECT:
        name = f"{currency} fetches"
        if currency not in latest:
            checks.append(Check("WARN", "collector", name, "no log entries yet"))
        elif streak[currency] >= 2:
            checks.append(Check("FAIL", "collector", name,
                                f"last {streak[currency]} runs failed: {latest[currency][1][:120]}"))
        elif latest[currency][0] == "FAIL":
            checks.append(Check("WARN", "collector", name,
                                f"latest run failed, once: {latest[currency][1][:120]}"))
        elif recent_fails.get(currency):
            checks.append(Check("WARN", "collector", name,
                                f"{recent_fails[currency]} failed attempt(s) in 24 h, since recovered"))
        else:
            checks.append(Check("OK", "collector", name, "no failures in 24 h"))
    return checks


_END = re.compile(r"^--- end (\S+ \S+Z), (\d+) day\(s\) fetched, (\d+) failed, "
                  r"(\d+) of (\d+) still missing ---")
_START = re.compile(r"^--- start (\S+ \S+Z)")


def archive_runs(text: str) -> list[dict]:
    """
    Split the archive log into runs. Each is a dict with `start`, `fetched`,
    `missing`, and exactly one of `ended` / `crashed` / `open` (still running,
    or killed before it could log an end). Runs from before the current log
    format carry no counts and are ignored.
    """
    runs: list[dict] = []
    current: dict | None = None
    last_line = ""
    for line in text.splitlines():
        m = _START.match(line)
        if m:
            if current:
                runs.append(current)   # a start with no end: killed
            current = {"start": dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%SZ")
                       .replace(tzinfo=dt.timezone.utc),
                       "fetched": 0, "missing": None, "state": "open", "error": ""}
            continue
        if current is None:
            continue
        m = _END.match(line)
        if m:
            current.update(fetched=int(m.group(2)), missing=int(m.group(4)), state="ended")
            runs.append(current)
            current = None
        elif line.startswith("--- FAILED ---"):
            current["state"] = "crashed"
        elif current["state"] == "crashed" and line.strip():
            last_line = line.strip()
            current["error"] = last_line
    if current:
        runs.append(current)
    # A crashed run stays "current" to collect its traceback; close it here.
    return runs


def archive_progress(runs: list[dict], missing: int, now: dt.datetime,
                     task_running: bool | None = None) -> list[Check]:
    """
    `task_running` is the scheduler's own word on whether a fetch is alive
    (None if it could not be asked). A run with no end line is only suspect
    when nothing is running: sleep stretches wall-clock time, and on 17
    September a healthy run took 9.8 hours and was wrongly reported killed.
    """
    if missing == 0:
        return [Check("OK", "archive", "progress", "archive complete")]
    if not runs:
        return [Check("WARN", "archive", "progress", f"{missing} days missing and no runs logged")]
    checks = []
    last = runs[-1]
    if last["state"] == "crashed":
        checks.append(Check("FAIL", "archive", "last run",
                            f"crashed: {last['error'][:140] or 'see archive_fetch.log'}"))
    elif last["state"] == "open" and task_running:
        checks.append(Check("OK", "archive", "last run",
                            f"running, started {_ago(now - last['start'])} ago"))
    elif last["state"] == "open" and now - last["start"] > ARCHIVE_RUN_LIMIT:
        checks.append(Check("WARN", "archive", "last run",
                            f"started {_ago(now - last['start'])} ago and never logged an end "
                            "(killed by the time limit or a power loss)"))
    elif last["state"] == "open":
        checks.append(Check("OK", "archive", "last run",
                            f"running, started {_ago(now - last['start'])} ago"))
    else:
        checks.append(Check("OK", "archive", "last run",
                            f"ended normally, {last['fetched']} day(s) fetched"))

    empty = 0
    for run in reversed([r for r in runs if r["state"] != "open"]):
        if run["fetched"]:
            break
        empty += 1
    level = "FAIL" if empty >= ARCHIVE_STUCK_RUNS else "OK"
    checks.append(Check(level, "archive", "progress",
                        f"{missing} days missing; {empty} consecutive run(s) without a new day"
                        + (" -- stuck" if level == "FAIL" else "")))
    return checks


def decide_alert(previous: dict, checks: list[Check], now: dt.datetime) -> tuple[str | None, dict]:
    """
    Returns (message or None, the problem record to store).

    Notifies on a new FAIL or WARN, on a FAIL that has persisted REALERT
    since the last notification, and once on recovery. Everything else is
    silent -- an alarm that fires every hour gets ignored.
    """
    problems = {c.key: c.level for c in checks if c.level != "OK"}
    before = previous.get("problems", {})
    last_alert = previous.get("last_alert")
    last_alert = dt.datetime.fromisoformat(last_alert) if last_alert else None

    fresh = [c for c in checks if c.level != "OK" and
             (before.get(c.key) != c.level)]
    # Only an escalation or a new problem counts as fresh; FAIL -> WARN is not.
    fresh = [c for c in fresh if not (before.get(c.key) == "FAIL" and c.level == "WARN")]
    fails = [c for c in checks if c.level == "FAIL"]

    message = None
    if fresh:
        worst = sorted(fresh, key=lambda c: c.level != "FAIL")
        message = "; ".join(f"{c.level} {c.area} {c.name}: {c.detail}" for c in worst)
    elif fails and (last_alert is None or now - last_alert >= REALERT):
        message = "still failing -- " + "; ".join(f"{c.area} {c.name}: {c.detail}" for c in fails)
    elif before and not problems:
        message = "all checks passing again"

    record = {"problems": problems,
              "last_alert": now.isoformat() if message else previous.get("last_alert")}
    return message, record


# --------------------------------------------------------------------------
# Gathering -- scheduler, files, disk
# --------------------------------------------------------------------------

def _powershell(script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW,
    )


def scheduled_tasks() -> dict[str, dict]:
    """State and settings for our tasks, straight from the task objects."""
    names = ",".join(f"'{n}'" for n in (COLLECTOR_TASK, ARCHIVE_TASK, TASK_NAME))
    script = f"""
$out = @{{}}
foreach ($n in @({names})) {{
  $t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
  if ($t) {{
    $i = $t | Get-ScheduledTaskInfo
    $out[$n] = @{{
      state = $t.State.ToString()
      lastResult = [int64]$i.LastTaskResult
      lastRun = if ($i.LastRunTime) {{ $i.LastRunTime.ToUniversalTime().ToString('o') }} else {{ '' }}
      batteryBlocked = [bool]$t.Settings.DisallowStartIfOnBatteries
      instances = $t.Settings.MultipleInstances.ToString()
      repeat = (@($t.Triggers | ForEach-Object {{ $_.Repetition.Interval }}) -join ',')
    }}
  }}
}}
$out | ConvertTo-Json -Compress
"""
    # Retried once: the query can time out in the minute after a wake, and a
    # failed query must never be read as "no tasks registered" -- that
    # produced a false FAIL notification on 15 September.
    for _ in range(2):
        try:
            result = _powershell(script)
            if result.returncode == 0:
                return json.loads(result.stdout or "{}")
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            pass
    return {"_error": {}}


def task_check(name: str, info: dict | None, area: str, required: bool,
               hourly: bool = True) -> Check:
    if info is None:
        return Check("FAIL" if required else "OK", area, "task",
                     "not registered" if required else "not registered (not needed)")
    problems = []
    if info["state"] == "Disabled":
        problems.append("disabled")
    if info["batteryBlocked"]:
        problems.append("blocked on battery -- will sit Queued and report success")
    if hourly and "PT1H" not in info["repeat"]:
        problems.append(f"not repeating hourly ({info['repeat'] or 'one-time'})")
    if info["instances"] != "IgnoreNew":
        problems.append(f"MultipleInstances={info['instances']} -- runs can overlap")
    if problems:
        return Check("FAIL", area, "task", "; ".join(problems))
    detail = f"{info['state'].lower()}, last result {info['lastResult']}"
    if info["state"] == "Queued":
        return Check("WARN", area, "task", detail + " -- queued, check power conditions")
    if info["lastResult"] not in _RUNNING_OR_NEW:
        return Check("WARN", area, "task", detail + " -- last run exited non-zero")
    return Check("OK", area, "task", detail)


def _stamp(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.stem.split("_")[-1], "%Y%m%dT%H%M%SZ").replace(
        tzinfo=dt.timezone.utc)


def collector_snapshots(now: dt.datetime, awake_from: dt.datetime | None) -> list[Check]:
    import pyarrow.parquet as pq

    checks = []
    for currency in COLLECT:
        files = sorted(RAW_DIR.glob(f"{currency.lower()}_chain_*.parquet"))
        last = _stamp(files[-1]) if files else None
        checks.append(freshness(currency, last, now, awake_from))
        if not files:
            continue
        # Footers only: cheap, and an unreadable footer is exactly what a
        # write killed half way leaves behind.
        try:
            newest = pq.read_metadata(files[-1]).num_rows
            recent = sorted(pq.read_metadata(f).num_rows for f in files[-25:-1])
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("FAIL", "collector", f"{currency} snapshot",
                                f"{files[-1].name} unreadable: {exc}"))
            continue
        median = recent[len(recent) // 2] if recent else newest
        if newest < ROW_FRACTION * median:
            checks.append(Check("FAIL", "collector", f"{currency} snapshot",
                                f"{files[-1].name} has {newest} rows against a median of {median}"))
        else:
            checks.append(Check("OK", "collector", f"{currency} snapshot",
                                f"{newest} rows (median {median})"))
    return checks


def archive_files(now: dt.datetime, verified: dict) -> tuple[list[Check], int, dict]:
    """Integrity of every archive day, re-reading only files that changed."""
    import pyarrow.parquet as pq

    from data.tardis import free_days

    days = free_days()
    have = {p.name for p in HIST_DIR.glob("*.parquet")} if HIST_DIR.exists() else set()
    missing = sum(
        1 for d in days
        if not all(f"{c.lower()}_chain_{d:%Y%m%d}.parquet" in have for c in ARCHIVE_CURRENCIES)
    )

    bad, still_verified = [], {}
    for path in sorted(HIST_DIR.glob("*.parquet")) if HIST_DIR.exists() else []:
        stat = path.stat()
        signature = [stat.st_size, int(stat.st_mtime)]
        if verified.get(path.name) == signature:
            still_verified[path.name] = signature
            continue
        try:
            table = pq.read_table(path, columns=["timestamp"])
            buckets = len(set(table.column("timestamp").to_pylist()))
            if table.num_rows < MIN_ARCHIVE_ROWS or buckets < MIN_BUCKETS:
                bad.append(f"{path.name} ({table.num_rows} rows, {buckets} buckets)")
                continue
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path.name} unreadable ({type(exc).__name__})")
            continue
        still_verified[path.name] = signature

    checks = [Check("FAIL" if bad else "OK", "archive", "files",
                    ("incomplete or unreadable: " + ", ".join(bad[:5])) if bad
                    else f"{len(still_verified)} files verified")]

    stale_tmp = [p.name for p in HIST_DIR.glob("*.tmp")
                 if now - dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) > TMP_AGE] \
        if HIST_DIR.exists() else []
    if stale_tmp:
        checks.append(Check("WARN", "archive", "temp files",
                            f"left by a killed write (safe to delete): {', '.join(stale_tmp[:5])}"))
    return checks, missing, still_verified


def disk_check() -> Check:
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    level = "FAIL" if free_gb < DISK_FAIL_GB else "WARN" if free_gb < DISK_WARN_GB else "OK"
    return Check(level, "system", "disk", f"{free_gb:.1f} GB free")


# --------------------------------------------------------------------------
# State, notification, commands
# --------------------------------------------------------------------------

def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def notify(title: str, body: str) -> bool:
    """A Windows toast, attributed to PowerShell so no app registration is needed."""
    def ps_quote(text: str) -> str:
        return escape(text).replace("'", "''")

    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast scenario="reminder"><visual><binding template="ToastGeneric"><text>{ps_quote(title)}</text><text>{ps_quote(body[:600])}</text></binding></visual><actions><action content="Dismiss" arguments="dismiss" activationType="system"/></actions></toast>')
$app = '{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\\WindowsPowerShell\\v1.0\\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""
    try:
        return _powershell(script, timeout=30).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def gather(state: dict, now: dt.datetime, record_run: bool) -> tuple[list[Check], dict]:
    runs = [dt.datetime.fromisoformat(t) for t in state.get("runs", [])]
    runs = [r for r in runs if now - r <= HISTORY_KEEP]
    tasks = scheduled_tasks()
    queried = "_error" not in tasks
    # When this is the hourly task, it is plainly installed whatever the
    # query said.
    watched = (record_run or TASK_NAME in tasks) and bool(runs)
    awake_from = awake_since(runs + [now], now) if watched else None

    checks: list[Check] = []
    # A failed query knows nothing about the tasks, so it must not report
    # them missing: one WARN below stands in for all of them.
    if queried:
        checks.append(task_check(COLLECTOR_TASK, tasks.get(COLLECTOR_TASK), "collector", required=True))
    checks += collector_snapshots(now, awake_from)
    log_lines = COLLECTOR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-500:] \
        if COLLECTOR_LOG.exists() else []
    checks += collector_failures(log_lines, now)

    file_checks, missing, verified = archive_files(now, state.get("verified_archive", {}))
    if queried:
        checks.append(task_check(ARCHIVE_TASK, tasks.get(ARCHIVE_TASK), "archive", required=missing > 0))
    archive_text = ARCHIVE_LOG.read_text(encoding="utf-8", errors="replace") \
        if ARCHIVE_LOG.exists() else ""
    running = (tasks.get(ARCHIVE_TASK) or {}).get("state") == "Running" if queried else None
    checks += archive_progress(archive_runs(archive_text), missing, now, running)
    checks += file_checks

    checks.append(disk_check())
    if not queried:
        checks.append(Check("WARN", "system", "scheduler",
                            "could not query scheduled tasks; task checks skipped this run"))
    if not record_run and queried:
        info = tasks.get(TASK_NAME)
        if info is None:
            checks.append(Check("WARN", "system", "watchdog",
                                "not installed -- nothing is watching between manual checks"))
        else:
            checks.append(task_check(TASK_NAME, info, "system", required=True))

    new_state = dict(state)
    new_state["verified_archive"] = verified
    new_state["runs"] = [r.isoformat() for r in (runs + [now] if record_run else runs)]
    return checks, new_state


def report(checks: list[Check]) -> None:
    for c in checks:
        print(f"[{c.level:^4}] {c.area:<9} {c.name:<16} {c.detail}")
    worst = "FAIL" if any(c.level == "FAIL" for c in checks) else \
        "WARN" if any(c.level == "WARN" for c in checks) else "OK"
    print(f"\n{worst}: {sum(c.level == 'FAIL' for c in checks)} fail, "
          f"{sum(c.level == 'WARN' for c in checks)} warn, {len(checks)} checks")


def check() -> int:
    checks, _ = gather(load_state(), _now(), record_run=False)
    report(checks)
    return 1 if any(c.level == "FAIL" for c in checks) else 0


def run() -> int:
    """The hourly task: check, record, notify on change. Never raises."""
    import traceback

    now = _now()
    state = load_state()
    try:
        checks, state = gather(state, now, record_run=True)
        message, record = decide_alert(state, checks, now)
        state.update(record)
        if message:
            ok = notify("BTC vol surface: " + ("recovered" if "passing again" in message
                                              else "needs attention"), message)
            if not ok:
                message += " [notification could not be shown]"
        fails = [c.key for c in checks if c.level == "FAIL"]
        warns = [c.key for c in checks if c.level == "WARN"]
        line = (f"{now:%Y-%m-%d %H:%M:%SZ}  "
                f"{'FAIL' if fails else 'WARN' if warns else 'OK  '}  "
                f"{len(fails)} fail {len(warns)} warn"
                + (f"  [{', '.join(fails + warns)}]" if fails or warns else "")
                + (f"  NOTIFIED: {message}" if message else ""))
        code = 1 if fails else 0
    except Exception:  # noqa: BLE001 - the watchdog must always leave a trace
        line = f"{now:%Y-%m-%d %H:%M:%SZ}  ERROR  watchdog crashed: " + \
            traceback.format_exc().strip().splitlines()[-1]
        notify("BTC vol surface: watchdog crashed", line)
        state.setdefault("runs", []).append(now.isoformat())
        code = 2
    save_state(state)
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HEALTH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return code


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True,
                          creationflags=_NO_WINDOW)


def install() -> int:
    if _schtasks("/query", "/tn", TASK_NAME).returncode == 0:
        print(f"'{TASK_NAME}' is already registered.")
        return 0
    command = f'"{PYTHONW}" "{ROOT / "scripts" / "health.py"}" run'
    # :16 sits half an hour away from the collector's :46, so each hour's
    # check sees that hour's snapshot rather than racing it.
    result = _schtasks("/create", "/tn", TASK_NAME, "/tr", command,
                       "/sc", "hourly", "/mo", "1", "/st", "00:16", "/f")
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return result.returncode
    applied = _powershell(
        f"Set-ScheduledTask -TaskName '{TASK_NAME}' -Settings (New-ScheduledTaskSettingsSet "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable "
        "-MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10))"
    )
    if applied.returncode != 0:
        _schtasks("/delete", "/tn", TASK_NAME, "/f")
        print("Could not apply task settings; task removed.\n" + (applied.stderr or applied.stdout))
        return applied.returncode or 1
    _schtasks("/run", "/tn", TASK_NAME)
    print(f"Registered '{TASK_NAME}', hourly at :16, and ran it once.")
    print("  python scripts/health.py        full report")
    print("  python scripts/health.py logs   one line per hourly verdict")
    return 0


def uninstall() -> int:
    result = _schtasks("/delete", "/tn", TASK_NAME, "/f")
    print(f"Removed '{TASK_NAME}'." if result.returncode == 0 else (result.stdout or result.stderr))
    return result.returncode


def logs(lines: int = 24) -> int:
    if not HEALTH_LOG.exists():
        print("no verdicts yet")
        return 0
    print("\n".join(HEALTH_LOG.read_text(encoding="utf-8").splitlines()[-lines:]))
    return 0


def notify_test() -> int:
    ok = notify("BTC vol surface: test", "If you can read this, the watchdog can reach you.")
    print("notification sent" if ok else "notification FAILED to show")
    return 0 if ok else 1


COMMANDS = {"check": check, "run": run, "install": install, "uninstall": uninstall,
            "logs": logs, "notify-test": notify_test}


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command not in COMMANDS:
        print(f"unknown command '{command}'; expected one of " + ", ".join(COMMANDS))
        return 2
    return COMMANDS[command]()


if __name__ == "__main__":
    raise SystemExit(main())
