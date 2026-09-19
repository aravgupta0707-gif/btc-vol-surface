"""
Download the Tardis free archive, resumably, as a detached background job.

    python scripts/fetch_archive.py start      register and run the task
    python scripts/fetch_archive.py status     progress, per currency
    python scripts/fetch_archive.py logs
    python scripts/fetch_archive.py stop       remove the task (keeps files)
    python scripts/fetch_archive.py run        run in the foreground instead

Why a scheduled task rather than a background process: the job takes hours,
and a process launched from a shell is a child of that shell. (An earlier
background process stopped at ~26 of 79 days with an empty stderr. That was
most likely not a crash but a deliberate `taskkill` from a second session,
which restarted the job to add ETH -- the design point stands regardless.)
The collector already proves a scheduled task survives, so the same
mechanism is used here.

The task repeats hourly and never runs two copies at once. No single run has
to finish the archive: a run that meets a rate limit or a dead connection
stops cleanly after backing off, and the next hour picks up where it left
off. A run with nothing left to fetch exits without a word, so leaving the
task registered after completion costs nothing; `stop` removes it.

Every run is resumable. Currency-days already on disk are skipped, files are
written atomically so a killed run cannot leave a truncated day that looks
finished, and an interruption costs only the day in flight.

Logging goes to a file because the task runs under pythonw, which has no
console to print to -- the same reason `scripts/collector.py` logs the way
it does.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "data" / "research" / "archive_fetch.log"
HIST_DIR = ROOT / "data" / "historical"
TASK_NAME = "BTC-Vol-Archive-Fetch"
PYTHONW = Path(sys.executable).with_name("pythonw.exe")
CURRENCIES = ("BTC", "ETH")


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True)


def _registered() -> bool:
    return _schtasks("/query", "/tn", TASK_NAME).returncode == 0


# Tardis rate-limits the free tier. The first attempt got 52 extractions
# through and was then refused for every remaining day, and the limit had
# cleared again within 22 hours -- so it is a rolling window on request
# count, not a hard quota. Pace requests and stop when refused.
DELAY_SECONDS = 20.0
BACKOFF_SECONDS = (60.0, 180.0, 420.0)


def _log(text: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _paths(day: dt.date) -> dict[str, Path]:
    return {c: HIST_DIR / f"{c.lower()}_chain_{day:%Y%m%d}.parquet"
            for c in CURRENCIES}


def _missing(days) -> list[dt.date]:
    return [d for d in days if not all(p.exists() for p in _paths(d).values())]


def classify(exc: BaseException) -> str:
    """
    'rate_limited', 'transient' or 'permanent': whether waiting could help.

    Permanent failures belong to the day -- a 404, or a file with no rows for
    these currencies -- so the run records them and moves on. The other two
    belong to the moment, and moving on would only fail the next day the
    same way: that is how one 429 burned 53 days, and how one connection
    reset ended a run and left the archive idle for three days.
    """
    import requests

    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else None
        if code == 429:
            return "rate_limited"
        return "transient" if code is not None and code >= 500 else "permanent"
    # Connection resets, timeouts and truncated gzip bodies all land here;
    # so does a full disk, where stopping is also the right response.
    if isinstance(exc, (requests.RequestException, OSError)):
        return "transient"
    return "permanent"


def run() -> int:
    """
    Do the download. This is what the task executes.

    The loop lives here rather than using `fetch_free_archive` because of how
    temporary failures have to be handled. That function logs a skip and
    moves on, which is right for a day that is genuinely broken but wrong for
    a rate limit or a dropped connection: the very next request fails too, so
    one 429 burned all 53 remaining days in a few seconds and left the queue
    looking attempted. Here a temporary failure backs off, retries, and if it
    persists **stops the run** -- leaving every untouched day genuinely
    untouched for the next hourly pass.
    """
    import time

    import requests

    from data.tardis import free_days, load_day_multi, write_parquet_atomic

    days = free_days()
    if not _missing(days):
        return 0  # complete: the hourly task stays silent rather than log noise

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    _log(f"--- start {stamp}, currencies {', '.join(CURRENCIES)} ---")
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    try:
        for day in days:
            paths = _paths(day)
            todo = [c for c in CURRENCIES if not paths[c].exists()]
            if not todo:
                continue

            error: BaseException | None = None
            for attempt, wait in enumerate((0.0, *BACKOFF_SECONDS)):
                if wait:
                    reason = ("rate limited" if classify(error) == "rate_limited"
                              else "connection problem")
                    _log(f"{day}  {reason}, waiting {wait:.0f}s "
                         f"(attempt {attempt}): {error}")
                    time.sleep(wait)
                try:
                    frames = load_day_multi(day, 60, todo)
                    for currency, frame in frames.items():
                        write_parquet_atomic(frame, paths[currency])
                        _log(f"{day}  {currency:4s} {len(frame):6d} rows  "
                             f"{frame['expiry_ts'].nunique():2d} expiries")
                    for currency in set(todo) - set(frames):
                        _log(f"{day}  {currency:4s} no rows in this day's file")
                    done += 1
                    break
                except (requests.RequestException, RuntimeError, OSError) as exc:
                    if classify(exc) == "permanent":
                        _log(f"{day}  FAILED: {exc}")
                        failed += 1
                        break
                    error = exc
            else:
                _log(f"{day}  still failing after {len(BACKOFF_SECONDS)} "
                     f"retries ({error}) -- stopping so the remaining days "
                     f"stay untried for the next run")
                break

            time.sleep(DELAY_SECONDS)

        # "end", not "done": a run that stopped early must not read as a
        # finished archive, so the line always carries what is still missing.
        _log(f"--- end {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ}, "
             f"{done} day(s) fetched, {failed} failed, "
             f"{len(_missing(days))} of {len(days)} still missing ---")
        return 0
    except Exception:  # noqa: BLE001 - the log is the only error channel here
        _log("--- FAILED ---")
        _log(traceback.format_exc())
        return 1


def start() -> int:
    if _registered():
        print(f"'{TASK_NAME}' is already registered. Use 'status'.")
        return 0
    command = f'"{PYTHONW}" "{ROOT / "scripts" / "fetch_archive.py"}" run'
    # Hourly, not once. A one-time task turned every clean stop -- a rate
    # limit, a dropped connection -- into a permanent one: the first
    # registration sat idle for three days after a single reset.
    result = _schtasks("/create", "/tn", TASK_NAME, "/tr", command,
                       "/sc", "hourly", "/mo", "1", "/f")
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return result.returncode

    # IgnoreNew matters as much as the battery flags: a day can take minutes
    # and a run can back off for eleven, so without it the next hour would
    # start a second copy writing the same files.
    settings = (
        "New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -StartWhenAvailable "
        "-MultipleInstances IgnoreNew "
        "-ExecutionTimeLimit (New-TimeSpan -Hours 8)"
    )
    applied = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"Set-ScheduledTask -TaskName '{TASK_NAME}' -Settings ({settings})"],
        capture_output=True, text=True,
    )
    if applied.returncode != 0:
        # The schtasks defaults park the task on battery while it reports
        # success, so a registration without these settings is not a
        # registration worth keeping.
        _schtasks("/delete", "/tn", TASK_NAME, "/f")
        print("Could not apply task settings; task removed.\n"
              + (applied.stderr or applied.stdout))
        return applied.returncode or 1
    _schtasks("/run", "/tn", TASK_NAME)
    print(f"Registered '{TASK_NAME}' hourly and started it (8 hour limit per run).")
    print("  python scripts/fetch_archive.py status")
    print("  python scripts/fetch_archive.py stop      <- removes the task")
    return 0


def stop() -> int:
    if not _registered():
        print(f"'{TASK_NAME}' is not registered.")
        return 0
    result = _schtasks("/delete", "/tn", TASK_NAME, "/f")
    print(f"Removed '{TASK_NAME}'. Downloaded days are untouched."
          if result.returncode == 0 else (result.stdout or result.stderr))
    return result.returncode


def status() -> int:
    from data.tardis import free_days

    print(f"task     {TASK_NAME}: "
          f"{'registered' if _registered() else 'not registered'}")
    if _registered():
        info = _schtasks("/query", "/tn", TASK_NAME, "/fo", "LIST", "/v").stdout
        for line in info.splitlines():
            for key in ("Status:", "Last Run Time:", "Last Result:"):
                if line.strip().startswith(key):
                    print(f"         {line.strip()}")

    days = free_days()
    print(f"\nprogress  {len(days)} free days available "
          f"({days[0]} .. {days[-1]})")
    for currency in CURRENCIES:
        have = sorted(HIST_DIR.glob(f"{currency.lower()}_chain_*.parquet")) \
            if HIST_DIR.exists() else []
        size = sum(p.stat().st_size for p in have) / 1e6
        print(f"  {currency:>5}  {len(have):3d}/{len(days)} days  {size:6.1f} MB"
              + (f"  {have[0].stem[-8:]} .. {have[-1].stem[-8:]}" if have else ""))
    missing = _missing(days)
    print(f"\n  {len(missing)} day(s) still missing"
          + (f", next {missing[0]}" if missing else
             " -- archive complete; 'stop' removes the hourly task"))
    if LOG_PATH.exists():
        print(f"\nlog      {LOG_PATH}")
    return 0


def logs(lines: int = 15) -> int:
    if not LOG_PATH.exists():
        print("no log yet")
        return 0
    print("\n".join(LOG_PATH.read_text(encoding="utf-8").splitlines()[-lines:]))
    return 0


COMMANDS = {"start": start, "stop": stop, "status": status, "logs": logs,
            "run": run}


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command not in COMMANDS:
        print(f"unknown command '{command}'; expected one of "
              + ", ".join(COMMANDS))
        return 2
    return COMMANDS[command]()


if __name__ == "__main__":
    raise SystemExit(main())
