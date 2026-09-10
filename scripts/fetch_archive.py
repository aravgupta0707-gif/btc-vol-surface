"""
Download the Tardis free archive, resumably, as a detached background job.

    python scripts/fetch_archive.py start      register and run the task
    python scripts/fetch_archive.py status     progress, per currency
    python scripts/fetch_archive.py logs
    python scripts/fetch_archive.py stop       remove the task (keeps files)
    python scripts/fetch_archive.py run        run in the foreground instead

Why a scheduled task rather than a background process: the job takes hours,
and a process launched from a shell is a child of that shell. An earlier
attempt with `Start-Process` reached 26 of 79 days and then stopped with an
empty stderr and no traceback -- killed with its parent, not crashed. The
collector already proves a scheduled task survives, so the same mechanism is
used here.

Every run is resumable. `fetch_free_archive` skips currency-days already on
disk, so an interruption costs only the day in flight, and re-running after
one is free.

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


def run() -> int:
    """
    Do the download. This is what the task executes.

    The loop lives here rather than using `fetch_free_archive` because of how
    a rate limit has to be handled. That function logs a skip and moves on,
    which is right for a one-off failure but wrong for a 429: the very next
    request is refused too, so one limit burned all 53 remaining days in a
    few seconds and left the queue looking attempted. Here a 429 backs off,
    retries, and if still refused **stops the run** -- leaving every
    untouched day genuinely untouched for the next pass.
    """
    import time

    import requests

    from data.tardis import free_days, load_day_multi

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    _log(f"--- start {stamp}, currencies {', '.join(CURRENCIES)} ---")
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    try:
        for day in free_days():
            paths = {
                c: HIST_DIR / f"{c.lower()}_chain_{day:%Y%m%d}.parquet"
                for c in CURRENCIES
            }
            todo = [c for c in CURRENCIES if not paths[c].exists()]
            if not todo:
                continue

            for attempt, wait in enumerate((0.0, *BACKOFF_SECONDS)):
                if wait:
                    _log(f"{day}  rate limited, waiting {wait:.0f}s "
                         f"(attempt {attempt})")
                    time.sleep(wait)
                try:
                    frames = load_day_multi(day, 60, todo)
                    for currency, frame in frames.items():
                        frame.to_parquet(paths[currency], index=False)
                        _log(f"{day}  {currency:4s} {len(frame):6d} rows  "
                             f"{frame['expiry_ts'].nunique():2d} expiries")
                    done += 1
                    break
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 429:
                        continue
                    _log(f"{day}  FAILED: {exc}")
                    failed += 1
                    break
                except (requests.RequestException, RuntimeError, OSError) as exc:
                    _log(f"{day}  FAILED: {exc}")
                    failed += 1
                    break
            else:
                _log(f"{day}  still rate limited after "
                     f"{len(BACKOFF_SECONDS)} retries -- stopping so the "
                     f"remaining days stay untried")
                break

            time.sleep(DELAY_SECONDS)

        _log(f"--- done {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ}, "
             f"{done} day(s) fetched, {failed} failed ---")
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
    # Start a minute out so the registration settles, then run it immediately.
    when = (dt.datetime.now() + dt.timedelta(minutes=1)).strftime("%H:%M")
    result = _schtasks("/create", "/tn", TASK_NAME, "/tr", command,
                       "/sc", "ONCE", "/st", when, "/f")
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return result.returncode

    settings = (
        "New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -StartWhenAvailable "
        "-ExecutionTimeLimit (New-TimeSpan -Hours 8)"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"Set-ScheduledTask -TaskName '{TASK_NAME}' -Settings ({settings})"],
        capture_output=True, text=True,
    )
    _schtasks("/run", "/tn", TASK_NAME)
    print(f"Registered and started '{TASK_NAME}' (8 hour limit).")
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
