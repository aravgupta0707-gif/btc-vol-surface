"""
Manage the hourly snapshot collector (Track A) as a Windows scheduled task.

    python scripts/collector.py status      what it is doing right now
    python scripts/collector.py logs        recent runs, newest last
    python scripts/collector.py install     register the hourly task
    python scripts/collector.py stop        disable it, keep it registered
    python scripts/collector.py start       re-enable it
    python scripts/collector.py uninstall   remove it entirely
    python scripts/collector.py run-once    take one snapshot by hand

The task is deliberately easy to see and easy to kill. It is registered
under a name you will recognise in Task Scheduler, every run appends a line
to data/raw/collector.log, and `status` reports the task state, the last and
next run times, and the snapshots actually on disk.

Measured cost of one run: 0.45s wall clock, 2.1 MB peak RAM, 41 KB written.
That is ~1 MB/day and ~0.4 GB/year, and the process is asleep 99.99% of the
time -- it exists for half a second an hour and then exits. It cannot
accumulate memory or hold a connection open, because nothing stays resident.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.fetch_deribit import COLLECT_CURRENCIES  # noqa: E402
LOG_PATH = ROOT / "data" / "raw" / "collector.log"
RAW_DIR = ROOT / "data" / "raw"
TASK_NAME = "BTC-Vol-Surface-Collector"
PYTHONW = Path(sys.executable).with_name("pythonw.exe")


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["schtasks", *args], capture_output=True, text=True, shell=False
    )


def _task_info() -> dict | None:
    """Parse `schtasks /query /v` into a dict, or None if not registered."""
    result = _schtasks("/query", "/tn", TASK_NAME, "/fo", "LIST", "/v")
    if result.returncode != 0:
        return None
    info = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    return info


def _apply_settings() -> subprocess.CompletedProcess:
    """
    Replace the settings `schtasks /create` defaults to.

    Those defaults are wrong for a laptop. `DisallowStartIfOnBatteries` is
    true out of the box, so on battery power the task never leaves the
    Queued state -- it reports Last Result 0 and silently collects nothing.
    That is exactly what happened on first install here.

    Also sets StartWhenAvailable, so a run missed while the machine was
    asleep fires on wake rather than being skipped, and a 10-minute
    execution limit so a wedged run cannot linger. A normal run is 0.45s.
    """
    settings = (
        "New-ScheduledTaskSettingsSet "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-StartWhenAvailable -MultipleInstances IgnoreNew "
        "-ExecutionTimeLimit (New-TimeSpan -Minutes 10)"
    )
    return subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"Set-ScheduledTask -TaskName '{TASK_NAME}' -Settings ({settings})",
        ],
        capture_output=True, text=True,
    )


def install(visible: bool = False) -> int:
    """
    Register the hourly task.

    `visible=True` runs it under python.exe, so a console window appears for
    about half a second every hour -- unmissable, but it briefly takes focus.
    The default uses pythonw.exe and stays quiet; `status`, `logs` and
    `watch` are then how you see what it is doing.
    """
    interpreter = Path(sys.executable) if visible else PYTHONW
    if not interpreter.exists():
        print(f"{interpreter.name} not found next to {sys.executable}")
        return 1
    if _task_info() is not None:
        print(f"'{TASK_NAME}' is already registered. Use 'status' to inspect it.")
        return 0

    command = f'"{interpreter}" "{ROOT / "scripts" / "collector.py"}" run-once'
    result = _schtasks(
        "/create", "/tn", TASK_NAME, "/tr", command,
        "/sc", "hourly", "/mo", "1", "/f",
    )
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return result.returncode

    settings = _apply_settings()
    if settings.returncode != 0:
        print("WARNING: task registered but settings were not applied:")
        print(settings.stdout or settings.stderr)
        print("On battery power it will sit Queued and collect nothing.")
        return settings.returncode

    print(f"Registered '{TASK_NAME}', running hourly"
          f"{' with a visible console window' if visible else ' quietly'}.")
    print()
    print("To see it:")
    print("  python scripts/collector.py status    state, last run, next run")
    print("  python scripts/collector.py watch     live tail of the log")
    print("  python scripts/collector.py logs      recent runs")
    print("To stop it:")
    print("  python scripts/collector.py stop      disable, keep registered")
    print("  python scripts/collector.py uninstall remove entirely")
    print("  Task Scheduler app -> Task Scheduler Library -> " + TASK_NAME)
    print()
    print("Cost per run: ~0.45s, ~2 MB RAM, 41 KB on disk. Nothing stays resident")
    print("between runs, so it cannot leak memory or hold a connection open.")
    return 0


def uninstall() -> int:
    if _task_info() is None:
        print(f"'{TASK_NAME}' is not registered; nothing to remove.")
        return 0
    result = _schtasks("/delete", "/tn", TASK_NAME, "/f")
    print(
        f"Removed '{TASK_NAME}'."
        if result.returncode == 0
        else (result.stdout or result.stderr)
    )
    print("Snapshots already collected are untouched in data/raw/.")
    return result.returncode


def set_enabled(enabled: bool) -> int:
    if _task_info() is None:
        print(f"'{TASK_NAME}' is not registered. Run 'install' first.")
        return 1
    result = _schtasks(
        "/change", "/tn", TASK_NAME, "/enable" if enabled else "/disable"
    )
    if result.returncode == 0:
        print(
            f"'{TASK_NAME}' {'enabled -- collecting hourly' if enabled else 'DISABLED -- not collecting'}."
        )
        if not enabled:
            print("It stays registered, so 'start' turns it back on.")
    else:
        print(result.stdout or result.stderr)
    return result.returncode


def status() -> int:
    info = _task_info()
    print(f"task     {TASK_NAME}")
    if info is None:
        print("state    NOT REGISTERED -- nothing is running")
        print("         'python scripts/collector.py install' to start collecting")
    else:
        state = info.get("Scheduled Task State", "?")
        marker = "COLLECTING" if state.lower() == "enabled" else "STOPPED"
        print(f"state    {marker}  (scheduled task is {state.lower()})")
        print(f"last run {info.get('Last Run Time', '?')}   "
              f"result {info.get('Last Result', '?')}")
        print(f"next run {info.get('Next Run Time', '?')}")
        print(f"runs as  {info.get('Task To Run', '?')}")

        # A task stuck in Queued is the failure mode that looks like success:
        # Last Result reads 0 and nothing is ever collected.
        if info.get("Status", "").strip().lower() == "queued":
            print()
            print("WARNING: status is Queued -- it is waiting on a condition and")
            print("         is NOT collecting. Usually battery power. Re-run")
            print("         'install' after 'uninstall' to reapply settings.")

    print()
    print("data     data/raw/")
    grand = 0
    for currency in COLLECT_CURRENCIES:
        snaps = (
            sorted(RAW_DIR.glob(f"{currency.lower()}_chain_*.parquet"))
            if RAW_DIR.exists() else []
        )
        size = sum(p.stat().st_size for p in snaps)
        grand += size
        span = (
            f"{snaps[0].name.split('_')[-1][:11]} .. "
            f"{snaps[-1].name.split('_')[-1][:11]}" if snaps else "-"
        )
        print(f"         {currency:>5}  {len(snaps):4d} snapshots  "
              f"{size / 1e6:6.1f} MB   {span}")
    print(f"         {'total':>5}  {grand / 1e6:20.1f} MB")
    print(f"log      {LOG_PATH if LOG_PATH.exists() else '(no runs logged yet)'}")
    return 0


def logs(lines: int = 20) -> int:
    if not LOG_PATH.exists():
        print("No runs logged yet.")
        return 0
    entries = LOG_PATH.read_text(encoding="utf-8").splitlines()
    print("\n".join(entries[-lines:]))
    return 0


NETWORK_WAIT_SECONDS = 120
NETWORK_POLL_SECONDS = 5


def wait_for_network(host: str = "www.deribit.com", port: int = 443,
                     limit: float = NETWORK_WAIT_SECONDS,
                     poll: float = NETWORK_POLL_SECONDS) -> float | None:
    """
    Block until `host` accepts a TCP connection; return seconds waited, or
    None if it never did within `limit`.

    The catch-up run fires the moment the laptop wakes, before Wi-Fi has
    reconnected. The fetch retries give up after about three seconds, so on
    18 and 19 September every book failed on wake and each catch-up snapshot
    -- an hour that can never be collected again -- was lost. The network
    takes tens of seconds to return; waiting for it costs nothing on a
    normal run, where the first probe succeeds at once.
    """
    import socket
    import time

    start = time.monotonic()
    while True:
        try:
            socket.create_connection((host, port), timeout=poll).close()
            return time.monotonic() - start
        except OSError:
            if time.monotonic() - start + poll > limit:
                return None
            time.sleep(poll)


def _describe(exc: BaseException) -> str:
    """The error and its root cause: 'failed after 3 attempts' alone says nothing."""
    text = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__
    if cause is not None:
        text += f" <- {type(cause).__name__}: {str(cause)[:160]}"
    return text


def run_once() -> int:
    """
    One snapshot per collected currency, appending a line each to the log.
    This is what the scheduled task runs.

    Each currency is attempted independently: one failing book must not cost
    us the others, since a missed hour cannot be collected later.
    """
    import datetime as dt
    import traceback

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines, code = [], 0
    waited = wait_for_network()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    if waited is None:
        lines.append(f"{stamp}   note  network unreachable after "
                     f"{NETWORK_WAIT_SECONDS}s; attempting anyway")
    elif waited >= 1:
        lines.append(f"{stamp}   note  waited {waited:.0f}s for the network (woke from sleep?)")
    for currency in COLLECT_CURRENCIES:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        try:
            from data.fetch_deribit import snapshot_chain, write_snapshot

            frame = snapshot_chain(currency)
            path = write_snapshot(frame, currency=currency)
            lines.append(
                f"{stamp}  {currency:>5}  ok    {len(frame):5d} instruments  "
                f"{frame['underlying_asset'].nunique()} underlying  {path.name}"
            )
        except Exception as exc:  # noqa: BLE001 - the log is the error channel
            lines.append(f"{stamp}  {currency:>5}  FAIL  {_describe(exc)}")
            traceback.print_exc()
            code = 1

    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return code


def watch() -> int:
    """Live tail of the log, so an hourly run is visible as it happens."""
    import time

    print(f"watching {LOG_PATH}")
    print("each line is one hourly run; Ctrl-C to stop watching "
          "(this does NOT stop collecting)\n")
    logs(10)
    position = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    try:
        while True:
            if LOG_PATH.exists() and LOG_PATH.stat().st_size > position:
                with LOG_PATH.open("r", encoding="utf-8") as handle:
                    handle.seek(position)
                    new = handle.read()
                    position = handle.tell()
                print(new, end="", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopped watching; the collector is still running")
    return 0


COMMANDS = {
    "status": status,
    "logs": logs,
    "watch": watch,
    "install": install,
    "uninstall": uninstall,
    "start": lambda: set_enabled(True),
    "stop": lambda: set_enabled(False),
    "run-once": run_once,
}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    command = args[0] if args else "status"
    if command not in COMMANDS:
        print(f"unknown command '{command}'")
        print("expected one of: " + ", ".join(COMMANDS))
        return 2
    sys.path.insert(0, str(ROOT))
    if command == "install":
        return install(visible="--visible" in flags)
    return COMMANDS[command]()


if __name__ == "__main__":
    raise SystemExit(main())
