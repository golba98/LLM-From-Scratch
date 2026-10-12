"""Open, or reuse, the one 120x30 Kitty dashboard window for a run.

Every training start reaches this through the trainer's viewer hook
(``LLM_TRAINING_VIEWER_COMMAND`` → ``training_dashboard.py --attach``). The window runs in
its own systemd user unit, so it outlives the supervisor (``KillMode=mixed``) and still
shows the final banner. Live dashboards register in ``$XDG_RUNTIME_DIR`` — never in
the run directory — so a resume reuses the open window instead of stacking a new one.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .training_notifications import WINDOW_ENVIRONMENT, WINDOW_SIZE
from .training_supervisor import process_start_ticks

TITLE = 'Codexa training dashboard'
SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/training_dashboard.py'


def registry_path(run_dir: Path) -> Path:
    runtime = Path(os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}')
    return runtime / f'codexa-dashboard-{Path(run_dir).name}.json'


def register(run_dir: Path) -> None:
    """Called by a live dashboard so later starts can reuse it."""
    path = registry_path(run_dir)
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(dict(pid=os.getpid(), start_ticks=process_start_ticks(os.getpid()),
                                             run_dir=str(run_dir))))
        temporary.replace(path)
    except OSError:
        pass   # reuse is a convenience; the dashboard works without it


def unregister(run_dir: Path) -> None:
    path = registry_path(run_dir)
    try:
        if json.loads(path.read_text()).get('pid') == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


def live_dashboard(run_dir: Path) -> int | None:
    """PID of a registered dashboard for this run that is still running, else None."""
    try:
        record = json.loads(registry_path(run_dir).read_text())
    except (OSError, ValueError):
        return None
    pid = record.get('pid')
    return pid if pid and process_start_ticks(pid) == record.get('start_ticks') else None


def window_command(run_dir: Path, *, marker: Path | None = None, interval: float = 2.0) -> list[str]:
    environment = {key: os.environ[key] for key in WINDOW_ENVIRONMENT if os.environ.get(key)}
    command = ['systemd-run', '--user', '--collect', '--no-block',
               f'--unit=codexa-dashboard-{Path(run_dir).name}-{int(time.time())}']
    command += [f'--setenv={key}={value}' for key, value in sorted(environment.items())]
    command += [shutil.which('kitty') or 'kitty', '--title', TITLE, '-o', 'remember_window_size=no',
                '-o', WINDOW_SIZE[0], '-o', WINDOW_SIZE[1],
                sys.executable, str(SCRIPT), '--run-dir', str(run_dir), '--interval', str(interval)]
    return command + (['--marker', str(marker)] if marker else [])


def open_window(run_dir: Path, *, marker: Path | None = None, interval: float = 2.0, runner=subprocess.run) -> dict:
    """Open a new dashboard window unless one for this run is already open."""
    existing = live_dashboard(run_dir)
    if existing:
        return dict(opened=False, pid=existing)
    command = window_command(run_dir, marker=marker, interval=interval)
    completed = runner(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(f'Could not open the dashboard window: {completed.stderr.strip()[-300:]}')
    return dict(opened=True, pid=None, unit=command[4].split('=', 1)[1])


def attach(status: Path, marker: Path, *, runner=subprocess.run, timeout: float = 14.0, poll: float = 0.2,
           wait_forever: bool = True) -> int:
    """Trainer viewer hook: make the dashboard visible, mark it attached, then live as long as it does."""
    run_dir = Path(status).resolve().parent.parent
    marker = Path(marker)
    result = open_window(run_dir, marker=marker, runner=runner)
    pid = result['pid']
    if pid:                                   # reuse: the open window is the attached viewer
        marker.write_text(json.dumps(dict(pid=pid, reused=True)))
    else:
        deadline = time.monotonic() + timeout
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise SystemExit('Dashboard window did not attach.')
            time.sleep(poll)
        pid = json.loads(marker.read_text())['pid']
    ticks = process_start_ticks(pid)
    while wait_forever and ticks is not None and process_start_ticks(pid) == ticks:
        time.sleep(2)                         # exits when the window closes → trainer records viewer_closed
    return pid
