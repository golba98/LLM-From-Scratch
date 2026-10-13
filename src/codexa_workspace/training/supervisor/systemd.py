"""systemd user units: the transient per-run supervisor unit and the persistent reconcile unit."""

from pathlib import Path
import subprocess
import sys


def systemd_launch(run_dir: Path, root: Path, *, unit: str, python: str, script: Path,
                   environment: dict, dry_run: bool = False) -> list[str]:
    """Start the supervisor as a transient user unit owned by the user manager (Linger=yes)."""
    command = ['systemd-run', '--user', f'--unit={unit}', '--collect', '--no-block',
               '--property=KillMode=mixed', '--property=TimeoutStopSec=900', '--property=Restart=no',
               f'--working-directory={root}', f'--setenv=CODEXA_SUPERVISOR_UNIT={unit}']
    command += [f'--setenv={key}={value}' for key, value in sorted(environment.items())]
    command += [python, str(script), 'run', '--run-dir', str(run_dir)]
    if not dry_run:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return command


RECONCILE_UNIT = 'codexa-train-reconcile.service'


def install_reconcile_unit(root: Path, python: str, script: Path) -> Path:
    """Persistent oneshot: at user-manager start it finalizes lost runs and deferred wakeups only."""
    path = Path.home() / '.config/systemd/user' / RECONCILE_UNIT
    path.parent.mkdir(parents=True, exist_ok=True)
    quote = lambda value: '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'
    text = ('[Unit]\nDescription=Codexa training reconcile (finalize interrupted runs; never resumes training)\n'
            'After=default.target\n\n[Service]\nType=oneshot\n'
            f'WorkingDirectory={root}\n'
            f'ExecStart={quote(python)} {quote(script)} reconcile\n'
            'TimeoutStartSec=3600\n\n[Install]\nWantedBy=default.target\n')
    path.write_text(text)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, timeout=60)
    subprocess.run(['systemctl', '--user', 'enable', RECONCILE_UNIT], check=True, capture_output=True, timeout=60)
    return path


def unit_active(unit: str) -> str:
    return subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()


def main_python() -> str:
    return sys.executable
