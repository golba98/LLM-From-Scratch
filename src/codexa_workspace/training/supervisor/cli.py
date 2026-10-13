"""Supervisor command line: ``run`` one prepared run, or ``reconcile`` every run after a restart.

Jobs prepare, approve, launch and resume themselves (for example
``scripts/foundation_continuation.py``) and start ``scripts/training_supervisor.py run``
in a transient systemd unit. The persistent ``codexa-train-reconcile.service`` unit runs
``reconcile`` at login. Neither command ever starts new training or consults an AI.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

from codexa_workspace.notifications import notifier
from codexa_workspace.paths import REPO_ROOT
from codexa_workspace.training.supervisor.runner import Supervisor, reconcile_all

_startup_deliver = notifier.deliver


def deliver(*args, **kwargs):
    """Late-bound: load the notifier as it is on disk at the terminal event, so notification
    fixes reach a long-running supervisor; fall back to the startup copy on error."""
    try:
        fresh = importlib.reload(notifier).deliver
    except Exception as error:   # a broken on-disk edit must never cost the notification
        print(f"notifier reload failed ({type(error).__name__}: {error}); using startup copy", file=sys.stderr)
        fresh = _startup_deliver
    return fresh(*args, **kwargs)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "reconcile"))
    parser.add_argument("--run-dir", type=Path, help="run: the prepared run directory (absolute or repo-relative)")
    args = parser.parse_args(arguments)
    if args.command == "run":
        if args.run_dir is None:
            parser.error("run requires --run-dir")
        run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
        result = Supervisor(run_dir, REPO_ROOT, deliver=deliver).run()
        print(json.dumps(dict(action=result["action"]), indent=2))
    else:
        print(json.dumps(reconcile_all(REPO_ROOT, deliver=deliver), indent=2, default=str))
    return 0
