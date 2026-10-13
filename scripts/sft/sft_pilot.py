#!/usr/bin/env python3
"""Authorization-time commands for the conversational SFT pilot. Run only by the user.

  approve  record the user's explicit grant for one exact frozen job hash (never overwrites)
  launch   launch through the OS supervisor; refuses unless the grant matches, readiness is
           READY FOR AUTHORIZATION and every preflight passes at this moment

Neither command is run by preparation or by an AI agent. Readiness is not authorization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    approve = commands.add_parser('approve')
    approve.add_argument('--experiment', type=Path, required=True)
    approve.add_argument('--job-sha256', required=True)
    approve.add_argument('--statement', required=True)
    start = commands.add_parser('launch')
    start.add_argument('--experiment', type=Path, required=True)
    start.add_argument('--provider', choices=('codex', 'claude'), required=True)
    args = parser.parse_args()
    from codexa_workspace import sft_phase4_data as data
    from codexa_workspace import sft_pilot_launch as launch
    from codexa_workspace import sft_pilot_runtime as runtime
    experiment = args.experiment.resolve()
    job = data.read(experiment)
    directory = experiment.parent.parent
    stamp = datetime.now(timezone.utc).isoformat()
    if args.command == 'approve':
        if args.job_sha256 != job['job_sha256']:
            raise SystemExit('The hash does not name this frozen job; nothing was approved.')
        print(launch.write_approval(directory, job, statement=args.statement, written_utc=stamp))
        return 0
    from codexa_workspace import sft_readiness
    result = launch.launch_authorized(
        directory, job, provider=args.provider, readiness=lambda: sft_readiness.evaluate(experiment),
        environment=runtime.TrainingGuardProvider('new'), measured=runtime.measure_resources(ROOT),
        written_utc=stamp)
    print(json.dumps({k: result[k] for k in ('launched', 'run_dir', 'command')}, indent=2))
    print(f"Monitor (read-only): {ROOT / '.venv/bin/python'} {ROOT / 'scripts/training_dashboard.py'} "
          f"--run-dir {result['run_dir']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
