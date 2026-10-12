"""Prepare or explicitly operate the isolated Codexa conversational successor."""

from pathlib import Path
import argparse
import sys
import time

STARTED = (time.time(), time.monotonic())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'prepare', 'start', 'view', 'logs', 'stop', 'resume', 'evaluate', '_supervise', '_worker'))
    parser.add_argument('--config', type=Path, default=Path('configs/chat-10h-v1.yaml'))
    parser.add_argument('--run', default='codexa-900m-chat-10h-v1')
    parser.add_argument('--stage', choices=('all', 'audit', 'candidates', 'benchmark', 'deterministic-benchmark', 'headroom-benchmark', 'pilots', 'resume-pilot', 'recovery', 'deterministic-recovery', 'diagnostics', 'finalize'), default='all')
    parser.add_argument('--tail', type=int, default=50)
    parser.add_argument('--suite', choices=('development', 'acceptance'), default='development')
    parser.add_argument('--resume-worker', action='store_true')
    arguments = parser.parse_args()
    from codexa_workspace.chat_preparation import main
    raise SystemExit(main(arguments, started=STARTED))
