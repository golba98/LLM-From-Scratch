"""Bounded offline investigation of the completed conversational experiment."""
import argparse
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('diagnostics', 'pilots', 'cache', 'protocol', 'pilot-teacher', 'end-weight-pilot', 'report'), required=True)
    parser.add_argument('--budget-seconds', type=int, default=14400)
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 14400:
        parser.error('Investigation ceiling is four hours.')
    from codexa_workspace.chat_investigation import investigate
    investigate(Path(__file__).resolve().parents[1], args.stage, started=STARTED, seconds=args.budget_seconds)
