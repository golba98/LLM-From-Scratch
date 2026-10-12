"""Prepare or explicitly authorize the proposed 20-minute legacy confirmation."""
import argparse
import json
import os
import signal
import yaml
from pathlib import Path
import sys
import time
STARTED = (time.time(), time.monotonic())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codexa_workspace.bounded_chat_confirmation import preflight, launch, supervise, worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'start', '_supervise', '_worker', 'view', 'stop', 'evaluate'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/chat-end3-confirmation-v1.yaml')
    parser.add_argument('--approval-file', type=Path)
    args = parser.parse_args()
    if args.action == 'preflight':
        result = preflight(ROOT, args.config)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['passed'] else 2)
    if args.action == 'start':
        if args.approval_file is None:
            parser.error('start requires separate explicit --approval-file')
        launch(ROOT, args.config, args.approval_file, STARTED)
    elif args.action in ('view', 'stop', 'evaluate'):
        document = yaml.safe_load(args.config.read_text())
        directory = ROOT / 'outputs/chat-training' / document['run_name']
        if not directory.exists():
            parser.error('No run exists; preparation does not create one.')
        if args.action == 'view':
            from llm_training.conversational_training import start_chat_viewer
            start_chat_viewer(directory / 'progress.json', title=document['run_name'])
        elif args.action == 'stop':
            process = json.loads((directory / 'process.json').read_text())
            proc = Path('/proc') / str(process['worker_pid'])
            ticks = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
            if ticks != process['worker_started_ticks'] or b'bounded_chat_confirmation.py' not in proc.joinpath('cmdline').read_bytes():
                raise RuntimeError('Worker identity changed; no signal sent.')
            os.kill(process['worker_pid'], signal.SIGTERM)
        else:
            from codexa_workspace.chat_operations import evaluate_run
            checkpoint = str(Path(document['spec']['initialization']['path']).parents[1] / 'codexa-900m-base-v1/latest.pt')
            print(json.dumps(evaluate_run(ROOT, dict(candidates=[checkpoint]), directory, directory, suite='development'), indent=2))
    elif args.action == '_supervise':
        raise SystemExit(supervise(ROOT, args.config))
    elif args.action == '_worker':
        raise SystemExit(worker(ROOT, args.config))
