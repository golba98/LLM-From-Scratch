"""Instant local training status from small atomic documents; never contacts an AI service."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from codexa_workspace.training.runs import resolve_run_dir
from codexa_workspace.training.supervisor.state import read_json, TERMINAL


def gpu_memory() -> dict | None:
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        used, total = (int(value) for value in output.split(','))
        return dict(used_mib=used, total_mib=total)
    except Exception:
        return None


def collect(run_dir: Path, *, gpu: bool = True) -> dict:
    run_dir = Path(run_dir)
    state = read_json(run_dir / 'supervisor/state.json') or {}
    progress = read_json(run_dir / 'metrics/progress.json') or {}
    heartbeat = read_json(run_dir / 'metrics/heartbeat.json') or {}
    budget = read_json(run_dir / 'metrics/budget.json') or {}
    completion = read_json(run_dir / 'supervisor/completion.json')
    notifications = read_json(run_dir / 'supervisor/notification-state.json')
    gates = read_json(run_dir / 'config/launch-gates.json') or {}
    wall = time.time()
    remaining = budget.get('remaining_seconds')
    if budget and state.get('state') not in TERMINAL and budget.get('last_recorded_utc'):
        remaining = max(0.0, budget['deadline_utc'] - wall)
    elapsed = None if not budget else budget['seconds'] - (remaining if remaining is not None else 0)
    step, total = progress.get('step'), progress.get('total_steps')
    return dict(run_id=run_dir.name, state=state.get('state', 'NOT_PREPARED'), phase=heartbeat.get('phase'),
        trainer_state=progress.get('state'), elapsed_seconds=elapsed, deadline_seconds=budget.get('seconds'),
        remaining_seconds=remaining, optimizer_steps=step, max_steps=total,
        progress_fraction=(step / total) if step is not None and total else None,
        processed_tokens=progress.get('total_tokens_seen'), training_loss=progress.get('training_loss'),
        validation_loss=progress.get('validation_loss'), best_validation_loss=progress.get('best_validation_loss'),
        learning_rate=progress.get('learning_rate'), tokens_per_second=progress.get('tokens_per_second'),
        eta_seconds=progress.get('eta_seconds'), peak_vram_bytes=progress.get('peak_vram_bytes'),
        gpu=gpu_memory() if gpu else None,
        heartbeat_age_seconds=None if 'ts_utc' not in heartbeat else round(wall - heartbeat['ts_utc'], 1),
        last_verified_checkpoint=progress.get('last_verified_checkpoint'),
        failure_reason=state.get('failure_reason') or (completion or {}).get('reason') if state.get('state') != 'COMPLETED' else None,
        last_error=progress.get('error') or ((completion or {}).get('errors') or [None])[-1],
        blocked_gates=gates.get('failed') if state.get('state') == 'BLOCKED' else None,
        completion_event_id=(completion or {}).get('event_id'),
        final_checkpoint_verification=(completion or {}).get('final_checkpoint_verification'),
        notification=None if not notifications else {name: value['status'] for name, value in notifications['channels'].items()},
        trainer_pid=(state.get('trainer') or {}).get('pid'), supervisor_unit=(state.get('supervisor') or {}).get('unit'))


def duration(value) -> str:
    if value is None:
        return '--'
    value = max(0, int(value))
    return f'{value // 3600:d}h{value // 60 % 60:02d}m{value % 60:02d}s'


def render(status: dict) -> str:
    number = lambda value, digits=4: '--' if value is None else f'{value:.{digits}f}'
    fraction = status['progress_fraction'] or 0.0
    bar = '#' * int(fraction * 40) + '-' * (40 - int(fraction * 40))
    checkpoint = status['last_verified_checkpoint'] or {}
    gpu = status['gpu'] or {}
    lines = [f"Codexa {status['run_id']}",
        f"  state      {status['state']}  (phase {status['phase'] or '--'}, trainer {status['trainer_state'] or '--'})",
        f"  progress   [{bar}] {100 * fraction:5.1f}%  updates {status['optimizer_steps'] if status['optimizer_steps'] is not None else '--'}/{status['max_steps'] or '--'}",
        f"  time       elapsed {duration(status['elapsed_seconds'])}  remaining {duration(status['remaining_seconds'])}  ETA {duration(status['eta_seconds'])}",
        f"  tokens     {status['processed_tokens'] or 0:,}  ({number(status['tokens_per_second'], 0)} tok/s)",
        f"  loss       train {number(status['training_loss'])}  val {number(status['validation_loss'])}  best val {number(status['best_validation_loss'])}",
        f"  lr         {status['learning_rate'] if status['learning_rate'] is not None else '--'}",
        f"  gpu        {gpu.get('used_mib', '--')}/{gpu.get('total_mib', '--')} MiB used; trainer peak reserved "
        f"{number((status['peak_vram_bytes'] or 0) / 1024**3, 2)} GiB",
        f"  checkpoint step {checkpoint.get('optimizer_step', '--')}  {checkpoint.get('path', '--')}",
        f"  heartbeat  {status['heartbeat_age_seconds'] if status['heartbeat_age_seconds'] is not None else '--'} s ago; trainer pid {status['trainer_pid'] or '--'}; unit {status['supervisor_unit'] or '--'}"]
    if status['failure_reason'] or status['last_error']:
        lines.append(f"  problem    {status['failure_reason'] or ''} {status['last_error'] or ''}".rstrip())
    if status['blocked_gates']:
        lines.append(f"  blocked by {', '.join(status['blocked_gates'])}")
    if status['completion_event_id']:
        lines.append(f"  final ckpt {status['final_checkpoint_verification']}; notifications {status['notification']}")
    return '\n'.join(lines)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Show Codexa training status instantly from local files '
                                                 '(no AI service is contacted).')
    parser.add_argument('--run-dir', type=Path, help='run directory (default: the most recently active run)')
    parser.add_argument('--json', action='store_true', help='compact JSON')
    parser.add_argument('--watch', type=float, nargs='?', const=10.0, metavar='SECONDS',
                        help='refresh locally every SECONDS (default 10)')
    args = parser.parse_args(arguments)
    run_dir = resolve_run_dir(args.run_dir)
    while True:
        status = collect(run_dir)
        text = json.dumps(status, separators=(',', ':')) if args.json else render(status)
        print(('\033[2J\033[H' if args.watch else '') + text, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)
