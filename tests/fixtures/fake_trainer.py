"""Scenario-driven stand-in for the real trainer: same files, no GPU, seconds not hours."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--step-seconds', type=float, default=.15)
    parser.add_argument('--budget-seconds', type=float, default=60)
    args = parser.parse_args()
    run, scenario = args.run_dir, args.scenario
    metrics = run / 'metrics'
    started = time.time()
    write(metrics / 'budget.json', dict(seconds=args.budget_seconds, started_utc=started,
        deadline_utc=started + args.budget_seconds, reserve_seconds=1, remaining_seconds=args.budget_seconds,
        elapsed_seconds=0, last_recorded_utc=started))
    beat = dict(pid=os.getpid(), phase='TRAINING', phase_started_utc=started, optimizer_step=0)
    stop = threading.Event()
    requested = []
    if scenario in ('stall', 'overrun', 'heartbeat_lost'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: requested.append(1))

    def heartbeat():
        while not stop.wait(.1):
            if scenario == 'heartbeat_lost' and beat['optimizer_step'] >= 2:
                continue
            write(metrics / 'heartbeat.json', dict(beat, ts_utc=time.time()))
    threading.Thread(target=heartbeat, daemon=True).start()

    def checkpoint(step: int, roles: tuple[str, ...], *, corrupt: bool = False) -> dict:
        slot = run / 'checkpoints' / f'slot-{step % 6}' / 'checkpoint.pt'
        slot.parent.mkdir(parents=True, exist_ok=True)
        payload = f'fake-checkpoint-{step}'.encode()
        slot.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        slot.with_suffix('.pt.sha256').write_text(f'{digest}  checkpoint.pt\n')
        if corrupt:
            slot.write_bytes(b'corrupted')
        index_path = run / 'checkpoints/recovery-index.json'
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        record = dict(path=str(slot.relative_to(run / 'checkpoints')), sha256=digest, bytes=len(payload),
                      optimizer_step=step, verified=True)
        for role in roles:
            index[role] = record
        write(index_path, index)
        return record

    def report(state: str, reason: str, step: int, **values) -> None:
        validations = [dict(optimizer_step=0, weighted_loss=3.2, sources=dict(a=3.1, b=3.3)),
                       dict(optimizer_step=step, weighted_loss=3.1, sources=dict(a=3.0, b=3.2))]
        write(metrics / 'report.json', dict(state=state, stop_reason=reason, run_id='fake-run',
            training_state=dict(optimizer_step=step, tokens_seen=step * 65536),
            counters=dict(validations=validations, baseline_validation_loss=3.2, best_validation_loss=3.1,
                          baseline_sources=dict(a=3.1, b=3.3), generations=[]),
            budget=dict(seconds=args.budget_seconds, elapsed_seconds=time.time() - started), **values))

    step = 0
    for step in range(1, args.steps + 1):
        time.sleep(args.step_seconds)
        if scenario == 'stall' and step >= 2:
            while True:
                time.sleep(1)
        if requested:
            checkpoint(step - 1, ('latest', 'final'))
            report('INTERRUPTED', 'INTERRUPTED_SIGNAL_15', step - 1, final_checkpoint_verified=True)
            return 0
        beat.update(optimizer_step=step)
        write(metrics / 'progress.json', dict(state='RUNNING', step=step, total_steps=args.steps,
            training_loss=3.0, tokens_per_second=8000., total_tokens_seen=step * 65536))
        if step == 2:
            checkpoint(2, ('latest',))
        if scenario in ('crash', 'nonfinite', 'checkpoint_failure') and step == 3:
            error = dict(crash=('RuntimeError', 'CUDA error: an illegal memory access'),
                         nonfinite=('FloatingPointError', 'Non-finite training loss: nan.'),
                         checkpoint_failure=('CheckpointPublicationError', 'No space left'))[scenario]
            report('FAILED', 'CHECKPOINT_FAILURE' if scenario == 'checkpoint_failure' else error[0], 3,
                   error_type=error[0], error=error[1], final_checkpoint_verified=False)
            return 1
        if scenario == 'early_stop' and step == 3:
            checkpoint(3, ('latest', 'final'))
            report('EARLY_STOPPED', 'GRADIENT_NORM_GUARD', 3, final_checkpoint_verified=True)
            return 0
    if scenario in ('overrun', 'heartbeat_lost'):
        while True:
            time.sleep(1)
    if scenario == 'no_report':
        return 0
    checkpoint(step, ('latest', 'final'), corrupt=scenario == 'corrupt_final')
    report('COMPLETED', 'STEP_CAP', step, final_checkpoint_verified=True)
    stop.set()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
