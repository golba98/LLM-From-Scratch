"""Run state for the OS-level training supervisor: states, transitions, classification, completion.

The process itself lives in ``runner``; systemd units in ``systemd``.
"""

import hashlib
import json
import os
from pathlib import Path
import time

from llm_training.session import atomic_document

STATES = ('PREPARING', 'VALIDATING', 'READY', 'BLOCKED', 'RUNNING', 'EARLY_EVALUATION', 'CHECKPOINTING',
          'COMPLETED', 'EARLY_STOPPED', 'FAILED', 'INTERRUPTED')
TERMINAL = ('COMPLETED', 'EARLY_STOPPED', 'FAILED', 'INTERRUPTED')
ACTIVE = ('RUNNING', 'EARLY_EVALUATION', 'CHECKPOINTING')
TRANSITIONS = {
    None: {'PREPARING', 'VALIDATING', 'BLOCKED'},
    'PREPARING': {'VALIDATING', 'BLOCKED'},
    'VALIDATING': {'READY', 'BLOCKED'},
    'BLOCKED': {'VALIDATING', 'PREPARING'},
    'READY': {'VALIDATING', 'RUNNING', 'BLOCKED', 'FAILED'},
    **{state: set(ACTIVE) | set(TERMINAL) for state in ACTIVE},
    **{state: set() for state in TERMINAL},
}
ACTIVITY_STATE = {'STARTING': 'RUNNING', 'TRAINING': 'RUNNING', 'FINALIZING': 'CHECKPOINTING',
                  'BASELINE': 'EARLY_EVALUATION', 'VALIDATING': 'EARLY_EVALUATION',
                  'GENERATION_EVAL': 'EARLY_EVALUATION', 'CHECKPOINTING': 'CHECKPOINTING'}
DEFAULT_POLICY = dict(poll_seconds=60, heartbeat_timeout=180, kill_grace=600, deadline_grace=600,
                      disk_reserve_bytes=100 * 1024**3,
                      stall_seconds=dict(STARTING=1800, BASELINE=1800, TRAINING=900, VALIDATING=1200,
                                         GENERATION_EVAL=1200, CHECKPOINTING=900, FINALIZING=1800))


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def boot_id() -> str:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def process_start_ticks(pid: int) -> int | None:
    """Kernel start time distinguishes a live trainer from a recycled PID."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, ProcessLookupError):
        return None


class RunState:
    """Atomic state document plus append-only, fsynced event journal."""

    def __init__(self, run_dir: Path) -> None:
        self.directory = Path(run_dir) / 'supervisor'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'state.json'
        self.events = self.directory / 'events.jsonl'

    def load(self) -> dict:
        return read_json(self.path) or {}

    def event(self, kind: str, **values) -> None:
        line = json.dumps(dict(at=now(), event=kind, **values), allow_nan=False)
        with self.events.open('a') as stream:
            stream.write(line + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def update(self, **values) -> dict:
        document = self.load()
        document.update(values, updated_utc=now())
        atomic_document(self.path, document)
        return document

    def transition(self, target: str, **values) -> dict:
        current = self.load().get('state')
        if target == current:
            return self.update(**values)
        if target not in TRANSITIONS.get(current, set()):
            raise ValueError(f'Illegal supervisor transition {current} -> {target}.')
        self.event('transition', source=current, target=target, **{k: v for k, v in values.items()
                                                                   if k in ('reason', 'failure_reason')})
        return self.update(state=target, **values)


def recovery_role(run_dir: Path, role: str, *, hash_payload: bool = True) -> dict | None:
    """Independently verify a published recovery role from index, sidecar and payload digest."""
    index = read_json(Path(run_dir) / 'checkpoints/recovery-index.json') or {}
    record = index.get(role)
    if not record:
        return None
    path = Path(run_dir) / 'checkpoints' / record['path']
    result = dict(role=role, path=str(path), optimizer_step=record.get('optimizer_step'), sha256=record.get('sha256'),
                  bytes=record.get('bytes'))
    sidecar = path.with_suffix(path.suffix + '.sha256')
    try:
        expected = sidecar.read_text().split()[0]
    except (FileNotFoundError, IndexError):
        return dict(result, verified=False, problem='missing checksum sidecar')
    if expected != record.get('sha256'):
        return dict(result, verified=False, problem='sidecar differs from index')
    if hash_payload:
        digest = hashlib.sha256()
        try:
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
        except FileNotFoundError:
            return dict(result, verified=False, problem='payload missing')
        if digest.hexdigest() != expected:
            return dict(result, verified=False, problem='payload digest mismatch')
    return dict(result, verified=True)


def classify(run_dir: Path, *, exit_status: dict | None, supervisor_reason: str | None,
             stop_signal: bool) -> dict:
    """Combine process status, trainer report and independent checkpoint verification."""
    report = read_json(Path(run_dir) / 'metrics/report.json') or {}
    trainer_state = report.get('state')
    errors = []
    if report.get('error'):
        errors.append(f"{report.get('error_type')}: {report['error']}")
    final = recovery_role(run_dir, 'final') if report.get('final_checkpoint_verified') else None
    if supervisor_reason in ('STALLED', 'DEADLINE_BACKSTOP', 'HEARTBEAT_LOST'):
        state, reason = 'FAILED', supervisor_reason
    elif supervisor_reason == 'DISK_RESERVE_WATCHDOG':
        state, reason = ('EARLY_STOPPED' if trainer_state in TERMINAL else 'FAILED'), supervisor_reason
    elif trainer_state not in TERMINAL:
        if stop_signal or (exit_status or {}).get('signal'):
            state, reason = 'INTERRUPTED', 'TRAINER_TERMINATED_WITHOUT_REPORT'
        else:
            state, reason = 'FAILED', 'TRAINER_EXITED_WITHOUT_REPORT'
    else:
        state, reason = trainer_state, report.get('stop_reason')
        if state == 'COMPLETED':
            code = (exit_status or {}).get('code')
            if code not in (0, None):
                state, reason = 'FAILED', f'NONZERO_EXIT_{code}'
            elif not (final and final['verified']):
                state, reason = 'FAILED', 'FINAL_CHECKPOINT_UNVERIFIED'
                errors.append(f"final checkpoint verification: {(final or {}).get('problem', 'no verified final role')}")
    latest = recovery_role(run_dir, 'latest', hash_payload=False)
    verification = 'verified' if final and final['verified'] else ('failed' if final else 'not_available')
    return dict(terminal_state=state, reason=reason, trainer_report_state=trainer_state, errors=errors,
                final_checkpoint=final, final_checkpoint_verification=verification,
                last_verified_checkpoint=latest, exit_status=exit_status, report=report)


def write_completion(run_dir: Path, classification: dict) -> dict:
    """Create the single terminal event; an existing record is never replaced."""
    path = Path(run_dir) / 'supervisor/completion.json'
    existing = read_json(path)
    if existing:
        return existing
    report = classification.pop('report')
    job = read_json(Path(run_dir) / 'config/job.json') or {}
    launch = read_json(Path(run_dir) / 'supervisor/launch.json') or {}
    counters = report.get('counters') or {}
    validations = counters.get('validations') or []
    training = report.get('training_state') or {}
    budget = report.get('budget') or read_json(Path(run_dir) / 'metrics/budget.json') or {}
    run_id = job.get('run_id') or launch.get('run_id') or Path(run_dir).name
    steps = training.get('optimizer_step')
    event = hashlib.sha256(f"{run_id}:{classification['terminal_state']}:{steps}:{report.get('run_id')}".encode()).hexdigest()
    spec = job.get('spec') or {}
    completion = dict(format='codexa-training-completion-v1', event_id=event, run_id=run_id,
        trainer_run_id=report.get('run_id'), job_sha256=job.get('job_sha256'), spec_sha256=job.get('spec_sha256'),
        created_utc=now(), **classification,
        elapsed_seconds=budget.get('elapsed_seconds'), deadline_seconds=budget.get('seconds'),
        optimizer_steps=steps, max_steps=(spec.get('training') or {}).get('max_steps'),
        processed_tokens=training.get('tokens_seen'),
        validation=dict(baseline=counters.get('baseline_validation_loss'), best=counters.get('best_validation_loss'),
                        final=validations[-1]['weighted_loss'] if validations else None,
                        final_step=validations[-1]['optimizer_step'] if validations else None,
                        baseline_sources=counters.get('baseline_sources'),
                        final_sources=validations[-1]['sources'] if validations else None),
        generations=counters.get('generations'),
        wakeup_provider=(launch.get('wakeup') or job.get('wakeup') or {}).get('provider'),
        paths=dict(run=str(run_dir), report=str(Path(run_dir) / 'metrics/report.json'),
                   metrics=str(Path(run_dir) / 'metrics/train_metrics.jsonl'),
                   validation=str(Path(run_dir) / 'metrics/validation-strata.jsonl'),
                   evaluations=str(Path(run_dir) / 'evaluations'),
                   trainer_log=str(Path(run_dir) / 'supervisor/trainer.log'),
                   supervisor_log=str(Path(run_dir) / 'supervisor/supervisor.log'),
                   events=str(Path(run_dir) / 'supervisor/events.jsonl'),
                   recovery_index=str(Path(run_dir) / 'checkpoints/recovery-index.json')))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(completion, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return completion
