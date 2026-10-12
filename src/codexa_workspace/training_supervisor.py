"""Lightweight OS-level supervisor for one approved training run. No LLM is consulted.

Authority boundary: the trainer owns in-loop guards (nonfinite, OOM, quality, disk
before save, deadline). The supervisor owns process-level policy only: stalls, the
hard deadline backstop, the disk reserve watchdog, signal forwarding, terminal
classification, final checkpoint verification and one notification/wakeup.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
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


class Supervisor:
    """Spawn or re-attach to one trainer, then wait on its pidfd; never polls an AI."""

    def __init__(self, run_dir: Path, root: Path, *, deliver=None) -> None:
        self.run_dir, self.root = Path(run_dir), Path(root)
        self.state = RunState(self.run_dir)
        self.launch = read_json(self.state.directory / 'launch.json')
        if not self.launch:
            raise FileNotFoundError('supervisor/launch.json is required.')
        self.policy = {**DEFAULT_POLICY, **self.launch.get('policy', {})}
        self.policy['stall_seconds'] = {**DEFAULT_POLICY['stall_seconds'], **self.launch.get('policy', {}).get('stall_seconds', {})}
        self.deliver = deliver
        self.stop_signal = False
        self.child = None
        self.log_path = self.state.directory / 'supervisor.log'

    def log(self, message: str) -> None:
        with self.log_path.open('a') as stream:
            stream.write(f'{now()} {message}\n')

    def _request_stop(self, number, frame) -> None:
        self.stop_signal = True

    def run(self) -> dict:
        if os.environ.get('CODEXA_WAKEUP_AGENT'):
            raise RuntimeError('A wakeup analysis agent may not supervise or launch training.')
        lock = (self.state.directory / 'lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log('another supervisor holds the run lock; exiting without action')
            return dict(action='already_supervised')
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        self.wake_read, wake_write = os.pipe()
        os.set_blocking(self.wake_read, False)
        os.set_blocking(wake_write, False)
        signal.set_wakeup_fd(wake_write)
        self.state.update(supervisor=dict(pid=os.getpid(), boot_id=boot_id(), started_utc=now(),
                                          unit=os.environ.get('CODEXA_SUPERVISOR_UNIT')))
        document = self.state.load()
        trainer = document.get('trainer')
        if document.get('state') in TERMINAL:
            self.log('run already terminal; ensuring notification delivery only')
            return self._finish(read_json(self.state.directory / 'completion.json'))
        if trainer:
            alive = (trainer.get('boot_id') == boot_id()
                     and process_start_ticks(trainer['pid']) == trainer.get('start_ticks'))
            if not alive:
                self.log('recorded trainer is gone; reconciling without relaunch')
                return self._terminate_and_classify(None, 'SUPERVISOR_LOST')
            self.log(f"re-attaching to live trainer pid {trainer['pid']}; no new trainer spawned")
            self.state.event('reattach', pid=trainer['pid'])
            return self._watch(trainer['pid'], child=False)
        if document.get('state') != 'READY':
            raise RuntimeError(f"Refusing to spawn from state {document.get('state')!r}; validate first.")
        return self._watch(self._spawn(), child=True)

    def _spawn(self) -> int:
        command = self.launch['command']
        environment = dict(os.environ, **self.launch.get('environment', {}))
        environment.pop('CODEXA_WAKEUP_AGENT', None)
        log = (self.state.directory / 'trainer.log').open('ab')
        self.child = subprocess.Popen(command, cwd=self.launch.get('cwd', str(self.root)), env=environment,
                                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        pid = self.child.pid
        record = dict(pid=pid, boot_id=boot_id(), start_ticks=process_start_ticks(pid), started_utc=now(),
                      started_epoch=time.time(), command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest())
        self.state.transition('RUNNING', trainer=record)
        self.state.event('spawn', pid=pid)
        self.log(f'spawned trainer pid {pid}')
        return pid

    def _watch(self, pid: int, *, child: bool) -> dict:
        policy = self.policy
        descriptor = os.pidfd_open(pid)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        poller.register(self.wake_read, select.POLLIN)
        trainer = self.state.load()['trainer']
        last_step, last_change = None, time.monotonic()
        reason, kill_at, forwarded = None, None, False
        try:
            while True:
                timeout = policy['poll_seconds']
                if kill_at is not None:
                    timeout = max(0.05, min(timeout, kill_at - time.monotonic()))
                ready = {fd for fd, _ in poller.poll(int(timeout * 1000))}
                if self.wake_read in ready:
                    try:
                        os.read(self.wake_read, 512)
                    except BlockingIOError:
                        pass
                if descriptor in ready:
                    break
                if self.stop_signal and not forwarded:
                    self.log('stop signal received; forwarding SIGTERM to trainer')
                    self.state.event('forward_stop', pid=pid)
                    os.kill(pid, signal.SIGTERM)
                    forwarded, kill_at = True, time.monotonic() + policy['kill_grace']
                    continue
                if kill_at is not None:
                    if time.monotonic() >= kill_at:
                        self.log('grace expired; sending SIGKILL')
                        self.state.event('kill', pid=pid, reason=reason)
                        os.kill(pid, signal.SIGKILL)
                        kill_at = time.monotonic() + 3600
                    continue
                heartbeat = read_json(self.run_dir / 'metrics/heartbeat.json') or {}
                progress = read_json(self.run_dir / 'metrics/progress.json') or {}
                budget = read_json(self.run_dir / 'metrics/budget.json') or {}
                wall = time.time()
                phase = heartbeat.get('phase', 'STARTING')
                step = heartbeat.get('optimizer_step')
                if step != last_step:
                    last_step, last_change = step, time.monotonic()
                limit = policy['stall_seconds'].get(phase, 1800)
                since_phase = wall - heartbeat.get('phase_started_utc', trainer['started_epoch'])
                since_step = time.monotonic() - last_change
                stalled_for = since_step if phase == 'TRAINING' else since_phase
                heartbeat_age = wall - heartbeat.get('ts_utc', trainer['started_epoch'])
                deadline = budget.get('deadline_utc', trainer['started_epoch'] + self.launch.get('deadline_seconds', 18000))
                free = shutil.disk_usage(self.run_dir).free
                if heartbeat_age > policy['heartbeat_timeout'] and wall - trainer['started_epoch'] > policy['heartbeat_timeout']:
                    reason = 'HEARTBEAT_LOST'
                elif stalled_for > limit:
                    reason = 'STALLED'
                elif wall > deadline + policy['deadline_grace']:
                    reason = 'DEADLINE_BACKSTOP'
                elif free < policy['disk_reserve_bytes']:
                    reason = 'DISK_RESERVE_WATCHDOG'
                target = ACTIVITY_STATE.get(phase, 'RUNNING')
                self.state.transition(target, phase=phase, optimizer_steps=step,
                    heartbeat_age_seconds=round(heartbeat_age, 1), stalled_for_seconds=round(stalled_for, 1),
                    disk_free_bytes=free, last_progress=dict(training_loss=progress.get('training_loss'),
                        validation_loss=progress.get('validation_loss'), tokens=progress.get('total_tokens_seen')))
                if reason:
                    self.log(f'{reason}: phase={phase} step={step} stalled_for={stalled_for:.0f}s heartbeat_age={heartbeat_age:.0f}s')
                    self.state.event('terminate', pid=pid, reason=reason)
                    os.kill(pid, signal.SIGTERM)
                    kill_at = time.monotonic() + (policy['kill_grace'] if reason == 'DISK_RESERVE_WATCHDOG'
                                                  else min(policy['kill_grace'], 120))
        finally:
            os.close(descriptor)
        status = None
        if child:
            code = self.child.wait()
            status = dict(code=code) if code >= 0 else dict(signal=-code)
        self.log(f'trainer pid {pid} exited: {status}')
        self.state.event('trainer_exit', pid=pid, status=status)
        return self._terminate_and_classify(status, reason)

    def _terminate_and_classify(self, status: dict | None, reason: str | None) -> dict:
        classification = classify(self.run_dir, exit_status=status, supervisor_reason=reason,
                                  stop_signal=self.stop_signal)
        if reason == 'SUPERVISOR_LOST' and classification['trainer_report_state'] not in TERMINAL:
            classification.update(terminal_state='INTERRUPTED', reason='SUPERVISOR_LOST_OR_HOST_RESTART')
        completion = write_completion(self.run_dir, classification)
        self.state.transition(completion['terminal_state'], failure_reason=completion['reason']
                              if completion['terminal_state'] != 'COMPLETED' else None,
                              completion_event_id=completion['event_id'])
        self.log(f"terminal {completion['terminal_state']} ({completion['reason']})")
        return self._finish(completion)

    def _finish(self, completion: dict | None) -> dict:
        if completion and self.deliver is not None:
            try:
                delivery = self.deliver(self.run_dir, self.root, defer_ai=self.stop_signal)
                self.state.event('delivery', ai=delivery['channels']['ai']['status'],
                                 desktop=delivery['channels']['desktop']['status'])
            except Exception as error:
                self.state.event('delivery_error', error=f'{type(error).__name__}: {error}')
                self.log(f'delivery error: {error}')
        return dict(action='finished', completion=completion)


def reconcile_all(root: Path, *, deliver=None) -> list[dict]:
    """Boot/login reconcile: never resumes training; finalizes lost runs and deferred deliveries."""
    results = []
    for state_path in sorted((Path(root) / 'outputs/training').glob('*/supervisor/state.json')):
        run_dir = state_path.parents[1]
        document = read_json(state_path) or {}
        state = document.get('state')
        if state in ACTIVE:
            lock = (run_dir / 'supervisor/lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                results.append(dict(run=run_dir.name, action='live_supervisor'))
                continue
            fcntl.flock(lock, fcntl.LOCK_UN)
            trainer = document.get('trainer') or {}
            if trainer.get('boot_id') == boot_id() and process_start_ticks(trainer.get('pid', -1)) == trainer.get('start_ticks'):
                results.append(dict(run=run_dir.name, action='orphan_trainer_alive_reattach_required'))
                continue
            supervisor = Supervisor(run_dir, root, deliver=deliver)
            reason = 'HOST_RESTART' if trainer.get('boot_id') != boot_id() else 'SUPERVISOR_LOST'
            supervisor.state.event('reconcile', reason=reason)
            classification = classify(run_dir, exit_status=None, supervisor_reason=None, stop_signal=True)
            if classification['trainer_report_state'] not in TERMINAL:
                classification.update(terminal_state='INTERRUPTED', reason=reason)
            completion = write_completion(run_dir, classification)
            supervisor.state.transition(completion['terminal_state'], failure_reason=completion['reason'],
                                        completion_event_id=completion['event_id'])
            supervisor._finish(completion)
            results.append(dict(run=run_dir.name, action='finalized', terminal_state=completion['terminal_state']))
        elif state in TERMINAL:
            notifications = read_json(run_dir / 'supervisor/notification-state.json') or {}
            ai = (notifications.get('channels') or {}).get('ai', {}).get('status')
            if deliver is not None and ai in (None, 'pending', 'deferred', 'failed_retryable', 'attempting'):
                state_document = deliver(run_dir, root)
                results.append(dict(run=run_dir.name, action='delivered', ai=state_document['channels']['ai']['status']))
            else:
                results.append(dict(run=run_dir.name, action='nothing_to_do', ai=ai))
    return results


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
