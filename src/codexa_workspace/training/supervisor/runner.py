"""The supervisor process: spawn or re-attach to one trainer, watch it, classify the end, notify.

Authority boundary: the trainer owns in-loop guards (nonfinite, OOM, quality, disk
before save, deadline). The supervisor owns process-level policy only: stalls, the
hard deadline backstop, the disk reserve watchdog, signal forwarding, terminal
classification, final checkpoint verification and one notification/wakeup. No LLM is consulted.
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
import time

from codexa_workspace.training.runs import run_states
from codexa_workspace.training.supervisor.state import (
    ACTIVE, ACTIVITY_STATE, DEFAULT_POLICY, TERMINAL, RunState, boot_id, classify, now, process_start_ticks, read_json,
    write_completion)


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
    for state_path in run_states(root):
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
