"""Notification policy: one popup per real terminal event, none for tests; mock backends only.

No test here reaches GNOME, notify-send or an agent CLI: backends are injected, and
the conftest downgrade keeps any un-injected path on the recording mock.
"""

import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from codexa_workspace import training_notifications as notifications
from codexa_workspace import training_supervisor as supervisor
from codexa_workspace.foundation_training import fingerprint, APPROVAL_FORMAT


class Desktop:
    name = 'test-desktop'

    def __init__(self, log: Path, *, available=True, success=True, delay=0., explode=False):
        self.log, self.ok, self.success, self.delay, self.explode = log, available, success, delay, explode

    def available(self):
        return (True, 'ok') if self.ok else (False, 'no session bus')

    def send(self, summary, body, *, urgency, replace_id):
        if self.explode:
            os._exit(9)  # simulate a crash after the popup call began
        time.sleep(self.delay)
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(summary=summary, body=body, urgency=urgency, replace_id=replace_id)) + '\n')
        return dict(success=self.success, exit_code=0 if self.success else 1, notification_id=41 if self.success else None)


class Agent:
    name = 'test-agent'

    def __init__(self, log: Path, *, submitted=True, success=True):
        self.log, self.submitted, self.success = log, submitted, success

    def submit(self, provider, run_dir, root, prompt):
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(provider=provider, prompt=prompt[:60])) + '\n')
        return dict(submitted=self.submitted, **({} if self.submitted else dict(error='exited immediately with 1')))

    def wait(self, provider, run_dir, handle, *, timeout=0):
        return dict(success=self.success, exit_code=0 if self.success else 1, session_id='s-1' if self.success else None)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def make_run(tmp_path: Path, *, run_id='codexa-real-run', state='COMPLETED', reason='STEP_CAP', verified=True,
             execution_class=None, grant=False, event='e1', step=1792) -> Path:
    run_dir = tmp_path / 'outputs/training' / run_id
    (run_dir / 'supervisor').mkdir(parents=True, exist_ok=True)
    launch = dict(run_id=run_id, wakeup=dict(provider='codex'))
    if execution_class:
        launch['execution_class'] = execution_class
    (run_dir / 'supervisor/launch.json').write_text(json.dumps(launch))
    if grant:
        (run_dir / 'config').mkdir(exist_ok=True)
        job = dict(format='codexa-foundation-training-job-v1', run_id=run_id, spec={})
        job['job_sha256'] = fingerprint(job)
        (run_dir / 'config/job.json').write_text(json.dumps(job))
        (run_dir / 'config/approval.json').write_text(json.dumps(dict(format=APPROVAL_FORMAT, approved=True,
                                                                      job_sha256=job['job_sha256'])))
    completion = dict(event_id=event, run_id=run_id, terminal_state=state, reason=reason, optimizer_steps=step,
                      final_checkpoint_verification='verified' if verified else 'not_available',
                      last_verified_checkpoint=dict(optimizer_step=128), wakeup_provider='codex',
                      validation={}, paths={}, errors=[])
    (run_dir / 'supervisor/completion.json').write_text(json.dumps(completion))
    return run_dir


@pytest.fixture
def logs(tmp_path):
    return tmp_path / 'desktop.jsonl', tmp_path / 'agent.jsonl'


def deliver(run_dir, logs, **options):
    desktop = options.pop('desktop', None) or Desktop(logs[0])
    agent = options.pop('agent', None) or Agent(logs[1])
    return notifications.deliver(run_dir, run_dir.parents[2], desktop_backend=desktop, ai_backend=agent, **options)


# 1, 2 — one popup for a real completion and for a real failure; truthful text.
def test_completed_and_failed_real_runs_each_notify_once(tmp_path, logs):
    done = make_run(tmp_path, execution_class='production')
    result = deliver(done, logs)
    assert result['channels']['desktop']['status'] == 'delivered' and result['channels']['ai']['status'] == 'delivered'
    popups = lines(logs[0])
    assert len(popups) == 1 and popups[0]['summary'] == 'Codexa — Training completed'
    assert 'final checkpoint verified' in popups[0]['body'] and 'Analysis requested' in popups[0]['body']
    failed = make_run(tmp_path, run_id='codexa-other', state='FAILED', reason='CUDA_OOM', verified=False,
                      execution_class='production', event='e2', step=300)
    deliver(failed, logs)
    popups = lines(logs[0])
    assert len(popups) == 2 and popups[1]['summary'] == 'Codexa — Training stopped'
    assert 'CUDA_OOM' in popups[1]['body'] and 'step 128 preserved' in popups[1]['body']
    assert 'verified (step' not in popups[1]['body'] and popups[1]['urgency'] == 'critical'


# 3, 5, 6, 11 — repeated polling, supervisor restart (re-deliver), event re-reads: silent; new runs independent.
def test_repeated_delivery_is_silent_and_new_runs_notify_independently(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    for _ in range(5):
        deliver(run_dir, logs)
    supervisor.reconcile_all(tmp_path, deliver=lambda d, r, **k: deliver(d, logs, **k))
    assert len(lines(logs[0])) == 1 and len(lines(logs[1])) == 1
    second = make_run(tmp_path, run_id='codexa-next-run', execution_class='production', event='e9')
    deliver(second, logs)
    assert len(lines(logs[0])) == 2 and lines(logs[0])[1]['replace_id'] is None  # other run's id never reused


# 4, 10 — the viewer and status tools never notify, whatever state they render.
def test_viewer_and_status_never_notify(tmp_path, logs, monkeypatch):
    import subprocess
    run_dir = make_run(tmp_path, state='FAILED', reason='DEADLINE_BACKSTOP', execution_class='production')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail(f'unexpected subprocess {a}'))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail(f'unexpected subprocess {a}'))
    from codexa_workspace.training_status import collect, render
    render(collect(run_dir, gpu=False))
    from codexa_workspace import training_dashboard
    training_dashboard.render_text(run_dir, width=100, height=40, gpu=False)
    assert not (run_dir / 'supervisor/desktop-notification.json').exists() and not logs[0].exists()


# 7 — two competing notifier processes cannot both send.
def _compete(run_dir, desktop_log, agent_log):
    deliver(Path(run_dir), (Path(desktop_log), Path(agent_log)), desktop=Desktop(Path(desktop_log), delay=.3))


def test_competing_processes_send_once(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    context = multiprocessing.get_context('fork')
    workers = [context.Process(target=_compete, args=(str(run_dir), str(logs[0]), str(logs[1]))) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
    assert [worker.exitcode for worker in workers] == [0] * 4
    assert len(lines(logs[0])) == 1 and len(lines(logs[1])) == 1


# 8, 9, 10 — simulated deadline / RuntimeError in non-production classes: zero popups, no agent.
@pytest.mark.parametrize('execution_class', ['integration_test', 'unit_test', 'simulation', None])
@pytest.mark.parametrize('reason', ['DEADLINE_BACKSTOP', 'RuntimeError'])
def test_simulated_failures_never_reach_real_backends(tmp_path, monkeypatch, execution_class, reason):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail(f'real backend invoked: {a}'))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail(f'real agent invoked: {a}'))
    run_dir = make_run(tmp_path, run_id='fake-run', state='FAILED', reason=reason, execution_class=execution_class)
    result = notifications.deliver(run_dir, tmp_path)   # no injected backends: policy chooses
    assert result['execution_class'] != 'production'
    record = json.loads((run_dir / 'supervisor/desktop-notification.json').read_text())
    assert record['backend'] == 'mock' and record['notification_state'] == 'delivered'
    mock = lines(run_dir / 'supervisor/notification-mock.jsonl')
    assert [row['channel'] for row in mock] == ['ai', 'desktop', 'agent_window'] and reason in mock[1]['body']


def test_classification_is_trusted_config_not_run_id_text(tmp_path, monkeypatch):
    monkeypatch.delenv(notifications.DOWNGRADE_ENVIRONMENT, raising=False)
    named_real = make_run(tmp_path, run_id='codexa-934m-real', execution_class=None)
    assert notifications.execution_class(named_real)['effective'] == 'unclassified'
    fake_but_granted = make_run(tmp_path, run_id='fake-run', grant=True)
    assert notifications.execution_class(fake_but_granted)['effective'] == 'production'
    tampered = make_run(tmp_path, run_id='tampered', grant=True)
    job = json.loads((tampered / 'config/job.json').read_text())
    job['spec'] = {'changed': True}
    (tampered / 'config/job.json').write_text(json.dumps(job))
    assert notifications.execution_class(tampered)['effective'] == 'unclassified'
    monkeypatch.setenv(notifications.DOWNGRADE_ENVIRONMENT, 'unit_test')
    assert notifications.execution_class(fake_but_granted)['effective'] == 'unit_test'
    monkeypatch.setenv(notifications.DOWNGRADE_ENVIRONMENT, 'production')   # cannot upgrade
    assert notifications.execution_class(named_real)['effective'] == 'unclassified'


# 12, 13 — failures recorded accurately; failed AI never marked successful; bounded retry.
def test_failed_delivery_and_failed_ai_are_recorded_honestly(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    result = deliver(run_dir, logs, desktop=Desktop(logs[0], success=False), agent=Agent(logs[1], success=False))
    assert result['channels']['desktop']['status'] == 'failed' and result['channels']['ai']['status'] == 'failed_retryable'
    deliver(run_dir, logs, agent=Agent(logs[1], success=False))
    desktop = json.loads((run_dir / 'supervisor/desktop-notification.json').read_text())
    ai = json.loads((run_dir / 'supervisor/ai-wakeup.json').read_text())
    assert desktop['notification_attempts'] == 1 and len(lines(logs[0])) == 1   # failed popup is not retried
    assert ai['status'] == 'failed_final' and ai['submissions'] == 2 and len(lines(logs[1])) == 2
    assert 'NOT automatically' in (run_dir / 'final-report/RESUME_MANUALLY.md').read_text()
    deliver(run_dir, logs)
    assert len(lines(logs[1])) == 2   # bounded


def test_submit_failure_is_not_reported_as_requested(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    deliver(run_dir, logs, agent=Agent(logs[1], submitted=False))
    body = lines(logs[0])[0]['body']
    assert 'Analysis requested' not in body and 'could not be requested' in body


# 14 — crash after claim: at-most-once, never resent; backend unavailable before send: retried later.
def test_crash_after_claim_is_indeterminate_and_not_resent(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    context = multiprocessing.get_context('fork')
    crash = context.Process(target=lambda: deliver(run_dir, logs, desktop=Desktop(logs[0], explode=True)))
    crash.start()
    crash.join(30)
    assert crash.exitcode == 9
    assert json.loads((run_dir / 'supervisor/desktop-notification.json').read_text())['notification_state'] == 'claimed'
    deliver(run_dir, logs)
    record = json.loads((run_dir / 'supervisor/desktop-notification.json').read_text())
    assert record['notification_state'] == 'indeterminate' and not logs[0].exists()


def test_unavailable_backend_is_retried_because_nothing_was_sent(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    assert deliver(run_dir, logs, desktop=Desktop(logs[0], available=False))['channels']['desktop']['status'] == 'backend_unavailable'
    assert deliver(run_dir, logs)['channels']['desktop']['status'] == 'delivered'
    assert len(lines(logs[0])) == 1 and len(lines(logs[1])) == 1


def test_legacy_state_from_old_deliver_is_honoured(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    legacy = dict(event_id='e1', channels=dict(desktop=dict(status='delivered', attempts=[{}]),
                                               ai=dict(status='delivered', attempts=[{}], session_id='old'),
                                               desktop_analysis=dict(status='delivered', attempts=[{}])))
    (run_dir / 'supervisor/notification-state.json').write_text(json.dumps(legacy))
    result = deliver(run_dir, logs)
    assert not logs[0].exists() and not logs[1].exists()
    assert result['channels']['desktop']['status'] == 'delivered' and result['channels']['ai']['session_id'] == 'old'


def test_dry_run_consumes_nothing(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    result = notifications.deliver(run_dir, tmp_path, dry_run=True)
    assert result['dry_run'] and not (run_dir / 'supervisor/desktop-notification.json').exists()
    assert deliver(run_dir, logs)['channels']['desktop']['status'] == 'delivered' and len(lines(logs[0])) == 1


# Interactive agent window: one per terminal event, after the AI outcome; read-only + on-request; mocks only.
class Window:
    name = 'test-window'

    def __init__(self, log: Path, *, available=True, success=True):
        self.log, self.ok, self.success = log, available, success

    def available(self):
        return (True, 'ok') if self.ok else (False, 'no graphical display in this environment')

    def open(self, *, unit, title, command, cwd):
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(unit=unit, title=title, command=command, cwd=str(cwd))) + '\n')
        return dict(success=self.success, exit_code=0 if self.success else 1, unit=unit)


def test_agent_window_resumes_analysis_session_once(tmp_path, logs):
    run_dir = make_run(tmp_path, state='EARLY_STOPPED', reason='MEMORY_HEADROOM_GATE', execution_class='production')
    windows = tmp_path / 'windows.jsonl'
    for _ in range(3):
        result = deliver(run_dir, logs, window_backend=Window(windows))
    opened = lines(windows)
    assert len(opened) == 1 and result['channels']['agent_window']['status'] == 'opened'
    command = opened[0]['command']
    assert command[1:3] == ['resume', 's-1'] and command[3:7] == ['-s', 'read-only', '-a', 'on-request']
    assert 'danger-full-access' not in command and opened[0]['unit'].startswith('codexa-agent-codexa-real-run-')
    assert 'EARLY_STOPPED' in opened[0]['title'] and len(lines(logs[0])) == 1


def test_agent_window_after_failed_analysis_starts_fresh_with_prompt(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    windows = tmp_path / 'windows.jsonl'
    deliver(run_dir, logs, agent=Agent(logs[1], success=False), window_backend=Window(windows))
    assert not windows.exists()                       # one AI retry remains: not final yet
    deliver(run_dir, logs, agent=Agent(logs[1], success=False), window_backend=Window(windows))
    command = lines(windows)[0]['command']
    assert 'resume' not in command and command[-1].startswith(notifications.wakeup_prompt(run_dir, 'codex')[:40])


def test_claude_window_command_and_disabled_window(tmp_path, logs):
    command = notifications.agent_window_command('claude', tmp_path, 'abc', None)
    assert command[1:] == ['--resume', 'abc', '--permission-mode', 'default']
    run_dir = make_run(tmp_path, execution_class='production')
    launch = json.loads((run_dir / 'supervisor/launch.json').read_text())
    (run_dir / 'supervisor/launch.json').write_text(json.dumps(dict(launch, agent_window=False)))
    windows = tmp_path / 'windows.jsonl'
    assert deliver(run_dir, logs, window_backend=Window(windows))['channels']['agent_window']['status'] == 'disabled'
    assert not windows.exists()


def test_agent_window_waits_for_display_then_opens_once(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    windows = tmp_path / 'windows.jsonl'
    first = deliver(run_dir, logs, window_backend=Window(windows, available=False))
    assert first['channels']['agent_window']['status'] == 'backend_unavailable' and not windows.exists()
    deliver(run_dir, logs, window_backend=Window(windows, success=False))
    deliver(run_dir, logs, window_backend=Window(windows))
    record = json.loads((run_dir / 'supervisor/agent-window.json').read_text())
    assert record['status'] == 'failed' and len(lines(windows)) == 1   # a failed open is not retried automatically


def test_injected_backends_never_open_a_real_window(tmp_path, logs, monkeypatch):
    import subprocess
    monkeypatch.delenv(notifications.DOWNGRADE_ENVIRONMENT, raising=False)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail(f'real window opened: {a}'))
    run_dir = make_run(tmp_path, execution_class='production')
    deliver(run_dir, logs)                             # production class, but desktop/ai are injected
    record = json.loads((run_dir / 'supervisor/agent-window.json').read_text())
    assert record['backend'] == 'mock' and record['status'] == 'opened'


def test_kitty_backend_runs_in_its_own_user_unit_without_secrets(tmp_path, monkeypatch):
    import subprocess
    seen = {}
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    monkeypatch.setattr(notifications.shutil, 'which', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(subprocess, 'run', lambda argv, **k: seen.setdefault('argv', argv) and
                        subprocess.CompletedProcess(argv, 0, '', ''))
    result = notifications.KittyAgentWindowBackend().open(unit='codexa-agent-x', title='t', cwd=tmp_path,
                                                          command=['codex', 'resume', 's'])
    argv = seen['argv']
    assert argv[:5] == ['systemd-run', '--user', '--collect', '--no-block', '--unit=codexa-agent-x']
    assert '--setenv=WAYLAND_DISPLAY=wayland-0' in argv and not any('sk-secret' in part for part in argv)
    assert 'initial_window_width=100c' in argv and 'initial_window_height=22c' in argv
    assert argv[-3:] == ['codex', 'resume', 's'] and result['success']


def test_manual_open_agent_records_reopen(tmp_path, logs):
    run_dir = make_run(tmp_path, execution_class='production')
    windows = tmp_path / 'windows.jsonl'
    deliver(run_dir, logs, window_backend=Window(windows))
    notifications.open_agent_window(run_dir, tmp_path, window_backend=Window(windows))
    record = json.loads((run_dir / 'supervisor/agent-window.json').read_text())
    assert len(lines(windows)) == 2 and record['history'][-1]['outcome'] == 'manual_open'
    assert 'manual' in lines(windows)[1]['unit'] and lines(windows)[1]['command'][1:3] == ['resume', 's-1']
