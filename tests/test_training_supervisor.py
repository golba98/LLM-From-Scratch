"""Supervisor, status and exactly-once wakeup against a fake trainer and stub agent CLIs.

No GPU, no real training, no API spend: `codex`, `claude` and `notify-send` are
stub executables placed first on PATH that record each invocation.
"""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codexa_workspace import training_supervisor as supervisor
from codexa_workspace import training_wakeup as wakeup
from codexa_workspace import training_notifications as notifications
from codexa_workspace.training_status import collect, render

FAKE = ROOT / 'tests/fixtures/fake_trainer.py'
SCRIPT = ROOT / 'scripts/training_supervisor.py'
WORKSPACE = ROOT.parent
# The same source-first search path as tools/codexa-dev: the startup guard checks it before
# the supervisor runs, so a development selection never silently resolves installed wheels.
SOURCE_FIRST = os.pathsep.join([*(str(WORKSPACE / name / 'src') for name in (
    'LLM-Architecture', 'LLM-Tokenizer', 'LLM-Data', 'LLM-Training', 'LLM-Inference', 'LLM-Memory',
    'LLM-Specialist')), str(ROOT / 'src'), str(ROOT)])
STARTED: list[subprocess.Popen] = []


def explicit_development_env(base: dict | None = None) -> dict:
    """Explicit selection for supervisor subprocesses. The real startup guard still validates the
    real workspace (pinned, committed, source-first) and fails closed otherwise."""
    return dict(os.environ if base is None else base, CODEXA_TRAINING_ENVIRONMENT='development',
                PYTHONPATH=SOURCE_FIRST)


@pytest.fixture(autouse=True)
def explicit_selection(monkeypatch, tmp_path):
    """In-process launcher calls read the selection too; every spawned process is reaped."""
    monkeypatch.setenv('CODEXA_TRAINING_ENVIRONMENT', 'development')
    yield
    while STARTED:
        process = STARTED.pop()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=30)
    for state_file in tmp_path.glob('**/supervisor/state.json'):
        trainer = json.loads(state_file.read_text()).get('trainer') or {}
        if trainer.get('pid') and supervisor.process_start_ticks(trainer['pid']) == trainer.get('start_ticks'):
            os.kill(trainer['pid'], signal.SIGKILL)


def wait_for(predicate, process: subprocess.Popen, *, timeout: float = 20, what: str = 'condition'):
    """Bounded polling that fails immediately, with the supervisor's stderr, if it exits first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=30)
            pytest.fail(f'supervisor exited ({process.returncode}) before {what}: {stderr[-2000:]}')
        time.sleep(.05)
    pytest.fail(f'timed out after {timeout} s waiting for {what}')
STUB = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
record = dict(name=name, argv=sys.argv[1:], stdin=sys.stdin.read() if name != 'notify-send' else None,
              wakeup_env=os.environ.get('CODEXA_WAKEUP_AGENT'))
with open(os.environ['CODEXA_TEST_STUB_CALLS'], 'a') as stream:
    stream.write(json.dumps(record) + '\\n')
if os.environ.get('CODEXA_TEST_STUB_FAIL') == name:
    sys.exit(1)
if name == 'codex':
    output = sys.argv[sys.argv.index('-o') + 1]
    Path(output).write_text('stub codex analysis')
    print(json.dumps(dict(type='thread.started', thread_id='stub-thread')))
if name == 'claude':
    print(json.dumps(dict(result='stub claude analysis', is_error=False, session_id='stub-session')))
'''


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('codex', 'claude', 'notify-send', 'kitty'):
        path = bin_dir / name
        path.write_text(STUB)
        path.chmod(0o755)
    calls_file = tmp_path / 'calls.jsonl'
    # Fail closed: the real agent CLIs live outside this PATH, so a broken stub can never
    # fall through to an authenticated, billed invocation.
    monkeypatch.setenv('PATH', f'{bin_dir}:/usr/bin:/bin')
    monkeypatch.setenv('CODEXA_TEST_STUB_CALLS', str(calls_file))
    monkeypatch.delenv('CODEXA_WAKEUP_AGENT', raising=False)
    for name in ('codex', 'claude'):
        assert shutil.which(name) == str(bin_dir / name)
    subprocess.run(['notify-send', 'sanity'], check=True, timeout=30)
    assert calls(calls_file, 'notify-send'), 'stub did not record; refusing to continue'
    calls_file.unlink()
    return calls_file


def calls(path: Path, name: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in rows if name is None or row['name'] == name]


POLICY = dict(poll_seconds=.2, heartbeat_timeout=2, kill_grace=1, deadline_grace=1, disk_reserve_bytes=0,
              stall_seconds=dict(TRAINING=1.5, STARTING=5))


def ready_run(tmp_path: Path, scenario: str, *, provider='codex', policy=None, **trainer) -> Path:
    run_dir = tmp_path / 'outputs/training/fake-run'
    state = supervisor.RunState(run_dir)
    state.transition('VALIDATING')
    state.transition('READY')
    command = [sys.executable, str(FAKE), '--run-dir', str(run_dir), '--scenario', scenario]
    for key, value in trainer.items():
        command += [f"--{key.replace('_', '-')}", str(value)]
    supervisor.atomic_document(run_dir / 'supervisor/launch.json', dict(run_id='fake-run', command=command,
        cwd=str(tmp_path), environment={}, wakeup=dict(provider=provider), deadline_seconds=60,
        execution_class='integration_test',
        policy={**POLICY, **(policy or {})}))
    return run_dir


def supervise(run_dir: Path, *, timeout=60, wait=True):
    process = subprocess.Popen([sys.executable, str(SCRIPT), 'run', '--run-dir', str(run_dir)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=explicit_development_env())
    STARTED.append(process)
    if wait:
        _, stderr = process.communicate(timeout=timeout)
        assert process.returncode == 0, stderr
    return process


def events(run_dir: Path, kind: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / 'supervisor/events.jsonl').read_text().splitlines()]
    return [row for row in rows if kind is None or row['event'] == kind]


def completion(run_dir: Path) -> dict:
    return json.loads((run_dir / 'supervisor/completion.json').read_text())


def mock(run_dir: Path, channel: str) -> list[dict]:
    path = run_dir / 'supervisor/notification-mock.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row['channel'] == channel]


def assert_silent_once(run_dir: Path, stubs: Path) -> None:
    """Test runs never reach real (or stub) notify-send/agents; the mock records one of each."""
    assert calls(stubs) == []
    assert len(mock(run_dir, 'desktop')) == 1 and len(mock(run_dir, 'ai')) == 1
    assert len(mock(run_dir, 'agent_window')) <= 1   # the interactive window is mocked too
    record = json.loads((run_dir / 'supervisor/desktop-notification.json').read_text())
    assert record['execution_class'] == 'integration_test' and record['backend'] == 'mock'


def test_success_records_one_mock_notification_only_at_terminal(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', steps=12, step_seconds=.25)
    process = supervise(run_dir, wait=False)

    def running():
        status = collect(run_dir, gpu=False)
        return status if status['state'] == 'RUNNING' and (status['optimizer_steps'] or 0) >= 3 else None
    status = wait_for(running, process, what='three optimizer steps')
    assert status['state'] == 'RUNNING' and mock(run_dir, 'desktop') == []  # nothing during progress
    process.communicate(timeout=60)
    done = completion(run_dir)
    assert done['terminal_state'] == 'COMPLETED' and done['final_checkpoint_verification'] == 'verified'
    assert done['optimizer_steps'] == 12 and done['validation']['baseline'] == 3.2
    assert_silent_once(run_dir, stubs)
    popup = mock(run_dir, 'desktop')[0]
    assert popup['summary'] == 'Codexa — Training completed' and 'final checkpoint verified' in popup['body']
    assert 'codex' in mock(run_dir, 'ai')[0]['command']
    handoff = (run_dir / 'supervisor/ai-handoff.md').read_text()
    assert 'COMPLETED' in handoff and 'Optimizer updates: 12 / ' in handoff and len(handoff) < 6000
    # Duplicate delivery, reconcile and a supervisor rerun are idempotent.
    notifications.deliver(run_dir, tmp_path)
    assert supervisor.reconcile_all(tmp_path, deliver=notifications.deliver)[0]['action'] == 'nothing_to_do'
    supervise(run_dir)
    assert_silent_once(run_dir, stubs)
    assert len(events(run_dir, 'spawn')) == 1
    assert collect(run_dir, gpu=False)['notification'] == dict(desktop='delivered', ai='delivered', agent_window='opened')


@pytest.mark.parametrize('scenario,state,reason', [
    ('crash', 'FAILED', 'RuntimeError'),
    ('nonfinite', 'FAILED', 'FloatingPointError'),
    ('checkpoint_failure', 'FAILED', 'CHECKPOINT_FAILURE'),
    ('early_stop', 'EARLY_STOPPED', 'GRADIENT_NORM_GUARD'),
    ('corrupt_final', 'FAILED', 'FINAL_CHECKPOINT_UNVERIFIED'),
    ('no_report', 'FAILED', 'TRAINER_EXITED_WITHOUT_REPORT'),
])
def test_terminal_classification(tmp_path, stubs, scenario, state, reason):
    run_dir = ready_run(tmp_path, scenario)
    supervise(run_dir)
    done = completion(run_dir)
    assert (done['terminal_state'], done['reason']) == (state, reason)
    assert json.loads((run_dir / 'supervisor/state.json').read_text())['state'] == state
    assert_silent_once(run_dir, stubs)
    body = mock(run_dir, 'desktop')[0]['body']
    if scenario in ('crash', 'checkpoint_failure'):
        assert done['last_verified_checkpoint']['optimizer_step'] == 2
        assert 'Last verified checkpoint: step 2 preserved' in body
    if scenario == 'corrupt_final':
        assert done['final_checkpoint_verification'] == 'failed' and 'NOT verified' in body


@pytest.mark.parametrize('scenario,reason,policy,trainer', [
    ('stall', 'STALLED', {}, {}),
    ('heartbeat_lost', 'HEARTBEAT_LOST', dict(stall_seconds=dict(TRAINING=60)), {}),
    ('overrun', 'DEADLINE_BACKSTOP', dict(stall_seconds=dict(TRAINING=60)), dict(budget_seconds=1)),
])
def test_process_level_watchdogs_terminate_and_fail_silently(tmp_path, stubs, scenario, reason, policy, trainer):
    run_dir = ready_run(tmp_path, scenario, policy=policy, **trainer)
    started = time.monotonic()
    supervise(run_dir)
    done = completion(run_dir)
    assert (done['terminal_state'], done['reason']) == ('FAILED', reason)
    assert done['exit_status'] == dict(signal=9)  # SIGTERM ignored, SIGKILL after grace
    assert time.monotonic() - started < 30
    assert_silent_once(run_dir, stubs)


def test_disk_reserve_watchdog_stops_gracefully(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', steps=200, step_seconds=.1, policy=dict(disk_reserve_bytes=1 << 62))
    supervise(run_dir)
    done = completion(run_dir)
    assert (done['terminal_state'], done['reason']) == ('EARLY_STOPPED', 'DISK_RESERVE_WATCHDOG')
    assert done['exit_status'] == dict(code=0) and done['last_verified_checkpoint'] is not None
    assert_silent_once(run_dir, stubs)


def test_stop_signal_forwards_and_defers_ai_until_reconcile(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', steps=200, step_seconds=.1)
    process = supervise(run_dir, wait=False)
    wait_for(lambda: (run_dir / 'metrics/progress.json').exists(), process, what='trainer progress')
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=30)
    done = completion(run_dir)
    assert done['terminal_state'] == 'INTERRUPTED' and done['reason'] == 'INTERRUPTED_SIGNAL_15'
    assert mock(run_dir, 'ai') == [] and len(mock(run_dir, 'desktop')) == 1
    assert 'deferred' in mock(run_dir, 'desktop')[0]['body']
    assert json.loads((run_dir / 'supervisor/notification-state.json').read_text())['channels']['ai']['status'] == 'deferred'
    result = supervisor.reconcile_all(tmp_path, deliver=notifications.deliver)
    assert result[0]['ai'] == 'delivered'
    supervisor.reconcile_all(tmp_path, deliver=notifications.deliver)
    assert_silent_once(run_dir, stubs)  # one popup in total, one AI request in total


def test_supervisor_restart_reattaches_without_duplicate_trainer(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', steps=30, step_seconds=.15)
    first = supervise(run_dir, wait=False)
    wait_for(lambda: json.loads((run_dir / 'supervisor/state.json').read_text()).get('trainer'), first,
             what='trainer spawn')
    trainer = json.loads((run_dir / 'supervisor/state.json').read_text())['trainer']
    first.kill()
    first.communicate(timeout=30)
    assert supervisor.process_start_ticks(trainer['pid']) == trainer['start_ticks']  # trainer survives
    supervise(run_dir)
    assert len(events(run_dir, 'spawn')) == 1 and len(events(run_dir, 'reattach')) == 1
    done = completion(run_dir)
    assert done['terminal_state'] == 'COMPLETED' and done['exit_status'] is None
    assert_silent_once(run_dir, stubs)


def test_reconcile_after_host_restart_marks_interrupted_once(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success')
    state = supervisor.RunState(run_dir)
    state.transition('RUNNING', trainer=dict(pid=999999, boot_id='previous-boot', start_ticks=1, started_epoch=time.time()))
    first = supervisor.reconcile_all(tmp_path, deliver=notifications.deliver)
    assert first[0]['action'] == 'finalized'
    done = completion(run_dir)
    assert (done['terminal_state'], done['reason']) == ('INTERRUPTED', 'HOST_RESTART')
    supervisor.reconcile_all(tmp_path, deliver=notifications.deliver)
    assert_silent_once(run_dir, stubs)
    assert len(events(run_dir, 'spawn')) == 0


def test_claude_adapter_command_is_restricted(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', provider='claude')
    supervise(run_dir)
    assert calls(stubs) == [] and mock(run_dir, 'ai')[0]['provider'] == 'claude'
    argv = mock(run_dir, 'ai')[0]['command']
    assert '--permission-mode' in argv and 'dontAsk' in argv and 'Bash' not in argv
    assert 'Write(final-report/**)' in argv and '--max-budget-usd' in argv


def test_illegal_transitions_and_spawn_refusals(tmp_path, stubs, monkeypatch):
    state = supervisor.RunState(tmp_path / 'outputs/training/x')
    with pytest.raises(ValueError):
        state.transition('RUNNING')
    run_dir = ready_run(tmp_path, 'success')
    supervisor.RunState(run_dir).transition('BLOCKED')
    result = subprocess.run([sys.executable, str(SCRIPT), 'run', '--run-dir', str(run_dir)], cwd=ROOT,
                            capture_output=True, text=True, env=explicit_development_env(), timeout=60)
    assert result.returncode != 0 and 'Refusing to spawn' in result.stderr
    monkeypatch.setenv('CODEXA_WAKEUP_AGENT', '1')
    result = subprocess.run([sys.executable, str(SCRIPT), 'launch', '--run-dir', str(run_dir)], cwd=ROOT,
                            capture_output=True, text=True, env=explicit_development_env(), timeout=60)
    assert result.returncode != 0 and 'wakeup analysis agent' in result.stderr
    assert not (run_dir / 'supervisor/completion.json').exists()


@pytest.mark.skipif(shutil.which('systemd-run') is None or subprocess.run(
    ['systemctl', '--user', 'is-system-running'], capture_output=True, text=True).stdout.strip() not in ('running', 'degraded'),
    reason='requires a systemd user manager')
def test_detached_systemd_unit_survives_launching_shell_exit(tmp_path, stubs):
    run_dir = ready_run(tmp_path, 'success', steps=15, step_seconds=.2)
    unit = f'codexa-test-{os.getpid()}-{int(time.time())}'
    # The unit does NOT inherit the pytest downgrade env: silence must come from launch.json alone.
    environment = explicit_development_env({key: os.environ[key] for key in ('PATH', 'CODEXA_TEST_STUB_CALLS', 'HOME')})
    command = supervisor.systemd_launch(run_dir, ROOT, unit=unit, python=sys.executable, script=SCRIPT,
                                        environment=environment, dry_run=True)
    # The launching shell exits immediately; the unit belongs to the user manager.
    subprocess.run(['bash', '-c', ' '.join(f"'{part}'" for part in command) + ' >/dev/null 2>&1 & exit 0'], check=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not mock(run_dir, 'ai'):
        # A unit that already ended without a notification will never produce one: stop waiting.
        if time.monotonic() > deadline - 55 and supervisor.unit_active(unit + '.service') in ('inactive', 'failed'):
            time.sleep(.5)
            break
        time.sleep(.2)
    done = completion(run_dir)
    assert done['terminal_state'] == 'COMPLETED'
    assert json.loads((run_dir / 'supervisor/state.json').read_text())['supervisor']['unit'] == unit
    assert_silent_once(run_dir, stubs)


def load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('training_supervisor_script', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('scenario,allowed', [('early_stop', False), ('crash', False)])
def test_resume_refuses_quality_and_failure_stops(tmp_path, stubs, scenario, allowed):
    run_dir = ready_run(tmp_path, scenario)
    supervise(run_dir)
    with pytest.raises(SystemExit, match='Not resumable'):
        load_script().resume(run_dir, provider='codex', confirm=run_dir.name)


def test_resume_of_resource_stop_is_gated_and_leaves_files_when_blocked(tmp_path, stubs, monkeypatch):
    run_dir = ready_run(tmp_path, 'early_stop')
    supervise(run_dir)
    done = json.loads((run_dir / 'supervisor/completion.json').read_text())
    done['reason'] = 'MEMORY_HEADROOM_GATE'
    (run_dir / 'supervisor/completion.json').write_text(json.dumps(done))
    script = load_script()
    monkeypatch.setattr(script, 'resume_gates', lambda d: dict(passed=False, failed=['gpu_vram'], gates=[
        dict(name='gpu_vram', passed=False, detail=dict(free_mib=13853))]))
    monkeypatch.setattr(script, 'launch', lambda *a, **k: pytest.fail('must not launch when gates fail'))
    assert script.resume(run_dir, provider='codex', confirm=run_dir.name) == 2
    assert (run_dir / 'supervisor/completion.json').exists() and not (run_dir / 'supervisor/segments').exists()
    with pytest.raises(SystemExit, match='exact run id'):
        script.resume(run_dir, provider='codex', confirm='other')


def test_every_start_uses_the_rich_dashboard_as_the_trainer_viewer(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('launcher', SCRIPT)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    run_dir = tmp_path / 'run'
    (run_dir / 'supervisor').mkdir(parents=True)
    for resume in (False, True):
        launch = launcher.write_launch(run_dir, provider='codex', resume=resume)
        command = json.loads(launch['environment']['LLM_TRAINING_VIEWER_COMMAND'])
        assert command[1].endswith('scripts/training_dashboard.py') and command[2] == '--attach'


def test_extend_only_a_deadline_stop_before_the_cap_and_restore_budget_when_blocked(tmp_path, stubs, monkeypatch):
    run_dir = ready_run(tmp_path, 'early_stop')
    supervise(run_dir)
    script = load_script()
    with pytest.raises(SystemExit, match='Not extendable'):
        script.extend(run_dir, provider='codex', confirm=run_dir.name, seconds=2400, statement='finish it')
    done = json.loads((run_dir / 'supervisor/completion.json').read_text())
    done.update(terminal_state='COMPLETED', reason='TRAINING_DEADLINE', optimizer_steps=5, max_steps=8,
                final_checkpoint_verification='verified')
    (run_dir / 'supervisor/completion.json').write_text(json.dumps(done))
    state = json.loads((run_dir / 'supervisor/state.json').read_text())
    (run_dir / 'supervisor/state.json').write_text(json.dumps(dict(state, state='COMPLETED')))
    budget = dict(seconds=100, started_utc=time.time() - 200, deadline_utc=time.time() - 100, reserve_seconds=10,
                  remaining_seconds=5, elapsed_seconds=95, last_recorded_utc=time.time() - 100)
    (run_dir / 'metrics').mkdir(exist_ok=True)
    (run_dir / 'metrics/budget.json').write_text(json.dumps(budget))
    with pytest.raises(SystemExit, match='statement'):
        script.extend(run_dir, provider='codex', confirm=run_dir.name, seconds=2400, statement=' ')
    monkeypatch.setattr(script, 'resume_gates', lambda d: dict(passed=False, failed=['gpu_vram'], gates=[]))
    monkeypatch.setattr(script, 'launch', lambda *a, **k: pytest.fail('must not launch when gates fail'))
    assert script.extend(run_dir, provider='codex', confirm=run_dir.name, seconds=2400, statement='finish it') == 2
    assert json.loads((run_dir / 'metrics/budget.json').read_text()) == budget
    assert not (run_dir / 'config/budget-extensions.jsonl').exists()
    seen = {}
    monkeypatch.setattr(script, 'resume_gates', lambda d: dict(passed=True, failed=[], gates=[]))
    monkeypatch.setattr(script, 'launch', lambda d, **k: seen.update(k) or 0)
    monkeypatch.setattr(script.supervisor, 'recovery_role', lambda d, role: dict(optimizer_step=5))
    (run_dir / 'metrics/report.json').write_text('{}')
    assert script.extend(run_dir, provider='codex', confirm=run_dir.name, seconds=2400, statement='finish it') == 0
    extended = json.loads((run_dir / 'metrics/budget.json').read_text())
    assert 2390 < extended['deadline_utc'] - time.time() <= 2400 and extended['started_utc'] == budget['started_utc']
    grant = json.loads((run_dir / 'config/budget-extensions.jsonl').read_text())
    assert grant['statement'] == 'finish it' and grant['from_step'] == 5 and grant['previous_budget'] == budget
    assert seen['resume'] is True and not (run_dir / 'supervisor/completion.json').exists()


def test_supervisor_without_selection_fails_closed_quickly(tmp_path, stubs):
    """No explicit selection: the startup guard refuses before the supervisor spawns anything."""
    run_dir = ready_run(tmp_path, 'success')
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('CODEXA_TRAINING_ENVIRONMENT', 'PYTHONPATH')}
    started = time.monotonic()
    result = subprocess.run([sys.executable, str(SCRIPT), 'run', '--run-dir', str(run_dir)], cwd=ROOT,
                            capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 2 and 'Explicit --environment' in result.stderr
    assert time.monotonic() - started < 10
    assert 'trainer' not in json.loads((run_dir / 'supervisor/state.json').read_text())
    assert not (run_dir / 'supervisor/completion.json').exists()
