"""Sole owner of terminal-outcome notifications: one desktop popup per real run, none for tests.

Execution class comes from trusted launch configuration (or, for runs launched before
the field existed, a self-verified approved job grant) — never from run-id text.
Only ``production`` runs may reach the real ``notify-send`` and agent CLIs; every
other class exercises the same policy against a recording mock backend.

Delivery semantics. Desktop: at-most-once per terminal event. A durable ``claimed``
record is fsynced before the popup is shown; a crash between the popup and the
result write leaves ``claimed`` → later read as ``indeterminate`` and never resent.
Nothing is resent after any invocation of the backend; only a pre-flight
``backend_unavailable`` (nothing was sent) may be retried. AI wakeup: its own record,
at most two submissions per event, success only on exit 0 plus a non-empty output.
Agent window: once the AI outcome is final, at most one interactive codex/claude Kitty
window per event (resuming the analysis session; read-only sandbox, on-request approvals),
in its own systemd user unit so it outlives the supervisor.
The viewer, status tool and trainer never call into this module.
"""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from llm_training.session import atomic_document

from .training_supervisor import read_json, process_start_ticks, boot_id
from . import run_reporting
from .training_wakeup import (agent_environment, handoff_text, prompt_text, provider_command,
                              resume_command, session_identity)

CLASSES = ('production', 'integration_test', 'unit_test', 'simulation', 'unclassified')
DOWNGRADE_ENVIRONMENT = 'CODEXA_EXECUTION_CLASS'
SUMMARY_FORMAT = 'codexa-notification-summary-v2'
MAX_AI_SUBMISSIONS = 2
TERMINAL_AI = ('delivered', 'failed_final', 'suppressed', 'disabled')


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def execution_class(run_dir: Path) -> dict:
    """Trusted classification; the environment may only downgrade, never upgrade."""
    run_dir = Path(run_dir)
    launch = read_json(run_dir / 'supervisor/launch.json') or {}
    if 'execution_class' in launch:
        declared = launch['execution_class'] if launch['execution_class'] in CLASSES else 'unclassified'
        source = 'launch.json'
    else:
        declared, source = 'unclassified', 'no trusted classification'
        job = read_json(run_dir / 'config/job.json')
        grant = read_json(run_dir / 'config/approval.json')
        if job and grant:
            from .foundation_training import fingerprint, APPROVAL_FORMAT
            unsigned = {key: value for key, value in job.items() if key != 'job_sha256'}
            if (fingerprint(unsigned) == job.get('job_sha256') and grant.get('format') == APPROVAL_FORMAT
                    and grant.get('approved') is True and grant.get('job_sha256') == job['job_sha256']):
                declared, source = 'production', 'approved job grant (legacy launch.json)'
    requested = os.environ.get(DOWNGRADE_ENVIRONMENT)
    effective, downgraded = declared, False
    if requested in CLASSES and requested != 'production' and declared == 'production':
        effective, downgraded = requested, True
    elif requested in CLASSES and requested != 'production':
        effective = requested if declared == 'unclassified' else declared
    return dict(declared=declared, effective=effective, source=source, downgraded=downgraded)


# ---------------------------------------------------------------- backends

class NotifySendBackend:
    """Real GNOME/freedesktop notifications via libnotify's CLI."""
    name = 'notify-send'

    def available(self) -> tuple[bool, str]:
        if shutil.which('notify-send') is None:
            return False, 'notify-send not installed'
        address = os.environ.get('DBUS_SESSION_BUS_ADDRESS', '')
        socket = address.split('unix:path=', 1)[1].split(',')[0] if 'unix:path=' in address else \
            str(Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')) / 'bus')
        if not Path(socket).exists():
            return False, f'session bus socket missing: {socket}'
        return True, 'ok'

    def send(self, summary: str, body: str, *, urgency: str, replace_id: int | None) -> dict:
        command = ['notify-send', '-p', '--app-name=Codexa', f'--urgency={urgency}']
        if replace_id:
            command += ['-r', str(replace_id)]
        completed = subprocess.run(command + [summary, body], capture_output=True, text=True, timeout=30)
        identifier = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ''
        return dict(success=completed.returncode == 0, exit_code=completed.returncode,
                    notification_id=int(identifier) if identifier.isdigit() else None,
                    stderr_tail=completed.stderr.strip()[-300:])


class RecordingDesktopBackend:
    """Mock: records the would-be popup; never touches the desktop."""
    name = 'mock'

    def __init__(self, run_dir: Path) -> None:
        self.log = Path(run_dir) / 'supervisor/notification-mock.jsonl'

    def available(self) -> tuple[bool, str]:
        return True, 'mock'

    def send(self, summary: str, body: str, *, urgency: str, replace_id: int | None) -> dict:
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(at=now(), channel='desktop', summary=summary, body=body,
                                         urgency=urgency, replace_id=replace_id)) + '\n')
        return dict(success=True, exit_code=0, notification_id=None, mock=True)


class CliAgentBackend:
    """Real agent CLI, submitted asynchronously so the popup can truthfully say 'requested'."""
    name = 'cli'

    def submit(self, provider: str, run_dir: Path, root: Path, prompt: str) -> dict:
        command, options = provider_command(provider, run_dir, root)
        report = Path(run_dir) / 'final-report'
        report.mkdir(parents=True, exist_ok=True)
        if shutil.which(command[0]) is None:
            return dict(submitted=False, error=f'{command[0]} CLI not found on PATH', command=command)
        stdout = (report / f'{provider}-events.jsonl').open('w')
        stderr = (report / f'{provider}-stderr.log').open('w')
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       cwd=options['cwd'], env=agent_environment(), text=True)
            process.stdin.write(prompt)
            process.stdin.close()
        except OSError as error:
            return dict(submitted=False, error=f'{type(error).__name__}: {error}', command=command)
        finally:
            stdout.close()
            stderr.close()
        time.sleep(3)
        code = process.poll()
        if code not in (None, 0):
            return dict(submitted=False, error=f'exited immediately with {code}', exit_code=code, command=command)
        return dict(submitted=True, pid=process.pid, start_ticks=process_start_ticks(process.pid),
                    command=command, output=str(options['output']), _process=process)

    def wait(self, provider: str, run_dir: Path, handle: dict, *, timeout: float = 2700) -> dict:
        process = handle['_process']
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return dict(success=False, error=f'timeout after {timeout}s')
        report = Path(run_dir) / 'final-report'
        stdout = (report / f'{provider}-events.jsonl').read_text()
        output = Path(handle['output'])
        result = dict(exit_code=code, session_id=session_identity(provider, stdout),
                      stderr_tail=(report / f'{provider}-stderr.log').read_text().strip()[-800:])
        if provider == 'claude':
            try:
                value = json.loads(stdout.strip().splitlines()[-1])
                result.update(is_error=value.get('is_error'), total_cost_usd=value.get('total_cost_usd'))
                if not output.exists() and isinstance(value.get('result'), str):
                    output.write_text(value['result'])
            except (json.JSONDecodeError, IndexError):
                result['is_error'] = True
        result['success'] = (code == 0 and output.exists() and output.stat().st_size > 0 and not result.get('is_error'))
        return result


class RecordingAgentBackend:
    """Mock: records the would-be invocation; no agent process is ever started."""
    name = 'mock'

    def __init__(self, run_dir: Path) -> None:
        self.log = Path(run_dir) / 'supervisor/notification-mock.jsonl'

    def submit(self, provider: str, run_dir: Path, root: Path, prompt: str) -> dict:
        command, _ = provider_command(provider, run_dir, root)
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(at=now(), channel='ai', provider=provider, command=command,
                                         prompt_chars=len(prompt))) + '\n')
        return dict(submitted=True, mock=True, command=command)

    def wait(self, provider: str, run_dir: Path, handle: dict, *, timeout: float = 0) -> dict:
        return dict(success=True, mock=True, session_id=None)


# ---------------------------------------------------------------- interactive agent window

# Non-secret variables only: these travel on the systemd-run command line. CLIs use their own stored login.
WINDOW_ENVIRONMENT = ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'SHELL', 'WAYLAND_DISPLAY', 'DISPLAY',
                      'XDG_RUNTIME_DIR', 'XDG_SESSION_TYPE', 'XDG_CURRENT_DESKTOP', 'DBUS_SESSION_BUS_ADDRESS')
WINDOW_SIZE = ('initial_window_width=100c', 'initial_window_height=22c')


def run_summary(run_dir: Path, completion: dict | None = None) -> dict | None:
    """Job-type-aware summary (run_reporting); None falls back to the legacy foundation-only text."""
    try:
        return run_reporting.build_summary(Path(run_dir), completion)
    except Exception:  # noqa: BLE001 - reporting must never block a terminal notification
        return None


def wakeup_prompt(run_dir: Path, provider: str, completion: dict | None = None) -> str:
    summary = run_summary(run_dir, completion)
    return run_reporting.prompt_text(run_dir, provider, summary) if summary else prompt_text(run_dir, provider)


def window_prompt(run_dir: Path, provider: str, completion: dict, session_id: str | None, ai_status: str) -> str | None:
    """None when resuming the analysis session (its transcript is the context)."""
    if session_id:
        return None
    if ai_status == 'delivered':
        return (f"The Codexa run {completion['run_id']} ended ({completion['terminal_state']}: {completion.get('reason')}). "
                f"Read {Path(run_dir) / 'final-report'} and {Path(run_dir) / 'supervisor/completion.json'}, "
                "summarise the outcome briefly, then wait for my instructions.")
    return wakeup_prompt(run_dir, provider, completion)


def agent_window_command(provider: str, root: Path, session_id: str | None, prompt: str | None) -> list[str]:
    """Interactive session: read-only sandbox and on-request approvals (overrides any permissive user config)."""
    executable = shutil.which(provider) or provider
    if provider == 'codex':
        command = [executable, 'resume', session_id] if session_id else [executable]
        command += ['-s', 'read-only', '-a', 'on-request', '-C', str(root)]
    else:
        command = [executable] + (['--resume', session_id] if session_id else []) + ['--permission-mode', 'default']
    return command + ([prompt] if prompt else [])


class KittyAgentWindowBackend:
    """A Kitty window in its own systemd user unit, so it outlives the supervisor (KillMode=mixed)."""
    name = 'kitty'

    def available(self) -> tuple[bool, str]:
        missing = [binary for binary in ('kitty', 'systemd-run') if shutil.which(binary) is None]
        if missing:
            return False, ', '.join(missing) + ' not installed'
        if not (os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY')):
            return False, 'no graphical display in this environment'
        return True, 'ok'

    def open(self, *, unit: str, title: str, command: list[str], cwd: Path) -> dict:
        environment = {key: os.environ[key] for key in WINDOW_ENVIRONMENT if os.environ.get(key)}
        environment['CODEXA_AGENT_WINDOW'] = '1'
        launcher = ['systemd-run', '--user', '--collect', '--no-block', f'--unit={unit}', f'--working-directory={cwd}']
        launcher += [f'--setenv={key}={value}' for key, value in sorted(environment.items())]
        launcher += [shutil.which('kitty'), '--title', title, '-o', 'remember_window_size=no',
                     '-o', WINDOW_SIZE[0], '-o', WINDOW_SIZE[1]] + command
        completed = subprocess.run(launcher, capture_output=True, text=True, timeout=60)
        return dict(success=completed.returncode == 0, exit_code=completed.returncode, unit=unit,
                    environment_keys=sorted(environment), stderr_tail=completed.stderr.strip()[-300:])


class RecordingWindowBackend:
    """Mock: records the would-be window; never starts kitty or an agent."""
    name = 'mock'

    def __init__(self, run_dir: Path) -> None:
        self.log = Path(run_dir) / 'supervisor/notification-mock.jsonl'

    def available(self) -> tuple[bool, str]:
        return True, 'mock'

    def open(self, *, unit: str, title: str, command: list[str], cwd: Path) -> dict:
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(at=now(), channel='agent_window', unit=unit, title=title,
                                         command=command, cwd=str(cwd))) + '\n')
        return dict(success=True, exit_code=0, unit=unit, mock=True)


def _window_spec(run_dir: Path, root: Path, completion: dict, provider: str, ai: dict, *, unit: str) -> dict:
    session_id = ai.get('session_id') if ai.get('status') == 'delivered' else None
    prompt = window_prompt(run_dir, provider, completion, session_id, ai.get('status'))
    return dict(unit=unit, cwd=Path(root), command=agent_window_command(provider, root, session_id, prompt),
                title=f"Codexa agent — {completion['run_id']}: {completion['terminal_state']}")


def _maybe_open_window(run_dir: Path, root: Path, completion: dict, provider: str | None, ai: dict, backend,
                       base: dict, enabled: bool) -> dict | None:
    """At-most-once per terminal event, once the AI outcome is final. Caller holds the notification lock."""
    path = Path(run_dir) / 'supervisor/agent-window.json'
    window, _ = _load(path, completion['event_id'])
    if window and window.get('status') == 'claimed':
        window['history'].append(dict(at=now(), outcome='indeterminate', note='interrupted while opening; not reopened'))
        window['status'] = 'indeterminate'
        atomic_document(path, window)
    if window and window['status'] != 'backend_unavailable':
        return window
    if not provider or ai.get('status') not in ('delivered', 'failed_final'):
        return window
    window = window or dict(base, status='pending', history=[])
    if not enabled:
        window.update(status='disabled', backend=backend.name)
        atomic_document(path, window)
        return window
    ok, why = backend.available()
    if not ok:
        window['history'].append(dict(at=now(), outcome='backend_unavailable', detail=why))
        window.update(status='backend_unavailable', backend=backend.name)
        atomic_document(path, window)
        return window
    spec = _window_spec(run_dir, root, completion, provider, ai,
                        unit=f"codexa-agent-{completion['run_id']}-{completion['event_id'][:8]}")
    window.update(status='claimed', backend=backend.name, provider=provider, unit=spec['unit'], title=spec['title'],
                  command=spec['command'], resumes_session=ai.get('session_id') if ai.get('status') == 'delivered' else None)
    atomic_document(path, window)        # durable claim BEFORE the window appears
    try:
        result = backend.open(**spec)
    except Exception as error:
        result = dict(success=False, error=f'{type(error).__name__}: {error}')
    window['history'].append(dict(at=now(), **result))
    window.update(status='opened' if result.get('success') else 'failed', finished_utc=now())
    atomic_document(path, window)
    return window


def open_agent_window(run_dir: Path, root: Path, *, provider: str | None = None, window_backend=None) -> dict:
    """Explicit user request to (re)open the agent window for the current terminal event."""
    run_dir, root = Path(run_dir), Path(root)
    supervisor = run_dir / 'supervisor'
    completion = read_json(supervisor / 'completion.json')
    if not completion:
        raise SystemExit('No terminal event yet: the run has not stopped or finished.')
    ai = read_json(supervisor / 'ai-wakeup.json') or {}
    if ai.get('terminal_event_id') != completion['event_id']:
        ai = {}
    launch = read_json(supervisor / 'launch.json') or {}
    provider = provider or ai.get('provider') or (launch.get('wakeup') or {}).get('provider') or 'codex'
    production = execution_class(run_dir)['effective'] == 'production'
    backend = window_backend or (KittyAgentWindowBackend() if production else RecordingWindowBackend(run_dir))
    ok, why = backend.available()
    if not ok:
        raise SystemExit(f'Cannot open the agent window: {why}')
    spec = _window_spec(run_dir, root, completion, provider, ai if ai.get('provider', provider) == provider else {},
                        unit=f"codexa-agent-{completion['run_id']}-manual-{int(time.time())}")
    result = backend.open(**spec)
    with (supervisor / 'notification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = supervisor / 'agent-window.json'
        window, _ = _load(path, completion['event_id'])
        window = window or dict(run_id=completion['run_id'], terminal_event_id=completion['event_id'],
                                event_type=completion['terminal_state'], status='opened', history=[])
        window['history'].append(dict(at=now(), outcome='manual_open', command=spec['command'], **result))
        if result.get('success'):
            window.update(status='opened', backend=backend.name, unit=spec['unit'], command=spec['command'])
        atomic_document(path, window)
    return dict(result, command=spec['command'], backend=backend.name)


# ---------------------------------------------------------------- records

def _load(path: Path, event_id: str) -> tuple[dict | None, dict | None]:
    """Return (record for this event, previous record of another event of this run)."""
    record = read_json(path)
    if record and record.get('terminal_event_id') != event_id:
        return None, record
    return record, None


def _import_legacy(supervisor: Path, completion: dict) -> None:
    """Adopt attempts made by the superseded in-memory deliver() as already done."""
    legacy = read_json(supervisor / 'notification-state.json')
    if not legacy or legacy.get('format') == SUMMARY_FORMAT or legacy.get('event_id') != completion['event_id']:
        return
    channels = legacy.get('channels', {})
    base = dict(run_id=completion['run_id'], terminal_event_id=completion['event_id'],
                event_type=completion['terminal_state'], imported_from='legacy notification-state.json',
                imported_utc=now())
    desktop_path = supervisor / 'desktop-notification.json'
    if not desktop_path.exists() and channels.get('desktop'):
        attempts = channels['desktop'].get('attempts', [])
        state = {'delivered': 'delivered', 'pending': None}.get(channels['desktop'].get('status'), 'failed' if attempts else None)
        if state:
            atomic_document(desktop_path, dict(base, notification_state=state, notification_attempts=len(attempts),
                                               backend='legacy', history=attempts))
    ai_path = supervisor / 'ai-wakeup.json'
    if not ai_path.exists() and channels.get('ai'):
        status = channels['ai'].get('status')
        attempts = channels['ai'].get('attempts', [])
        if status and status != 'pending':
            atomic_document(ai_path, dict(base, status='failed_retryable' if status == 'attempting' else status,
                                          submissions=len(attempts), session_id=channels['ai'].get('session_id'),
                                          resume_command=channels['ai'].get('resume_command'), backend='legacy',
                                          history=attempts))


def outcome_message(completion: dict, ai_phrase: str | None) -> tuple[str, str, str]:
    """Concise, truthful text: 'verified' only after the integrity check passed."""
    state, reason, run_id = completion['terminal_state'], completion.get('reason'), completion['run_id']
    verified = completion.get('final_checkpoint_verification') == 'verified'
    if state == 'COMPLETED':
        summary = 'Codexa — Training completed'
        try:
            label = run_reporting.outcome_label(Path((completion.get('paths') or {})['run']))
        except Exception:  # noqa: BLE001 - fall back to a neutral label
            label = 'training run'
        body = f"{run_id}: {label} finished ({completion.get('optimizer_steps')} updates)"
        body += '; final checkpoint verified' if verified else '; final checkpoint NOT verified'
        urgency = 'normal' if verified else 'critical'
    else:
        summary = 'Codexa — Training stopped'
        last = (completion.get('last_verified_checkpoint') or {}).get('optimizer_step')
        if completion.get('final_checkpoint_verification') == 'failed':
            # The supervisor's 'last verified' role is index/sidecar-checked only; after a payload
            # integrity failure on the final role, claim nothing about preserved integrity.
            kept = f'final checkpoint NOT verified (integrity check failed at step {completion.get("optimizer_steps")})'
        elif verified:
            kept = f'final checkpoint verified (step {completion.get("optimizer_steps")})'
        else:
            kept = f'last verified checkpoint: step {last} preserved' if last is not None else 'no verified checkpoint'
        body = f"{run_id}: {state} — {reason}. {kept[0].upper() + kept[1:]}"
        urgency = 'critical' if state == 'FAILED' else 'normal'
    if ai_phrase:
        body += f'. {ai_phrase}'
    return summary, body + '.', urgency


def _manual_fallback(run_dir: Path, root: Path, provider: str, status: str) -> None:
    report = Path(run_dir) / 'final-report'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'prompt.txt').write_text(wakeup_prompt(run_dir, provider))
    command, options = provider_command(provider, run_dir, root)
    quoted = ' '.join(json.dumps(part) if (' ' in part or '(' in part) else part for part in command)
    (report / 'RESUME_MANUALLY.md').write_text(
        '# Automatic AI wakeup did not complete\n\n'
        f'Status: `{status}` (see `supervisor/ai-wakeup.json`). The AI was NOT automatically analysed.\n\n'
        'Request the analysis manually (fresh session):\n\n```sh\n'
        f'cd {json.dumps(str(options["cwd"]))} && {quoted} < {json.dumps(str(report / "prompt.txt"))}\n```\n')


def _summary(completion: dict, classification: dict, desktop: dict | None, ai: dict | None,
             window: dict | None = None) -> dict:
    return dict(format=SUMMARY_FORMAT, event_id=completion['event_id'], run_id=completion['run_id'],
                execution_class=classification['effective'], updated_utc=now(),
                records=dict(desktop='desktop-notification.json', ai='ai-wakeup.json', agent_window='agent-window.json'),
                channels=dict(desktop=dict(status=(desktop or {}).get('notification_state', 'pending'),
                                           backend=(desktop or {}).get('backend')),
                              ai=dict(status=(ai or {}).get('status', 'pending'), backend=(ai or {}).get('backend'),
                                      session_id=(ai or {}).get('session_id'),
                                      resume_command=(ai or {}).get('resume_command')),
                              agent_window=dict(status=(window or {}).get('status', 'pending'),
                                                backend=(window or {}).get('backend'),
                                                command=(window or {}).get('command'))))


def deliver(run_dir: Path, root: Path, *, provider: str | None = None, defer_ai: bool = False,
            dry_run: bool = False, desktop_backend=None, ai_backend=None, window_backend=None,
            ai_timeout: float = 2700) -> dict:
    """Process one terminal event idempotently. Compatible with the supervisor's deliver contract."""
    run_dir, root = Path(run_dir), Path(root)
    supervisor = run_dir / 'supervisor'
    completion = json.loads((supervisor / 'completion.json').read_text())
    classification = execution_class(run_dir)
    production = classification['effective'] == 'production'
    launch = read_json(supervisor / 'launch.json') or {}
    provider = provider or (launch.get('wakeup') or {}).get('provider') or completion.get('wakeup_provider')
    if dry_run:
        mock = RecordingDesktopBackend(run_dir)
        summary, body, urgency = outcome_message(completion, 'AI analysis would be requested' if provider else None)
        with mock.log.open('a') as stream:
            stream.write(json.dumps(dict(at=now(), channel='dry_run', classification=classification,
                summary=summary, body=body, provider=provider,
                command=provider_command(provider, run_dir, root)[0] if provider else None,
                agent_window=agent_window_command(provider, root, None, '<prompt>') if provider else None)) + '\n')
        return dict(_summary(completion, classification, None, None), dry_run=True,
                    channels=dict(desktop=dict(status='dry_run'), ai=dict(status='dry_run'),
                                  agent_window=dict(status='dry_run')))
    # A real window only when nothing is injected: tests exercising production policy never open Kitty.
    real_window = production and desktop_backend is None and ai_backend is None
    window_backend = window_backend or (KittyAgentWindowBackend() if real_window else RecordingWindowBackend(run_dir))
    window_enabled = launch.get('agent_window', True) is not False
    desktop_backend = desktop_backend or (NotifySendBackend() if production else RecordingDesktopBackend(run_dir))
    ai_backend = ai_backend or (CliAgentBackend() if production else RecordingAgentBackend(run_dir))
    desktop_path, ai_path = supervisor / 'desktop-notification.json', supervisor / 'ai-wakeup.json'
    lock_path = supervisor / 'notification.lock'
    handle = None

    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _import_legacy(supervisor, completion)
        summary_document = run_summary(run_dir, completion)
        if summary_document and not (supervisor / run_reporting.SUMMARY_FILE).exists():
            run_reporting.write_summary(summary_document, supervisor / run_reporting.SUMMARY_FILE)
        if not (supervisor / 'ai-handoff.md').exists():
            (supervisor / 'ai-handoff.md').write_text(run_reporting.handoff_text(summary_document)
                                                      if summary_document else handoff_text(completion))
        base = dict(run_id=completion['run_id'], terminal_event_id=completion['event_id'],
                    event_type=completion['terminal_state'], execution_class=classification['effective'],
                    classification_source=classification['source'])

        # ---- AI wakeup (own record) first, so the single popup can state the truth.
        ai, previous_ai = _load(ai_path, completion['event_id'])
        if ai is None:
            ai = dict(base, status='pending', submissions=0, history=[],
                      previous_event=(previous_ai or {}).get('terminal_event_id'))
        if ai['status'] == 'submitted':
            # In flight while either the waiting notifier or the agent process itself is alive.
            alive = ai.get('boot_id') == boot_id() and any(
                ai.get(pid) and process_start_ticks(ai[pid]) == ai.get(ticks)
                for pid, ticks in (('waiter_pid', 'waiter_ticks'), ('pid', 'start_ticks')))
            if not alive:
                ai['history'].append(dict(at=now(), outcome='abandoned', note='submitter ended before recording a result'))
                ai['status'] = 'failed_final' if ai['submissions'] >= MAX_AI_SUBMISSIONS else 'failed_retryable'
        ai_phrase = None
        if ai['status'] in TERMINAL_AI or ai['status'] == 'submitted':
            pass
        elif not provider:
            ai.update(status='disabled')
        elif defer_ai:
            ai.update(status='deferred', deferred_utc=now())
            ai_phrase = 'AI analysis deferred until next start'
        elif ai['submissions'] < MAX_AI_SUBMISSIONS:
            result = ai_backend.submit(provider, run_dir, root, wakeup_prompt(run_dir, provider, completion))
            ai['submissions'] += 1
            entry = {key: value for key, value in result.items() if not key.startswith('_')}
            ai['history'].append(dict(at=now(), outcome='submitted' if result['submitted'] else 'submit_failed', **entry))
            ai['backend'] = ai_backend.name
            if result['submitted']:
                ai.update(status='submitted', provider=provider, pid=result.get('pid'),
                          start_ticks=result.get('start_ticks'), boot_id=boot_id(),
                          waiter_pid=os.getpid(), waiter_ticks=process_start_ticks(os.getpid()))
                handle = result
                ai_phrase = 'Analysis requested' + (' (mock)' if getattr(ai_backend, 'name', '') == 'mock' else '')
            else:
                ai['status'] = 'failed_final' if ai['submissions'] >= MAX_AI_SUBMISSIONS else 'failed_retryable'
        if ai['status'] in ('failed_retryable', 'failed_final') and provider:
            _manual_fallback(run_dir, root, provider, ai['status'])
            ai_phrase = 'AI analysis could not be requested; see final-report/RESUME_MANUALLY.md'
        if ai_phrase is None:
            mock = ' (mock)' if ai.get('backend') == 'mock' else ''
            ai_phrase = {'submitted': f'Analysis requested{mock}', 'delivered': f'Analysis requested{mock}',
                         'deferred': 'AI analysis deferred until next start'}.get(ai['status'])
        atomic_document(ai_path, ai)

        # ---- Desktop: at-most-once for this terminal event.
        desktop, previous_desktop = _load(desktop_path, completion['event_id'])
        if desktop is None or desktop.get('notification_state') == 'backend_unavailable':
            desktop = desktop or dict(base, notification_state='pending', notification_attempts=0, history=[],
                                      previous_event=(previous_desktop or {}).get('terminal_event_id'))
            ok, why = desktop_backend.available()
            if not ok:
                desktop['history'].append(dict(at=now(), outcome='backend_unavailable', detail=why))
                desktop.update(notification_state='backend_unavailable', backend=desktop_backend.name)
                atomic_document(desktop_path, desktop)
            else:
                summary, body, urgency = outcome_message(completion, ai_phrase)
                replace = (previous_desktop or {}).get('desktop_notification_id') \
                    if (previous_desktop or {}).get('run_id') == completion['run_id'] else None
                desktop.update(notification_state='claimed', claimed_utc=now(), claimed_pid=os.getpid(),
                               notification_attempts=desktop['notification_attempts'] + 1,
                               backend=desktop_backend.name, summary=summary, body=body)
                atomic_document(desktop_path, desktop)          # durable claim BEFORE the popup
                try:
                    result = desktop_backend.send(summary, body, urgency=urgency, replace_id=replace)
                except Exception as error:
                    result = dict(success=False, error=f'{type(error).__name__}: {error}')
                desktop['history'].append(dict(at=now(), **result))
                desktop.update(notification_state='delivered' if result.get('success') else 'failed',
                               desktop_notification_id=result.get('notification_id'), finished_utc=now())
                atomic_document(desktop_path, desktop)
        elif desktop.get('notification_state') == 'claimed':
            desktop['history'].append(dict(at=now(), outcome='indeterminate',
                note='claimed by an interrupted process; popup may or may not have appeared; not resent'))
            desktop['notification_state'] = 'indeterminate'
            atomic_document(desktop_path, desktop)
        # ---- Interactive agent window, once the AI outcome is final (e.g. reconcile, or submit failed).
        window = _maybe_open_window(run_dir, root, completion, provider, ai, window_backend, base, window_enabled)
        atomic_document(supervisor / 'notification-state.json', _summary(completion, classification, desktop, ai, window))

    if handle is None:
        return _summary(completion, classification, desktop, ai, window)

    # ---- Wait for the agent without holding the lock; record the outcome under the lock.
    result = ai_backend.wait(provider, run_dir, handle, timeout=ai_timeout)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ai = read_json(ai_path)
        ai['history'].append(dict(at=now(), outcome='success' if result.get('success') else 'failed', **result))
        if result.get('success'):
            ai.update(status='delivered', session_id=result.get('session_id'),
                      resume_command=resume_command(provider, result.get('session_id')) if ai_backend.name != 'mock' else None)
        else:
            ai['status'] = 'failed_final' if ai['submissions'] >= MAX_AI_SUBMISSIONS else 'failed_retryable'
            _manual_fallback(run_dir, root, provider, ai['status'])
        atomic_document(ai_path, ai)
        desktop = read_json(desktop_path)
        window = _maybe_open_window(run_dir, root, completion, provider, ai, window_backend,
                                    {key: ai.get(key) for key in ('run_id', 'terminal_event_id', 'event_type',
                                                              'execution_class', 'classification_source')},
                                    window_enabled)
        atomic_document(supervisor / 'notification-state.json', _summary(completion, classification, desktop, ai, window))
        return _summary(completion, classification, desktop, ai, window)
