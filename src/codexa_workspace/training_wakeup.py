"""Exactly-once terminal notification and one fresh AI analysis invocation.

A durable outbox (notification-state.json) keyed by the completion event id records
every attempt. Each provider call is a NEW noninteractive agent run, never a
resumption of the conversation that launched training.
"""

import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from llm_training.session import atomic_document

WAKEUP_ENVIRONMENT = 'CODEXA_WAKEUP_AGENT'
MAX_AI_ATTEMPTS = 2
PASSTHROUGH = re.compile(r'^(PATH|HOME|USER|LOGNAME|LANG|LC_[A-Z]+|TERM|SHELL|TMPDIR|XDG_[A-Z_]+|'
                         r'DBUS_SESSION_BUS_ADDRESS|(ANTHROPIC|OPENAI|CODEX|CLAUDE|CODEXA_TEST)_[A-Z0-9_]+)$')
INSTRUCTION = ("Codexa 934M's five-hour foundational training experiment has reached a terminal state. "
               "The Python supervisor has finished monitoring. Read the run's structured completion report, "
               "validate the final checkpoint and metrics, compare against the baseline, diagnose any failures, "
               "and produce the final training assessment. Do not start another training run.")


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def agent_environment() -> dict:
    """Minimal inherited environment; credentials stay in the CLIs' own stores and are never logged."""
    environment = {key: value for key, value in os.environ.items() if PASSTHROUGH.match(key)}
    environment[WAKEUP_ENVIRONMENT] = '1'
    return environment


def fmt(value, digits=5):
    return '--' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def handoff_text(completion: dict) -> str:
    """Compact handoff; never pastes logs."""
    v = completion.get('validation', {})
    f = completion.get('final_checkpoint') or {}
    last = completion.get('last_verified_checkpoint') or {}
    lines = [f"# Codexa training handoff — {completion['run_id']}", '', INSTRUCTION, '',
        '## Run', f"- Run ID: `{completion['run_id']}` (trainer run_id `{completion.get('trainer_run_id')}`)",
        f"- Job SHA256: `{completion.get('job_sha256')}`; spec SHA256: `{completion.get('spec_sha256')}`",
        f"- Terminal state: **{completion['terminal_state']}** — reason `{completion.get('reason')}`",
        f"- Elapsed: {fmt(completion.get('elapsed_seconds'), 0)} s of {fmt(completion.get('deadline_seconds'), 0)} s budget",
        f"- Optimizer updates: {completion.get('optimizer_steps')} / {completion.get('max_steps')}",
        f"- Processed target tokens: {completion.get('processed_tokens')}",
        '', '## Validation (corpus-token-weighted CE; per-source in completion.json)',
        f"- Starting (no-update baseline): {fmt(v.get('baseline'))}",
        f"- Best: {fmt(v.get('best'))}", f"- Final/latest: {fmt(v.get('final'))} at step {v.get('final_step')}",
        '', '## Checkpoints',
        f"- Final checkpoint: `{f.get('path')}`", f"- Final verification: {completion.get('final_checkpoint_verification')}",
        f"- Last verified recoverable checkpoint: `{last.get('path')}` (step {last.get('optimizer_step')})",
        '', '## Errors', *(f"- {item}" for item in (completion.get('errors') or ['none recorded'])),
        '', '## Paths (inspect sections as needed; do not modify)',
        *(f"- {name}: `{path}`" for name, path in completion.get('paths', {}).items()),
        '', '## Constraints',
        '- Analysis and reporting only. Do not launch, resume or schedule training; do not delete or move assets.',
        '- Neutral/direct generations require genuine human semantic review; do not fabricate review outcomes.']
    return '\n'.join(lines) + '\n'


def prompt_text(run_dir: Path, provider: str) -> str:
    output = run_dir / 'final-report' / ('ai-analysis.md' if provider == 'codex' else 'ai-analysis-claude.md')
    target = ('Your final message is saved verbatim as' if provider == 'codex' else 'Write your assessment to')
    return (f"{INSTRUCTION}\n\nStructured handoff: {run_dir / 'supervisor/ai-handoff.md'}\n"
            f"Completion record: {run_dir / 'supervisor/completion.json'}\n"
            f"{target} {output}. Keep it concise: verdict, metrics vs baseline, checkpoint validity, "
            "failure diagnosis if any, and recommended next decision (which must be a human decision).")


def provider_command(provider: str, run_dir: Path, root: Path, *, max_budget_usd: float = 5.0) -> tuple[list[str], dict]:
    """Return argv and options for a supported installed CLI; the prompt goes through stdin."""
    report = run_dir / 'final-report'
    if provider == 'codex':
        return (['codex', 'exec', '--skip-git-repo-check', '-s', 'read-only', '-C', str(root), '--json',
                 '-o', str(report / 'ai-analysis.md'), '-'], dict(cwd=root, output=report / 'ai-analysis.md'))
    if provider == 'claude':
        return (['claude', '-p', '--output-format', 'json', '--permission-mode', 'dontAsk',
                 '--max-budget-usd', str(max_budget_usd), '--add-dir', str(run_dir),
                 '--allowedTools', 'Read', 'Glob', 'Grep', 'Edit(final-report/**)', 'Write(final-report/**)'],
                dict(cwd=run_dir, output=report / 'ai-analysis-claude.md'))
    raise ValueError(f'Unsupported wakeup provider {provider!r}.')


def session_identity(provider: str, stdout: str) -> str | None:
    """Best-effort extraction of the new session id so the user can resume that fresh session."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            for key in ('session_id', 'thread_id', 'conversation_id'):
                if isinstance(value.get(key), str):
                    return value[key]
    return None


def invoke_provider(provider: str, run_dir: Path, root: Path, prompt: str, *, timeout: float = 2700,
                    dry_run: bool = False) -> dict:
    command, options = provider_command(provider, run_dir, root)
    (run_dir / 'final-report').mkdir(parents=True, exist_ok=True)
    if dry_run:
        return dict(dry_run=True, command=command, cwd=str(options['cwd']), prompt_chars=len(prompt), success=False)
    if shutil.which(command[0]) is None:
        return dict(success=False, error=f'{command[0]} CLI not found on PATH', command=command)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, input=prompt, cwd=options['cwd'], env=agent_environment(),
                                   capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(success=False, error=f'timeout after {timeout}s', command=command)
    events = run_dir / 'final-report' / f'{provider}-events.jsonl'
    events.write_text(completed.stdout)
    output = options['output']
    result = dict(command=command, exit_code=completed.returncode, seconds=round(time.monotonic() - started, 1),
                  stderr_tail=completed.stderr.strip()[-800:], session_id=session_identity(provider, completed.stdout),
                  output=str(output), events=str(events))
    if provider == 'claude':
        try:
            value = json.loads(completed.stdout.strip().splitlines()[-1])
            result['is_error'] = value.get('is_error')
            result['total_cost_usd'] = value.get('total_cost_usd')
            if not output.exists() and isinstance(value.get('result'), str):
                output.write_text(value['result'])
        except (json.JSONDecodeError, IndexError):
            result['is_error'] = True
    result['success'] = (completed.returncode == 0 and output.exists() and output.stat().st_size > 0
                         and not result.get('is_error'))
    return result


def resume_command(provider: str, session_id: str | None) -> str:
    if provider == 'codex':
        return f'codex exec resume {session_id}' if session_id else 'codex resume'
    return f'claude --resume {session_id}' if session_id else 'claude --continue'


def desktop_notify(summary: str, body: str, *, urgency: str = 'normal') -> dict:
    if shutil.which('notify-send') is None:
        return dict(success=False, error='notify-send unavailable')
    completed = subprocess.run(['notify-send', '--app-name=Codexa', f'--urgency={urgency}', summary, body],
                               capture_output=True, text=True, timeout=30)
    return dict(success=completed.returncode == 0, exit_code=completed.returncode,
                stderr_tail=completed.stderr.strip()[-300:])


def notification_text(completion: dict) -> tuple[str, str, str]:
    state, reason = completion['terminal_state'], completion.get('reason')
    verified = completion.get('final_checkpoint_verification') == 'verified'
    if state == 'COMPLETED' and verified:
        return ('Codexa training complete', f"{completion['run_id']}: {completion.get('optimizer_steps')} updates, "
                f"final checkpoint verified. Results ready for analysis.", 'normal')
    last = (completion.get('last_verified_checkpoint') or {}).get('optimizer_step')
    preserved = f'Last verified checkpoint: step {last}.' if last is not None else 'No verified checkpoint from this run.'
    return (f'Codexa training {state.lower().replace("_", " ")}', f"{completion['run_id']}: {reason}. {preserved} Review required.",
            'critical' if state == 'FAILED' else 'normal')


def deliver(run_dir: Path, root: Path, *, provider: str | None = None, defer_ai: bool = False,
            dry_run: bool = False, notifier=desktop_notify, invoker=invoke_provider) -> dict:
    """Idempotent delivery: one desktop notice and at most one successful AI call per event."""
    run_dir = Path(run_dir)
    supervisor = run_dir / 'supervisor'
    completion = json.loads((supervisor / 'completion.json').read_text())
    with (supervisor / 'notification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = supervisor / 'notification-state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['event_id'] != completion['event_id']:
                raise ValueError('A different terminal event was already recorded for this run.')
        else:
            state = dict(event_id=completion['event_id'], run_id=completion['run_id'], created_utc=now(),
                         provider=provider or completion.get('wakeup_provider'),
                         channels=dict(desktop=dict(status='pending', attempts=[]),
                                       ai=dict(status='pending', attempts=[]),
                                       desktop_analysis=dict(status='pending', attempts=[])))
            atomic_document(path, state)
        provider = state['provider']
        handoff = supervisor / 'ai-handoff.md'
        if not handoff.exists():
            handoff.write_text(handoff_text(completion))

        def record(channel: str, **values) -> None:
            state['channels'][channel].update(values)
            atomic_document(path, state)

        desktop = state['channels']['desktop']
        if desktop['status'] != 'delivered' and len(desktop['attempts']) < 2:
            summary, body = notification_text(completion)[:2]
            if provider and not dry_run and not defer_ai:
                body += f' AI analysis requested via {provider} CLI (fresh session).'
            result = dict(at=now(), **(dict(dry_run=True) if dry_run else notifier(summary, body, urgency=notification_text(completion)[2])))
            desktop['attempts'].append(result)
            record('desktop', status='delivered' if result.get('success') else ('dry_run' if dry_run else 'failed'))

        ai = state['channels']['ai']
        for attempt in ai['attempts']:
            if attempt.get('status') == 'attempting':
                attempt.update(status='abandoned', note='delivery process ended before recording a result')
        if not provider:
            record('ai', status='disabled')
        elif ai['status'] in ('delivered', 'failed_final', 'disabled'):
            pass
        elif defer_ai:
            record('ai', status='deferred', deferred_utc=now(), note='host stopping; reconcile delivers at next start')
        elif dry_run:
            ai['attempts'].append(dict(at=now(), status='dry_run',
                **invoker(provider, run_dir, root, prompt_text(run_dir, provider), dry_run=True)))
            record('ai', status='dry_run')
        elif len(ai['attempts']) < MAX_AI_ATTEMPTS:
            attempt = dict(at=now(), status='attempting', provider=provider, pid=os.getpid())
            ai['attempts'].append(attempt)
            record('ai', status='attempting')
            result = invoker(provider, run_dir, root, prompt_text(run_dir, provider))
            attempt.update(status='success' if result.get('success') else 'failed', finished=now(), **result)
            if result.get('success'):
                record('ai', status='delivered', session_id=result.get('session_id'),
                       resume_command=resume_command(provider, result.get('session_id')))
            else:
                record('ai', status='failed_final' if len(ai['attempts']) >= MAX_AI_ATTEMPTS else 'failed_retryable')
        if ai['status'] not in ('delivered', 'dry_run', 'disabled', 'deferred'):
            manual = run_dir / 'final-report' / 'RESUME_MANUALLY.md'
            manual.parent.mkdir(parents=True, exist_ok=True)
            command, options = provider_command(provider or 'codex', run_dir, root)
            manual.write_text('# Automatic AI wakeup did not complete\n\n'
                f'Status: `{ai["status"]}`. Attempts are recorded in `{path}`. The AI was NOT automatically analysed.\n\n'
                'Run this to request the analysis manually (fresh session):\n\n```sh\n'
                f'cd {json.dumps(str(options["cwd"]))} && {" ".join(json.dumps(part) if " " in part or "(" in part else part for part in command)} '
                f'< {json.dumps(str(run_dir / "final-report/prompt.txt"))}\n```\n')
            (run_dir / 'final-report' / 'prompt.txt').write_text(prompt_text(run_dir, provider or 'codex'))
        analysis = state['channels']['desktop_analysis']
        if ai['status'] == 'delivered' and analysis['status'] != 'delivered' and not analysis['attempts']:
            result = notifier('Codexa analysis ready', f"{provider} wrote {ai['attempts'][-1].get('output')}")
            analysis['attempts'].append(dict(at=now(), **result))
            record('desktop_analysis', status='delivered' if result.get('success') else 'failed')
        elif ai['status'] == 'failed_final' and not analysis['attempts']:
            result = notifier('Codexa AI wakeup failed', f"See {run_dir / 'final-report/RESUME_MANUALLY.md'}", urgency='critical')
            analysis['attempts'].append(dict(at=now(), **result))
            record('desktop_analysis', status='delivered' if result.get('success') else 'failed')
        state['updated_utc'] = now()
        atomic_document(path, state)
        return state


def test_provider(provider: str, directory: Path, root: Path) -> dict:
    """Harmless authenticated CLI check with a trivial prompt in an isolated directory."""
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / 'final-report'
    report.mkdir(exist_ok=True)
    command, options = provider_command(provider, directory, root, max_budget_usd=0.5)
    prompt = ('This is a connectivity test. Do not read or modify any files. '
              + ('Reply with exactly: OK' if provider == 'codex' else
                 'Use the Write tool to create final-report/ai-analysis-claude.md containing exactly OK, then reply OK.'))
    started = time.monotonic()
    completed = subprocess.run(command, input=prompt, cwd=options['cwd'], env=agent_environment(),
                               capture_output=True, text=True, timeout=600)
    output = options['output']
    result = dict(provider=provider, command=command, exit_code=completed.returncode,
                  seconds=round(time.monotonic() - started, 1), session_id=session_identity(provider, completed.stdout),
                  output_exists=output.exists(), output=output.read_text()[:200] if output.exists() else None,
                  stdout_tail=completed.stdout.strip()[-600:], stderr_tail=completed.stderr.strip()[-600:])
    result['passed'] = completed.returncode == 0 and output.exists() and 'OK' in output.read_text()
    atomic_document(directory / f'{provider}-test.json', result)
    return result
