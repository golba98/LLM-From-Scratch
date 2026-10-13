"""AI provider adapters for the run-end analysis: commands, environment, session ids and the
fallback handoff/prompt text. Delivery itself (outbox, exactly-once) lives in ``notifier``.

Each provider call is a NEW noninteractive agent run, never a resumption of the
conversation that launched training.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import time

from llm_training.session import atomic_document

WAKEUP_ENVIRONMENT = 'CODEXA_WAKEUP_AGENT'
PASSTHROUGH = re.compile(r'^(PATH|HOME|USER|LOGNAME|LANG|LC_[A-Z]+|TERM|SHELL|TMPDIR|XDG_[A-Z_]+|'
                         r'DBUS_SESSION_BUS_ADDRESS|(ANTHROPIC|OPENAI|CODEX|CLAUDE|CODEXA_TEST)_[A-Z0-9_]+)$')
INSTRUCTION = ("Codexa 934M's five-hour foundational training experiment has reached a terminal state. "
               "The Python supervisor has finished monitoring. Read the run's structured completion report, "
               "validate the final checkpoint and metrics, compare against the baseline, diagnose any failures, "
               "and produce the final training assessment. Do not start another training run.")


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


def resume_command(provider: str, session_id: str | None) -> str:
    if provider == 'codex':
        return f'codex exec resume {session_id}' if session_id else 'codex resume'
    return f'claude --resume {session_id}' if session_id else 'claude --continue'


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
