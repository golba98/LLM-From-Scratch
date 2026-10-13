"""The training-environment contract: source-first search paths and the guard adapter.

``tools/training_environment_guard.py`` is the enforcement authority. This module only
adapts it (``TrainingGuardProvider``) and builds the search path and session environment
that supervised training children inherit.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from typing import Callable, Protocol

from codexa_workspace.hashing import file_sha256
from codexa_workspace.paths import REPO_ROOT, SCRIPTS, TOOLS, WORKSPACE

SOURCE_REPOSITORIES = ('LLM-Architecture', 'LLM-Tokenizer', 'LLM-Data', 'LLM-Training', 'LLM-Inference',
                       'LLM-Memory', 'LLM-Specialist')
SESSION_ENVIRONMENT = ('WAYLAND_DISPLAY', 'DISPLAY', 'DBUS_SESSION_BUS_ADDRESS', 'XDG_RUNTIME_DIR',
                       'XDG_SESSION_TYPE', 'PATH', 'HOME', 'LANG')
GUARD = TOOLS / 'training_environment_guard.py'
SUPERVISOR_SCRIPT = SCRIPTS / 'training_supervisor.py'


class LaunchRefused(RuntimeError):
    """Training launch or resume prerequisites are not satisfied."""


def source_first_pythonpath() -> str:
    """The search path tools/codexa-dev exports: component sources first, never installed wheels."""
    return os.pathsep.join([*(str(WORKSPACE / name / 'src') for name in SOURCE_REPOSITORIES),
                            str(REPO_ROOT / 'src'), str(REPO_ROOT)])


def supervisor_available(*, which: Callable = shutil.which, runner: Callable = subprocess.run) -> dict:
    """systemd user manager, linger and the supervisor script; read-only."""
    def output(command):
        try:
            return runner(command, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''
    record = dict(systemd_run=bool(which('systemd-run')), systemctl=bool(which('systemctl')),
                  user_manager=output(['systemctl', '--user', 'is-system-running']),
                  linger=output(['loginctl', 'show-user', os.environ.get('USER', ''), '-p', 'Linger', '--value']),
                  reconcile_unit=output(['systemctl', '--user', 'is-enabled', 'codexa-train-reconcile.service']),
                  supervisor_script=SUPERVISOR_SCRIPT.is_file(), dashboard=importlib.util.find_spec('codexa_workspace.monitoring.dashboard') is not None)
    record['available'] = bool(record['systemd_run'] and record['systemctl'] and record['supervisor_script']
                               and record['user_manager'] in ('running', 'degraded') and record['linger'] == 'yes')
    return record


class EnvironmentProvider(Protocol):
    def resolve(self) -> dict: ...


def training_guard():
    """Codex's guard module (tools/training_environment_guard.py), imported read-only."""
    try:
        import training_environment_guard as guard
    except ImportError:
        spec = importlib.util.spec_from_file_location('training_environment_guard',
                                                      GUARD)
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
    return guard


class TrainingGuardProvider:
    """Adapter over Codex's enforcement: ``training_environment_guard.inspect('development', op)``.

    Runs the guard's own construction-free inspection in this process, so it judges the very
    interpreter, import origins and source trees that would train. A valid development verdict
    maps to the runtime's ``current`` context; any error yields ``invalid`` and is rejected.
    The guard is owned by Codex; this adapter never modifies or relaxes it.
    """

    interface = 'tools/training_environment_guard.py inspect(development, new) + import-time trainer wrappers'

    def __init__(self, operation: str = 'new', inspector=None):
        self.operation, self.inspector = operation, inspector

    def resolve(self) -> dict:
        guard_path = GUARD
        try:
            inspector = self.inspector or training_guard().inspect
            report = inspector('development', self.operation)
        except Exception as error:  # an unusable guard is never treated as a pass
            return dict(context='invalid', interface=self.interface, errors=[f'{type(error).__name__}: {error}'])
        valid = report.get('valid') is True and report.get('selected') == 'development' and not report.get('errors')
        return dict(context='current' if valid else 'invalid', interface=self.interface,
                    guard_sha256=file_sha256(guard_path) if guard_path.exists() else None,
                    operation=self.operation, errors=report.get('errors', []), python=report.get('python'),
                    architecture=report.get('architecture'), tokenizer_sha256=report.get('tokenizer_sha256'),
                    component_revisions=report.get('component_revisions'), enforcement='codex-guard')


def check_environment(provider: EnvironmentProvider, required_context: str) -> dict:
    resolved = provider.resolve()
    context = resolved.get('context')
    if context not in ('current', 'historical'):
        errors = '; '.join(resolved.get('errors') or [])[:1000]
        raise LaunchRefused(f'Ambiguous training environment {context!r}; refuse to proceed. {errors}'.strip())
    if context != required_context:
        raise LaunchRefused(f'Environment context {context!r} differs from required {required_context!r}.')
    return resolved
