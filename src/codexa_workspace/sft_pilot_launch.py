"""Authorized launch path for the conversational SFT pilot. Never invoked during preparation.

The pilot reuses the hash-bound OS supervisor (``training_supervisor``) unchanged: this module
only writes the run's ``supervisor/launch.json`` and starts ``scripts/training_supervisor.py run``
as a transient user unit. Both the supervisor and its trainer child run source-first with
``CODEXA_TRAINING_ENVIRONMENT=development``, so Codex's startup guard and the import-time
trainer wrappers validate the committed development identity before any optimizer exists.

``launch_authorized`` launches nothing unless the frozen job verifies, a matching user
approval exists, the readiness verdict is READY FOR AUTHORIZATION and every preflight check
passes. The process launcher is injected so the whole path is testable without systemd.
"""
from __future__ import annotations

import inspect as _inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from codexa_workspace import sft_pilot_runtime as runtime
from codexa_workspace.sft_pilot_runtime import LaunchRefused

ROOT = runtime.ROOT
WORKSPACE = ROOT.parent
PYTHON = ROOT / '.venv/bin/python'
SUPERVISOR_SCRIPT = ROOT / 'scripts/training_supervisor.py'
SOURCE_REPOSITORIES = ('LLM-Architecture', 'LLM-Tokenizer', 'LLM-Data', 'LLM-Training', 'LLM-Inference',
                       'LLM-Memory', 'LLM-Specialist')
SESSION_ENVIRONMENT = ('WAYLAND_DISPLAY', 'DISPLAY', 'DBUS_SESSION_BUS_ADDRESS', 'XDG_RUNTIME_DIR',
                       'XDG_SESSION_TYPE', 'PATH', 'HOME', 'LANG')
READY = 'READY FOR AUTHORIZATION'
RESUMABLE_STOPS = ('MEMORY_HEADROOM_GATE', 'DISK_HEADROOM', 'DISK_RESERVE_WATCHDOG')

# The approved pilot configuration. Any deviation is refused rather than silently accepted.
APPROVED_TRAINING = dict(micro_batch_size=1, gradient_accumulation_steps=8, learning_rate=5e-6, weight_decay=.1,
                         warmup_steps=8, max_steps=128, gradient_clip=1.0, precision='bf16')
APPROVED_OPTIMIZATION = dict(optimizer='adamw8bit', minimum_learning_rate=5e-7, weight_dtype='fp32', autocast='bf16',
                             loss_normalization='supervised_tokens', tf32=False)


def source_first_pythonpath() -> str:
    """The search path tools/codexa-dev exports: component sources first, never installed wheels."""
    return os.pathsep.join([*(str(WORKSPACE / name / 'src') for name in SOURCE_REPOSITORIES),
                            str(ROOT / 'src'), str(ROOT)])


def child_environment() -> dict:
    return dict(PYTHONPATH=source_first_pythonpath(), CODEXA_TRAINING_ENVIRONMENT='development',
                LLM_INTEGRATION_ROOT=str(ROOT),
                LLM_TRAINING_VIEWER_COMMAND=json.dumps([str(PYTHON), str(ROOT / 'scripts/training_dashboard.py'),
                                                        '--attach']))


def run_directory(job: dict) -> Path:
    return Path(job['spec']['outputs']['root'])


def build_launch(run_dir: Path, job: dict, *, provider: str, written_utc: str) -> dict:
    """The supervisor's launch.json for this pilot (schema of scripts/training_supervisor.py write_launch)."""
    if provider not in ('codex', 'claude'):
        raise ValueError('provider must be codex or claude')
    return dict(run_id=job['run_id'], cwd=str(ROOT),
                command=[str(PYTHON), '-m', 'codexa_workspace.sft_pilot_trainer', '--run-dir', str(run_dir)],
                environment=child_environment(), wakeup=dict(provider=provider),
                deadline_seconds=job['spec']['budget']['total_seconds'], agent_window=True,
                execution_class='production', policy={}, job_sha256=job['job_sha256'], written_utc=written_utc)


# ---------------------------------------------------------------- configuration and runtime support

def validate_training_config(spec: dict) -> dict:
    """The frozen spec must equal the approved pilot exactly; unsupported settings are refused."""
    problems = [f'training.{k}={spec["training"].get(k)!r} (approved {v!r})'
                for k, v in APPROVED_TRAINING.items() if spec['training'].get(k) != v]
    problems += [f'optimization.{k}={spec["optimization"].get(k)!r} (approved {v!r})'
                 for k, v in APPROVED_OPTIMIZATION.items() if spec['optimization'].get(k) != v]
    if spec['data'].get('maximum_serialized_tokens') != 512 or spec['data'].get('presentations') != 1024:
        problems.append('data: context limit 512 and exactly 1,024 presentations are required')
    if spec['evaluation'].get('updates') != [0, 32, 64, 128]:
        problems.append('evaluation updates must be 0, 32, 64, 128')
    if spec['checkpoint'].get('publications') != [64, 'final']:
        problems.append('checkpoints must be published at update 64 and final')
    if spec['initialization'].get('policy') != 'weights_only_fresh_optimizer_and_scheduler':
        problems.append('initialization must be weights-only with a fresh optimizer and scheduler')
    if spec['training']['max_steps'] * spec['training']['gradient_accumulation_steps'] \
            * spec['training']['micro_batch_size'] != spec['data'].get('presentations'):
        problems.append('updates x accumulation x microbatch must equal the presentations')
    if problems:
        raise LaunchRefused('Unsupported or unapproved training configuration: ' + '; '.join(problems))
    return dict(approved=True, training=APPROVED_TRAINING, optimization=APPROVED_OPTIMIZATION)


def runtime_support(*, cuda: bool = True) -> dict:
    """Confirm the training stack supports every setting, by signature and capability only.

    Imports modules and inspects signatures; constructs no model and no optimizer.
    """
    import torch
    from llm_architecture.runtime import resolve_precision
    from llm_data.chat_dataset import collate_chat  # noqa: F401
    from llm_training.pretrain_run import FixedOrderSampler  # noqa: F401
    from llm_training import training
    parameters = set(_inspect.signature(training.train_model).parameters)
    required = {'max_steps', 'gradient_accumulation_steps', 'gradient_clip', 'warmup_steps', 'peak_learning_rate',
                'minimum_learning_rate', 'loss_normalization', 'deadline_monotonic', 'stop_requested',
                'on_optimizer_step', 'max_epochs'}
    missing = sorted(required - parameters)
    optimizer_parameters = set(_inspect.signature(training.create_adamw_optimizer).parameters)
    schedule = [training.cosine_learning_rate(step, warmup_steps=8, max_steps=128, peak_learning_rate=5e-6,
                                              minimum_learning_rate=5e-7) for step in (0, 7, 8, 128)]
    import bitsandbytes.optim
    result = dict(train_model_parameters_missing=missing,
                  optimizer_factory_accepts=sorted({'optimizer_name', 'weight_decay', 'betas', 'epsilon'}
                                                   & optimizer_parameters),
                  adamw8bit_available=hasattr(bitsandbytes.optim, 'AdamW8bit'),
                  schedule_samples=dict(zip(('step0', 'step7', 'step8', 'step128'), schedule)),
                  fixed_order_sampler=True, collate_chat=True)
    if cuda:
        result.update(cuda_available=torch.cuda.is_available(),
                      bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
        if result['cuda_available']:
            policy = resolve_precision('bf16', torch.device('cuda'))
            result['precision_policy'] = str(policy)
    if missing or not result['adamw8bit_available'] or len(result['optimizer_factory_accepts']) != 4 \
            or (cuda and not result.get('bf16_supported')):
        raise LaunchRefused(f'Training runtime does not support the approved configuration: {result}')
    if abs(schedule[-1] - 5e-7) > 1e-12:
        raise LaunchRefused(f'Cosine schedule does not end at the minimum learning rate: {schedule}')
    return result


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
                  supervisor_script=SUPERVISOR_SCRIPT.is_file(), dashboard=(ROOT / 'scripts/training_dashboard.py').is_file())
    record['available'] = bool(record['systemd_run'] and record['systemctl'] and record['supervisor_script']
                               and record['user_manager'] in ('running', 'degraded') and record['linger'] == 'yes')
    return record


def resume_allowed(run_dir: Path, job: dict, *, latest: dict | None) -> dict:
    """Resume only after a resource stop of this exact job, from an independently verified checkpoint."""
    completion_path = Path(run_dir) / 'supervisor/completion.json'
    if not completion_path.exists():
        raise LaunchRefused('No terminal record: nothing to resume.')
    completion = json.loads(completion_path.read_text())
    if completion.get('reason') not in RESUMABLE_STOPS:
        raise LaunchRefused(f"Not resumable: {completion.get('terminal_state')}/{completion.get('reason')}")
    recorded = json.loads((Path(run_dir) / 'config/job.json').read_text())
    if recorded.get('job_sha256') != job['job_sha256'] or recorded.get('code_identity') != job['code_identity']:
        raise LaunchRefused('Unsafe resume: job or code identity differs from the interrupted run.')
    if not (latest and latest.get('verified')):
        raise LaunchRefused('Unsafe resume: no independently verified latest checkpoint.')
    return dict(resumable=True, reason=completion['reason'], from_step=latest.get('optimizer_step'))


# ---------------------------------------------------------------- authorization boundary

def approval_for(directory: Path, job: dict) -> dict:
    """The user's grant lives beside the frozen job; it must name this exact job hash."""
    return runtime.approval_valid(Path(directory), job)


def write_approval(directory: Path, job: dict, *, statement: str, written_utc: str) -> Path:
    """Record an explicit user grant. Only ever run by the user at authorization time."""
    if not statement.strip():
        raise ValueError('An explicit authorization statement is required.')
    runtime.verify_job(job)
    path = Path(directory) / 'config/approval.json'
    if path.exists():
        raise FileExistsError('An approval already exists for this experiment; never overwrite a grant.')
    path.write_text(json.dumps(dict(format=runtime.APPROVAL_FORMAT, approved=True, job_sha256=job['job_sha256'],
                                    statement=statement.strip(), granted_utc=written_utc), indent=2) + '\n')
    return path


def launch_authorized(directory: Path, job: dict, *, provider: str, readiness: Callable[[], dict],
                      environment: runtime.EnvironmentProvider, measured: dict, written_utc: str,
                      launcher: Callable | None = None, supervisor: Callable[[], dict] = supervisor_available,
                      verify_manifest: Callable | None = None) -> dict:
    """The single authorized launch path. Every check runs first; any failure launches nothing."""
    if os.environ.get('CODEXA_WAKEUP_AGENT'):
        raise LaunchRefused('A wakeup analysis agent may not launch training.')
    runtime.verify_job(job)
    approval = approval_for(directory, job)
    verdict = readiness()
    if verdict.get('verdict') != READY:
        raise LaunchRefused(f"Readiness is {verdict.get('verdict')!r}, not {READY!r}: {verdict.get('blocking')}")
    environment_record = runtime.check_environment(environment, job['spec']['environment']['required_context'])
    expected = job['spec']['environment'].get('component_revisions') or {}
    detected = {name: value.get('detected') for name, value in (environment_record.get('component_revisions') or {}).items()}
    if expected and detected != expected:
        raise LaunchRefused(f'Source revisions differ from the frozen job: expected {expected}, detected {detected}')
    validate_training_config(job['spec'])
    prepared = runtime.prepare(job, verify_manifest=verify_manifest, measured=measured)
    if not prepared['passed']:
        raise LaunchRefused(f"Launch gates failed: {[c['name'] for c in prepared['checks'] if not c['passed']]}")
    available = supervisor()
    if not available.get('available'):
        raise LaunchRefused(f'OS supervisor unavailable: {available}')
    run_dir = run_directory(job)
    if run_dir.exists():
        raise LaunchRefused(f'Run namespace {run_dir} already exists; never overwrite or reuse a run.')

    from codexa_workspace import training_supervisor
    (run_dir / 'config').mkdir(parents=True)
    training_supervisor.atomic_document(run_dir / 'config/job.json', job)
    training_supervisor.atomic_document(run_dir / 'config/approval.json', approval)
    state = training_supervisor.RunState(run_dir)
    state.transition('VALIDATING')
    state.transition('READY', launch_gates=dict(readiness=verdict.get('verdict'), prepare='passed'))
    launch = build_launch(run_dir, job, provider=provider, written_utc=written_utc)
    training_supervisor.atomic_document(run_dir / 'supervisor/launch.json', launch)
    environment_for_unit = {key: os.environ[key] for key in SESSION_ENVIRONMENT if os.environ.get(key)}
    environment_for_unit.update(PYTHONPATH=source_first_pythonpath(), CODEXA_TRAINING_ENVIRONMENT='development')
    start = launcher or training_supervisor.systemd_launch
    command = start(run_dir, ROOT, unit=f'codexa-train-{run_dir.name}', python=str(PYTHON),
                    script=SUPERVISOR_SCRIPT, environment=environment_for_unit)
    state.event('systemd_launch', unit=f'codexa-train-{run_dir.name}')
    return dict(launched=True, run_dir=str(run_dir), command=command, launch=launch)
