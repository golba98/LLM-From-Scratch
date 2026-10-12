"""SFT pilot runtime adapter: prepare / evaluate modes and a separately authorized train mode.

Prepare reads data and configuration and constructs no model or optimizer.
Evaluate loads weights for inference only, under ``NoTrainingGuard``.
Train is reachable only through ``train_authorized`` with a matching approval
grant, a passing environment contract and passing resource gates; nothing in
prepare or evaluate transitions into it. Preparation never calls train mode.

The environment contract is Codex's guard (tools/training_environment_guard.py): the
development selection, committed and pinned source, source-first imports and the repaired
architecture. Job format v2 (Step 5A) binds the guard's report and the proposed 1,200 s cap.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = ROOT / 'outputs/training/codexa-934m-foundation-5h-v1'
INITIALIZATION = FOUNDATION / 'checkpoints/slot-0/checkpoint.pt'
INITIALIZATION_SHA256 = 'ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e'
EXPECTED_PARAMETERS = 934_356_480
JOB_FORMAT = 'codexa-sft-pilot-job-v2'
APPROVAL_FORMAT = 'codexa-sft-pilot-approval-v1'
RUN_ID = 'codexa-934m-sft-feasibility-pilot-v1'
EVALUATION_UPDATES = (0, 32, 64, 128)
CHECKPOINT_UPDATES = (64, 'final')
# Proposed in Step 5A from measured evaluation timing; a cap is only ever granted by the user.
PROPOSED_BUDGET = dict(total_seconds=1200, finalization_reserve_seconds=180, automatic_extension=False)
# Intermediate evaluations keep every stop and acceptance signal (greedy conversation, gold END and
# premature END, reference CE); nucleus decoding and the 48 historical prompts run at the final update.
REDUCED_EVALUATION_UPDATES = (32, 64)
CODE_FILES = ('src/codexa_workspace/sft_pilot_runtime.py', 'src/codexa_workspace/sft_pilot_trainer.py',
              'src/codexa_workspace/sft_pilot_launch.py', 'src/codexa_workspace/sft_pilot_evaluation.py',
              'src/codexa_workspace/sft_phase4_data.py', 'src/codexa_workspace/sft_feasibility.py',
              'src/codexa_workspace/training_supervisor.py', 'scripts/training_supervisor.py',
              'tools/training_environment_guard.py')
COMPONENT_MODULES = ('llm_training.training', 'llm_training.recovery', 'llm_training.session',
                     'llm_training.checkpointing', 'llm_training.pretrain_run', 'llm_training.config',
                     'llm_data.chat_dataset',
                     'llm_tokenizer.sft', 'llm_tokenizer.tokenizer', 'llm_architecture.model',
                     'llm_architecture.checkpointing', 'llm_architecture.runtime', 'llm_inference.generate')
NON_PRODUCTION_CLASSES = ('unit_test', 'integration_test', 'simulation')


class TrainingProhibited(RuntimeError):
    """An optimizer, backward pass, weight update or checkpoint write was attempted outside train mode."""


class LaunchRefused(RuntimeError):
    """Train mode prerequisites are not satisfied."""


# ---------------------------------------------------------------- canonical gates

def canonical_gates() -> dict:
    """Read the production gates from their owner; never redefine or lower them here."""
    from codexa_workspace import foundation_training as owner
    return dict(startup_free_vram_mib=owner.VRAM_FREE_MIB, available_ram_bytes=owner.HOST_RAM_BYTES,
                free_disk_bytes=owner.DISK_FLOOR_BYTES, recovery_reserve_bytes=owner.RECOVERY_RESERVE_BYTES,
                in_run_free_vram_bytes=1536 * 1024**2)


def validate_gates(gates: dict) -> dict:
    canonical = canonical_gates()
    weakened = {k: (gates.get(k), v) for k, v in canonical.items() if gates.get(k) is None or gates[k] < v}
    if weakened:
        raise ValueError(f'Resource gates may not be lowered or omitted: {weakened}')
    return gates


def measure_resources(path: Path = ROOT) -> dict:
    """Read-only host measurements (nvidia-smi, /proc/meminfo, statvfs)."""
    vram = None
    try:
        done = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
        name, total, used, free = [x.strip() for x in done.stdout.strip().splitlines()[0].split(',')]
        vram = dict(name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        vram = None
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    return dict(vram=vram, available_ram_bytes=available, free_disk_bytes=shutil.disk_usage(path).free,
                measured_utc=datetime.now(timezone.utc).isoformat())


def resource_checks(gates: dict, measured: dict, *, required_disk_bytes: int | None = None) -> list[dict]:
    gates = validate_gates(gates)
    vram = measured.get('vram') or {}
    disk_needed = max(gates['free_disk_bytes'], gates['recovery_reserve_bytes'] + (required_disk_bytes or 0))
    return [
        dict(name='startup_free_vram', passed=vram.get('free_mib', -1) >= gates['startup_free_vram_mib'],
             measured=vram.get('free_mib'), required=gates['startup_free_vram_mib'], unit='MiB'),
        dict(name='available_ram', passed=(measured.get('available_ram_bytes') or -1) >= gates['available_ram_bytes'],
             measured=measured.get('available_ram_bytes'), required=gates['available_ram_bytes'], unit='bytes'),
        dict(name='free_disk_and_recovery_reserve', passed=measured['free_disk_bytes'] >= disk_needed,
             measured=measured['free_disk_bytes'], required=disk_needed, unit='bytes')]


# ---------------------------------------------------------------- no-training guard

@contextmanager
def no_training_guard(*models):
    """Make optimizer construction, backward passes and torch.save raise; freeze parameters."""
    import torch
    patches = []

    def refuse(what):
        def raiser(*args, **kwargs):
            raise TrainingProhibited(f'{what} is prohibited outside authorized train mode.')
        return raiser

    targets = [(torch.optim.Optimizer, '__init__', 'Optimizer construction'),
               (torch.Tensor, 'backward', 'Backpropagation'),
               (torch.autograd, 'backward', 'Backpropagation'),
               (torch, 'save', 'Checkpoint write')]
    try:
        import bitsandbytes.optim as bnb_optim  # noqa: F401
        for name in ('AdamW8bit', 'Adam8bit', 'PagedAdamW8bit'):
            if hasattr(bnb_optim, name):
                targets.append((getattr(bnb_optim, name), '__init__', f'bitsandbytes {name} construction'))
    except Exception:
        pass
    for owner, name, what in targets:
        patches.append((owner, name, owner.__dict__.get(name, getattr(owner, name))))
        setattr(owner, name, refuse(what))
    for model in models:
        model.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise TrainingProhibited('Parameters must be frozen during preparation and evaluation.')
    try:
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


# ---------------------------------------------------------------- environment contract

class EnvironmentProvider(Protocol):
    def resolve(self) -> dict: ...


class IdentityScriptProvider:
    """Read-only context report over scripts/environment_identity.py (report + context_verdict).

    Used for inference-only evaluation. It rejects ambiguity but is not the enforcement
    authority for training; ``TrainingGuardProvider`` (Codex's guard) is.
    """

    interface = 'scripts/environment_identity.py:report/context_verdict'

    def resolve(self) -> dict:
        spec = importlib.util.spec_from_file_location('_codexa_environment_identity',
                                                      ROOT / 'scripts/environment_identity.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.report()
        return dict(context=module.context_verdict(data), executable=data['executable'],
                    bootstrap_active=data['bootstrap_active'], architecture=data['architecture'],
                    historical_reproducibility=data['historical_reproducibility']['status'],
                    interface=self.interface, enforcement='context report (evaluation only)')


def training_guard():
    """Codex's guard module (tools/training_environment_guard.py), imported read-only."""
    try:
        import training_environment_guard as guard
    except ImportError:
        spec = importlib.util.spec_from_file_location('training_environment_guard',
                                                      ROOT / 'tools/training_environment_guard.py')
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
        guard_path = ROOT / 'tools/training_environment_guard.py'
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


# ---------------------------------------------------------------- frozen job

def file_sha256(path) -> str:
    from codexa_workspace.sft_phase4_data import digest
    return digest(path)


def code_identity(root: Path = ROOT) -> dict:
    import importlib
    identities = {name: file_sha256(root / name) for name in CODE_FILES}
    for name in COMPONENT_MODULES:
        identities[name] = file_sha256(Path(importlib.import_module(name).__file__))
    return identities


def build_spec(directory: Path, manifest: dict, *, step_zero: dict | None, step_zero_path: Path | None = None,
               manifest_path: Path | None = None, environment_report: dict | None = None,
               budget_analysis: str | None = None) -> dict:
    """The complete frozen experiment specification. Non-runnable until separately authorized.

    ``directory`` holds the frozen data manifest; ``step_zero_path`` and ``manifest_path`` default
    to that directory's sealed artifacts. ``environment_report`` is the guard's development
    report at freeze time: the component revisions it names must still be HEAD at launch.
    """
    report = environment_report or {}
    revisions = {name: value.get('detected') for name, value in (report.get('component_revisions') or {}).items()}
    from codexa_workspace import sft_pilot_evaluation as evaluation
    from codexa_workspace.sft_phase4_data import FOUNDATION as _foundation, read
    foundation = read(_foundation / 'config/job.json')['spec']
    return dict(
        format=JOB_FORMAT, run_name=RUN_ID, purpose='conversational SFT feasibility pilot', device='cuda',
        runnable=False, authorization_required=True, viewer_required=True,
        expected_parameters=EXPECTED_PARAMETERS, model=foundation['model'],
        initialization=dict(path=str(INITIALIZATION), sha256=INITIALIZATION_SHA256,
                            policy='weights_only_fresh_optimizer_and_scheduler',
                            note='Fresh AdamW8bit and scheduler; never an exact resume of the foundation optimizer.'),
        tokenizer=dict(path=manifest['tokenizer']['path'], sha256=manifest['tokenizer']['sha256']),
        environment=dict(required_context='current',
                         reason='Source-first components include the masked KV-cache repair and the nonfinite-logit check.',
                         codex_selection='development', enforcement_interface=TrainingGuardProvider.interface,
                         provider_for_train_mode='TrainingGuardProvider(operation=new)',
                         provider_for_evaluation=IdentityScriptProvider.interface,
                         child_python=str(ROOT / '.venv/bin/python'), child_imports='source-first PYTHONPATH',
                         guard_sha256=report.get('guard_sha256'), component_revisions=revisions,
                         architecture_sha256=(report.get('architecture') or {}).get('sha256'),
                         tokenizer_sha256=report.get('tokenizer_sha256')),
        data=dict(manifest=str(manifest_path or Path(directory) / 'manifest/phase4-manifest.json'),
                  manifest_sha256=manifest['manifest_sha256'],
                  train=manifest['datasets']['train']['path'], train_sha256=manifest['datasets']['train']['sha256'],
                  validation=manifest['datasets']['validation']['path'],
                  validation_sha256=manifest['datasets']['validation']['sha256'],
                  examples=512, validation_examples=32, maximum_serialized_tokens=512, truncation=False,
                  epochs=2, epoch_seeds=[73, 74], presentations=1024, padding='dynamic (microbatch 1, no padding)',
                  order='manifest orders[0] then orders[1] (ConversationSampler seed 73, bucket 1, epochs 0/1)',
                  reference=dict((k, foundation['data'][k]) for k in (
                      'corpus_path', 'dtype', 'validation_window_ids', 'validation_window_ids_sha256',
                      'validation_sources', 'validation_sources_sha256', 'source_loss_weights'))),
        training=dict(micro_batch_size=1, gradient_accumulation_steps=8, learning_rate=5e-6, weight_decay=.1,
                      warmup_steps=8, max_steps=128, gradient_clip=1.0, precision='bf16', checkpoint_interval=64,
                      evaluation_interval=32, seed=73),
        optimization=dict(optimizer='adamw8bit', betas=[.9, .95], epsilon=1e-8, minimum_learning_rate=5e-7,
                          schedule='linear warmup 8 then cosine to 5e-7', weight_dtype='fp32', autocast='bf16',
                          gradient_checkpointing=True, loss_normalization='supervised_tokens', loss_weights=None,
                          tf32=False, allocator='default', compile=False, minimum_headroom_bytes=1536 * 1024**2),
        evaluation=dict(updates=list(EVALUATION_UPDATES),
                        step_zero='sealed artifact computed in Phase 4 (same code and settings)',
                        step_zero_sha256=(step_zero or {}).get('summary_sha256'),
                        step_zero_path=str(step_zero_path or Path(directory) / 'evaluation/step-zero/summary.json'),
                        reduced_updates=list(REDUCED_EVALUATION_UPDATES),
                        schedule=dict(intermediate='conversation greedy-16, gold END/premature END and assistant CE, '
                                                   'reference CE (every stop and acceptance signal)',
                                      final='full step-zero schedule: conversation and 48 historical prompts, '
                                            'greedy and nucleus, gold, reference'),
                        suites=evaluation.suite_identity(),
                        decoding={k: dict(v.__dict__) for k, v in evaluation.decoding_configs().items()},
                        primary_decoding='greedy', secondary_decoding='nucleus',
                        eos_token_id=2, end_token_id=7, pad_token_id=0, stop_sequences=[[7]],
                        gold='32 frozen validation conversations: assistant CE, gold END NLL, premature END probability',
                        reference='256 frozen 2048-token windows: Wikipedia and FineWeb-Edu CE'),
        acceptance=dict(evaluation.ACCEPTANCE),
        guards=dict(gates=canonical_gates(), nonfinite='immediate FAILED', oom='immediate FAILED (no save)',
                    headroom='resource stop below 1.5 GiB free VRAM (resumable only by explicit resume)',
                    deterioration='two consecutive evaluation deteriorations in severe degeneration, '
                                  'or reference CE above +0.1 at two consecutive points: EARLY_STOPPED',
                    update_cap=128),
        checkpoint=dict(backend='legacy', publications=list(CHECKPOINT_UPDATES), reserve_bytes=100 * 1024**3,
                        minimum_save_seconds=60, estimated_checkpoint_bytes=5636391032),
        budget=dict(PROPOSED_BUDGET, status='proposed; granted only by the user approval of this job hash',
                    feasibility=budget_analysis or 'see config/budget-analysis.json'),
        outputs=dict(root=str(ROOT / 'outputs/training' / RUN_ID), overwrite=False),
        recovery=dict(resume='only after a resource-guard or signal interruption, with explicit confirmation',
                      extension='never automatic', rollback='retain the verified foundation checkpoint; no promotion'))


def fingerprint(value) -> str:
    from codexa_workspace.sft_phase4_data import canonical_sha256
    return canonical_sha256(value)


def build_job(directory: Path, manifest: dict, *, step_zero: dict | None, **spec_options) -> dict:
    spec = build_spec(directory, manifest, step_zero=step_zero, **spec_options)
    job = dict(format=JOB_FORMAT, run_id=RUN_ID, spec=spec, spec_sha256=fingerprint(spec),
               code_identity=code_identity(), no_automatic_follow_up_training=True,
               launch_command=None, authorization='NOT AUTHORIZED: requires a separate explicit user grant (Step 5B).')
    job['job_sha256'] = fingerprint(job)
    return job


def verify_job(job: dict) -> dict:
    claimed = job['job_sha256']
    body = {k: v for k, v in job.items() if k != 'job_sha256'}
    if fingerprint(body) != claimed:
        raise LaunchRefused('Job hash mismatch.')
    if fingerprint(job['spec']) != job['spec_sha256']:
        raise LaunchRefused('Spec hash mismatch.')
    return job


# ---------------------------------------------------------------- prepare mode

@dataclass
class Check:
    name: str
    passed: bool
    detail: object = None


def prepare(job: dict, *, verify_manifest: Callable[[Path], dict] | None = None,
            measured: dict | None = None, environment: EnvironmentProvider | None = None) -> dict:
    """Validate everything needed for a future launch. Constructs no model and no optimizer."""
    from codexa_workspace.sft_phase4_data import digest, read
    from llm_architecture.model import ModelConfig
    checks: list[Check] = []

    def check(name, function):
        try:
            detail = function()
            checks.append(Check(name, True, detail))
        except Exception as error:  # every failure is reported, none is skipped
            checks.append(Check(name, False, f'{type(error).__name__}: {error}'))

    spec = job['spec']
    check('job and spec hashes', lambda: bool(verify_job(job)))
    check('non-runnable draft', lambda: _require(spec['runnable'] is False and job['launch_command'] is None
                                                 and spec['authorization_required'] is True, 'draft must be non-runnable'))
    check('manifest present and reproducible', lambda: (verify_manifest or _verify_manifest)(Path(spec['data']['manifest'])))
    check('manifest identity bound', lambda: _require(read(spec['data']['manifest'])['manifest_sha256']
                                                      == spec['data']['manifest_sha256'], 'manifest hash drift'))
    check('initialization checkpoint hash', lambda: _require(digest(spec['initialization']['path'])
                                                             == INITIALIZATION_SHA256 == spec['initialization']['sha256'],
                                                             'checkpoint hash mismatch'))
    check('tokenizer hash', lambda: _require(digest(spec['tokenizer']['path']) == spec['tokenizer']['sha256'],
                                             'tokenizer hash mismatch'))
    check('architecture identity', lambda: _architecture(spec))
    check('resource gates canonical', lambda: validate_gates(spec['guards']['gates']))
    check('hyperparameters', lambda: _hyperparameters(spec))
    check('output namespace fresh and isolated', lambda: _fresh_namespace(spec['outputs']))
    check('training configuration approved', lambda: _approved_configuration(spec))
    check('evaluation cadence and checkpoints', lambda: _require(spec['evaluation']['updates'] == list(EVALUATION_UPDATES)
                                                                 and spec['checkpoint']['publications'] == list(CHECKPOINT_UPDATES),
                                                                 'cadence drift'))
    check('step-zero sealed', lambda: _require(spec['evaluation']['step_zero_sha256'] is not None, 'step zero missing'))
    if measured is not None:
        try:
            items = resource_checks(spec['guards']['gates'], measured,
                                    required_disk_bytes=2 * spec['checkpoint']['estimated_checkpoint_bytes'])
        except Exception as error:  # a weakened or missing gate is a failed check, never a crash or a pass
            items = [dict(name='gates', passed=False, detail=f'{type(error).__name__}: {error}')]
        for item in items:
            checks.append(Check(f"resource {item['name']}", item['passed'], item))
    if environment is not None:
        check('environment contract', lambda: check_environment(environment, spec['environment']['required_context']))
    return dict(mode='prepare', optimizer_constructed=False, model_constructed=False,
                checks=[c.__dict__ for c in checks], passed=all(c.passed for c in checks))


def _fresh_namespace(outputs: dict) -> dict:
    """A new run root under outputs/training, never inside a historical run, never with results.

    The authorized launcher creates only config/ and supervisor/ before the trainer starts.
    """
    root = Path(outputs['root']).resolve()
    _require(outputs.get('overwrite') is False, 'overwrite must be False')
    _require(root.parent == (ROOT / 'outputs/training').resolve(), f'{root} is not a direct child of outputs/training')
    _require(not root.is_relative_to(FOUNDATION.resolve()) and root != FOUNDATION.resolve(), 'inside a historical run')
    if root.exists():
        produced = sorted(p.name for p in root.iterdir() if p.name not in ('config', 'supervisor'))
        _require(not produced, f'output namespace already has results {produced}; never overwrite')
    return dict(root=str(root), exists=root.exists())


def _approved_configuration(spec: dict) -> dict:
    from codexa_workspace.sft_pilot_launch import validate_training_config
    return validate_training_config(spec)


def _require(condition, message):
    if not condition:
        raise ValueError(message)
    return True


def _verify_manifest(path: Path) -> dict:
    from codexa_workspace.sft_phase4_data import verify_manifest
    return verify_manifest(path)


def _architecture(spec: dict) -> dict:
    import torch
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    with torch.device('meta'):
        model = LanguageModel(ModelConfig(**spec['model']))
    count = count_parameters(model)
    _require(count == spec['expected_parameters'] == EXPECTED_PARAMETERS, f'parameter count {count}')
    _require(model.lm_head.weight is model.token_embeddings.weight, 'tied embeddings')
    _require(spec['model']['context_length'] >= spec['data']['maximum_serialized_tokens'], 'context too short')
    return dict(parameters=count, tied=True, meta_device_only=True)


def _hyperparameters(spec: dict) -> dict:
    from llm_training.config import TrainingConfig
    config = TrainingConfig(**spec['training'])
    expected = dict(micro_batch_size=1, gradient_accumulation_steps=8, learning_rate=5e-6, weight_decay=.1,
                    warmup_steps=8, max_steps=128, gradient_clip=1.0, precision='bf16')
    _require(all(getattr(config, k) == v for k, v in expected.items()), 'training hyperparameter drift')
    optimization = spec['optimization']
    _require(optimization['optimizer'] == 'adamw8bit' and optimization['minimum_learning_rate'] == 5e-7
             and optimization['gradient_checkpointing'] is True and optimization['tf32'] is False
             and optimization['minimum_headroom_bytes'] >= canonical_gates()['in_run_free_vram_bytes'],
             'optimization drift')
    _require(config.max_steps * config.gradient_accumulation_steps * config.micro_batch_size
             == spec['data']['presentations'] == 1024, 'presentation count drift')
    return dict(config.__dict__)


# ---------------------------------------------------------------- stop policy

@dataclass
class StopPolicy:
    """Pure stop decisions for the future train mode; unit-tested with deterministic fixtures."""

    max_updates: int = 128
    minimum_headroom_bytes: int = 1536 * 1024**2
    reference_ce_increase_max: float = .1
    deterioration_patience: int = 2
    history: list = field(default_factory=list)
    reference_bad: int = 0
    degeneration_bad: int = 0

    def after_update(self, *, step: int, loss: float, gradient_norm: float, free_vram_bytes: int | None,
                     training_deadline_passed: bool) -> str | None:
        import math
        if not (math.isfinite(loss) and math.isfinite(gradient_norm)):
            return 'NONFINITE'
        if free_vram_bytes is not None and free_vram_bytes < self.minimum_headroom_bytes:
            return 'MEMORY_HEADROOM_GATE'
        if training_deadline_passed:
            return 'TRAINING_DEADLINE'
        if step >= self.max_updates:
            return 'STEP_CAP'
        return None

    def after_evaluation(self, *, step: int, step_zero: dict, result: dict) -> str | None:
        """Two consecutive deteriorations (degeneration or reference retention) stop the run."""
        if not result.get('finite', False):
            return 'NONFINITE'
        severe = result['generation']['conversation/greedy']['severe_degeneration']
        previous = self.history[-1]['severe'] if self.history else \
            step_zero['generation']['conversation/greedy']['severe_degeneration']
        self.degeneration_bad = self.degeneration_bad + 1 if severe > previous else 0
        over = any(result['reference']['sources'][s] - step_zero['reference']['sources'][s] > self.reference_ce_increase_max
                   for s in step_zero['reference']['sources'])
        self.reference_bad = self.reference_bad + 1 if over else 0
        self.history.append(dict(step=step, severe=severe, reference_over_limit=over))
        if self.degeneration_bad >= self.deterioration_patience:
            return 'DEGENERATION_DETERIORATION'
        if self.reference_bad >= self.deterioration_patience:
            return 'REFERENCE_CE_RETENTION'
        return None

    @staticmethod
    def classify_exception(error: BaseException) -> str:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return 'CUDA_OOM'
        if isinstance(error, FloatingPointError):
            return 'NONFINITE'
        if isinstance(error, KeyboardInterrupt):
            return 'INTERRUPTED'
        return type(error).__name__


def terminal_state(stop_reason: str | None) -> str:
    """Always one of the OS supervisor's terminal states (COMPLETED, EARLY_STOPPED, FAILED, INTERRUPTED)."""
    if stop_reason in (None, 'STEP_CAP', 'TRAINING_DEADLINE'):
        return 'COMPLETED'
    if stop_reason in ('NONFINITE', 'CUDA_OOM', 'FINAL_SAVE_RESERVE_EXHAUSTED', 'CHECKPOINT_FAILURE'):
        return 'FAILED'
    if stop_reason.startswith('INTERRUPTED'):
        return 'INTERRUPTED'
    return 'EARLY_STOPPED'


def outcome(stop_reason: str | None) -> str:
    """Experiment-level reading of a completed run: a deadline stop before the update cap is inconclusive."""
    return 'INCONCLUSIVE_DEADLINE' if stop_reason == 'TRAINING_DEADLINE' else (stop_reason or 'STEP_CAP')


# ---------------------------------------------------------------- authorization and train mode

def approval_valid(run_dir: Path, job: dict) -> dict:
    path = Path(run_dir) / 'config/approval.json'
    if not path.exists():
        raise LaunchRefused('No approval grant: training is not authorized.')
    approval = json.loads(path.read_text())
    if (approval.get('format') != APPROVAL_FORMAT or approval.get('approved') is not True
            or approval.get('job_sha256') != job['job_sha256'] or not str(approval.get('statement', '')).strip()):
        raise LaunchRefused('Approval grant does not match this frozen job.')
    return approval


def train_authorized(run_dir: Path, job: dict, *, environment: EnvironmentProvider,
                     measured: dict, verify_manifest: Callable[[Path], dict] | None = None,
                     trainer: Callable | None = None) -> dict:
    """The only path into training. Every precondition is re-checked; any failure refuses."""
    if os.environ.get('CODEXA_EXECUTION_CLASS') in NON_PRODUCTION_CLASSES and trainer is None:
        raise LaunchRefused('Non-production execution class may not start real training.')
    if os.environ.get('CODEXA_WAKEUP_AGENT'):
        raise LaunchRefused('A wakeup analysis agent may not start training.')
    verify_job(job)
    approval = approval_valid(run_dir, job)
    environment_record = check_environment(environment, job['spec']['environment']['required_context'])
    prepared = prepare(job, verify_manifest=verify_manifest, measured=measured)
    if not prepared['passed']:
        failed = [c['name'] for c in prepared['checks'] if not c['passed']]
        raise LaunchRefused(f'Launch gates failed: {failed}')
    if trainer is None:
        from codexa_workspace.sft_pilot_trainer import run_sft_pilot as trainer
    return trainer(job, Path(run_dir), approval=approval, environment=environment_record)


def launch_plan(run_dir: Path, job: dict, *, provider: str = 'codex') -> dict:
    """The launch.json the supervisor would consume (sft_pilot_launch.build_launch). Returned, never written."""
    from codexa_workspace.sft_pilot_launch import build_launch
    plan = build_launch(Path(run_dir), job, provider=provider, written_utc='not written')
    return dict(plan, status='NOT AUTHORIZED - launched only by sft_pilot_launch.launch_authorized after approval')
