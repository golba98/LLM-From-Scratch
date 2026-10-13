"""Train mode for the conversational SFT pilot. NOT AUTHORIZED: never run during preparation.

Reachable only through ``sft_pilot_runtime.train_authorized`` (approval grant, environment
contract and resource gates re-checked). The run directory layout matches what the
existing OS supervisor reads: ``metrics/{heartbeat,progress,budget,report}.json`` and a
verified ``final`` recovery role. Heavy dependencies are injected through ``Components`` so
the control flow is unit-tested with fakes, without constructing a torch optimizer or
running backpropagation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time
from typing import Callable

from codexa_workspace import training_telemetry
from codexa_workspace.sft_pilot_runtime import StopPolicy, outcome, terminal_state


@dataclass
class Components:
    """Everything that touches CUDA, autograd or disk-heavy state, injectable for tests."""

    heartbeat: Callable          # (path) -> object with set(phase=None, **values) and stop()
    budget: Callable             # (seconds, reserve) -> object with remaining, reserve_seconds, training_deadline, document()
    datasets: Callable           # (spec) -> (train_dataset, validation_dataset)
    model: Callable              # (spec) -> model with initialization weights loaded
    optimizer: Callable          # (model, spec) -> optimizer  (constructed only here, after every check)
    train_loop: Callable         # (model, optimizer, train_dataset, spec, budget, on_update, stop_requested) -> state
    evaluate: Callable           # (model, validation_dataset, spec, step) -> evaluation summary dict
    store: Callable              # (run_dir, spec) -> object with save(model, optimizer, state, *, final) -> record, verify(record, state) -> dict
    free_vram: Callable          # () -> int | None
    write: Callable              # (path, document) -> None
    viewer: Callable | None = None   # (progress_path, title) -> process; the required visible dashboard


def run_sft_pilot(job: dict, run_dir: Path, *, approval: dict, environment: dict,
                  components: Components | None = None) -> dict:
    spec = job['spec']
    parts = components or default_components()
    run_dir = Path(run_dir)
    metrics = run_dir / 'metrics'
    if (metrics / 'report.json').exists():
        raise FileExistsError('Fresh run namespace required; never overwrite a run.')
    metrics.mkdir(parents=True, exist_ok=True)
    evaluations = run_dir / 'evaluations'
    # Per-update telemetry (append-only JSONL). A telemetry failure is recorded, never fatal to training.
    telemetry_errors = []
    telemetry = training_telemetry.TelemetryWriter(
        metrics / training_telemetry.FILENAME, run_id=spec['run_name'], job_kind='sft',
        clip=spec['training'].get('gradient_clip'))

    def observe(method, *args, **values):
        try:
            method(*args, **values)
        except Exception as error:  # noqa: BLE001 - observability must not stop a run
            telemetry_errors.append(f'{type(error).__name__}: {error}'[:500])
    budget = parts.budget(spec['budget']['total_seconds'], spec['budget']['finalization_reserve_seconds'])
    heartbeat = parts.heartbeat(metrics / 'heartbeat.json')
    policy = StopPolicy(max_updates=spec['training']['max_steps'],
                        minimum_headroom_bytes=spec['optimization']['minimum_headroom_bytes'],
                        reference_ce_increase_max=spec['acceptance']['reference_ce_increase_max'])
    step_zero = json.loads(Path(spec['evaluation']['step_zero_path']).read_text())
    report = dict(run_name=spec['run_name'], job_sha256=job['job_sha256'], approval=approval,
                  environment=environment, initialized_from=spec['initialization'], exact_resume=False,
                  state='STARTING', checkpoints=[], evaluations=[], final_checkpoint_verified=False,
                  telemetry=str(metrics / training_telemetry.FILENAME), telemetry_errors=telemetry_errors)
    progress = dict(phase='conversational-sft-pilot', run_name=spec['run_name'], state='STARTING', step=0,
                    total_steps=spec['training']['max_steps'], pid=os.getpid())
    stop_reason = None
    model = optimizer = state = None

    def document():
        parts.write(metrics / 'progress.json', progress)
        parts.write(metrics / 'budget.json', budget.document())

    def evaluate(step: int) -> None:
        nonlocal stop_reason
        heartbeat.set('GENERATION_EVAL', optimizer_step=step)
        started = time.monotonic()
        summary = parts.evaluate(model, validation, spec, step)
        parts.write(evaluations / f'step-{step:05d}.json', summary)
        observe(telemetry.evaluation, step, summary, seconds=time.monotonic() - started)
        report['evaluations'].append(dict(step=step, summary_path=str(evaluations / f'step-{step:05d}.json')))
        decision = policy.after_evaluation(step=step, step_zero=step_zero, result=summary)
        if decision:
            stop_reason = decision
        heartbeat.set('TRAINING')

    def save(role: str, step: int, final: bool) -> dict:
        started = time.monotonic()
        try:
            record = store.save(model, optimizer, state_holder['state'], final=final)
        except Exception as error:
            observe(telemetry.checkpoint, step, role=role, record=None, seconds=time.monotonic() - started,
                    error=f'{type(error).__name__}: {error}'[:500])
            raise
        observe(telemetry.checkpoint, step, role=role, record=record, seconds=time.monotonic() - started)
        return record

    def on_update(step: int, loss: float, gradient_norm: float, record=None) -> None:
        nonlocal stop_reason
        progress.update(state='RUNNING', step=step, training_loss=loss, gradient_norm=gradient_norm,
                        remaining_seconds=budget.remaining)
        heartbeat.set(optimizer_step=step, last_step_utc=time.time())
        free_vram = parts.free_vram()
        observe(telemetry.update, record if record is not None else
                dict(optimizer_step=step, training_loss=loss, gradient_norm=gradient_norm), free_vram_bytes=free_vram)
        decision = policy.after_update(step=step, loss=loss, gradient_norm=gradient_norm,
                                       free_vram_bytes=free_vram,
                                       training_deadline_passed=time.monotonic() >= budget.training_deadline)
        if decision in ('NONFINITE',):
            raise FloatingPointError('Nonfinite loss or gradient norm.')
        if decision and decision != 'STEP_CAP':
            stop_reason = decision
        if step in spec['evaluation']['updates'] and step != 0 and stop_reason is None:
            evaluate(step)
        if step in [u for u in spec['checkpoint']['publications'] if isinstance(u, int)] and stop_reason is None:
            heartbeat.set('CHECKPOINTING')
            report['checkpoints'].append(save('intermediate', step, final=False))
            heartbeat.set('TRAINING')
        document()

    state_holder = {}
    viewer = None
    try:
        heartbeat.set('STARTING')
        document()
        if spec.get('viewer_required') and parts.viewer is not None:
            # The 120x30 Rich dashboard (LLM_TRAINING_VIEWER_COMMAND); no update runs without it.
            viewer = parts.viewer(metrics / 'progress.json', f"Codexa SFT pilot: {spec['run_name']}")
        train, validation = parts.datasets(spec)
        model = parts.model(spec)
        store = parts.store(run_dir, spec)
        optimizer = parts.optimizer(model, spec)
        heartbeat.set('TRAINING')

        def stop_requested() -> bool:
            return stop_reason is not None

        try:
            state = parts.train_loop(model, optimizer, train, spec, budget, on_update, stop_requested, state_holder)
        except TimeoutError:
            # train_model rolls an interrupted accumulation back to the last committed update.
            stop_reason = stop_reason or 'TRAINING_DEADLINE'
            current = state_holder['state']
            state = current.to_dict() if hasattr(current, 'to_dict') else dict(current)
        if stop_reason is None and state['optimizer_step'] < spec['training']['max_steps']:
            # train_model also returns early, without an exception, once its deadline has passed.
            stop_reason = ('TRAINING_DEADLINE' if time.monotonic() >= budget.training_deadline
                           else 'TRAINING_INCOMPLETE')
        heartbeat.set('FINALIZING')
        if stop_reason is None and state['optimizer_step'] not in spec['evaluation']['updates']:
            evaluate(state['optimizer_step'])
        if budget.remaining <= spec['checkpoint']['minimum_save_seconds']:
            stop_reason = stop_reason or 'FINAL_SAVE_RESERVE_EXHAUSTED'
        else:
            # Save the live training state object (as at update 64); ``state`` is its JSON summary.
            final_state = state_holder['state']
            record = save('final', state['optimizer_step'], final=True)
            report['checkpoints'].append(record)
            report['final_checkpoint'] = store.verify(record, final_state)
            report['final_checkpoint_verified'] = True
        report.update(state=terminal_state(stop_reason), stop_reason=stop_reason or 'STEP_CAP',
                      outcome=outcome(stop_reason), training_state=state)
    except BaseException as error:
        reason = StopPolicy.classify_exception(error)
        report.update(state='INTERRUPTED' if reason == 'INTERRUPTED' else 'FAILED', stop_reason=reason,
                      error_type=type(error).__name__, error=str(error)[:2000],
                      note='No weights are saved after an exception; the last verified checkpoint is retained.')
        if not isinstance(error, (FloatingPointError, KeyboardInterrupt)) and reason != 'CUDA_OOM':
            raise
    finally:
        progress.update(state=report['state'], stop_reason=report.get('stop_reason'))
        report['viewer_closed'] = viewer is not None and viewer.poll() is not None
        heartbeat.set('FINALIZING', terminal_state=report['state'])
        heartbeat.stop()
        observe(telemetry.stop, state=report['state'], reason=report.get('stop_reason'),
                step=(report.get('training_state') or {}).get('optimizer_step', progress.get('step')),
                error=report.get('error'), error_type=report.get('error_type'))
        observe(telemetry.close)
        parts.write(metrics / 'report.json', report)
        document()
        del model, optimizer
        gc.collect()
    return report


# ---------------------------------------------------------------- real components (authorized runs only)

def default_components(device: str = 'cuda') -> Components:
    """The real components. Production uses CUDA; the wiring smoke test runs them on CPU with a toy model."""
    import torch
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_tokenizer.tokenizer import load_tokenizer
    from llm_training.checkpointing import SchedulerState, build_checkpoint_payload
    from llm_training.config import ProjectConfig, TrainingConfig
    from llm_training.pretrain_run import FixedOrderSampler, Heartbeat, verify_final
    from llm_training.recovery import RecoveryStore
    from llm_training.session import SessionBudget, atomic_document
    from llm_training.training import (TrainingState, create_adamw_optimizer, resolve_precision,
                                       set_deterministic_seed, train_model)
    from codexa_workspace import sft_pilot_evaluation as evaluation

    def datasets(spec):
        tokenizer = load_tokenizer(spec['tokenizer']['path'])
        train = IndexedChatDataset(Path(spec['data']['train']), tokenizer, context=512)
        validation = IndexedChatDataset(Path(spec['data']['validation']), tokenizer, context=512)
        if (train.sha256, validation.sha256) != (spec['data']['train_sha256'], spec['data']['validation_sha256']):
            raise ValueError('Dataset fingerprint drift.')
        datasets.tokenizer = tokenizer
        return train, validation

    def model(spec):
        set_deterministic_seed(spec['training']['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        network = LanguageModel(ModelConfig(**spec['model']))
        payload = read_native_checkpoint(spec['initialization']['path'],
                                         expected_sha256=spec['initialization']['sha256'])
        network.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        gc.collect()
        if count_parameters(network) != spec['expected_parameters']:
            raise ValueError('Architecture drift.')
        network.set_gradient_checkpointing(True)
        return network.to(device)

    def optimizer(network, spec):
        o = spec['optimization']
        return create_adamw_optimizer(network, learning_rate=spec['training']['learning_rate'],
                                      weight_decay=spec['training']['weight_decay'],
                                      optimizer_name=o['optimizer'], betas=tuple(o['betas']), epsilon=o['epsilon'])

    def train_loop(network, optim, train, spec, budget, on_update, stop_requested, holder):
        manifest = json.loads(Path(spec['data']['manifest']).read_text())
        positions = [i for order in manifest['orders'] for i in order['indices']]
        loader = torch.utils.data.DataLoader(train, batch_size=1, sampler=FixedOrderSampler(positions),
                                             collate_fn=collate_chat, pin_memory=device == 'cuda')
        t = spec['training']
        state = TrainingState()
        holder['state'] = state

        def callback(current, metrics):
            holder['state'] = current
            on_update(current.optimizer_step, metrics.training_loss, metrics.gradient_norm, metrics)

        final, _ = train_model(network, loader, optim, device=torch.device(device),
                               precision=resolve_precision(t['precision'], torch.device(device)),
                               max_steps=t['max_steps'], gradient_accumulation_steps=t['gradient_accumulation_steps'],
                               gradient_clip=t['gradient_clip'], warmup_steps=t['warmup_steps'],
                               peak_learning_rate=t['learning_rate'],
                               minimum_learning_rate=spec['optimization']['minimum_learning_rate'],
                               state=state, seed=t['seed'], run_name=spec['run_name'],
                               deadline_monotonic=budget.training_deadline,
                               loss_normalization=spec['optimization']['loss_normalization'],
                               on_optimizer_step=callback, stop_requested=stop_requested, max_epochs=1)
        holder['state'] = final
        return final.to_dict()

    def evaluate(network, validation, spec, step):
        tokenizer = datasets.tokenizer
        decodes = evaluation.decoding_configs()
        # Intermediate updates keep every stop/acceptance signal; the full schedule runs at the end.
        full = step not in spec['evaluation'].get('reduced_updates', ())
        if not full:
            decodes = {'greedy': decodes['greedy']}
        rows = evaluation.generate_suite(network, tokenizer, evaluation.conversational_cases(),
                                         suite='conversation', decodes=decodes, device=device)
        if full:
            rows += evaluation.generate_suite(network, tokenizer, evaluation.historical_cases(),
                                              suite='historical', decodes=decodes, device=device)
        gold, _ = evaluation.gold_validation(network, validation, device=device)
        reference = evaluation.reference_validation(network, spec['data']['reference'], device=device)
        finite = all(bool(torch.isfinite(p).all()) for p in network.parameters())
        network.train()
        return dict(step=step, schedule='full' if full else 'reduced', generation=evaluation.summarize_generations(rows),
                    gold=gold, reference=reference, finite=finite, rows=rows)

    class Store:
        def __init__(self, run_dir, spec):
            self.spec = spec
            self.store = RecoveryStore(Path(run_dir) / 'checkpoints', reserve_bytes=spec['checkpoint']['reserve_bytes'])
            self.project = ProjectConfig(ModelConfig(**spec['model']), TrainingConfig(**spec['training']))
            from codexa_workspace.sft_pilot_runtime import fingerprint
            self.config_hash = fingerprint(spec)

        def save(self, network, optim, state, *, final):
            t = self.spec['training']
            payload = build_checkpoint_payload(model=network, optimizer=optim, scaler=None, state=state,
                scheduler=SchedulerState(t['warmup_steps'], t['max_steps'], t['learning_rate'],
                                         self.spec['optimization']['minimum_learning_rate']),
                config=self.project, run_name=self.spec['run_name'], run_id=self.spec['run_name'],
                tokenizer_reference=self.spec['tokenizer']['path'], tokenizer_sha256=self.spec['tokenizer']['sha256'],
                resume_context=dict(configuration_sha256=self.config_hash,
                                    data_manifest_sha256=self.spec['data']['manifest_sha256']))
            try:
                return self.store.save(payload, final=final)
            finally:
                del payload

        def verify(self, record, state):
            return verify_final(self.store, record, run_id=self.spec['run_name'],
                                step=record['optimizer_step'], config_hash=self.config_hash)

    def free_vram():
        return torch.cuda.mem_get_info()[0] if device == 'cuda' and torch.cuda.is_available() else None

    def heartbeat(path):
        return Heartbeat(path).start()

    def viewer(status, title):
        from llm_training.conversational_training import start_chat_viewer
        return start_chat_viewer(status, title=title)

    return Components(heartbeat=heartbeat, budget=SessionBudget.start, datasets=datasets, model=model,
                      optimizer=optimizer, train_loop=train_loop, evaluate=evaluate, store=Store,
                      free_vram=free_vram, write=atomic_document, viewer=viewer if device == 'cuda' else None)


def main(arguments: list[str] | None = None) -> int:  # pragma: no cover - supervisor entry, authorized runs only
    """Supervisor entry. Refuses without a matching approval grant (train_authorized re-checks all gates).

    Codex's guard runs first, before torch or the optimizer factory is imported: an implicit,
    historical or dirty-source environment exits here, before any model or optimizer exists.
    """
    from codexa_workspace import sft_pilot_runtime as runtime
    runtime.training_guard().require_environment('new')
    parser = argparse.ArgumentParser(description='Approved SFT pilot trainer entry (supervisor child only).')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(arguments)
    job = json.loads((args.run_dir / 'config/job.json').read_text())
    report = runtime.train_authorized(args.run_dir, job, environment=runtime.TrainingGuardProvider('new'),
                                      measured=runtime.measure_resources(args.run_dir))
    return 0 if report['state'] in ('COMPLETED', 'EARLY_STOPPED') else 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
