"""Bounded real-model evidence, corrective pilots and frozen readiness gates."""

from dataclasses import replace
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from llm_training.session import atomic_document, SessionBudget
from codexa_workspace.chat_preparation import (audit, compare_candidates, consume_gpu_budget,
    experiment_environment, make_spec, preflight, ranking)


def tensor_fingerprint(value) -> str:
    """Hash every tensor byte and scalar in a nested optimizer/model state."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def retention_and_sampled(root: Path, config: dict, directory: Path) -> dict:
    """Fixed replay diagnostic and secondary decoding; neither is a held-out benchmark."""
    import numpy as np
    from llm_architecture.checkpointing import read_native_checkpoint, file_sha256
    from llm_architecture.model import LanguageModel, ModelConfig
    from llm_training.training import evaluate, resolve_precision
    from codexa_workspace.assets import resolve_input
    from codexa_workspace.chat_evaluation import evaluate_candidate
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_tokenizer.tokenizer import load_tokenizer
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Final diagnostics require CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.')
    torch.use_deterministic_algorithms(True)
    deterministic_reports = []
    for value in config['candidates']:
        path = resolve_input(root, value)
        target = directory / 'deterministic-candidates' / path.parent.name
        if (target / 'report.json').exists():
            deterministic_reports.append(json.loads((target / 'report.json').read_text()))
            continue
        started = time.monotonic()
        try:
            deterministic_reports.append(evaluate_candidate(path, resolve_input(root, config['tokenizer']), target))
        finally:
            consume_gpu_budget(directory, started, label='deterministic-baseline-' + path.parent.name,
                               ceiling=config['preparation_gpu_seconds'])
    atomic_document(directory / 'deterministic-candidate-comparison.json', deterministic_reports)
    reports = []
    for value in config['candidates']:
        path = resolve_input(root, value)
        target = directory / 'sampled-candidates' / path.parent.name
        if (target / 'report.json').exists():
            reports.append(json.loads((target / 'report.json').read_text()))
            continue
        started = time.monotonic()
        try:
            reports.append(evaluate_candidate(path, resolve_input(root, config['tokenizer']), target, sampled=True))
        finally:
            consume_gpu_budget(directory, started, label='sampled-' + path.parent.name, ceiling=config['preparation_gpu_seconds'])
    atomic_document(directory / 'sampled-comparison.json', reports)
    stream = resolve_input(root, 'data/tokenized/base-v1/mixed/train.bin')
    offsets = [0, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 4000000000]
    tokens = np.memmap(stream, dtype=np.uint16, mode='r')
    windows = np.stack([tokens[o:o + 513].astype(np.int64) for o in offsets])
    window_digest = hashlib.sha256(windows.tobytes()).hexdigest()
    loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(windows[:, :-1].copy()),
        torch.from_numpy(windows[:, 1:].copy())), batch_size=1)
    audit_report = json.loads((directory / 'data-audit.json').read_text())
    validation = IndexedChatDataset(Path(audit_report['splits']['validation']['path']),
        load_tokenizer(resolve_input(root, config['tokenizer'])))
    validation_loader = DataLoader(Subset(validation, audit_report['splits']['validation']['possibly_clean_indices']),
                                   batch_size=1, collate_fn=collate_chat)
    choices = [resolve_input(root, value) for value in config['candidates']]
    pilots_path = directory / 'pilot-comparison.json'
    if pilots_path.exists():
        choices.append(Path(json.loads(pilots_path.read_text())[-1]['checkpoint']))
    losses = []
    for path in choices:
        started = time.monotonic()
        model = None
        try:
            payload = read_native_checkpoint(path)
            model = LanguageModel(ModelConfig(**payload['config']['model']))
            model.load_state_dict(payload['model_state_dict'], strict=True)
            del payload
            model.to('cuda')
            loss, count = evaluate(model, loader, device=torch.device('cuda'),
                precision=resolve_precision('bf16', torch.device('cuda')))
            val_loss, val_count = evaluate(model, validation_loader, device=torch.device('cuda'),
                precision=resolve_precision('bf16', torch.device('cuda')))
            losses.append(dict(checkpoint=str(path), sha256=file_sha256(path), loss=loss, tokens=count,
                               common_conversational_validation_loss=val_loss, common_validation_tokens=val_count,
                               seconds=time.monotonic() - started))
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()
            consume_gpu_budget(directory, started, label='retention-' + path.parent.name, ceiling=config['preparation_gpu_seconds'])
    report = dict(path=str(stream), offsets=offsets, window_sha256=window_digest, results=losses,
        common_validation_indices=audit_report['splits']['validation']['possibly_clean_indices'],
        population='Eight fixed 512-target-token windows from previously trained base TRAINING stream.',
        limitation='Retention diagnostic only: exposed training material, not missing base validation or proven generalization.')
    atomic_document(directory / 'retention-diagnostic.json', report)
    return report


def benchmark(root: Path, config: dict, directory: Path, candidate: dict, *, deterministic: bool = False,
              memory_cap: bool = False) -> dict:
    """Benchmark complete local conversations, including the worst accepted shape."""
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_architecture.model import LanguageModel, ModelConfig
    from llm_tokenizer.tokenizer import load_tokenizer
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_training.conversational_training import start_chat_viewer
    from llm_training.training import create_adamw_optimizer, train_model, resolve_precision, evaluate
    from codexa_workspace.assets import resolve_input
    target = directory / ('benchmark-headroom' if memory_cap else 'benchmark-deterministic' if deterministic else 'benchmark')
    target.mkdir(parents=True, exist_ok=False)
    status = target / 'progress.json'
    atomic_document(status, dict(phase='conversational', state='STARTING', pid=os.getpid(), step=0,
        total_steps=6, run_name='934M bounded benchmark', kind='934M decoder', lineage=candidate['checkpoint']))
    viewer = start_chat_viewer(status, title='Codexa 934M bounded optimization benchmark')
    started = time.monotonic()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    payload = read_native_checkpoint(Path(candidate['checkpoint']))
    tokenizer = load_tokenizer(resolve_input(root, config['tokenizer']))
    data = IndexedChatDataset(resolve_input(root, config['data']) / 'train.jsonl', tokenizer)
    audit_report = json.loads((directory / 'data-audit.json').read_text())
    val = IndexedChatDataset(resolve_input(root, config['data']) / 'validation.jsonl', tokenizer)
    validation = DataLoader(Subset(val, audit_report['splits']['validation']['possibly_clean_indices'][:8]),
                            batch_size=1, collate_fn=collate_chat)
    short = [i for i, entry in enumerate(data.index) if 450 <= entry.tokens <= 512][:8]
    long = sorted(range(len(data)), key=lambda i: data.index[i].tokens, reverse=True)[:8]
    if len(short) < 8:
        raise ValueError('No complete short-sequence benchmark population available.')
    free_initial, total_memory = torch.cuda.mem_get_info()
    fraction = (free_initial - config['minimum_gpu_headroom_bytes'] - 64 * 1024**2) / total_memory
    if memory_cap:
        torch.cuda.set_per_process_memory_fraction(fraction)
    model = LanguageModel(ModelConfig(**payload['config']['model'])).to('cuda')
    policy = resolve_precision('bf16', torch.device('cuda'))
    results = []
    options = [('fixed2048-long-m1', long, 1, 2048, True, 0, True),
               ('dynamic-long-m1', long, 1, None, True, 0, True),
               ('dynamic-short-m1', short, 1, None, True, 0, True),
               ('dynamic-short-m2', short, 2, None, True, 0, True),
               ('dynamic-short-workers2', short, 1, None, True, 2, True),
               ('dynamic-short-no-checkpointing', short, 1, None, False, 0, True),
               ('dynamic-short-unpinned', short, 1, None, True, 0, False)]
    if deterministic:
        options = [('dynamic-long-m1', long, 1, None, True, 0, True),
                   ('dynamic-short-m1', short, 1, None, True, 0, True),
                   ('dynamic-mixed-accum32', short + long, 1, None, True, 0, True)]
    optimizer = None
    try:
        for name, indices, microbatch, fixed, checkpointing, workers, pinned in options:
            del optimizer
            optimizer = None
            gc.collect(); torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            # Persistent model already occupies VRAM. Additional grads/state ~5.7 GB.
            minimum_incremental = 5700000000 + (1500000000 if not checkpointing else 500000000)
            if free < minimum_incremental + config['minimum_gpu_headroom_bytes']:
                results.append(dict(name=name, skipped=True, reason='predicted memory/headroom gate', free_bytes=free))
                continue
            print('Benchmarking', name, flush=True)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            model.set_gradient_checkpointing(checkpointing)
            optimizer = create_adamw_optimizer(model, learning_rate=1e-5, weight_decay=.1, optimizer_name='adamw8bit')
            before = model.final_norm.weight.detach().clone()
            loader = DataLoader(Subset(data, indices), batch_size=microbatch, num_workers=workers,
                collate_fn=partial(collate_chat, fixed_length=fixed), pin_memory=pinned,
                generator=torch.Generator().manual_seed(42))
            def update(state, metrics):
                if deterministic:
                    torch.cuda.empty_cache()
                atomic_document(status, dict(phase='conversational', state='RUNNING', pid=os.getpid(),
                    step=state.optimizer_step, total_steps=6, run_name=name, kind='934M decoder',
                    lineage=candidate['checkpoint'], training_loss=metrics.training_loss,
                    total_tokens_seen=state.tokens_seen, tokens_per_second=metrics.tokens_per_second,
                    eta_seconds=metrics.step_time_seconds * (6 - state.optimizer_step),
                    peak_vram_bytes=metrics.peak_reserved_vram_bytes, viewer_closed=viewer.poll() is not None))
            try:
                state, records = train_model(model, loader, optimizer, device=torch.device('cuda'), precision=policy,
                    max_steps=6, gradient_accumulation_steps=32 if name == 'dynamic-mixed-accum32' else 1,
                    gradient_clip=1, warmup_steps=0,
                    peak_learning_rate=1e-5, loss_normalization='supervised_tokens', on_optimizer_step=update)
                steady = records[2:]
                elapsed = sum(r.step_time_seconds for r in steady)
                validation_start = time.monotonic()
                validation_loss, validation_tokens = evaluate(model, validation, device=torch.device('cuda'), precision=policy)
                free_after, _ = torch.cuda.mem_get_info()
                result = dict(name=name, skipped=False, microbatch=microbatch, fixed_padding=fixed,
                    gradient_checkpointing=checkpointing, workers=workers, pinned=pinned,
                    actual_sequence_lengths=[data.index[i].tokens for i in indices],
                    cold_step_seconds=records[0].step_time_seconds, steady_step_seconds=statistics.median(r.step_time_seconds for r in steady),
                    assistant_tokens_per_second=sum(r.tokens_processed for r in steady) / elapsed,
                    context_tokens_per_second=sum(r.context_tokens_processed for r in steady) / elapsed,
                    optimizer_updates_per_second=len(steady) / elapsed,
                    peak_allocated_bytes=max(r.peak_allocated_vram_bytes for r in records),
                    peak_reserved_bytes=max(r.peak_reserved_vram_bytes for r in records), free_bytes_after=free_after,
                    headroom_passed=free_after >= config['minimum_gpu_headroom_bytes'],
                    finite_gradients=all(math.isfinite(r.gradient_norm) for r in records),
                    parameter_update=not torch.equal(before, model.final_norm.weight),
                    validation_loss=validation_loss, validation_tokens=validation_tokens,
                    validation_seconds=time.monotonic() - validation_start,
                    metrics=[r.to_dict() for r in records])
            except torch.cuda.OutOfMemoryError as error:
                result = dict(name=name, skipped=False, failed=True, reason='CUDA OOM', error=str(error))
                model.zero_grad(set_to_none=True)
            results.append(result)
            atomic_document(target / 'results.partial.json', results)
        report = dict(results=results, seconds=time.monotonic() - started,
            allocator_memory_cap=memory_cap, memory_fraction=fraction if memory_cap else None,
            free_gpu_bytes_before_model=free_initial,
            deterministic_algorithms=deterministic, cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            optimizer_alternatives={'adamw_full_precision': 'Not attempted: model+grad+FP32 Adam state alone ~14.95 GB before activations.',
                'torch_compile': 'Disabled; optional compile benchmark deferred until an initialization meets quality gates.',
                'tf32': 'Not enabled: BF16 matrix operations are the selected arithmetic path.'})
        atomic_document(target / 'report.json', report)
        return report
    finally:
        atomic_document(status, dict(phase='conversational', state='COMPLETED', pid=os.getpid(), step=6,
                                     total_steps=6, run_name='benchmark', kind='934M decoder'))
        del model, optimizer, payload
        gc.collect(); torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(previous_determinism)
        consume_gpu_budget(directory, started, label='benchmark', ceiling=config['preparation_gpu_seconds'])


def recovery_smoke(root: Path, config: dict, directory: Path, candidate: dict, *, deterministic: bool = False) -> dict:
    """Compare uninterrupted and full-state restored 934M optimizer continuations."""
    from llm_architecture.model import LanguageModel
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_tokenizer.tokenizer import load_tokenizer
    from llm_training.config import load_config
    from llm_training.training import create_adamw_optimizer, train_model, resolve_precision, set_deterministic_seed
    from llm_training.checkpointing import build_checkpoint_payload, SchedulerState, load_checkpoint
    from llm_training.recovery import RecoveryStore
    from llm_training.conversational_training import start_chat_viewer
    from codexa_workspace.assets import resolve_input
    target = directory / ('recovery-deterministic' if deterministic else 'recovery')
    target.mkdir(parents=True, exist_ok=False)
    status = target / 'progress.json'
    atomic_document(status, dict(phase='conversational', state='STARTING', pid=os.getpid(), step=0,
        total_steps=4, run_name='934M full-state recovery', kind='934M decoder', lineage=candidate['checkpoint']))
    start_chat_viewer(status, title='Codexa 934M checkpoint recovery verification')
    started = time.monotonic()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True)
    set_deterministic_seed(42)
    project = load_config(root / config['architecture'])
    project = replace(project, training=replace(project.training, learning_rate=1e-5, warmup_steps=0,
        max_steps=4, gradient_accumulation_steps=1, checkpoint_interval=2, evaluation_interval=2))
    data = IndexedChatDataset(resolve_input(root, config['data']) / 'train.jsonl',
        load_tokenizer(resolve_input(root, config['tokenizer'])))
    indices = [i for i, entry in enumerate(data.index) if 450 <= entry.tokens <= 512][:8]
    loader = DataLoader(Subset(data, indices), batch_size=1, collate_fn=collate_chat,
                        generator=torch.Generator().manual_seed(42), pin_memory=True)
    model = LanguageModel(project.model).to('cuda'); model.set_gradient_checkpointing(True)
    initial = read_native_checkpoint(Path(candidate['checkpoint']))
    model.load_state_dict(initial['model_state_dict'], strict=True)
    del initial
    optimizer = create_adamw_optimizer(model, learning_rate=1e-5, weight_decay=.1, optimizer_name='adamw8bit')
    scheduler = SchedulerState(0, 4, 1e-5, 1e-6)
    policy = resolve_precision('bf16', torch.device('cuda'))
    store = RecoveryStore(target / 'recoverable')
    def train(state=None, micro_steps=None):
        return train_model(model, loader, optimizer, device=torch.device('cuda'), precision=policy,
            max_steps=4, max_micro_steps=micro_steps, gradient_accumulation_steps=1, gradient_clip=1,
            warmup_steps=0, peak_learning_rate=1e-5, state=state, loss_normalization='supervised_tokens')
    def payload(state):
        return build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
            scheduler=scheduler, config=project, run_name='934M-recovery-smoke', run_id='isolated-recovery',
            tokenizer_reference=str(resolve_input(root, config['tokenizer'])), tokenizer_sha256=config['tokenizer_sha256'],
            resume_context={'configuration_sha256': 'recovery-smoke', 'data_manifest_sha256':
                json.loads((directory / 'data-audit.json').read_text())['manifest_sha256']})
    try:
        state, first = train(micro_steps=2)
        before_weights = tensor_fingerprint(model.state_dict())
        before_optimizer = tensor_fingerprint(optimizer.state_dict())
        saved = store.save(payload(state), best=True)
        reference_state, reference_metrics = train(state=state)
        reference_weights = tensor_fingerprint(model.state_dict())
        reference_optimizer = tensor_fingerprint(optimizer.state_dict())
        reference_counters = reference_state.to_dict()
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()
        model = LanguageModel(project.model).to('cuda'); model.set_gradient_checkpointing(True)
        optimizer = create_adamw_optimizer(model, learning_rate=1e-5, weight_decay=.1, optimizer_name='adamw8bit')
        restore_start = time.monotonic()
        restored = load_checkpoint(store.resolve(), model=model, optimizer=optimizer, scaler=None, expected_config=project)
        restore_seconds = time.monotonic() - restore_start
        restored_weights_exact = tensor_fingerprint(model.state_dict()) == before_weights
        restored_optimizer_exact = tensor_fingerprint(optimizer.state_dict()) == before_optimizer
        resumed_state, resumed_metrics = train(state=restored.state)
        result = dict(weights_exact_after_restore=restored_weights_exact, optimizer_exact_after_restore=restored_optimizer_exact,
            scheduler_exact=restored.scheduler == scheduler, counters_exact=resumed_state.to_dict() == reference_counters,
            continuation_weights_bitwise_equal=tensor_fingerprint(model.state_dict()) == reference_weights,
            continuation_optimizer_bitwise_equal=tensor_fingerprint(optimizer.state_dict()) == reference_optimizer,
            losses_reference=[m.training_loss for m in reference_metrics],
            losses_resumed=[m.training_loss for m in resumed_metrics],
            restore_seconds=restore_seconds, save=saved, finite_gradients=all(math.isfinite(m.gradient_norm) for m in resumed_metrics))
        result['passed'] = all(result[k] for k in ('weights_exact_after_restore', 'optimizer_exact_after_restore',
            'scheduler_exact', 'counters_exact', 'continuation_weights_bitwise_equal', 'continuation_optimizer_bitwise_equal', 'finite_gradients'))
        result['final_save'] = store.save(payload(resumed_state), final=True)
        result['seconds'] = time.monotonic() - started
        result['deterministic_algorithms'] = deterministic
        result['cublas_workspace_config'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        atomic_document(target / 'report.json', result)
        return result
    finally:
        atomic_document(status, dict(phase='conversational', state='COMPLETED', pid=os.getpid(), step=4,
                                    total_steps=4, run_name='recovery smoke', kind='934M decoder'))
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(previous_determinism)
        consume_gpu_budget(directory, started, label='full-model-recovery', ceiling=config['preparation_gpu_seconds'])


def pilots(root: Path, config: dict, directory: Path, candidate: dict) -> list[dict]:
    """Corrective phase if all baselines fail; otherwise comparable LR/mixture probes."""
    from llm_training.chat_run import run_chat
    from llm_training.recovery import RecoveryStore
    from codexa_workspace.chat_evaluation import evaluate_candidate, evaluate_loaded, make_suite
    target = directory / 'pilots'
    target.mkdir(parents=True, exist_ok=True)
    passed = candidate['metrics']['entry_gate_passed']
    choices = [(1e-5, False, 128)] if not passed else [(lr, False, 32) for lr in config['learning_rate_candidates']] + [(1e-5, True, 32)]
    first_report = target / 'corrective-00/report.json'
    if (not passed and first_report.exists()
            and json.loads(first_report.read_text()).get('stop_reason') == 'MEMORY_HEADROOM_GATE'
            and os.environ.get('PYTORCH_ALLOC_CONF') == 'expandable_segments:True'):
        choices.append((1e-5, False, 128))
        second_report = target / 'corrective-01/report.json'
        if second_report.exists() and json.loads(second_report.read_text()).get('stop_reason') == 'MEMORY_HEADROOM_GATE':
            choices.append((1e-5, False, 128))
    results = []
    for i, (lr, balanced, steps) in enumerate(choices):
        run_dir = target / f'{"corrective" if not passed else "pilot"}-{i:02d}'
        if (run_dir / 'evaluation' / 'report.json').exists():
            results.append(json.loads((run_dir / 'evaluation' / 'report.json').read_text()))
            continue
        spec = make_spec(root, config, directory, candidate, purpose='preparation')
        spec['training'].update(learning_rate=lr, warmup_steps=min(8, steps - 1), max_steps=steps,
                                checkpoint_interval=32, evaluation_interval=32)
        spec['data']['balanced'] = balanced
        spec['optimization']['allocator'] = os.environ.get('PYTORCH_ALLOC_CONF', 'default')
        spec['optimization']['release_cached_segments_at_boundary'] = not passed and i == 2
        spec['evaluation']['validation_batches'] = 32
        spec['evaluation']['validation_reserve_seconds'] = 45
        spec['evaluation']['quality_reserve_seconds'] = 60
        # Measured recovery publication cost and a full development generation reserve.
        save_seconds = json.loads((directory / 'recovery/report.json').read_text())['final_save']['write_verify_seconds']
        reserve = max(180, save_seconds * 2 + candidate['seconds'] * 1.5 + 60)
        spec['checkpoint']['minimum_save_seconds'] = save_seconds * 1.5 + 15
        budget = SessionBudget.start(1200, min(reserve, 600))
        started = time.monotonic()
        counter = [0]
        def quality(model, tokenizer, deadline):
            counter[0] += 1
            evaluation = evaluate_loaded(model, tokenizer, make_suite(), deadline=deadline,
                output=run_dir / 'development' / f'check-{counter[0]:03}.jsonl')
            atomic_document(run_dir / 'development' / f'check-{counter[0]:03}.json', evaluation)
            return evaluation['metrics']
        try:
            report = run_chat(spec, run_dir, budget, quality_evaluator=quality)
            store = RecoveryStore(run_dir / 'checkpoints')
            evaluation = evaluate_candidate(store.resolve('final') if 'final' in store.index() else store.resolve(),
                Path(spec['tokenizer']['path']), run_dir / 'evaluation', deadline=budget.deadline_monotonic)
            report['budget'] = budget.document()
            atomic_document(run_dir / 'report.json', report)
            atomic_document(run_dir / 'budget.json', budget.document())
            evaluation.update(pilot_spec=spec, training_report=report, learning_rate=lr, balanced=balanced)
            atomic_document(run_dir / 'evaluation/report.json', evaluation)
            results.append(evaluation)
        finally:
            consume_gpu_budget(directory, started, label=run_dir.name, ceiling=config['preparation_gpu_seconds'])
    atomic_document(directory / 'pilot-comparison.json', results)
    return results


def resume_pilot(root: Path, config: dict, directory: Path) -> dict:
    """Restore only the isolated interrupted pilot under its original 20-minute deadline."""
    from llm_training.chat_run import run_chat
    from llm_training.recovery import RecoveryStore
    from codexa_workspace.chat_evaluation import evaluate_candidate, evaluate_loaded, make_suite
    run_dir = directory / 'pilots/corrective-02'
    original = json.loads((run_dir / 'report.json').read_text())
    if original['state'] != 'INTERRUPTED':
        raise ValueError('Only the explicitly interrupted preparation pilot may resume.')
    spec = json.loads((run_dir / 'run_metadata.json').read_text())['spec']
    budget = SessionBudget.resume(json.loads((run_dir / 'budget.json').read_text()))
    atomic_document(run_dir / 'interruption-report.json', original)
    started = time.monotonic()
    counter = [0]
    def quality(model, tokenizer, deadline):
        counter[0] += 1
        result = evaluate_loaded(model, tokenizer, make_suite(), deadline=deadline,
            output=run_dir / 'development' / f'resumed-{counter[0]:03}.jsonl')
        atomic_document(run_dir / 'development' / f'resumed-{counter[0]:03}.json', result)
        return result['metrics']
    try:
        report = run_chat(spec, run_dir, budget, resume=True, quality_evaluator=quality)
        store = RecoveryStore(run_dir / 'checkpoints')
        result = evaluate_candidate(store.resolve('final'), Path(spec['tokenizer']['path']),
            run_dir / 'evaluation-resumed', deadline=budget.deadline_monotonic)
        report['budget'] = budget.document()
        atomic_document(run_dir / 'report.json', report)
        atomic_document(run_dir / 'budget.json', budget.document())
        result.update(pilot_spec=spec, training_report=report, learning_rate=spec['training']['learning_rate'],
                      balanced=spec['data']['balanced'])
        atomic_document(run_dir / 'evaluation-resumed/report.json', result)
        records = json.loads((directory / 'pilot-comparison.json').read_text())
        records.append(result)
        atomic_document(directory / 'pilot-comparison.json', records)
        return result
    finally:
        consume_gpu_budget(directory, started, label='corrective-02-resume', ceiling=config['preparation_gpu_seconds'])


def finalize(root: Path, config: dict, directory: Path) -> dict:
    """Freeze measured choices; failed or missing mandatory gates mean BLOCKED."""
    from llm_architecture.checkpointing import file_sha256
    comparison = directory / 'deterministic-candidate-comparison.json'
    candidates = json.loads((comparison if comparison.exists() else directory / 'candidate-comparison.json').read_text())
    eligible = [r for r in candidates if r['metrics']['entry_gate_passed']]
    candidate = max(eligible or candidates, key=ranking)
    spec = make_spec(root, config, directory, candidate)
    benchmark_report = json.loads((directory / 'benchmark/report.json').read_text())
    recovery_report = json.loads((directory / 'recovery/report.json').read_text())
    deterministic_recovery = directory / 'recovery-deterministic/report.json'
    deterministic_benchmark = directory / 'benchmark-deterministic/report.json'
    if deterministic_recovery.exists() and deterministic_benchmark.exists():
        alternative = json.loads(deterministic_recovery.read_text())
        if alternative['passed']:
            recovery_report = alternative
            benchmark_report = json.loads(deterministic_benchmark.read_text())
            spec['optimization'].update(deterministic_algorithms=True, cublas_workspace_config=':4096:8')
    pilot_results = json.loads((directory / 'pilot-comparison.json').read_text()) if (directory / 'pilot-comparison.json').exists() else []
    technical = [r for r in pilot_results if r['training_report'].get('training_state', {}).get('optimizer_step', 0) >= 32
                 and r['training_report'].get('stop_reason') != 'MEMORY_HEADROOM_GATE']
    if technical:
        measured = max(technical, key=lambda r: r['training_report']['training_state']['optimizer_step'])
        spec['optimization'].update({key: measured['pilot_spec']['optimization'][key]
            for key in ('allocator', 'release_cached_segments_at_boundary') if key in measured['pilot_spec']['optimization']})
    safe_benchmarks = [r for r in benchmark_report['results'] if not r.get('skipped') and not r.get('failed') and r['headroom_passed']]
    worst = next((r for r in safe_benchmarks if r['name'] == 'dynamic-long-m1'), None)
    representative = next((r for r in safe_benchmarks if r['name'] == 'dynamic-short-m1'), None)
    improving = [r for r in pilot_results if r['metrics']['entry_gate_passed'] and
                 r['metrics']['macro_task_accuracy'] >= candidate['metrics']['macro_task_accuracy'] and
                 r['metrics']['severe_degeneration_rate'] <= candidate['metrics']['severe_degeneration_rate']]
    if improving:
        winner = max(improving, key=lambda r: (ranking(r), -r['learning_rate']))
        spec['training']['learning_rate'] = winner['learning_rate']
        spec['data']['balanced'] = winner['balanced']
    train_steps = 5000
    reserve = max(300, recovery_report['final_save']['write_verify_seconds'] * 2 + candidate['restore_and_evaluation_seconds'] * 12 + 120)
    spec['finalization_reserve_seconds'] = min(reserve, config['wall_clock_seconds'] / 4)
    spec['checkpoint']['minimum_save_seconds'] = recovery_report['final_save']['write_verify_seconds'] * 1.5 + 15
    if worst:
        # Conservative worst-length microstep projection; actual pilot steps are preferred.
        step_seconds = worst['steady_step_seconds'] * spec['training']['gradient_accumulation_steps']
        measured_steps = []
        measured_paths = set()
        for result in pilot_results:
            path = Path(result['checkpoint']).parents[2] / 'train_metrics.jsonl'
            if path.exists() and path not in measured_paths:
                measured_steps.extend(json.loads(line)['step_time_seconds'] for line in path.read_text().splitlines())
                measured_paths.add(path)
        if measured_steps:
            step_seconds = statistics.median(measured_steps)
        unique = json.loads((directory / 'data-audit.json').read_text())['splits']['train']['totals']['examples']
        train_steps = min(config['maximum_optimizer_steps'], unique // (spec['training']['micro_batch_size'] * spec['training']['gradient_accumulation_steps']),
                          int((config['wall_clock_seconds'] - spec['finalization_reserve_seconds']) * .95 / step_seconds))
        spec['training'].update(max_steps=max(1, train_steps), warmup_steps=min(100, max(1, round(train_steps * .02))))
        interval = max(1, round(1200 / step_seconds))
        spec['training'].update(checkpoint_interval=interval, evaluation_interval=interval)
    environment = experiment_environment(root)
    checks = json.loads((directory / 'automated-checks.json').read_text()) if (directory / 'automated-checks.json').exists() else {}
    gates = dict(dataset_audit=(directory / 'data-audit.json').exists(), initialization_quality=bool(eligible),
        worst_shape_memory=worst is not None, full_optimizer_recovery=bool(recovery_report.get('passed')),
        realistic_accumulation_memory=bool(technical),
        behavioral_pilot=bool(improving), deadline_and_signal_tests=bool(checks.get('deadline_and_signal_tests')),
        regression_tests=bool(checks.get('regression_tests')),
        installed_implementations=all(not r['source_import_differences'] for name, r in environment['repositories'].items()
                                      if name != 'LLM-Specialist'))
    gates['held_out_exposure_provenance'] = False  # Historical flags do not prove sealed evaluation access history.
    headroom_path = directory / 'benchmark-headroom/report.json'
    headroom_report = json.loads(headroom_path.read_text()) if headroom_path.exists() else None
    gates['strict_live_memory_headroom'] = bool(headroom_report and all(
        not r.get('failed') and not r.get('skipped') and r.get('headroom_passed') for r in headroom_report['results']))
    manifest = dict(readiness='READY' if all(gates.values()) else 'BLOCKED', mandatory_gates=gates,
        spec=spec, configuration_file=config['_configuration_path'], configuration_file_sha256=file_sha256(config['_configuration_path']),
        required_free_gpu_bytes=max((r['peak_reserved_bytes'] for r in safe_benchmarks), default=0) + config['minimum_gpu_headroom_bytes'],
        selected_provisional_candidate=candidate['checkpoint'], selected_candidate_passed=bool(eligible),
        planned_checkpoint_slots=6, checkpoint_bytes=recovery_report['final_save']['bytes'],
        limitations=['No candidate passed the entry floor unless initialization_quality is true.',
            'Historical exposure flags cannot certify held-out validation/test; no proven generalization claim.',
            'Full optimizer state restoration and bitwise resumed continuation are separate recovery checks.'],
        environment=environment, promotion='NOT_AUTHORIZED', production_launched=False)
    atomic_document(directory / 'launch-manifest.json', manifest)
    from codexa_workspace.chat_report import write_report
    write_report(root, directory, manifest)
    return manifest


def regression_checks(root: Path, directory: Path) -> dict:
    """Record actual offline suite outcomes before sealing launch evidence."""
    commands = [
        [sys.executable, '-m', 'pytest', '-q', '-rs'],
        [sys.executable, '-m', 'pytest', '-q', '../LLM-Architecture/tests', '../LLM-Tokenizer/tests',
         '../LLM-Data/tests', '../LLM-Training/tests', '../LLM-Inference/tests', 'tests/test_chat_preparation.py'],
        [sys.executable, '-m', 'pip', 'check'],
    ]
    records = []
    for i, command in enumerate(commands):
        started = time.monotonic()
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=300, env={**os.environ, 'OMP_NUM_THREADS': '2'})
        log = directory / f'regression-{i:02d}.txt'
        log.write_text(result.stdout)
        records.append(dict(command=command, returncode=result.returncode, seconds=time.monotonic()-started,
                            log=str(log)))
    checks = dict(regression_tests=all(r['returncode'] == 0 for r in records),
                  deadline_and_signal_tests=records[1]['returncode'] == 0, results=records,
                  skips='Five optional sklearn specialist tests are skipped in the generative environment.')
    atomic_document(directory / 'automated-checks.json', checks)
    return checks


def prepare(root: Path, config: dict, directory: Path, *, stage: str) -> None:
    """Run bounded preparation stages; this function cannot call production launch."""
    directory.mkdir(parents=True, exist_ok=True)
    if stage == 'all':
        preflight(root, config, directory)
    if stage in ('all', 'audit') and not (directory / 'data-audit.json').exists():
        audit(root, config, directory)
    if stage in ('all', 'candidates'):
        compare_candidates(root, config, directory)
    if stage == 'diagnostics':
        retention_and_sampled(root, config, directory)
    if stage == 'resume-pilot':
        resume_pilot(root, config, directory)
    if stage in ('all', 'benchmark', 'deterministic-benchmark', 'headroom-benchmark', 'recovery', 'deterministic-recovery', 'pilots', 'finalize'):
        candidates = json.loads((directory / 'candidate-comparison.json').read_text())
        eligible = [r for r in candidates if r['metrics']['entry_gate_passed']]
        candidate = max(eligible or candidates, key=ranking)
        print('Provisional technical candidate:', candidate['checkpoint'], 'quality gate:', bool(eligible), flush=True)
        if stage == 'deterministic-recovery':
            recovery_smoke(root, config, directory, candidate, deterministic=True)
        if stage == 'deterministic-benchmark':
            benchmark(root, config, directory, candidate, deterministic=True)
        if stage == 'headroom-benchmark':
            benchmark(root, config, directory, candidate, deterministic=True, memory_cap=True)
        if stage == 'finalize':
            regression_checks(root, directory)
        if stage in ('all', 'benchmark') and not (directory / 'benchmark/report.json').exists():
            benchmark(root, config, directory, candidate)
        if stage in ('all', 'recovery') and not (directory / 'recovery/report.json').exists():
            recovery_smoke(root, config, directory, candidate)
        if stage in ('all', 'pilots'):
            pilots(root, config, directory, candidate)
        if stage in ('all', 'finalize'):
            if stage == 'all':
                regression_checks(root, directory)
            print(json.dumps(finalize(root, config, directory), indent=2))
