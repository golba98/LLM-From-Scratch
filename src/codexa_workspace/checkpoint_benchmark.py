"""Small lossless checkpoint benchmarks; never opens historical outputs for writes."""

from dataclasses import asdict
import json
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch

from llm_architecture.checkpointing import read_native_checkpoint
from llm_architecture.model import ModelConfig
from llm_training.config import TrainingConfig
from llm_training.checkpointing import CheckpointManager, capture_random_states
from llm_training.recovery import RecoveryStore


def allocated(path: Path) -> int:
    """Inode-accounted bytes, excluding symlink targets."""
    return sum(p.stat().st_blocks * 512 for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def extent_usage(path: Path) -> str:
    result = subprocess.run(['btrfs', 'filesystem', 'du', '-s', '--raw', str(path)],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def payload() -> dict:
    torch.manual_seed(73)
    return dict(format_version=1, created_at_utc='bounded-fixture',
        model_state_dict={'weight': torch.randn(1024, 1024)},
        optimizer_state_dict={'state': {0: {'exp_avg': torch.randn(1024, 1024),
            'exp_avg_sq': torch.rand(1024, 1024), 'step': torch.tensor(1.)}},
            'param_groups': [{'params': [0], 'lr': .001}]},
        scheduler_state=dict(warmup_steps=0, max_steps=10, peak_learning_rate=.001,
                             minimum_learning_rate=.0001),
        grad_scaler_state_dict=None, random_states=capture_random_states(),
        training_state=dict(micro_step=1, optimizer_step=1, tokens_seen=4,
            completed_epochs=0, batches_in_epoch=1, best_validation_loss=None),
        config=dict(model=asdict(ModelConfig(vocab_size=16, context_length=4, num_layers=1,
            hidden_size=8, num_heads=2, intermediate_size=16)),
            training=asdict(TrainingConfig(1, 1, .001, 0., 0, 10, 1000., 'fp32', 1, 1, 73))),
        run_name='bounded', run_id='bounded', tokenizer_reference=None, tokenizer_sha256=None,
        resume_context={'configuration_sha256': 'bounded', 'microstep_within_gradient_accumulation': 0,
                        'budget': {'elapsed_seconds': 0}})


def benchmark_backend(directory: Path, backend: str) -> dict:
    """Six small states, with three identical-tensor finalization pairs."""
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    torch.set_num_threads(2)
    data = payload()
    directory.mkdir(parents=True, exist_ok=False)
    if backend == 'copy-manager':
        manager = CheckpointManager(directory, 'bounded')
        store = None
    else:
        store = RecoveryStore(directory / 'run', backend=backend, reserve_bytes=0)
    rows = []
    for i in range(6):
        changed = i % 2 == 0
        if i and changed:
            data['model_state_dict']['weight'].add_(.0001)
            data['optimizer_state_dict']['state'][0]['exp_avg'].add_(.0001)
        data['created_at_utc'] = f'fixture-{i}'
        data['resume_context']['budget']['elapsed_seconds'] = i
        data['training_state']['optimizer_step'] = 1 + i // 2
        started_cpu, started = time.process_time(), time.perf_counter()
        if store is None:
            path = manager.save(data, is_best=i == 0, milestone=changed)
        else:
            store.save(data, best=i == 0, final=i == 5)
            path = store.resolve()
        save_seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - started_cpu
        started = time.perf_counter()
        restored = read_native_checkpoint(path)
        load_seconds = time.perf_counter() - started
        for actual, expected in [(restored['model_state_dict']['weight'], data['model_state_dict']['weight']),
            (restored['optimizer_state_dict']['state'][0]['exp_avg'], data['optimizer_state_dict']['state'][0]['exp_avg']),
            (restored['optimizer_state_dict']['state'][0]['exp_avg_sq'], data['optimizer_state_dict']['state'][0]['exp_avg_sq'])]:
            assert torch.equal(actual, expected) and actual.dtype == expected.dtype
        assert restored['resume_context'] == data['resume_context']
        del restored
        rows.append(dict(changed=changed, save_seconds=save_seconds, load_seconds=load_seconds,
                         cpu_seconds=cpu_seconds, allocated_bytes=allocated(directory)))
        if allocated(directory) > 128 * 1024**2:
            raise RuntimeError('Bounded per-backend artifact ceiling exceeded.')
    return dict(backend=backend, synthetic=True, tensor_bytes=3 * 1024 * 1024 * 4, rows=rows,
        median_changed_save_seconds=statistics.median(r['save_seconds'] for r in rows if r['changed']),
        median_load_seconds=statistics.median(r['load_seconds'] for r in rows),
        unchanged_growth_bytes=rows[1]['allocated_bytes'] - rows[0]['allocated_bytes'],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_growth_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_rss,
        extent_usage=extent_usage(directory), exact=True)


def benchmark(root: Path, directory: Path) -> dict:
    """Run isolated subprocess measurements with a total 512 MiB ceiling."""
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    script = root / 'scripts/storage/benchmark_checkpoint_storage.py'
    for backend in ['legacy', 'optimized', 'copy-manager']:
        result = subprocess.run([str(root / '.venv/bin/python'), str(script), '--worker', backend,
            '--output', str(directory / backend)], capture_output=True, text=True, check=True)
        results.append(json.loads(result.stdout))
        if allocated(directory) > 512 * 1024**2:
            raise RuntimeError('Total benchmark storage ceiling exceeded.')
    legacy, optimized = results[:2]
    report = dict(results=results, total_allocated_bytes=allocated(directory),
        unchanged_growth_reduction=1 - optimized['unchanged_growth_bytes'] / legacy['unchanged_growth_bytes'],
        changed_save_ratio=optimized['median_changed_save_seconds'] / legacy['median_changed_save_seconds'],
        load_ratio=optimized['median_load_seconds'] / legacy['median_load_seconds'],
        full_size_qualified=False, garbage_collection_enabled=False)
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
