"""Matched V2/V3 restore and continuous-rotation benchmarks on disposable data."""

import importlib.util
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import time

import torch

from llm_architecture.checkpointing import read_native_checkpoint
from llm_training.recovery import RecoveryStore
from llm_training.checkpoint_reclamation import create_disposable_fixture_store
from .checkpoint_benchmark import payload, allocated, extent_usage


def baseline_modules(directory: Path):
    modules = []
    for name in ('component', 'recovery'):
        package = 'llm_architecture' if name == 'component' else 'llm_training'
        spec = importlib.util.spec_from_file_location(package + '._codexa_v2_' + name,
            directory / f'baseline_{name}_v2.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    component, recovery = modules
    recovery.read_components = component.read_components
    recovery.manifest_document = component.manifest_document
    recovery.ComponentStore = component.ComponentStore
    return component, recovery


def _summary(samples: list, cpu: list) -> dict:
    return dict(median_seconds=statistics.median(samples),
        p95_seconds=statistics.quantiles(samples, n=20, method='inclusive')[18],
        median_cpu_seconds=statistics.median(cpu), samples=samples)


def memory_worker(engine: str, directory: Path, baseline: Path) -> dict:
    """Fresh-process RSS measurement; CUDA is not initialized."""
    torch.set_num_threads(2)
    if engine == 'legacy':
        store = RecoveryStore(directory)
        read = lambda: read_native_checkpoint(store.resolve())
    elif engine == 'v2':
        _, module = baseline_modules(baseline)
        store = module.OptimizedRecoveryStore(directory, reserve_bytes=0)
        read = store.load
    else:
        store = RecoveryStore(directory, backend='optimized', reserve_bytes=0)
        read = store.load
    def resident():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
        raise RuntimeError('Linux RSS measurement unavailable.')
    before = resident()
    samples, stop = [before], threading.Event()
    def sample():
        while not stop.wait(.001):
            samples.append(resident())
    observer = threading.Thread(target=sample)
    observer.start()
    value = read()
    # Access all mapped payloads, representing CPU consumption after verification.
    checksum = sum(float(t.sum()) for t in (
        value['model_state_dict']['weight'],
        value['optimizer_state_dict']['state'][0]['exp_avg'],
        value['optimizer_state_dict']['state'][0]['exp_avg_sq']))
    after = resident()
    stop.set()
    observer.join()
    peak = max([after, *samples])
    return dict(initial_rss_kib=before, loaded_rss_kib=after,
                sampled_peak_rss_kib=peak, resident_growth_kib=max(0, after-before),
                additional_sampled_peak_kib=max(0, peak-before), checksum=checksum,
                method='isolated process, /proc/self/status RSS sampled every 1 ms')


def benchmark(root: Path, directory: Path, baseline: Path) -> dict:
    """At most 512 MiB persistent fixtures, with deletion only in fixture grants."""
    directory.mkdir(parents=True, exist_ok=False)
    _, old = baseline_modules(baseline)
    torch.set_num_threads(2)
    torch.manual_seed(73)
    cases = []
    for size in (12, 96):
        case = directory / f'{size}MiB'
        case.mkdir()
        data = payload()
        width = size // 12 * 1024
        data['model_state_dict']['weight'] = torch.randn(width, 1024)
        state = data['optimizer_state_dict']['state'][0]
        state['exp_avg'], state['exp_avg_sq'] = torch.randn(width, 1024), torch.rand(width, 1024)
        bound = (size + 1) * 1024**2
        legacy = RecoveryStore(case / 'legacy')
        v2 = old.OptimizedRecoveryStore(case / 'v2', reserve_bytes=0,
                                       retention=dict(max_checkpoint_bytes=bound))
        v3, _ = create_disposable_fixture_store(case, max_checkpoint_bytes=bound)
        stores = dict(legacy=legacy, v2=v2, v3=v3)
        first_saves = {}
        for name, store in stores.items():
            started, cpu = time.perf_counter(), time.process_time()
            store.save(data)
            first_saves[name] = dict(wall_seconds=time.perf_counter()-started,
                                    cpu_seconds=time.process_time()-cpu)
        legacy_path = legacy.resolve()
        readers = dict(legacy=lambda: read_native_checkpoint(legacy_path),
                       v2=v2.load, v3=v3.load)
        for read in readers.values():
            value = read()
            assert torch.equal(value['model_state_dict']['weight'], data['model_state_dict']['weight'])
            assert torch.equal(value['optimizer_state_dict']['state'][0]['exp_avg'], state['exp_avg'])
            assert value['resume_context'] == data['resume_context']
            del value
        samples, cpu_samples = {k: [] for k in readers}, {k: [] for k in readers}
        names = list(readers)
        for trial in range(55):
            order = names[trial % 3:] + names[:trial % 3]
            if trial % 2:
                order.reverse()
            for name in order:
                started, cpu = time.perf_counter(), time.process_time()
                value = readers[name]()
                elapsed, consumed = time.perf_counter()-started, time.process_time()-cpu
                del value
                if trial >= 5:
                    samples[name].append(elapsed)
                    cpu_samples[name].append(consumed)
        results = {k: _summary(samples[k], cpu_samples[k]) for k in readers}
        memory = {}
        for name, store in stores.items():
            process = subprocess.run([sys.executable, '-B',
                str(root / 'scripts/benchmark_checkpoint_storage_v3.py'),
                '--memory-worker', name, '--checkpoint-directory', str(store.directory),
                '--baseline', str(baseline)], capture_output=True, text=True, check=True)
            memory[name] = json.loads(process.stdout)
        ratio = results['v3']['median_seconds'] / results['legacy']['median_seconds']
        cases.append(dict(size_mib=size, synthetic=True, nonzero=True, repetitions=50,
            results=results, first_save=first_saves, isolated_memory=memory,
            restore_ratio_vs_legacy=ratio,
            restore_ratio_vs_v2=results['v3']['median_seconds']/results['v2']['median_seconds'],
            latency_gate_passed=ratio <= 1.10, exact=True, extent_usage=extent_usage(case)))
        if allocated(baseline) > 512 * 1024**2:
            raise RuntimeError('Total V3 artifact budget exceeded.')
    # Sixty-two representative 12 MiB unique states, not metadata-only duplicates.
    rotation = directory / 'rotation'
    rotation.mkdir()
    v3, _ = create_disposable_fixture_store(rotation, max_checkpoint_bytes=13 * 1024**2)
    legacy = RecoveryStore(rotation / 'legacy')
    data = payload()
    rows = []
    for step in range(1, 63):
        data['training_state']['optimizer_step'] = step
        data['model_state_dict']['weight'].add_(.0001)
        data['optimizer_state_dict']['state'][0]['exp_avg'].add_(.0001)
        data['optimizer_state_dict']['state'][0]['exp_avg_sq'].add_(.0001)
        row = dict(step=step)
        for name, store in (('legacy', legacy), ('v3', v3)):
            start, cpu = time.perf_counter(), time.process_time()
            store.save(data, best=step == 1, final=step == 62)
            row[name] = dict(save_seconds=time.perf_counter()-start,
                            cpu_seconds=time.process_time()-cpu,
                            allocated_bytes=allocated(store.directory if name == 'legacy' else Path(store.reclamation_authorization.namespace)))
        rows.append(row)
        if allocated(baseline) > 512 * 1024**2:
            raise RuntimeError('Total V3 artifact budget exceeded during rotation.')
    rotation_results = {name: _summary([r[name]['save_seconds'] for r in rows[6:]],
                                     [r[name]['cpu_seconds'] for r in rows[6:]])
                        for name in ('legacy', 'v3')}
    report = dict(cases=cases, rotation=dict(publications=62, rows=rows,
        steady_state=rotation_results, final_retained_snapshots=len(list((v3.store_root/'snapshots').glob('*/*.ckpt.json'))),
        extent_usage=extent_usage(rotation)), allocated_bytes=allocated(baseline),
        all_latency_gates_passed=all(c['latency_gate_passed'] for c in cases),
        full_size_gpu_qualified=False, real_store_reclamation_executed=False)
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def restore_existing(root: Path, directory: Path, baseline: Path,
                     report_name: str = 'restore-corrected.json') -> dict:
    """Corrected one-pass comparison reuses fixtures and never publishes checkpoints."""
    _, old = baseline_modules(baseline)
    torch.set_num_threads(2)
    cases = []
    for size in (12, 96):
        case = directory / f'{size}MiB'
        legacy = RecoveryStore(case/'legacy')
        v2 = old.OptimizedRecoveryStore(case/'v2', reserve_bytes=0)
        v3_path = next(case.glob('codexa-v3-fixture-*/run'))
        v3 = RecoveryStore(v3_path, backend='optimized', reserve_bytes=0)
        path = legacy.resolve()  # Resolve outside timing: every timed loader hashes once.
        readers = dict(legacy=lambda: read_native_checkpoint(path), v2=v2.load, v3=v3.load)
        samples, cpu = {k: [] for k in readers}, {k: [] for k in readers}
        names = list(readers)
        for trial in range(55):
            order = names[trial % 3:] + names[:trial % 3]
            if trial % 2:
                order.reverse()
            for name in order:
                start, consumed = time.perf_counter(), time.process_time()
                value = readers[name]()
                elapsed, usage = time.perf_counter()-start, time.process_time()-consumed
                del value
                if trial >= 5:
                    samples[name].append(elapsed)
                    cpu[name].append(usage)
        results = {k: _summary(samples[k], cpu[k]) for k in readers}
        memory = {}
        for name, store in (('legacy',legacy),('v2',v2),('v3',v3)):
            process = subprocess.run([sys.executable, '-B',
                str(root/'scripts/benchmark_checkpoint_storage_v3.py'), '--memory-worker', name,
                '--checkpoint-directory', str(store.directory), '--baseline', str(baseline)],
                capture_output=True, text=True, check=True)
            memory[name] = json.loads(process.stdout)
        ratio = results['v3']['median_seconds']/results['legacy']['median_seconds']
        cases.append(dict(size_mib=size, synthetic=True, results=results, repetitions=50,
            isolated_memory=memory, restore_ratio_vs_legacy=ratio,
            restore_ratio_vs_v2=results['v3']['median_seconds']/results['v2']['median_seconds'],
            latency_gate_passed=ratio <= 1.10))
    report = dict(cases=cases, integrity_passes_per_load=1, warm_cache=True,
                  all_latency_gates_passed=all(c['latency_gate_passed'] for c in cases))
    if Path(report_name).name != report_name:
        raise ValueError('Report name must remain within this fixture directory.')
    target = directory/report_name
    with target.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    return report
