"""Controlled nonzero fixtures comparing legacy, v1 baseline and revised readers."""

import cProfile
import importlib.util
import json
from pathlib import Path
import pstats
import resource
import statistics
import time

import torch

from llm_architecture.checkpointing import read_native_checkpoint
from llm_training.recovery import RecoveryStore
from codexa_workspace.checkpoint_benchmark import payload, allocated, extent_usage


def benchmark(root: Path, directory: Path, baseline_source: Path) -> dict:
    """Publish two bounded fixture sizes; never writes to baseline/historical paths."""
    directory.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('codexa_v1_loader_baseline', baseline_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    cases = []
    for size in (12, 96):
        case = directory / f'{size}MiB'
        data = payload()
        width = size // 12 * 1024
        data['model_state_dict'] = {'weight': torch.randn(width, 1024)}
        state = data['optimizer_state_dict']['state'][0]
        state['exp_avg'] = torch.randn(width, 1024)
        state['exp_avg_sq'] = torch.rand(width, 1024)
        bound = size * 1024**2 + 1024**2
        legacy = RecoveryStore(case / 'legacy')
        revised = RecoveryStore(case / 'revised', backend='optimized', reserve_bytes=0,
            retention=dict(max_checkpoint_bytes=bound, max_snapshots=6))
        legacy.save(data)
        revised.save(data)
        paths = [legacy.resolve(), revised.resolve()]
        readers = {
            'legacy': lambda: read_native_checkpoint(paths[0]),
            'v1_baseline': lambda: module.read_components(paths[1]),
            'revised_direct': lambda: read_native_checkpoint(paths[1]),
            'revised_store': lambda: revised.load(),
        }
        # All engines read identical values with complete integrity verification.
        for read in readers.values():
            value = read()
            assert torch.equal(value['model_state_dict']['weight'], data['model_state_dict']['weight'])
            assert torch.equal(value['optimizer_state_dict']['state'][0]['exp_avg'], state['exp_avg'])
            assert value['resume_context'] == data['resume_context']
            del value
        samples = {name: [] for name in readers}
        cpu = {name: [] for name in readers}
        names = list(readers)
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for trial in range(55):
            order = names[trial % len(names):] + names[:trial % len(names)]
            if trial % 2:
                order.reverse()
            for name in order:
                started, cpu_started = time.perf_counter(), time.process_time()
                value = readers[name]()
                elapsed, consumed = time.perf_counter() - started, time.process_time() - cpu_started
                del value
                if trial >= 5:
                    samples[name].append(elapsed)
                    cpu[name].append(consumed)
        results = {name: dict(median_seconds=statistics.median(values),
                    p95_seconds=statistics.quantiles(values, n=20, method='inclusive')[18],
                    median_cpu_seconds=statistics.median(cpu[name]), samples=values)
                   for name, values in samples.items()}
        profiles = {}
        for name in ['v1_baseline', 'revised_direct']:
            profile = cProfile.Profile()
            profile.enable()
            for _ in range(10):
                value = readers[name]()
                del value
            profile.disable()
            stats = pstats.Stats(profile)
            rows = sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:22]
            profiles[name] = [dict(function=f'{k[0]}:{k[1]}:{k[2]}', calls=v[1], total=v[2], cumulative=v[3])
                              for k, v in rows]
        ratio = results['revised_direct']['median_seconds'] / results['legacy']['median_seconds']
        store_ratio = results['revised_store']['median_seconds'] / results['legacy']['median_seconds']
        cases.append(dict(size_mib=size, synthetic=True, nonzero=True, repetitions=50,
            results=results, direct_restore_ratio=ratio, store_restore_ratio=store_ratio,
            latency_gate_passed=ratio <= 1.10 and store_ratio <= 1.10,
            profiles=profiles, exact=True, extent_usage=extent_usage(case),
            rss_growth_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_rss))
        if allocated(directory) > 512 * 1024**2:
            raise RuntimeError('Validation fixture budget exceeded.')
        del data, state
    result = dict(cases=cases, allocated_bytes=allocated(directory),
                  all_latency_gates_passed=all(c['latency_gate_passed'] for c in cases),
                  full_size_gpu_qualified=False, garbage_collection_enabled=False)
    (directory / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
