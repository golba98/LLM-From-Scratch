"""Offline preparation evidence and explicit, gated operator launch commands."""

from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

from codexa_workspace.assets import workspace_root, resolve_input, output_root
from llm_training.session import atomic_document


def read_configuration(path: Path) -> tuple[Path, dict, Path]:
    """Resolve the configured workspace and all physical inputs explicitly."""
    root = workspace_root()
    path = path if path.is_absolute() else root / path
    config = yaml.safe_load(path.read_text())
    config['_configuration_path'] = str(path)
    if not 0 < config['wall_clock_seconds'] <= 36000 or config['preparation_gpu_seconds'] > 7200:
        raise ValueError('Session budget must be positive and at most ten hours; preparation GPU budget is at most two hours.')
    if Path(config['run_name']).name != config['run_name']:
        raise ValueError('Unsafe run identity.')
    backend = config.get('checkpoint_storage_backend', 'legacy')
    if backend not in ('legacy', 'optimized'):
        raise ValueError('Unknown checkpoint storage backend.')
    if config.get('checkpoint_retention') is not None and backend != 'optimized':
        raise ValueError('Checkpoint retention requires the optimized backend.')
    mode = config.get('checkpoint_reclamation_mode', 'disabled')
    if mode not in ('disabled', 'simulate') or (mode != 'disabled' and backend != 'optimized'):
        raise ValueError('Reclamation config supports optimized simulation only; no execution authority.')
    if config.get('checkpoint_reclamation_authorization') is not None:
        raise ValueError('Training config cannot supply deletion authority.')
    if config.get('checkpoint_optimizer_storage_evidence') is not None:
        if backend != 'optimized':
            raise ValueError('Optimizer storage evidence requires optimized storage.')
        config['checkpoint_optimizer_storage_evidence'] = str(
            resolve_input(root,config['checkpoint_optimizer_storage_evidence']))
    if config.get('checkpoint_store_root'):
        store = Path(config['checkpoint_store_root'])
        store = store if store.is_absolute() else output_root(root) / store
        if not store.resolve().is_relative_to(output_root(root).resolve()):
            raise ValueError('New checkpoint stores must stay under the configured output root.')
        config['checkpoint_store_root'] = str(store.absolute())
    preparation = output_root(root) / 'preparation' / config['run_name']
    return root, config, preparation


def hardware() -> dict:
    """Observe desktop coexistence and RAM without changing processes/settings."""
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key] = int(value.strip().split()[0]) * 1024
    return {'gpu': gpu, 'free_gpu_bytes': int(gpu.split(',')[2]) * 1024**2,
            'host_memory_bytes': memory,
            'gpu_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                                     '--format=csv'], text=True)}


def experiment_environment(root: Path) -> dict:
    """Record revisions, dirty status and imported implementations/wheel identities."""
    import torch
    import importlib.metadata
    manifest = json.loads((root / 'compatibility.json').read_text())
    records = {}
    for name, details in {**manifest['components'], 'LLM-From-Scratch': {'path': '.'}}.items():
        repo = (root / details['path']).resolve()
        package = 'codexa_workspace' if name == 'LLM-From-Scratch' else 'llm_' + name.split('-', 1)[1].lower()
        try:
            module = importlib.import_module(package)
            installed = Path(module.__file__).parent
            if package == 'codexa_workspace':
                installed = Path(importlib.metadata.distribution('codexa-workspace').locate_file('codexa_workspace'))
            source = repo / 'src' / package
            differences = [str(p.relative_to(source)) for p in source.rglob('*.py')
                if not (installed / p.relative_to(source)).exists()
                or hashlib.sha256(p.read_bytes()).digest() != hashlib.sha256((installed / p.relative_to(source)).read_bytes()).digest()]
            module_path = str(module.__file__)
        except ImportError:
            differences, module_path = ['not installed in this profile'], None
        records[name] = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'dirty': subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True).strip(),
            'module': module_path, 'source_import_differences': differences,
            'source_tree_sha256': hashlib.sha256(b''.join(str(p.relative_to(repo)).encode() + p.read_bytes()
                for p in sorted((repo / 'src').rglob('*.py')))).hexdigest()}
    return dict(python=sys.version, executable=sys.executable, pytorch=torch.__version__, cuda=torch.version.cuda,
                bf16=torch.cuda.is_bf16_supported(), repositories=records,
                bitsandbytes=importlib.metadata.version('bitsandbytes'),
                cuda_allocator=os.environ.get('PYTORCH_ALLOC_CONF', 'default'),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def audit(root: Path, config: dict, directory: Path) -> dict:
    """Verify existing content/masks without writing transformed datasets."""
    from llm_architecture.checkpointing import file_sha256
    from llm_tokenizer.tokenizer import load_tokenizer
    from llm_data.chat_dataset import IndexedChatDataset, verify_split_isolation
    tokenizer_path = resolve_input(root, config['tokenizer'])
    if file_sha256(tokenizer_path) != config['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch.')
    tokenizer = load_tokenizer(tokenizer_path)
    data_path = resolve_input(root, config['data'])
    manifest = json.loads((data_path / 'dataset_manifest.json').read_text())
    datasets = {}
    report = {'manifest': manifest, 'manifest_sha256': file_sha256(data_path / 'dataset_manifest.json'), 'splits': {}}
    for split in ('train', 'validation', 'test'):
        started = time.monotonic()
        print(f'Auditing every {split} record and loss mask (read-only)', flush=True)
        dataset = IndexedChatDataset(data_path / f'{split}.jsonl', tokenizer, verify_tokens=True)
        if dataset.sha256 != manifest['output_sha256'][split]:
            raise ValueError('Dataset hash mismatch.')
        datasets[split] = dataset
        groups = {}
        for entry in dataset.index:
            groups[entry.group] = groups.get(entry.group, False) or entry.historical_exposure
        clean = [i for i, e in enumerate(dataset.index) if not groups[e.group]]
        report['splits'][split] = dict(path=str(dataset.path), sha256=dataset.sha256,
            totals=dict(dataset.audit), groups=len(groups), possibly_clean_groups=sum(not v for v in groups.values()),
            possibly_clean_indices=clean, seconds=time.monotonic() - started)
        atomic_document(directory / 'data-audit.partial.json', report)
    report['split_isolation'] = verify_split_isolation(datasets)
    report['limitations'] = ['Historical exposure flags denote potential training exposure, not a complete per-run access log.',
        'Original SFT validation and previous Stage 2 evaluation exposure remain; clean flags are not a sealed test guarantee.',
        'Assistant-token totals count repeated turns/prefixes across distinct conversation paths.',
        'No semantic paraphrase or factual-quality certification. Missing original base validation is not replaced.']
    # Record exact prompt occurrences without evaluating the protected acceptance suite.
    from codexa_workspace.chat_evaluation import make_suite
    from llm_data.data.general_chat import normalized
    prompt_sets = {name: {normalized(r['prompt']) for r in make_suite(acceptance=name == 'acceptance')}
                   for name in ('development', 'acceptance')}
    matches = {name: 0 for name in prompt_sets}
    for line in (data_path / 'train.jsonl').open():
        row = json.loads(line)
        for message in row['messages']:
            if message['role'] == 'user':
                for name, prompts in prompt_sets.items():
                    matches[name] += normalized(message['content']) in prompts
    report['exact_normalized_prompt_matches_in_training'] = matches
    atomic_document(directory / 'data-audit.json', report)
    return report


def preflight(root: Path, config: dict, directory: Path) -> dict:
    """Read-only launch checks; a missing readiness gate never launches training."""
    from llm_architecture.checkpointing import file_sha256, verify_checkpoint_checksum
    if Path(sys.executable).resolve() != (root / '.venv/bin/python').resolve() or sys.prefix != str(root / '.venv'):
        raise RuntimeError('Use the exact generative .venv/bin/python environment.')
    disk = shutil.disk_usage(output_root(root).parent)
    required = max(180 * 1024**3, config['minimum_disk_headroom_bytes'] + 80 * 1024**3)
    storage_plan = None
    if config.get('checkpoint_storage_backend', 'legacy') == 'optimized':
        from llm_training.preflight import (estimate_component_storage, checkpoint_publication_bound,
            estimate_bounded_component_storage)
        # Current successor uses 100-step checkpoint and evaluation intervals.
        storage_plan = estimate_component_storage(model_bytes=934356480 * 4,
            optimizer_bytes=934356480 * 8 + 1024**2,
            updated_generations=checkpoint_publication_bound(
                max_steps=config.get('maximum_optimizer_steps', 5000),
                checkpoint_interval=100, evaluation_interval=100))
        if config.get('checkpoint_retention') is not None:
            model_bytes, optimizer_bytes = 934356480*4, 934356480*8+1024**2
            options=dict(config['checkpoint_retention'])
            if config.get('checkpoint_optimizer_storage_evidence'):
                from llm_training.preflight import verified_optimizer_storage_evidence
                evidence=verified_optimizer_storage_evidence(
                    Path(config['checkpoint_optimizer_storage_evidence']),
                    model_config=asdict(load_config(root/config['architecture']).model),
                    optimizer_name='adamw8bit')
                model_bytes,optimizer_bytes=evidence['model_bytes'],evidence['optimizer_bytes']
                options.setdefault('max_checkpoint_bytes',evidence['declared_checkpoint_bound'])
            storage_plan = estimate_bounded_component_storage(model_bytes=model_bytes,
                optimizer_bytes=optimizer_bytes, retention=options)
        required = max(required, storage_plan['required_free_bytes'])
    if disk.free < required:
        raise RuntimeError(f'Insufficient storage: free={disk.free}, required={required}; no pruning permitted.')
    data = resolve_input(root, config['data'])
    token = resolve_input(root, config['tokenizer'])
    if file_sha256(token) != config['tokenizer_sha256']:
        raise ValueError('Tokenizer drift.')
    manifest = json.loads((data / 'dataset_manifest.json').read_text())
    for split, digest in manifest['output_sha256'].items():
        if file_sha256(data / f'{split}.jsonl') != digest:
            raise ValueError(f'{split} dataset drift.')
    candidates = {str(resolve_input(root, p)): verify_checkpoint_checksum(resolve_input(root, p)) for p in config['candidates']}
    disk = shutil.disk_usage(output_root(root).parent)
    hw = hardware()
    evidence = dict(hardware=hw, disk_free_bytes=disk.free, checkpoint_storage_plan=storage_plan, candidate_sha256=candidates,
        tokenizer_sha256=config['tokenizer_sha256'], dataset_manifest_sha256=file_sha256(data / 'dataset_manifest.json'),
        environment=experiment_environment(root), historical_inputs_read_only=True)
    if disk.free < config['minimum_disk_headroom_bytes'] + 80 * 1024**3:
        raise RuntimeError('Insufficient headroom for bounded preparation and recovery retention.')
    if hw['host_memory_bytes']['MemAvailable'] < config.get('minimum_host_available_bytes', 12 * 1024**3):
        raise RuntimeError('Host available RAM is below the checkpoint restore reserve; no process was altered.')
    directory.mkdir(parents=True, exist_ok=True)
    atomic_document(directory / 'preflight.json', evidence)
    return evidence


def consume_gpu_budget(directory: Path, started: float, *, label: str, ceiling: int) -> None:
    """Conservatively count entire CUDA workload processes, including their startup."""
    path = directory / 'gpu-budget.json'
    state = json.loads(path.read_text()) if path.exists() else {'seconds': 0, 'operations': []}
    elapsed = time.monotonic() - started
    state['seconds'] += elapsed
    state['operations'].append({'label': label, 'seconds': elapsed})
    atomic_document(path, state)
    if state['seconds'] > ceiling:
        raise TimeoutError('Preparation GPU ceiling exhausted; remaining gates are blocked.')


@contextmanager
def preparation_guard(directory: Path, ceiling: int):
    """Bound preparation independently and conservatively account abandoned stages."""
    directory.mkdir(parents=True, exist_ok=True)
    ledger, active = directory / 'gpu-budget.json', directory / 'gpu-active.json'
    state = json.loads(ledger.read_text()) if ledger.exists() else {'seconds': 0, 'operations': []}
    if active.exists():
        prior = json.loads(active.read_text())
        command = Path(f"/proc/{prior['pid']}/cmdline")
        if command.exists() and b'chat_training.py' in command.read_bytes():
            raise RuntimeError('Another preparation stage is active; no process was terminated.')
        state['seconds'] = max(state['seconds'], prior['base_seconds'] + max(0, time.time() - prior['started_utc']))
        atomic_document(ledger, state)
    remaining = ceiling - state['seconds']
    if remaining <= 1:
        raise TimeoutError('Preparation ceiling is exhausted, including abandoned-stage accounting.')
    started = time.monotonic()
    base = state['seconds']
    atomic_document(active, dict(pid=os.getpid(), started_utc=time.time(), base_seconds=base))
    old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, int(remaining)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        state = json.loads(ledger.read_text()) if ledger.exists() else state
        state['seconds'] = max(state['seconds'], base + time.monotonic() - started)
        atomic_document(ledger, state)
        active.unlink(missing_ok=True)


def compare_candidates(root: Path, config: dict, directory: Path) -> list[dict]:
    """Only development results can influence initialization selection."""
    from codexa_workspace.chat_evaluation import evaluate_candidate, fingerprint, make_suite
    from llm_architecture.checkpointing import verify_checkpoint_checksum
    reports = []
    for value in config['candidates']:
        path = resolve_input(root, value)
        target = directory / 'candidates' / path.parent.name
        if (target / 'report.json').exists():
            cached = json.loads((target / 'report.json').read_text())
            if (cached['checkpoint_sha256'] != verify_checkpoint_checksum(path)
                    or cached['tokenizer_sha256'] != config['tokenizer_sha256']
                    or cached['suite_sha256'] != fingerprint(make_suite())
                    or cached['decoding']['do_sample']):
                raise ValueError('Cached candidate identity differs; preserve it and use a new preparation identity.')
            reports.append(cached)
            continue
        print('Evaluating candidate:', path.parent.name, flush=True)
        started = time.monotonic()
        try:
            report = evaluate_candidate(path, resolve_input(root, config['tokenizer']), target)
            reports.append(report)
            print(json.dumps({'candidate': path.parent.name, **report['metrics']}), flush=True)
        finally:
            consume_gpu_budget(directory, started, label=path.parent.name, ceiling=config['preparation_gpu_seconds'])
    atomic_document(directory / 'candidate-comparison.json', reports)
    return reports


def make_spec(root: Path, config: dict, directory: Path, candidate: dict, *, purpose: str = 'production') -> dict:
    """Combine frozen measured choices with the unchanged main architecture."""
    from llm_training.config import load_config
    from llm_architecture.checkpointing import file_sha256
    from codexa_workspace.chat_evaluation import fingerprint, make_suite
    audit_report = json.loads((directory / 'data-audit.json').read_text())
    training = asdict(load_config(root / config['architecture']).training)
    training.update(micro_batch_size=config['micro_batch_size'], gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=1e-5, warmup_steps=50, max_steps=5000, seed=config['seed'], checkpoint_interval=100,
        evaluation_interval=100, precision='bf16', weight_decay=config['weight_decay'], gradient_clip=config['gradient_clip'])
    splits = audit_report['splits']
    return dict(purpose=purpose, model=asdict(load_config(root / config['architecture']).model), training=training,
        initialization={'path': candidate['checkpoint'], 'sha256': candidate['checkpoint_sha256'], 'semantics': 'weights_initialization'},
        tokenizer={'path': str(resolve_input(root, config['tokenizer'])), 'sha256': config['tokenizer_sha256']},
        data=dict(train=splits['train']['path'], validation=splits['validation']['path'],
            train_sha256=splits['train']['sha256'], validation_sha256=splits['validation']['sha256'],
            manifest_sha256=audit_report['manifest_sha256'], maximum_epochs=1, workers=0, balanced=False,
            bucket_size=128, sampling='seeded_no_replacement_epoch_shuffle', packing=False,
            sequence_policy='complete_conversation_dynamic_right_padding',
            validation_indices=splits['validation']['possibly_clean_indices'], cache_bytes=64 * 1024**2),
        optimization=dict(optimizer='adamw8bit', gradient_checkpointing=True, compile=False, tf32=False,
                          loss_normalization='supervised_tokens',
                          betas=[.9, .95], epsilon=1e-8, minimum_headroom_bytes=config['minimum_gpu_headroom_bytes']),
        evaluation=dict(validation_batches=96, quality_reserve_seconds=120, validation_reserve_seconds=120,
            development_suite_sha256=fingerprint(make_suite()), acceptance_suite_sha256=fingerprint(make_suite(acceptance=True)),
            regression_policy=config.get('evaluation', {}).get('regression_policy'),
            regression_patience=config.get('evaluation', {}).get('regression_patience', 2)),
        checkpoint=dict(minimum_save_seconds=120, slots=6,
            storage_backend=config.get('checkpoint_storage_backend', 'legacy'),
            store_root=config.get('checkpoint_store_root'), reserve_bytes=100 * 1024**3,
            retention=config.get('checkpoint_retention'),
            optimizer_storage_evidence=config.get('checkpoint_optimizer_storage_evidence'),
            reclamation_mode=config.get('checkpoint_reclamation_mode', 'disabled')), baseline_quality=candidate['metrics'],
        wall_clock_seconds=config['wall_clock_seconds'], finalization_reserve_seconds=min(1200, config['wall_clock_seconds'] / 4), evaluation_suite_version=config['evaluation_suite_version'],
        architecture_sha256=file_sha256(root / config['architecture']))


def ranking(report: dict) -> tuple:
    metrics = report['metrics']
    return (metrics['macro_task_accuracy'], metrics['END_rate'], -metrics['severe_degeneration_rate'])


def main(args, *, started: tuple[float, float] | None = None) -> int:
    """Execute one explicitly requested operation, never an automatic production launch."""
    root, config, preparation = read_configuration(args.config)
    if args.run != config['run_name']:
        raise ValueError('Run must match its configuration identity.')
    run_dir = output_root(root) / 'chat-training' / args.run
    if args.action == 'preflight':
        result = preflight(root, config, preparation)
        print(json.dumps(result, indent=2))
        manifest = preparation / 'launch-manifest.json'
        print('READINESS:', json.loads(manifest.read_text())['readiness'] if manifest.exists() else 'BLOCKED: preparation gates pending')
        return 0
    if args.action == 'prepare':
        from codexa_workspace.chat_validation import prepare
        if args.stage in ('audit', 'finalize'):
            prepare(root, config, preparation, stage=args.stage)
        else:
            with preparation_guard(preparation, config['preparation_gpu_seconds']):
                prepare(root, config, preparation, stage=args.stage)
        return 0
    if args.action in ('start', 'resume'):
        from codexa_workspace.chat_operations import launch
        return launch(root, config, preparation, run_dir, resume=args.action == 'resume', started=started)
    if args.action in ('_supervise', '_worker'):
        from codexa_workspace.chat_operations import supervise, worker
        return (supervise if args.action == '_supervise' else worker)(root, config, preparation, run_dir, resume=args.resume_worker)
    if args.action == 'view':
        from llm_training.conversational_training import start_chat_viewer
        start_chat_viewer(run_dir / 'progress.json', title=f'Codexa {args.run}')
        return 0
    if args.action == 'logs':
        path = run_dir / 'train_metrics.jsonl'
        if not path.exists():
            raise FileNotFoundError('No production metrics exist; no production job has run.')
        subprocess.run(['tail', '-n', str(args.tail), str(path)], check=True)
        return 0
    if args.action == 'stop':
        record = json.loads((run_dir / 'process.json').read_text())
        pid = record['worker_pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        if 'chat_training.py _worker' not in command or args.run not in command or ticks != record['worker_started_ticks']:
            raise RuntimeError('PID does not identify this run worker; no signal sent.')
        os.kill(pid, signal.SIGTERM)
        print('Graceful stop requested:', pid)
        return 0
    if args.action == 'evaluate':
        from codexa_workspace.chat_operations import evaluate_run
        evaluate_run(root, config, preparation, run_dir, suite=args.suite)
        return 0
    return 1
