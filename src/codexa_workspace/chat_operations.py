"""Explicit operator launches, isolated watchdogs and sealed final comparison."""

import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

from llm_training.session import SessionBudget, atomic_document


def permitted_experimental_launch(manifest: dict) -> bool:
    """Recognize a recorded user-authorized bounded experiment without falsifying gates."""
    authorization = manifest.get('experimental_authorization', {})
    waived = {'initialization_quality', 'behavioral_pilot', 'held_out_exposure_provenance', 'strict_live_memory_headroom'}
    failed = {key for key, passed in manifest['mandatory_gates'].items() if not passed}
    return (authorization.get('explicit_user_request') is True
            and manifest['spec']['wall_clock_seconds'] == 7200
            and manifest['spec']['purpose'] == 'experimental'
            and failed <= waived)


def launch(root: Path, config: dict, preparation: Path, directory: Path, *,
           resume: bool, started: tuple[float, float] | None) -> int:
    """Launch only after mandatory evidence gates; never called by preparation."""
    from codexa_workspace.chat_preparation import preflight, experiment_environment
    manifest = json.loads((preparation / 'launch-manifest.json').read_text())
    if not permitted_experimental_launch(manifest) and (manifest['readiness'] not in ('READY', 'CONDITIONALLY READY') or not all(manifest['mandatory_gates'].values())):
        raise RuntimeError('Production blocked: ' + ', '.join(k for k, v in manifest['mandatory_gates'].items() if not v))
    evidence = preflight(root, config, preparation)
    from llm_architecture.checkpointing import file_sha256
    if file_sha256(Path(manifest['configuration_file'])) != manifest['configuration_file_sha256']:
        raise ValueError('Configuration changed after readiness evidence was sealed.')
    selected = manifest['spec']['initialization']
    if file_sha256(root / config['architecture']) != manifest['spec']['architecture_sha256']:
        raise ValueError('Architecture configuration changed after preparation.')
    dataset_manifest_hash=evidence['dataset_manifest_sha256']
    if manifest['spec']['data'].get('mixture_manifest'):
        dataset_manifest_hash=file_sha256(Path(manifest['spec']['data']['train']))
    if dataset_manifest_hash != manifest['spec']['data']['manifest_sha256']:
        raise ValueError('Dataset manifest identity drift.')
    if os.environ.get('PYTORCH_ALLOC_CONF', 'default') != manifest['spec']['optimization'].get('allocator', 'default'):
        raise ValueError('Launch requires the explicitly measured CUDA allocator environment.')
    workspace = manifest['spec']['optimization'].get('cublas_workspace_config')
    if workspace and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
        raise ValueError('Launch requires the validated CUBLAS_WORKSPACE_CONFIG environment.')
    if evidence['candidate_sha256'][selected['path']] != selected['sha256']:
        raise ValueError('Selected initialization identity drift.')
    if evidence['hardware']['free_gpu_bytes'] < manifest['required_free_gpu_bytes']:
        raise RuntimeError('Current GPU memory cannot preserve measured training headroom.')
    current = experiment_environment(root)
    for name, frozen in manifest['environment']['repositories'].items():
        record = current['repositories'][name]
        if (record['commit'] != frozen['commit'] or record['source_import_differences']
                or record['source_tree_sha256'] != frozen['source_tree_sha256']):
            if name != 'LLM-Specialist':
                raise ValueError(f'Implementation changed or wheel is stale: {name}.')
    for line in evidence['hardware']['gpu_processes'].splitlines()[1:]:
        pid = line.split(',', 1)[0].strip()
        command_path = Path('/proc') / pid / 'cmdline'
        if command_path.exists():
            command = command_path.read_bytes().decode(errors='replace')
            if any(marker in command for marker in ('train_chat_stage2.py', 'chat_training.py', 'train.py', 'train_sft')):
                raise RuntimeError(f'Conflicting GPU training process {pid}; no process was terminated.')
    if resume:
        previous_report = directory / 'report.json'
        if previous_report.exists() and json.loads(previous_report.read_text())['state'] in ('COMPLETED', 'EARLY_STOPPED'):
            raise ValueError('A completed/stopped phase requires a new run identity, not additional training on the sealed test.')
        if not directory.exists():
            raise FileNotFoundError('No production run exists to resume.')
        budget = SessionBudget.resume(json.loads((directory / 'budget.json').read_text()))
    else:
        directory.mkdir(parents=True, exist_ok=False)
        seconds = manifest['spec']['wall_clock_seconds']
        budget = SessionBudget.start(seconds, manifest['spec']['finalization_reserve_seconds'])
        if started:
            budget.started_utc, budget.deadline_utc = started[0], started[0] + seconds
            budget.deadline_monotonic = started[1] + seconds
    atomic_document(directory / 'budget.json', budget.document())
    atomic_document(directory / 'launch.json', manifest)
    arguments = [sys.executable, str(root / 'scripts/chat_training.py'), '_supervise', '--config',
                 config['_configuration_path'], '--run', config['run_name']]
    if resume:
        arguments.append('--resume-worker')
    with (directory / 'console.log').open('a') as console:
        child = subprocess.Popen(arguments, cwd=root, stdout=console, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(f'Explicit operator launch: supervisor PID {child.pid}; {directory}')
    return 0


def supervise(root: Path, config: dict, preparation: Path, directory: Path, *, resume: bool) -> int:
    """Own a single run lock and enforce the original deadline independently."""
    lock_path = directory.parent / '.training.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = SessionBudget.resume(json.loads((directory / 'budget.json').read_text()))
        command = [sys.executable, str(root / 'scripts/chat_training.py'), '_worker', '--config',
                   config['_configuration_path'], '--run', config['run_name']]
        if resume:
            command.append('--resume-worker')
        child = subprocess.Popen(command, cwd=root, start_new_session=True)
        atomic_document(directory / 'process.json', {'supervisor_pid': os.getpid(), 'worker_pid': child.pid,
            'run_name': config['run_name'], 'deadline_utc': budget.deadline_utc,
            'worker_started_ticks': Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]})
        return watch_owned_process(child, budget, directory)


def watch_owned_process(child: subprocess.Popen, budget: SessionBudget, directory: Path) -> int:
    """Bound a separately-sessioned owned worker without signalling unrelated PIDs."""
    while child.poll() is None and budget.remaining > 0:
        time.sleep(min(1.0, budget.remaining))
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        atomic_document(directory / 'hard-deadline.json', {'state': 'DEADLINE_KILLED',
            'budget': budget.document(), 'recovery': 'last published recovery-index checkpoint only'})
    return child.returncode


def worker(root: Path, config: dict, preparation: Path, directory: Path, *, resume: bool) -> int:
    """Train and finalize under the supervisor's persisted original budget."""
    from llm_training.chat_run import run_chat
    from codexa_workspace.chat_evaluation import evaluate_loaded, make_suite
    launch_document = json.loads((directory / 'launch.json').read_text())
    if not all(launch_document['mandatory_gates'].values()) and not permitted_experimental_launch(launch_document):
        raise RuntimeError('Mandatory preparation gate failed; worker refused.')
    budget = SessionBudget.resume(json.loads((directory / 'budget.json').read_text()))
    spec = launch_document['spec']
    development_cases=make_suite()
    if spec['evaluation'].get('development_suite_path'):
        from .chat_evaluation import fingerprint
        development_cases=json.loads(Path(spec['evaluation']['development_suite_path']).read_text())
        if (fingerprint(development_cases)!=spec['evaluation']['development_suite_sha256']
                or any(case.get('protected') for case in development_cases)):
            raise ValueError('Development suite identity drift or protected-case exposure.')
    quality_counter = [0]
    def quality(model, tokenizer, deadline):
        quality_counter[0] += 1
        evaluation = evaluate_loaded(model, tokenizer, development_cases, deadline=deadline,
            output=directory / 'development' / f'check-{quality_counter[0]:04}.jsonl')
        atomic_document(directory / 'development' / f'check-{quality_counter[0]:04}.json', evaluation)
        return evaluation['metrics']
    report = run_chat(spec, directory, budget, resume=resume, quality_evaluator=quality)
    if spec['purpose'] != 'experimental' and report['state'] in ('COMPLETED', 'EARLY_STOPPED') and budget.remaining > 0:
        try:
            evaluate_run(root, config, preparation, directory, suite='acceptance', deadline=budget.deadline_monotonic)
            report['final_acceptance'] = 'COMPLETE'
        except TimeoutError as error:
            report['final_acceptance'] = 'INCOMPLETE_WITHIN_BUDGET'
            report['final_acceptance_error'] = str(error)
        except Exception as error:
            report['final_acceptance'] = 'FAILED'
            report['final_acceptance_error'] = f'{type(error).__name__}: {error}'
    report['budget'] = budget.document()
    atomic_document(directory / 'report.json', report)
    atomic_document(directory / 'budget.json', budget.document())
    return 1 if report.get('final_acceptance') == 'FAILED' else 0


def evaluate_run(root: Path, config: dict, preparation: Path, directory: Path, *,
                 suite: str, deadline: float | None = None) -> dict:
    """Compare base/init/best/final after training; never promote or overwrite exports."""
    from codexa_workspace.assets import resolve_input
    from codexa_workspace.chat_evaluation import evaluate_candidate, make_suite
    from llm_training.recovery import RecoveryStore
    report = json.loads((directory / 'report.json').read_text())
    if report['state'] not in ('COMPLETED', 'EARLY_STOPPED', 'INTERRUPTED'):
        raise ValueError('Evaluate only a stopped production run.')
    if suite == 'acceptance' and report['state'] == 'INTERRUPTED':
        raise ValueError('Protected acceptance is sealed until the training phase is finished.')
    launch_document = json.loads((directory / 'launch.json').read_text())
    spec = launch_document['spec']
    store = RecoveryStore(directory / 'checkpoints',
        backend=spec['checkpoint'].get('storage_backend', 'legacy'),
        store_root=Path(spec['checkpoint']['store_root']) if spec['checkpoint'].get('store_root') else None)
    from .chat_evaluation import fingerprint
    cases=make_suite(acceptance=suite=='acceptance')
    suite_path=spec['evaluation'].get(suite+'_suite_path')
    if suite_path:
        cases=json.loads(Path(suite_path).read_text())
        if fingerprint(cases)!=spec['evaluation'][suite+'_suite_sha256']:
            raise ValueError('Evaluation suite identity drift.')
        if suite=='development' and any(case.get('protected') for case in cases):
            raise ValueError('Protected cases cannot be development evaluations.')
    comparisons = {'base': resolve_input(root, config['candidates'][0]),
        'initialization': Path(spec['initialization']['path']), 'best': store.resolve('best'),
        'final': store.resolve('final') if 'final' in store.index() else store.resolve()}
    if 'best_validation' in store.index():
        comparisons['best_validation']=store.resolve('best_validation')
    target = directory / 'evaluation' / suite
    if (target / 'comparison.json').exists():
        raise FileExistsError('Protected comparison already exists; preserve its first acceptance result.')
    results = {}
    for label, checkpoint in comparisons.items():
        for sampled in (False, True):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Final comparison did not fit the reserved session time.')
            key = label + ('-sampled' if sampled else '-greedy')
            path = target / key
            if (path / 'report.json').exists():
                from llm_architecture.checkpointing import file_sha256
                cached = json.loads((path / 'report.json').read_text())
                from codexa_workspace.chat_evaluation import fingerprint
                if (cached['checkpoint_sha256'] != file_sha256(checkpoint)
                        or cached['suite_sha256'] != fingerprint(cases)
                        or cached['decoding']['do_sample'] != sampled):
                    raise ValueError('Completed evaluation identity drift; no artifact overwritten.')
                results[key] = cached
                continue
            if path.exists():
                shutil.move(path, path.with_name(key + '-incomplete-' + uuid.uuid4().hex[:8]))
            results[key] = evaluate_candidate(checkpoint, Path(spec['tokenizer']['path']), target / key,
                cases=cases, sampled=sampled, deadline=deadline)
    comparison = {'results': results, 'promotion': 'NOT_PERFORMED',
        'limitations': ['Scoring is versioned; independent human semantic acceptance remains separate.',
                       'Corpus overlap and exposure limitations remain in the preparation data audit.']}
    atomic_document(target / 'comparison.json', comparison)
    return comparison
