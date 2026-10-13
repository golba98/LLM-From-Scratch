"""Operator adapter for a separately approved 20-minute END3 confirmation.

Delegates all training mathematics, recovery and viewing to existing packages.
No default/backend change or readiness-gate waiver is made.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml
from llm_architecture.checkpointing import file_sha256, verify_checkpoint_checksum
from llm_training.session import SessionBudget, atomic_document
from .chat_preparation import hardware


def validate_spec(spec: dict) -> None:
    """Keep the adapter restricted to the proposed bounded experiment."""
    if (spec['purpose'] != 'experimental' or spec['wall_clock_seconds'] != 1200
            or spec['finalization_reserve_seconds'] != 180
            or spec['training']['max_steps'] > 512
            or spec['checkpoint']['storage_backend'] != 'legacy'
            or spec['checkpoint'].get('reclamation_mode') != 'disabled'
            or spec['checkpoint']['reserve_bytes'] < 100 * 1024**3):
        raise ValueError('Only the declared bounded legacy experiment is supported.')


def preflight(root: Path, config: Path) -> dict:
    """Check identities and current resources without creating a training run."""
    document = yaml.safe_load(config.read_text())
    spec = document['spec']
    validate_spec(spec)
    checkpoint = Path(spec['initialization']['path'])
    if verify_checkpoint_checksum(checkpoint) != spec['initialization']['sha256']:
        raise ValueError('Initialization identity drift.')
    if file_sha256(Path(spec['tokenizer']['path'])) != spec['tokenizer']['sha256']:
        raise ValueError('Tokenizer identity drift.')
    for split in ('train', 'validation'):
        if file_sha256(Path(spec['data'][split])) != spec['data'][split + '_sha256']:
            raise ValueError('Dataset identity drift: ' + split)
    manifest = Path(spec['data']['train']).parent / 'dataset_manifest.json'
    if file_sha256(manifest) != spec['data']['manifest_sha256']:
        raise ValueError('Dataset manifest identity drift.')
    if file_sha256(root / 'configs/1b.yaml') != document['architecture_sha256']:
        raise ValueError('Architecture identity drift.')
    if os.environ.get('PYTORCH_ALLOC_CONF', 'default') != spec['optimization'].get('allocator', 'default'):
        raise ValueError('Allocator differs from the measured configuration.')
    workspace = spec['optimization'].get('cublas_workspace_config')
    if workspace and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
        raise ValueError('CUBLAS workspace environment differs.')
    from dataclasses import asdict
    from llm_training.config import load_config
    if spec['model'] != asdict(load_config(root / 'configs/1b.yaml').model):
        raise ValueError('Proposed model differs from the protected architecture.')
    observed = hardware()
    free = shutil.disk_usage(root).free
    gates = dict(disk_180_GiB=free >= 180*1024**3,
        host_12_GiB=observed['host_memory_bytes']['MemAvailable'] >= 12*1024**3,
        measured_gpu_headroom=observed['free_gpu_bytes'] >= document['required_free_gpu_bytes'])
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().decode(errors='replace')
        except (FileNotFoundError, PermissionError):
            continue
        if any(marker in command for marker in ('chat_training.py\x00_worker', 'chat_training.py\x00_supervise', 'bounded_chat_confirmation.py\x00_worker', 'train_chat_stage2.py')):
            raise RuntimeError('Conflicting training process: ' + proc.name)
    return dict(gates=gates, passed=all(gates.values()), free_bytes=free, hardware=observed,
                operator_files_sha256={str(p):file_sha256(p) for p in (root / 'scripts/legacy/bounded_chat_confirmation.py', Path(__file__))},
                config_sha256=file_sha256(config), spec=spec, run_name=document['run_name'],
                requires_separate_approval=True, quality_entry_gate_passed=False,
                purpose='Replication of END3 termination effect, not a production-quality model claim')


def launch(root: Path, config: Path, approval: Path, started: tuple[float, float]) -> None:
    """Launch only with explicit experiment-specific approval and fresh preflight."""
    authorization = json.loads(approval.read_text())
    evidence = preflight(root, config)
    if not evidence['passed']:
        raise RuntimeError('Resource gate failed: ' + repr(evidence['gates']))
    if (authorization.get('authorize_bounded_training') is not True
            or authorization.get('run_name') != evidence['run_name']
            or authorization.get('configuration_sha256') != evidence['config_sha256']
            or authorization.get('accept_experimental_quality_limitations') is not True):
        raise ValueError('Separate explicit approval must name this config and experimental limitations.')
    directory = root / 'outputs/chat-training' / evidence['run_name']
    directory.mkdir(parents=True, exist_ok=False)
    budget = SessionBudget.start(1200, 180)
    budget.started_utc, budget.deadline_utc = started[0], started[0] + 1200
    budget.deadline_monotonic = started[1] + 1200
    if budget.remaining <= 180:
        raise TimeoutError('Startup consumed the usable budget; no training started.')
    atomic_document(directory / 'budget.json', budget.document())
    atomic_document(directory / 'launch.json', dict(evidence=evidence, spec=evidence['spec'], authorization=authorization))
    with (directory / 'console.log').open('x') as console:
        child = subprocess.Popen([sys.executable, str(root / 'scripts/legacy/bounded_chat_confirmation.py'),
            '_supervise', '--config', str(config.absolute())], cwd=root,
            stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, directory=str(directory))))


def supervise(root: Path, config: Path) -> int:
    """Reuse the existing hard-deadline watchdog and shared single-run lock."""
    from .chat_operations import watch_owned_process
    name = yaml.safe_load(config.read_text())['run_name']
    directory = root / 'outputs/chat-training' / name
    with (directory.parent / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = SessionBudget.resume(json.loads((directory / 'budget.json').read_text()))
        child = subprocess.Popen([sys.executable, str(root / 'scripts/legacy/bounded_chat_confirmation.py'),
            '_worker', '--config', str(config.absolute())], cwd=root, start_new_session=True)
        atomic_document(directory / 'process.json', dict(supervisor_pid=os.getpid(), worker_pid=child.pid,
            deadline_utc=budget.deadline_utc,
            worker_started_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        return watch_owned_process(child, budget, directory)


def worker(root: Path, config: Path) -> int:
    """Use existing CPU/GPU training engine, viewer and behavioral diagnostics."""
    from llm_training.chat_run import run_chat
    from .chat_evaluation import evaluate_loaded, make_suite
    doc = yaml.safe_load(config.read_text())
    directory = root / 'outputs/chat-training' / doc['run_name']
    launch_document = json.loads((directory / 'launch.json').read_text())
    if launch_document['evidence']['config_sha256'] != file_sha256(config):
        raise ValueError('Config changed since explicit approval.')
    for path, expected in launch_document['evidence']['operator_files_sha256'].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError('Operator implementation changed since preflight.')
    budget = SessionBudget.resume(json.loads((directory / 'budget.json').read_text()))
    count = [0]
    def quality(model, tokenizer, deadline):
        count[0] += 1
        output = directory / 'development' / f'check-{count[0]:04}.jsonl'
        result = evaluate_loaded(model, tokenizer, make_suite(), deadline=deadline, output=output)
        atomic_document(output.with_suffix('.json'), result)
        return result['metrics']
    report = run_chat(doc['spec'], directory, budget, quality_evaluator=quality)
    atomic_document(directory / 'report.json', report)
    return 0
