"""Frozen five-hour foundational continued-pretraining job, launch gates and trainer entry.

The pilot proposal remains the source of every hyperparameter; this job changes
only its wall-clock, update and generation-schedule ceilings.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from llm_training.session import atomic_document



RUN_ID = 'codexa-934m-foundation-5h-v1'
JOB_FORMAT = 'codexa-foundation-training-job-v1'
APPROVAL_FORMAT = 'codexa-foundation-training-approval-v1'
PREPARED = Path('outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared')
WALL_CLOCK_SECONDS = 18000
FINALIZATION_RESERVE_SECONDS = 600
MAX_STEPS = 1792
GENERATION_STEPS = [896]
GIB = 1024**3
DISK_FLOOR_BYTES = 180 * GIB
RECOVERY_RESERVE_BYTES = 100 * GIB
NATIVE_PAYLOAD_BYTES = 5636391032
HOST_RAM_BYTES = 16 * GIB
VRAM_FREE_MIB = 13928
OVERRIDDEN_BLOCKERS = ['neutral-format evidence and genuine review']
CODE_FILES = ('src/codexa_workspace/foundation_training.py', 'src/codexa_workspace/training_supervisor.py',
              'src/codexa_workspace/training_wakeup.py', 'src/codexa_workspace/foundation_diagnosis.py')
COMPONENT_MODULES = ('llm_training.pretrain_run', 'llm_training.training', 'llm_training.recovery',
                     'llm_training.session', 'llm_training.checkpointing', 'llm_data.token_views',
                     'llm_data.token_data', 'llm_architecture.model')
FOCUSED_TESTS = (('LLM-Training', ['tests/test_pretrain_run.py', 'tests/test_readiness_recovery.py']),
                 ('LLM-From-Scratch', ['tests/test_training_supervisor.py', 'tests/test_pretraining_readiness.py']))


def fingerprint(value: object) -> str:
    """Canonical JSON identity, identical to chat_evaluation.fingerprint without its torch import."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path) -> str:
    """Streaming digest without importing torch (keeps the supervisor CLI lightweight)."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_directory(root: Path, run_id: str = RUN_ID) -> Path:
    return root / 'outputs/training' / run_id


def code_identity(root: Path) -> dict:
    """Hash the integration sources and the installed component modules actually imported."""
    import importlib
    identities = {name: file_sha256(root / name) for name in CODE_FILES}
    for name in COMPONENT_MODULES:
        identities[name] = file_sha256(Path(importlib.import_module(name).__file__))
    return identities


def frozen_order(prepared: Path, selection: dict, count: int) -> np.ndarray:
    """Seed-42 traversal identical to the readiness draw, extended to the 5h budget."""
    import torch
    train = np.load(prepared / 'train_window_ids.npy', mmap_mode='r', allow_pickle=False)
    generator = torch.Generator().manual_seed(selection['seed'])
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(len(train), generator=generator)[:count].numpy()
    result = np.asarray(train[order], dtype='<u4')
    pilot = np.load(prepared / 'pilot_window_order.npy', allow_pickle=False)
    if not np.array_equal(result[:len(pilot)], pilot):
        raise ValueError('Extended order does not reproduce the frozen pilot prefix.')
    if len(np.unique(result)) != len(result):
        raise ValueError('Extended order repeats a window.')
    return result


def build_spec(root: Path, config_dir: Path) -> tuple[dict, dict]:
    """Derive the 5h specification from the validated pilot proposal; write its data arrays."""
    from .pretraining_readiness import validate_proposal, validate_selection
    import yaml
    prepared = root / PREPARED
    proposal = json.loads((prepared / 'foundation-pilot.proposed.json').read_text())
    validate_proposal(proposal)
    selection = validate_selection(prepared)
    if proposal['data_selection_sha256'] != selection['selection_sha256']:
        raise ValueError('Proposal and selection identities differ.')
    exposure = json.loads((prepared / 'pilot-source-exposure.json').read_text())
    if file_sha256(prepared / 'pilot_window_order.npy') != exposure['pilot_window_order_sha256']:
        raise ValueError('Pilot order fingerprint drift.')
    config = yaml.safe_load(Path(proposal['configuration_file']).read_text())
    training = dict(config['training'])
    if training['max_steps'] != 256 or training['precision'] != 'bf16':
        raise ValueError('Unexpected pilot configuration.')
    training['max_steps'] = MAX_STEPS
    microbatches = MAX_STEPS * training['gradient_accumulation_steps'] * training['micro_batch_size']
    config_dir.mkdir(parents=True, exist_ok=True)
    order_path = config_dir / 'train_window_order.npy'
    order = frozen_order(prepared, selection, microbatches)
    if order_path.exists():
        if not np.array_equal(np.load(order_path, allow_pickle=False), order):
            raise ValueError('Existing frozen order differs.')
    else:
        np.save(order_path, order, allow_pickle=False)
    validation = np.load(prepared / 'validation_window_ids.npy', allow_pickle=False)
    documents = json.loads((prepared / 'validation-documents.json').read_text())
    sources = {str(row['window_id']): row['source'] for row in documents}
    if sorted(int(key) for key in sources) != [int(value) for value in validation]:
        raise ValueError('Validation windows and source attributions differ.')
    sources_path = config_dir / 'validation_sources.json'
    serialized = json.dumps(sources, sort_keys=True, indent=1) + '\n'
    if sources_path.exists() and sources_path.read_text() != serialized:
        raise ValueError('Existing validation source map differs.')
    sources_path.write_text(serialized)
    guards = proposal['quality_guards']
    spec = dict(format='codexa-foundation-pretraining-spec-v1', run_name=RUN_ID, device='cuda',
        viewer_required=True, expected_parameters=934356480,
        model=proposal['architecture'], training=training,
        initialization=dict(path=proposal['initialization']['path'], sha256=proposal['initialization']['sha256'],
                            policy=proposal['initialization']['policy']),
        tokenizer=dict(path=proposal['tokenizer_path'], sha256=proposal['tokenizer_sha256']),
        data=dict(corpus_path=selection['corpus_path'], dtype='uint16', selection_sha256=selection['selection_sha256'],
                  corpus_input_identity=selection['corpus_input_identity'],
                  train_window_order=str(order_path), train_window_order_sha256=file_sha256(order_path),
                  pilot_prefix_sha256=exposure['pilot_window_order_sha256'],
                  validation_window_ids=str(prepared / 'validation_window_ids.npy'),
                  validation_window_ids_sha256=selection['view_arrays']['validation_window_ids.npy']['sha256'],
                  validation_sources=str(sources_path), validation_sources_sha256=file_sha256(sources_path),
                  source_loss_weights=selection['source_loss_weights']),
        optimization=dict(optimizer=proposal['optimizer']['name'], betas=proposal['optimizer']['betas'],
                          epsilon=proposal['optimizer']['epsilon'], minimum_learning_rate=proposal['minimum_learning_rate'],
                          gradient_checkpointing=proposal['gradient_checkpointing'],
                          loss_normalization=proposal['loss_normalization'], tf32=False, allocator='default',
                          minimum_headroom_bytes=1536 * 1024**2),
        evaluation=dict(cadence_updates=proposal['evaluation']['cadence_updates'], validation_batches=None,
                        generation_steps=GENERATION_STEPS, generation_forms=['neutral', 'direct'],
                        generation_cases=proposal['evaluation']['generation_cases'],
                        generation_reserve_seconds=420, validation_overrun_seconds=0),
        guards=dict(gradient_norm_threshold=guards['gradient_norm_threshold'],
                    gradient_norm_patience=guards['gradient_norm_patience'],
                    source_validation_CE_increase=guards['source_validation_CE_increase'],
                    validation_patience=guards['validation_patience']),
        checkpoint=dict(backend=proposal['checkpoint_backend'], interval=training['checkpoint_interval'],
                        minimum_save_seconds=120, reserve_bytes=RECOVERY_RESERVE_BYTES,
                        estimated_checkpoint_bytes=NATIVE_PAYLOAD_BYTES),
        budget=dict(seconds=WALL_CLOCK_SECONDS, reserve_seconds=FINALIZATION_RESERVE_SECONDS))
    derivation = dict(proposal_sha256=file_sha256(prepared / 'foundation-pilot.proposed.json'),
        proposal_configuration_sha256=proposal['configuration_sha256'],
        changed_from_pilot=dict(wall_clock_seconds=[proposal['wall_clock_seconds'], WALL_CLOCK_SECONDS],
            max_steps=[256, MAX_STEPS], maximum_targets=[proposal['maximum_targets'], microbatches * 2048],
            generation_steps=['every 128 updates', GENERATION_STEPS + ['baseline', 'final']],
            finalization_reserve_seconds=[proposal['finalization_reserve_seconds'], FINALIZATION_RESERVE_SECONDS]),
        overridden_blockers=OVERRIDDEN_BLOCKERS)
    return spec, derivation


def prepare(root: Path, run_dir: Path) -> dict:
    """Freeze job.json; refuse to replace a different existing job."""
    config_dir = run_dir / 'config'
    spec, derivation = build_spec(root, config_dir)
    from llm_training.pretrain_run import identity
    job = dict(format=JOB_FORMAT, run_id=RUN_ID, spec=spec, spec_sha256=identity(spec),
               derivation=derivation, code_identity=code_identity(root),
               wakeup=dict(provider='codex'), no_automatic_follow_up_training=True)
    job['job_sha256'] = fingerprint(job)
    path = config_dir / 'job.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['job_sha256'] != job['job_sha256']:
            if (run_dir / 'metrics/run_metadata.json').exists():
                raise ValueError('A started run cannot be re-frozen with a different job.')
            path.replace(config_dir / f"job.superseded-{existing['job_sha256'][:12]}.json")
    atomic_document(path, job)
    return job


def approve(run_dir: Path, job_sha256: str, statement: str) -> dict:
    """Bind a user's explicit grant to one exact frozen job hash."""
    job = json.loads((run_dir / 'config/job.json').read_text())
    if job['job_sha256'] != job_sha256:
        raise ValueError('Approval hash does not match the frozen job.')
    grant = dict(format=APPROVAL_FORMAT, approved=True, job_sha256=job_sha256, run_id=job['run_id'],
                 statement=statement, overridden_blockers=OVERRIDDEN_BLOCKERS,
                 recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic_document(run_dir / 'config/approval.json', grant)
    return grant


def verify_job(root: Path, run_dir: Path) -> dict:
    """Reject stale code, drifted specification or a missing/mismatched grant."""
    job = json.loads((run_dir / 'config/job.json').read_text())
    unsigned = {key: value for key, value in job.items() if key != 'job_sha256'}
    if fingerprint(unsigned) != job['job_sha256']:
        raise ValueError('Frozen job document was modified.')
    if code_identity(root) != job['code_identity']:
        raise ValueError('Code identity changed after the job was frozen.')
    grant_path = run_dir / 'config/approval.json'
    grant = json.loads(grant_path.read_text()) if grant_path.exists() else {}
    if grant.get('format') != APPROVAL_FORMAT or grant.get('approved') is not True or grant.get('job_sha256') != job['job_sha256']:
        raise ValueError('Job-specific approval is missing or mismatched.')
    return job


def host_available_bytes() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable not reported.')


def gpu_snapshot() -> dict:
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    if query.returncode:
        return dict(available=False, error=query.stderr.strip()[-300:])
    name, total, used, free = [value.strip() for value in query.stdout.strip().splitlines()[0].split(',')]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    return dict(available=True, name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free),
                compute_apps=[line.strip() for line in apps.stdout.splitlines() if line.strip()])


def run_focused_tests(root: Path) -> dict:
    """Bounded CPU-only regression gate; never touches CUDA or real assets."""
    results = {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    for repository, paths in FOCUSED_TESTS:
        cwd = root.parent / repository
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *paths]
        started = time.monotonic()
        completed = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=1800)
        results[repository] = dict(command=command, exit_code=completed.returncode,
                                   summary=(completed.stdout.strip().splitlines() or [''])[-1],
                                   seconds=round(time.monotonic() - started, 1))
    return results


def validate_launch(root: Path, run_dir: Path, *, run_tests: bool = True) -> dict:
    """Evaluate every mandatory gate; any failure blocks launch without lowering a threshold."""
    gates = []

    def gate(name, check):
        try:
            observed = check()
            passed, detail = observed if isinstance(observed, tuple) else (True, observed)
        except Exception as error:
            passed, detail = False, f'{type(error).__name__}: {error}'
        gates.append(dict(name=name, passed=bool(passed), detail=detail))
        return passed

    job = {}

    def job_gate():
        job.update(verify_job(root, run_dir))
        return True, dict(job_sha256=job['job_sha256'])
    gate('approval_and_code_identity', job_gate)
    spec = job.get('spec')
    if spec:
        rebuilt, _ = build_spec(root, run_dir / 'config')
        gate('frozen_specification', lambda: (rebuilt == spec, 'rebuilt specification equals frozen job'))
        gate('base_checkpoint_sha256', lambda: (file_sha256(spec['initialization']['path']) == spec['initialization']['sha256'],
                                                spec['initialization']['sha256']))
        gate('tokenizer_sha256', lambda: (file_sha256(spec['tokenizer']['path']) == spec['tokenizer']['sha256'],
                                          spec['tokenizer']['sha256']))

        def data_gate():
            data = spec['data']
            stat = os.stat(data['corpus_path'])
            identity = data['corpus_input_identity']
            same = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) == (
                identity['device'], identity['inode'], identity['bytes'], identity['mtime_ns'])
            order = np.load(data['train_window_order'], allow_pickle=False)
            pilot = np.load(root / PREPARED / 'pilot_window_order.npy', allow_pickle=False)
            checks = dict(corpus_identity=same,
                order_sha=file_sha256(data['train_window_order']) == data['train_window_order_sha256'],
                pilot_prefix=bool(np.array_equal(order[:len(pilot)], pilot)),
                validation_sha=file_sha256(data['validation_window_ids']) == data['validation_window_ids_sha256'],
                sources_sha=file_sha256(data['validation_sources']) == data['validation_sources_sha256'],
                held_out_disjoint=not np.intersect1d(order, np.load(data['validation_window_ids'])).size)
            return all(checks.values()), checks
        gate('indexed_dataset_and_validation_selection', data_gate)

        def optimizer_gate():
            t, o = spec['training'], spec['optimization']
            expected = dict(lr=2e-5, min_lr=2e-6, warmup=20, max_steps=MAX_STEPS, accumulation=32, microbatch=1,
                            optimizer='adamw8bit', betas=[.9, .95], eps=1e-8, wd=.1, clip=1., precision='bf16', seed=42)
            observed = dict(lr=t['learning_rate'], min_lr=o['minimum_learning_rate'], warmup=t['warmup_steps'],
                            max_steps=t['max_steps'], accumulation=t['gradient_accumulation_steps'],
                            microbatch=t['micro_batch_size'], optimizer=o['optimizer'], betas=o['betas'],
                            eps=o['epsilon'], wd=t['weight_decay'], clip=t['gradient_clip'],
                            precision=t['precision'], seed=t['seed'])
            return observed == expected, observed
        gate('optimizer_and_schedule', optimizer_gate)

        def checkpoint_gate():
            c = spec['checkpoint']
            ok = (c['backend'] == 'legacy' and c['interval'] == 64 and c['reserve_bytes'] >= RECOVERY_RESERVE_BYTES)
            return ok, c
        gate('checkpoint_policy', checkpoint_gate)
    if run_tests:
        def tests_gate():
            results = run_focused_tests(root)
            return all(value['exit_code'] == 0 for value in results.values()), results
        gate('focused_tests', tests_gate)

    def disk_gate():
        run_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(run_dir).free
        peak = 7 * NATIVE_PAYLOAD_BYTES + GIB
        return (free >= DISK_FLOOR_BYTES and free - peak >= RECOVERY_RESERVE_BYTES,
                dict(free_bytes=free, free_gib=round(free / GIB, 3), floor_gib=180,
                     projected_peak_gib=round(peak / GIB, 2), reserve_gib=100))
    gate('disk_capacity', disk_gate)
    gate('host_ram', lambda: (host_available_bytes() >= HOST_RAM_BYTES,
                              dict(available_gib=round(host_available_bytes() / GIB, 2), required_gib=16)))

    def gpu_gate():
        snapshot = gpu_snapshot()
        return snapshot.get('available') and snapshot['free_mib'] >= VRAM_FREE_MIB, dict(snapshot, required_free_mib=VRAM_FREE_MIB)
    gate('gpu_vram', gpu_gate)

    def cuda_gate():
        probe = subprocess.run([sys.executable, '-c', 'import torch, bitsandbytes, json; '
            'print(json.dumps(dict(cuda=torch.cuda.is_available(), bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '
            'torch=torch.__version__, bnb=bitsandbytes.__version__)))'], capture_output=True, text=True, timeout=300)
        if probe.returncode:
            return False, probe.stderr.strip()[-500:]
        value = json.loads(probe.stdout.strip().splitlines()[-1])
        return value['cuda'] and value['bf16'], value
    gate('cuda_bf16_bitsandbytes', cuda_gate)

    def namespace_gate():
        started = (run_dir / 'metrics/run_metadata.json').exists() or (run_dir / 'checkpoints').exists()
        state_path = run_dir / 'supervisor/state.json'
        state = json.loads(state_path.read_text())['state'] if state_path.exists() else None
        return not started and state in (None, 'PREPARING', 'VALIDATING', 'READY', 'BLOCKED'), dict(started=started, state=state)
    gate('fresh_namespace', namespace_gate)
    gate('visible_viewer_environment', lambda: (bool(shutil.which('kitty')) and bool(
        os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY')),
        dict(kitty=shutil.which('kitty'), wayland=os.environ.get('WAYLAND_DISPLAY'), display=os.environ.get('DISPLAY'))))
    gate('not_inside_wakeup_agent', lambda: (not os.environ.get('CODEXA_WAKEUP_AGENT'), 'wakeup agents cannot launch training'))
    gate('baseline_evaluation', lambda: (True, 'enforced in-run: baseline validation must complete before any optimizer update'))
    passed = all(item['passed'] for item in gates)
    record = dict(run_id=RUN_ID, checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  passed=passed, failed=[item['name'] for item in gates if not item['passed']], gates=gates)
    (run_dir / 'config').mkdir(parents=True, exist_ok=True)
    atomic_document(run_dir / 'config/launch-gates.json', record)
    with (run_dir / 'config/launch-gates-history.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(checked_utc=record['checked_utc'], passed=passed, failed=record['failed'])) + '\n')
    return record


def generation_evaluator(spec: dict):
    """Frozen 24-concept neutral/direct cases (48), greedy 128 tokens, saved for human review.

    Mirrors foundation_diagnosis.evaluate_model (hash-frozen by the earlier diagnostic job),
    whose aggregate assumes all five formats and cannot summarize a neutral/direct subset.
    """
    from llm_tokenizer.tokenizer import load_tokenizer
    from .foundation_diagnosis import make_cases, scoring, likelihood_probe
    tokenizer = load_tokenizer(Path(spec['tokenizer']['path']))
    forms = tuple(spec['evaluation']['generation_forms'])
    cases = [case for case in make_cases(tokenizer) if case['form'] in forms]
    if len(cases) != spec['evaluation']['generation_cases']:
        raise ValueError('Frozen generation case count drift.')

    def evaluate(model, directory: Path, deadline: float) -> dict:
        import torch
        from contextlib import nullcontext
        from llm_inference.generate import GenerationConfig, generate_sequences
        device = spec['device']
        decode = GenerationConfig(max_new_tokens=128, repetition_penalty=1, seed=42, do_sample=False)
        output = Path(directory) / 'outputs.jsonl'
        rows = []
        model.eval()
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()

        def finite_decode_logits(_model, _inputs, result):
            if not torch.isfinite(result[0][:, -1, :]).all().item():
                raise FloatingPointError('Nonfinite free-generation logits; no answer is scored.')
        with torch.inference_mode(), context, model.register_forward_hook(finite_decode_logits):
            for case in cases:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Generation deadline reached.')
                inputs = torch.tensor([case['input_ids']], device=device, dtype=torch.long)
                generated = generate_sequences(model, inputs, eos_token_id=2, pad_token_id=0,
                                               stop_sequences=[[7]], config=decode).sequences[0]
                text = tokenizer.decode(list(generated.visible_token_ids), skip_special_tokens=True).strip()
                row = dict(**case, response=text, raw_token_ids=list(generated.token_ids),
                           terminator=generated.terminating_token_id, termination=generated.termination_cause,
                           generated_tokens=generated.generated_token_count,
                           metrics=scoring(text, case, generated.terminating_token_id),
                           likelihood=likelihood_probe(model, tokenizer, case, device=device))
                with output.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        by_form = {form: [row for row in rows if row['form'] == form] for form in forms}
        return dict(cases=len(rows), outputs_path=str(output), semantic_review_pending=len(rows),
                    exact_surface_by_form={form: sum(r['metrics']['exact_surface_correct'] for r in values) / len(values)
                                           for form, values in by_form.items() if values},
                    expected_preferred_by_likelihood=sum(r['likelihood']['expected']['mean_token_nll']
                                                         < r['likelihood']['distractor']['mean_token_nll'] for r in rows))
    return evaluate


def trainer_main(arguments: list[str] | None = None) -> int:
    """Entry the supervisor spawns; refuses unapproved or drifted jobs."""
    from llm_training.pretrain_run import run_pretraining
    from llm_training.session import SessionBudget
    parser = argparse.ArgumentParser(description='Run the approved frozen foundational job.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(arguments)
    job = verify_job(args.root, args.run_dir)
    spec = job['spec']
    if args.resume:
        budget = SessionBudget.resume(json.loads((args.run_dir / 'metrics/budget.json').read_text()))
    else:
        budget = SessionBudget.start(spec['budget']['seconds'], spec['budget']['reserve_seconds'])
    report = run_pretraining(spec, args.run_dir, budget, resume=args.resume,
                             generation_evaluator=generation_evaluator(spec))
    return 0 if report['state'] in ('COMPLETED', 'EARLY_STOPPED', 'INTERRUPTED') else 1


if __name__ == '__main__':
    raise SystemExit(trainer_main())
