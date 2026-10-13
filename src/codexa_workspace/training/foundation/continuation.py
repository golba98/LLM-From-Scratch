"""Bounded 10-hour continued pretraining of the 934M foundation: frozen job, readiness, launch, trainer.

A *new* job, never an exact continuation of ``codexa-934m-foundation-5h-v1``:

- initialization: the verified foundation checkpoint ``ca779a16…`` (weights only) with a fresh
  AdamW8bit optimizer and a fresh, independently specified cosine schedule. The predecessor's
  optimizer moments are deliberately not reused: its schedule had already decayed to its minimum,
  its data order is exhausted, its code identity is superseded, and exact resume semantics would
  misrepresent this run (see FOUNDATION_10H_EXPERIMENT_PLAN.md);
- data: the same seed-42 permutation of the selection's eligible windows that the 5h run used for
  its first 57,344 windows, continued from position 57,344. The new order is therefore disjoint
  from the 5h run, from the 256 held-out validation windows and from base-v1's reconstructed
  positions (all excluded by the selection), and deterministic;
- budget: <= 36,000 s wall-clock from trainer start, including startup and finalization
  (``SessionBudget``), with an update cap derived conservatively from measured timings;
- telemetry, validation, periodic capability scoring and full evaluations through
  ``llm_training.pretrain_run`` and ``capability_benchmark``, bound to a sealed step-zero baseline.

Nothing here launches training on import. ``launch_authorized`` and ``resume_authorized`` refuse
unless the user's grant names this exact job hash and every gate passes at that moment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable

import numpy as np

from codexa_workspace.hashing import file_sha256, fingerprint
from codexa_workspace.paths import PYTHON, REPO_ROOT as ROOT, SCRIPTS, WORKSPACE
from codexa_workspace.training.environment import LaunchRefused, SUPERVISOR_SCRIPT

RUN_ID = 'codexa-934m-foundation-continuation-10h-v1'
JOB_FORMAT = 'codexa-foundation-continuation-job-v1'
SPEC_FORMAT = 'codexa-foundation-pretraining-spec-v2'
APPROVAL_FORMAT = 'codexa-foundation-continuation-approval-v1'
BASELINE_FORMAT = 'codexa-foundation-step-zero-baseline-v1'
PREDECESSOR = ROOT / 'outputs/training/codexa-934m-foundation-5h-v1'
INITIALIZATION = PREDECESSOR / 'checkpoints/slot-0/checkpoint.pt'
INITIALIZATION_SHA256 = 'ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e'
PREPARED = ROOT / 'outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared'
WALL_CLOCK_SECONDS = 36000
UPDATE_TOKENS = 32 * 2048
CADENCE_UPDATES = 224          # validation + periodic capability scoring (~30 min at 8.1 s/update)
CHECKPOINT_UPDATES = 896       # verified recovery checkpoint (~2 h)
GENERATION_UPDATES = (896, 1792, 2688)   # full evaluations (~2 h) before the final one
RESUMABLE_STOPS = ('MEMORY_HEADROOM_GATE', 'DISK_HEADROOM', 'DISK_RESERVE_WATCHDOG')
READY, CONDITIONAL, NOT_READY = 'READY FOR AUTHORIZATION', 'CONDITIONALLY READY', 'NOT READY'
# Everything that decides what this job trains on and how it is evaluated and supervised.
CODE_FILES = ('src/codexa_workspace/training/foundation/continuation.py',
              'src/codexa_workspace/training/foundation/selection.py',
              'src/codexa_workspace/training/gates.py', 'src/codexa_workspace/training/environment.py',
              'src/codexa_workspace/training/supervisor/state.py', 'src/codexa_workspace/training/supervisor/runner.py',
              'src/codexa_workspace/training/supervisor/cli.py',
              'src/codexa_workspace/evaluation/capability/benchmark.py',
              'src/codexa_workspace/evaluation/capability/suites/foundation_capability_v1.json',
              'src/codexa_workspace/evaluation/capability/suites/foundation_capability_v2.json',
              'src/codexa_workspace/evaluation/language_suite.py', 'src/codexa_workspace/evaluation/generation.py',
              'src/codexa_workspace/hashing.py',
              'scripts/training_supervisor.py', 'scripts/foundation_continuation.py',
              'tools/training_environment_guard.py')
COMPONENT_MODULES = ('llm_training.pretrain_run', 'llm_training.training', 'llm_training.telemetry',
                     'llm_training.recovery', 'llm_training.session', 'llm_training.checkpointing',
                     'llm_training.config', 'llm_data.token_views', 'llm_data.token_data',
                     'llm_architecture.model', 'llm_architecture.checkpointing', 'llm_architecture.runtime',
                     'llm_inference.generate', 'llm_tokenizer.tokenizer')


def read(path) -> dict:
    return json.loads(Path(path).read_text())


def code_identity(root: Path = ROOT) -> dict:
    import importlib
    identities = {name: file_sha256(root / name) for name in CODE_FILES}
    for name in COMPONENT_MODULES:
        identities[name] = file_sha256(Path(importlib.import_module(name).__file__))
    return identities


def run_directory() -> Path:
    return ROOT / 'outputs/training' / RUN_ID


# ---------------------------------------------------------------- data order

def predecessor_spec() -> dict:
    return read(PREDECESSOR / 'config/job.json')['spec']


def continuation_order(prepared: Path, selection: dict, *, offset: int, count: int) -> np.ndarray:
    """Positions [offset, offset+count) of the seed-42 permutation the predecessor used for [0, offset)."""
    import torch
    train = np.load(prepared / 'train_window_ids.npy', mmap_mode='r', allow_pickle=False)
    generator = torch.Generator().manual_seed(selection['seed'])
    torch.empty((), dtype=torch.int64).random_(generator=generator)       # identical draw to the predecessor
    permutation = torch.randperm(len(train), generator=generator).numpy()
    if offset + count > len(permutation):
        raise ValueError('Requested order exceeds the eligible window population.')
    return np.asarray(train[permutation[offset:offset + count]], dtype='<u4'), \
        np.asarray(train[permutation[:offset]], dtype='<u4')


def build_order(config_dir: Path, *, max_steps: int) -> dict:
    """Write the frozen order and prove its disjointness; returns its identity record."""
    from codexa_workspace.training.foundation.selection import validate_selection
    selection = validate_selection(PREPARED)
    previous = predecessor_spec()
    consumed = previous['training']['max_steps'] * previous['training']['gradient_accumulation_steps']
    order, prefix = continuation_order(PREPARED, selection, offset=consumed, count=max_steps * 32)
    predecessor = np.load(previous['data']['train_window_order'], allow_pickle=False)
    if not np.array_equal(prefix, predecessor):
        raise ValueError('The permutation prefix does not reproduce the predecessor order.')
    validation = np.load(PREPARED / 'validation_window_ids.npy', allow_pickle=False)
    historical = np.load(PREPARED / 'historical_window_ids.npy', allow_pickle=False)
    checks = dict(unique=len(np.unique(order)) == len(order),
                  disjoint_from_predecessor=not np.intersect1d(order, predecessor).size,
                  disjoint_from_validation=not np.intersect1d(order, validation).size,
                  disjoint_from_base_v1_reconstructed=not np.intersect1d(order, historical).size)
    if not all(checks.values()):
        raise ValueError(f'Frozen order disjointness failed: {checks}')
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / 'train_window_order.npy'
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), order):
            raise ValueError('An existing frozen order differs; never overwrite it.')
    else:
        np.save(path, order, allow_pickle=False)
    return dict(path=str(path), sha256=file_sha256(path), windows=len(order), offset=consumed,
                selection_sha256=selection['selection_sha256'], seed=selection['seed'],
                derivation='positions [offset, offset+windows) of the predecessor seed-42 randperm over train_window_ids',
                predecessor_order_sha256=previous['data']['train_window_order_sha256'], checks=checks)


# ---------------------------------------------------------------- timing model

def step_time_statistics() -> dict:
    """Measured steady-state update times of the predecessor (same model, context, batch and optimizer)."""
    rows = [json.loads(line) for line in (PREDECESSOR / 'metrics/train_metrics.jsonl').read_text().splitlines()]
    times = sorted(r['step_time_seconds'] for r in rows)
    pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]
    return dict(source=str(PREDECESSOR / 'metrics/train_metrics.jsonl'), updates=len(times),
                median=pick(.5), p95=pick(.95), p99=pick(.99), maximum=times[-1],
                tokens_per_update=rows[-1]['tokens_processed'],
                peak_reserved_vram_bytes=max(r['peak_reserved_vram_bytes'] or 0 for r in rows))


def timing_model(measured: dict, *, max_steps: int, reserve_seconds: int) -> dict:
    """Wall-clock plan for ``max_steps`` using the p95 update time and measured evaluation costs."""
    steps = measured['steps']
    validations = max_steps // CADENCE_UPDATES
    periodic = len([s for s in range(CADENCE_UPDATES, max_steps, CADENCE_UPDATES)])
    full_mid = len([s for s in GENERATION_UPDATES if s < max_steps])
    checkpoints = len([s for s in range(CHECKPOINT_UPDATES, max_steps, CHECKPOINT_UPDATES)])
    items = dict(startup=measured['startup_seconds'], baseline_validation=measured['validation_seconds'],
                 optimizer_updates=max_steps * steps['p95'],
                 validations=validations * measured['validation_seconds'],
                 periodic_capability=periodic * measured['periodic_seconds'],
                 full_evaluations_mid=full_mid * measured['full_seconds'],
                 checkpoints=checkpoints * measured['checkpoint_seconds'])
    final = dict(validation=measured['validation_seconds'], save_and_verify=2 * measured['checkpoint_seconds'],
                 full_evaluation=measured['full_seconds'])
    total = sum(items.values()) + reserve_seconds
    return dict(max_steps=max_steps, tokens=max_steps * UPDATE_TOKENS, items=items, final_within_reserve=final,
                final_needed_seconds=sum(final.values()), reserve_seconds=reserve_seconds,
                planned_total_seconds=total, cap_seconds=WALL_CLOCK_SECONDS, slack_seconds=WALL_CLOCK_SECONDS - total,
                fits=total <= WALL_CLOCK_SECONDS and sum(final.values()) <= reserve_seconds,
                median_basis_tokens=max_steps * UPDATE_TOKENS,
                note='p95 update time; the SessionBudget deadline remains the hard limit whatever the cap')


def choose_max_steps(measured: dict, *, reserve_seconds: int, slack_fraction: float = .10) -> int:
    """Largest multiple of the cadence whose p95 plan leaves >= slack_fraction of the cap unused."""
    best = None
    for steps in range(CADENCE_UPDATES, 8000, CADENCE_UPDATES):
        plan = timing_model(measured, max_steps=steps, reserve_seconds=reserve_seconds)
        if plan['fits'] and plan['slack_seconds'] >= slack_fraction * WALL_CLOCK_SECONDS:
            best = steps
    if best is None:
        raise ValueError('No update cap fits the 10-hour budget with the required slack.')
    return best


# ---------------------------------------------------------------- preparation: sealed baseline and data evidence

def environment_record() -> dict:
    """The guard's development report at freeze: these component revisions must still be HEAD at launch."""
    from codexa_workspace.training.environment import TrainingGuardProvider
    report = TrainingGuardProvider('new').resolve()
    if report['context'] != 'current':
        raise ValueError(f"Development environment invalid: {report.get('errors')}")
    revisions = {name: value.get('detected') for name, value in (report.get('component_revisions') or {}).items()}
    return dict(required_context='current', selection='development', operation='new',
                interface=report['interface'], guard_sha256=report['guard_sha256'],
                architecture_sha256=(report.get('architecture') or {}).get('sha256'),
                tokenizer_sha256=report.get('tokenizer_sha256'), component_revisions=revisions,
                integration_head=git(ROOT, 'rev-parse', 'HEAD'), child_python=str(PYTHON),
                child_imports='source-first PYTHONPATH (tools/codexa-dev equivalent)')


def step_zero_baseline(output: Path) -> dict:
    """Inference-only sealed baseline of the initialization checkpoint (no optimizer, no backward pass)."""
    import torch
    from torch.utils.data import DataLoader
    from llm_data.token_data import MemmapTokenDataset
    from llm_data.token_views import IndexedTokenWindowDataset
    from llm_training.pretrain_run import SourcedWindows, collate_sourced, load_validation, weighted_validation
    from llm_training.training import evaluate, resolve_precision
    from llm_tokenizer.tokenizer import load_tokenizer
    from codexa_workspace.evaluation.capability import benchmark as bench
    from codexa_workspace.evaluation.generation import load_inference_model
    from codexa_workspace.evaluation.language_suite import generation_evaluator
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous = predecessor_spec()
    spec = dict(previous, device='cuda')
    started = time.monotonic()
    model, metadata = load_inference_model(INITIALIZATION, INITIALIZATION_SHA256, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    timings = dict(load=time.monotonic() - started)
    if metadata['requires_grad']:
        raise RuntimeError('Baseline model must be frozen.')
    # Validation CE exactly as the trainer computes it (same windows, sources, weights, precision).
    source = MemmapTokenDataset(previous['data']['corpus_path'], dtype='uint16', context_length=2048)
    windows, sources = load_validation(previous)
    loader = DataLoader(SourcedWindows(IndexedTokenWindowDataset(source, windows, vocab_size=16384), sources),
                        batch_size=1, collate_fn=collate_sourced, pin_memory=True)
    details = {}
    t0 = time.monotonic()
    with torch.inference_mode():
        evaluate(model, loader, device=torch.device('cuda'), precision=resolve_precision('bf16', torch.device('cuda')),
                 non_blocking=True, details=details)
    timings['validation'] = time.monotonic() - t0
    validation = dict(weighted_loss=weighted_validation(details, previous['data']['source_loss_weights']),
                      sources={k: v['loss'] for k, v in details['sources'].items()}, tokens=details['tokens'],
                      validation_window_ids_sha256=previous['data']['validation_window_ids_sha256'])
    # Historical 48-prompt suite (neutral + direct), as the trainer's full evaluation runs it.
    t0 = time.monotonic()
    (output / 'historical').mkdir()
    historical = generation_evaluator(spec)(model, output / 'historical', time.monotonic() + 3600)
    timings['historical_48'] = time.monotonic() - t0
    tokenizer = load_tokenizer(Path(previous['tokenizer']['path']))
    suites = bench.load_suites(('v1', 'v2'))
    with bench.inference_context(model, 'cuda'):
        t0 = time.monotonic()
        periodic_rows = bench.evaluate_items(model, tokenizer, suites['items'], forms=('plain',), device='cuda')
        timings['periodic_scoring'] = time.monotonic() - t0
        t0 = time.monotonic()
        rows = bench.evaluate_items(model, tokenizer, suites['items'], forms=('plain', 'chat'),
                                    decodes=bench.decoding_configs(('greedy',)), device='cuda')
        timings['full_scoring_and_greedy'] = time.monotonic() - t0
        t0 = time.monotonic()
        penalty = bench.evaluate_items(model, tokenizer, suites['items'], forms=('plain', 'chat'),
                                       decodes=bench.decoding_configs(('greedy_repetition_penalty_1.3',)), device='cuda')
        timings['repetition_penalty_decode'] = time.monotonic() - t0
    deterministic = [dict(id=r['id'], top1=r['top1'], margin=round(r['margin'], 6)) for r in periodic_rows] == \
        [dict(id=r['id'], top1=r['top1'], margin=round(r['margin'], 6)) for r in rows if r['form'] == 'plain']
    by_key = {(r['id'], r['form']): r for r in penalty}
    for row in rows:
        extra = by_key[(row['id'], row['form'])].get('generations', {})
        row.setdefault('generations', {}).update(extra)
    rows_path = output / 'capability-rows.jsonl'
    with rows_path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    peak = torch.cuda.max_memory_reserved()
    document = dict(format=BASELINE_FORMAT, sealed_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint=dict(path=str(INITIALIZATION), sha256=INITIALIZATION_SHA256),
                    tokenizer_sha256=previous['tokenizer']['sha256'], model=metadata['model_config'],
                    validation=validation, historical=historical,
                    capability=dict(suites=suites['identity'], suites_sha256=suites['sha256'], rows_path=str(rows_path),
                                    rows_sha256=file_sha256(rows_path), summary=bench.summarize(rows),
                                    periodic_plain_matches_full_plain=deterministic,
                                    decodes=dict(greedy=bench.decoding_configs(('greedy',))['greedy'].__dict__,
                                                 penalty=bench.decoding_configs(('greedy_repetition_penalty_1.3',))
                                                 ['greedy_repetition_penalty_1.3'].__dict__)),
                    timings_seconds=timings, peak_cuda_reserved_bytes=peak, environment=environment_record(),
                    torch=torch.__version__, optimizer_constructed=False, backward_calls=0, training_updates=0)
    document['summary_sha256'] = fingerprint(document)
    (output / 'summary.json').write_text(json.dumps(document, indent=1, default=str) + '\n')
    del model
    torch.cuda.empty_cache()
    return document


def verify_payload_and_sources(order_path: Path, output: Path) -> dict:
    """Full corpus checksum, token-range scan of every selected window, exact source accounting (metadata only)."""
    from collections import Counter
    from codexa_workspace.training.foundation.selection import index_rows, targets_before, validate_selection
    selection = validate_selection(PREPARED)
    manifest = read(selection['canonical_manifest_path'])
    started = time.monotonic()
    corpus_sha = file_sha256(selection['corpus_path'])
    checksum_seconds = time.monotonic() - started
    order = np.load(order_path, allow_pickle=False)
    corpus = np.memmap(selection['corpus_path'], dtype='<u2', mode='r')
    context = selection['context_length']
    maximum, eos, out_of_range = 0, 0, 0
    for start in range(0, len(order), 4096):
        block = np.sort(order[start:start + 4096]).astype(np.int64)
        tokens = np.stack([corpus[w * context:(w + 1) * context + 1] for w in block])
        maximum = max(maximum, int(tokens.max()))
        out_of_range += int((tokens >= manifest['tokenizer_actual_vocab_size']).sum())
        eos += int((tokens == 2).sum())
    window_count = selection['index_observations']['windows']
    mask = np.zeros(window_count, dtype=bool)
    mask[order] = True
    prefix = np.concatenate([np.zeros(1, dtype='uint32'), np.cumsum(mask, dtype='uint32')])
    totals, observations = Counter(), {}
    index = Path(selection['canonical_manifest_path']).parent / 'train_index.json'
    for row in index_rows(index, token_count=manifest['train_token_count'], expected_sha256=selection['index_sha256'],
                          observations=observations):
        totals[row['source']] += (targets_before(row['token_end'], mask, prefix, context)
                                  - targets_before(row['token_start'], mask, prefix, context))
    expected_targets = len(order) * context
    checks = dict(corpus_checksum=corpus_sha == manifest['output_checksums']['train'] == selection['corpus_sha256'],
                  tokens_in_vocabulary=out_of_range == 0, source_accounting_complete=sum(totals.values()) == expected_targets,
                  index_checksum=observations.get('sha256') == selection['index_sha256'])
    shares = {k: v / sum(totals.values()) for k, v in totals.items()}
    record = dict(passed=all(checks.values()), checks=checks, corpus_sha256=corpus_sha, checksum_seconds=checksum_seconds,
                  corpus_tokens=manifest['train_token_count'], selected_windows=len(order),
                  selected_targets=expected_targets, maximum_token_id=maximum, eos_tokens_in_selection=eos,
                  targets_by_source=dict(totals), source_token_shares=shares,
                  eligible_source_token_shares=read(PREPARED / 'pilot-source-exposure.json')['eligible_source_token_shares'],
                  packing=manifest['packing_policy'], tokenizer_sha256=manifest['tokenizer_sha256'],
                  limitations=selection['historical_exposure_limitation'] + [
                      'Base-v1 exclusion is positional and reconstructed; content-level near-duplicates are not excluded.'],
                  summary=dict(corpus_checksum=checks['corpus_checksum'], source_token_shares=shares))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(record, indent=1, default=str) + '\n')
    return record


# ---------------------------------------------------------------- specification and frozen job

def build_spec(directory: Path, *, max_steps: int, reserve_seconds: int, generation_reserve_seconds: int,
               order: dict, baseline: dict, suites: dict, environment: dict, plan: dict) -> dict:
    previous = predecessor_spec()
    training = dict(previous['training'], max_steps=max_steps, checkpoint_interval=CHECKPOINT_UPDATES,
                    evaluation_interval=CADENCE_UPDATES)
    sources = directory / 'config/validation_sources.json'
    if not sources.exists():
        shutil.copyfile(previous['data']['validation_sources'], sources)
    if file_sha256(sources) != previous['data']['validation_sources_sha256']:
        raise ValueError('Validation source map drift.')
    from codexa_workspace.training.foundation.selection import validate_selection
    selection = validate_selection(PREPARED)
    return dict(
        format=SPEC_FORMAT, run_name=RUN_ID, purpose='foundation continued pretraining', device='cuda',
        viewer_required=True, expected_parameters=previous['expected_parameters'], model=previous['model'],
        training=training,
        initialization=dict(path=str(INITIALIZATION), sha256=INITIALIZATION_SHA256,
                            policy='weights_only_fresh_optimizer_and_scheduler',
                            predecessor_run='codexa-934m-foundation-5h-v1', exact_resume=False),
        tokenizer=previous['tokenizer'],
        data=dict(corpus_path=selection['corpus_path'], dtype='uint16', selection_sha256=selection['selection_sha256'],
                  corpus_input_identity=selection['corpus_input_identity'],
                  train_window_order=order['path'], train_window_order_sha256=order['sha256'], order=order,
                  validation_window_ids=str(PREPARED / 'validation_window_ids.npy'),
                  validation_window_ids_sha256=previous['data']['validation_window_ids_sha256'],
                  validation_sources=str(sources), validation_sources_sha256=file_sha256(sources),
                  source_loss_weights=selection['source_loss_weights']),
        optimization=dict(previous['optimization']),
        evaluation=dict(cadence_updates=CADENCE_UPDATES, validation_batches=None,
                        generation_steps=[s for s in GENERATION_UPDATES if s < max_steps],
                        generation_forms=['neutral', 'direct'], generation_cases=48,
                        generation_reserve_seconds=generation_reserve_seconds, validation_overrun_seconds=0,
                        capability=dict(suites=['v1', 'v2'], suites_sha256=suites['sha256'],
                                        full_decodes=['greedy'], full_generate_forms=['plain', 'chat'],
                                        periodic='plain-form teacher-forced scoring of every frozen item',
                                        baseline_path=baseline['path'], baseline_sha256=baseline['summary_sha256'])),
        guards=dict(previous['guards'], capability_patience=2),
        checkpoint=dict(previous['checkpoint'], interval=CHECKPOINT_UPDATES),
        budget=dict(seconds=WALL_CLOCK_SECONDS, reserve_seconds=reserve_seconds, automatic_extension=False),
        environment=environment, timing_model=plan)


def freeze(directory: Path, spec: dict, *, evidence: dict) -> dict:
    job = dict(format=JOB_FORMAT, run_id=RUN_ID, spec=spec, spec_sha256=fingerprint(spec),
               code_identity=code_identity(), evidence=evidence,
               wakeup=dict(provider='claude'), no_automatic_follow_up_training=True,
               authorization='NOT AUTHORIZED: requires a separate explicit user grant naming this job hash.')
    job['job_sha256'] = fingerprint(job)
    path = Path(directory) / 'config/job.json'
    if path.exists():
        raise FileExistsError('A frozen job already exists in this preparation directory; never overwrite it.')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(job, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    return job


def verify_job(job: dict) -> dict:
    unsigned = {k: v for k, v in job.items() if k != 'job_sha256'}
    if job.get('format') != JOB_FORMAT or fingerprint(unsigned) != job['job_sha256']:
        raise LaunchRefused('Frozen job document was modified.')
    if fingerprint(job['spec']) != job['spec_sha256']:
        raise LaunchRefused('Frozen specification was modified.')
    return job


def approval_path(directory: Path) -> Path:
    return Path(directory) / 'config/approval.json'


def write_approval(directory: Path, job: dict, *, statement: str, written_utc: str) -> Path:
    """Record the user's explicit grant for one exact job hash. Never overwrites a grant."""
    verify_job(job)
    if not statement or not statement.strip():
        raise ValueError('An explicit authorization statement is required.')
    path = approval_path(directory)
    if path.exists():
        raise FileExistsError('An approval already exists; never overwrite a grant.')
    grant = dict(format=APPROVAL_FORMAT, approved=True, job_sha256=job['job_sha256'], run_id=RUN_ID,
                 statement=statement.strip(), granted_utc=written_utc,
                 scope='one 10-hour foundation continuation; no extension, resume beyond resource stops, or follow-up run')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(grant, stream, indent=1)
        stream.write('\n')
    return path


def approval_valid(directory: Path, job: dict) -> dict:
    path = approval_path(directory)
    grant = read(path) if path.exists() else {}
    if grant.get('format') != APPROVAL_FORMAT or grant.get('approved') is not True \
            or grant.get('job_sha256') != job['job_sha256'] or not str(grant.get('statement', '')).strip():
        raise LaunchRefused('A user approval naming this exact job hash is required.')
    return grant


# ---------------------------------------------------------------- readiness

def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_reproducibility(job: dict) -> dict:
    compatibility = read(ROOT / 'compatibility.json')
    pins = {name: dict(pinned=record['commit'], head=git((ROOT / record['path']).resolve(), 'rev-parse', 'HEAD'))
            for name, record in compatibility['components'].items()}
    dirty = {name: git((ROOT / record['path']).resolve(), 'status', '--porcelain', '--untracked-files=all', '--', 'src')
             for name, record in compatibility['components'].items()}
    dirty['LLM-From-Scratch'] = git(ROOT, 'status', '--porcelain', '--untracked-files=all', '--', 'src', 'scripts', 'tools')
    uncommitted = [name for name in CODE_FILES if not git(ROOT, 'ls-files', '--', name) or git(ROOT, 'status', '--porcelain', '--', name)]
    current = code_identity()
    return dict(pins=pins, pins_match=all(p['pinned'] == p['head'] for p in pins.values()),
                dirty={k: v.splitlines() for k, v in dirty.items() if v}, clean=not any(dirty.values()),
                integration_head=git(ROOT, 'rev-parse', 'HEAD'), uncommitted_code_files=uncommitted,
                code_identity_matches_job=current == job['code_identity'],
                code_identity_differences=sorted(k for k in set(current) | set(job['code_identity'])
                                                 if current.get(k) != job['code_identity'].get(k)))


def verify_data(spec: dict) -> dict:
    data = spec['data']
    stat = os.stat(data['corpus_path'])
    identity = data['corpus_input_identity']
    order = np.load(data['train_window_order'], allow_pickle=False)
    validation = np.load(data['validation_window_ids'], allow_pickle=False)
    historical = np.load(PREPARED / 'historical_window_ids.npy', allow_pickle=False)
    predecessor = np.load(predecessor_spec()['data']['train_window_order'], allow_pickle=False)
    eligible = np.load(PREPARED / 'train_window_ids.npy', mmap_mode='r', allow_pickle=False)
    # st_dev is not a stable identity on btrfs (anonymous subvolume device ids change across mounts:
    # 54 at selection, 50 after a remount, with identical inode, size, timestamps and full SHA-256).
    # Identity is inode + size + mtime + ctime here, and the full payload checksum in the evidence.
    checks = dict(
        corpus_identity=(stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        == (identity['inode'], identity['bytes'], identity['mtime_ns'], identity['ctime_ns']),
        order_sha=file_sha256(data['train_window_order']) == data['train_window_order_sha256'],
        order_length=len(order) == spec['training']['max_steps'] * spec['training']['gradient_accumulation_steps'],
        order_unique=len(np.unique(order)) == len(order),
        order_within_eligible=bool(np.isin(order, eligible).all()),
        disjoint_from_predecessor=not np.intersect1d(order, predecessor).size,
        disjoint_from_validation=not np.intersect1d(order, validation).size,
        disjoint_from_base_v1_reconstructed=not np.intersect1d(order, historical).size,
        validation_sha=file_sha256(data['validation_window_ids']) == data['validation_window_ids_sha256'],
        sources_sha=file_sha256(data['validation_sources']) == data['validation_sources_sha256'])
    return dict(passed=all(checks.values()), checks=checks,
                device=dict(current=stat.st_dev, at_selection=identity['device'], informational=True))


def evaluate_readiness(job_path: Path, *, environment=None, resources: Callable[[], dict] | None = None,
                       supervisor: Callable[[], dict] | None = None, run_tests: bool = False) -> dict:
    """Every check runs; engineering failures make NOT READY, launch-time resources CONDITIONALLY READY."""
    from codexa_workspace.training import environment, gates as resource_gates
    from codexa_workspace.evaluation.capability import benchmark as bench
    directory = Path(job_path).parent.parent
    job = read(job_path)
    spec = job['spec']
    checks = []

    def check(category, name, function, kind='engineering'):
        try:
            result = function()
            passed = bool(result if isinstance(result, bool) else (result or {}).get('passed', True))
            detail = result
        except Exception as error:
            passed, detail = False, f'{type(error).__name__}: {error}'
        checks.append(dict(category=category, name=name, kind=kind, passed=passed, detail=detail))
        return detail

    check('configuration', 'job and spec hashes', lambda: bool(verify_job(job)))
    check('configuration', 'not authorized at freeze; no launch command', lambda: job['authorization'].startswith('NOT AUTHORIZED'))
    check('configuration', 'hyperparameters equal the verified predecessor (only cap, cadence, checkpoints differ)',
          lambda: hyperparameters(spec))
    check('configuration', 'timing model fits 36,000 s with the final work inside the reserve',
          lambda: dict(passed=spec['timing_model']['fits'] and spec['budget']['seconds'] == WALL_CLOCK_SECONDS
                       and spec['budget']['automatic_extension'] is False, plan=spec['timing_model']))
    check('configuration', 'runtime supports the configuration (capability only; no optimizer constructed)', runtime_support)
    guard = check('source identity', 'guard: development, operation new',
                  lambda: (lambda r: dict(r, passed=r['context'] == 'current'))(
                      (environment or environment.TrainingGuardProvider('new')).resolve()))
    check('source identity', 'component revisions equal the frozen job',
          lambda: isinstance(guard, dict) and {k: v.get('detected') for k, v in (guard.get('component_revisions') or {}).items()}
          == spec['environment']['component_revisions'])
    check('source identity', 'repaired architecture and tokenizer identity',
          lambda: isinstance(guard, dict) and (guard.get('architecture') or {}).get('repaired') is True
          and guard.get('tokenizer_sha256') == spec['tokenizer']['sha256'])
    check('git reproducibility', 'pins, clean executed trees, committed code identity',
          lambda: (lambda g: dict(g, passed=g['pins_match'] and g['clean'] and not g['uncommitted_code_files']
                                  and g['code_identity_matches_job']))(git_reproducibility(job)))
    check('checkpoint', 'initialization checkpoint bytes (foundation ca779a16…)',
          lambda: file_sha256(spec['initialization']['path']) == spec['initialization']['sha256'] == INITIALIZATION_SHA256)
    check('tokenizer', 'tokenizer bytes', lambda: file_sha256(spec['tokenizer']['path']) == spec['tokenizer']['sha256'])
    check('dataset', 'corpus identity, frozen order and disjointness', lambda: verify_data(spec))
    check('dataset', 'payload and source verification evidence',
          lambda: (lambda e: dict(passed=e['passed'], summary=e.get('summary')))(read(directory / 'evidence/data-verification.json')))
    check('evaluation', 'frozen capability suites', lambda: bench.load_suites(tuple(spec['evaluation']['capability']['suites']))['sha256']
          == spec['evaluation']['capability']['suites_sha256'])
    check('evaluation', 'sealed step-zero baseline bound to checkpoint, suites, tokenizer and code',
          lambda: baseline_binding(spec))
    check('safety', 'canonical resource gates (never lowered)', lambda: bool(resource_gates.validate_gates(gates(spec))))
    check('safety', 'stop policy (nonfinite, OOM, headroom, gradient, source CE, capability, deadline, checkpoint)',
          lambda: stop_policy(spec))
    check('safety', 'fresh isolated run namespace', lambda: dict(passed=not run_directory().exists(), path=str(run_directory())))
    check('supervisor', 'OS supervisor, linger and read-only dashboard',
          lambda: (lambda s: dict(s, passed=s['available']))((supervisor or environment.supervisor_available)()))
    if run_tests:
        check('tests', 'focused CPU regression suites', focused_tests)
    measured = (resources or (lambda: resource_gates.measure_resources(ROOT)))()
    peak = 7 * spec['checkpoint']['estimated_checkpoint_bytes'] + 1024**3
    for item in resource_gates.resource_checks(gates(spec), measured, required_disk_bytes=peak):
        check('resources', f"launch-time {item['name']}", lambda item=item: dict(item), kind='launch-time')
    engineering = [c for c in checks if not c['passed'] and c['kind'] == 'engineering']
    launch_time = [c for c in checks if not c['passed'] and c['kind'] == 'launch-time']
    verdict = NOT_READY if engineering else (CONDITIONAL if launch_time else READY)
    try:
        granted = bool(approval_valid(directory, job))
    except LaunchRefused:
        granted = False
    return dict(verdict=verdict, job_sha256=job['job_sha256'], run_id=RUN_ID, checks=checks,
                engineering_failures=len(engineering), launch_time_conditions=len(launch_time),
                blocking=[c['name'] for c in engineering + launch_time],
                authorization=dict(granted=granted, path=str(approval_path(directory)),
                                   note='Technical readiness is never authorization.'),
                checked_utc=datetime.now(timezone.utc).isoformat())


def gates(spec: dict) -> dict:
    from codexa_workspace.training.gates import canonical_gates
    canonical = canonical_gates()
    return dict(canonical, in_run_free_vram_bytes=spec['optimization']['minimum_headroom_bytes'],
                recovery_reserve_bytes=spec['checkpoint']['reserve_bytes'])


def hyperparameters(spec: dict) -> dict:
    previous = predecessor_spec()
    t, p = spec['training'], previous['training']
    differences = sorted(k for k in set(t) | set(p) if t.get(k) != p.get(k))
    allowed = {'max_steps', 'checkpoint_interval', 'evaluation_interval'}
    optimization_same = spec['optimization'] == previous['optimization']
    expected = dict(learning_rate=2e-5, weight_decay=.1, warmup_steps=20, gradient_clip=1.0, precision='bf16',
                    micro_batch_size=1, gradient_accumulation_steps=32, seed=42)
    values_ok = all(t.get(k) == v for k, v in expected.items())
    return dict(passed=set(differences) <= allowed and optimization_same and values_ok
                and spec['optimization']['optimizer'] == 'adamw8bit' and spec['optimization']['minimum_learning_rate'] == 2e-6,
                differences=differences, optimization_same=optimization_same)


def stop_policy(spec: dict) -> dict:
    g = spec['guards']
    from codexa_workspace.training.gates import canonical_gates
    ok = (g['gradient_norm_threshold'] == 10.0 and g['gradient_norm_patience'] == 3
          and g['source_validation_CE_increase'] == 0.1 and g['validation_patience'] == 2
          and g['capability_patience'] == 2
          and spec['optimization']['minimum_headroom_bytes'] >= canonical_gates()['in_run_free_vram_bytes']
          and spec['checkpoint']['minimum_save_seconds'] >= 120)
    return dict(passed=ok, guards=g)


def runtime_support() -> dict:
    import inspect
    import torch
    import bitsandbytes.optim
    from llm_training import pretrain_run
    parameters = set(inspect.signature(pretrain_run.run_pretraining).parameters)
    result = dict(periodic_evaluator='periodic_evaluator' in parameters, generation_evaluator='generation_evaluator' in parameters,
                  adamw8bit=hasattr(bitsandbytes.optim, 'AdamW8bit'), cuda=torch.cuda.is_available(),
                  bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
    result['passed'] = all(result.values())
    return result


def baseline_binding(spec: dict) -> dict:
    from codexa_workspace.evaluation.capability import benchmark as bench
    capability = spec['evaluation']['capability']
    document = bench.load_baseline(Path(capability['baseline_path']), capability['baseline_sha256'])
    checks = dict(checkpoint=document['checkpoint']['sha256'] == spec['initialization']['sha256'],
                  tokenizer=document['tokenizer_sha256'] == spec['tokenizer']['sha256'],
                  suites=document['capability']['suites_sha256'] == capability['suites_sha256'],
                  validation_windows=document['validation']['validation_window_ids_sha256'] == spec['data']['validation_window_ids_sha256'],
                  no_training=document['optimizer_constructed'] is False and document['backward_calls'] == 0,
                  components=document['environment']['component_revisions'] == spec['environment']['component_revisions'])
    return dict(passed=all(checks.values()), checks=checks, summary_sha256=document['summary_sha256'])


def focused_tests() -> dict:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    suites = (('LLM-Training', ['tests/test_pretrain_run.py', 'tests/test_telemetry.py']),
              ('LLM-From-Scratch', ['tests/training/test_foundation_continuation.py',
                                    'tests/evaluation/test_capability_benchmark.py',
                                    'tests/training/test_supervisor.py', 'tests/notifications/test_reporting.py']))
    results = {}
    for repository, paths in suites:
        done = subprocess.run([str(PYTHON), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *paths],
                              cwd=WORKSPACE / repository, env=environment, capture_output=True, text=True, timeout=1800)
        results[repository] = dict(exit_code=done.returncode, summary=(done.stdout.strip().splitlines() or [''])[-1])
    return dict(passed=all(r['exit_code'] == 0 for r in results.values()), results=results)


# ---------------------------------------------------------------- authorized launch and resume

def child_environment() -> dict:
    from codexa_workspace.training.environment import source_first_pythonpath
    return dict(PYTHONPATH=source_first_pythonpath(), CODEXA_TRAINING_ENVIRONMENT='development',
                LLM_INTEGRATION_ROOT=str(ROOT),
                LLM_TRAINING_VIEWER_COMMAND=json.dumps([str(PYTHON), str(SCRIPTS / 'training_dashboard.py'), '--attach']))


def build_launch(run_dir: Path, job: dict, *, provider: str, resume: bool, written_utc: str) -> dict:
    if provider not in ('codex', 'claude'):
        raise ValueError('provider must be codex or claude')
    command = [str(PYTHON), '-m', 'codexa_workspace.training.foundation.continuation', 'train', '--run-dir', str(run_dir)]
    if resume:
        command.append('--resume')
    return dict(run_id=RUN_ID, cwd=str(ROOT), command=command, environment=child_environment(),
                wakeup=dict(provider=provider), deadline_seconds=WALL_CLOCK_SECONDS, agent_window=True,
                execution_class='production', policy=dict(deadline_grace=120), job_sha256=job['job_sha256'],
                written_utc=written_utc)


def _start_unit(run_dir: Path, launch: dict, *, unit: str, launcher: Callable | None) -> list:
    from codexa_workspace.training.supervisor import state as training_supervisor, systemd as supervisor_systemd
    from codexa_workspace.training.environment import SESSION_ENVIRONMENT, source_first_pythonpath
    training_supervisor.atomic_document(run_dir / 'supervisor/launch.json', launch)
    environment = {key: os.environ[key] for key in SESSION_ENVIRONMENT if os.environ.get(key)}
    environment.update(PYTHONPATH=source_first_pythonpath(), CODEXA_TRAINING_ENVIRONMENT='development')
    start = launcher or supervisor_systemd.systemd_launch
    command = start(run_dir, ROOT, unit=unit, python=str(PYTHON), script=SUPERVISOR_SCRIPT, environment=environment)
    training_supervisor.RunState(run_dir).event('systemd_launch', unit=unit)
    return command


def launch_authorized(directory: Path, job: dict, *, provider: str, readiness: Callable[[], dict],
                      written_utc: str, launcher: Callable | None = None) -> dict:
    """The single launch path. Every check runs first; any failure launches nothing."""
    if os.environ.get('CODEXA_WAKEUP_AGENT'):
        raise LaunchRefused('A wakeup analysis agent may not launch training.')
    verify_job(job)
    approval = approval_valid(directory, job)
    verdict = readiness()
    if verdict.get('verdict') != READY:
        raise LaunchRefused(f"Readiness is {verdict.get('verdict')!r}, not {READY!r}: {verdict.get('blocking')}")
    run_dir = run_directory()
    if run_dir.exists():
        raise LaunchRefused(f'Run namespace {run_dir} already exists; never overwrite or reuse a run.')
    from codexa_workspace.training.supervisor import state as training_supervisor, systemd as supervisor_systemd
    (run_dir / 'config').mkdir(parents=True)
    training_supervisor.atomic_document(run_dir / 'config/job.json', job)
    training_supervisor.atomic_document(run_dir / 'config/approval.json', approval)
    training_supervisor.atomic_document(run_dir / 'config/launch-readiness.json', verdict)
    state = training_supervisor.RunState(run_dir)
    state.transition('VALIDATING')
    state.transition('READY', launch_gates=dict(readiness=verdict['verdict']))
    launch = build_launch(run_dir, job, provider=provider, resume=False, written_utc=written_utc)
    command = _start_unit(run_dir, launch, unit=f'codexa-train-{RUN_ID}', launcher=launcher)
    return dict(launched=True, run_dir=str(run_dir), command=command, launch=launch)


def resume_authorized(job: dict, *, confirm: str, provider: str, resources: Callable[[], dict], written_utc: str,
                      launcher: Callable | None = None) -> dict:
    """Exact resume after a resource stop or interruption only; the original deadline is kept."""
    from codexa_workspace.training.supervisor import state as training_supervisor, systemd as supervisor_systemd
    from codexa_workspace.training.gates import resource_checks
    if os.environ.get('CODEXA_WAKEUP_AGENT'):
        raise LaunchRefused('A wakeup analysis agent may not resume training.')
    run_dir = run_directory()
    if confirm != RUN_ID:
        raise LaunchRefused('--confirm-run must repeat the exact run id.')
    recorded = verify_job(read(run_dir / 'config/job.json'))
    if recorded['job_sha256'] != job['job_sha256'] or code_identity() != recorded['code_identity']:
        raise LaunchRefused('Unsafe resume: job or code identity differs from the interrupted run.')
    approval_valid(run_dir, recorded)
    state = training_supervisor.read_json(run_dir / 'supervisor/state.json') or {}
    reason = (training_supervisor.read_json(run_dir / 'supervisor/completion.json') or {}).get('reason')
    if not (state.get('state') == 'INTERRUPTED' or (state.get('state') == 'EARLY_STOPPED' and reason in RESUMABLE_STOPS)):
        raise LaunchRefused(f"Not resumable: {state.get('state')} ({reason}). Only interruptions and resource stops resume.")
    latest = training_supervisor.recovery_role(run_dir, 'latest')
    if not (latest and latest['verified']):
        raise LaunchRefused('Unsafe resume: no independently verified latest checkpoint.')
    spec = recorded['spec']
    failed = [c for c in resource_checks(gates(spec), resources(), required_disk_bytes=7 * spec['checkpoint']['estimated_checkpoint_bytes'])
              if not c['passed']]
    if failed:
        raise LaunchRefused(f'Resource gates failed: {failed}')
    budget = training_supervisor.read_json(run_dir / 'metrics/budget.json')
    remaining = min(budget['deadline_utc'] - time.time(), budget['remaining_seconds'] - (time.time() - budget['last_recorded_utc']))
    if remaining <= budget['reserve_seconds'] + 900:
        raise LaunchRefused(f'The original 10-hour budget is nearly spent ({remaining:.0f} s left); no resume.')
    segment = run_dir / 'supervisor/segments' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    segment.mkdir(parents=True)
    for name in ('state.json', 'completion.json', 'notification-state.json', 'desktop-notification.json', 'ai-wakeup.json',
                 'agent-window.json', 'ai-handoff.md', 'launch.json', 'run-summary.json'):
        if (run_dir / 'supervisor' / name).exists():
            shutil.move(str(run_dir / 'supervisor' / name), segment / name)
    if (run_dir / 'final-report').exists():
        shutil.move(str(run_dir / 'final-report'), segment / 'final-report')
    run_state = training_supervisor.RunState(run_dir)
    run_state.event('manual_resume', segment=str(segment), checkpoint_step=latest['optimizer_step'],
                    previous_state=state.get('state'), previous_reason=reason)
    run_state.update(state='READY', resumed_from=latest, previous_segment=str(segment))
    launch = build_launch(run_dir, recorded, provider=provider, resume=True, written_utc=written_utc)
    command = _start_unit(run_dir, launch, unit=f'codexa-train-{RUN_ID}-resume-{int(time.time())}', launcher=launcher)
    return dict(resumed=True, run_dir=str(run_dir), from_step=latest['optimizer_step'], command=command)


# ---------------------------------------------------------------- trainer entry (supervisor child only)

def verify_run(run_dir: Path) -> dict:
    job = verify_job(read(Path(run_dir) / 'config/job.json'))
    if code_identity() != job['code_identity']:
        raise LaunchRefused('Code identity changed after the job was frozen.')
    approval_valid(Path(run_dir), job)
    return job


def train(run_dir: Path, *, resume: bool) -> dict:  # pragma: no cover - authorized runs only
    from codexa_workspace.training.environment import training_guard
    training_guard().require_environment('new')
    from llm_training.pretrain_run import run_pretraining
    from llm_training.session import SessionBudget
    from codexa_workspace.evaluation.capability import benchmark as bench
    job = verify_run(run_dir)
    spec = job['spec']
    capability = spec['evaluation']['capability']
    baseline = bench.load_baseline(Path(capability['baseline_path']), capability['baseline_sha256'])
    if resume:
        budget = SessionBudget.resume(read(Path(run_dir) / 'metrics/budget.json'))
    else:
        budget = SessionBudget.start(spec['budget']['seconds'], spec['budget']['reserve_seconds'])
    return run_pretraining(spec, Path(run_dir), budget, resume=resume,
                           generation_evaluator=bench.full_evaluator(spec, baseline),
                           periodic_evaluator=bench.periodic_evaluator(spec, baseline, Path(run_dir) / 'evaluations'))


def main(arguments: list[str] | None = None) -> int:  # pragma: no cover - supervisor entry
    parser = argparse.ArgumentParser(description='Supervisor child entry for the approved 10-hour continuation.')
    parser.add_argument('command', choices=('train',))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(arguments)
    report = train(args.run_dir, resume=args.resume)
    return 0 if report['state'] in ('COMPLETED', 'EARLY_STOPPED', 'INTERRUPTED') else 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
