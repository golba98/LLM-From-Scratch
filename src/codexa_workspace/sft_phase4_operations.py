"""Phase 4 operations: sealed step-zero inference, prepare/preflight checks, budget, preservation.

Everything here runs under ``no_training_guard`` (optimizer construction, backward and
torch.save raise). No train mode is reachable from this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from codexa_workspace import sft_phase4_data as data
from codexa_workspace import sft_pilot_runtime as runtime

GIB = 1024**3
STEP_ZERO_MINIMUM_FREE_VRAM_MIB = 8 * 1024  # inference-only bound; training gates are separate and unchanged


def _manifest(directory: Path) -> dict:
    return data.read(Path(directory) / 'manifest/phase4-manifest.json')


def _training_units_active() -> list[str]:
    done = subprocess.run(['systemctl', '--user', 'list-units', '--type=service', '--state=running', '--no-legend'],
                          capture_output=True, text=True, timeout=30)
    return [line.split()[0] for line in done.stdout.splitlines()
            if any(word in line.lower() for word in ('codexa', 'train', 'supervisor'))]


def _versions() -> dict:
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def step_zero(directory: Path) -> dict:
    """Seal the step-zero baseline with the exact evaluation code the pilot would use."""
    import torch
    from llm_data.chat_dataset import IndexedChatDataset
    from llm_tokenizer.tokenizer import load_tokenizer
    from codexa_workspace import sft_pilot_evaluation as evaluation
    directory = Path(directory)
    target = directory / 'evaluation/step-zero'
    if (target / 'summary.json').exists():
        raise FileExistsError('Step zero is sealed; never overwrite it.')
    target.mkdir(parents=True, exist_ok=True)
    verified = data.verify_manifest(directory / 'manifest/phase4-manifest.json')
    manifest = _manifest(directory)
    environment = runtime.IdentityScriptProvider().resolve()
    if environment['context'] != 'current':
        raise runtime.LaunchRefused(f"Step zero must run in the current source-first context, got {environment['context']!r}.")
    units = _training_units_active()
    if units:
        raise RuntimeError(f'Training/supervisor units are active; GPU is not available for evaluation: {units}')
    measured = runtime.measure_resources(directory)
    if (measured['vram'] or {}).get('free_mib', 0) < STEP_ZERO_MINIMUM_FREE_VRAM_MIB:
        raise RuntimeError(f'Insufficient free VRAM for bounded inference: {measured["vram"]}')
    checkpoint_before = data.digest(runtime.INITIALIZATION)
    if checkpoint_before != runtime.INITIALIZATION_SHA256:
        raise ValueError('Initialization checkpoint hash mismatch.')
    job = data.read(data.FOUNDATION / 'config/job.json')['spec']
    tokenizer = load_tokenizer(manifest['tokenizer']['path'])
    validation = IndexedChatDataset(Path(manifest['datasets']['validation']['path']), tokenizer, context=512)
    timings = {}
    started = time.monotonic()
    torch.manual_seed(42)
    with runtime.no_training_guard():
        model, metadata = evaluation.load_inference_model(runtime.INITIALIZATION, runtime.INITIALIZATION_SHA256)
        with runtime.no_training_guard(model):
            timings['load'] = time.monotonic() - started
            torch.cuda.reset_peak_memory_stats()
            decodes = evaluation.decoding_configs()
            t = time.monotonic()
            conversation = evaluation.generate_suite(model, tokenizer, evaluation.conversational_cases(),
                                                     suite='conversation', decodes=decodes, device='cuda')
            timings['conversation_generation'] = time.monotonic() - t
            t = time.monotonic()
            historical = evaluation.generate_suite(model, tokenizer, evaluation.historical_cases(),
                                                   suite='historical', decodes=decodes, device='cuda')
            timings['historical_generation'] = time.monotonic() - t
            t = time.monotonic()
            gold, per_record = evaluation.gold_validation(model, validation, device='cuda')
            timings['gold_validation'] = time.monotonic() - t
            t = time.monotonic()
            reference = evaluation.reference_validation(model, job['data'], device='cuda')
            timings['reference_validation'] = time.monotonic() - t
            t = time.monotonic()
            repeat = evaluation.generate_suite(model, tokenizer, evaluation.conversational_cases(), suite='conversation',
                                               decodes={'greedy': decodes['greedy']}, device='cuda')
            timings['determinism_repeat_greedy16'] = time.monotonic() - t
            finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
    timings['total'] = time.monotonic() - started
    del model
    torch.cuda.empty_cache()
    rows = conversation + historical
    greedy = {r['id']: r for r in conversation if r['decode'] == 'greedy'}
    determinism = dict(cases=len(repeat), identical_tokens=sum(r['raw_token_ids'] == greedy[r['id']]['raw_token_ids'] for r in repeat))
    prior = {(r['suite'], r['id'], r['decode']): r for r in data.jsonlines(evaluation.HISTORICAL_GENERATIONS)
             if r['decode'] in ('greedy', 'nucleus')}
    comparison = dict(
        greedy_identical_tokens=sum(r['raw_token_ids'] == prior[(r['suite'], r['id'], r['decode'])]['raw_token_ids']
                                    for r in rows if r['decode'] == 'greedy'),
        greedy_cases=sum(r['decode'] == 'greedy' for r in rows),
        nucleus_identical_tokens=sum(r['raw_token_ids'] == prior[(r['suite'], r['id'], r['decode'])]['raw_token_ids']
                                     for r in rows if r['decode'] == 'nucleus'),
        note='Historical diagnostic used the historical .venv wheels; this run uses the current source-first '
             'context. Differences are attributed to environment, not to the checkpoint.')
    checkpoint_after = data.digest(runtime.INITIALIZATION)
    summary = dict(format='codexa-sft-pilot-step-zero-v1', sealed_utc=datetime.now(timezone.utc).isoformat(),
                   checkpoint=dict(path=str(runtime.INITIALIZATION), sha256_before=checkpoint_before,
                                   sha256_after=checkpoint_after, unchanged=checkpoint_before == checkpoint_after),
                   manifest_sha256=verified['manifest_sha256'], validation_sha256=verified['validation_sha256'],
                   suites=evaluation.suite_identity(), generation=evaluation.summarize_generations(rows),
                   gold=gold, reference=reference, finite=finite, determinism=determinism,
                   historical_comparison=comparison, model=metadata, peak_cuda_memory_bytes=peak,
                   timings_seconds=timings, environment=environment, versions=_versions(),
                   numerical=dict(weights='fp32', autocast='bf16', tf32=False, kv_cache=True,
                                  memory_fraction=.5),
                   optimizer_constructed=False, backward_calls=0, training_updates=0)
    summary['summary_sha256'] = data.canonical_sha256({k: v for k, v in summary.items() if k != 'sealed_utc'})
    data.write_rows(target / 'generations.jsonl', rows)
    data.write_rows(target / 'determinism-repeat.jsonl', repeat)
    data.write_rows(target / 'gold-per-record.jsonl', per_record)
    data.write(target / 'summary.json', summary)
    if not summary['checkpoint']['unchanged']:
        raise RuntimeError('Checkpoint hash changed during evaluation.')
    return {k: summary[k] for k in ('generation', 'gold', 'reference', 'determinism', 'historical_comparison',
                                    'timings_seconds', 'peak_cuda_memory_bytes', 'summary_sha256')}


def job(directory: Path) -> dict:
    directory = Path(directory)
    if (directory / 'config/job.json').exists():
        raise FileExistsError('The Phase 4 job is frozen; never overwrite it (Step 5A freezes job v2 separately).')
    manifest = _manifest(directory)
    step = data.read(directory / 'evaluation/step-zero/summary.json')
    with runtime.no_training_guard():
        frozen = runtime.build_job(directory, manifest, step_zero=step)
    data.write(directory / 'config/job.json', frozen)
    return dict(job_sha256=frozen['job_sha256'], spec_sha256=frozen['spec_sha256'], runnable=frozen['spec']['runnable'])


def prepare(directory: Path) -> dict:
    directory = Path(directory)
    frozen = data.read(directory / 'config/job.json')
    with runtime.no_training_guard():
        result = runtime.prepare(frozen, measured=runtime.measure_resources(directory),
                                 environment=runtime.IdentityScriptProvider())
    guard = runtime.TrainingGuardProvider('new').resolve()
    result['checks'].append(dict(name='Codex training guard (development, operation=new)',
                                 passed=guard['context'] == 'current', detail=guard))
    result['passed'] = all(c['passed'] for c in result['checks'])
    data.write(directory / 'evidence/prepare-checks.json', result)
    return dict(passed=result['passed'], failed=[c['name'] for c in result['checks'] if not c['passed']])


def preflight(directory: Path) -> dict:
    measured = runtime.measure_resources(Path(directory))
    checks = runtime.resource_checks(runtime.canonical_gates(), measured, required_disk_bytes=2 * 5636391032)
    result = dict(measured=measured, gates=runtime.canonical_gates(), checks=checks,
                  note='Snapshot only; a future launch must re-run every gate at launch time.')
    data.write(Path(directory) / 'evidence/resource-preflight.json', result)
    return result


def budget(directory: Path) -> dict:
    """Measured inference costs plus explicitly labeled historical estimates; no optimizer was run."""
    directory = Path(directory)
    if (directory / 'config/budget-recommendation.json').exists():
        raise FileExistsError('The Phase 4 budget recommendation is frozen; Step 5A writes budget-analysis.json.')
    step = data.read(directory / 'evaluation/step-zero/summary.json')
    manifest = _manifest(directory)
    timings = step['timings_seconds']
    round_full = (timings['conversation_generation'] + timings['historical_generation'] +
                  timings['gold_validation'] + timings['reference_validation'])
    round_reduced = (timings['conversation_generation'] / 2 + timings['gold_validation'] +
                     timings['reference_validation'])  # greedy-16 only (nucleus half of the suite timing is an estimate)
    tokens = manifest['token_accounting']['train']['total']['serialized_tokens']
    per_update_tokens = tokens * 2 / 128
    estimates = dict(
        startup_seconds=dict(value=35, basis='historical weights-only starts 24-34 s plus measured load '
                             f"{timings['load']:.1f} s", kind='estimate'),
        optimizer_seconds=dict(value=round(128 * 1.33, 1), basis='historical 2048-context chat pilots, ga 8: '
                               f'<= 1.33 s/update median; pilot averages {per_update_tokens:.0f} tokens/update '
                               '(far fewer than the ~8k historical), so this is a conservative upper estimate',
                               kind='estimate'),
        checkpoint_seconds=dict(value=2 * 25, basis='historical legacy saves 20-29 s each (64 and final)', kind='estimate'),
        evaluation_round_full_seconds=dict(value=round(round_full, 1), basis='measured step-zero (all suites)', kind='measured'),
        evaluation_round_reduced_seconds=dict(value=round(round_reduced, 1),
            basis='measured gold+reference; greedy-16 taken as half the measured greedy+nucleus conversation time', kind='derived'),
        finalization_reserve_seconds=dict(value=180, basis='proposal', kind='fixed'))
    scenarios = {}
    for name, rounds in (('proposal_full_at_32_64_128', [round_full] * 3),
                         ('reduced_at_32_64_full_at_128', [round_reduced, round_reduced, round_full])):
        work = (estimates['startup_seconds']['value'] + estimates['optimizer_seconds']['value'] +
                estimates['checkpoint_seconds']['value'] + sum(rounds))
        total = work + 180
        scenarios[name] = dict(work_seconds=round(work, 1), total_with_reserve_seconds=round(total, 1),
                               fits_900=total <= 900, fits_1200=total <= 1200)
    recommendation = dict(format='codexa-sft-pilot-budget-recommendation-v1', frozen_budget=runtime.PROPOSED_BUDGET,
        estimates=estimates, scenarios=scenarios,
        memory=dict(inference_peak=dict(value=step['peak_cuda_memory_bytes'], kind='measured'),
                    training_512_context_peak=dict(value=13226737664, kind='estimated upper bound',
                        basis='historical 2048-context foundation trainer peak reserved 12.32 GiB; 512-context activations are smaller'),
                    optimizer_state=dict(kind='unverified', basis='no optimizer may be constructed in Phase 4')),
        recommendation=None, decision_required=True,
        note='The frozen specification keeps 900 s. Any revised cap is a user decision at authorization; '
             'no gate or acceptance criterion is weakened.')
    if not scenarios['proposal_full_at_32_64_128']['fits_900']:
        recommendation['recommendation'] = (
            'Authorize a 1,200 s cap (matching run_chat\'s existing 128-step / 20-minute preparation cap) '
            'with the step-zero baseline sealed in Phase 4, or keep 900 s only with nucleus/historical generation '
            'deferred to the final evaluation (all mandatory gates still evaluated at 32/64/128) if that scenario fits.')
    data.write(directory / 'config/budget-recommendation.json', recommendation)
    return recommendation


def preservation(directory: Path) -> dict:
    """Rehash Codex's 88 protected artifacts plus the checkpoint, tokenizer and source splits."""
    prior = data.read(data.PRIOR / 'evidence/environment.json')['protected']
    checks = [dict(path=path, expected=identity['sha256'], unchanged=data.digest(path) == identity['sha256'])
              for path, identity in sorted(prior.items())]
    result = dict(artifacts=len(checks), unchanged=sum(c['unchanged'] for c in checks),
                  all_unchanged=all(c['unchanged'] for c in checks),
                  checkpoint_sha256=data.digest(runtime.INITIALIZATION), checks=checks,
                  checked_utc=datetime.now(timezone.utc).isoformat())
    data.write(Path(directory) / 'evidence/preservation-check.json', result)
    return {k: v for k, v in result.items() if k != 'checks'}


def report(directory: Path) -> dict:
    from codexa_workspace.sft_phase4_report import write_report
    return write_report(Path(directory))
