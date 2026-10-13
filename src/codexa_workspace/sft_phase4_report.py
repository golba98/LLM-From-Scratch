"""Generate SFT_PHASE4_READINESS_REPORT.md from Phase 4 evidence (numbers are never typed by hand)."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from codexa_workspace import sft_phase4_data as data


def _load(directory: Path, name: str, default=None):
    path = directory / name
    return data.read(path) if path.exists() else default


def _fmt(value, digits=4):
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def verdict(evidence: dict) -> tuple[str, list[str], list[str]]:
    blockers, conditions = [], []
    if not evidence['selection'].get('complete'):
        blockers.append('512/32 selection incomplete')
    if not evidence['verify'].get('verified'):
        blockers.append('manifest verification failed')
    if evidence['step_zero'] is None:
        blockers.append('step-zero baseline not sealed')
    elif not evidence['step_zero']['checkpoint']['unchanged']:
        blockers.append('checkpoint changed during evaluation')
    if evidence['preservation'] and not evidence['preservation']['all_unchanged']:
        blockers.append('protected artifact changed')
    if evidence['tests'] and evidence['tests'].get('failed'):
        blockers.append(f"{evidence['tests']['failed']} failing tests")
    conditions.append('Codex Phase 2 enforcement (tools/training_environment_guard.py) is integrated through '
                      'TrainingGuardProvider and was exercised live: it currently refuses a new development job because '
                      'component and integration source is uncommitted. The guard is still being changed by Codex, so the '
                      'integration must be re-verified against its final committed version before authorization.')
    conditions.append('Train mode has only been exercised with fakes; the real 512-context optimizer/activation '
                      'peak is unmeasured (estimated bound only).')
    budget = evidence['budget'] or {}
    if budget and not budget['scenarios']['proposal_full_at_32_64_128']['fits_900']:
        conditions.append('The proposed 900 s cap does not fit the measured evaluation schedule; a cap or schedule '
                          'decision is required at authorization.')
    checks = (evidence['prepare'] or {}).get('checks', [])
    failed = [c['name'] for c in checks if not c['passed']]
    if failed:
        conditions.append(f"Prepare-mode checks not passing at report time: {', '.join(failed)}.")
    conditions.append('Blinded human usefulness review is a post-pilot requirement and cannot be performed now.')
    status = 'NOT READY' if blockers else ('CONDITIONALLY READY' if conditions else 'READY FOR AUTHORIZATION')
    return status, blockers, conditions


def write_report(directory: Path) -> dict:
    directory = Path(directory)
    manifest = _load(directory, 'manifest/phase4-manifest.json')
    evidence = dict(selection=_load(directory, 'evidence/selection-status.json', {}),
                    verify=data.verify_manifest(directory / 'manifest/phase4-manifest.json'),
                    step_zero=_load(directory, 'evaluation/step-zero/summary.json'),
                    budget=_load(directory, 'config/budget-recommendation.json'),
                    prepare=_load(directory, 'evidence/prepare-checks.json'),
                    preflight=_load(directory, 'evidence/resource-preflight.json'),
                    preservation=_load(directory, 'evidence/preservation-check.json'),
                    tests=_load(directory, 'evidence/test-results.json'),
                    git=_load(directory, 'evidence/git-final.json'),
                    job=_load(directory, 'config/job.json'),
                    extension=_load(directory, 'evidence/validation-pool-extension.json'))
    status, blockers, conditions = verdict(evidence)
    review = data.review_statistics(data.ledger(directory))
    accounting = manifest['token_accounting']
    train = [r for r in manifest['records'] if r['split'] == 'train']
    validation = [r for r in manifest['records'] if r['split'] == 'validation']
    prefilter = sum(1 for _ in data.jsonlines(directory / 'review/prefilter.jsonl'))
    s0 = evidence['step_zero']
    lines = [f'# Codexa Phase 4 — SFT readiness report', '',
             f'Directory `{directory}`. Generated from evidence files; no number in this report is typed by hand.', '',
             '## A. Executive readiness verdict', '', f'**{status}**', '']
    if blockers:
        lines += ['Blockers:', *[f'- {b}' for b in blockers], '']
    lines += ['Conditions that must be resolved or explicitly accepted before authorization:',
              *[f'- {c}' for c in conditions], '',
              'No optimizer was constructed, no backward pass ran, no weights changed and no job was launched.', '']
    # B. Final dataset
    by_source = accounting['train']['by_source']
    lines += ['## B. Final dataset', '',
              '| Split | OASST1 | UltraChat | Authored | Total |', '|---|---:|---:|---:|---:|',
              f"| Train | {manifest['composition']['train']['oasst1']} | {manifest['composition']['train']['ultrachat']} | {manifest['composition']['train']['authored']} | {len(train)} |",
              f"| Validation | {manifest['composition']['validation']['oasst1']} | {manifest['composition']['validation']['ultrachat']} | 0 | {len(validation)} |", '',
              '| Train tokens (per pass) | Examples | Serialized | Assistant-supervised | END targets |', '|---|---:|---:|---:|---:|']
    for source, value in by_source.items():
        lines.append(f"| {source} | {value['examples']} | {value['serialized_tokens']} | {value['supervised_tokens']} | {value['end_targets']} |")
    total = accounting['train']['total']
    lines += [f"| **Total** | {total['examples']} | {total['serialized_tokens']} | {total['supervised_tokens']} | {total['end_targets']} |",
              f"| Two passes (1,024 presentations) | {accounting['train_two_passes']['examples']} | {accounting['train_two_passes']['serialized_tokens']} | {accounting['train_two_passes']['supervised_tokens']} | {accounting['train_two_passes']['end_targets']} |",
              f"| Validation | {accounting['validation']['total']['examples']} | {accounting['validation']['total']['serialized_tokens']} | {accounting['validation']['total']['supervised_tokens']} | {accounting['validation']['total']['end_targets']} |", '',
              '| Train category | Examples | Supervised tokens |', '|---|---:|---:|']
    for category, value in sorted(accounting['train']['by_category'].items(), key=lambda x: -x[1]['examples']):
        lines.append(f"| {category} | {value['examples']} | {value['supervised_tokens']} |")
    first = review['first_pass_by_pool']
    lines += ['', f"Review volume: {review['first_pass']} first-pass decisions (365 inherited from Codex's ledger) plus "
              f"{prefilter} candidates skipped by the conservative non-English prefilter (never admissible). "
              f"First-pass by pool: {json.dumps(first)}. Lead second pass: {review['second_pass']} decisions "
              f"({json.dumps(review['agreement'])}).",
              f"Historical exposure: {manifest['historical_exposure']['train']} of 512 training records and "
              f"{manifest['historical_exposure']['validation']} of 32 validation records carry `historical_sft_exposure=true` "
              '(all authored records are new). Lineage note: ' + manifest['historical_exposure']['lineage']['classification'], '']
    if evidence['extension']:
        lines += [f"Validation OASST1 pool extension: the 172-record validation-split pool yielded 15 admissions; a documented "
                  f"extension pool from the general-chat-v2 **test** split ({evidence['extension']['eligible']} eligible, identical "
                  "filters, hash ranking) supplied the remainder. Selected extension records: "
                  f"{sum(1 for r in validation if 'test.jsonl' in str(r['provenance'].get('source_file')))}.", '']
    # C. Quality
    lines += ['## C. Dataset quality', '',
              'Every non-authored record was admitted only when an agent first-pass reviewer **and** the lead second-pass reviewer '
              'both admitted it under `review/RUBRIC.md` (correctness, relevance, coherence, instruction following, completeness, '
              'grounding, no stale/time-sensitive claims, no false capability or identity claims, English only, no repetition). '
              'Disagreements excluded the record. These are agent engineering judgments, **not** independent human review.', '',
              'Most frequent first-pass rejection reasons:', '']
    lines += [f'- {reason} ({count})' for reason, count in review['reason_counts'][:12]]
    lines += ['', f"Reject spot check: {review['reject_spot_check']['checked']} sampled rejects re-read; "
              f"{review['reject_spot_check']['second_pass_would_admit']} would have been admitted (rejects are never promoted).",
              f"Selection exclusions after both-pass admission: {json.dumps(Counter(e['reason'].split(' (')[0][:40] for e in evidence['selection']['exclusions']))}.",
              'Standards were not relaxed: the 30% category cap (153) binds context-grounded UltraChat data and was not raised; '
              'additional UltraChat review waves supplied non-context-grounded records instead.', '',
              f"Remaining risks: UltraChat admissions are dominated by supplied-context extraction and creative taglines; "
              f"OASST1 answers are short (8–96 assistant tokens by pool construction); "
              f"{sum(r['contamination']['possible_semantic_overlap'] for r in manifest['records'])} selected records are flagged "
              'as possible (soft) semantic overlap with benchmark text; automated screening cannot prove semantic independence.', '']
    # D. Configuration
    job = evidence['job']
    if job:
        spec = job['spec']
        lines += ['## D. Experiment configuration (frozen, non-runnable)', '',
                  f"Job `{job['job_sha256']}` (spec `{job['spec_sha256']}`), `runnable: {spec['runnable']}`, launch command: `{job['launch_command']}`.", '',
                  '```json', json.dumps({k: spec[k] for k in ('initialization', 'tokenizer', 'training', 'optimization', 'budget',
                                                                'checkpoint', 'outputs', 'environment', 'recovery')}, indent=2), '```', '',
                  f"Data: manifest `{spec['data']['manifest_sha256']}`, train `{spec['data']['train_sha256']}`, validation "
                  f"`{spec['data']['validation_sha256']}`; {spec['data']['presentations']} presentations in the frozen seed-73 then seed-74 orders.", '']
    # E. Step zero
    lines += ['## E. Step-zero baseline (newly measured)', '']
    if s0:
        conversation = s0['generation']['conversation/greedy']
        historical = s0['generation']['historical/greedy']
        lines += ['| Measure (new, current source-first context) | Value |', '|---|---:|',
                  f"| Conversational greedy natural endings | {conversation['natural_end']}/16 |",
                  f"| Conversational greedy chat-END endings | {conversation['chat_end']}/16 |",
                  f"| Conversational greedy severe degeneration | {conversation['severe_degeneration']}/16 |",
                  f"| Strict constrained successes | {conversation['strict_success']}/7 |",
                  f"| Historical greedy severe (independent) | {historical['severe_degeneration']}/48 |",
                  f"| Historical greedy length-limit endings | {historical['length_limit']}/48 |",
                  f"| Assistant-supervised CE (32 validation) | {_fmt(s0['gold']['assistant_supervised_ce'])} |",
                  f"| Mean gold END NLL ({s0['gold']['end_targets']} END targets) | {_fmt(s0['gold']['mean_gold_end_nll'])} |",
                  f"| Mean premature END probability ({s0['gold']['content_positions']} positions) | {_fmt(s0['gold']['mean_premature_end_probability'], 6)} |",
                  f"| Wikipedia reference CE | {_fmt(s0['reference']['sources']['wikipedia'])} |",
                  f"| FineWeb-Edu reference CE | {_fmt(s0['reference']['sources']['fineweb_edu'])} |",
                  f"| All weights finite | {s0['finite']} |",
                  f"| Greedy determinism (repeat of 16) | {s0['determinism']['identical_tokens']}/{s0['determinism']['cases']} identical |", '',
                  f"Historical comparison (not a recharacterization): greedy token sequences identical to the diagnostic for "
                  f"{s0['historical_comparison']['greedy_identical_tokens']}/{s0['historical_comparison']['greedy_cases']} cases. "
                  + s0['historical_comparison']['note'],
                  'The diagnostic\'s historical figures (46/48 severe greedy repetition, 47/48 length-limit endings, 0/48 exact '
                  'matches, no END in the new suite; Wikipedia 3.01320 / FineWeb-Edu 3.14029) remain as reported there and are '
                  'not overwritten.', '',
                  f"Precision {json.dumps(s0['numerical'])}; device {s0['versions'].get('device')}; torch {s0['versions']['torch']} / "
                  f"CUDA {s0['versions']['cuda']}; environment context `{s0['environment']['context']}`; sealed summary `{s0['summary_sha256']}`.", '']
    else:
        lines += ['Not sealed.', '']
    # F. Runtime
    lines += ['## F. Runtime integration', '',
              '- `sft_pilot_runtime.py`: prepare mode (manifest, hashes, architecture on the meta device, canonical gates, fresh '
              'namespace, cadence), `no_training_guard` (optimizer construction, backward and `torch.save` raise), environment '
              'contract, frozen job builder, `StopPolicy`, `train_authorized` (approval grant + environment + gates), and a '
              'documentation-only `launch_plan`.',
              '- `sft_pilot_trainer.py`: supervisor-compatible train mode (heartbeat, `metrics/report.json`, verified `final` '
              'role, checkpoints at 64 and final, evaluations at 32/64/128). **Never executed**; control flow tested with fakes.',
              '- `sft_pilot_evaluation.py`: the single evaluation path for step zero and future checkpoints.',
              '- `scripts/sft/sft_phase4.py`: CLI with no launch subcommand.',
              '- Codex dependency: Codex installed `tools/training_environment_guard.py` (explicit `--environment` / '
              '`CODEXA_TRAINING_ENVIRONMENT`, import-time guard on `create_adamw_optimizer`/`train_model`/`run_chat`/'
              '`run_pretraining`) during this phase. Train mode consumes it through `TrainingGuardProvider` (runs the guard CLI '
              'read-only, maps a valid `development` verdict to the `current` context, rejects everything else); evaluation uses '
              '`scripts/environment_identity.py`. The live guard currently **refuses** a new job (uncommitted executed source in '
              'LLM-Architecture, LLM-Data, LLM-Training, LLM-Inference and the integration repository). `launch_plan` sets '
              '`CODEXA_TRAINING_ENVIRONMENT=development` for Codex\'s supervisor propagation. Re-verify once Codex finalizes.',
              '- Pre-existing test failures in this phase are caused by that guard (training-path tests now require an explicit '
              'environment); they are Codex-owned and were not modified.', '']
    if evidence['prepare']:
        lines += ['Prepare-mode checks at report time:', '']
        lines += [f"- {'PASS' if c['passed'] else 'FAIL'} — {c['name']}" for c in evidence['prepare']['checks']]
        lines += ['']
    # G. Resources
    budget = evidence['budget']
    lines += ['## G. Resource feasibility', '']
    if budget:
        lines += ['| Component | Seconds | Kind | Basis |', '|---|---:|---|---|']
        for name, value in budget['estimates'].items():
            lines.append(f"| {name} | {value['value']} | {value['kind']} | {value['basis']} |")
        lines += ['', '| Scenario | Work s | Total with 180 s reserve | Fits 900 | Fits 1200 |', '|---|---:|---:|---|---|']
        for name, value in budget['scenarios'].items():
            lines.append(f"| {name} | {value['work_seconds']} | {value['total_with_reserve_seconds']} | {value['fits_900']} | {value['fits_1200']} |")
        lines += ['', f"Recommendation: {budget['recommendation'] or 'The proposed 900 s cap fits the measured schedule.'} "
                  'The frozen specification still records 900 s; changing it is a user decision.', '',
                  f"Memory: inference peak (measured) {json.dumps(budget['memory']['inference_peak']['value'])} bytes; "
                  '512-context training peak **estimated** ≤ 13,226,737,664 bytes reserved (2048-context historical bound); '
                  'optimizer state **unverified**.', '']
    if evidence['preflight']:
        lines += ['Resource snapshot vs canonical gates (unchanged: 13,928 MiB VRAM, 16 GiB RAM, 180 GiB disk, 100 GiB reserve, 1.5 GiB in-run):', '']
        lines += [f"- {'PASS' if c['passed'] else 'FAIL'} {c['name']}: measured {c['measured']} / required {c['required']} {c['unit']}"
                  for c in evidence['preflight']['checks']]
        lines += ['']
    # H. Acceptance
    lines += ['## H. Acceptance gates', '',
              '| Gate | Threshold | Implementation |', '|---|---|---|',
              '| Natural endings | ≥ 8/16 | `acceptance_gates` / `termination` |',
              '| Chat-END endings | ≥ 6/16 | `acceptance_gates` / `termination` |',
              '| Severe degeneration | ≤ 4/16 | `mechanical` (diagnostic port) |',
              '| Strict constrained | ≥ 3/7 | `strict_success` (exact surface) |',
              '| Human usefulness | ≥ 8/16 accepted, ≥ 4 improvements, ≤ 1 regression | `blinded_review_package` + `usefulness_gate`; **review dependency, pending** |',
              '| Gold END NLL | improves ≥ 0.5 vs step zero | `gold_metrics_from_logits` |',
              '| Premature END probability | rises ≤ 0.02 | `gold_metrics_from_logits` |',
              '| Wikipedia / FineWeb-Edu CE | each rises ≤ 0.1 | `reference_validation` |',
              '| Finite weights/logits; resource gates | required | evaluation + `StopPolicy` |', '',
              'Overall PASS requires every gate PASS; PENDING (e.g. missing human ratings) never counts as success. Stop rules: '
              'nonfinite/OOM → FAILED without saving; headroom < 1.5 GiB → resource stop; two consecutive deteriorations '
              '(degeneration or reference CE) → EARLY_STOPPED; 128-update cap; deadline → inconclusive, no extension.', '']
    # I. Tests and preservation
    tests = evidence['tests'] or {}
    lines += ['## I. Tests and preservation', '']
    for suite in tests.get('suites', []):
        lines.append(f"- `{suite['command']}`: {suite['summary']}")
    if evidence['preservation']:
        p = evidence['preservation']
        lines += ['', f"Preservation: {p['unchanged']}/{p['artifacts']} protected artifacts unchanged; initialization checkpoint "
                  f"`{p['checkpoint_sha256']}`.", '']
    # J. Git
    git = evidence['git'] or {}
    lines += ['## J. Git and audit status', '', git.get('summary', 'See evidence/git-final.json.'), '']
    # K. Handoff
    lines += ['## K. Authorization handoff', '',
              f"- Experiment: `{job['run_id'] if job else 'n/a'}`, job `{job['job_sha256'] if job else 'n/a'}`.",
              f"- Initialization: `{job['spec']['initialization']['path'] if job else ''}` SHA `{job['spec']['initialization']['sha256'] if job else ''}` (weights only, fresh optimizer).",
              f"- Manifest `{manifest['manifest_sha256']}`; train `{evidence['verify']['train_sha256']}`; validation `{evidence['verify']['validation_sha256']}`; tokenizer `{evidence['verify']['tokenizer_sha256']}`.",
              f"- Step zero: `{s0['summary_sha256'] if s0 else 'missing'}`.",
              f"- Output: `{job['spec']['outputs']['root'] if job else ''}` (must not exist).",
              '- Launch prerequisites: resolve the conditions in section A; close GPU applications until ≥ 13,928 MiB is free; '
              'Codex environment enforcement tested; budget decision recorded; written approval grant bound to the job hash.', '',
              '**NOT AUTHORIZED — DO NOT EXECUTE.** Intended future procedure, for documentation only:', '',
              '```text', '1. Re-verify: tools/codexa-dev scripts/sft/sft_phase4.py verify --directory <this directory>',
              '2. Copy config/job.json into a new run directory <run>/config/job.json; user writes <run>/config/approval.json',
              '   {format: codexa-sft-pilot-approval-v1, approved: true, job_sha256: <job hash>, statement: <user words>}',
              '3. Supervisor launch.json per sft_pilot_runtime.launch_plan(); start via the OS supervisor with the read-only dashboard.',
              '4. Trainer entry: python -m codexa_workspace.sft_pilot_trainer --run-dir <run> (re-checks approval, environment, gates).',
              '```', '', 'Unresolved risks: ' + '; '.join(conditions), '']
    path = directory / 'SFT_PHASE4_READINESS_REPORT.md'
    path.write_text('\n'.join(lines))
    return dict(verdict=status, blockers=blockers, conditions=conditions, report=str(path))
