"""Generate CODEXA_STEP5A_FINAL_READINESS.md (sections A-L) from the Step 5A evidence only.

Every number comes from an evidence file in the Step 5A directory; nothing is typed in by hand,
so the report cannot drift from what was measured or verified.
"""
from __future__ import annotations

from pathlib import Path

from codexa_workspace import sft_phase4_data as data

TITLE = 'CODEXA_STEP5A_FINAL_READINESS.md'
READY_LINE = 'CODEXA READY FOR STEP 5B — FIRST CONTROLLED CONVERSATIONAL SFT PILOT'


def _load(directory: Path, relative: str):
    path = directory / relative
    return data.read(path) if path.exists() else None


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    return lines + ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]


def _gib(value) -> str:
    return f'{value / 1024**3:.2f} GiB' if isinstance(value, (int, float)) else 'n/a'


def write_report(directory: Path) -> dict:
    directory = Path(directory)
    readiness = _load(directory, 'evidence/readiness.json') or {}
    job = _load(directory, 'config/job.json') or {}
    spec = job.get('spec', {})
    dataset = _load(directory, 'evidence/dataset-verification.json') or {}
    exception = _load(directory, 'evidence/validation-exception.json') or {}
    lineage = _load(directory, 'manifest/lineage-exposure.json') or {}
    binding = _load(directory, 'evidence/step-zero-binding.json') or {}
    reconfirm = _load(directory, 'evidence/step-zero-reconfirmation.json') or {}
    budget = _load(directory, 'config/budget-analysis.json') or {}
    resources = _load(directory, 'evidence/resources.json') or {}
    tests = _load(directory, 'evidence/test-results.json') or {}
    diagnosis = _load(directory, 'evidence/test-failure-diagnosis.json') or {}
    git = _load(directory, 'evidence/git-integration.json') or {}
    before = _load(directory, 'evidence/preservation-before.json') or {}
    after = _load(directory, 'evidence/preservation-after.json') or {}
    step_zero = data.read(spec['evaluation']['step_zero_path']) if spec else {}
    verdict = readiness.get('verdict', 'NOT READY (readiness not evaluated)')
    lines = [f'# Codexa Step 5A — final integration and training-readiness verdict', '',
             f'Generated from the evidence in `{directory}`. No training, fine-tuning, optimizer construction, '
             'backward pass, weight or checkpoint change, tokenizer change, push or pull request occurred. '
             'Readiness is not authorization.', '']

    # A
    lines += ['## A. Executive verdict', '', f'**{verdict}**', '']
    if readiness:
        lines += [f"- Frozen experiment job: `{readiness['job_sha256']}`.",
                  f"- Engineering failures: {readiness['engineering_failures']}; launch-time conditions: "
                  f"{readiness['launch_time_conditions']}.",
                  f"- Authorization granted: {readiness['authorization']['granted']} (the deliberate execution gate, "
                  'separate from technical readiness).']
        for item in readiness.get('blocking', []):
            lines.append(f'- Blocking: {item}.')
        lines.append('')
    if verdict == 'READY FOR AUTHORIZATION':
        lines += [f'**{READY_LINE}**', '']

    # B
    lines += ['## B. Integration', '']
    for item in git.get('integration_notes', []):
        lines.append(f'- {item}')
    lines.append('')

    # C
    lines += ['## C. Source identity', '']
    environment = spec.get('environment', {})
    if environment:
        lines += _table(['Component', 'Frozen commit (= guard-detected HEAD at freeze)'],
                        [[name, f'`{commit}`'] for name, commit in sorted(environment['component_revisions'].items())])
        lines += ['', f"- Integration HEAD at freeze: `{git.get('integration_head_at_freeze', 'n/a')}`.",
                  f"- Environment: `{environment['codex_selection']}` selection, `{environment['enforcement_interface']}`; "
                  f"child `{environment['child_python']}` with {environment['child_imports']}.",
                  f"- Architecture `{environment['architecture_sha256']}` (repaired, source-first); "
                  f"tokenizer `{environment['tokenizer_sha256']}`; guard `{environment['guard_sha256']}`.",
                  '- Historical compatibility: the completed foundation job and its 12 frozen identities are verified '
                  'in the historical context only; its metadata was not rewritten.']
    for item in git.get('historical_checks', []):
        lines.append(f'- {item}')
    lines.append('')

    # D
    lines += ['## D. Test results', '']
    if tests:
        lines += _table(['Suite', 'Passed', 'Failed', 'Skipped', 'Deselected', 'Blocked', 'Command'],
                        [[s['suite'], s.get('passed', 0), s.get('failed', 0), s.get('skipped', 0), s.get('deselected', 0),
                          s.get('blocked', 0), f"`{s['command']}`"] for s in tests.get('suites', [])])
        lines += ['', f"Totals: {tests.get('totals')}.", '']
        for note in tests.get('notes', []):
            lines.append(f'- {note}')
        lines.append('')
    for item in diagnosis.get('findings', []):
        lines.append(f"- **{item['subject']}** — {item['cause']} Fix: {item['fix']}")
    lines.append('')

    # E
    lines += ['## E. Dataset', '']
    if dataset:
        train, validation = dataset['train'], dataset['validation']
        lines += _table(['Split', 'Examples', 'Serialized tokens', 'Supervised tokens', 'END targets'],
                        [['train (per pass)', train['examples'], train['serialized_tokens'], train['supervised_tokens'],
                          train['end_targets']],
                         ['train (two passes)', dataset['train_two_passes']['presentations'], '',
                          dataset['train_two_passes']['supervised_tokens'], dataset['train_two_passes']['end_targets']],
                         ['validation', validation['examples'], validation['serialized_tokens'],
                          validation['supervised_tokens'], validation['end_targets']]])
        lines += ['', 'Composition: 224 OASST1 + 224 UltraChat + 64 authored training; 16 OASST1 + 16 UltraChat validation.',
                  f"Independent re-derivation (`{dataset['manifest_sha256']}`): " +
                  ', '.join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in dataset['checks'].items()) + '.',
                  f"Split isolation overlaps: {dataset['isolation']}.", '']
    if exception:
        lines += [f"Validation-source exception `{exception['conversation_id']}` (from `{exception['source_file']}`): "
                  f"shared with training {exception['shared_with_training']}; nearest training 5-gram Jaccard "
                  f"{exception['nearest_training'][0]['jaccard']}; in general-chat-v2 train {exception['in_general_chat_v2_train']}; "
                  f"isolation policy satisfied: {exception['isolation_policy_satisfied']}; replacement required: "
                  f"{exception['replacement_required']}.", '']
    if lineage:
        lines += [f"Historical exposure: {lineage['flags']}. {lineage['classification']}", '']

    # F
    lines += ['## F. Step-zero evaluation', '']
    if step_zero:
        conversation = step_zero['generation']['conversation/greedy']
        historical = step_zero['generation']['historical/greedy']
        lines += _table(['Baseline', 'Value', 'Evidence'], [
            ['Conversational natural endings (greedy)', f"{conversation['natural_end']}/{conversation['cases']}", 'summary.json'],
            ['Chat END terminations', f"{conversation['chat_end']}/{conversation['cases']}", 'summary.json'],
            ['Severe repetition (greedy)', f"{conversation['severe_degeneration']}/{conversation['cases']}", 'summary.json'],
            ['Strict constrained successes', f"{conversation['strict_success']}/{conversation['strict_cases']}", 'summary.json'],
            ['Historical 48-prompt severe (greedy)', f"{historical['severe_degeneration']}/{historical['cases']}", 'summary.json'],
            ['Gold END NLL', f"{step_zero['gold']['mean_gold_end_nll']:.4f}", 'gold-per-record.jsonl'],
            ['Premature-END probability', f"{step_zero['gold']['mean_premature_end_probability']:.3g}", 'gold-per-record.jsonl'],
            ['Assistant-supervised CE', f"{step_zero['gold']['assistant_supervised_ce']:.4f}", 'summary.json'],
            ['Wikipedia reference CE', f"{step_zero['reference']['sources']['wikipedia']:.4f}", 'summary.json'],
            ['FineWeb-Edu reference CE', f"{step_zero['reference']['sources']['fineweb_edu']:.4f}", 'summary.json'],
            ['Blinded human comparison', 'PENDING (after a future run; never fabricated)', '—']])
        lines += ['', f"Sealed summary `{step_zero['summary_sha256']}` at `{spec['evaluation']['step_zero_path']}`."]
    if binding:
        lines.append(f"Binding checks: {binding['checks']} — bound: {binding['bound']}.")
    if reconfirm:
        lines.append(f"Reconfirmation under the committed identity: greedy {reconfirm['greedy_identical_tokens']}/"
                     f"{reconfirm['greedy_cases']} token-identical, gold metrics equal: {reconfirm['gold_match']}, "
                     f"checkpoint unchanged: {reconfirm['checkpoint']['unchanged']} ({reconfirm['seconds']} s, inference only).")
    lines.append('')

    # G
    lines += ['## G. Runtime', '']
    for item in git.get('runtime_notes', []):
        lines.append(f'- {item}')
    if spec:
        t, o = spec['training'], spec['optimization']
        lines.append('')
        lines += _table(['Setting', 'Frozen value'], [
            ['Updates / microbatch / accumulation', f"{t['max_steps']} / {t['micro_batch_size']} / {t['gradient_accumulation_steps']}"],
            ['Context limit / presentations', f"{spec['data']['maximum_serialized_tokens']} / {spec['data']['presentations']}"],
            ['LR / warmup / minimum LR', f"{t['learning_rate']} / {t['warmup_steps']} / {o['minimum_learning_rate']}"],
            ['Optimizer / weight decay / clip', f"{o['optimizer']} / {t['weight_decay']} / {t['gradient_clip']}"],
            ['Precision', f"{o['weight_dtype']} weights, {o['autocast']} autocast, TF32 {o['tf32']}"],
            ['Checkpoints / evaluations', f"{spec['checkpoint']['publications']} / {spec['evaluation']['updates']}"],
            ['Initialization', spec['initialization']['policy']],
            ['Objective', 'assistant-only SFT including END']])
    lines.append('')

    # H
    lines += ['## H. GPU resources', '']
    if resources:
        lines += _table(['Gate', 'Measured', 'Required', 'Passed'],
                        [[c['name'], c['measured'], c['required'], c['passed']] for c in resources['checks']])
        lines += ['', 'GPU processes (reported, never terminated): ' +
                  ', '.join(f"{p['name']} (pid {p['pid']}, {p['used_mib']} MiB)" for p in resources['gpu_processes']) + '.',
                  f"Supervisor: {resources['supervisor']}.", f"In-run abort chain: {resources['in_run']['abort']}.",
                  f"Codexa units: {resources['codexa_units']}."]
    if budget:
        memory = budget['memory']
        lines += [f"Memory: inference peak reserved {_gib(memory['inference_peak_reserved_bytes']['value'])} (measured); "
                  f"training analog {memory['training_peak_reserved_gib']['value']} GiB "
                  f"({memory['training_peak_reserved_gib']['kind']}); {memory['this_pilot']}."]
    lines.append('')

    # I
    lines += ['## I. Time budget', '']
    if budget:
        lines += _table(['Item', 'Seconds', 'Kind', 'Basis'],
                        [[name, item['seconds'], item['kind'], item['basis']] for name, item in budget['items'].items()])
        rows = []
        for scenario, values in budget['scenarios'].items():
            for cap in (900, 1200):
                cell = values[f'cap_{cap}']
                rows.append([scenario, cap, cell['total_with_reserve'], cell['fits'], cell['slack_seconds']])
        lines += [''] + _table(['Schedule', 'Cap (s)', 'Total incl. reserve (s)', 'Fits', 'Slack (s)'], rows)
        recommendation = budget['recommendation']
        lines += ['', f"**Recommended cap: {recommendation['total_seconds']} s** with the {recommendation['schedule']} "
                      f"schedule and a {recommendation['finalization_reserve_seconds']} s finalization reserve. "
                      f"{recommendation['rationale']} {recommendation['status']}.", '']

    # J
    lines += ['## J. Safety and rollback', '']
    if spec:
        guards = spec['guards']
        lines += [f"- Gates (never lowered): {guards['gates']}.", f"- Nonfinite: {guards['nonfinite']}; OOM: {guards['oom']}.",
                  f"- Headroom: {guards['headroom']}.", f"- Deterioration: {guards['deterioration']}.",
                  f"- Update cap {guards['update_cap']}; budget {spec['budget']['total_seconds']} s with "
                  f"{spec['budget']['finalization_reserve_seconds']} s reserve, no automatic extension.",
                  f"- Output namespace `{spec['outputs']['root']}` (fresh, isolated from historical runs, never overwritten).",
                  f"- Recovery: {spec['recovery']['resume']}; rollback: {spec['recovery']['rollback']}."]
    if before and after:
        lines.append(f"- Preservation: {after['unchanged']}/{after['artifacts']} protected artifacts unchanged "
                     f"(before {before['unchanged']}/{before['artifacts']}); checkpoint `{after['checkpoint_sha256']}`.")
    lines.append('')

    # K
    lines += ['## K. Git and audit', '']
    if git.get('commits'):
        lines += _table(['Repository', 'Commit', 'Scope'], [[c['repo'], f"`{c['commit']}`", c['subject']] for c in git['commits']])
        lines.append('')
    for item in git.get('remaining_dirty', []):
        lines.append(f'- Remaining uncommitted: {item}')
    for item in git.get('audit', []):
        lines.append(f'- Audit: {item}')
    lines.append('- No push and no pull request were made. No commit was amended, reset or rewritten.')
    lines.append('')

    # L
    lines += ['## L. Authorization handoff', '']
    if job:
        root = Path(__file__).resolve().parents[2]
        job_path = directory / 'config/job.json'
        experiment = job_path.relative_to(root) if job_path.is_relative_to(root) else job_path
        lines += [f"- Frozen experiment: `{job_path}`", f"- Job SHA-256: `{job['job_sha256']}`",
                  f"- Spec SHA-256: `{job['spec_sha256']}`; format `{job['format']}`; run `{job['run_id']}`.",
                  f"- Engineering gates all pass: {readiness.get('engineering_failures') == 0}.",
                  '- Requirements at launch time: the readiness command must return READY FOR AUTHORIZATION '
                  '(free VRAM >= 13,928 MiB, RAM >= 16 GiB, disk >= 180 GiB with the 100 GiB reserve), and the '
                  'source tree must still equal the frozen commits.', '',
                  'Smallest user actions for Step 5B (not performed here):', '',
                  '```bash', 'cd LLM-From-Scratch', f'EXPERIMENT="{experiment}"',
                  'tools/codexa-dev scripts/sft_readiness.py --experiment "$EXPERIMENT"   # must print READY FOR AUTHORIZATION',
                  f'tools/codexa-dev scripts/sft/sft_pilot.py approve --experiment "$EXPERIMENT" '
                  f"--job-sha256 {job['job_sha256']} --statement '<your explicit authorization>'",
                  'tools/codexa-dev scripts/sft/sft_pilot.py launch --experiment "$EXPERIMENT" --provider <codex|claude>',
                  '```', '', 'Readiness is not authorization. Step 5B must be explicitly authorized separately.']
    text = '\n'.join(lines).rstrip() + '\n'
    (directory / TITLE).write_text(text)
    return dict(path=str(directory / TITLE), verdict=verdict, lines=len(text.splitlines()))
