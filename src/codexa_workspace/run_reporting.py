"""Job-type-aware terminal reporting: run summary, AI handoff and wakeup prompt.

``completion.json`` (written once by the hash-bound supervisor) was designed for foundation
pretraining: for an SFT run its validation and generation fields are null and its handoff
text called every run "the five-hour foundational training experiment". This module reads
the job type from ``config/job.json`` and the run's real artifacts instead:

- foundation: corpus validation CE (baseline/best/final), generation checkpoints, ``best`` role;
- sft: the sealed step-zero baseline, every ``evaluations/step-*.json``, the project's own
  acceptance gates, and a best *candidate* checkpoint (never promoted).

Execution success (terminal state, verified final checkpoint) and quality evaluation are
reported separately. A metric with no evidence is the string ``unavailable``, never a guess.
Nothing here modifies a run's existing records: ``write_summary`` creates a new, separately
identifiable ``supervisor/run-summary.json`` (or any path given) and refuses to overwrite.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from codexa_workspace import training_telemetry

FORMAT = 'codexa-run-summary-v1'
UNAVAILABLE = 'unavailable'
SUMMARY_FILE = 'run-summary.json'


def _read(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _value(value):
    return UNAVAILABLE if value is None else value


# ---------------------------------------------------------------- job identity

def job_kind(job: dict | None) -> str:
    job = job or {}
    spec = job.get('spec') or {}
    formats = ' '.join(str(x) for x in (job.get('format'), spec.get('format')) if x)
    if 'sft' in formats or str(spec.get('purpose', '')).lower().startswith('conversational sft'):
        return 'sft'
    if 'foundation' in formats or 'pretraining' in formats:
        return 'foundation'
    return 'unknown'


def _budget_seconds(spec: dict) -> float | None:
    budget = spec.get('budget')
    if isinstance(budget, dict):
        return budget.get('total_seconds') or budget.get('seconds')
    return budget if isinstance(budget, (int, float)) else None


def _duration(seconds: float | None) -> str:
    if not seconds:
        return 'unknown budget'
    return f'{seconds / 3600:g} h budget' if seconds >= 3600 else f'{seconds:g} s budget'


def experiment(job: dict | None, run_dir: Path) -> dict:
    job = job or {}
    spec = job.get('spec') or {}
    kind = job_kind(job)
    name = spec.get('run_name') or job.get('run_id') or Path(run_dir).name
    seconds = _budget_seconds(spec)
    purpose = spec.get('purpose') or {'foundation': 'foundation continued pretraining',
                                      'sft': 'conversational supervised fine-tuning'}.get(kind, 'training run')
    return dict(name=name, kind=kind, purpose=purpose, budget_seconds=seconds,
                label=f'{purpose} ({_duration(seconds)})', job_format=job.get('format'),
                job_sha256=job.get('job_sha256'), spec_sha256=job.get('spec_sha256'))


# ---------------------------------------------------------------- execution (independent of quality)

def execution(completion: dict) -> dict:
    verified = completion.get('final_checkpoint_verification') == 'verified'
    state = completion.get('terminal_state')
    succeeded = state == 'COMPLETED' and verified and not completion.get('errors')
    return dict(outcome='SUCCEEDED' if succeeded else ('COMPLETED_UNVERIFIED' if state == 'COMPLETED' else state),
                terminal_state=state, stop_reason=completion.get('reason'),
                exit_code=(completion.get('exit_status') or {}).get('code'),
                optimizer_updates=_value(completion.get('optimizer_steps')),
                max_updates=_value(completion.get('max_steps')),
                processed_tokens=_value(completion.get('processed_tokens')),
                elapsed_seconds=_value(completion.get('elapsed_seconds')),
                deadline_seconds=_value(completion.get('deadline_seconds')),
                final_checkpoint=completion.get('final_checkpoint') or UNAVAILABLE,
                final_checkpoint_verification=completion.get('final_checkpoint_verification') or UNAVAILABLE,
                errors=completion.get('errors') or [])


# ---------------------------------------------------------------- SFT quality

def _sft_point(step: int, evaluation: dict) -> dict:
    conversation = (evaluation.get('generation') or {}).get('conversation/greedy') or {}
    gold, reference = evaluation.get('gold') or {}, (evaluation.get('reference') or {}).get('sources') or {}
    return dict(step=step, schedule=evaluation.get('schedule', 'baseline' if step == 0 else UNAVAILABLE),
                natural_end=_value(conversation.get('natural_end')), chat_end=_value(conversation.get('chat_end')),
                severe_repetition=_value(conversation.get('severe_degeneration')),
                strict_success=_value(conversation.get('strict_success')),
                conversational_cases=_value(conversation.get('cases')), strict_cases=_value(conversation.get('strict_cases')),
                mean_generated_tokens=_value(conversation.get('mean_generated_tokens')),
                validation_assistant_ce=_value(gold.get('assistant_supervised_ce')),
                gold_end_nll=_value(gold.get('mean_gold_end_nll')),
                premature_end_probability=_value(gold.get('mean_premature_end_probability')),
                reference_ce_wikipedia=_value(reference.get('wikipedia')),
                reference_ce_fineweb_edu=_value(reference.get('fineweb_edu')))


def sft_quality(run_dir: Path, job: dict, completion: dict) -> dict:
    spec = job.get('spec') or {}
    step_zero_path = (spec.get('evaluation') or {}).get('step_zero_path')
    step_zero = _read(step_zero_path) if step_zero_path else None
    evaluations = {}
    for path in sorted((Path(run_dir) / 'evaluations').glob('step-*.json')):
        document = _read(path)
        if document is not None:
            evaluations[int(path.stem.split('-')[1])] = document
    result = dict(baseline_path=step_zero_path or UNAVAILABLE, evaluation_steps=sorted(evaluations),
                  baseline=_sft_point(0, step_zero) if step_zero else UNAVAILABLE,
                  trajectory=[_sft_point(step, evaluations[step]) for step in sorted(evaluations)])
    if not evaluations:
        return dict(result, outcome=UNAVAILABLE, final=UNAVAILABLE, acceptance=UNAVAILABLE, best_candidate=UNAVAILABLE,
                    note='No evaluation artifacts were found for this run.')
    final_step = max(evaluations)
    result['final'] = _sft_point(final_step, evaluations[final_step])
    if step_zero is None:
        return dict(result, outcome=UNAVAILABLE, acceptance=UNAVAILABLE, best_candidate=UNAVAILABLE,
                    note='The sealed step-zero baseline is missing; no gate can be computed.')
    from codexa_workspace.sft_pilot_evaluation import acceptance_gates
    resource_stop = completion.get('reason') in ('MEMORY_HEADROOM_GATE', 'DISK_HEADROOM', 'DISK_RESERVE_WATCHDOG')
    resources = None if completion.get('terminal_state') is None else not resource_stop
    try:
        acceptance = acceptance_gates(step_zero, evaluations[final_step], None, resources_passed=resources)
    except ValueError as error:                       # suite drift is reported, never hidden
        return dict(result, outcome=UNAVAILABLE, acceptance=UNAVAILABLE, best_candidate=UNAVAILABLE,
                    note=f'Acceptance gates could not be computed: {error}')
    result.update(final_step=final_step, acceptance=acceptance, outcome=acceptance['overall'],
                  best_candidate=_best_candidate(run_dir, step_zero, evaluations, acceptance_gates))
    return result


def _best_candidate(run_dir: Path, step_zero: dict, evaluations: dict, gates) -> dict | str:
    """Among evaluated steps that have a published checkpoint: fewest failing automatic gates, then lowest
    validation CE. A candidate only: never promoted, and human review is still required."""
    report = _read(Path(run_dir) / 'metrics/report.json') or {}
    checkpoints = {c.get('optimizer_step'): c for c in report.get('checkpoints') or [] if c.get('verified')}
    ranked = []
    for step, evaluation in evaluations.items():
        if step not in checkpoints:
            continue
        try:
            statuses = gates(step_zero, evaluation, None)['gates']
        except ValueError:
            continue
        failures = sorted(name for name, gate in statuses.items() if gate['status'] == 'FAIL')
        ce = (evaluation.get('gold') or {}).get('assistant_supervised_ce', float('inf'))
        ranked.append((len(failures), ce, -step, step, failures))
    if not ranked:
        return UNAVAILABLE
    failures_count, ce, _, step, failures = min(ranked)
    checkpoint = checkpoints[step]
    return dict(step=step, path=str(Path(run_dir) / 'checkpoints' / checkpoint['path']) if checkpoint.get('path') else UNAVAILABLE,
                sha256=checkpoint.get('sha256'), failing_gates=failures,
                validation_assistant_ce=ce, promoted=False,
                note='Candidate for review only; not promoted. Failing automatic gates still apply.')


# ---------------------------------------------------------------- foundation quality

def foundation_quality(run_dir: Path, completion: dict) -> dict:
    validation = completion.get('validation') or {}
    baseline, final, best = validation.get('baseline'), validation.get('final'), validation.get('best')
    if baseline is None or final is None:
        outcome = UNAVAILABLE
    else:
        outcome = 'VALIDATION_IMPROVED' if final < baseline else 'VALIDATION_NOT_IMPROVED'
    from codexa_workspace.training_supervisor import recovery_role
    best_role = recovery_role(run_dir, 'best', hash_payload=False)
    counters = (_read(Path(run_dir) / 'metrics/report.json') or {}).get('counters') or {}

    def capability(entry: dict) -> dict:
        comparison = (entry.get('comparison') or {})
        plain = comparison.get('plain', comparison) if isinstance(comparison, dict) else {}
        overall = plain.get('overall') or {}
        return dict(step=entry.get('optimizer_step'), label=entry.get('label', 'periodic'),
                    top1=_value(overall.get('top1')), baseline_top1=_value(overall.get('baseline_top1')),
                    n=_value(overall.get('n')), gained=_value(overall.get('gained')), lost=_value(overall.get('lost')),
                    p_two_sided=_value(overall.get('p_two_sided')),
                    mean_margin_change=_value(overall.get('mean_margin_change')),
                    degraded=_value(entry.get('degraded')),
                    skipped=entry.get('skipped') or entry.get('error'))
    full = [capability(g) for g in counters.get('generations') or [] if g.get('label') != 'baseline']
    periodic = [capability(g) for g in counters.get('periodic') or []]
    return dict(outcome=outcome, validation_baseline=_value(baseline), validation_best=_value(best),
                capability_full=full or UNAVAILABLE, capability_periodic=periodic or UNAVAILABLE,
                validation_final=_value(final), validation_final_step=_value(validation.get('final_step')),
                baseline_sources=validation.get('baseline_sources') or UNAVAILABLE,
                final_sources=validation.get('final_sources') or UNAVAILABLE,
                generations=completion.get('generations') or UNAVAILABLE,
                best_candidate=dict(best_role, promoted=False) if best_role else UNAVAILABLE,
                note='Validation CE is a training signal, not a behavioural quality verdict; generations '
                     'require genuine human semantic review.')


# ---------------------------------------------------------------- telemetry

def telemetry_summary(run_dir: Path) -> dict:
    path = Path(run_dir) / 'metrics' / training_telemetry.FILENAME
    if path.exists():
        return dict(source=str(path), **training_telemetry.summarize(path))
    legacy = Path(run_dir) / 'metrics/train_metrics.jsonl'   # llm_training logger (foundation runs)
    rows = []
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if not rows:
        return dict(source=UNAVAILABLE, available=False,
                    note='This run predates per-update telemetry; loss history and peak VRAM are unavailable.')
    peaks = [r['peak_reserved_vram_bytes'] for r in rows if r.get('peak_reserved_vram_bytes') is not None]
    return dict(source=str(legacy), available=True, updates=len(rows), last_step=rows[-1].get('optimizer_step'),
                first_loss=rows[0].get('training_loss'), last_loss=rows[-1].get('training_loss'),
                peak_reserved_vram_bytes=max(peaks) if peaks else UNAVAILABLE,
                max_gradient_norm_pre_clip=max((r.get('gradient_norm') or 0.0) for r in rows))


# ---------------------------------------------------------------- summary, handoff and prompt

def build_summary(run_dir: Path, completion: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    completion = completion or _read(run_dir / 'supervisor/completion.json') or {}
    job = _read(run_dir / 'config/job.json') or {}
    identity = experiment(job, run_dir)
    if identity['kind'] == 'sft':
        quality = sft_quality(run_dir, job, completion)
    elif identity['kind'] == 'foundation':
        quality = foundation_quality(run_dir, completion)
    else:
        quality = dict(outcome=UNAVAILABLE, note='Unknown job type; no quality evaluation is defined.')
    return dict(format=FORMAT, run_dir=str(run_dir), event_id=completion.get('event_id'), experiment=identity,
                execution=execution(completion), quality=quality, telemetry=telemetry_summary(run_dir),
                human_review='PENDING', promotion='none (never automatic)')


def write_summary(summary: dict, path: Path) -> Path:
    """Create a new summary file; an existing one is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(summary, stream, indent=2, default=str)
        stream.write('\n')
    return path


def instruction(summary: dict) -> str:
    e = summary['experiment']
    return (f"Codexa run `{e['name']}` — {e['label']} — has reached a terminal state. The Python supervisor has "
            "finished monitoring. Read the run summary and completion record, validate the final checkpoint, "
            "compare the final results against the baseline, report execution and quality outcomes separately, "
            "diagnose any failures, and produce the final assessment. Do not start another training run.")


def _fmt(value, digits=4):
    if value == UNAVAILABLE or value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f'{value:.2e}' if 0 < abs(value) < 0.01 else f'{value:.{digits}f}'
    return str(value)


def handoff_text(summary: dict, completion: dict | None = None) -> str:
    e, x, q, t = summary['experiment'], summary['execution'], summary['quality'], summary['telemetry']
    final = x['final_checkpoint'] if isinstance(x['final_checkpoint'], dict) else {}
    lines = [f"# Codexa training handoff — {e['name']}", '', instruction(summary), '',
             '## Experiment', f"- Name: `{e['name']}`", f"- Type: **{e['kind']}** — {e['label']}",
             f"- Job SHA256: `{e['job_sha256']}`; format `{e['job_format']}`", '',
             '## Execution (did the run complete safely?)',
             f"- Outcome: **{x['outcome']}** — terminal state `{x['terminal_state']}`, stop reason `{x['stop_reason']}`",
             f"- Optimizer updates: {x['optimizer_updates']} / {x['max_updates']}; processed tokens: {x['processed_tokens']}",
             f"- Elapsed: {_fmt(x['elapsed_seconds'], 0)} s of {_fmt(x['deadline_seconds'], 0)} s",
             f"- Final checkpoint: `{final.get('path', UNAVAILABLE)}` (step {final.get('optimizer_step', UNAVAILABLE)}), "
             f"verification {x['final_checkpoint_verification']}",
             f"- Errors: {'; '.join(x['errors']) if x['errors'] else 'none recorded'}", '',
             '## Quality (did the model get better?)', f"- Automatic outcome: **{q.get('outcome')}** (human review PENDING)"]
    if e['kind'] == 'sft' and isinstance(q.get('final'), dict) and isinstance(q.get('baseline'), dict):
        keys = [('natural_end', 'Natural endings (greedy)'), ('chat_end', 'Chat END'),
                ('severe_repetition', 'Severe repetition'), ('strict_success', 'Strict exact-format'),
                ('validation_assistant_ce', 'Validation assistant CE'), ('gold_end_nll', 'Gold END NLL'),
                ('premature_end_probability', 'Premature END probability'),
                ('reference_ce_wikipedia', 'Reference CE Wikipedia'), ('reference_ce_fineweb_edu', 'Reference CE FineWeb-Edu')]
        steps = [p['step'] for p in q['trajectory']]
        lines += ['', '| Metric | Baseline (step 0) | ' + ' | '.join(f'Update {s}' for s in steps) + ' |',
                  '|---|---|' + '---|' * len(steps)]
        for key, label in keys:
            lines.append(f"| {label} | {_fmt(q['baseline'][key])} | "
                         + ' | '.join(_fmt(p[key]) for p in q['trajectory']) + ' |')
        if isinstance(q.get('acceptance'), dict):
            lines += ['', '### Acceptance gates (final evaluation vs sealed baseline)', '| Gate | Measured | Threshold | Result |',
                      '|---|---|---|---|']
            lines += [f"| {name} | {_fmt(g.get('measured'))} | {g.get('threshold')} | {g['status']} |"
                      for name, g in q['acceptance']['gates'].items()]
        candidate = q.get('best_candidate')
        lines += ['', '### Best candidate checkpoint (not promoted)',
                  f"- {UNAVAILABLE}" if not isinstance(candidate, dict) else
                  f"- Update {candidate['step']}: `{candidate['path']}` sha256 `{candidate['sha256']}`; failing gates: "
                  f"{', '.join(candidate['failing_gates']) or 'none'}"]
    elif e['kind'] == 'foundation':
        lines += [f"- Validation CE: baseline {_fmt(q.get('validation_baseline'))}, best {_fmt(q.get('validation_best'))}, "
                  f"final {_fmt(q.get('validation_final'))} at step {q.get('validation_final_step')}"]
        candidate = q.get('best_candidate')
        lines.append(f"- Best checkpoint role: {UNAVAILABLE if not isinstance(candidate, dict) else candidate.get('path')}")
        sources = q.get('final_sources') if isinstance(q.get('final_sources'), dict) else {}
        base_sources = q.get('baseline_sources') if isinstance(q.get('baseline_sources'), dict) else {}
        for name in sorted(set(sources) | set(base_sources)):
            lines.append(f"- {name} CE: baseline {_fmt(base_sources.get(name))} -> final {_fmt(sources.get(name))}")
        for title, entries in (('Full capability evaluations (plain form, paired vs sealed baseline)', q.get('capability_full')),
                               ('Periodic capability scoring (plain form)', q.get('capability_periodic'))):
            lines += ['', f'### {title}']
            if not isinstance(entries, list):
                lines.append(f'- {UNAVAILABLE}')
                continue
            lines += ['| Step | Label | Top-1 | Baseline top-1 | Gained | Lost | McNemar p | Mean margin change | Degraded |',
                      '|---|---|---|---|---|---|---|---|---|']
            lines += [f"| {e['step']} | {e['label']} | {e['top1']}/{e['n']} | {e['baseline_top1']} | {e['gained']} | {e['lost']} | "
                      f"{_fmt(e['p_two_sided'])} | {_fmt(e['mean_margin_change'])} | {e['degraded']}"
                      + (f" ({e['skipped']})" if e.get('skipped') else '') + ' |' for e in entries]
        lines.append('- Lower CE without capability gains must be reported as such; repetition changes alone are not capability.')
    if q.get('note'):
        lines.append(f"- Note: {q['note']}")
    lines += ['', '## Telemetry']
    if t.get('available'):
        lines += [f"- Status: **{t.get('status', UNAVAILABLE)}**" + (f"; failures: {t.get('telemetry_failures')}" if t.get('telemetry_failures') else ''),
                  f"- Source: `{t['source']}`; updates {t.get('updates')}; loss {_fmt(t.get('first_loss'))} → {_fmt(t.get('last_loss'))}",
                  f"- Peak reserved VRAM: {_fmt(t.get('peak_reserved_vram_bytes'))} bytes; "
                  f"max pre-clip gradient norm {_fmt(t.get('max_gradient_norm_pre_clip'))}; "
                  f"clipped updates {t.get('clipped_updates', UNAVAILABLE)}"]
    else:
        lines.append(f"- {t.get('note', UNAVAILABLE)}")
    lines += ['', '## Paths (inspect as needed; do not modify)', f"- run: `{summary['run_dir']}`",
              f"- completion: `{Path(summary['run_dir']) / 'supervisor/completion.json'}`",
              f"- run summary: `{Path(summary['run_dir']) / 'supervisor' / SUMMARY_FILE}`",
              f"- evaluations: `{Path(summary['run_dir']) / 'evaluations'}`",
              f"- telemetry: `{t.get('source', UNAVAILABLE)}`"]
    if e['kind'] == 'sft':
        lines.append(f"- sealed baseline: `{q.get('baseline_path', UNAVAILABLE)}`")
    lines += ['', '## Constraints',
              '- Analysis and reporting only. Do not launch, resume or schedule training; do not delete or move assets.',
              '- Never promote a checkpoint. Generations require genuine human semantic review; do not fabricate it.']
    return '\n'.join(lines) + '\n'


def prompt_text(run_dir: Path, provider: str, summary: dict) -> str:
    run_dir = Path(run_dir)
    output = run_dir / 'final-report' / ('ai-analysis.md' if provider == 'codex' else 'ai-analysis-claude.md')
    target = 'Your final message is saved verbatim as' if provider == 'codex' else 'Write your assessment to'
    return (f"{instruction(summary)}\n\nStructured handoff: {run_dir / 'supervisor/ai-handoff.md'}\n"
            f"Run summary: {run_dir / 'supervisor' / SUMMARY_FILE}\n"
            f"Completion record: {run_dir / 'supervisor/completion.json'}\n"
            f"{target} {output}. Keep it concise: execution verdict, quality verdict with metrics vs baseline, "
            "checkpoint validity, failure diagnosis if any, and the recommended next decision (a human decision).")


def outcome_label(run_dir: Path) -> str:
    """Short phrase for the desktop popup, e.g. 'conversational SFT feasibility pilot'."""
    return experiment(_read(Path(run_dir) / 'config/job.json'), Path(run_dir))['purpose']


def main(arguments: list[str] | None = None) -> int:
    """Render a run summary/handoff for an existing run without touching its records."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--write-summary', type=Path, help='create a NEW summary JSON at this path (never overwrites)')
    parser.add_argument('--write-handoff', type=Path, help='create a NEW handoff Markdown at this path (never overwrites)')
    args = parser.parse_args(arguments)
    summary = build_summary(args.run_dir)
    if args.write_summary:
        write_summary(summary, args.write_summary)
    text = handoff_text(summary)
    if args.write_handoff:
        descriptor = os.open(args.write_handoff, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
