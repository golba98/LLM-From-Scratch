"""Job-type-aware run reporting: foundation summaries, SFT labelling, handoff text and notifier wiring.

Synthetic run directories only; mocked desktop and AI backends (never notify-send/codex/claude).
"""
import json

import pytest

from codexa_workspace.notifications import reporting
from codexa_workspace.notifications import notifier as notifications
from codexa_workspace.training import telemetry
from tests.notifications.test_notifier import Agent, Desktop


def evaluation(step, *, natural=11, severe=5, strict=0, ce=2.26, end=1.14, premature=0.007, wiki=3.04, fineweb=3.17,
               schedule='reduced'):
    return dict(step=step, schedule=schedule, finite=True,
                generation={'conversation/greedy': dict(cases=16, natural_end=natural, chat_end=natural, document_eos=0,
                                                        length_limit=16 - natural, severe_degeneration=severe,
                                                        historical_severe=severe, strict_cases=7, strict_success=strict,
                                                        mean_generated_tokens=50.0, empty_responses=0, seconds=8.0)},
                gold=dict(assistant_supervised_ce=ce, mean_gold_end_nll=end, mean_premature_end_probability=premature),
                reference=dict(sources=dict(wikipedia=wiki, fineweb_edu=fineweb)), rows=[{'response': 'x'}])


def sft_run(tmp_path, *, evaluations=None, checkpoints=(64, 128), state='COMPLETED', reason='STEP_CAP'):
    run = tmp_path / 'outputs/training/codexa-sft-test'
    for name in ('config', 'supervisor', 'evaluations', 'metrics'):
        (run / name).mkdir(parents=True, exist_ok=True)
    step_zero = tmp_path / 'step-zero.json'
    step_zero.write_text(json.dumps(evaluation(0, natural=0, severe=15, strict=0, ce=3.02, end=18.16, premature=1e-6,
                                               wiki=3.01, fineweb=3.14, schedule='full')))
    job = dict(format='codexa-sft-pilot-job-v2', run_id='codexa-sft-test', job_sha256='j' * 64,
               spec=dict(format='codexa-sft-pilot-job-v2', run_name='codexa-sft-test',
                         purpose='conversational SFT feasibility pilot', budget=dict(total_seconds=1200),
                         training=dict(max_steps=128), evaluation=dict(step_zero_path=str(step_zero))))
    (run / 'config/job.json').write_text(json.dumps(job))
    for step, document in (evaluations if evaluations is not None else {
            32: evaluation(32), 64: evaluation(64, severe=4, ce=2.30), 128: evaluation(128, schedule='full')}).items():
        (run / f'evaluations/step-{step:05d}.json').write_text(json.dumps(document))
    (run / 'metrics/report.json').write_text(json.dumps(dict(checkpoints=[
        dict(path=f'slot-{i}/checkpoint.pt', sha256=f'{step:064d}', optimizer_step=step, verified=True)
        for i, step in enumerate(checkpoints)])))
    completion = dict(event_id='e-sft', run_id='codexa-sft-test', terminal_state=state, reason=reason,
                      optimizer_steps=128, max_steps=128, processed_tokens=51026, elapsed_seconds=331.0,
                      deadline_seconds=1200, final_checkpoint=dict(path=str(run / 'checkpoints/slot-1/checkpoint.pt'),
                                                                   optimizer_step=128, verified=True),
                      final_checkpoint_verification='verified', errors=[], exit_status=dict(code=0),
                      validation=dict(baseline=None, best=None, final=None), generations=None, wakeup_provider='claude',
                      paths=dict(run=str(run)))
    (run / 'supervisor/completion.json').write_text(json.dumps(completion))
    (run / 'supervisor/launch.json').write_text(json.dumps(dict(run_id='codexa-sft-test', wakeup=dict(provider='claude'),
                                                                execution_class='unit_test')))
    return run


def foundation_run(tmp_path):
    run = tmp_path / 'outputs/training/codexa-foundation-test'
    for name in ('config', 'supervisor'):
        (run / name).mkdir(parents=True, exist_ok=True)
    (run / 'config/job.json').write_text(json.dumps(dict(
        format='codexa-foundation-training-job-v1', run_id='codexa-foundation-test',
        spec=dict(format='codexa-foundation-pretraining-spec-v1', run_name='codexa-foundation-test',
                  budget=dict(seconds=18000, reserve_seconds=600)))))
    completion = dict(event_id='e-f', run_id='codexa-foundation-test', terminal_state='COMPLETED', reason='STEP_CAP',
                      optimizer_steps=1792, max_steps=1792, final_checkpoint_verification='verified', errors=[],
                      validation=dict(baseline=3.113, best=3.091, final=3.091, final_step=1792), paths=dict(run=str(run)))
    (run / 'supervisor/completion.json').write_text(json.dumps(completion))
    return run


def test_job_kinds_are_distinguished():
    assert reporting.job_kind(dict(format='codexa-sft-pilot-job-v2')) == 'sft'
    assert reporting.job_kind(dict(format='codexa-foundation-training-job-v1')) == 'foundation'
    assert reporting.job_kind(dict(spec=dict(purpose='conversational SFT feasibility pilot'))) == 'sft'
    assert reporting.job_kind({}) == 'unknown' and reporting.job_kind(None) == 'unknown'


def test_sft_run_reports_execution_and_marks_quality_unavailable(tmp_path):
    summary = reporting.build_summary(sft_run(tmp_path))
    e, x, q = summary['experiment'], summary['execution'], summary['quality']
    assert (e['name'], e['kind'], e['label']) == ('codexa-sft-test', 'sft', 'conversational SFT feasibility pilot (1200 s budget)')
    assert x['outcome'] == 'SUCCEEDED' and (x['optimizer_updates'], x['max_updates'], x['stop_reason']) == (128, 128, 'STEP_CAP')
    assert q['outcome'] == 'unavailable' and 'pre-rewrite-2026-10-13' in q['note']   # SFT gates were retired
    assert summary['human_review'] == 'PENDING'


def test_unavailable_evidence_is_explicit(tmp_path):
    summary = reporting.build_summary(sft_run(tmp_path, evaluations={}))
    assert summary['quality']['outcome'] == 'unavailable'
    assert summary['telemetry']['available'] is False and 'predates per-update telemetry' in summary['telemetry']['note']
    assert 'unavailable' in reporting.handoff_text(summary)


def test_sft_handoff_names_the_experiment(tmp_path):
    text = reporting.handoff_text(reporting.build_summary(sft_run(tmp_path)))
    assert 'five-hour' not in text and 'foundational' not in text
    assert 'conversational SFT feasibility pilot (1200 s budget)' in text and '**SUCCEEDED**' in text
    assert '**unavailable**' in text and 'pre-rewrite-2026-10-13' in text


def test_foundation_summary_keeps_validation_view(tmp_path):
    summary = reporting.build_summary(foundation_run(tmp_path))
    assert summary['experiment']['label'] == 'foundation continued pretraining (5 h budget)'
    assert summary['quality']['outcome'] == 'VALIDATION_IMPROVED' and summary['quality']['best_candidate'] == 'unavailable'
    text = reporting.handoff_text(summary)
    assert 'baseline 3.1130, best 3.0910, final 3.0910' in text and 'conversational' not in text


def test_telemetry_is_reported_when_present(tmp_path):
    run = sft_run(tmp_path)
    writer = telemetry.TelemetryWriter(run / 'metrics/telemetry.jsonl', run_id='codexa-sft-test', clip=1.0)
    writer.update(dict(optimizer_step=1, training_loss=2.5, gradient_norm=5.0, peak_reserved_vram_bytes=12,
                       device='cuda'))   # VRAM counts as measured only when the device is known to be CUDA
    writer.close()
    t = reporting.build_summary(run)['telemetry']
    assert t['available'] and t['clipped_updates'] == 1 and t['peak_reserved_vram_bytes'] == 12


def test_summary_files_are_new_and_never_overwritten(tmp_path):
    summary = reporting.build_summary(sft_run(tmp_path))
    path = reporting.write_summary(summary, tmp_path / 'summary.json')
    with pytest.raises(FileExistsError):
        reporting.write_summary(summary, path)


def test_notifier_writes_summary_and_sft_handoff_without_touching_completion(tmp_path):
    run = sft_run(tmp_path)
    before = (run / 'supervisor/completion.json').read_bytes()
    log = tmp_path / 'agent.jsonl'
    notifications.deliver(run, tmp_path, desktop_backend=Desktop(tmp_path / 'desktop.jsonl'), ai_backend=Agent(log))
    assert (run / 'supervisor/completion.json').read_bytes() == before
    summary = json.loads((run / 'supervisor/run-summary.json').read_text())
    assert summary['format'] == reporting.FORMAT and summary['quality']['outcome'] == 'unavailable'
    handoff = (run / 'supervisor/ai-handoff.md').read_text()
    assert 'conversational SFT' in handoff and 'five-hour' not in handoff
    prompt = json.loads(log.read_text().splitlines()[0])['prompt']
    assert prompt.startswith('Codexa run `codexa-sft-test` — conversational SFT')
    popup = json.loads((tmp_path / 'desktop.jsonl').read_text().splitlines()[0])
    assert 'conversational SFT feasibility pilot finished (128 updates)' in popup['body']


def test_reporting_failure_falls_back_to_the_legacy_handoff(tmp_path, monkeypatch):
    run = sft_run(tmp_path)
    monkeypatch.setattr(reporting, 'build_summary', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('broken')))
    notifications.deliver(run, tmp_path, desktop_backend=Desktop(tmp_path / 'd.jsonl'), ai_backend=Agent(tmp_path / 'a.jsonl'))
    assert (run / 'supervisor/ai-handoff.md').exists() and not (run / 'supervisor/run-summary.json').exists()


def test_foundation_handoff_reports_capability_against_the_sealed_baseline(tmp_path):
    run = foundation_run(tmp_path)
    (run / 'metrics').mkdir()
    entry = lambda step, label, lost: dict(optimizer_step=step, label=label, degraded=False,
                                           comparison=dict(plain=dict(overall=dict(n=266, top1=120 - lost, baseline_top1=120,
                                                                                   gained=4, lost=lost, p_two_sided=0.5,
                                                                                   mean_margin_change=0.12))))
    (run / 'metrics/report.json').write_text(json.dumps(dict(counters=dict(
        generations=[dict(optimizer_step=0, label='baseline', skipped='sealed'), entry(896, 'mid', 2), entry(3360, 'final', 1)],
        periodic=[dict(optimizer_step=224, degraded=False, comparison=dict(overall=dict(n=266, top1=121, baseline_top1=120,
                                                                                         gained=3, lost=2, p_two_sided=1.0,
                                                                                         mean_margin_change=0.01)))]))))
    summary = reporting.build_summary(run)
    assert [e['step'] for e in summary['quality']['capability_full']] == [896, 3360]
    text = reporting.handoff_text(summary)
    assert '| 896 | mid | 118/266 | 120 | 4 | 2 |' in text and '| 224 | periodic | 121/266 |' in text
    assert 'Lower CE without capability gains' in text and 'five-hour' not in text
