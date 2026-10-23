"""Tiny real CPU runs of the indexed-window pretraining controller."""

from dataclasses import asdict
import hashlib
import json
import os
import signal

import numpy as np
import pytest
import torch

from llm_architecture.model import LanguageModel
from llm_training.checkpointing import build_checkpoint_payload, SchedulerState
from llm_training.recovery import RecoveryStore
from llm_training.session import SessionBudget
from llm_training.training import TrainingState
import llm_training.pretrain_run as pretrain
from test_chat_runtime import project


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def job(tmp_path):
    config = project()
    torch.manual_seed(5)
    corpus = tmp_path / 'corpus.bin'
    np.random.default_rng(3).integers(0, 16, 4 * 64 + 1, dtype=np.uint16).tofile(corpus)
    order = np.random.default_rng(4).permutation(np.arange(0, 48))[:16].astype('<u4')
    np.save(tmp_path / 'order.npy', order)
    validation = np.array([50, 51, 52, 53], dtype='<u4')
    np.save(tmp_path / 'validation.npy', validation)
    (tmp_path / 'sources.json').write_text(json.dumps({'50': 'a', '51': 'a', '52': 'b', '53': 'b'}))
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('fixture')
    source = LanguageModel(config.model)
    initial = RecoveryStore(tmp_path / 'initial')
    initial.save(build_checkpoint_payload(model=source, optimizer=torch.optim.AdamW(source.parameters()),
        scaler=None, state=TrainingState(), scheduler=SchedulerState(0, 1, .01, .001), config=config,
        run_name='fixture', run_id='fixture', tokenizer_reference=None, tokenizer_sha256=None,
        resume_context={}))
    training = asdict(config.training)
    training.update(max_steps=8, gradient_accumulation_steps=2, warmup_steps=1, evaluation_interval=4,
                    checkpoint_interval=4)
    return dict(format='test', run_name='tiny', device='cpu', viewer_required=False,
        model=asdict(config.model), training=training,
        initialization=dict(path=str(initial.resolve()), sha256=digest(initial.resolve())),
        tokenizer=dict(path=str(tokenizer), sha256=digest(tokenizer)),
        data=dict(corpus_path=str(corpus), dtype='uint16', selection_sha256='selection',
                  train_window_order=str(tmp_path / 'order.npy'), train_window_order_sha256=digest(tmp_path / 'order.npy'),
                  validation_window_ids=str(tmp_path / 'validation.npy'),
                  validation_window_ids_sha256=digest(tmp_path / 'validation.npy'),
                  validation_sources=str(tmp_path / 'sources.json'),
                  validation_sources_sha256=digest(tmp_path / 'sources.json'),
                  source_loss_weights={'a': .6, 'b': .4}),
        optimization=dict(optimizer='adamw', betas=[.9, .95], epsilon=1e-8, minimum_learning_rate=.001,
                          gradient_checkpointing=False, loss_normalization='supervised_tokens',
                          minimum_headroom_bytes=0),
        evaluation=dict(cadence_updates=4, validation_batches=None, generation_steps=[4],
                        generation_reserve_seconds=0, validation_overrun_seconds=0),
        guards=dict(gradient_norm_threshold=10., gradient_norm_patience=3,
                    source_validation_CE_increase=.1, validation_patience=2),
        checkpoint=dict(backend='legacy', interval=4, minimum_save_seconds=.01, reserve_bytes=0,
                        estimated_checkpoint_bytes=0))


def run(spec, directory, **options):
    return pretrain.run_pretraining(spec, directory, SessionBudget.start(120, 10), heartbeat_interval=.05, **options)


def test_fresh_run_completes_with_verified_final_and_generation_schedule(job, tmp_path):
    calls = []
    def generation(model, target, deadline):
        calls.append(target.name)
        return dict(cases=1)
    report = run(job, tmp_path / 'run', generation_evaluator=generation)
    assert report['state'] == 'COMPLETED' and report['stop_reason'] == 'STEP_CAP'
    assert report['final_checkpoint_verified'] and report['final_checkpoint']['optimizer_step'] == 8
    assert calls == ['generation-baseline-step-00000', 'generation-mid-step-00004', 'generation-final-step-00008']
    assert [v['optimizer_step'] for v in report['counters']['validations']] == [0, 4, 8]
    heartbeat = json.loads((tmp_path / 'run/metrics/heartbeat.json').read_text())
    assert heartbeat['terminal_state'] == 'COMPLETED' and heartbeat['optimizer_step'] == 8
    progress = json.loads((tmp_path / 'run/metrics/progress.json').read_text())
    assert progress['state'] == 'COMPLETED' and progress['last_verified_checkpoint']['optimizer_step'] == 8
    assert json.loads((tmp_path / 'run/metrics/report.json').read_text())['state'] == 'COMPLETED'
    with pytest.raises(FileExistsError):
        run(job, tmp_path / 'run')


def test_interrupt_then_resume_is_bitwise_equal_and_keeps_budget(job, tmp_path):
    reference = run(job, tmp_path / 'reference')
    expected = torch.load(RecoveryStore(tmp_path / 'reference/checkpoints').resolve('final'), weights_only=False)

    def interrupt(model, target, deadline):
        if 'mid' in target.name:
            os.kill(os.getpid(), signal.SIGTERM)
        return {}
    report = run(job, tmp_path / 'split', generation_evaluator=interrupt)
    assert report['state'] == 'INTERRUPTED' and report['training_state']['optimizer_step'] == 4
    budget_document = json.loads((tmp_path / 'split/metrics/budget.json').read_text())
    budget = SessionBudget.resume(budget_document)
    assert budget.started_utc == budget_document['started_utc']
    resumed = pretrain.run_pretraining(job, tmp_path / 'split', budget, resume=True, heartbeat_interval=.05)
    assert resumed['state'] == 'COMPLETED' and resumed['run_id'] == report['run_id']
    actual = torch.load(RecoveryStore(tmp_path / 'split/checkpoints').resolve('final'), weights_only=False)
    for key, value in expected['model_state_dict'].items():
        assert torch.equal(value, actual['model_state_dict'][key]), key
    assert expected['training_state'] == actual['training_state']
    assert reference['final_checkpoint_verified'] and resumed['final_checkpoint_verified']


def test_gradient_guard_early_stops_with_verified_state(job, tmp_path):
    job['guards'].update(gradient_norm_threshold=1e-12, gradient_norm_patience=2)
    report = run(job, tmp_path / 'run')
    assert report['state'] == 'EARLY_STOPPED' and report['stop_reason'] == 'GRADIENT_NORM_GUARD'
    assert report['training_state']['optimizer_step'] == 2 and report['final_checkpoint_verified']


def test_source_validation_guard_requires_patience(job, tmp_path):
    job['training'].update(max_steps=16)
    job['data']['train_window_order'] = str(tmp_path / 'order32.npy')
    np.save(tmp_path / 'order32.npy', np.arange(0, 32, dtype='<u4')[::-1].copy())
    job['data']['train_window_order_sha256'] = digest(tmp_path / 'order32.npy')
    job['guards'].update(source_validation_CE_increase=-10.)
    report = run(job, tmp_path / 'run')
    assert report['stop_reason'] == 'SOURCE_CE_DETERIORATION' and report['training_state']['optimizer_step'] == 8


def test_disk_reserve_stops_without_publishing(job, tmp_path):
    job['checkpoint'].update(reserve_bytes=1 << 62)
    report = run(job, tmp_path / 'run')
    assert report['state'] == 'EARLY_STOPPED' and report['stop_reason'] == 'DISK_HEADROOM'
    assert not report['final_checkpoint_verified'] and report['checkpoints'] == []


def test_checkpoint_failure_and_nonfinite_are_failures(job, tmp_path, monkeypatch):
    monkeypatch.setattr(RecoveryStore, 'save', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(pretrain.CheckpointPublicationError):
        run(job, tmp_path / 'checkpoint')
    report = json.loads((tmp_path / 'checkpoint/metrics/report.json').read_text())
    assert report['state'] == 'FAILED' and report['stop_reason'] == 'CHECKPOINT_FAILURE'
    monkeypatch.undo()

    def explode(*args, **kwargs):
        raise FloatingPointError('Non-finite training loss: nan.')
    monkeypatch.setattr(pretrain, 'train_model', explode)
    with pytest.raises(FloatingPointError):
        run(job, tmp_path / 'nan')
    report = json.loads((tmp_path / 'nan/metrics/report.json').read_text())
    assert report['state'] == 'FAILED' and report['error_type'] == 'FloatingPointError'
    assert json.loads((tmp_path / 'nan/metrics/heartbeat.json').read_text())['terminal_state'] == 'FAILED'


def test_order_drift_and_wakeup_agent_are_refused(job, tmp_path, monkeypatch):
    job['data']['train_window_order_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='order fingerprint'):
        run(job, tmp_path / 'drift')
    monkeypatch.setenv(pretrain.WAKEUP_ENVIRONMENT, '1')
    with pytest.raises(RuntimeError, match='wakeup'):
        run(job, tmp_path / 'agent')


def test_diagnostic_generation_errors_are_recorded_not_fatal(job, tmp_path):
    def broken(model, target, deadline):
        raise ZeroDivisionError('aggregate bug')
    report = run(job, tmp_path / 'run', generation_evaluator=broken)
    assert report['state'] == 'COMPLETED' and report['final_checkpoint_verified']
    assert [g['error'] for g in report['counters']['generations']] == ['ZeroDivisionError: aggregate bug'] * 3

    def nonfinite(model, target, deadline):
        raise FloatingPointError('Nonfinite free-generation logits')
    with pytest.raises(FloatingPointError):
        run(job, tmp_path / 'nan', generation_evaluator=nonfinite)


# ---------------------------------------------------------------- per-update telemetry and capability guard

from llm_training import telemetry as records


def effective(directory):
    return records.read_telemetry(directory / 'metrics' / records.FILENAME)


def test_fresh_run_writes_complete_telemetry(job, tmp_path):
    report = run(job, tmp_path / 'run', generation_evaluator=lambda model, target, deadline: dict(cases=1))
    assert report['telemetry_status'] == 'complete' and report['telemetry_failures'] == []
    result = effective(tmp_path / 'run')
    kinds = [r['kind'] for r in result['records']]
    updates = [r for r in result['records'] if r['kind'] == 'update']
    assert [r['step'] for r in updates] == list(range(1, 9))
    assert all(r['learning_rate'] is not None and r['micro_step'] == 2 * r['step'] for r in updates)
    assert all(r['vram_driver_free_bytes'] is None for r in updates)          # CPU: unmeasured, not zero
    assert all(r['vram_peak_reserved_bytes'] is None and r['vram_measured'] is False for r in updates)
    assert all(r['gates']['headroom']['passed'] is None and r['gates']['decision'] is None for r in updates)
    assert all(r['gradient_norm_post_clip'] <= r['gradient_norm_pre_clip'] + 1e-9 for r in updates)
    assert [r['step'] for r in result['records'] if r['kind'] == 'evaluation:validation'] == [0, 4, 8]
    assert [r['step'] for r in result['records'] if r['kind'].startswith('evaluation:generation')] == [0, 4, 8]
    assert [(r['step'], r['role'].split('+')[0]) for r in result['records'] if r['kind'].startswith('checkpoint')] \
        == [(4, 'latest'), (8, 'final')]
    assert kinds[-1] == 'stop' and result['records'][-1]['reason'] == 'STEP_CAP'
    assert result['records'][-1]['final_checkpoint_verified'] is True


def test_resume_telemetry_reports_each_update_once(job, tmp_path):
    def interrupt(model, target, deadline):
        if 'mid' in target.name:
            os.kill(os.getpid(), signal.SIGTERM)
        return {}
    run(job, tmp_path / 'split', generation_evaluator=interrupt)
    budget = SessionBudget.resume(json.loads((tmp_path / 'split/metrics/budget.json').read_text()))
    pretrain.run_pretraining(job, tmp_path / 'split', budget, resume=True, heartbeat_interval=.05)
    result = effective(tmp_path / 'split')
    assert result['attempts'] == [1, 2]
    assert result['sessions'][1]['resumed'] is True and result['sessions'][1]['resumed_from_step'] == 4
    assert [r['step'] for r in result['records'] if r['kind'] == 'update'] == list(range(1, 9))
    stops = [r for r in result['records'] if r['kind'] == 'stop']
    assert [s['state'] for s in stops] == ['INTERRUPTED', 'COMPLETED']        # different steps: both kept


def test_telemetry_failure_is_reported_never_fatal(job, tmp_path, monkeypatch):
    monkeypatch.setattr(records.TelemetryWriter, 'update',
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    report = run(job, tmp_path / 'run')
    assert report['state'] == 'COMPLETED' and report['final_checkpoint_verified']
    assert report['telemetry_status'] == 'degraded' and len(report['telemetry_failures']) == 8
    assert records.summarize(tmp_path / 'run/metrics/telemetry.jsonl')['telemetry_failures'][0].startswith('update')
    assert records.summarize(tmp_path / 'run/metrics/telemetry.jsonl')['status'] == 'degraded'


def test_telemetry_that_cannot_open_is_unavailable_not_silent(job, tmp_path, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise PermissionError('read-only metrics directory')
    monkeypatch.setattr(records.TelemetryWriter, '__init__', refuse)
    report = run(job, tmp_path / 'run')
    assert report['state'] == 'COMPLETED' and report['telemetry_status'] == 'unavailable'
    assert 'PermissionError' in report['telemetry_open_error']


def test_periodic_evaluator_runs_after_each_cadence_validation(job, tmp_path):
    calls = []
    def periodic(model, step, deadline):
        calls.append(step)
        return dict(accuracy=.5, degraded=False, rows=['not copied'])
    report = run(job, tmp_path / 'run', periodic_evaluator=periodic)
    assert calls == [4] and report['state'] == 'COMPLETED'                    # step 8 is the final step
    assert report['counters']['periodic'] == [dict(optimizer_step=4, accuracy=.5, degraded=False)]
    assert [r['step'] for r in effective(tmp_path / 'run')['records'] if r['kind'] == 'evaluation:periodic'] == [4]


def test_capability_guard_requires_consecutive_degradation(job, tmp_path):
    job['training'].update(max_steps=16)
    job['data']['train_window_order'] = str(tmp_path / 'order32.npy')
    np.save(tmp_path / 'order32.npy', np.arange(0, 32, dtype='<u4')[::-1].copy())
    job['data']['train_window_order_sha256'] = digest(tmp_path / 'order32.npy')
    job['guards']['capability_patience'] = 2
    noisy = iter([True, False, True])                                         # never twice in a row
    report = run(job, tmp_path / 'noisy', periodic_evaluator=lambda m, s, d: dict(degraded=next(noisy)))
    assert report['state'] == 'COMPLETED' and report['stop_reason'] == 'STEP_CAP'
    report = run(job, tmp_path / 'bad', periodic_evaluator=lambda m, s, d: dict(degraded=True))
    assert report['state'] == 'EARLY_STOPPED' and report['stop_reason'] == 'CAPABILITY_DEGRADATION'
    assert report['training_state']['optimizer_step'] == 8 and report['final_checkpoint_verified']
    gates = [r for r in effective(tmp_path / 'bad')['records'] if r['kind'] == 'gate:capability']
    assert gates and gates[0]['detail']['consecutive'] == 2


def test_periodic_evaluator_errors_are_recorded_not_fatal(job, tmp_path):
    report = run(job, tmp_path / 'run', periodic_evaluator=lambda m, s, d: 1 / 0)
    assert report['state'] == 'COMPLETED'
    assert report['counters']['periodic'][0]['error'].startswith('ZeroDivisionError')
