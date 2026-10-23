"""llm_training.telemetry: append-only per-update records, crash recovery and failure accounting.

Synthetic records only; no model, optimizer or backward pass.
"""
import json
import time

import torch

from llm_training import telemetry
from llm_training.training import TrainingMetrics


def metrics(step, *, loss=2.0, norm=0.5, tokens=40, seconds=0.1):
    return TrainingMetrics(
        run_name='r', run_id='r', micro_step=step * 8, optimizer_step=step, completed_epochs=0, training_loss=loss,
        validation_loss=None, learning_rate=5e-6, tokens_processed=tokens, total_tokens_seen=step * tokens,
        tokens_per_second=tokens / seconds, step_time_seconds=seconds, forward_backward_seconds=seconds * .9,
        optimizer_update_seconds=seconds * .1, gradient_norm=norm, allocated_vram_bytes=4_000,
        reserved_vram_bytes=5_000, peak_allocated_vram_bytes=6_000, peak_reserved_vram_bytes=7_000 + step,
        device='cuda', precision='bf16', timestamp_utc='t')


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_post_clip_norm_matches_torch():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    pre = float(torch.nn.utils.clip_grad_norm_([parameter], 1.0))
    assert abs(telemetry.post_clip_norm(pre, 1.0) - float(parameter.grad.norm())) < 1e-6
    assert telemetry.post_clip_norm(0.5, 1.0) == 0.5 and telemetry.post_clip_norm(None, 1.0) is None


def test_update_fields_rolling_throughput_and_gates(tmp_path):
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0, job_kind='foundation',
                                       recovery=dict(resume=False))
    writer.update(metrics(1, tokens=100, seconds=1.0), free_vram_bytes=None, gates=dict(decision=None))
    writer.update(metrics(2, norm=5.0, tokens=300, seconds=1.0), free_vram_bytes=3_000_000)
    writer.close()
    session, first, second = lines(tmp_path / 't.jsonl')
    assert session['job_kind'] == 'foundation' and session['recovery'] == dict(resume=False)
    assert first['vram_driver_free_bytes'] is None and first['gates'] == dict(decision=None)
    assert second['rolling_tokens_per_second'] == 200.0 and second['rolling_window_updates'] == 2
    assert second['clipped'] is True and abs(second['gradient_norm_post_clip'] - 1.0) < 1e-6


def test_duplicates_refused_and_nonfinite_is_null(tmp_path):
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0)
    assert writer.update(metrics(1, loss=float('nan'))) and not writer.update(metrics(1))
    writer.close()
    assert [r['kind'] for r in lines(tmp_path / 't.jsonl')] == ['session', 'update']
    assert lines(tmp_path / 't.jsonl')[1]['training_loss'] is None


def test_same_step_evaluations_of_different_schedules_are_distinct(tmp_path):
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0)
    writer.evaluation(4, dict(weighted_loss=3.0), seconds=1.0, schedule='validation')
    writer.evaluation(4, dict(accuracy=.5, rows=[1]), seconds=2.0, schedule='periodic')
    writer.close()
    kinds = [r['kind'] for r in telemetry.read_telemetry(tmp_path / 't.jsonl')['records']]
    assert kinds == ['evaluation:validation', 'evaluation:periodic']
    assert 'rows' not in lines(tmp_path / 't.jsonl')[2]['metrics']


def test_torn_tail_and_resume_rollback(tmp_path):
    path = tmp_path / 't.jsonl'
    first = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    for step in range(1, 11):
        first.update(metrics(step, loss=3.0))
    first.close()
    original = path.read_bytes()
    with path.open('ab') as stream:
        stream.write(b'{"format":"codexa-training-telemetry-v1","kind":"upd')
    second = telemetry.TelemetryWriter(path, run_id='r', clip=1.0, resumed_from_step=8)
    for step in range(9, 13):
        second.update(metrics(step, loss=1.0))
    second.close()
    assert path.read_bytes().startswith(original)
    result = telemetry.read_telemetry(path)
    updates = [r for r in result['records'] if r['kind'] == 'update']
    assert [r['step'] for r in updates] == list(range(1, 13)) and result['corrupt_lines'] == 1
    assert {r['step']: r['training_loss'] for r in updates}[10] == 1.0


def test_safe_records_failures_and_summary_status(tmp_path, monkeypatch):
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0)
    assert writer.safe('update', metrics(1)) and writer.status == 'complete'
    monkeypatch.setattr(telemetry.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('io error')))
    assert writer.safe('gate', 1, name='headroom', decision='stop') is False
    monkeypatch.undo()
    assert writer.status == 'degraded' and writer.failures[0].startswith('gate: OSError')
    writer.stop(state='COMPLETED', reason='STEP_CAP', step=1)
    writer.close()
    summary = telemetry.summarize(tmp_path / 't.jsonl')
    assert summary['status'] == 'degraded' and summary['stop']['reason'] == 'STEP_CAP'
    assert telemetry.summarize(tmp_path / 'missing.jsonl')['status'] == 'unavailable'


def test_fsync_batched_and_cheap(tmp_path, monkeypatch):
    calls = []
    real = telemetry.os.fsync
    monkeypatch.setattr(telemetry.os, 'fsync', lambda fd: (calls.append(fd), real(fd)))
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0, fsync_every=16)
    started = time.perf_counter()
    for step in range(1, 2001):
        writer.update(metrics(step), free_vram_bytes=1)
    elapsed = time.perf_counter() - started
    writer.close()
    assert len(calls) == 1 + 2000 // 16 + 1 and elapsed < 2.0


def test_off_cuda_vram_is_null_not_zero(tmp_path):
    from dataclasses import replace
    cpu = replace(metrics(1), device='cpu', allocated_vram_bytes=0, peak_reserved_vram_bytes=0)
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0)
    writer.update(cpu)
    writer.update(metrics(2))
    writer.close()
    cpu_row, cuda_row = lines(tmp_path / 't.jsonl')[1:]
    assert cpu_row['vram_allocated_bytes'] is None and cpu_row['vram_peak_reserved_bytes'] is None
    assert cpu_row['vram_measured'] is False and cuda_row['vram_measured'] is True
    assert cuda_row['vram_peak_reserved_bytes'] == 7_002


def test_status_distinguishes_unavailable_degraded_and_complete(tmp_path, monkeypatch):
    assert telemetry.summarize(tmp_path / 'none.jsonl')['status'] == 'unavailable'
    writer = telemetry.TelemetryWriter(tmp_path / 'early.jsonl', run_id='r', clip=1.0)
    writer.stop(state='FAILED', reason='ValueError', step=0)
    writer.close()
    assert telemetry.summarize(tmp_path / 'early.jsonl')['status'] == 'complete'          # observed, no updates
    writer = telemetry.TelemetryWriter(tmp_path / 'lost.jsonl', run_id='r', clip=1.0)
    monkeypatch.setattr(telemetry.TelemetryWriter, 'update', lambda self, *a, **k: (_ for _ in ()).throw(OSError('x')))
    writer.safe('update', metrics(1))
    writer.stop(state='COMPLETED', reason='STEP_CAP', step=1)
    writer.close()
    assert telemetry.summarize(tmp_path / 'lost.jsonl')['status'] == 'degraded'
