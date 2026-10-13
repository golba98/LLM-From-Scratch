"""Per-update training telemetry: append-only JSONL, crash recovery, resume de-duplication.

Synthetic records only: no model, no optimizer, no backward pass.
"""
import json
import time

from llm_training.training import TrainingMetrics

from codexa_workspace.training import telemetry


def metrics(step, *, loss=2.0, norm=0.5, tokens=40, total=None):
    return TrainingMetrics(
        run_name='r', run_id='r', micro_step=step * 8, optimizer_step=step, completed_epochs=0, training_loss=loss,
        validation_loss=None, learning_rate=5e-6, tokens_processed=tokens, total_tokens_seen=total or step * tokens,
        tokens_per_second=400.0, step_time_seconds=0.1, forward_backward_seconds=0.09, optimizer_update_seconds=0.01,
        gradient_norm=norm, allocated_vram_bytes=4_000, reserved_vram_bytes=5_000, peak_allocated_vram_bytes=6_000,
        peak_reserved_vram_bytes=7_000 + step, device='cuda', precision='bf16', timestamp_utc='t')


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_post_clip_norm_matches_torch_clipping_rule():
    assert telemetry.post_clip_norm(0.5, 1.0) == 0.5                        # below the threshold: unchanged
    assert abs(telemetry.post_clip_norm(5.0, 1.0) - 1.0) < 1e-6              # clipped to the threshold
    assert telemetry.post_clip_norm(None, 1.0) is None and telemetry.post_clip_norm(float('inf'), 1.0) is None
    assert telemetry.post_clip_norm(3.0, None) == 3.0
    import torch
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])                      # norm 5
    pre = float(torch.nn.utils.clip_grad_norm_([parameter], 1.0))
    assert abs(pre - 5.0) < 1e-6 and abs(telemetry.post_clip_norm(pre, 1.0) - float(parameter.grad.norm())) < 1e-6


def test_update_records_carry_every_engineering_field(tmp_path):
    path = tmp_path / 'telemetry.jsonl'
    writer = telemetry.TelemetryWriter(path, run_id='r', clip=1.0, job_kind='sft')
    assert writer.update(metrics(1, norm=5.0), free_vram_bytes=3_000_000)
    writer.close()
    session, update = lines(path)
    assert session['kind'] == 'session' and session['attempt'] == 1 and session['job_kind'] == 'sft'
    expected = {'optimizer_step', 'micro_step', 'training_loss', 'learning_rate', 'gradient_norm_pre_clip',
                'gradient_norm_post_clip', 'clipped', 'supervised_tokens', 'supervised_tokens_total',
                'vram_allocated_bytes', 'vram_reserved_bytes', 'vram_peak_allocated_bytes', 'vram_peak_reserved_bytes',
                'vram_driver_free_bytes', 'step_tokens_per_second', 'effective_tokens_per_second',
                'optimizer_seconds_total', 'evaluation_seconds_total', 'forward_backward_seconds'}
    assert expected <= set(update)
    assert update['gradient_norm_pre_clip'] == 5.0 and abs(update['gradient_norm_post_clip'] - 1.0) < 1e-6
    assert update['clipped'] is True and update['gradient_norm_post_clip_source'] == 'derived'
    assert (update['micro_step'], update['supervised_tokens'], update['vram_driver_free_bytes']) == (8, 40, 3_000_000)


def test_duplicate_update_in_one_attempt_is_not_written(tmp_path):
    path = tmp_path / 'telemetry.jsonl'
    writer = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    assert writer.update(metrics(1)) and not writer.update(metrics(1))
    writer.close()
    assert [r['kind'] for r in lines(path)] == ['session', 'update']


def test_nonfinite_values_are_strict_json_nulls(tmp_path):
    path = tmp_path / 'telemetry.jsonl'
    writer = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    writer.update(metrics(1, loss=float('nan'), norm=float('inf')))
    writer.close()
    update = lines(path)[1]                                                   # json.loads would reject NaN tokens
    assert update['training_loss'] is None and update['gradient_norm_pre_clip'] is None
    assert update['gradient_norm_post_clip'] is None


def test_torn_tail_is_preserved_and_the_next_attempt_starts_on_a_new_line(tmp_path):
    path = tmp_path / 'telemetry.jsonl'
    writer = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    for step in (1, 2):
        writer.update(metrics(step))
    writer.close()
    original = path.read_bytes()
    with path.open('ab') as stream:                                           # crash mid-record
        stream.write(b'{"format":"codexa-training-telemetry-v1","kind":"upd')
    second = telemetry.TelemetryWriter(path, run_id='r', clip=1.0, resumed_from_step=2)
    second.update(metrics(3))
    second.close()
    assert path.read_bytes().startswith(original)                             # nothing rewritten
    result = telemetry.read_telemetry(path)
    assert result['corrupt_lines'] == 1 and result['attempts'] == [1, 2]
    assert [r['step'] for r in result['records'] if r['kind'] == 'update'] == [1, 2, 3]
    assert telemetry.read_telemetry(path)['records'][0]['attempt'] == 1


def test_resume_rolls_back_unrecovered_updates_without_duplicates(tmp_path):
    path = tmp_path / 'telemetry.jsonl'
    first = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    for step in range(1, 11):                                                 # crash after 10; last checkpoint at 8
        first.update(metrics(step, loss=3.0))
    first.close()
    second = telemetry.TelemetryWriter(path, run_id='r', clip=1.0, resumed_from_step=8)
    for step in range(9, 13):
        second.update(metrics(step, loss=1.0))
    second.close()
    updates = [r for r in telemetry.read_telemetry(path)['records'] if r['kind'] == 'update']
    assert [r['step'] for r in updates] == list(range(1, 13))                 # each step exactly once
    assert {r['step']: r['attempt'] for r in updates}[9] == 2 and {r['step']: r['attempt'] for r in updates}[8] == 1
    assert {r['step']: r['training_loss'] for r in updates}[10] == 1.0          # the replayed update wins
    assert telemetry.read_telemetry(path)['raw_records'] == 2 + 10 + 4


def test_boundaries_are_fsynced_and_updates_batched(tmp_path, monkeypatch):
    calls = []
    real = telemetry.os.fsync
    monkeypatch.setattr(telemetry.os, 'fsync', lambda fd: (calls.append(fd), real(fd)))
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0, fsync_every=16)
    assert len(calls) == 1                                                    # session record
    for step in range(1, 33):
        writer.update(metrics(step))
    assert len(calls) == 3                                                    # every 16 updates
    writer.evaluation(32, dict(schedule='reduced', rows=[1, 2, 3], gold={'x': 1}), seconds=1.5)
    writer.checkpoint(32, role='intermediate', record=dict(path='p', sha256='s', bytes=1, verified=True), seconds=2.0)
    writer.stop(state='COMPLETED', reason='STEP_CAP', step=32)
    assert len(calls) == 6
    writer.close()
    evaluation = next(r for r in lines(tmp_path / 't.jsonl') if r['kind'].startswith('evaluation'))
    assert 'rows' not in evaluation['metrics'] and evaluation['metrics']['gold'] == {'x': 1}


def test_writing_is_cheap(tmp_path):
    writer = telemetry.TelemetryWriter(tmp_path / 't.jsonl', run_id='r', clip=1.0)
    started = time.perf_counter()
    for step in range(1, 2001):
        writer.update(metrics(step), free_vram_bytes=1)
    elapsed = time.perf_counter() - started
    writer.close()
    assert elapsed < 2.0, f'2,000 updates took {elapsed:.2f} s'               # well under 1 ms per update


def test_summary_reports_unavailable_when_absent_and_statistics_when_present(tmp_path):
    empty = telemetry.summarize(tmp_path / 'missing.jsonl')
    assert empty['available'] is False and empty['peak_reserved_vram_bytes'] is None and empty['stop'] is None
    path = tmp_path / 't.jsonl'
    writer = telemetry.TelemetryWriter(path, run_id='r', clip=1.0)
    for step, norm in enumerate([0.5, 5.0, 2.0], start=1):
        writer.update(metrics(step, loss=3.0 - step, norm=norm), free_vram_bytes=10 - step)
    writer.stop(state='COMPLETED', reason='STEP_CAP', step=3)
    writer.close()
    summary = telemetry.summarize(path)
    assert (summary['updates'], summary['last_step'], summary['clipped_updates']) == (3, 3, 2)
    assert (summary['first_loss'], summary['last_loss'], summary['min_driver_free_vram_bytes']) == (2.0, 0.0, 7)
    assert summary['peak_reserved_vram_bytes'] == 7_003 and summary['stop']['reason'] == 'STEP_CAP'


# ---------------------------------------------------------------- the SFT trainer with fakes
