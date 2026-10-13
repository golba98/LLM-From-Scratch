"""Rendering tests for the read-only Rich dashboard over synthetic run directories."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from codexa_workspace.monitoring import reader as run_data, render
@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    # pytest's tmp dir is a small tmpfs; disk alerts have their own explicit test below.
    monkeypatch.setattr(run_data, 'disk_free', lambda path: 220 * 1024**3)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def run(tmp_path: Path, *, state='RUNNING', phase='TRAINING', step=400, total=1792, free_gpu=2.6 * 1024**3,
        terminal=None, reason=None, step_seconds=8.5, remaining=12000, heartbeat_age=2.0, gates=None) -> Path:
    r = tmp_path / 'codexa-934m-test'
    now = time.time()
    write(r / 'config/job.json', dict(job_sha256='ab' * 32, wakeup=dict(provider='codex'), spec=dict(
        model=dict(context_length=2048), training=dict(max_steps=total, warmup_steps=20, gradient_accumulation_steps=32,
        micro_batch_size=1), evaluation=dict(cadence_updates=64), optimization=dict(minimum_headroom_bytes=int(1.5 * 1024**3)),
        guards=dict(gradient_norm_threshold=10., source_validation_CE_increase=.1),
        initialization=dict(path='/base/latest.pt', sha256='80' * 32), tokenizer=dict(sha256='a7' * 32))))
    write(r / 'supervisor/state.json', dict(state=state, failure_reason=reason, trainer=dict(pid=1),
                                            supervisor=dict(pid=2, unit='codexa-train-x')))
    if gates:
        write(r / 'config/launch-gates.json', dict(passed=False, failed=gates))
    if state not in ('BLOCKED',):
        write(r / 'metrics/budget.json', dict(seconds=18000, started_utc=now - (18000 - remaining),
              deadline_utc=now + remaining, reserve_seconds=600, elapsed_seconds=18000 - remaining))
        write(r / 'metrics/heartbeat.json', dict(phase=phase, optimizer_step=step, ts_utc=now - heartbeat_age,
              phase_started_utc=now - 5, last_step_utc=now - 3, run_id='trainer-1'))
        write(r / 'metrics/progress.json', dict(state='RUNNING', step=step, total_steps=total, training_loss=2.95,
              learning_rate=1.9e-5, tokens_per_second=8000., total_tokens_seen=step * 65536, gradient_norm=.43,
              free_gpu_bytes=free_gpu, peak_vram_bytes=13 * 1024**3, next_checkpoint_step=(step // 64 + 1) * 64,
              activity=phase))
        metrics = []
        for i in range(1, step + 1):
            stamp = datetime.fromtimestamp(now - (step - i) * step_seconds, timezone.utc).isoformat()
            metrics.append(json.dumps(dict(optimizer_step=i, training_loss=3.1 - i * 1e-4, tokens_per_second=8000.,
                                           step_time_seconds=step_seconds, timestamp_utc=stamp)))
        write(r / 'metrics/train_metrics.jsonl', '\n'.join(metrics) + '\n')
        write(r / 'metrics/validation-strata.jsonl', '\n'.join(json.dumps(dict(optimizer_step=s, weighted_loss=3.113 - s * 1e-5,
              sources=dict(fineweb_edu=3.162 - s * 1e-5, wikipedia=3.036 - s * 1e-5))) for s in range(0, step + 1, 64)) + '\n')
        slot = r / 'checkpoints/slot-0/checkpoint.pt'
        write(slot, 'x')
        record = dict(path='slot-0/checkpoint.pt', optimizer_step=step // 64 * 64, verified=True, sha256='c3' * 32,
                      bytes=5636070865, write_verify_seconds=12.)
        write(r / 'checkpoints/recovery-index.json', dict(latest=record, best_validation=record))
    if terminal:
        write(r / 'supervisor/completion.json', dict(event_id='e', run_id=r.name, terminal_state=terminal, reason=reason,
              optimizer_steps=step, processed_tokens=step * 65536, elapsed_seconds=3600.,
              final_checkpoint_verification='verified' if terminal != 'FAILED' else 'not_available',
              final_checkpoint=dict(path=str(r / 'checkpoints/slot-0/checkpoint.pt'), optimizer_step=step) if terminal != 'FAILED' else None,
              last_verified_checkpoint=dict(optimizer_step=step // 64 * 64),
              validation=dict(baseline=3.113, best=3.109, final=3.109), errors=['RuntimeError: CUDA out of memory'] if terminal == 'FAILED' else []))
        write(r / 'supervisor/desktop-notification.json', dict(terminal_event_id='e', notification_state='delivered', backend='notify-send'))
        write(r / 'supervisor/ai-wakeup.json', dict(terminal_event_id='e', status='submitted', provider='codex'))
    return r


def text(run_dir, **options):
    options.setdefault('gpu', False)
    return render.render_text(run_dir, **options)


SCENARIOS = {
    'running': dict(),
    'validating': dict(phase='VALIDATING', state='EARLY_EVALUATION'),
    'checkpointing': dict(phase='CHECKPOINTING', state='CHECKPOINTING'),
    'early_stopped': dict(state='EARLY_STOPPED', terminal='EARLY_STOPPED', reason='MEMORY_HEADROOM_GATE', step=129),
    'completed': dict(state='COMPLETED', terminal='COMPLETED', reason='STEP_CAP', step=1792),
    'failed': dict(state='FAILED', terminal='FAILED', reason='CUDA_OOM', step=300),
    'interrupted': dict(state='INTERRUPTED', terminal='INTERRUPTED', reason='HOST_RESTART', step=640),
    'blocked': dict(state='BLOCKED', reason='launch gates failed', gates=['disk_capacity', 'gpu_vram']),
}
BADGE = dict(running='RUNNING', validating='VALIDATING', checkpointing='CHECKPOINTING', early_stopped='EARLY STOP',
             completed='COMPLETED', failed='FAILED', interrupted='INTERRUPTED', blocked='BLOCKED')


@pytest.mark.parametrize('name', SCENARIOS)
@pytest.mark.parametrize('width', [50, 80, 120])
def test_every_state_renders_within_width(tmp_path, name, width):
    output = text(run(tmp_path, **SCENARIOS[name]), width=width)
    assert all(len(line) <= width for line in output.splitlines())
    assert BADGE[name] in output and 'never stops training' in output.replace('\n', ' ')


def test_running_progress_eta_and_metrics(tmp_path):
    r = run(tmp_path, step=448, step_seconds=8.5, remaining=14000)
    output = text(r, width=120)
    assert '448 / 1,792' in output and '25.0%' in output
    derived = run_data.derive(run_data.RunReader(r, gpu=False).read())
    assert derived['per_step'] == pytest.approx(8.5, rel=1e-3)
    assert derived['eta'] == pytest.approx((1792 - 448) * 8.5, rel=1e-3)
    assert derived['projected_stop'] is None and 'No alerts' in output
    assert 'FineWeb-Edu' in output and 'Wikipedia' in output and 'update 448' in output
    assert 'one final-outcome notification will be sent by the supervisor' in output


def test_deadline_projection_and_vram_alerts(tmp_path):
    r = run(tmp_path, step=400, step_seconds=10., remaining=9000, free_gpu=1.2 * 1024**3)
    output = text(r, width=120)
    derived = run_data.derive(run_data.RunReader(r, gpu=False).read())
    assert abs(derived['projected_stop'] - (400 + int((9000 - 600) / 10.))) <= 1
    assert 'will stop near update 1,2' in output and '(TRAINING_DEADLINE)' in output
    assert 'Free VRAM 1.20 GiB near the 1.5 GiB stop guard' in output


def test_stale_heartbeat_is_flagged(tmp_path):
    output = text(run(tmp_path, heartbeat_age=400), width=120)
    assert 'NO HEARTBEAT' in output and 'Heartbeat stale' in output


def test_terminal_banners(tmp_path):
    done = text(run(tmp_path, **SCENARIOS['completed']), width=120)
    assert 'Run finished' in done and '✔ COMPLETED' in done and 'verified' in done
    assert 'Desktop notification delivered once' in done and 'codex analysis requested' in done
    failed = text(run(tmp_path / 'f', **SCENARIOS['failed']), width=120)
    assert '✖ FAILED' in failed and 'CUDA_OOM' in failed and 'CUDA out of memory' in failed
    blocked = text(run(tmp_path / 'b', **SCENARIOS['blocked']), width=120)
    assert 'Launch blocked: disk_capacity, gpu_vram' in blocked


def test_small_height_drops_secondary_panels_first(tmp_path):
    r = run(tmp_path)
    tall = text(r, width=120)
    short = text(r, width=120, height=22)
    assert len(short.splitlines()) <= 22 < len(tall.splitlines())
    assert 'RUNNING' in short and 'Alerts' in short


def test_details_view_exposes_raw_operational_fields(tmp_path):
    output = text(run(tmp_path), width=140, details=True)
    assert 'Details (raw)' in output and 'abababab' in output and 'codexa-train-x' in output
    assert 'Role latest' in output and '/base/latest.pt' in output


def test_low_disk_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(run_data, 'disk_free', lambda path: 105 * 1024**3)
    assert 'near the 100 GiB recovery reserve' in text(run(tmp_path), width=120)


def test_missing_files_render_gracefully(tmp_path):
    empty = tmp_path / 'nothing-yet'
    empty.mkdir()
    output = text(empty, width=80)
    assert 'NOT PREPARED' in output


def test_incremental_tail_reads_only_new_lines(tmp_path):
    r = run(tmp_path, step=64)
    reader = run_data.RunReader(r, gpu=False)
    assert len(reader.read()['metrics']) == 64
    with (r / 'metrics/train_metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(optimizer_step=65, training_loss=3.0, tokens_per_second=8000.,
                                     step_time_seconds=8.5, timestamp_utc=datetime.now(timezone.utc).isoformat())) + '\n')
        stream.write('{"optimizer_step": 66, "partial')            # torn write must be ignored for now
    rows = reader.read()['metrics']
    assert rows[-1]['optimizer_step'] == 65 and reader.metrics.offset < (r / 'metrics/train_metrics.jsonl').stat().st_size


def test_eta_ignores_resume_downtime_gap(tmp_path):
    r = run(tmp_path, step=134, step_seconds=8.5, remaining=15000)
    rows = [json.loads(line) for line in (r / 'metrics/train_metrics.jsonl').read_text().splitlines()]
    for row in rows[:129]:   # updates before a 21-minute stop/resume gap
        row['timestamp_utc'] = datetime.fromtimestamp(epoch_of(row) - 1260, timezone.utc).isoformat()
    (r / 'metrics/train_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    derived = run_data.derive(run_data.RunReader(r, gpu=False).read())
    assert derived['per_step'] == pytest.approx(8.5, rel=1e-3) and derived['projected_stop'] is None


def epoch_of(row):
    return datetime.fromisoformat(row['timestamp_utc']).timestamp()


# Every training start shows this dashboard: the trainer's viewer hook opens (or reuses) one 120x30 window.
from codexa_workspace.monitoring import window as dashboard_window
@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path / 'runtime'
    path.mkdir()
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(path))
    return path


def test_attach_opens_one_window_in_its_own_unit(tmp_path, runtime):
    import subprocess
    r = run(tmp_path)
    marker = r / 'metrics/progress.attached'
    seen = []

    def runner(command, **_):
        seen.append(command)
        marker.write_text(json.dumps(dict(pid=os.getpid())))   # the dashboard's first frame
        return subprocess.CompletedProcess(command, 0, '', '')

    assert dashboard_window.attach(r / 'metrics/progress.json', marker, runner=runner, wait_forever=False) == os.getpid()
    command = seen[0]
    assert command[:4] == ['systemd-run', '--user', '--collect', '--no-block'] and command[4].startswith('--unit=codexa-dashboard-')
    assert 'initial_window_width=120c' in command and 'initial_window_height=30c' in command
    assert command[-2:] == ['--marker', str(marker)] and str(dashboard_window.SCRIPT) in command


def test_attach_reuses_the_open_dashboard(tmp_path, runtime):
    r = run(tmp_path)
    dashboard_window.register(r)                               # this process plays the open dashboard
    marker = r / 'metrics/progress.attached'
    runner = lambda *a, **k: pytest.fail('a second window must not be opened')
    assert dashboard_window.attach(r / 'metrics/progress.json', marker, runner=runner, wait_forever=False) == os.getpid()
    assert json.loads(marker.read_text()) == dict(pid=os.getpid(), reused=True)
    dashboard_window.unregister(r)
    assert dashboard_window.live_dashboard(r) is None


def test_attach_fails_when_the_window_never_appears(tmp_path, runtime):
    import subprocess
    r = run(tmp_path)
    runner = lambda command, **_: subprocess.CompletedProcess(command, 0, '', '')
    with pytest.raises(SystemExit, match='did not attach'):
        dashboard_window.attach(r / 'metrics/progress.json', r / 'metrics/progress.attached', runner=runner,
                                timeout=0.3, wait_forever=False)


def test_stale_registration_is_ignored(tmp_path, runtime):
    r = run(tmp_path)
    dashboard_window.registry_path(r).write_text(json.dumps(dict(pid=os.getpid(), start_ticks=-1)))
    assert dashboard_window.live_dashboard(r) is None
