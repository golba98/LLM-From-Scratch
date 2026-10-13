"""10-hour foundation continuation: order, timing, freezing, authorization, launch and resume.

Mocks and synthetic fixtures only: no model, no optimizer, no systemd unit, no GPU.
"""
import json

import numpy as np
import pytest
import torch

from codexa_workspace import foundation_continuation as job

MEASURED = dict(steps=dict(median=8.11, p95=8.19, p99=8.59, maximum=9.1), startup_seconds=99, validation_seconds=25,
                periodic_seconds=60, full_seconds=400, checkpoint_seconds=24)


# ---------------------------------------------------------------- data order

def test_continuation_order_extends_the_predecessor_permutation_disjointly(tmp_path):
    eligible = np.arange(10, 1010, dtype='<u4')
    np.save(tmp_path / 'train_window_ids.npy', eligible)
    order, prefix = job.continuation_order(tmp_path, dict(seed=42), offset=100, count=64)
    generator = torch.Generator().manual_seed(42)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    permutation = torch.randperm(len(eligible), generator=generator).numpy()
    assert np.array_equal(prefix, eligible[permutation[:100]]) and np.array_equal(order, eligible[permutation[100:164]])
    assert not np.intersect1d(order, prefix).size and len(np.unique(order)) == 64
    with pytest.raises(ValueError, match='exceeds'):
        job.continuation_order(tmp_path, dict(seed=42), offset=990, count=64)


# ---------------------------------------------------------------- timing model

def test_update_cap_fits_ten_hours_with_slack_and_final_work_in_reserve():
    steps = job.choose_max_steps(MEASURED, reserve_seconds=900)
    plan = job.timing_model(MEASURED, max_steps=steps, reserve_seconds=900)
    assert steps % job.CADENCE_UPDATES == 0 and plan['fits']
    assert plan['planned_total_seconds'] <= 0.9 * job.WALL_CLOCK_SECONDS
    bigger = job.timing_model(MEASURED, max_steps=steps + job.CADENCE_UPDATES, reserve_seconds=900)
    assert bigger['slack_seconds'] < 0.1 * job.WALL_CLOCK_SECONDS
    assert plan['final_needed_seconds'] <= plan['reserve_seconds']
    # Optimizer time alone is never mistaken for elapsed time: every overhead is in the total.
    assert plan['planned_total_seconds'] > steps * MEASURED['steps']['p95'] + 900


def test_impossible_budget_and_undersized_reserve_are_refused():
    with pytest.raises(ValueError, match='No update cap'):
        job.choose_max_steps(dict(MEASURED, steps=dict(p95=200.0)), reserve_seconds=900)
    assert not job.timing_model(MEASURED, max_steps=224, reserve_seconds=100)['fits']   # final work > reserve


# ---------------------------------------------------------------- frozen job and authorization

def frozen(spec=None, **overrides):
    document = dict(format=job.JOB_FORMAT, run_id=job.RUN_ID,
                    spec=spec or dict(run_name=job.RUN_ID, checkpoint=dict(estimated_checkpoint_bytes=1)), code_identity={},
                    authorization='NOT AUTHORIZED: test', **overrides)
    document['spec_sha256'] = job.fingerprint(document['spec'])
    document['job_sha256'] = job.fingerprint(document)
    return document


def test_tampered_job_is_refused():
    document = frozen()
    assert job.verify_job(document)
    document['spec']['run_name'] = 'other'
    with pytest.raises(job.LaunchRefused, match='modified'):
        job.verify_job(document)


def test_approval_is_bound_to_the_exact_hash_and_never_overwritten(tmp_path):
    document = frozen()
    (tmp_path / 'config').mkdir()
    with pytest.raises(ValueError, match='explicit'):
        job.write_approval(tmp_path, document, statement='  ', written_utc='t')
    with pytest.raises(job.LaunchRefused, match='approval'):
        job.approval_valid(tmp_path, document)
    job.write_approval(tmp_path, document, statement='I authorize this 10-hour run', written_utc='t')
    assert job.approval_valid(tmp_path, document)['job_sha256'] == document['job_sha256']
    with pytest.raises(FileExistsError):
        job.write_approval(tmp_path, document, statement='again', written_utc='t')
    other = frozen(evidence='different')
    with pytest.raises(job.LaunchRefused):
        job.approval_valid(tmp_path, other)                     # the SFT or any other grant never transfers


def test_freeze_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'code_identity', lambda root=job.ROOT: {'x': 'y'})
    first = job.freeze(tmp_path, dict(run_name=job.RUN_ID), evidence={})
    assert first['wakeup'] == dict(provider='claude') and first['authorization'].startswith('NOT AUTHORIZED')
    with pytest.raises(FileExistsError):
        job.freeze(tmp_path, dict(run_name=job.RUN_ID), evidence={})


# ---------------------------------------------------------------- launch path

@pytest.fixture
def world(tmp_path, monkeypatch):
    run_dir = tmp_path / 'outputs/training' / job.RUN_ID
    monkeypatch.setattr(job, 'run_directory', lambda: run_dir)
    monkeypatch.delenv('CODEXA_WAKEUP_AGENT', raising=False)
    prep = tmp_path / 'prep'
    (prep / 'config').mkdir(parents=True)
    document = frozen()
    calls = []

    def launcher(run_dir, root, *, unit, python, script, environment):
        calls.append(dict(run_dir=run_dir, unit=unit, script=str(script), environment=environment))
        return ['systemd-run', '--user', f'--unit={unit}']
    return dict(prep=prep, job=document, run_dir=run_dir, calls=calls, launcher=launcher)


def launch(world, **options):
    return job.launch_authorized(world['prep'], world['job'], provider='claude', written_utc='t',
                                 launcher=world['launcher'],
                                 readiness=options.pop('readiness', lambda: dict(verdict=job.READY)), **options)


def test_launch_refuses_without_the_users_grant(world):
    with pytest.raises(job.LaunchRefused, match='approval'):
        launch(world)
    assert not world['calls'] and not world['run_dir'].exists()


def test_launch_refuses_unless_readiness_is_ready(world):
    job.write_approval(world['prep'], world['job'], statement='go', written_utc='t')
    with pytest.raises(job.LaunchRefused, match='CONDITIONALLY'):
        launch(world, readiness=lambda: dict(verdict=job.CONDITIONAL, blocking=['launch-time startup_free_vram']))
    assert not world['calls']


def test_launch_refuses_inside_a_wakeup_agent_and_an_existing_namespace(world, monkeypatch):
    job.write_approval(world['prep'], world['job'], statement='go', written_utc='t')
    monkeypatch.setenv('CODEXA_WAKEUP_AGENT', '1')
    with pytest.raises(job.LaunchRefused, match='wakeup'):
        launch(world)
    monkeypatch.delenv('CODEXA_WAKEUP_AGENT')
    world['run_dir'].mkdir(parents=True)
    with pytest.raises(job.LaunchRefused, match='already exists'):
        launch(world)
    assert not world['calls']


def test_authorized_launch_writes_the_supervised_unit(world):
    job.write_approval(world['prep'], world['job'], statement='go', written_utc='t')
    result = launch(world)
    run_dir = world['run_dir']
    launch_doc = json.loads((run_dir / 'supervisor/launch.json').read_text())
    assert result['launched'] and world['calls'][0]['unit'] == f'codexa-train-{job.RUN_ID}'
    assert world['calls'][0]['script'].endswith('scripts/training_supervisor.py')
    assert world['calls'][0]['environment']['CODEXA_TRAINING_ENVIRONMENT'] == 'development'
    assert launch_doc['command'][-4:] == ['codexa_workspace.foundation_continuation', 'train', '--run-dir', str(run_dir)]
    assert '--resume' not in launch_doc['command'] and launch_doc['deadline_seconds'] == 36000
    assert launch_doc['policy'] == dict(deadline_grace=120) and launch_doc['wakeup'] == dict(provider='claude')
    assert launch_doc['execution_class'] == 'production' and launch_doc['agent_window'] is True
    viewer = json.loads(launch_doc['environment']['LLM_TRAINING_VIEWER_COMMAND'])
    assert viewer[-2:] == [viewer[-2], '--attach'] and viewer[-2].endswith('training_dashboard.py')
    assert json.loads((run_dir / 'config/job.json').read_text())['job_sha256'] == world['job']['job_sha256']
    assert json.loads((run_dir / 'config/approval.json').read_text())['job_sha256'] == world['job']['job_sha256']


# ---------------------------------------------------------------- resume

def stopped_run(world, *, state='EARLY_STOPPED', reason='MEMORY_HEADROOM_GATE', remaining=20000.0):
    job.write_approval(world['prep'], world['job'], statement='go', written_utc='t')
    launch(world)
    run_dir = world['run_dir']
    (run_dir / 'supervisor/state.json').write_text(json.dumps(dict(state=state)))
    (run_dir / 'supervisor/completion.json').write_text(json.dumps(dict(reason=reason)))
    (run_dir / 'metrics').mkdir(exist_ok=True)
    import time
    now = time.time()
    (run_dir / 'metrics/budget.json').write_text(json.dumps(dict(deadline_utc=now + remaining, remaining_seconds=remaining,
                                                                  last_recorded_utc=now, reserve_seconds=900)))
    return run_dir


def resume(world, monkeypatch, *, verified=True, resources_ok=True, confirm=job.RUN_ID):
    from codexa_workspace import training_supervisor
    monkeypatch.setattr(job, 'code_identity', lambda root=job.ROOT: {})
    monkeypatch.setattr(training_supervisor, 'recovery_role',
                        lambda run_dir, role, **k: dict(verified=verified, optimizer_step=448) if verified is not None else None)
    monkeypatch.setattr(job, 'gates', lambda spec: dict(startup_free_vram_mib=13928, available_ram_bytes=16 * 1024**3,
                                                        free_disk_bytes=180 * 1024**3, recovery_reserve_bytes=100 * 1024**3,
                                                        in_run_free_vram_bytes=1536 * 1024**2))
    measured = dict(vram=dict(free_mib=14000 if resources_ok else 9000), available_ram_bytes=20 * 1024**3,
                    free_disk_bytes=300 * 1024**3)
    return job.resume_authorized(world['job'], confirm=confirm, provider='claude', resources=lambda: measured,
                                 written_utc='t', launcher=world['launcher'])


def test_resume_only_after_a_resource_stop_with_a_verified_checkpoint(world, monkeypatch):
    stopped_run(world, state='COMPLETED', reason='STEP_CAP')
    with pytest.raises(job.LaunchRefused, match='Not resumable'):
        resume(world, monkeypatch)


def test_resume_refusals_and_success(world, monkeypatch):
    run_dir = stopped_run(world)
    with pytest.raises(job.LaunchRefused, match='confirm-run'):
        resume(world, monkeypatch, confirm='other')
    with pytest.raises(job.LaunchRefused, match='verified latest'):
        resume(world, monkeypatch, verified=False)
    with pytest.raises(job.LaunchRefused, match='Resource gates'):
        resume(world, monkeypatch, resources_ok=False)
    result = resume(world, monkeypatch)
    assert result['resumed'] and result['from_step'] == 448
    assert '--resume' in json.loads((run_dir / 'supervisor/launch.json').read_text())['command']
    assert any((run_dir / 'supervisor/segments').iterdir())


def test_resume_refused_when_the_original_budget_is_spent(world, monkeypatch):
    stopped_run(world, remaining=1000.0)
    with pytest.raises(job.LaunchRefused, match='nearly spent'):
        resume(world, monkeypatch)


# ---------------------------------------------------------------- policy checks

def test_lowered_gates_and_relaxed_guards_are_rejected():
    from codexa_workspace.sft_pilot_runtime import validate_gates
    spec = dict(optimization=dict(minimum_headroom_bytes=1536 * 1024**2), checkpoint=dict(reserve_bytes=100 * 1024**3,
                minimum_save_seconds=120), guards=dict(gradient_norm_threshold=10.0, gradient_norm_patience=3,
                source_validation_CE_increase=0.1, validation_patience=2, capability_patience=2))
    assert validate_gates(job.gates(spec)) and job.stop_policy(spec)['passed']
    with pytest.raises(ValueError, match='lowered'):
        validate_gates(job.gates(dict(spec, optimization=dict(minimum_headroom_bytes=1024**3))))
    relaxed = dict(spec, guards=dict(spec['guards'], capability_patience=5))
    assert not job.stop_policy(relaxed)['passed']


def test_verify_run_refuses_changed_code_or_missing_grant(tmp_path, monkeypatch):
    document = frozen()
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/job.json').write_text(json.dumps(document))
    monkeypatch.setattr(job, 'code_identity', lambda root=job.ROOT: {'changed': 'yes'})
    with pytest.raises(job.LaunchRefused, match='Code identity'):
        job.verify_run(tmp_path)
    monkeypatch.setattr(job, 'code_identity', lambda root=job.ROOT: {})
    with pytest.raises(job.LaunchRefused, match='approval'):
        job.verify_run(tmp_path)


def test_corpus_identity_ignores_the_unstable_btrfs_device_number(tmp_path, monkeypatch):
    import os
    corpus = tmp_path / 'train.bin'
    corpus.write_bytes(b'\0' * 64)
    stat = os.stat(corpus)
    identity = dict(device=stat.st_dev + 4, inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    ctime_ns=stat.st_ctime_ns)
    order = tmp_path / 'order.npy'
    np.save(order, np.array([5, 6], dtype='<u4'))
    for name, values in (('validation.npy', [1]), ('historical_window_ids.npy', [2]), ('train_window_ids.npy', [5, 6, 7]),
                         ('predecessor.npy', [7])):
        np.save(tmp_path / name, np.array(values, dtype='<u4'))
    (tmp_path / 'sources.json').write_text('{}')
    monkeypatch.setattr(job, 'PREPARED', tmp_path)
    monkeypatch.setattr(job, 'predecessor_spec', lambda: dict(data=dict(train_window_order=str(tmp_path / 'predecessor.npy'))))
    spec = dict(training=dict(max_steps=1, gradient_accumulation_steps=2),
                data=dict(corpus_path=str(corpus), corpus_input_identity=identity, train_window_order=str(order),
                          train_window_order_sha256=job.file_sha256(order), validation_window_ids=str(tmp_path / 'validation.npy'),
                          validation_window_ids_sha256=job.file_sha256(tmp_path / 'validation.npy'),
                          validation_sources=str(tmp_path / 'sources.json'),
                          validation_sources_sha256=job.file_sha256(tmp_path / 'sources.json')))
    result = job.verify_data(spec)
    assert result['passed'] and result['device']['informational'] is True
    corpus.write_bytes(b'\1' * 64)                                   # content and timestamps change
    assert not job.verify_data(spec)['checks']['corpus_identity']
