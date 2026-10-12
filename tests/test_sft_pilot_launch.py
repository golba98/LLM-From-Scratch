"""The authorized SFT launch path, end to end, with a mocked process launcher and optimizer.

No systemd unit, trainer process, torch optimizer or backward pass is ever created: the
launcher, the trainer and the optimizer factory are recorders. Real checks (job hashes,
meta-device architecture, canonical resource gates, configuration approval) run unmodified.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from codexa_workspace import sft_phase4_data as data
from codexa_workspace import sft_pilot_launch as launch
from codexa_workspace import sft_pilot_runtime as rt
from codexa_workspace import sft_pilot_trainer as trainer
from codexa_workspace import training_supervisor

PHASE4 = data.BASE / '20261012T145429Z-643f4379'
needs_phase4 = pytest.mark.skipif(not (PHASE4 / 'manifest/phase4-manifest.json').exists(),
                                  reason='Phase 4 frozen manifest not present (ignored asset)')
READY = dict(verdict=launch.READY, blocking=[])
REVISIONS = {'LLM-Data': 'a' * 40, 'LLM-Training': 'b' * 40}
MEASURED = dict(vram=dict(free_mib=14500), available_ram_bytes=20 * 1024**3, free_disk_bytes=300 * 1024**3)


class Provider:
    def __init__(self, context='current', revisions=None):
        self.context, self.revisions = context, REVISIONS if revisions is None else revisions

    def resolve(self):
        return dict(context=self.context, errors=[] if self.context == 'current' else ['uncommitted executed source'],
                    component_revisions={k: dict(expected=v, detected=v) for k, v in self.revisions.items()})


def rehash(job):
    job['spec_sha256'] = rt.fingerprint(job['spec'])
    body = {k: v for k, v in job.items() if k != 'job_sha256'}
    job['job_sha256'] = rt.fingerprint(body)
    return job


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An isolated workspace: tiny stand-in checkpoint/tokenizer files and a private outputs root."""
    if not (PHASE4 / 'manifest/phase4-manifest.json').exists():
        pytest.skip('Phase 4 frozen manifest not present (ignored asset)')
    root = tmp_path / 'LLM-From-Scratch'
    (root / 'outputs/training').mkdir(parents=True)
    checkpoint, tokenizer = tmp_path / 'checkpoint.pt', tmp_path / 'tokenizer.json'
    checkpoint.write_bytes(b'stand-in weights')
    tokenizer.write_bytes(b'stand-in tokenizer')
    monkeypatch.setattr(rt, 'ROOT', root)
    monkeypatch.setattr(rt, 'INITIALIZATION_SHA256', data.digest(checkpoint))
    monkeypatch.setattr(training_supervisor, 'systemd_launch',
                        lambda *a, **k: pytest.fail('the real systemd launcher must never run in tests'))
    monkeypatch.delenv('CODEXA_WAKEUP_AGENT', raising=False)
    manifest = data.read(PHASE4 / 'manifest/phase4-manifest.json')
    step_zero = data.read(PHASE4 / 'evaluation/step-zero/summary.json')
    job = rt.build_job(PHASE4, manifest, step_zero=step_zero, environment_report=dict(
        component_revisions={k: dict(expected=v, detected=v) for k, v in REVISIONS.items()}))
    job['spec']['initialization'].update(path=str(checkpoint), sha256=data.digest(checkpoint))
    job['spec']['tokenizer'].update(path=str(tokenizer), sha256=data.digest(tokenizer))
    job['spec']['outputs']['root'] = str(root / 'outputs/training' / rt.RUN_ID)
    experiment = tmp_path / 'experiment'
    (experiment / 'config').mkdir(parents=True)
    calls = []
    return dict(job=rehash(job), experiment=experiment, calls=calls, root=root)


def approve(world, job=None):
    return launch.write_approval(world['experiment'], job or world['job'], statement='I authorize this pilot.',
                                 written_utc='2026-10-12T00:00:00Z')


def attempt(world, *, job=None, provider=None, readiness=READY, measured=MEASURED, supervisor=None,
            verify_manifest=None):
    calls = world['calls']

    def launcher(run_dir, root, **kwargs):
        calls.append(('launcher', kwargs['unit'], kwargs['environment']['CODEXA_TRAINING_ENVIRONMENT']))
        return ['systemd-run', '(mocked)']

    def ready():
        calls.append(('readiness',))
        return readiness

    return launch.launch_authorized(
        world['experiment'], job or world['job'], provider='codex', readiness=ready,
        environment=provider or Provider(), measured=measured, written_utc='2026-10-12T00:00:00Z',
        launcher=launcher, supervisor=supervisor or (lambda: dict(available=True)),
        verify_manifest=verify_manifest or (lambda path: calls.append(('manifest',)) or dict(verified=True)))


# ---------------------------------------------------------------- the connected, authorized path

def test_authorized_path_is_connected_and_optimizer_comes_last(world, monkeypatch):
    approve(world)
    result = attempt(world)
    run_dir = Path(result['run_dir'])
    assert result['launched'] and [c[0] for c in world['calls']] == ['readiness', 'manifest', 'launcher']
    assert world['calls'][-1][2] == 'development'
    launch_json = json.loads((run_dir / 'supervisor/launch.json').read_text())
    assert launch_json['command'][1:3] == ['-m', 'codexa_workspace.sft_pilot_trainer']
    assert launch_json['environment']['PYTHONPATH'].split(os.pathsep)[0].endswith('LLM-Architecture/src')
    assert launch_json['deadline_seconds'] == 1200 and launch_json['job_sha256'] == world['job']['job_sha256']
    assert json.loads((run_dir / 'supervisor/state.json').read_text())['state'] == 'READY'
    assert json.loads((run_dir / 'config/job.json').read_text()) == world['job']

    # The supervisor's child then re-checks everything before the optimizer factory is reached.
    order = []
    original_approval = rt.approval_valid
    monkeypatch.setattr(rt, 'approval_valid', lambda *a: order.append('approval') or original_approval(*a))
    monkeypatch.setenv('CODEXA_EXECUTION_CLASS', 'production')

    class RecordingProvider(Provider):
        def resolve(self):
            order.append('environment')
            return super().resolve()

    def fake_trainer(job, run_dir, *, approval, environment):
        order.append('trainer')
        parts = trainer.Components(
            heartbeat=lambda path: type('H', (), dict(set=lambda *a, **k: None, stop=lambda s: None))(),
            budget=lambda s, r: type('B', (), dict(remaining=s, reserve_seconds=r, training_deadline=float('inf'),
                                                   document=lambda self: {}))(),
            datasets=lambda spec: order.append('datasets') or ('t', 'v'),
            model=lambda spec: order.append('model') or 'm',
            optimizer=lambda model, spec: order.append('optimizer') or 'o',
            train_loop=lambda *a: a[-1].update(state=dict(optimizer_step=0)) or dict(optimizer_step=0),
            evaluate=lambda *a: pytest.fail('no evaluation in this structural check'),
            store=lambda run_dir, spec: type('S', (), dict(save=lambda *a, **k: dict(), verify=lambda *a: {}))(),
            free_vram=lambda: None, write=lambda path, value: None)
        job = json.loads(json.dumps(job))
        job['spec']['evaluation']['updates'] = [0]   # structural: no evaluations, no stop decisions
        job['spec']['checkpoint']['minimum_save_seconds'] = 0
        return trainer.run_sft_pilot(job, Path(run_dir) / 'structural', approval=approval, environment=environment,
                                     components=parts)

    report = rt.train_authorized(run_dir, world['job'], environment=RecordingProvider(), measured=MEASURED,
                                 verify_manifest=lambda path: order.append('manifest') or dict(verified=True),
                                 trainer=fake_trainer)
    assert order[:3] == ['approval', 'environment', 'manifest']
    assert order.index('optimizer') > order.index('trainer') > order.index('manifest')
    assert order[-3:] == ['datasets', 'model', 'optimizer'] and report['final_checkpoint_verified']


# ---------------------------------------------------------------- every refusal launches nothing

def refused(world, match, **kwargs):
    with pytest.raises(rt.LaunchRefused, match=match):
        attempt(world, **kwargs)
    assert not any(call[0] == 'launcher' for call in world['calls'])
    assert not Path(world['job']['spec']['outputs']['root']).exists()


def test_missing_authorization(world):
    refused(world, 'not authorized')


def test_approval_for_another_job(world):
    other = rehash(json.loads(json.dumps(world['job'])) | dict(run_id='other'))
    approve(world, other)
    refused(world, 'does not match')


def test_readiness_must_be_ready_for_authorization(world):
    approve(world)
    refused(world, 'Readiness', readiness=dict(verdict='CONDITIONALLY READY', blocking=['startup_free_vram']))


@pytest.mark.parametrize('context', ['mixed', None, 'invalid'])
def test_missing_or_ambiguous_environment(world, context):
    approve(world)
    refused(world, 'Ambiguous', provider=Provider(context))


def test_mismatched_source_revisions(world):
    approve(world)
    refused(world, 'Source revisions differ', provider=Provider(revisions={**REVISIONS, 'LLM-Data': 'c' * 40}))


def defect(world, mutate):
    job = json.loads(json.dumps(world['job']))
    mutate(job['spec'])
    job = rehash(job)
    approve(world, job)
    return job


@pytest.mark.parametrize('mutate,match', [
    (lambda s: s['initialization'].update(sha256='0' * 64), 'initialization checkpoint hash'),
    (lambda s: s['tokenizer'].update(sha256='0' * 64), 'tokenizer hash'),
    (lambda s: s['guards']['gates'].update(startup_free_vram_mib=1000), 'resource gates canonical'),
    (lambda s: s['outputs'].update(root=str(rt.FOUNDATION / 'sft')), 'output namespace'),
    (lambda s: s['evaluation'].update(step_zero_sha256=None), 'step-zero sealed'),
])
def test_preflight_defects_refuse(world, mutate, match):
    job = defect(world, mutate)
    with pytest.raises(rt.LaunchRefused, match=match):
        attempt(world, job=job)
    assert not any(call[0] == 'launcher' for call in world['calls'])


@pytest.mark.parametrize('mutate', [
    lambda s: s['optimization'].update(optimizer='adamw'),
    lambda s: s['training'].update(learning_rate=1e-5),
    lambda s: s['training'].update(max_steps=256),
    lambda s: s['training'].update(gradient_accumulation_steps=4),
    lambda s: s['data'].update(maximum_serialized_tokens=2048),
    lambda s: s['initialization'].update(policy='exact_resume'),
    lambda s: s['checkpoint'].update(publications=['final']),
])
def test_unsupported_or_unapproved_configuration(world, mutate):
    job = defect(world, mutate)
    with pytest.raises(rt.LaunchRefused, match='Unsupported or unapproved'):
        attempt(world, job=job)
    assert not any(call[0] == 'launcher' for call in world['calls'])


def test_changed_dataset_manifest(world):
    approve(world)
    def changed(path):
        raise ValueError('Manifest hash mismatch')
    refused(world, 'manifest present and reproducible', verify_manifest=changed)


def test_insufficient_resources(world):
    approve(world)
    refused(world, 'resource startup_free_vram', measured=dict(MEASURED, vram=dict(free_mib=13518)))


def test_existing_run_namespace(world):
    approve(world)
    results = Path(world['job']['spec']['outputs']['root']) / 'metrics'
    results.mkdir(parents=True)
    with pytest.raises(rt.LaunchRefused, match='output namespace'):
        attempt(world)
    assert not any(call[0] == 'launcher' for call in world['calls'])


def test_supervisor_unavailable(world):
    approve(world)
    refused(world, 'supervisor unavailable', supervisor=lambda: dict(available=False, linger='no'))


def test_tampered_job_and_wakeup_agent(world, monkeypatch):
    approve(world)
    tampered = json.loads(json.dumps(world['job']))
    tampered['spec']['budget']['total_seconds'] = 3600
    refused(world, 'hash mismatch', job=tampered)
    monkeypatch.setenv('CODEXA_WAKEUP_AGENT', '1')
    refused(world, 'wakeup analysis agent')


def test_approval_is_never_overwritten(world):
    approve(world)
    with pytest.raises(FileExistsError):
        approve(world)
    with pytest.raises(ValueError, match='statement'):
        launch.write_approval(world['experiment'] / 'other', world['job'], statement=' ', written_utc='x')


# ---------------------------------------------------------------- resume identity

def _stopped(tmp_path, job, reason):
    run = tmp_path / 'run'
    (run / 'supervisor').mkdir(parents=True)
    (run / 'config').mkdir()
    (run / 'supervisor/completion.json').write_text(json.dumps(dict(terminal_state='EARLY_STOPPED', reason=reason)))
    (run / 'config/job.json').write_text(json.dumps(job))
    return run


def test_resume_only_after_resource_stop_with_identical_identity(tmp_path):
    job = dict(job_sha256='j', code_identity=dict(a='1'))
    verified = dict(verified=True, optimizer_step=40)
    assert launch.resume_allowed(_stopped(tmp_path / 'a', job, 'MEMORY_HEADROOM_GATE'), job, latest=verified)['resumable']
    with pytest.raises(rt.LaunchRefused, match='Not resumable'):
        launch.resume_allowed(_stopped(tmp_path / 'b', job, 'DEGENERATION_DETERIORATION'), job, latest=verified)
    with pytest.raises(rt.LaunchRefused, match='Unsafe resume'):
        launch.resume_allowed(_stopped(tmp_path / 'c', dict(job, code_identity=dict(a='2')), 'MEMORY_HEADROOM_GATE'),
                              job, latest=verified)
    with pytest.raises(rt.LaunchRefused, match='Unsafe resume'):
        launch.resume_allowed(_stopped(tmp_path / 'd', job, 'MEMORY_HEADROOM_GATE'), job, latest=dict(verified=False))


# ---------------------------------------------------------------- implicit starts

@pytest.mark.parametrize('selection,message', [(None, 'Explicit --environment'),
                                               ('historical', 'New training jobs require')])
def test_trainer_entry_refuses_implicit_or_historical_start(tmp_path, selection, message):
    """The real trainer entry stops at Codex's guard before torch, a model or an optimizer."""
    environment = {k: v for k, v in os.environ.items() if k not in ('CODEXA_TRAINING_ENVIRONMENT', 'PYTHONPATH')}
    environment['PYTHONPATH'] = launch.source_first_pythonpath()
    if selection:
        environment['CODEXA_TRAINING_ENVIRONMENT'] = selection
    probe = ('import sys, runpy; sys.argv = ["trainer", "--run-dir", sys.argv[1]]; '
             'runpy.run_module("codexa_workspace.sft_pilot_trainer", run_name="__main__"); ')
    result = subprocess.run([str(launch.PYTHON), '-c', probe + 'print("torch" in sys.modules)', str(tmp_path / 'run')],
                            capture_output=True, text=True, env=environment, timeout=120, cwd=str(launch.ROOT))
    assert result.returncode != 0 and message in result.stderr
    assert not (tmp_path / 'run').exists()


def test_importing_trainer_constructs_nothing():
    source = Path(trainer.__file__).read_text()
    assert "if __name__ == '__main__'" in source
    assert 'default_components()' in source and trainer.main.__doc__.count('guard') >= 1
