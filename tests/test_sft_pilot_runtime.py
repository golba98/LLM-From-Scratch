"""SFT pilot runtime: modes, guards, environment contract, authorization and stop policy.

No real optimizer is constructed and no backward pass runs; train-mode control flow is
exercised with injected fakes only.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from codexa_workspace import sft_pilot_runtime as rt
from codexa_workspace import sft_pilot_trainer as trainer


# ---------------------------------------------------------------- gates

def test_canonical_gates_match_owner_and_cannot_be_lowered():
    gates = rt.canonical_gates()
    assert gates == dict(startup_free_vram_mib=13928, available_ram_bytes=16 * 1024**3,
                         free_disk_bytes=180 * 1024**3, recovery_reserve_bytes=100 * 1024**3,
                         in_run_free_vram_bytes=1536 * 1024**2)
    assert rt.validate_gates(dict(gates))
    for key in gates:
        with pytest.raises(ValueError, match='may not be lowered'):
            rt.validate_gates({**gates, key: gates[key] - 1})
    with pytest.raises(ValueError):
        rt.validate_gates({k: v for k, v in gates.items() if k != 'free_disk_bytes'})


def test_resource_checks_report_each_gate():
    measured = dict(vram=dict(free_mib=13927), available_ram_bytes=16 * 1024**3, free_disk_bytes=181 * 1024**3)
    checks = {c['name']: c['passed'] for c in rt.resource_checks(rt.canonical_gates(), measured)}
    assert checks == dict(startup_free_vram=False, available_ram=True, free_disk_and_recovery_reserve=True)


# ---------------------------------------------------------------- no-training guard

def test_guard_blocks_optimizer_backward_and_save_then_restores(tmp_path):
    layer = torch.nn.Linear(2, 2)
    with rt.no_training_guard(layer):
        assert not any(p.requires_grad for p in layer.parameters())
        with pytest.raises(rt.TrainingProhibited):
            torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.)
        with pytest.raises(rt.TrainingProhibited):
            torch.zeros(1, requires_grad=True).sum().backward()
        with pytest.raises(rt.TrainingProhibited):
            torch.autograd.backward([torch.zeros(1, requires_grad=True).sum()])
        with pytest.raises(rt.TrainingProhibited):
            torch.save({}, tmp_path / 'x.pt')
    assert torch.save is not None and 'raiser' not in torch.save.__name__
    assert torch.Tensor.backward.__name__ == 'backward'


def test_guard_blocks_bitsandbytes_optimizer_when_installed():
    bnb = pytest.importorskip('bitsandbytes.optim')
    with rt.no_training_guard():
        with pytest.raises(rt.TrainingProhibited):
            bnb.AdamW8bit([torch.nn.Parameter(torch.zeros(1))])


# ---------------------------------------------------------------- environment contract

class Provider:
    def __init__(self, context):
        self.context = context

    def resolve(self):
        return dict(context=self.context)


@pytest.mark.parametrize('context', ['mixed', None, 'unknown'])
def test_ambiguous_environment_rejected(context):
    with pytest.raises(rt.LaunchRefused, match='Ambiguous'):
        rt.check_environment(Provider(context), 'current')


def test_environment_mismatch_rejected_and_match_accepted():
    with pytest.raises(rt.LaunchRefused, match='differs'):
        rt.check_environment(Provider('historical'), 'current')
    assert rt.check_environment(Provider('current'), 'current')['context'] == 'current'


# ---------------------------------------------------------------- job identity and authorization

def _job(tmp_path):
    spec = dict(runnable=False, authorization_required=True, environment=dict(required_context='current'),
                budget=dict(total_seconds=900), run_name='x')
    job = dict(format=rt.JOB_FORMAT, run_id='x', spec=spec, spec_sha256=rt.fingerprint(spec), launch_command=None)
    job['job_sha256'] = rt.fingerprint(job)
    return job


def test_job_hash_tamper_rejected(tmp_path):
    job = _job(tmp_path)
    assert rt.verify_job(job)
    tampered = json.loads(json.dumps(job))
    tampered['spec']['budget']['total_seconds'] = 1200
    with pytest.raises(rt.LaunchRefused):
        rt.verify_job(tampered)


def test_unauthorized_train_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv('CODEXA_EXECUTION_CLASS', raising=False)
    job = _job(tmp_path)
    called = []
    fake_trainer = lambda *a, **k: called.append(1)
    with pytest.raises(rt.LaunchRefused, match='not authorized'):
        rt.train_authorized(tmp_path, job, environment=Provider('current'), measured={}, trainer=fake_trainer)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/approval.json').write_text(json.dumps(dict(
        format=rt.APPROVAL_FORMAT, approved=True, job_sha256='0' * 64, statement='yes')))
    with pytest.raises(rt.LaunchRefused, match='does not match'):
        rt.train_authorized(tmp_path, job, environment=Provider('current'), measured={}, trainer=fake_trainer)
    (tmp_path / 'config/approval.json').write_text(json.dumps(dict(
        format=rt.APPROVAL_FORMAT, approved=True, job_sha256=job['job_sha256'], statement='')))
    with pytest.raises(rt.LaunchRefused, match='does not match'):
        rt.train_authorized(tmp_path, job, environment=Provider('current'), measured={}, trainer=fake_trainer)
    (tmp_path / 'config/approval.json').write_text(json.dumps(dict(
        format=rt.APPROVAL_FORMAT, approved=True, job_sha256=job['job_sha256'], statement='granted')))
    with pytest.raises(rt.LaunchRefused, match='Ambiguous'):
        rt.train_authorized(tmp_path, job, environment=Provider('mixed'), measured={}, trainer=fake_trainer)
    assert called == []


def test_non_production_class_cannot_reach_real_trainer(tmp_path):
    assert os.environ.get('CODEXA_EXECUTION_CLASS') == 'unit_test'
    with pytest.raises(rt.LaunchRefused, match='Non-production'):
        rt.train_authorized(tmp_path, _job(tmp_path), environment=Provider('current'), measured={})


def test_prepare_reports_failures_without_constructing_model(tmp_path):
    job = _job(tmp_path)
    job['spec'].update(data=dict(manifest=str(tmp_path / 'missing.json'), manifest_sha256='0'),
                       initialization=dict(path=str(tmp_path / 'ckpt.pt'), sha256='0'),
                       tokenizer=dict(path=str(tmp_path / 'tok.json'), sha256='0'))
    with rt.no_training_guard():
        result = rt.prepare(job)
    failed = {c['name'] for c in result['checks'] if not c['passed']}
    assert {'job and spec hashes', 'manifest present and reproducible', 'initialization checkpoint hash',
            'tokenizer hash'} <= failed
    assert result['passed'] is False and result['optimizer_constructed'] is False


def test_launch_plan_is_documentation_only_and_source_first(tmp_path):
    before = set(tmp_path.rglob('*'))
    job = _job(tmp_path)
    plan = rt.launch_plan(tmp_path, job)
    assert set(tmp_path.rglob('*')) == before and 'NOT AUTHORIZED' in plan['status']
    assert plan['environment']['CODEXA_TRAINING_ENVIRONMENT'] == 'development'
    first = plan['environment']['PYTHONPATH'].split(os.pathsep)[0]
    assert first.endswith('LLM-Architecture/src') and '.venv' not in plan['environment']['PYTHONPATH']
    assert plan['command'][1:3] == ['-m', 'codexa_workspace.sft_pilot_trainer']


def test_preparation_modules_do_not_import_training_execution():
    for name in ('sft_pilot_runtime.py', 'sft_phase4_data.py', 'sft_pilot_evaluation.py', 'sft_phase4_operations.py'):
        source = (Path(rt.__file__).parent / name).read_text()
        assert 'create_adamw_optimizer' not in source and 'train_model' not in source, name


# ---------------------------------------------------------------- stop policy

def test_stop_policy_update_decisions():
    policy = rt.StopPolicy()
    common = dict(gradient_norm=1., free_vram_bytes=4 * 1024**3, training_deadline_passed=False)
    assert policy.after_update(step=10, loss=2., **common) is None
    assert policy.after_update(step=10, loss=float('nan'), **common) == 'NONFINITE'
    assert policy.after_update(step=10, loss=2., gradient_norm=float('inf'), free_vram_bytes=None,
                               training_deadline_passed=False) == 'NONFINITE'
    assert policy.after_update(step=10, loss=2., **{**common, 'free_vram_bytes': 1024**3}) == 'MEMORY_HEADROOM_GATE'
    assert policy.after_update(step=10, loss=2., **{**common, 'training_deadline_passed': True}) == 'TRAINING_DEADLINE'
    assert policy.after_update(step=128, loss=2., **common) == 'STEP_CAP'


def _eval(severe, wiki=3.0, finite=True):
    return dict(generation={'conversation/greedy': dict(severe_degeneration=severe)},
                reference=dict(sources=dict(wikipedia=wiki, fineweb_edu=3.1)), finite=finite)


def test_stop_policy_requires_two_consecutive_deteriorations():
    step0 = _eval(10)
    policy = rt.StopPolicy()
    assert policy.after_evaluation(step=32, step_zero=step0, result=_eval(11)) is None
    assert policy.after_evaluation(step=64, step_zero=step0, result=_eval(12)) == 'DEGENERATION_DETERIORATION'
    policy = rt.StopPolicy()
    assert policy.after_evaluation(step=32, step_zero=step0, result=_eval(11)) is None
    assert policy.after_evaluation(step=64, step_zero=step0, result=_eval(9)) is None
    policy = rt.StopPolicy()
    assert policy.after_evaluation(step=32, step_zero=step0, result=_eval(10, wiki=3.2)) is None
    assert policy.after_evaluation(step=64, step_zero=step0, result=_eval(10, wiki=3.2)) == 'REFERENCE_CE_RETENTION'
    assert rt.StopPolicy().after_evaluation(step=32, step_zero=step0, result=_eval(1, finite=False)) == 'NONFINITE'


def test_exception_classification_and_terminal_states():
    assert rt.StopPolicy.classify_exception(torch.cuda.OutOfMemoryError('x')) == 'CUDA_OOM'
    assert rt.StopPolicy.classify_exception(FloatingPointError()) == 'NONFINITE'
    assert rt.terminal_state(None) == 'COMPLETED' and rt.terminal_state('STEP_CAP') == 'COMPLETED'
    assert rt.terminal_state('TRAINING_DEADLINE') == 'COMPLETED'
    assert rt.outcome('TRAINING_DEADLINE') == 'INCONCLUSIVE_DEADLINE' and rt.outcome(None) == 'STEP_CAP'
    assert rt.terminal_state('CUDA_OOM') == 'FAILED' and rt.terminal_state('MEMORY_HEADROOM_GATE') == 'EARLY_STOPPED'


# ---------------------------------------------------------------- train-mode control flow with fakes

class FakeBudget:
    def __init__(self, seconds, reserve):
        self.remaining, self.reserve_seconds = seconds, reserve
        self.training_deadline = float('inf')

    def document(self):
        return dict(remaining=self.remaining)


class FakeHeartbeat:
    def __init__(self, path):
        self.phases = []

    def set(self, phase=None, **values):
        if phase:
            self.phases.append(phase)

    def stop(self):
        self.phases.append('STOPPED')


def _components(tmp_path, *, losses=None, evaluations=None, free=4 * 1024**3, error=None, saves=None):
    saves = [] if saves is None else saves
    writes = {}

    def train_loop(model, optimizer, train, spec, budget, on_update, stop_requested, holder):
        if error:
            raise error
        step = 0
        for step, loss in enumerate(losses or [2.] * 128, start=1):
            holder['state'] = dict(optimizer_step=step)
            on_update(step, loss, 1.)
            if stop_requested():
                break
        return dict(optimizer_step=step)

    class Store:
        def __init__(self, run_dir, spec):
            pass

        def save(self, model, optimizer, state, *, final):
            saves.append((final, state['optimizer_step']))
            return dict(optimizer_step=state['optimizer_step'], final=final)

        def verify(self, record, state):
            return dict(verified=True)

    def evaluate(model, validation, spec, step):
        return (evaluations or {}).get(step, _eval(10))

    parts = trainer.Components(heartbeat=FakeHeartbeat, budget=FakeBudget, datasets=lambda spec: ('train', 'val'),
                               model=lambda spec: 'model', optimizer=lambda model, spec: 'fake-optimizer',
                               train_loop=train_loop, evaluate=evaluate, store=Store, free_vram=lambda: free,
                               write=lambda path, value: writes.__setitem__(str(path), value))
    return parts, saves, writes


def _train_job(tmp_path):
    (tmp_path / 'step0.json').write_text(json.dumps(_eval(10)))
    spec = dict(run_name='fake', training=dict(max_steps=128), optimization=dict(minimum_headroom_bytes=1536 * 1024**2),
                acceptance=dict(reference_ce_increase_max=.1), budget=dict(total_seconds=900, finalization_reserve_seconds=180),
                evaluation=dict(updates=[0, 32, 64, 128], step_zero_path=str(tmp_path / 'step0.json')),
                checkpoint=dict(publications=[64, 'final'], minimum_save_seconds=60), initialization={})
    return dict(spec=spec, job_sha256='j')


def test_fake_train_completes_with_checkpoints_at_64_and_final(tmp_path):
    parts, saves, writes = _components(tmp_path)
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['state'] == 'COMPLETED' and report['final_checkpoint_verified']
    assert saves == [(False, 64), (True, 128)]
    assert [e['step'] for e in report['evaluations']] == [32, 64, 128]
    assert str(tmp_path / 'run/metrics/report.json') in writes


def test_fake_train_nonfinite_fails_without_saving(tmp_path):
    parts, saves, _ = _components(tmp_path, losses=[2.] * 10 + [float('nan')])
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['state'] == 'FAILED' and report['stop_reason'] == 'NONFINITE' and saves == []


def test_fake_train_oom_fails_without_saving(tmp_path):
    parts, saves, _ = _components(tmp_path, error=torch.cuda.OutOfMemoryError('oom'))
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['state'] == 'FAILED' and report['stop_reason'] == 'CUDA_OOM' and saves == []


def test_fake_train_headroom_stops_early_and_keeps_final(tmp_path):
    parts, saves, _ = _components(tmp_path, free=1024**3)
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['state'] == 'EARLY_STOPPED' and report['stop_reason'] == 'MEMORY_HEADROOM_GATE'
    assert saves == [(True, 1)]


def test_fake_train_degeneration_deterioration_stops_at_64(tmp_path):
    parts, saves, _ = _components(tmp_path, evaluations={32: _eval(11), 64: _eval(12)})
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['stop_reason'] == 'DEGENERATION_DETERIORATION' and report['state'] == 'EARLY_STOPPED'
    assert saves == [(True, 64)]


def test_fake_train_refuses_existing_run(tmp_path):
    (tmp_path / 'run/metrics').mkdir(parents=True)
    (tmp_path / 'run/metrics/report.json').write_text('{}')
    parts, _, _ = _components(tmp_path)
    with pytest.raises(FileExistsError):
        trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)


# ---------------------------------------------------------------- Codex training-guard adapter

def _inspector(report=None, error=None):
    def inspect(environment, operation):
        assert (environment, operation) == ('development', 'new')
        if error:
            raise error
        return report
    return inspect


def test_training_guard_provider_maps_valid_development_to_current():
    report = dict(selected='development', valid=True, errors=[], component_revisions={'LLM-Data': dict(detected='a')})
    resolved = rt.check_environment(rt.TrainingGuardProvider('new', inspector=_inspector(report)), 'current')
    assert resolved['context'] == 'current' and resolved['enforcement'] == 'codex-guard'
    assert resolved['component_revisions'] == {'LLM-Data': dict(detected='a')}


@pytest.mark.parametrize('report,error', [
    (dict(selected='development', valid=False, errors=['Integration training source has uncommitted changes']), None),
    (dict(selected='historical', valid=True, errors=[]), None),
    (dict(selected='development', valid=True, errors=['late error']), None),
    (None, RuntimeError('guard unavailable'))])
def test_training_guard_provider_rejects_invalid_or_unavailable(report, error):
    provider = rt.TrainingGuardProvider('new', inspector=_inspector(report, error))
    with pytest.raises(rt.LaunchRefused, match='Ambiguous'):
        rt.check_environment(provider, 'current')


def test_real_guard_is_consulted_in_process():
    """The default provider runs Codex's real inspection; on a dirty or unpinned tree it refuses."""
    resolved = rt.TrainingGuardProvider('new').resolve()
    guard = rt.training_guard().inspect('development', 'new')
    assert resolved['context'] == ('current' if guard['valid'] else 'invalid')
    assert resolved['errors'] == guard['errors']


def test_every_trainer_terminal_state_is_known_to_the_supervisor():
    """The hash-bound supervisor classifies only its own TERMINAL states; anything else becomes a failure."""
    from codexa_workspace.training_supervisor import TERMINAL
    reasons = [None, 'STEP_CAP', 'TRAINING_DEADLINE', 'NONFINITE', 'CUDA_OOM', 'FINAL_SAVE_RESERVE_EXHAUSTED',
               'CHECKPOINT_FAILURE', 'INTERRUPTED', 'MEMORY_HEADROOM_GATE', 'DEGENERATION_DETERIORATION',
               'REFERENCE_CE_RETENTION', 'RuntimeError']
    assert {rt.terminal_state(r) for r in reasons} <= set(TERMINAL)
    # The exception path writes FAILED or INTERRUPTED only.
    assert {'FAILED', 'INTERRUPTED'} <= set(TERMINAL)


def test_fake_train_deadline_during_accumulation_finalizes_gracefully(tmp_path):
    parts, saves, _ = _components(tmp_path, losses=[2.] * 40)

    def interrupted(model, optimizer, train, spec, budget, on_update, stop_requested, holder):
        for step in range(1, 41):
            holder['state'] = dict(optimizer_step=step)
            on_update(step, 2., 1.)
        raise TimeoutError('Training deadline reached during accumulation.')
    parts.train_loop = interrupted
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert (report['state'], report['stop_reason'], report['outcome']) == ('COMPLETED', 'TRAINING_DEADLINE',
                                                                          'INCONCLUSIVE_DEADLINE')
    assert saves == [(True, 40)] and report['final_checkpoint_verified']


def test_fake_train_early_return_is_never_reported_as_step_cap(tmp_path):
    parts, saves, _ = _components(tmp_path, losses=[2.] * 20)
    report = trainer.run_sft_pilot(_train_job(tmp_path), tmp_path / 'run', approval={}, environment={}, components=parts)
    assert report['stop_reason'] == 'TRAINING_INCOMPLETE' and report['state'] == 'EARLY_STOPPED'
    assert saves == [(True, 20)]
