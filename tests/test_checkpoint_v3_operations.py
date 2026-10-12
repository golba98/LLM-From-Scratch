"""Capacity and GPU authorization gates are tested without creating CUDA models."""

import time
import json

import pytest

from llm_training.preflight import project_continuous_storage
from codexa_workspace import checkpoint_gpu_qualification as qualification


def test_capacity_schedule_and_reserve_unchanged():
    result=project_continuous_storage(checkpoint_bytes=5636070993)
    assert [r['publications_per_run'] for r in result['periods']]==[14,32,62]
    assert result['required_free_bytes']>=180*1024**3
    assert not result['hypothetical_reclamation_is_available_space']
    assert result['periods'][2]['v1_append_only_upper_bytes']==62*5636070993
    with pytest.raises(ValueError):
        project_continuous_storage(checkpoint_bytes=1,retained_states=2,active_runs=2)


def test_gpu_plan_does_not_create_output_and_execute_refuses_failed_capacity(tmp_path,monkeypatch):
    checkpoint=tmp_path/'fixture.pt'
    checkpoint.write_bytes(b'not a model: plan uses size only')
    monkeypatch.setattr(qualification,'hardware',lambda:{'free_gpu_bytes':0,'host_memory_bytes':{}})
    monkeypatch.setattr(qualification.shutil,'disk_usage',lambda p:type('Capacity',(),{'free':1})())
    output=tmp_path/'new-qualification'
    plan=qualification.qualification_plan(checkpoint,output)
    assert not output.exists() and not plan['storage_gate_passed']
    # Even the approval path is not read when the capacity gate fails.
    with pytest.raises(RuntimeError,match='reserve failed'):
        qualification.execute_qualification(plan,tmp_path/'missing-approval.json',started=time.monotonic())
    assert not output.exists()
    with pytest.raises(ValueError):
        qualification.qualification_plan(checkpoint,output,updates=100)


def test_explicit_optimizer_evidence_reverified_and_no_optimizer_substitution(tmp_path,monkeypatch):
    import llm_architecture.checkpoint_inspection as inspection
    from llm_training.preflight import verified_optimizer_storage_evidence
    actual=dict(sha256='verified',config={'model':{'width':8}},complete_optimizer_state=True,
        file_bytes=128,components={'model_state_dict':{'bytes':32},
                                  'optimizer_state_dict':{'bytes':8,'by_dtype':{'torch.uint8':8}}})
    supplied=dict(format='codexa-storage-evidence-v1',path=str(tmp_path/'fixture.pt'),sha256='verified')
    path=tmp_path/'evidence.json'
    path.write_text(json.dumps(supplied))
    calls=[]
    def verify(checkpoint):
        calls.append(checkpoint)
        return actual
    monkeypatch.setattr(inspection,'inspect_checkpoint_storage',verify)
    result=verified_optimizer_storage_evidence(path,model_config={'width':8},optimizer_name='adamw8bit')
    assert result['optimizer_bytes']==8 and calls
    with pytest.raises(ValueError,match='match'):
        verified_optimizer_storage_evidence(path,model_config={'width':8},optimizer_name='adamw')
    actual['sha256']='changed'
    with pytest.raises(ValueError,match='match'):
        verified_optimizer_storage_evidence(path,model_config={'width':8},optimizer_name='adamw8bit')


def test_gpu_comparison_is_bitwise_for_signed_zero_and_nan_payloads():
    import torch
    import pytest
    from codexa_workspace.checkpoint_gpu_qualification import _equal
    _equal(torch.tensor(float('nan')), torch.tensor(float('nan')))
    with pytest.raises(ValueError, match='Non-bitwise'):
        _equal(torch.tensor(0.), torch.tensor(-0.))
