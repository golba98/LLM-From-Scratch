"""Preparation gates cannot authorize GPU work or bypass scientific review."""

import hashlib
import json
import time

import pytest
import yaml

from codexa_workspace import conversational_readiness as readiness


def test_candidate_process_isolation_preserves_original_deadline(tmp_path,monkeypatch):
    root=tmp_path/'root';directory=root/'job';directory.mkdir(parents=True)
    scripts=root/'scripts';scripts.mkdir()
    (scripts/'conversational_readiness.py').write_text('''
import sys,json,time
from pathlib import Path
a=sys.argv;directory=Path(a[a.index('--output')+1]);name=a[a.index('--candidate')+1]
started=float(a[a.index('--original-started-monotonic')+1]);assert 0<started+3600-time.monotonic()<100
target=directory/'generations'/name;target.mkdir()
(target/'report.json').write_text(json.dumps({'candidate':name}))
''')
    observed=dict(disk_gate_passed=True,host_gate_passed=True,gpu_gate_passed=True)
    plan=dict(output=str(directory/'generations'),resources=observed,job_sha256='job',
              max_wall_clock_seconds=3600,candidates={'repair':{'path':'fixture','sha256':'identity'}})
    monkeypatch.setattr(readiness,'baseline_plan',lambda *a:plan)
    monkeypatch.setattr(readiness,'resource_preflight',lambda *a:observed)
    approval=directory/'approval.json';approval.write_text(json.dumps(dict(
        format='codexa-inference-baseline-approval-v1',approved=True,job_sha256='job')))
    result=readiness.run_baselines(root,directory,approval,started=time.monotonic()-3501)
    assert result['repair']['candidate']=='repair'
    process=json.loads((directory/'candidate-repair.supervisor/process.json').read_text())
    assert 0<process['budget']['remaining_seconds']<100


def test_incomplete_comparison_cannot_change_scientific_protocol(tmp_path,monkeypatch):
    output=tmp_path/'generations';output.mkdir()
    keys=dict(candidates={},suite_sha256='new',decoding={},validation={},scoring_sha256='s',evaluator_sha256='e')
    plan=dict(output=str(output),job_sha256='job',resources=dict(disk_gate_passed=True,
        host_gate_passed=True,gpu_gate_passed=True),max_wall_clock_seconds=3600,**keys)
    (output/'completed-evidence.json').write_text(json.dumps(dict(plan={**keys,'suite_sha256':'old'},results={})))
    monkeypatch.setattr(readiness,'baseline_plan',lambda *a:plan)
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(dict(format='codexa-inference-baseline-approval-v1',
        approved=True,job_sha256='job')))
    with pytest.raises(ValueError,match='scientific protocol'):
        readiness.run_baselines(tmp_path,tmp_path,approval,resume_incomplete=True)


def test_inference_approval_is_scoped_and_resources_cannot_be_waived(tmp_path):
    approval=tmp_path/'approval.json'
    plan={'job_sha256':'fixed','resources':{'disk_gate_passed':True,'host_gate_passed':False,'gpu_gate_passed':True}}
    approval.write_text(json.dumps({'format':'codexa-inference-baseline-approval-v1','approved':True,'job_sha256':'wrong'}))
    with pytest.raises(ValueError):
        readiness.validate_baseline_approval(plan,approval)
    approval.write_text(json.dumps({'format':'codexa-inference-baseline-approval-v1','approved':True,'job_sha256':'fixed'}))
    with pytest.raises(RuntimeError,match='resource gate'):
        readiness.validate_baseline_approval(plan,approval)
    plan['resources']['host_gate_passed']=True
    readiness.validate_baseline_approval(plan,approval)


def test_proposal_cannot_promote_backend_or_reduce_reserve(tmp_path):
    path=tmp_path/'proposal.yaml'
    path.write_text(yaml.safe_dump({'format':'codexa-readiness-proposal-v1','spec':{
        'checkpoint':{'storage_backend':'optimized','reclamation_mode':'disabled'},
        'minimum_disk_free_bytes':180*1024**3}}))
    with pytest.raises(ValueError,match='backend'):
        readiness.proposal_preflight(tmp_path,path)


def test_materialization_refuses_unknown_initializer_before_writing(tmp_path):
    proposal=tmp_path/'proposal.yaml';selection=tmp_path/'selection.json'
    proposal.write_text(yaml.safe_dump({'format':'codexa-readiness-proposal-v1'}))
    selection.write_text(json.dumps({'initialization':None}))
    with pytest.raises(ValueError,match='initializer'):
        readiness.materialize_launch(tmp_path,proposal,selection,tmp_path/'no-review',tmp_path/'no-qualification')
    assert not (tmp_path/'outputs').exists()


def test_data_admission_requires_all_pending_records(tmp_path,monkeypatch):
    output=tmp_path/'outputs';output.mkdir()
    monkeypatch.setattr(readiness,'workspace_root',lambda:tmp_path)
    monkeypatch.setattr(readiness,'output_root',lambda root:output)
    source=tmp_path/'source.jsonl'
    source.write_text(json.dumps({'conversation_id':'pending','review_status':'pending_human'})+'\n')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'format':'codexa-chat-mixture-v1','sources':[
        {'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    review=tmp_path/'review.json'
    decision={'approved':True,'reviewer':'fixture reviewer','method':'human_manual_review',
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'approved_record_ids':[]}
    review.write_text(json.dumps(decision))
    with pytest.raises(ValueError,match='Missing'):
        readiness.admit_mixture(manifest,review,output/'admitted.json')
    assert not (output/'admitted.json').exists()
    decision['approved_record_ids']=['pending'];review.write_text(json.dumps(decision))
    result=readiness.admit_mixture(manifest,review,output/'admitted.json')
    assert result['status']=='HUMAN_ADMITTED'
    assert result['review_sha256']==hashlib.sha256(review.read_bytes()).hexdigest()


def test_no_baseline_is_never_a_selected_initializer(tmp_path):
    comparison=tmp_path/'comparison.json';comparison.write_text(json.dumps({'results':{}}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'reviewer':'fixture',
        'method':'human_manual_review','comparison_sha256':hashlib.sha256(comparison.read_bytes()).hexdigest()}))
    result=readiness.select_initializer(comparison,review)
    assert result['initializer'] is None and result['status']=='BLOCKED'


def test_validation_stratification_uses_unique_roots_and_fixed_seed(tmp_path):
    path=tmp_path/'validation.jsonl'
    path.write_text(''.join(json.dumps({'source':source,'root_id':str(i//2)})+'\n'
        for source in ('a','b') for i in range(12)))
    selected=readiness.stratified_validation_indices(path,per_source=4)
    assert selected==readiness.stratified_validation_indices(path,per_source=4)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(selected)==8 and len({(rows[i]['source'],rows[i]['root_id']) for i in selected})==8
