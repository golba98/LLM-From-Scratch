"""Small fixtures prove prompt, approval and deadline behavior without real models."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from codexa_workspace import foundation_diagnosis as diagnosis
from codexa_workspace.chat_evaluation import fingerprint
from llm_tokenizer.tokenizer import SPECIAL_TOKENS


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([str(diagnosis.concepts())], trainers.BpeTrainer(
        vocab_size=512, special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tokenizer


def test_matched_forms_do_not_inject_bos(tokenizer):
    rows = diagnosis.make_cases(tokenizer)
    assert len(rows) == 120 and len({x['id'] for x in rows}) == 120
    for row in rows:
        assert row['input_ids'] == tokenizer.encode(row['prompt'], add_special_tokens=False).ids
        assert 1 not in row['input_ids']
        if row['form'] in ('neutral', 'direct'):
            assert not any(x < 8 for x in row['input_ids'])
        else:
            assert row['input_ids'][-1] == 6
            assert row['input_ids'].count(7) == (3 if row['form'] == 'multi_turn' else 1)
            assert row['answer_token_ids']['expected'] == tokenizer.encode(row['expected'], add_special_tokens=False).ids
    for concept in diagnosis.concepts():
        matched = [x for x in rows if x['concept'] == concept['id']]
        assert len(matched) == 5
        assert {x['expected'] for x in matched} == {concept['expected']}


def test_invalid_or_duplicate_arithmetic_reference_is_rejected(tokenizer, monkeypatch):
    original = diagnosis.concepts()
    mutated = [dict(x) for x in original]
    next(x for x in mutated if x['category']=='arithmetic')['expected'] = '999'
    monkeypatch.setattr(diagnosis,'concepts',lambda:mutated)
    with pytest.raises(ValueError,match='reference'):
        diagnosis.make_cases(tokenizer)


def test_keyword_and_contradiction_are_not_semantic_success():
    case = dict(expected='Rome', category='factual', form='chat')
    score = diagnosis.scoring('Rome is the capital of France.', case, 7)
    assert not score['exact_surface_correct']
    assert score['semantic_correct'] is None
    assert score['ended_with_END']
    assert diagnosis.scoring('Rome', case, 7)['exact_surface_correct']
    plain = diagnosis.scoring('Rome', dict(case, form='neutral'), None)
    assert plain['missing_chat_END'] is None
    assert not plain['chat_turn_stop_expected']


def test_arithmetic_exactness_and_case():
    case = dict(expected='63', category='arithmetic', form='short')
    assert diagnosis.scoring('63', case, 7)['arithmetic_exact_correct']
    assert not diagnosis.scoring('The result is 63, or perhaps 53.', case, 7)['arithmetic_exact_correct']
    case.update(expected='C', category='factual')
    assert not diagnosis.scoring('c', case, 7)['exact_surface_correct']


def test_likelihood_requires_finite_values(tokenizer):
    case = diagnosis.make_cases(tokenizer)[0]
    class Nonfinite:
        def __call__(self, ids):
            return torch.full((*ids.shape, tokenizer.get_vocab_size()), float('nan')), None
    with pytest.raises(FloatingPointError):
        diagnosis.likelihood_probe(Nonfinite(), tokenizer, case, device='cpu')


def test_cache_and_uncached_cpu_generation_agree(tokenizer):
    from llm_architecture.model import LanguageModel, ModelConfig
    from llm_inference.generate import GenerationConfig, generate_sequences
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        model = LanguageModel(ModelConfig(vocab_size=tokenizer.get_vocab_size(), context_length=512,
            num_layers=1, hidden_size=16, num_heads=2, intermediate_size=32)).eval()
        case = diagnosis.make_cases(tokenizer)[0]
        ids = torch.tensor([case['input_ids']])
        results = [generate_sequences(model, ids, eos_token_id=2, pad_token_id=0,
            stop_sequences=[[7]], config=GenerationConfig(max_new_tokens=8, use_kv_cache=cache))
            for cache in (True, False)]
        assert results[0].sequences[0].token_ids == results[1].sequences[0].token_ids
        with torch.inference_mode():
            probes = diagnosis.likelihood_probe(model, tokenizer, case, device='cpu')
        assert probes['expected']['mean_token_nll'] > 0
        assert 0 <= probes['expected']['first_token_probability'] <= 1
    finally:
        torch.set_num_threads(original_threads)


def approved_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'prefix', str(tmp_path/'.venv'))
    monkeypatch.setattr(diagnosis, 'code_identity', lambda root: {'fixture': 'code'})
    monkeypatch.setattr(diagnosis, 'resource_preflight', lambda root: dict(
        disk_gate_passed=True, host_gate_passed=True, gpu_gate_passed=True))
    directory = tmp_path/'diagnostic'
    directory.mkdir()
    suite = []
    tokenizer_path = tmp_path/'tokenizer.json'
    tokenizer_path.write_text('fixture')
    plan = dict(format=diagnosis.FORMAT, optimizer_updates=0, suite_sha256=fingerprint(suite),
        code_identity={'fixture': 'code'}, tokenizer_path=str(tokenizer_path),
        tokenizer_sha256=diagnosis.file_sha256(tokenizer_path),
        approval_format='codexa-foundation-diagnostic-approval-v1', candidates={'base': {}})
    plan['job_sha256'] = fingerprint(plan)
    (directory/'plan.json').write_text(json.dumps(plan))
    (directory/'suite.json').write_text(json.dumps(suite))
    approval = tmp_path/'approval.json'
    approval.write_text(json.dumps(dict(format=plan['approval_format'], approved=True,
                                        job_sha256=plan['job_sha256'])))
    return directory, approval


def test_approval_drift_resource_and_deadline_refusal(tmp_path, monkeypatch):
    directory, approval = approved_fixture(tmp_path, monkeypatch)
    assert diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()+100)
    monkeypatch.setattr(diagnosis, 'code_identity', lambda root: {'fixture': 'changed'})
    with pytest.raises(ValueError, match='implementation drift'):
        diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()+100)
    monkeypatch.setattr(diagnosis, 'code_identity', lambda root: {'fixture': 'code'})
    with pytest.raises(TimeoutError):
        diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()-1)
    monkeypatch.setattr(diagnosis, 'resource_preflight', lambda root: dict(
        disk_gate_passed=False, host_gate_passed=True, gpu_gate_passed=True))
    with pytest.raises(RuntimeError, match='no model loaded'):
        diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()+100)
    monkeypatch.setattr(diagnosis, 'resource_preflight', lambda root: dict(
        disk_gate_passed=True, host_gate_passed=True, gpu_gate_passed=True))
    (directory/'suite.json').write_text('[{}]')
    with pytest.raises(ValueError, match='suite drift'):
        diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()+100)
    (directory/'suite.json').write_text('[]')
    approval.write_text(json.dumps(dict(format='codexa-inference-baseline-approval-v1', approved=True)))
    with pytest.raises(ValueError, match='approval'):
        diagnosis.validate_job(tmp_path, directory, approval, time.monotonic()+100)


def test_artifact_quota_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnosis, 'LIMIT_BYTES', 10)
    (tmp_path/'owned.json').write_text('x'*11)
    with pytest.raises(RuntimeError, match='allowance'):
        diagnosis.check_artifact_budget(tmp_path)


def test_failed_owned_child_keeps_original_clock(tmp_path, monkeypatch):
    directory, approval = approved_fixture(tmp_path, monkeypatch)
    started = time.monotonic()-90
    seen = {}
    class Failed:
        pid = 123
        def poll(self):
            return 1
    monkeypatch.setattr(diagnosis.subprocess, 'Popen', lambda command, **kw: Failed())
    def watch(child, budget, control):
        seen['deadline'] = budget.deadline_monotonic
        return 1
    monkeypatch.setattr('codexa_workspace.chat_operations.watch_owned_process', watch)
    with pytest.raises(RuntimeError, match='Candidate base failed'):
        diagnosis.run(tmp_path, directory, approval, started)
    assert seen['deadline'] == started+3600
    failure = json.loads((directory/'incomplete.json').read_text())
    assert failure['completed_candidates'] == []
    assert failure['training_updates'] == 0
    assert failure['budget']['deadline_monotonic'] == started+3600


def test_child_rejects_deadline_reset(tmp_path, monkeypatch):
    directory, approval = approved_fixture(tmp_path, monkeypatch)
    supervisor = directory/'supervisor'
    supervisor.mkdir()
    (supervisor/'budget.json').write_text(json.dumps({'deadline_monotonic': time.monotonic()+100}))
    with pytest.raises(ValueError, match='reset'):
        diagnosis.candidate(tmp_path, directory, approval, 'base', time.monotonic()+200)


def test_full_review_pairing_and_drift(tmp_path, tokenizer):
    cases = diagnosis.make_cases(tokenizer)
    results = {}
    for name in ('base','repair','balanced','original_2h','END3'):
        target = tmp_path/name
        target.mkdir()
        rows = [dict(case, response='fixture response', metrics=dict(exact_surface_correct=False)) for case in cases]
        path = target/'outputs.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        results[name] = dict(outputs_path=str(path))
    (tmp_path/'comparison.json').write_text(json.dumps(dict(results=results)))
    package = diagnosis.prepare_blinded_review(tmp_path, results)
    mapping = json.loads((tmp_path/'private-review-mapping.json').read_text())
    export = dict(format='codexa-human-review-export-v1', package_sha256=package['package_sha256'],
        reviewer='Synthetic fixture reviewer; not real approval', method='human_manual_review', decisions={})
    for item in package['items']:
        alias,case_id = item['id'].split(':')
        export['decisions'][item['id']] = dict(decision='approve' if mapping[alias]=='base'
            and case_id.endswith(('-neutral','-direct')) else 'reject')
    decisions = tmp_path/'fixture-review.json'
    decisions.write_text(json.dumps(export))
    result = diagnosis.reviewed_comparison(tmp_path, decisions)
    assert result['reviewed_responses'] == 600
    assert result['corrective_sft_diagnostic_signals'] == [
        dict(candidate='base', form='neutral'), dict(candidate='base', form='direct')]
    assert not result['training_authorized']
    key = next(iter(export['decisions']))
    export['decisions'][key]['decision'] = 'pending'
    decisions.write_text(json.dumps(export))
    with pytest.raises(ValueError, match='Every diagnostic response'):
        diagnosis.reviewed_comparison(tmp_path, decisions)
    export['decisions'][key]['decision'] = 'reject'
    decisions.write_text(json.dumps(export))
    path = Path(results['base']['outputs_path'])
    rows = [json.loads(line) for line in path.open()]
    rows[0]['response'] = 'changed after review'
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    with pytest.raises(ValueError, match='changed after review'):
        diagnosis.reviewed_comparison(tmp_path, decisions)


def test_every_generation_step_checks_finiteness(tmp_path, tokenizer):
    class Nonfinite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(context_length=2048, vocab_size=tokenizer.get_vocab_size())
            self.supports_kv_cache = False
        def forward(self, ids):
            return torch.full((*ids.shape,self.config.vocab_size),float('nan')), None
    model = Nonfinite()
    with pytest.raises(FloatingPointError, match='free-generation'):
        diagnosis.evaluate_model(model, tokenizer, diagnosis.make_cases(tokenizer)[:5],
            tmp_path, time.monotonic()+100, device='cpu')
    assert not model._forward_hooks


def test_bounded_corpus_sample_reads_index_not_whole_data(tmp_path, tokenizer):
    import numpy as np
    target = tmp_path/'data/tokenized/base-v1/mixed'
    target.mkdir(parents=True)
    ids = tokenizer.encode('A small fixture document.', add_special_tokens=False).ids + [2]
    np.asarray(ids*256,dtype=np.uint16).tofile(target/'train.bin')
    records = [dict(source='fineweb_edu' if i%2 else 'wikipedia', document_id=str(i),
        document_ordinal=i, token_start=i*len(ids), token_end=(i+1)*len(ids)) for i in range(256)]
    (target/'train_index.json').write_text('[\n'+',\n'.join(json.dumps(x) for x in records)+'\n]')
    result = diagnosis.corpus_sample(tmp_path,tokenizer)
    assert result['counts'] == dict(fineweb_edu=128, wikipedia=128)
    assert result['index_bytes_read'] < 4*1024**2
    assert all(row['byte_roundtrip_ids_equal'] for row in result['documents'])


def test_cpu_diagnostic_persists_all_forms_without_semantic_scores(tmp_path, tokenizer):
    class Terminates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(context_length=2048, vocab_size=tokenizer.get_vocab_size())
            self.supports_kv_cache = False
        def forward(self, ids):
            logits = torch.zeros((*ids.shape, self.config.vocab_size))
            logits[:,:,-1] = 1
            logits[:,:,2] = 100
            return logits,None
    result = diagnosis.evaluate_model(Terminates(), tokenizer, diagnosis.make_cases(tokenizer)[:5],
        tmp_path, time.monotonic()+100, device='cpu')
    assert result['cases'] == 5 and result['training_updates'] == 0
    rows = [json.loads(line) for line in (tmp_path/'outputs.jsonl').open()]
    assert {x['form'] for x in rows} == set(diagnosis.FORMS)
    assert all(x['metrics']['semantic_correct'] is None for x in rows)
    assert all(x['terminator'] == 2 and x['raw_token_ids'] == [2] for x in rows)


def test_screening_review_cannot_authorize_dataset_or_model(tmp_path):
    from codexa_workspace.qualification_review import write_review_page, import_decisions
    package = write_review_page(tmp_path/'screening', dict(kind='screening', items=[]))
    decisions = tmp_path/'decisions.json'
    decisions.write_text(json.dumps(dict(package_sha256=package['package_sha256'],
        reviewer='Synthetic fixture reviewer', method='human_manual_review', decisions={})))
    with pytest.raises(ValueError, match='cannot authorize'):
        import_decisions(tmp_path/'screening/package.json', decisions, tmp_path/'no-admission.json')
    assert not (tmp_path/'no-admission.json').exists()
