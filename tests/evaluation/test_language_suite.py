"""The frozen language suite: case construction, exact-surface scoring and likelihood probes (tiny CPU fixtures)."""

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from codexa_workspace.evaluation import language_suite as suite
from llm_tokenizer.tokenizer import SPECIAL_TOKENS


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([str(suite.concepts())], trainers.BpeTrainer(
        vocab_size=512, special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tokenizer


def test_matched_forms_do_not_inject_bos(tokenizer):
    rows = suite.make_cases(tokenizer)
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
    for concept in suite.concepts():
        matched = [x for x in rows if x['concept'] == concept['id']]
        assert len(matched) == 5
        assert {x['expected'] for x in matched} == {concept['expected']}


def test_invalid_or_duplicate_arithmetic_reference_is_rejected(tokenizer, monkeypatch):
    original = suite.concepts()
    mutated = [dict(x) for x in original]
    next(x for x in mutated if x['category']=='arithmetic')['expected'] = '999'
    monkeypatch.setattr(suite,'concepts',lambda:mutated)
    with pytest.raises(ValueError,match='reference'):
        suite.make_cases(tokenizer)


def test_keyword_and_contradiction_are_not_semantic_success():
    case = dict(expected='Rome', category='factual', form='chat')
    score = suite.scoring('Rome is the capital of France.', case, 7)
    assert not score['exact_surface_correct']
    assert score['semantic_correct'] is None
    assert score['ended_with_END']
    assert suite.scoring('Rome', case, 7)['exact_surface_correct']
    plain = suite.scoring('Rome', dict(case, form='neutral'), None)
    assert plain['missing_chat_END'] is None
    assert not plain['chat_turn_stop_expected']


def test_arithmetic_exactness_and_case():
    case = dict(expected='63', category='arithmetic', form='short')
    assert suite.scoring('63', case, 7)['arithmetic_exact_correct']
    assert not suite.scoring('The result is 63, or perhaps 53.', case, 7)['arithmetic_exact_correct']
    case.update(expected='C', category='factual')
    assert not suite.scoring('c', case, 7)['exact_surface_correct']


def test_likelihood_requires_finite_values(tokenizer):
    case = suite.make_cases(tokenizer)[0]
    class Nonfinite:
        def __call__(self, ids):
            return torch.full((*ids.shape, tokenizer.get_vocab_size()), float('nan')), None
    with pytest.raises(FloatingPointError):
        suite.likelihood_probe(Nonfinite(), tokenizer, case, device='cpu')


def test_cache_and_uncached_cpu_generation_agree(tokenizer):
    from llm_architecture.model import LanguageModel, ModelConfig
    from llm_inference.generate import GenerationConfig, generate_sequences
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        model = LanguageModel(ModelConfig(vocab_size=tokenizer.get_vocab_size(), context_length=512,
            num_layers=1, hidden_size=16, num_heads=2, intermediate_size=32)).eval()
        case = suite.make_cases(tokenizer)[0]
        ids = torch.tensor([case['input_ids']])
        results = [generate_sequences(model, ids, eos_token_id=2, pad_token_id=0,
            stop_sequences=[[7]], config=GenerationConfig(max_new_tokens=8, use_kv_cache=cache))
            for cache in (True, False)]
        assert results[0].sequences[0].token_ids == results[1].sequences[0].token_ids
        with torch.inference_mode():
            probes = suite.likelihood_probe(model, tokenizer, case, device='cpu')
        assert probes['expected']['mean_token_nll'] > 0
        assert 0 <= probes['expected']['first_token_probability'] <= 1
    finally:
        torch.set_num_threads(original_threads)


def test_screening_review_cannot_authorize_dataset_or_model(tmp_path):
    from codexa_workspace.review.qualification import write_review_page, import_decisions
    package = write_review_page(tmp_path/'screening', dict(kind='screening', items=[]))
    decisions = tmp_path/'decisions.json'
    decisions.write_text(json.dumps(dict(package_sha256=package['package_sha256'],
        reviewer='Synthetic fixture reviewer', method='human_manual_review', decisions={})))
    with pytest.raises(ValueError, match='cannot authorize'):
        import_decisions(tmp_path/'screening/package.json', decisions, tmp_path/'no-admission.json')
    assert not (tmp_path/'no-admission.json').exists()
