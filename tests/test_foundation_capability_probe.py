"""Capability probe: suite identity and the knowledge/format classification (no model is loaded)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('foundation_capability_probe', ROOT / 'scripts/foundation_capability_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_suite_is_complete_and_unambiguous():
    categories = {}
    for item in probe.PROBES:
        categories[item['category']] = categories.get(item['category'], 0) + 1
        if item['gold'] is not None:
            assert item['gold'] not in item['distractors'] and len(set(item['distractors'])) == len(item['distractors'])
            assert item['plain'] and item['chat']
    assert categories == dict(factual=8, arithmetic=8, copy_extract=8, sentence_completion=8, grammar=8, reasoning=8,
                              exact_format=8, conversational=8)
    assert len({item['id'] for item in probe.PROBES}) == len(probe.PROBES)
    assert probe.suite_sha256() == probe.suite_sha256()


def test_chat_prompt_matches_the_sft_serialization():
    assert probe.chat_prompt('Hi') == '<|user|>Hi<|end|><|assistant|>'


def test_first_candidate_prefers_the_earliest_whole_word_mention():
    candidates = ['Paris', 'Rome', 'London']
    assert probe.first_candidate('The capital of France is Paris, not Rome.', candidates) == 'Paris'
    assert probe.first_candidate('Rome. Paris is elsewhere.', candidates) == 'Rome'
    assert probe.first_candidate('Parisian food', candidates) is None             # no partial-word matches
    assert probe.first_candidate('paris', candidates) == 'Paris'                  # case-insensitive fallback
    assert probe.first_candidate('4 apples', ['4', '14']) == '4'


def test_classification_separates_format_generation_and_knowledge_failures():
    assert probe.classify(gold_rank=1, stated_gold=True, exact=True) == 'correct_exact'
    assert probe.classify(gold_rank=1, stated_gold=True, exact=False) == 'knows_wrong_format'
    assert probe.classify(gold_rank=1, stated_gold=False, exact=False) == 'knows_generates_other'
    assert probe.classify(gold_rank=3, stated_gold=True, exact=False) == 'does_not_know'
    assert probe.classify(gold_rank=2, stated_gold=False, exact=False) == 'does_not_know'
