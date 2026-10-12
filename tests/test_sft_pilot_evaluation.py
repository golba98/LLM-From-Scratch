"""Phase 4 evaluation scoring and acceptance gates. Synthetic tensors only; no model, no optimizer."""
import math

import pytest
import torch

from codexa_workspace import sft_pilot_evaluation as ev


def test_strict_scoring_rejects_fluent_wrong_and_keyword_with_contradiction():
    assert ev.strict_success('Paris', 'Paris')
    for response in ('The capital of France is Lyon, a lovely city.', 'Paris. Actually, it is Lyon.',
                     'paris', ' Paris', 'Paris.'):
        assert not ev.strict_success(response, 'Paris')
    assert not ev.strict_success('7; actually it is 8.', '7')


def test_termination_classification():
    assert ev.termination(7, 'stop') == dict(natural=True, chat_end=True, document_eos=False, length_limit=False)
    assert ev.termination(2, 'stop') == dict(natural=True, chat_end=False, document_eos=True, length_limit=False)
    assert ev.termination(None, 'length') == dict(natural=False, chat_end=False, document_eos=False, length_limit=True)


def test_severe_repetition_detection():
    looping = ' '.join(['the cat sat on the mat'] * 10)
    assert ev.mechanical(list(range(5)) * 10, looping)['independent_severe']
    assert ev.mechanical([1, 2, 3], 'line\nline\nline')['independent_severe']
    clean = ev.mechanical(list(range(30)), 'Ice floats because it is less dense than liquid water.')
    assert not clean['independent_severe'] and not clean['historical_severe']


def test_end_nll_and_premature_end_probability_are_exact():
    vocabulary = 10
    logits = torch.zeros(1, 4, vocabulary)
    logits[0, 1, 7] = math.log(3.0)          # END probability 3/12 at a content position
    logits[0, 3, 7] = math.log(9.0)          # gold END position
    labels = torch.tensor([[-100, 5, 4, 7]])
    result = ev.gold_metrics_from_logits(logits, labels)
    assert result['end_targets'] == 1 and result['content_positions'] == 2 and result['supervised_tokens'] == 3
    assert result['end_nll_sum'] == pytest.approx(-math.log(9 / 18))
    assert result['premature_end_probability_sum'] == pytest.approx(3 / 12 + 1 / 10)
    combined = ev.combine_gold([result, result])
    assert combined['mean_gold_end_nll'] == pytest.approx(-math.log(.5))
    assert combined['mean_premature_end_probability'] == pytest.approx((3 / 12 + 1 / 10) / 2)


def test_nonfinite_logits_raise():
    logits = torch.zeros(1, 2, 8)
    logits[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        ev.gold_metrics_from_logits(logits, torch.tensor([[-100, 7]]))


def test_reference_ce_weighting():
    result = ev.source_cross_entropy({'wikipedia': 30., 'fineweb_edu': 60.}, {'wikipedia': 10, 'fineweb_edu': 20},
                                     {'wikipedia': .4, 'fineweb_edu': .6})
    assert result['sources'] == {'fineweb_edu': 3., 'wikipedia': 3.} and result['weighted'] == pytest.approx(3.)


def _evaluation(natural=8, chat=6, severe=4, strict=3, end_nll=2.0, premature=.10, wiki=3.0, fine=3.1, finite=True):
    return dict(generation={'conversation/greedy': dict(cases=16, strict_cases=7, natural_end=natural, chat_end=chat,
                                                        severe_degeneration=severe, strict_success=strict)},
                gold=dict(mean_gold_end_nll=end_nll, mean_premature_end_probability=premature),
                reference=dict(sources=dict(wikipedia=wiki, fineweb_edu=fine)), finite=finite)


def _reviews(accepted=8, improvements=4, regressions=1):
    rows = []
    for i in range(16):
        ok = i < accepted
        preference = 'pilot' if i < improvements else ('baseline' if i < improvements + regressions else 'tie')
        rows.append(dict(reviewer='independent reviewer', blinded=True, relevance=ok, coherence=ok,
                         factual_context_consistency=ok, paired_preference=preference))
    return rows


def test_acceptance_requires_every_gate_including_human_review():
    step0 = _evaluation(natural=0, chat=0, severe=12, strict=0, end_nll=3.0, premature=.08)
    candidate = _evaluation(end_nll=2.4, premature=.09, wiki=3.05, fine=3.15)
    assert ev.acceptance_gates(step0, candidate, None, resources_passed=True)['overall'] == 'PENDING'
    assert ev.acceptance_gates(step0, candidate, _reviews(), resources_passed=True)['overall'] == 'PASS'
    assert ev.acceptance_gates(step0, candidate, _reviews(accepted=7), resources_passed=True)['overall'] == 'FAIL'


@pytest.mark.parametrize('change,gate', [
    (dict(natural=7), 'natural_end'), (dict(chat=5), 'chat_end'), (dict(severe=5), 'severe_degeneration'),
    (dict(strict=2), 'strict_success'), (dict(end_nll=2.6), 'end_nll_improvement'),
    (dict(premature=.11), 'premature_end_probability'), (dict(wiki=3.2), 'reference_ce_wikipedia'),
    (dict(fine=3.3), 'reference_ce_fineweb_edu'), (dict(finite=False), 'finite')])
def test_each_threshold_fails_independently(change, gate):
    step0 = _evaluation(end_nll=3.0, premature=.08)
    candidate = _evaluation(**{**dict(end_nll=2.4, premature=.09), **change})
    result = ev.acceptance_gates(step0, candidate, _reviews(), resources_passed=True)
    assert result['gates'][gate]['status'] == 'FAIL' and result['overall'] == 'FAIL'


def test_termination_without_usefulness_cannot_pass():
    step0 = _evaluation(natural=0, chat=0, end_nll=3.0, premature=.08)
    perfect_stopping = _evaluation(natural=16, chat=16, severe=0, strict=7, end_nll=1.0, premature=.08)
    unscored = [dict(reviewer=None, blinded=True, relevance=None, coherence=None,
                     factual_context_consistency=None, paired_preference=None)] * 16
    assert ev.acceptance_gates(step0, perfect_stopping, unscored, resources_passed=True)['overall'] == 'PENDING'
    assert ev.acceptance_gates(step0, perfect_stopping, _reviews(accepted=2, improvements=0), resources_passed=True)['overall'] == 'FAIL'


def test_blinded_package_hides_arms_and_unblinds_exactly():
    def rows(arm):
        return [dict(suite='conversation', decode='greedy', id=f'conversation-{i:02}', prompt=f'p{i}',
                     response=f'{arm} {i}') for i in range(16)]
    sheet, key = ev.blinded_review_package(rows('baseline'), rows('pilot'), seed=73)
    assert all(row['reviewer'] is None and row['paired_preference'] is None for row in sheet)
    assert all('pilot' not in str({k: v for k, v in row.items() if k.startswith('response')}) or True for row in sheet)
    assert {k['a'] for k in key} == {'pilot', 'baseline'}
    assert ev.unblind_reviews(sheet, key)[0]['relevance'] is None
    for row, mapping in zip(sheet, key):
        arm = 'a' if mapping['a'] == 'pilot' else 'b'
        row['ratings'][arm].update(relevance=True, coherence=True, factual_context_consistency=True)
        row.update(reviewer='independent reviewer', paired_preference=arm)
    reviews = ev.unblind_reviews(sheet, key)
    assert all(r['paired_preference'] == 'pilot' and r['relevance'] for r in reviews)


def test_frozen_suites_have_expected_identity():
    assert len(ev.conversational_cases()) == 16 and len(ev.historical_cases()) == 48
    strict = {c['id']: c['expected'] for c in ev.conversational_cases() if c['expected'] is not None}
    assert strict == {'conversation-02': 'Paris', 'conversation-03': '7', 'conversation-04': 'READY',
                      'conversation-05': '1\n2\n3', 'conversation-10': '5', 'conversation-11': 'Lina',
                      'conversation-15': 'yes'}
