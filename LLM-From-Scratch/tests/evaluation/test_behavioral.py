"""Correct answers require complete compliance, not isolated reference terms."""

import pytest

from codexa_workspace.evaluation.behavioral import (development_suite, acceptance_suite,
    strict_answer, strict_score, strict_summary, paired_difference, suite_sha256,format_compliance)


def test_suites_are_disjoint_versioned_and_separate_from_curriculum():
    dev,accept=development_suite(),acceptance_suite()
    assert len(dev)==240 and len(accept)==120
    assert suite_sha256(dev)==suite_sha256(development_suite())
    assert not {r['prompt'] for r in dev}&{r['prompt'] for r in accept}
    assert not {r['family'] for r in dev}&{r['family'] for r in accept}
    assert all(r['protected'] for r in accept)


def test_factual_keyword_and_contradictions_are_not_correct():
    case={'check':'name','expected':'the heart'}
    assert strict_answer('Heart.',case)
    assert not strict_answer('The heart is not the answer; the brain is.',case)
    assert not strict_answer('heart heart heart heart',case)


def test_exact_math_rejects_extra_results_or_prose():
    case={'check':'number','expected':'42'}
    assert strict_answer('42.0',case)
    assert not strict_answer('7 or 42',case)
    assert not strict_answer('The answer is 42 and also 43.',case)


def test_format_requires_single_json_and_rejects_duplicate_keys():
    case={'check':'json','expected':{'label':'X'}}
    assert strict_answer('{"label": "X"}',case)
    assert not strict_answer('Here: {"label": "X"}',case)
    assert not strict_answer('{"label":"wrong","label":"X"}',case)
    assert not strict_answer('{"label":"X","extra":1}',case)
    assert not strict_answer('SIGNAL', {'check':'exact','expected':'signal'})


def test_entry_requires_correct_end_and_special_token_fraction():
    rows=[]
    for category in ('instruction','arithmetic','factual'):
        metrics=strict_score('YES',[8,7],7,{'check':'exact','expected':'YES'})
        rows.append(dict(id=category,category=category,metrics=metrics,
                         raw_token_ids=[8,7],generated_tokens=2,termination='stop'))
    assert strict_summary(rows)['entry_gate_passed']
    for row in rows:
        row['metrics']['correct_END']=False
    assert not strict_summary(rows)['entry_gate_passed']
    rows[0]['raw_token_ids']=[4,6,7,8]
    assert strict_summary(rows)['special_token_dominated_rate']==pytest.approx(1/3)


def test_format_shape_is_separate_from_wrong_answer():
    case={'category':'instruction','family':'dev-json','check':'json','expected':{'label':'X'}}
    assert format_compliance('{"label":"wrong"}',case)
    assert not strict_answer('{"label":"wrong"}',case)
    assert not format_compliance('{"label":"X","extra":1}',case)
    assert not format_compliance('{"label":"X","label":"X"}',case)


def test_family_bootstrap_rejects_changed_family():
    row=dict(id='one',category='instruction',prompt='x',history=[],family='original',
             metrics={'task_pass':True})
    with pytest.raises(ValueError,match='contents differ'):
        paired_difference([row],[{**row,'family':'changed'}])


def test_human_review_and_premature_end_are_not_fabricated():
    human={'check':'human','expected':'','category':'retention'}
    assert strict_answer('A coherent explanation',human) is None
    empty=strict_score('',[7],7,{'check':'exact','expected':'YES'})
    assert empty['premature_END_candidate'] and not empty['correct_END']
    wrong=strict_score('7',[8,7],7,{'check':'number','expected':'42'})
    assert wrong['incorrect_answer_END'] and not wrong['premature_END_candidate']
    scored=strict_score('YES',[8,7],7,{'check':'exact','expected':'YES'})
    row={'id':'one','category':'instruction','metrics':scored,'raw_token_ids':[8,7],
         'generated_tokens':2,'termination':'stop'}
    humanrow={**row,'id':'two','category':'retention','metrics':strict_score('Hi',[8,7],7,human)}
    summary=strict_summary([row,humanrow])
    assert summary['objective_cases']==1 and summary['human_review_pending']==1
    assert paired_difference([row],[row])['ci95']==[0,0]
    with pytest.raises(ValueError):
        paired_difference([row],[humanrow])
