"""Offline admission/scoring checks. Never invoke a real trainer or optimizer."""
import copy
import sys
import sqlite3
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from codexa_workspace import sft_feasibility as pilot


def test_authored_answers_and_wrong_answers():
    rows=pilot.authored_specs()
    assert len(rows)==64 and len({r['prompt'] for r in rows})==64
    for row in rows:
        assert pilot.correct(row)
        wrong=copy.deepcopy(row)
        wrong['answer']='Fluent but incorrect answer.'
        assert not pilot.correct(wrong)
    arithmetic=[r['answer'] for r in rows if r['category']=='arithmetic']
    assert len(set(arithmetic))==16


@pytest.mark.parametrize('response',['17.',' 17','17\n','17; actually it is 18.','18'])
def test_surface_accuracy_rejects_cleanup_or_contradiction(response):
    assert not pilot.strict_reference_success(response,'17')
    assert pilot.strict_reference_success('17','17')


def test_natural_termination_does_not_substitute_for_usefulness():
    assert pilot.usefulness_gate([])=='PENDING'
    unscored=[dict(natural_end=True,severe_repetition=False,reviewer=None) for _ in range(16)]
    assert pilot.usefulness_gate(unscored)=='PENDING'
    reviews=[dict(reviewer='real-review-fixture',blinded=True,relevance=True,
        coherence=True,factual_context_consistency=False,paired_preference='tie') for _ in range(16)]
    assert pilot.usefulness_gate(reviews)=='FAIL'
    for row in reviews: row['factual_context_consistency']=True
    for row in reviews[:4]:row['paired_preference']='pilot'
    assert pilot.usefulness_gate(reviews)=='PASS'
    for row in reviews[4:6]:row['paired_preference']='baseline'
    assert pilot.usefulness_gate(reviews)=='FAIL'


def test_preparation_does_not_import_training_execution():
    source=Path(pilot.__file__).read_text()
    assert 'import llm_training.training' not in source
    assert 'import llm_training.chat_run' not in source
    assert 'torch.optim' not in source


def test_baseline_refuses_unfrozen_gold_set(tmp_path):
    with pytest.raises(RuntimeError,match='not frozen and reviewed'):
        pilot.baseline(tmp_path)


def test_reviewed_sql_example_on_unsorted_fixture():
    with sqlite3.connect(':memory:') as database:
        database.execute('CREATE TABLE Students (grade INTEGER)')
        grades=[43,91,7,65,88,12,56,73,29,100,34,81]
        database.executemany('INSERT INTO Students VALUES (?)',[(n,) for n in grades])
        ascending=[x[0] for x in database.execute('SELECT * FROM Students ORDER BY grade ASC;')]
        highest=[x[0] for x in database.execute('SELECT * FROM Students ORDER BY grade DESC LIMIT 10;')]
    assert ascending==sorted(grades)
    assert highest==sorted(grades,reverse=True)[:10]
