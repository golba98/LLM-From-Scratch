"""Checks for reproducible paired regression diagnosis."""

import pytest

from codexa_workspace.chat_investigation import paired_changes


def test_paired_changes_attribute_improvement_and_regression():
    def row(identity, passed, repeated, stopped):
        return dict(id=identity, category='instruction', prompt='Say yes.', response='yes',
                    metrics=dict(task_pass=passed, severe_degeneration=repeated, ended_with_END=stopped))
    before = [row('a', True, False, True), row('b', False, True, False)]
    after = [row('b', True, False, True), row('a', False, True, False)]
    counts = paired_changes(before, after)['counts']
    assert counts == dict(task_improved=1, task_regressed=1, degeneration_new=1,
                          degeneration_fixed=1, lost_END=1, gained_END=1)


def test_paired_changes_reject_population_drift():
    with pytest.raises(ValueError, match='populations differ'):
        paired_changes([{'id': 'a'}], [{'id': 'b'}])
