"""Historical repair leakage patterns cannot recur in prospective split creation."""

import pytest

from llm_data.data.repair_split import split_repair_records


def make_row(index, *, root=None, content=None):
    return dict(source='oasst1', root_id=root or f'root-{index}', conversation_id=f'item-{index}',
                messages=[dict(role='user', content=f'Question {index}'),
                          dict(role='assistant', content=content or f'Answer {index}')])


def test_root_and_duplicate_bridges_stay_together_with_reproducible_order_independent_splits():
    rows = [make_row(i) for i in range(40)]
    rows[1]['root_id'] = rows[0]['root_id']
    rows[2]['messages'] = rows[1]['messages']
    rows[3]['root_id'] = rows[2]['root_id']
    train, validation = split_repair_records(rows, seed=42)
    assignments = {row['conversation_id']: name for name, values in [('train', train), ('validation', validation)]
                   for row in values}
    assert len({assignments[f'item-{i}'] for i in range(4)}) == 1
    reverse_train, reverse_validation = split_repair_records(list(reversed(rows)), seed=42)
    assert {row['conversation_id'] for row in train} == {row['conversation_id'] for row in reverse_train}
    assert {row['duplicate_group'] for row in train}.isdisjoint(row['duplicate_group'] for row in validation)
    assert {row['root_id'] for row in train}.isdisjoint(row['root_id'] for row in validation)


def test_repeated_anchor_content_under_distinct_ids_is_not_a_validation_population():
    rows = [make_row(i, root='same') for i in range(8)]
    with pytest.raises(ValueError, match='empty repair split'):
        split_repair_records(rows, seed=42)


def test_missing_provenance_rejects_before_output():
    rows = [make_row(i) for i in range(40)]
    del rows[0]['root_id']
    with pytest.raises(ValueError, match='identities'):
        split_repair_records(rows, seed=42)


def test_preexisting_duplicate_components_are_preserved():
    rows = [make_row(i) for i in range(40)]
    rows[7]['duplicate_group'] = rows[19]['duplicate_group'] = 'known-related-component'
    train, validation = split_repair_records(rows, seed=42)
    assignments = {row['conversation_id']: name for name, values in [('train', train), ('validation', validation)]
                   for row in values}
    assert assignments['item-7'] == assignments['item-19']
