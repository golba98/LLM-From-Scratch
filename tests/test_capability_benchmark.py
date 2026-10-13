"""Frozen capability benchmark: suite identity, order control, statistics and paired comparison.

No model is loaded; scoring is exercised with a tiny deterministic fake where needed.
"""
import json

import pytest

from codexa_workspace import capability_benchmark as bench


def test_suites_are_frozen_and_large_enough():
    suites = bench.load_suites()
    items = suites['items']
    assert suites['identity']['v1']['content_sha256'] == bench.SUITES['v1'][1]
    for category in bench.CATEGORIES:
        new = [i for i in items if i['suite'] == 'v2' and i['category'] == category]
        old = [i for i in items if i['suite'] == 'v1' and i['category'] == category]
        assert len(new) >= 30 and len(old) == 8
        for item in new:
            assert item['gold'] not in item['distractors'] and len(set(item['distractors'])) == len(item['distractors'])
    v1_prompts = {i['plain'] for i in items if i['suite'] == 'v1'}
    assert not any(i['plain'] in v1_prompts for i in items if i['suite'] == 'v2' and i['plain'])


def test_v1_is_byte_identical_to_the_original_probe_suite():
    import importlib.util
    from pathlib import Path
    root = Path(bench.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('probe', root / 'scripts/foundation_capability_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    document = json.loads((bench.BENCHMARKS / 'foundation_capability_v1.json').read_text())
    assert document['items'] == probe.PROBES and document['content_sha256'] == probe.suite_sha256()


def test_tampered_suite_is_refused(tmp_path, monkeypatch):
    document = json.loads((bench.BENCHMARKS / 'foundation_capability_v2.json').read_text())
    document['items'][0]['gold'] = 'Bonn'
    (tmp_path / 'foundation_capability_v2.json').write_text(json.dumps(document))
    monkeypatch.setattr(bench, 'BENCHMARKS', tmp_path)
    with pytest.raises(ValueError, match='drifted'):
        bench.load_suites(('v2',))


def test_grammar_options_render_in_both_orders():
    item = next(i for i in bench.load_suites(('v2',))['items'] if i['category'] == 'grammar')
    plain = bench.variants(item, 'plain')
    assert plain == [item['plain'].format(options='has/have'), item['plain'].format(options='have/has')]
    chat = bench.variants(item, 'chat')
    assert 'has or have' in chat[0] and 'have or has' in chat[1] and chat[0].startswith('<|user|>')
    assert bench.variants(dict(plain=None, chat='Hi'), 'plain') == []


def test_wilson_and_mcnemar():
    low, high = bench.wilson(16, 32)
    assert 0.33 < low < 0.34 and 0.66 < high < 0.67
    assert bench.wilson(0, 0) == (None, None)
    result = bench.mcnemar(gained=0, lost=10)
    assert result['p_loss_one_sided'] == 1 / 1024 and result['p_two_sided'] == 2 / 1024
    assert bench.mcnemar(3, 3)['p_two_sided'] == 1.0


def row(identifier, top1, margin, category='factual', form='plain'):
    return dict(id=identifier, category=category, form=form, variants=[{}], top1=top1, margin=margin,
                candidate_count=5, top1_length_normalized=top1, order_sensitive=False, gold_candidate_probability=.5)


def test_compare_requires_identical_items_and_flags_only_significant_loss():
    base = [row(f'i{n}', True, 1.0) for n in range(40)]
    noisy = [row(f'i{n}', n >= 3, 0.9) for n in range(40)]                     # 3 lost: not significant
    assert bench.compare(base, noisy)['degraded'] is False
    bad = [row(f'i{n}', n >= 12, 0.5) for n in range(40)]                      # 12 lost, 0 gained
    result = bench.compare(base, bad)
    assert result['degraded'] is True and result['overall']['p_loss_one_sided'] < 0.01
    improved_margin = [row(f'i{n}', n >= 12, 1.5) for n in range(40)]         # loss with higher margin
    assert bench.compare(base, improved_margin)['degraded'] is False
    with pytest.raises(ValueError, match='identical'):
        bench.compare(base, bad[:-1])


def test_summary_counts_and_classification():
    rows = [row('a', True, 1.0), row('b', False, -1.0, category='arithmetic')]
    rows[0]['generations'] = {'greedy': dict(stated_gold=True, exact=False, termination=dict(natural=False),
                                             severe_repetition=True, classification='knows_wrong_format')}
    summary = bench.summarize(rows)['plain']
    assert summary['overall']['n'] == 2 and summary['overall']['top1'] == 1
    assert summary['categories']['factual']['generation:greedy']['classes']['knows_wrong_format'] == 1
    assert bench.classify(top1=True, stated_gold=False, exact=False) == 'knows_generates_other'
    assert bench.first_candidate('It is Tokyo, not Seoul.', ['Tokyo', 'Seoul']) == 'Tokyo'


def test_baseline_binding_detects_tampering(tmp_path):
    rows_path = tmp_path / 'rows.jsonl'
    rows_path.write_text(json.dumps(row('a', True, 1.0)) + '\n')
    import hashlib
    document = dict(format='x', capability=dict(rows_path=str(rows_path),
                                                rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest()))
    document['summary_sha256'] = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    (tmp_path / 'summary.json').write_text(json.dumps(document))
    loaded = bench.load_baseline(tmp_path / 'summary.json', document['summary_sha256'])
    assert loaded['rows'][0]['id'] == 'a'
    with pytest.raises(ValueError, match='identity'):
        bench.load_baseline(tmp_path / 'summary.json', '0' * 64)
    rows_path.write_text(json.dumps(row('a', False, 1.0)) + '\n')
    with pytest.raises(ValueError, match='modified'):
        bench.load_baseline(tmp_path / 'summary.json', document['summary_sha256'])
