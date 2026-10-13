"""Step 5A verification and the GO / NO-GO classification.

Real-asset checks skip when the ignored Phase 4 directory is absent; verdict logic is synthetic.
"""
import json
from pathlib import Path

import pytest

from codexa_workspace import sft_phase4_data as data
from codexa_workspace import sft_readiness as readiness
from codexa_workspace import sft_step5a_verify as verify

MANIFEST = verify.PHASE4 / 'manifest/phase4-manifest.json'
SUMMARY = verify.PHASE4 / 'evaluation/step-zero/summary.json'
needs_phase4 = pytest.mark.skipif(not MANIFEST.exists(), reason='Phase 4 frozen manifest not present (ignored asset)')


def check(name, passed, kind=readiness.ENGINEERING):
    return dict(category='x', name=name, kind=kind, passed=passed)


def test_verdict_classification_never_upgrades():
    assert readiness.classify([check('a', True), check('b', True, readiness.LAUNCH_TIME)])[0] == readiness.READY
    assert readiness.classify([check('a', True), check('vram', False, readiness.LAUNCH_TIME)])[0] == readiness.CONDITIONAL
    assert readiness.classify([check('a', False), check('vram', False, readiness.LAUNCH_TIME)])[0] == readiness.NOT_READY
    assert readiness.classify([check('a', False, 'unknown-kind')])[0] == readiness.NOT_READY
    assert readiness.classify([])[0] == readiness.NOT_READY


def test_documented_probe_uses_the_slowest_row():
    probe = verify._documented_probe('dynamic-short-m1')
    assert probe['rows'] >= 1 and probe['median_microbatch_seconds'] >= 0.146
    with pytest.raises(ValueError):
        verify._documented_probe('no-such-probe')


@needs_phase4
def test_step_zero_binding_recomputes_the_sealed_hash(tmp_path):
    binding = verify.step_zero_binding(SUMMARY, MANIFEST)
    assert binding['checks']['summary_hash'] and binding['checks']['no_training'] and binding['checks']['checkpoint']
    tampered = data.read(SUMMARY)
    tampered['gold']['mean_gold_end_nll'] = 0.0
    (tmp_path / 'summary.json').write_text(json.dumps(tampered))
    assert not verify.step_zero_binding(tmp_path / 'summary.json', MANIFEST)['checks']['summary_hash']


@needs_phase4
def test_lineage_has_no_sft_ancestor_but_nothing_is_called_unseen():
    lineage = verify.lineage_exposure(MANIFEST)
    assert lineage['sft_model_in_initialization_lineage'] is False
    assert lineage['corpus_sources'] == ['fineweb_edu', 'wikipedia'] and lineage['continuation_is_range_of_mixed']
    assert 'unseen' in lineage['classification'] and 'cannot be excluded' in lineage['classification']
    assert lineage['flags'] == dict(train_historical_sft_exposure=445, validation_historical_sft_exposure=32)


@needs_phase4
def test_validation_exception_is_isolated_from_training():
    result = verify.validation_exception(MANIFEST, '0c903975-f9da-4938-b897-07a6f1766390')
    assert not any(result['shared_with_training'].values()) and not result['near_duplicate_of_training']
    assert result['target_five_gram_shingles'] > 0 and result['isolation_policy_satisfied']


@needs_phase4
def test_independent_dataset_rederivation():
    result = verify.verify_dataset(MANIFEST)
    assert result['verified'], {k: v for k, v in result['checks'].items() if not v}
    assert result['train']['supervised_tokens'] == data.read(MANIFEST)['token_accounting']['train']['total']['supervised_tokens']
    assert result['train_two_passes']['presentations'] == 1024
    assert result['token_ids'] == dict(pad=0, bos=1, eos=2, system=4, user=5, assistant=6, end=7)


def test_report_without_evidence_is_not_ready(tmp_path):
    from codexa_workspace.sft_step5a_report import TITLE, READY_LINE, write_report
    result = write_report(tmp_path)
    text = (tmp_path / TITLE).read_text()
    assert result['verdict'].startswith('NOT READY') and READY_LINE not in text
    assert '## L. Authorization handoff' in text
