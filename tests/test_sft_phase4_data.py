"""Phase 4 SFT data: screening helpers (synthetic) and the frozen manifest (real local assets).

Real-asset tests skip when the ignored Phase 4 output directory or tokenizer is absent.
"""
import json
from collections import Counter
from pathlib import Path
import random
import shutil

import pytest

from codexa_workspace import sft_phase4_data as d


def _phase4_directory():
    if not d.BASE.exists():
        return None
    candidates = sorted(p for p in d.BASE.iterdir() if (p / 'manifest/phase4-manifest.json').exists())
    return candidates[-1] if candidates else None


DIRECTORY = _phase4_directory()
needs_manifest = pytest.mark.skipif(DIRECTORY is None, reason='Phase 4 frozen manifest not present (ignored asset)')


def _row(text, identity='x', role_texts=None):
    return dict(conversation_id=identity, messages=role_texts or [dict(role='user', content=text),
                                                                  dict(role='assistant', content='Fine.')])


def test_non_english_prefilter_only_flags_clear_cases():
    spanish = _row('La fotosíntesis es el proceso mediante el cual las plantas verdes convierten la luz solar '
                   'en energía química que se almacena en moléculas de glucosa dentro de sus células.')
    mixed = _row('¿Cuál es la mejor manera de aprender a programar en Python para principiantes que no saben nada?')
    english = _row('What is the best way to learn programming in Python for beginners who know nothing at all?')
    short = _row('Hola amigo')
    assert d.prefilter_reject(spanish)
    assert d.prefilter_reject(english) is None
    assert d.prefilter_reject(short) is None
    assert d.prefilter_reject(mixed) is None  # shares English function words: left for review, never auto-admitted


def test_contamination_screen_hard_and_soft():
    screen = d.ContaminationScreen(['What is the capital of France?',
                                   'Explain why ice floats on water in two sentences.'])
    assert screen.screen(_row('What is the capital of France?'))['hard_reject']
    assert screen.screen(_row('Please explain why ice floats on water in two sentences for me'))['hard_reject']
    clean = screen.screen(_row('Name three primary colours used in painting.'))
    assert not clean['hard_reject'] and not clean['possible_semantic_overlap']


def test_near_duplicate_and_template_helpers():
    a = d.shingles('the quick brown fox jumps over the lazy dog today', 5)
    assert d.jaccard(a, a) == 1 and d.jaccard(a, set()) == 0
    template = 'Based on the text material above, generate the response to the following question or instruction'
    candidates = {str(i): dict(record=_row(f'{template}: topic number {i} with extra words here')) for i in range(12)}
    assert d.shingles(template) <= d.template_shingles(candidates)


def test_authored_records_regenerate_exactly():
    tokenizer_path = Path(d.tokenizer_identity()['path'])
    if not tokenizer_path.exists():
        pytest.skip('tokenizer asset absent')
    from llm_tokenizer.tokenizer import load_tokenizer
    rows = list(d.jsonlines(d.PRIOR / 'data/authored.jsonl'))
    checked = d.verify_authored(rows, load_tokenizer(tokenizer_path))
    assert len(checked) == 64 and Counter(r['authored_kind'] for r in checked) == dict.fromkeys(
        ('word-only', 'formatting', 'arithmetic', 'supplied-context'), 16)
    tampered = [dict(rows[0], messages=[rows[0]['messages'][0], dict(role='assistant', content='wrong')])] + rows[1:]
    with pytest.raises(ValueError):
        d.verify_authored(tampered, load_tokenizer(tokenizer_path))


@needs_manifest
def test_frozen_composition_identity_and_isolation():
    manifest = d.read(DIRECTORY / 'manifest/phase4-manifest.json')
    train = [r for r in manifest['records'] if r['split'] == 'train']
    validation = [r for r in manifest['records'] if r['split'] == 'validation']
    assert len(train) == 512 and len({r['conversation_id'] for r in train}) == 512
    assert Counter(r['source'] for r in train) == dict(oasst1=224, ultrachat=224, authored=64)
    assert len(validation) == 32 and Counter(r['source'] for r in validation) == dict(oasst1=16, ultrachat=16)
    for field in ('conversation_id', 'root_id', 'duplicate_group', 'content_sha256'):
        assert not {r[field] for r in train} & {r[field] for r in validation}, field
    assert manifest['presentations'] == 1024 and [o['seed'] for o in manifest['orders']] == [73, 74]
    assert all(sorted(o['indices']) == list(range(512)) for o in manifest['orders'])
    assert manifest['orders'][0]['indices'] != manifest['orders'][1]['indices']
    for record in manifest['records']:
        assert record['serialized_tokens'] <= 512 and record['assistant_supervised_tokens'] > 0
        assert record['end_target_positions'] and record['provenance'] and record['content_sha256']
        assert record['admission']['first']['independent_human_review'] is False
        assert record['contamination']['hard_reject'] is False
        if record['source'] != 'authored':
            assert record['admission']['first']['disposition'] == record['admission']['second']['disposition'] == 'admit'
    caps = Counter(r['category'] for r in train)
    assert max(caps.values()) <= d.CATEGORY_CAP and caps['ack-refusal-branding'] <= d.ACK_CAP


@needs_manifest
def test_token_accounting_matches_masks():
    manifest = d.read(DIRECTORY / 'manifest/phase4-manifest.json')
    train = [r for r in manifest['records'] if r['split'] == 'train']
    total = manifest['token_accounting']['train']['total']
    assert total['serialized_tokens'] == sum(r['serialized_tokens'] for r in train)
    assert total['supervised_tokens'] == sum(r['assistant_supervised_tokens'] for r in train)
    assert total['end_targets'] == sum(len(r['end_target_positions']) for r in train)
    assert manifest['token_accounting']['train_two_passes']['supervised_tokens'] == 2 * total['supervised_tokens']


@needs_manifest
def test_serialization_masks_end_and_shift_on_real_records():
    from llm_tokenizer.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(d.tokenizer_identity()['path'])
    meta, records = d.verify_records(DIRECTORY / 'data/validation.jsonl', tokenizer)
    assert meta['token_ids'] == dict(pad=0, bos=1, eos=2, system=4, user=5, assistant=6, end=7)
    assert len(records) == 32 and all(r['end_targets'] >= 1 for r in records)


@needs_manifest
def test_manifest_verification_detects_drift(tmp_path):
    path = DIRECTORY / 'manifest/phase4-manifest.json'
    assert d.verify_manifest(path)['verified']
    manifest = d.read(path)
    # 1. Modified data content.
    validation = tmp_path / 'validation.jsonl'
    lines = (DIRECTORY / 'data/validation.jsonl').read_text().splitlines()
    first = json.loads(lines[0])
    first['messages'][-1]['content'] += ' Extra.'
    validation.write_text('\n'.join([json.dumps(first)] + lines[1:]) + '\n')
    changed = json.loads(json.dumps(manifest))
    changed['datasets']['validation']['path'] = str(validation)
    _write_rehashed(tmp_path / 'm1.json', changed)
    with pytest.raises(ValueError):
        d.verify_manifest(tmp_path / 'm1.json')
    # 2. Reordered presentation.
    changed = json.loads(json.dumps(manifest))
    order = changed['orders'][0]['indices']
    order[0], order[1] = order[1], order[0]
    _write_rehashed(tmp_path / 'm2.json', changed)
    with pytest.raises(ValueError, match='order'):
        d.verify_manifest(tmp_path / 'm2.json')
    # 3. Different tokenizer identity.
    changed = json.loads(json.dumps(manifest))
    changed['tokenizer']['sha256'] = '0' * 64
    _write_rehashed(tmp_path / 'm3.json', changed)
    with pytest.raises(ValueError, match='Tokenizer'):
        d.verify_manifest(tmp_path / 'm3.json')
    # 4. Edited manifest without rehash.
    changed = json.loads(json.dumps(manifest))
    changed['presentations'] = 1023
    d.write(tmp_path / 'm4.json', changed)
    with pytest.raises(ValueError, match='hash'):
        d.verify_manifest(tmp_path / 'm4.json')


def _write_rehashed(path, manifest):
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    body['manifest_sha256'] = d.canonical_sha256(body)
    d.write(path, body)


@needs_manifest
def test_order_matches_independent_seeded_shuffle():
    manifest = d.read(DIRECTORY / 'manifest/phase4-manifest.json')
    for order in manifest['orders']:
        expected = list(range(512))
        random.Random(order['seed']).shuffle(expected)
        assert order['indices'] == expected
