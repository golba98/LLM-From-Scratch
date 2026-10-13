"""Offline safety checks for diagnostic-only file relocation."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'diagnostic_organization',
    Path(__file__).resolve().parents[2] / 'scripts/diagnostics/organize_post_training_generation_diagnostic.py',
)
organization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(organization)


def populate(directory):
    for name in organization.REPORTS:
        (directory / name).write_text('# Original report\n')
    (directory / 'generations-final.jsonl').write_text('{"raw_text":"unchanged"}\n')
    (directory / 'retry.log').write_text('Original failed attempt\n')


def test_relocation_preserves_every_file_and_is_idempotent(tmp_path):
    populate(tmp_path)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    inventory = organization.organize(tmp_path)
    assert inventory['original_file_count'] == len(original)
    assert len({x['destination'] for x in inventory['files']}) == len(original)
    for entry in inventory['files']:
        assert (tmp_path / entry['destination']).read_bytes() == original[entry['original']]
        assert not (tmp_path / entry['original']).exists()
    assert organization.organize(tmp_path) == inventory
    assert json.loads((tmp_path / 'evidence/verification/organization-manifest.json').read_text()) == inventory


def test_changed_evidence_rejected_on_repeat(tmp_path):
    populate(tmp_path)
    organization.organize(tmp_path)
    (tmp_path / 'evidence/generation/generations-final.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Changed evidence'):
        organization.organize(tmp_path)


def test_missing_reports_do_not_move_any_evidence(tmp_path):
    (tmp_path / 'generations-final.jsonl').write_text('original')
    with pytest.raises(ValueError, match='Missing report'):
        organization.organize(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / 'generations-final.jsonl']


def test_raw_output_headings_are_not_report_sections():
    text = '# Report\n\n## Findings\n\n```text\n## Raw model output\n```\n\n## Checks\nPassed'
    assert list(organization.sections(text)) == ['Findings', 'Checks']
    lowered = organization.lower_headings(text)
    assert '### Findings' in lowered
    assert '\n## Raw model output\n' in lowered
